/* Copyright 2016 Carnegie Mellon University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scanner/util/memory.h"
#include "scanner/util/cuda.h"

#include <sys/syscall.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <cassert>
#include <mutex>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace scanner {

// Allocations in Scanner differ from manual memory management in three
// respects:
//
// 1. Scanner uses different allocators for different devices, e.g. the `new`
//    keyword for CPU allocations and cudaMalloc for GPU memory. The System and
//    Pool allocators are parameterized by a DeviceHandle to determine which
//    device they are allocating for.
//
// 2. The pool allocator up-front allocates a large memory pool and then does
// its
//    own simple memory allocation in place of the system allocator for the
//    respective device. Memory pools avoid potentially expensive and
//    synchronous
//    calls to the system allocator, e.g. cudaMalloc which synchronizes the
//    whole
//    GPU.
//
// 3. Block allocations allow ops to allocate a single block of memory
// for
//    their returned elements instead of allocating individually for each element. This
//    again reduces the number of cudaMallocs if not using a memory pool.
//    Regardless of pool usage, blocks can also be copied in a single memcpy
//    instead of many, which reduces memcpy calls. To avoid complexity in the
//    core Scanner engine, it is oblivious to whether a u8* in an output element
//    is from a block or an individual allocation. Instead, the allocation
//    runtime does reference counting when the engine calls free on a memory
//    block, e.g. if a memory block is allocated for 96 elements (96 different
//    pointers in the same block), then each free to a pointer into the block
//    decrements a reference counter until freeing the block at 0 refs.
//
// The user can dictate usage of the memory pool with the MemoryPoolConfig, but
// cannot directly call into it. Users can only ask for normal memory segments
// or block memory segments, the former of which is allocated by the system
// and the latter by the pool if it exists.

class Allocator {
 public:
  virtual ~Allocator(){};

  virtual u8* allocate(size_t size) = 0;
  virtual void free(u8* buffer) = 0;
};

class SystemAllocator : public Allocator {
 public:
  SystemAllocator(DeviceHandle device, bool pinned = false)
      : device_(device), pinned_(pinned) {}

  u8* allocate(size_t size) {
    if (device_.type == DeviceType::CPU) {
      try {
        if (pinned_) {
          u8* buff;
          CUDA_PROTECT({ CU_CHECK(cudaMallocHost((void**)&buff, size)); });
          return buff;
        } else {
          return new u8[size];
        }
      } catch (const std::bad_alloc& e) {
        LOG(FATAL) << "CPU memory allocation failed: " << e.what();
      }
    } else if (device_.type == DeviceType::GPU) {
      u8* buffer;
      CUDA_PROTECT({
        CU_CHECK(cudaSetDevice(device_.id));
        CU_CHECK(cudaMalloc((void**)&buffer, size));
      })
      return buffer;
    }
  }

  void free(u8* buffer) {
    if (device_.type == DeviceType::CPU) {
      if (pinned_) {
        CUDA_PROTECT({ CU_CHECK(cudaFreeHost(buffer)); });
      } else {
        delete[] buffer;
      }
    } else if (device_.type == DeviceType::GPU) {
      CUDA_PROTECT({
        CU_CHECK(cudaSetDevice(device_.id));
        CU_CHECK(cudaFree(buffer));
      });
    }
  }

  size_t alignment() {
    if (device_.type == DeviceType::CPU) {
      return 16;
    } else if (device_.type == DeviceType::GPU) {
      return 256;
    }
  }

 private:
  DeviceHandle device_;
  bool pinned_;
};

bool pointer_in_buffer(u8* ptr, u8* buf_start, u8* buf_end) {
  return (size_t)ptr >= (size_t)buf_start && (size_t)ptr < (size_t)buf_end;
}

class PoolAllocator : public Allocator {
 public:
  PoolAllocator(DeviceHandle device, SystemAllocator* allocator,
                size_t pool_size)
      : device_(device), system_allocator(allocator), pool_size_(pool_size) {
    pool_ = system_allocator->allocate(pool_size_);
  }

  ~PoolAllocator() { system_allocator->free(pool_); }

  u8* allocate(size_t size) {
    Allocation alloc;
    alloc.length = size;

    std::lock_guard<std::mutex> guard(lock_);
    bool found = false;
    i32 num_alloc = allocations_.size();
    for (i32 i = 0; i < num_alloc; ++i) {
      Allocation lower;
      if (i == 0) {
        lower.offset = 0;
        lower.length = 0;
      } else {
        lower = allocations_[i - 1];
      }
      Allocation higher = allocations_[i];
      assert(higher.offset >= lower.offset + lower.length);
      size_t base = align(lower.offset + lower.length);
      if ((higher.offset - base) >= size) {
        alloc.offset = base;
        allocations_.insert(allocations_.begin() + i, alloc);
        found = true;
        break;
      }
    }

    if (!found) {
      if (num_alloc > 0) {
        Allocation& last = allocations_[num_alloc - 1];
        alloc.offset = align(last.offset + last.length);
      } else {
        alloc.offset = 0;
      }
      allocations_.push_back(alloc);
    }

    LOG_IF(FATAL, alloc.offset + alloc.length >= pool_size_)
        << "Exceeded pool size";

    u8* buffer = pool_ + alloc.offset;
    return buffer;
  }

  size_t align(size_t ptr) {
    size_t alignment = system_allocator->alignment();
    size_t remainder = ptr % alignment;
    if (remainder != 0) {
      return ptr + (alignment - remainder);
    } else {
      return ptr;
    }
  }

  void free(u8* buffer) {
    LOG_IF(FATAL, !pointer_in_buffer(buffer, pool_, pool_ + pool_size_))
        << "Pool allocator tried to free buffer not in pool";

    std::lock_guard<std::mutex> guard(lock_);
    i32 index;
    bool found = find_buffer(buffer, index);
    LOG_IF(FATAL, !found) << "Attempted to free unallocated buffer in pool";

    Allocation& alloc = allocations_[index];
    allocations_.erase(allocations_.begin() + index);
  }

 private:
  bool find_buffer(u8* buffer, i32& index) {
    i32 num_alloc = allocations_.size();
    for (i32 i = 0; i < num_alloc; ++i) {
      Allocation alloc = allocations_[i];
      if ((size_t)buffer == (size_t)pool_ + alloc.offset) {
        index = i;
        return true;
      }
    }
    return false;
  }

  typedef struct {
    size_t offset;
    size_t length;
  } Allocation;

  DeviceHandle device_;
  u8* pool_ = nullptr;
  size_t pool_size_;
  std::mutex lock_;
  std::vector<Allocation> allocations_;

  SystemAllocator* system_allocator;
};

class BlockAllocator {
 public:
  BlockAllocator(Allocator* allocator) : allocator_(allocator) {}

  ~BlockAllocator() {
    std::lock_guard<std::mutex> guard(lock_);

    for (Allocation& alloc : allocations_) {
      assert(alloc.refs > 0);
      allocator_->free(alloc.buffer);
    }
  }

  u8* allocate(size_t size, i32 refs) {
    u8* buffer = allocator_->allocate(size);

    Allocation alloc;
    alloc.buffer = buffer;
    alloc.size = size;
    alloc.refs = refs;

    std::lock_guard<std::mutex> guard(lock_);
    allocations_.push_back(alloc);

    return buffer;
  }

  void free(u8* buffer) {
    std::lock_guard<std::mutex> guard(lock_);

    i32 index;
    bool found = find_buffer(buffer, index);
    LOG_IF(FATAL, !found) << "Block allocator freed non-block buffer";

    Allocation& alloc = allocations_[index];
    assert(alloc.refs > 0);
    alloc.refs -= 1;

    if (alloc.refs == 0) {
      allocator_->free(alloc.buffer);
      allocations_.erase(allocations_.begin() + index);
      return;
    }
  }

  bool buffers_in_same_block(std::vector<u8*> buffers) {
    assert(buffers.size() > 0);

    std::lock_guard<std::mutex> guard(lock_);
    i32 base_index;
    bool found = find_buffer(buffers[0], base_index);
    if (!found) {
      return false;
    }

    for (i32 i = 1; i < buffers.size(); ++i) {
      i32 index;
      found = find_buffer(buffers[i], index);
      if (!found || base_index != index) {
        return false;
      }
    }

    return true;
  }

  bool buffer_in_block(u8* buffer) {
    std::lock_guard<std::mutex> guard(lock_);
    i32 index;
    return find_buffer(buffer, index);
  }

 private:
  bool find_buffer(u8* buffer, i32& index) {
    i32 num_alloc = allocations_.size();
    for (i32 i = 0; i < num_alloc; ++i) {
      Allocation alloc = allocations_[i];
      if (pointer_in_buffer(buffer, alloc.buffer, alloc.buffer + alloc.size)) {
        index = i;
        return true;
      }
    }
    return false;
  }

  typedef struct {
    u8* buffer;
    size_t size;
    i32 refs;
  } Allocation;

  std::mutex lock_;
  std::vector<Allocation> allocations_;
  Allocator* allocator_;
};

static SystemAllocator* cpu_system_allocator = nullptr;
static std::map<i32, SystemAllocator*> gpu_system_allocators;
static PoolAllocator* cpu_pool_allocator = nullptr;
static BlockAllocator* cpu_block_allocator = nullptr;
static std::map<i32, PoolAllocator*> gpu_pool_allocators;
static std::map<i32, BlockAllocator*> gpu_block_allocators;

void init_memory_allocators(MemoryPoolConfig config,
                            std::vector<i32> gpu_device_ids) {
  cpu_system_allocator = new SystemAllocator(CPU_DEVICE, config.pinned_cpu());
  Allocator* cpu_block_allocator_base = cpu_system_allocator;
  if (config.cpu().use_pool()) {
    struct sysinfo info;
    i32 err = sysinfo(&info);
    LOG_IF(FATAL, err < 0) << "sysinfo failed: " << strerror(errno);
    size_t total_mem = info.totalram;
    LOG_IF(FATAL, config.cpu().free_space() > total_mem)
        << "Requested CPU free space (" << config.cpu().free_space() << ") "
        << "larger than total CPU memory size ( " << total_mem << ")";
    cpu_pool_allocator =
        new PoolAllocator(CPU_DEVICE, cpu_system_allocator,
                          total_mem - config.cpu().free_space());
    cpu_block_allocator_base = cpu_pool_allocator;
  }
  cpu_block_allocator = new BlockAllocator(cpu_block_allocator_base);

#ifdef HAVE_CUDA
  for (i32 device_id : gpu_device_ids) {
    DeviceHandle device = {DeviceType::GPU, device_id};
    SystemAllocator* gpu_system_allocator = new SystemAllocator(device);
    gpu_system_allocators[device.id] = gpu_system_allocator;
    Allocator* gpu_block_allocator_base = gpu_system_allocator;
    if (config.gpu().use_pool()) {
      cudaDeviceProp prop;
      CU_CHECK(cudaGetDeviceProperties(&prop, device_id));
      size_t total_mem = prop.totalGlobalMem;
      LOG_IF(FATAL, config.gpu().free_space() > total_mem)
          << "Requested GPU free space (" << config.gpu().free_space() << ") "
          << "larger than total GPU memory size ( " << total_mem << ") "
          << "on device " << device_id;
      gpu_pool_allocators[device.id] = new PoolAllocator(
          device, gpu_system_allocator, total_mem - config.gpu().free_space());
      gpu_block_allocator_base = gpu_pool_allocators[device.id];
    }
    gpu_block_allocators[device.id] =
        new BlockAllocator(gpu_block_allocator_base);
  }
#endif
}

void destroy_memory_allocators() {
  delete cpu_block_allocator;
  if (cpu_pool_allocator) {
    delete cpu_pool_allocator;
    cpu_pool_allocator = nullptr;
  }
  delete cpu_system_allocator;

#ifdef HAVE_CUDA
  for (auto entry : gpu_block_allocators) {
    delete entry.second;
  }
  for (auto entry : gpu_pool_allocators) {
    delete entry.second;
  }
  for (auto entry : gpu_system_allocators) {
    delete entry.second;
  }
  gpu_block_allocators.clear();
  gpu_pool_allocators.clear();
  gpu_system_allocators.clear();
#endif
}

SystemAllocator* system_allocator_for_device(DeviceHandle device) {
  if (device.type == DeviceType::CPU) {
    return cpu_system_allocator;
  } else if (device.type == DeviceType::GPU) {
    CUDA_PROTECT({/* dummy to trigger cuda check */});
    return gpu_system_allocators.at(device.id);
  } else {
    LOG(FATAL) << "Tried to allocate buffer of unsupported device type";
  }
}

BlockAllocator* block_allocator_for_device(DeviceHandle device) {
  if (device.type == DeviceType::CPU) {
    return cpu_block_allocator;
  } else if (device.type == DeviceType::GPU) {
    CUDA_PROTECT({/* dummy to trigger cuda check */});
    return gpu_block_allocators.at(device.id);
  } else {
    LOG(FATAL) << "Tried to allocate buffer of unsupported device type";
  }
}

u8* new_buffer(DeviceHandle device, size_t size) {
  assert(size > 0);
  SystemAllocator* allocator = system_allocator_for_device(device);
  return allocator->allocate(size);
}

u8* new_block_buffer(DeviceHandle device, size_t size, i32 refs) {
  assert(size > 0);
  BlockAllocator* allocator = block_allocator_for_device(device);
  return allocator->allocate(size, refs);
}

void delete_buffer(DeviceHandle device, u8* buffer) {
  assert(buffer != nullptr);
  BlockAllocator* block_allocator = block_allocator_for_device(device);
  if (block_allocator->buffer_in_block(buffer)) {
    block_allocator->free(buffer);
  } else {
    SystemAllocator* system_allocator = system_allocator_for_device(device);
    system_allocator->free(buffer);
  }
}

// FIXME(wcrichto): case if transferring between two different GPUs
void memcpy_buffer(u8* dest_buffer, DeviceHandle dest_device,
                   const u8* src_buffer, DeviceHandle src_device, size_t size) {
  if (dest_device.type == DeviceType::CPU &&
      src_device.type == DeviceType::CPU) {
    memcpy(dest_buffer, src_buffer, size);
  } else {
    assert(!(dest_device.type == DeviceType::GPU &&
             src_device.type == DeviceType::GPU &&
             dest_device.id != src_device.id));
    CUDA_PROTECT({
      CU_CHECK(cudaSetDevice(src_device.id));
      CU_CHECK(cudaMemcpy(dest_buffer, src_buffer, size, cudaMemcpyDefault));
    });
  }
}

#define NUM_CUDA_STREAMS 32

// TODO(wcrichto): implement CPU-CPU transfer
void memcpy_vec(std::vector<u8*> dest_buffers, DeviceHandle dest_device,
                const std::vector<u8*> src_buffers, DeviceHandle src_device,
                std::vector<size_t> sizes) {
  assert(dest_device.type == DeviceType::GPU ||
         src_device.type == DeviceType::GPU);
  assert(src_device.can_copy_to(dest_device));
  assert(dest_buffers.size() > 0);
  assert(src_buffers.size() > 0);
  assert(dest_buffers.size() == src_buffers.size());

#ifdef HAVE_CUDA
  thread_local std::vector<cudaStream_t> streams;
  if (streams.size() == 0) {
    streams.resize(NUM_CUDA_STREAMS);
    for (i32 i = 0; i < NUM_CUDA_STREAMS; ++i) {
      cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
  }

  BlockAllocator* dest_allocator = block_allocator_for_device(dest_device);
  BlockAllocator* src_allocator = block_allocator_for_device(src_device);

  if (src_device.type == DeviceType::GPU) {
    CU_CHECK(cudaSetDevice(src_device.id));
  } else if (dest_device.type == DeviceType::GPU) {
    CU_CHECK(cudaSetDevice(dest_device.id));
  }

  // In the case where the dest and src vectors are each respectively drawn
  // from a single block, we do a single memcpy from one block to the other.
  if (dest_allocator->buffers_in_same_block(dest_buffers) &&
      src_allocator->buffers_in_same_block(src_buffers)) {
    size_t total_size = 0;
    for (auto size : sizes) {
      total_size += size;
    }

    CU_CHECK(cudaMemcpyAsync(dest_buffers[0], src_buffers[0], total_size,
                             cudaMemcpyDefault, streams[0]));
    CU_CHECK(cudaStreamSynchronize(streams[0]));
  } else {
    i32 n = dest_buffers.size();

    for (i32 i = 0; i < n; ++i) {
      CU_CHECK(cudaMemcpyAsync(dest_buffers[i], src_buffers[i], sizes[i],
                               cudaMemcpyDefault,
                               streams[i % NUM_CUDA_STREAMS]));
    }

    for (i32 i = 0; i < std::min(n, NUM_CUDA_STREAMS); ++i) {
      cudaStreamSynchronize(streams[i]);
    }
  }
#else
  LOG(FATAL) << "Cuda not installed";
#endif
}
}
