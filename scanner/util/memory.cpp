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

#include <unistd.h>
#include <cassert>
#include <mutex>
#include <cmath>

#ifdef __linux
#include <sys/syscall.h>
#include <sys/sysinfo.h>
#elif __APPLE__
#include <sys/sysctl.h>
#endif

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace scanner {
namespace {

size_t get_total_ram() {
  size_t total_mem;

#ifdef __linux__
  struct sysinfo info;
  i32 err = sysinfo(&info);
  LOG_IF(FATAL, err < 0) << "sysinfo failed: " << strerror(errno);
  total_mem = info.totalram;
#elif __APPLE__
  int mem_size;
  size_t length = sizeof(mem_size);
  i32 err = sysctlbyname("hw.memsize", &mem_size, &length, NULL, 0);
  LOG_IF(FATAL, err < 0) << "sysctlbyname failed: " << strerror(err);
  total_mem = mem_size;
#endif

  return total_mem;
}

}

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
//    their returned elements instead of allocating individually for each
//    element. This
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
  SystemAllocator(DeviceHandle device)
    : device_(device) {
  }

  ~SystemAllocator() {
    // Device reset ensures cuda-memcheck will work
    if (device_.type == DeviceType::GPU) {
      CUDA_PROTECT({
        CU_CHECK(cudaSetDevice(device_.id));
        CU_CHECK(cudaDeviceReset());
      });
    }
  }

  u8* allocate(size_t size) {
    if (device_.type == DeviceType::CPU) {
      try {
        return new u8[size];
      } catch (const std::bad_alloc& e) {
        LOG(FATAL) << "CPU memory allocation failed (" << size << " bytes): " << e.what();
      }
    } else if (device_.type == DeviceType::GPU) {
      u8* buffer;
      CUDA_PROTECT({
        CU_CHECK(cudaSetDevice(device_.id));
        CU_CHECK(cudaMalloc((void**)&buffer, size));
      });
      return buffer;
    }
  }

  void free(u8* buffer) {
    if (device_.type == DeviceType::CPU) {
      delete[] buffer;
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

  ~PoolAllocator() {
    system_allocator->free(pool_);
  }

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
    {
      std::lock_guard<std::mutex> guard(map_lock_);
      for (auto& it : allocation_map_) {
        auto alloc = it.second;
        assert(alloc->refs > 0);
        allocator_->free(alloc->buffer);
      }
      allocation_map_.clear();
    }

    {
      std::lock_guard<std::mutex> guard(vec_lock_);
      for (Allocation& alloc : allocations_) {
        assert(alloc.refs > 0);
        allocator_->free(alloc.buffer);
      }
      allocations_.clear();
    }
  }

  u8* allocate(size_t size, i32 refs, std::string call_file = "",
               i32 call_line = 0) {
    u8* buffer = allocator_->allocate(size);

    Allocation alloc;
    alloc.buffer = buffer;
    alloc.size = size;
    alloc.refs = refs;
    alloc.call_file = call_file;
    alloc.call_line = call_line;

    {
      std::lock_guard<std::mutex> guard(vec_lock_);
      allocations_.push_back(alloc);
    }

    current_memory_allocated_ += alloc.size;
    max_memory_allocated_ =
        std::max(current_memory_allocated_, max_memory_allocated_);

    return buffer;
  }

  u8* allocate_sizes(const std::vector<size_t>& sizes, std::string call_file = "", i32 call_line = 0) {
    size_t total_size = 0;
    for (size_t size : sizes) { total_size += size; }
    u8* buffer = allocator_->allocate(total_size);

    std::shared_ptr<Allocation> alloc(new Allocation);
    alloc->buffer = buffer;
    alloc->size = total_size;
    alloc->refs = sizes.size();
    alloc->call_file = call_file;
    alloc->call_line = call_line;

    {
      std::lock_guard<std::mutex> guard(map_lock_);
      u8* cursor = buffer;
      for (size_t size : sizes) {
        allocation_map_.insert({cursor, alloc});
        cursor += size;
      }
    }

    current_memory_allocated_ += alloc->size;
    max_memory_allocated_ =
        std::max(current_memory_allocated_, max_memory_allocated_);

    return buffer;
  }

  void add_refs(u8* buffer, size_t refs) {
    {
      std::lock_guard<std::mutex> guard(map_lock_);
      auto it = allocation_map_.find(buffer);
      if (it != allocation_map_.end()) {
        std::shared_ptr<Allocation> alloc = it->second;
        assert(alloc->refs > 0);
        alloc->refs += refs;
        for (size_t i = 0; i < refs; ++i) {
          allocation_map_.insert({buffer, alloc});
        }
        return;
      }
    }

    {
      std::lock_guard<std::mutex> guard(vec_lock_);

      i32 index;
      bool found = find_buffer(buffer, index);
      LOG_IF(FATAL, !found)
        << "Block allocator tried to add ref to non-block buffer";

      Allocation& alloc = allocations_[index];
      assert(alloc.refs > 0);

      alloc.refs += refs;
    }
  }

  void free(u8* buffer) {
    {
      std::lock_guard<std::mutex> guard(map_lock_);
      auto it = allocation_map_.find(buffer);
      if (it != allocation_map_.end()) {
        std::shared_ptr<Allocation> alloc = it->second;
        allocation_map_.erase(it);

        assert(alloc->refs > 0);
        alloc->refs -= 1;

        if (alloc->refs == 0) {
          current_memory_allocated_ -= alloc->size;
          allocator_->free(alloc->buffer);
        }

        return;
      }
    }

    {
      std::lock_guard<std::mutex> guard(vec_lock_);

      i32 index;
      bool found = find_buffer(buffer, index);
      LOG_IF(FATAL, !found) << "Block allocator freed non-block buffer";

      Allocation& alloc = allocations_[index];
      assert(alloc.refs > 0);
      alloc.refs -= 1;

      if (alloc.refs == 0) {
        current_memory_allocated_ -= alloc.size;

        allocator_->free(alloc.buffer);
        allocations_.erase(allocations_.begin() + index);
      }
    }
  }

  std::vector<int> buffers_in_same_block(std::vector<u8*> buffers) {
    assert(buffers.size() > 0);

    {
      std::lock_guard<std::mutex> guard(map_lock_);
      auto it = allocation_map_.find(buffers[0]);
      if (it != allocation_map_.end()) {
        u8* prev_buffer = it->second->buffer;

        std::vector<int> block_sizes;
        int block_size = 1;
        for (i32 i = 1; i < buffers.size(); ++i) {
          auto it2 = allocation_map_.find(buffers[i]);
          if (it2 == allocation_map_.end() || it2->second->buffer != prev_buffer) {
            block_sizes.push_back(block_size);
            block_size = 1;
            prev_buffer = it2->second->buffer;
          } else {
            block_size += 1;
          }
        }

        return block_sizes;
      }
    }

    {
      std::lock_guard<std::mutex> guard(vec_lock_);

      i32 base_index = -1;
      std::vector<int> block_sizes;
      int block_size = 1;
      for (i32 i = 0; i < buffers.size(); ++i) {
        i32 index = -1;
        bool found = find_buffer(buffers[i], index);
        if (!found || base_index != index) {
          block_sizes.push_back(block_size);
          block_size = 1;
          base_index = index;
        } else {
          block_size += 1;
        }
      }

      return block_sizes;
    }
  }

  bool buffer_in_block(u8* buffer) {
    {
      std::lock_guard<std::mutex> guard(map_lock_);
      if (allocation_map_.find(buffer) != allocation_map_.end()) {
        return true;
      }
    }

    {
      std::lock_guard<std::mutex> guard(vec_lock_);
      i32 index;
      return find_buffer(buffer, index);
    }
  }

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

  const std::vector<Allocation> allocations() {
    std::lock_guard<std::mutex> guard1(map_lock_);
    std::lock_guard<std::mutex> guard2(vec_lock_);

    std::vector<Allocation> allocs_copy = allocations_;
    for (auto& it : allocation_map_) {
      allocs_copy.push_back(*(it.second));
    }

    return allocations_;
  }

  u64 current_memory_allocated() {
    return current_memory_allocated_;
  }

  u64 max_memory_allocated() {
    return max_memory_allocated_;
  }

 private:
  std::mutex map_lock_;
  std::mutex vec_lock_;
  std::vector<Allocation> allocations_;
  std::unordered_multimap<u8*, std::shared_ptr<Allocation>> allocation_map_;
  Allocator* allocator_;

  u64 current_memory_allocated_ = 0;
  u64 max_memory_allocated_ = 0;
};

class LinkedAllocator {
 public:
  LinkedAllocator(std::map<DeviceHandle, Allocator*> allocators)
    : allocators_(allocators) {}

  ~LinkedAllocator() {
    std::lock_guard<std::mutex> guard(lock_);
    for (Allocation& alloc : allocations_) {
      for (auto kv : alloc.buffers) {
        auto& allocator = allocators_.at(kv.first);
        allocator->free(kv.second);
      }
    }
    allocations_.clear();
  }

  u8* allocate(DeviceHandle device, size_t size, i32 refs) {
    auto& allocator = allocators_.at(device);
    u8* buffer = allocator->allocate(size);

    Allocation alloc;
    alloc.buffers[device] = buffer;
    alloc.size = size;
    alloc.refs[device] = refs;

    std::lock_guard<std::mutex> guard(lock_);
    allocations_.push_back(alloc);

    return buffer;
  }

  void add_refs(DeviceHandle device, u8* buffer, size_t refs) {
    auto& allocator = allocators_.at(device);

    std::lock_guard<std::mutex> guard(lock_);

    i32 index;
    bool found = find_buffer(device, buffer, index);
    LOG_IF(FATAL, !found)
        << "Block allocator tried to add ref to non-block buffer";

    Allocation& alloc = allocations_[index];
    alloc.refs[device] += refs;
  }

  void copy_or_add_refs(DeviceHandle source_device, u8* source_buffer,
                        size_t refs, DeviceHandle target_device,
                        u8*& target_buffer) {
    std::lock_guard<std::mutex> guard(lock_);

    // Check if buffer exists
    i32 index;
    bool found = find_buffer(source_device, source_buffer, index);
    LOG_IF(FATAL, !found)
        << "Linked allocator tried to copy or add ref to non-block buffer";
    // Check if requested device exists
    Allocation& alloc = allocations_[index];
    if (alloc.refs.count(target_device) > 0) {
      // Add ref
      alloc.refs[target_device] += refs;
    } else {
      // Copy
      auto& allocator = allocators_.at(source_device);
      u8* new_buffer = allocator->allocate(alloc.size);
      memcpy_buffer(new_buffer, target_device,
                    alloc.buffers[source_device], source_device, alloc.size);
      alloc.refs[target_device] = refs;
      alloc.buffers[target_device] = new_buffer;
    }
    // Set target_buffer to same offset as it would be in the allocation that
    // source_buffer is from
    target_buffer = (u64)(source_buffer - alloc.buffers[source_device]) +
                    alloc.buffers[target_device];
  }

  void free(DeviceHandle device, u8* buffer) {
    auto& allocator = allocators_.at(device);

    std::lock_guard<std::mutex> guard(lock_);

    i32 index;
    bool found = find_buffer(device, buffer, index);
    LOG_IF(FATAL, !found) << "Block allocator freed non-block buffer";

    Allocation& alloc = allocations_[index];
    assert(alloc.refs[device] > 0);
    alloc.refs[device] -= 1;

    if (alloc.refs[device] == 0) {
      allocator->free(alloc.buffers[device]);
      alloc.buffers.erase(device);
      alloc.refs.erase(device);
      if (alloc.refs.size() == 0) {
        allocations_.erase(allocations_.begin() + index);
      }
    }
  }

  std::vector<int> buffers_in_same_block(DeviceHandle device, std::vector<u8*> buffers) {
    assert(buffers.size() > 0);

    std::lock_guard<std::mutex> guard(lock_);
    i32 base_index;
    bool found = find_buffer(device, buffers[0], base_index);
    if (!found) {
      return {};
    }

    for (i32 i = 1; i < buffers.size(); ++i) {
      i32 index;
      found = find_buffer(device, buffers[i], index);
      if (!found || base_index != index) {
        return {};
      }
    }

    return {};
  }

  bool buffer_in_block(DeviceHandle device, u8* buffer) {
    std::lock_guard<std::mutex> guard(lock_);
    i32 index;
    return find_buffer(device, buffer, index);
  }

 private:
  bool find_buffer(DeviceHandle device, u8* buffer, i32& index) {
    auto& allocations = allocations_;
    i32 num_alloc = allocations_.size();
    for (i32 i = 0; i < num_alloc; ++i) {
      Allocation alloc = allocations_[i];
      if (alloc.buffers.count(device) > 0) {
        u8* alloc_buffer = alloc.buffers[device];
        if (pointer_in_buffer(buffer, alloc_buffer,
                              alloc_buffer + alloc.size)) {
          index = i;
          return true;
        }
      }
    }
    return false;
  }

  typedef struct {
    std::map<DeviceHandle, u8*> buffers;
    size_t size;
    std::map<DeviceHandle, i32> refs;
  } Allocation;

  std::mutex lock_;
  i64 last_allocation_id_;
  std::vector<Allocation> allocations_;
  std::map<DeviceHandle, Allocator*> allocators_;
};

static std::unique_ptr<SystemAllocator> cpu_system_allocator;
static std::map<i32, SystemAllocator*> gpu_system_allocators;
static PoolAllocator* cpu_pool_allocator = nullptr;
static std::unique_ptr<BlockAllocator> cpu_block_allocator;
static std::map<i32, PoolAllocator*> gpu_pool_allocators;
static std::map<i32, BlockAllocator*> gpu_block_allocators;
static std::unique_ptr<LinkedAllocator> linked_allocator;

//#define USE_LINKED_ALLOCATOR

#define PINNED_BUFFER_SIZE (32<<20)
static std::map<i32, u8*> pinned_cpu_buffers;
static std::map<i32, std::mutex> pinned_cpu_locks;

void init_memory_allocators(MemoryPoolConfig config,
                            std::vector<i32> gpu_device_ids) {
  cpu_system_allocator.reset(new SystemAllocator(CPU_DEVICE));
  Allocator* cpu_block_allocator_base = cpu_system_allocator.get();
  if (config.cpu().use_pool()) {
    size_t total_mem = get_total_ram();
    LOG_IF(FATAL, config.cpu().free_space() > total_mem)
        << "Requested CPU free space (" << config.cpu().free_space() << ") "
        << "larger than total CPU memory size ( " << total_mem << ")";
    cpu_pool_allocator =
        new PoolAllocator(CPU_DEVICE, cpu_system_allocator.get(),
                          total_mem - config.cpu().free_space());
    cpu_block_allocator_base = cpu_pool_allocator;
  }
#ifdef USE_LINKED_ALLOCATOR
  std::map<DeviceHandle, Allocator*> allocators;
  allocators[CPU_DEVICE] = cpu_block_allocator_base;
#else
  cpu_block_allocator.reset(new BlockAllocator(cpu_block_allocator_base));
#endif

#ifdef HAVE_CUDA
  for (i32 device_id : gpu_device_ids) {
    cudaSetDevice(device_id);
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
#ifdef USE_LINKED_ALLOCATOR
    allocators[device] = gpu_block_allocator_base;
#else
    gpu_block_allocators[device.id] =
        new BlockAllocator(gpu_block_allocator_base);
#endif
    CU_CHECK(cudaMallocHost((void**)&pinned_cpu_buffers[device.id],
                            PINNED_BUFFER_SIZE));
  }
#endif
#ifdef USE_LINKED_ALLOCATOR
  linked_allocator.reset(new LinkedAllocator(allocators));
#endif
}

void destroy_memory_allocators() {
  linked_allocator.reset(nullptr);
  cpu_block_allocator.reset(nullptr);
  if (cpu_pool_allocator) {
    delete cpu_pool_allocator;
    cpu_pool_allocator = nullptr;
  }
  cpu_system_allocator.reset(nullptr);

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
  for (auto entry : pinned_cpu_buffers) {
    cudaFreeHost(entry.second);
  }
  gpu_block_allocators.clear();
  gpu_pool_allocators.clear();
  gpu_system_allocators.clear();
  pinned_cpu_buffers.clear();
#endif
}

SystemAllocator* system_allocator_for_device(DeviceHandle device) {
  if (device.type == DeviceType::CPU) {
    return cpu_system_allocator.get();
  } else if (device.type == DeviceType::GPU) {
    CUDA_PROTECT({/* dummy to trigger cuda check */});
    return gpu_system_allocators.at(device.id);
  } else {
    LOG(FATAL) << "Tried to allocate buffer of unsupported device type";
  }
}

BlockAllocator* block_allocator_for_device(DeviceHandle device) {
  if (device.type == DeviceType::CPU) {
    return cpu_block_allocator.get();
  } else if (device.type == DeviceType::GPU) {
    CUDA_PROTECT({/* dummy to trigger cuda check */});
    return gpu_block_allocators.at(device.id);
  } else {
    LOG(FATAL) << "Tried to allocate buffer of unsupported device type";
  }
}

u8* new_buffer_(DeviceHandle device, size_t size, std::string call_file,
               i32 call_line) {
  return new_block_buffer_size_(device, size, 1, call_file, call_line);
}

u8* new_block_buffer_(DeviceHandle device, size_t size, i32 refs,
                     std::string call_file, i32 call_line) {
  LOG_IF(FATAL, size == 0) << "Cannot allocate zero-length buffer";
#ifdef USE_LINKED_ALLOCATOR
  return linked_allocator->allocate(device, size, refs);
#else
  BlockAllocator* allocator = block_allocator_for_device(device);
  return allocator->allocate(size, refs, call_file, call_line);
#endif
}

u8* new_block_buffer_size_(DeviceHandle device, size_t size, i32 copies,
                            std::string call_file, i32 call_line) {
  std::vector<size_t> sizes;
  for (i32 i = 0; i < copies; ++i) { sizes.push_back(size); }
  return new_block_buffer_sizes_(device, sizes, call_file, call_line);
}

u8* new_block_buffer_sizes_(DeviceHandle device, const std::vector<size_t>& sizes,
                            std::string call_file, i32 call_line) {
  LOG_IF(FATAL, sizes.size() == 0) << "Cannot allocate zero-length buffer";
#ifdef USE_LINKED_ALLOCATOR
  LOG(FATAL) << "Linked allocator doesn't implement size-list based allocation";
#else
  BlockAllocator* allocator = block_allocator_for_device(device);
  return allocator->allocate_sizes(sizes, call_file, call_line);
#endif
}




void add_buffer_ref(DeviceHandle device, u8* buffer) {
  add_buffer_refs(device, buffer, 1);
}

void add_buffer_refs(DeviceHandle device, u8* buffer, i32 refs) {
  assert(buffer != nullptr);
#ifdef USE_LINKED_ALLOCATOR
  return linked_allocator->add_refs(device, buffer, refs);
#else
  BlockAllocator* block_allocator = block_allocator_for_device(device);
  block_allocator->add_refs(buffer, refs);
#endif
}

void delete_buffer(DeviceHandle device, u8* buffer) {
  assert(buffer != nullptr);
#ifdef USE_LINKED_ALLOCATOR
  linked_allocator->free(device, buffer);
#else
  BlockAllocator* block_allocator = block_allocator_for_device(device);
  if (block_allocator->buffer_in_block(buffer)) {
    block_allocator->free(buffer);
  } else {
    SystemAllocator* system_allocator = system_allocator_for_device(device);
    system_allocator->free(buffer);
  }
#endif
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
      if (size <= PINNED_BUFFER_SIZE) {
        if (dest_device.type == DeviceType::CPU) {
          CU_CHECK(cudaMemcpy(pinned_cpu_buffers[src_device.id], src_buffer,
                              size, cudaMemcpyDefault));
          memcpy(dest_buffer, pinned_cpu_buffers[src_device.id], size);
        } else if (src_device.type == DeviceType::CPU) {
          memcpy(pinned_cpu_buffers[dest_device.id], src_buffer, size);
          CU_CHECK(cudaMemcpy(dest_buffer, pinned_cpu_buffers[dest_device.id], size,
                              cudaMemcpyDefault));
        } else {
          CU_CHECK(cudaMemcpy(dest_buffer, src_buffer, size, cudaMemcpyDefault));
        }
      } else {
        CU_CHECK(cudaMemcpy(dest_buffer, src_buffer, size, cudaMemcpyDefault));
      }
    });
  }
}

#define NUM_CUDA_STREAMS 32

// TODO(wcrichto): implement CPU-CPU transfer
void memcpy_vec(std::vector<u8*>& dest_buffers, DeviceHandle dest_device,
                const std::vector<u8*>& src_buffers, DeviceHandle src_device,
                const std::vector<size_t>& sizes) {
  assert(src_device.can_copy_to(dest_device));
  assert(dest_buffers.size() > 0);
  assert(src_buffers.size() > 0);
  assert(dest_buffers.size() == src_buffers.size());

#ifndef USE_LINKED_ALLOCATOR
  BlockAllocator* dest_allocator = block_allocator_for_device(dest_device);
  BlockAllocator* src_allocator = block_allocator_for_device(src_device);
#endif

  size_t total_size = 0;
  for (auto size : sizes) {
    total_size += size;
  }

  // In the case where buffers in the dest and src vectors are each respectively drawn
  // from a single block, we do a single memcpy from one block to the other.
  std::vector<int> src_from_same_block;
  std::vector<int> dest_from_same_block;
#ifdef USE_LINKED_ALLOCATOR
  dest_from_same_block =
      linked_allocator->buffers_in_same_block(dest_device, dest_buffers);
  src_from_same_block =
      linked_allocator->buffers_in_same_block(src_device, src_buffers);
#else
  dest_from_same_block = dest_allocator->buffers_in_same_block(dest_buffers);
  src_from_same_block = src_allocator->buffers_in_same_block(src_buffers);
#endif

  if (dest_device.type == DeviceType::GPU ||
      src_device.type == DeviceType::GPU) {
#ifdef HAVE_CUDA
    static thread_local std::vector<cudaStream_t> streams;
    if (streams.size() == 0) {
      streams.resize(NUM_CUDA_STREAMS);
      for (i32 i = 0; i < NUM_CUDA_STREAMS; ++i) {
        CU_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      }
    }

    // Attempts to copy elements from one block in the source to one block in
    // the dest
    i64 src_block = 0;
    i64 src_elements_used = 0;
    i64 src_block_offset = 0;
    i64 dest_block = 0;
    i64 dest_elements_used = 0;
    i64 dest_block_offset = 0;
    while (src_block < src_from_same_block.size() &&
           dest_block < dest_from_same_block.size()) {
      i64 src_elements = src_from_same_block[src_block];
      i64 src_elements_left = src_elements - src_elements_used;
      i64 dest_elements = dest_from_same_block[dest_block];
      i64 dest_elements_left = dest_elements - dest_elements_used;
      i64 elements_to_copy = std::min(dest_elements_left, src_elements_left);
      memcpy_buffer(dest_buffers[dest_block_offset], dest_device,
                    src_buffers[src_block_offset], src_device, total_size);
      src_elements_used += elements_to_copy;
      if (src_elements_used == src_elements) {
        src_block += 1;
        src_elements_used = 0;
        src_block_offset += src_elements;
      }
      dest_elements_used += elements_to_copy;
      if (dest_elements_used == dest_elements) {
        dest_block += 1;
        dest_elements_used = 0;
        dest_block_offset += dest_elements;
      }
    }
#else
    LOG(FATAL) << "Cuda not installed";
#endif
  } else {
    i64 src_block = 0;
    i64 src_elements_used = 0;
    i64 src_block_offset = 0;
    i64 dest_block = 0;
    i64 dest_elements_used = 0;
    i64 dest_block_offset = 0;
    while (src_block < src_from_same_block.size() &&
           dest_block < dest_from_same_block.size()) {
      i64 src_elements = src_from_same_block[src_block];
      i64 src_elements_left = src_elements - src_elements_used;
      i64 dest_elements = dest_from_same_block[dest_block];
      i64 dest_elements_left = dest_elements - dest_elements_used;
      i64 elements_to_copy = std::min(dest_elements_left, src_elements_left);
      memcpy(dest_buffers[dest_block_offset], src_buffers[src_block_offset],
             total_size);
      src_elements_used += elements_to_copy;
      if (src_elements_used == src_elements) {
        src_block += 1;
        src_elements_used = 0;
        src_block_offset += src_elements;
      }
      dest_elements_used += elements_to_copy;
      if (dest_elements_used == dest_elements) {
        dest_block += 1;
        dest_elements_used = 0;
        dest_block_offset += dest_elements;
      }
    }
  }
}

void copy_or_ref_buffers(std::vector<u8*>& dest_buffers,
                         DeviceHandle dest_device,
                         const std::vector<u8*>& src_buffers,
                         DeviceHandle src_device,
                         const std::vector<size_t>& sizes) {
  assert(src_device.can_copy_to(dest_device));
  assert(src_buffers.size() > 0);

#ifdef USE_LINKED_ALLOCATOR
  // If source buffers are all from same block, this will perform only one
  // copy. However, it will perform multiple lookups in the allocator.
  dest_buffers.resize(src_buffers.size());
  for (i32 i = 0; i < dest_buffers.size(); ++i) {
    linked_allocator->copy_or_add_refs(src_device, src_buffers[i], 1,
                                       dest_device, dest_buffers[i]);
  }
#else
  BlockAllocator* dest_allocator = block_allocator_for_device(dest_device);
  if (dest_device.is_same_address_space(src_device)) {
    for (auto& buf : src_buffers) {
      dest_buffers.push_back(buf);
      dest_allocator->add_refs(buf, 1);
    }
  } else {
    u8* dest_buff = dest_allocator->allocate_sizes(sizes);
    for (size_t size : sizes) {
      dest_buffers.push_back(dest_buff);
      dest_buff += size;
    }
    memcpy_vec(dest_buffers, dest_device, src_buffers, src_device, sizes);
  }
#endif
}

u64 current_memory_allocated(DeviceHandle device) {
  BlockAllocator* block_allocator = block_allocator_for_device(device);
  return block_allocator->current_memory_allocated();
}

u64 max_memory_allocated(DeviceHandle device) {
  BlockAllocator* block_allocator = block_allocator_for_device(device);
  return block_allocator->max_memory_allocated();
}

const std::vector<Allocation> allocator_allocations(DeviceHandle device) {
  BlockAllocator* block_allocator = block_allocator_for_device(device);
  return block_allocator->allocations();
}

}
