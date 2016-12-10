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

#include <cassert>
#include <mutex>
#include <sys/syscall.h>
#include <unistd.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "scanner/util/cuda.h"
#endif

namespace scanner {

class SystemAllocator {
public:
  SystemAllocator(DeviceType device_type, i32 device_id) :
    device_type_(device_type),
    device_id_(device_id) {
    assert(device_type == DeviceType::CPU || device_type == DeviceType::GPU);
  }

  u8* allocate(size_t size) {
    if (device_type_ == DeviceType::CPU) {
      return new u8[size];
    } else if (device_type_ == DeviceType::GPU) {
      u8* buffer;
      CU_CHECK(cudaSetDevice(device_id_));
      CU_CHECK(cudaMalloc((void**) &buffer, size));
      return buffer;
    }
  }

  void free(u8* buffer) {
    if (device_type_ == DeviceType::CPU) {
      delete buffer;
    } else if (device_type_ == DeviceType::GPU) {
      CU_CHECK(cudaSetDevice(device_id_));
      CU_CHECK(cudaFree(buffer));
    }
  }

private:
  DeviceType device_type_;
  i32 device_id_;
};

class PoolAllocator {
public:
  PoolAllocator(DeviceType device_type, i32 device_id, SystemAllocator* allocator) :
    device_type_(device_type),
    device_id_(device_id),
    system_allocator(allocator) {
    assert(device_type_ == DeviceType::CPU || device_type_ == DeviceType::GPU);
    pool_ = system_allocator->allocate(pool_size);
  }

  ~PoolAllocator() {
    system_allocator->free(pool_);
  }

  u8* allocate(size_t size) {
    Allocation alloc;
    alloc.length = size;
    alloc.refs = 1;

    std::lock_guard<std::mutex> guard(lock);
    bool found = false;
    i32 num_alloc = allocations.size();
    for (i32 i = 0; i < num_alloc; ++i) {
      Allocation lower;
      if (i == 0) {
        lower.offset = 0;
        lower.length = 0;
      } else {
        lower = allocations[i-1];
      }
      Allocation higher = allocations[i];
      assert(higher.offset >= lower.offset + lower.length);
      if ((higher.offset - (lower.offset + lower.length)) >= size) {
        alloc.offset = lower.offset + lower.length;
        allocations.insert(allocations.begin() + i, alloc);
        found = true;
        break;
      }
    }

    if (!found) {
      if (num_alloc > 0) {
        Allocation& last = allocations[num_alloc - 1];
        alloc.offset = last.offset + last.length;
      } else {
        alloc.offset = 0;
      }
      allocations.push_back(alloc);
    }

    u8* buffer = pool_ + alloc.offset;
    LOG_IF(FATAL, alloc.offset + alloc.length > pool_size)
      << "Exceeded pool size";

    return buffer;
  }

  void free(u8* buffer) {
    if (!buffer_in_pool(buffer)) {
      system_allocator->free(buffer);
      return;
    }

    std::lock_guard<std::mutex> guard(lock);
    i32 index;
    bool found = find_buffer(buffer, index);
    LOG_IF(FATAL, !found)
      << "Attempted to free unallocated buffer (did you forget to setref?)";

    Allocation& alloc = allocations[index];
    LOG_IF(FATAL, alloc.refs == 0)
      << "Attempted to free buffer with no refs";

    alloc.refs -= 1;
    if (alloc.refs == 0) {
      allocations.erase(allocations.begin() + index);
    }
  }

  void setref(u8* buffer, i32 refs) {
    std::lock_guard<std::mutex> guard(lock);
    i32 index;
    bool found = find_buffer(buffer, index);
    LOG_IF(FATAL, !found) << "Attempted to setref unallocated buffer";

    assert(allocations[index].refs == 1);
    allocations[index].refs = refs;
  }

  bool buffer_in_pool(u8* buffer) {
    return (size_t) buffer >= (size_t) pool_ &&
      (size_t) buffer <= (size_t) (pool_ + pool_size);
  }

  bool buffers_in_same_block(u8* buf1, u8* buf2) {
    std::lock_guard<std::mutex> guard(lock);
    i32 i1, i2;
    bool f1 = find_buffer(buf1, i1);
    bool f2 = find_buffer(buf2, i2);
    assert(f1 && f2);
    return i1 == i2;
  }

private:
  bool find_buffer(u8* buffer, i32& index) {
    i32 num_alloc = allocations.size();
    for (i32 i = 0; i < num_alloc; ++i) {
      Allocation alloc = allocations[i];
      if ((size_t) buffer >= (size_t) (pool_ + alloc.offset) &&
          (size_t) buffer < (size_t) (pool_ + alloc.offset + alloc.length)) {
        index = i;
        return true;
      }
    }
    return false;
  }

  typedef struct {
    i64 offset;
    i64 length;
    i64 refs;
  } Allocation;

  DeviceType device_type_;
  i32 device_id_;
  u8* pool_ = nullptr;
  const i64 pool_size = 4L*1024L*1024L*1024L;
  std::mutex lock;
  std::vector<Allocation> allocations;

  SystemAllocator* system_allocator;
};

static SystemAllocator* cpu_system_allocator = nullptr;
static std::map<i32, SystemAllocator*> gpu_system_allocators;
static PoolAllocator* cpu_pool_allocator = nullptr;
static std::map<i32, PoolAllocator*> gpu_pool_allocators;

void init_memory_allocators(std::vector<i32> gpu_device_ids, bool use_pool) {
  cpu_system_allocator = new SystemAllocator(DeviceType::CPU, 0);
  if (use_pool) {
    cpu_pool_allocator =
      new PoolAllocator(DeviceType::CPU, 0, cpu_system_allocator);
  }
#ifdef HAVE_CUDA
  for (i32 device_id : gpu_device_ids) {
    SystemAllocator* gpu_system_allocator =
      new SystemAllocator(DeviceType::GPU, device_id);
    gpu_system_allocators[device_id] = gpu_system_allocator;
    if (use_pool) {
      gpu_pool_allocators[device_id] =
        new PoolAllocator(DeviceType::GPU, device_id, gpu_system_allocator);
    }
  }
#endif
}

SystemAllocator* system_allocator_for_device(DeviceType type, i32 device_id) {
  if (type == DeviceType::CPU) {
    return cpu_system_allocator;
  } else if (type == DeviceType::GPU) {
    return gpu_system_allocators[device_id];
  } else {
    LOG(FATAL) << "Tried to allocate buffer of unsupported device type";
  }
}

PoolAllocator* pool_allocator_for_device(DeviceType type, i32 device_id) {
  if (type == DeviceType::CPU) {
    return cpu_pool_allocator;
  } else if (type == DeviceType::GPU) {
    return gpu_pool_allocators[device_id];
  } else {
    LOG(FATAL) << "Tried to allocate buffer of unsupported device type";
  }
}

u8* new_buffer(DeviceType type, i32 device_id, size_t size) {
  assert(size > 0);
  SystemAllocator* allocator = system_allocator_for_device(type, device_id);
  return allocator->allocate(size);
}

u8* new_buffer_from_pool(DeviceType type, i32 device_id, size_t size) {
  assert(size > 0);
  PoolAllocator* allocator = pool_allocator_for_device(type, device_id);
  assert(allocator != nullptr);
  return allocator->allocate(size);
}

void delete_buffer(DeviceType type, i32 device_id, u8* buffer) {
  assert(buffer != nullptr);
  PoolAllocator* pool_allocator = pool_allocator_for_device(type, device_id);
  SystemAllocator* system_allocator = system_allocator_for_device(type, device_id);
  if (pool_allocator != nullptr && pool_allocator->buffer_in_pool(buffer)) {
    pool_allocator->free(buffer);
  } else {
    system_allocator->free(buffer);
  }
}

// FIXME(wcrichto): case if transferring between two different GPUs
void memcpy_buffer(u8* dest_buffer, DeviceType dest_type, i32 dest_device_id,
                   const u8* src_buffer, DeviceType src_type, i32 src_device_id,
                   size_t size) {
#ifdef HAVE_CUDA
  CU_CHECK(cudaMemcpy(dest_buffer, src_buffer, size, cudaMemcpyDefault));
#else
  assert(dest_type == DeviceType::CPU);
  assert(dest_type == src_type);
  memcpy(dest_buffer, src_buffer, size);
#endif
}

#define NUM_CUDA_STREAMS 32

void memcpy_vec(std::vector<u8*> dest_buffers, DeviceType dest_type, i32 dest_device_id,
                const std::vector<u8*> src_buffers, DeviceType src_type, i32 src_device_id,
                std::vector<size_t> sizes) {
  thread_local std::vector<cudaStream_t> streams;
  if (streams.size() == 0) {
    streams.resize(NUM_CUDA_STREAMS);
    for (i32 i = 0; i < NUM_CUDA_STREAMS; ++i) {
      cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
  }

  assert(dest_buffers.size() > 0);
  assert(src_buffers.size() > 0);
  assert(dest_buffers.size() == src_buffers.size());

  PoolAllocator* dest_allocator = pool_allocator_for_device(dest_type, dest_device_id);
  PoolAllocator* src_allocator = pool_allocator_for_device(src_type, src_device_id);
  if (dest_allocator->buffer_in_pool(dest_buffers[0]) &&
      src_allocator->buffer_in_pool(src_buffers[0]) &&
      (dest_buffers.size() == 1 ||
       (dest_allocator->buffers_in_same_block(dest_buffers[0], dest_buffers[1]) &&
        src_allocator->buffers_in_same_block(src_buffers[0], src_buffers[1]))))
  {
    size_t total_size = 0;
    for (auto size : sizes) {
      total_size += size;
    }

    cudaMemcpyAsync(dest_buffers[0], src_buffers[0], total_size,
                    cudaMemcpyDefault, streams[0]);
    cudaStreamSynchronize(streams[0]);
  } else {
#ifdef HAVE_CUDA
    i32 n = dest_buffers.size();

    for (i32 i = 0; i < n; ++i) {
      CU_CHECK(cudaMemcpyAsync(dest_buffers[i], src_buffers[i], sizes[i],
                               cudaMemcpyDefault, streams[i % NUM_CUDA_STREAMS]));
    }

    for (i32 i = 0; i < std::min(n, NUM_CUDA_STREAMS); ++i) {
      cudaStreamSynchronize(streams[i]);
    }
#else
    LOG(FATAL) << "Not yet implemented";
#endif
  }
}

void setref_buffer(DeviceType type, i32 device_id, u8* buffer, i32 refs) {
  assert(buffer != nullptr);
  PoolAllocator* allocator = pool_allocator_for_device(type, device_id);
  assert(allocator != nullptr);
  allocator->setref(buffer, refs);
}
}
