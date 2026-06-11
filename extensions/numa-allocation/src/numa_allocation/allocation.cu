#include <torch/extension.h>
#include <cuda_runtime.h>
#include <numa.h>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>

constexpr size_t PAGE_SIZE = 4096;

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA ERROR (%s:%d): %s\n",                      \
                   __FILE__, __LINE__, cudaGetErrorString(err));            \
      std::exit(err);                                                       \
    }                                                                       \
  } while (0)

torch::Tensor zeros_numa_onnode(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    bool pin_memory,
    std::vector<int64_t> priority_nodes) {
  // Only support bfloat16, float16, and float32.
  if (dtype != at::kBFloat16 && dtype != at::kHalf && dtype != at::kFloat) {
    throw std::runtime_error("Unsupported dtype. Only bfloat16, float16, and float32 are supported.");
  }

  // Compute total number of elements.
  size_t numel = 1;
  for (const auto s : sizes) {
    numel *= s;
  }

  // Determine element size.
  size_t element_size = (dtype == at::kFloat) ? 4 : 2;
  size_t bytes = numel * element_size;
  // Align allocation size to a page boundary.
  size_t alloc_size = ((bytes + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;

  void* ptr = nullptr;
  bool allocated_on_numa = false;

  // Try NUMA allocation using the provided priority node list.
  if (!priority_nodes.empty()) {
    int max_node = numa_max_node();  // Highest node id available.
    for (auto node : priority_nodes) {
      // Valid node: should be between 0 and max_node (inclusive).
      if (node < 0 || node > max_node) continue;
      ptr = numa_alloc_onnode(alloc_size, node);
      if (ptr != nullptr) {
        allocated_on_numa = true;
        break;
      }
    }
  }

  // Fallback: use posix_memalign if NUMA allocation was not successful.
  if (ptr == nullptr) {
    int ret = posix_memalign(&ptr, PAGE_SIZE, alloc_size);
    if (ret != 0 || ptr == nullptr) {
      throw std::bad_alloc();
    }
  }

  // Zero-initialize the allocated memory.
  std::memset(ptr, 0, alloc_size);

  // Optionally pin the memory for CUDA usage.
  if (pin_memory) {
    CUDA_CHECK(cudaHostRegister(ptr, alloc_size, cudaHostRegisterMapped));
  }

  // Define a custom deleter to clean up the memory.
  auto deleter = [pin_memory, allocated_on_numa, alloc_size](void* p) {
    if (pin_memory) {
      CUDA_CHECK(cudaHostUnregister(p));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (allocated_on_numa) {
      numa_free(p, alloc_size);
    } else {
      free(p);
    }
  };

  // Create a tensor that uses the allocated memory.
  auto options = torch::TensorOptions().dtype(dtype).device(at::kCPU);
  auto tensor = torch::from_blob(ptr, sizes, deleter, options);
  return tensor;
}

torch::Tensor zeros_numa_on_nodemask(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    bool pin_memory,
    std::vector<int64_t> interleave_nodes) {
  // Only support bfloat16, float16, and float32.
  if (dtype != at::kBFloat16 && dtype != at::kHalf && dtype != at::kFloat) {
    throw std::runtime_error("Unsupported dtype. Only bfloat16, float16, and float32 are supported.");
  }

  // Compute total number of elements.
  size_t numel = 1;
  for (const auto s : sizes) {
    numel *= s;
  }

  // Determine element size.
  size_t element_size = (dtype == at::kFloat) ? 4 : 2;
  size_t bytes = numel * element_size;
  // Align allocation size to a page boundary.
  size_t alloc_size = ((bytes + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;

  // Ensure interleave_nodes is non-empty.
  if (interleave_nodes.empty()) {
    throw std::runtime_error("interleave_nodes must be non-empty.");
  }

  // Allocate a nodemask and set bits for the desired nodes.
  struct bitmask* nodemask = numa_allocate_nodemask();
  if (!nodemask) {
    throw std::runtime_error("Failed to allocate nodemask.");
  }
  numa_bitmask_clearall(nodemask);
  int max_node = numa_max_node();
  for (auto node : interleave_nodes) {
    if (node < 0 || node > max_node) continue;
    numa_bitmask_setbit(nodemask, node);
  }

  // Use numa_alloc_interleaved_subset to allocate memory across the given nodes.
  void* ptr = numa_alloc_interleaved_subset(alloc_size, nodemask);
  // Free the nodemask now that allocation is done.
  numa_free_nodemask(nodemask);

  if (ptr == nullptr) {
    throw std::bad_alloc();
  }

  // Zero-initialize the allocated memory.
  std::memset(ptr, 0, alloc_size);

  // Optionally pin the memory for CUDA usage.
  if (pin_memory) {
    CUDA_CHECK(cudaHostRegister(ptr, alloc_size, cudaHostRegisterMapped));
  }

  // Define a custom deleter to clean up the memory.
  auto deleter = [pin_memory, alloc_size](void* p) {
    if (pin_memory) {
      CUDA_CHECK(cudaHostUnregister(p));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    numa_free(p, alloc_size);
  };

  // Create a tensor that uses the allocated memory.
  auto options = torch::TensorOptions().dtype(dtype).device(at::kCPU);
  auto tensor = torch::from_blob(ptr, sizes, deleter, options);
  return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("zeros_numa_onnode", &zeros_numa_onnode, "NUMA aware zero initialization");
  m.def("zeros_numa_on_nodemask", &zeros_numa_on_nodemask, "NUMA aware zero initialization using interleaved nodemask");
}
