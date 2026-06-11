#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <numeric>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA ERROR (%s:%d): %s\n",                      \
                   __FILE__, __LINE__, cudaGetErrorString(err));            \
      std::exit(err);                                                       \
    }                                                                       \
  } while (0)

// Frees POSIX-aligned memory previously registered with CUDA.
void posix_memory_free(void* ptr) {
  if (!ptr) {
      std::fprintf(stderr, "WARNING: Attempted to free a null pointer.\n");
      return;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaHostUnregister(ptr)); 
  free(ptr);
}

// Frees CUDA pinned host memory allocated with cudaHostAlloc.
void cuda_host_memory_free(void* ptr) {
  if (!ptr) {
      std::fprintf(stderr, "WARNING: Attempted to free a null pointer.\n");
      return;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFreeHost(ptr));
}

// Creates a zero-initialized tensor with pinned memory using cudaHostAlloc.
torch::Tensor zeros_cuda_host_alloc_pinned(const std::vector<int64_t>& shape, c10::ScalarType dtype) {
  const size_t DMA_ALIGNMENT = 4096;
  size_t dtype_bytes = c10::elementSize(dtype);
  size_t numel = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
  size_t size = numel * dtype_bytes;
  size_t allocate_size = ((size + DMA_ALIGNMENT - 1) / DMA_ALIGNMENT) * DMA_ALIGNMENT;
  
  void* data_ptr = nullptr;
  CUDA_CHECK(cudaHostAlloc(&data_ptr, allocate_size, cudaHostAllocPortable));
  std::memset(data_ptr, 0, allocate_size);  

  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
  auto tensor = torch::from_blob(data_ptr, shape, cuda_host_memory_free, options);
  
  return tensor;
}

// Copies a CPU tensor to pinned memory using cudaHostAlloc.
torch::Tensor to_cuda_host_alloc_pinned(torch::Tensor& src) {
  TORCH_CHECK(src.device().is_cpu(), 
      "ERROR: Expected a CPU tensor, but received a tensor on device: ", 
      src.device().str());

  const size_t DMA_ALIGNMENT = 4096;
  size_t size = src.numel() * src.element_size(); 
  size_t allocate_size = ((size + DMA_ALIGNMENT - 1) / DMA_ALIGNMENT) * DMA_ALIGNMENT;

  void* data_ptr = nullptr;
  CUDA_CHECK(cudaHostAlloc(&data_ptr, allocate_size, cudaHostAllocPortable));

  auto tensor = torch::from_blob(data_ptr, src.sizes(), cuda_host_memory_free, src.options());
  tensor.copy_(src);

  return tensor;
}

// Copies a CPU tensor to aligned pinned memory using posix_memalign and CUDA registration.
torch::Tensor to_posix_memalign_pinned(torch::Tensor& src) {
  TORCH_CHECK(src.device().is_cpu(), 
              "ERROR: Expected a CPU tensor, but received a tensor on device: ", 
              src.device().str());

  const size_t DMA_ALIGNMENT = 4096;
  size_t size = src.numel() * src.element_size(); 
  size_t allocate_size = ((size + DMA_ALIGNMENT - 1) / DMA_ALIGNMENT) * DMA_ALIGNMENT;

  void* data_ptr = nullptr;
  int ret = posix_memalign(&data_ptr, DMA_ALIGNMENT, allocate_size);
  if (ret != 0) {
      throw std::runtime_error("ERROR: Failed to allocate aligned memory. posix_memalign returned: " + std::to_string(ret));
  }

  CUDA_CHECK(cudaHostRegister(data_ptr, allocate_size, cudaHostRegisterPortable)); 
  CUDA_CHECK(cudaDeviceSynchronize());

  auto tensor = torch::from_blob(data_ptr, src.sizes(), posix_memory_free, src.options());
  tensor.copy_(src);

  return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("zeros_cuda_host_alloc_pinned", &zeros_cuda_host_alloc_pinned,
    "Allocates zero-initialized pinned CPU memory directly using `cudaHostAlloc`. "
    "Returns a tensor utilizing this memory, optimized for CUDA device transfers.");
  m.def("to_cuda_host_alloc_pinned", &to_cuda_host_alloc_pinned,
    "Copies a CPU tensor to pinned memory allocated directly with `cudaHostAlloc`. "
    "Returns a new tensor with the same shape and options, optimized for CUDA device transfers.");
  m.def("to_posix_memalign_pinned", &to_posix_memalign_pinned,
    "Copies a CPU tensor to 4096-byte aligned memory allocated with `posix_memalign` and registered as pinned with `cudaHostRegister`. "
    "Returns a new tensor with the same shape and options, offering explicit alignment control.");
}