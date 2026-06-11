# Zero Overhead Pinned Memory

A PyTorch extension that provides alternative methods to create pinned CPU memory tensors, bypassing the extra memory overhead introduced by PyTorch’s power-of-two allocation strategy. This extension offers **three distinct manual allocation methods** to optimize memory usage and flexibility for your specific needs.

## Motivation

In PyTorch, when you set `pin_memory=True` or use `.pin_memory()` on a tensor, the allocation request is handled by PyTorch’s C++ backend. The requested size is rounded up to the nearest power of two (via a function like `c10::llvm::PowerOf2Ceil`) before allocation with CUDA’s memory routines (e.g., `cudaHostAlloc`). For more details, see [Discussion on GitHub Issue #95823](https://github.com/pytorch/pytorch/issues/95823). This approach aims to:

- **Simplify Caching and Reduce Fragmentation:** Uniform block sizes streamline the CUDA caching allocator’s bookkeeping, improving memory reuse and reducing fragmentation.

While this benefits dynamic memory management, it can allocate significantly more pinned memory than needed—especially for large tensors whose sizes aren’t powers of two. In scenarios like DeepSpeed, where pinned memory persists until the process terminates, the fragmentation prevention becomes less relevant, amplifying the memory waste.

This repository provides three alternative methods to avoid this overhead:

- **`to_posix_memalign_pinned`:** Allocates aligned memory using `posix_memalign` (4096-byte alignment) and registers it with CUDA via `cudaHostRegister`, minimizing overhead beyond minimal metadata.
- **`to_cuda_host_alloc_pinned`:** Allocates pinned memory directly with `cudaHostAlloc` and copies data from an existing tensor, offering a straightforward alternative.
- **`zeros_cuda_host_alloc_pinned`:** Creates a zero-initialized tensor directly in pinned memory using `cudaHostAlloc`, eliminating the need for an intermediate unpinned tensor.

## Features

1. **Pinned Memory Tensor Creation**  
   The extension provides three CUDA extension functions:
   - **`to_posix_memalign_pinned(tensor)`**  
     Copies an existing CPU tensor to pinned memory allocated with `posix_memalign` and registered with `cudaHostRegister`.
   - **`to_cuda_host_alloc_pinned(tensor)`**  
     Copies an existing CPU tensor to pinned memory allocated with `cudaHostAlloc`.
   - **`zeros_cuda_host_alloc_pinned(shape, dtype)`**  
     Creates a new zero-initialized tensor in pinned memory allocated with `cudaHostAlloc`.

   These functions bypass PyTorch’s power-of-two allocation, reducing memory overhead. The first two copy data from an existing tensor, while the third is ideal for initializing pinned zero tensors directly.

2. **Memory Usage Comparison**  
   Scripts compare memory usage between PyTorch’s standard pinned allocator and the three manual approaches.

3. **Performance Benchmark**  
   Code evaluates CPU-to-GPU transfer speeds and allocation overhead across:
   - PyTorch’s built-in pinned memory (`pin_memory=True`)
   - `to_posix_memalign_pinned`
   - `to_cuda_host_alloc_pinned`
   - `zeros_cuda_host_alloc_pinned`

4. **Correctness Tests**  
   Tests ensure proper allocation, registration, and deallocation of pinned memory, maintaining data integrity during GPU transfers.

## Installation

1. **Prerequisites:**  
   Ensure you have already follow the prerequisites in the main [README](../../README.md) to set up the NVIDIA driver, CUDA toolkit, and compatible PyTorch version.

2. **Install the Extension from Source:**  
   ```bash
   pip install .
   ```

## Usage

After installation, import the functions from `zero_overhead_pinned_memory`:

- **`to_posix_memalign_pinned`**  
  ```python
  import torch
  from zero_overhead_pinned_memory import to_posix_memalign_pinned

  cpu_tensor = torch.randn(100, 100)
  pinned_tensor = to_posix_memalign_pinned(cpu_tensor)
  ```
  Allocates pinned memory via `posix_memalign` and registers it with CUDA.

- **`to_cuda_host_alloc_pinned`**  
  ```python
  import torch
  from zero_overhead_pinned_memory import to_cuda_host_alloc_pinned

  cpu_tensor = torch.randn(100, 100)
  pinned_tensor = to_cuda_host_alloc_pinned(cpu_tensor)
  ```
  Allocates pinned memory using `cudaHostAlloc`.

- **`zeros_cuda_host_alloc_pinned`**  
  ```python
  import torch
  from zero_overhead_pinned_memory import zeros_cuda_host_alloc_pinned

  pinned_tensor = zeros_cuda_host_alloc_pinned((100, 100), torch.float32)
  ```
  Creates a zero-initialized pinned tensor with `cudaHostAlloc`, useful for direct initialization without an unpinned intermediate.

### 1. Running Tests

Verify correctness with:
```bash
pytest test_correctness.py
```
Tests confirm pinned memory allocation, registration, and data consistency on GPU transfers.

### 2. Memory Usage Comparison

Compare memory usage:
```bash
python memory_usage.py
```
This measures host memory usage for PyTorch’s pinned allocator versus the three manual methods, executed in separate processes for accuracy.

### 3. Performance Benchmark

Evaluate allocation and transfer performance:
```bash
python performance_benchmark.py
```
Compares:
1. PyTorch’s built-in pinned memory (`pin_memory=True`)
2. `to_posix_memalign_pinned`
3. `to_cuda_host_alloc_pinned`
4. `zeros_cuda_host_alloc_pinned`

Outputs average allocation times (CPU creation + pinning) and GPU transfer times.


## Reference

For CUDA host memory details, see:
- [CUDA Runtime API: cudaHostRegister](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g81fd4101862bbefdb42a62d60e515eea)
- [CUDA Runtime API: cudaHostAlloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902)