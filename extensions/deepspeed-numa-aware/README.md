# DeepSpeed-0.16.0

This repository modifies DeepSpeed-0.16.0 to support NUMA-aware tensor allocation for CPU offloading configurations. To enable this feature, you must patch the tensor allocation method for tensor creation function. An example of how to apply this patch is provided later in this README.

## Installation

1. **Prerequisites:**  
   Ensure you have already follow the prerequisites in the main [README](../../README.md) to set up the NVIDIA driver, CUDA toolkit, and compatible PyTorch version.

2. **Install the Extension from Source:**  
   ```bash
   pip install .
   ```

## How to Apply the Patch

In CPU offloading scenarios, tensor allocation primarily involves compute weights, compute gradients, master weights, gradients, optimizer states, and more. This modified DeepSpeed separates each allocation type into distinct functions, making it easier to patch later. The patch function must follow this signature:

```python
def _zeros_patch_example(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
```

Below is an example implementation:

```python
import torch
import deepspeed

def _zeros_patch_example(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    return torch.zeros(shape=shape, dtype=dtype, device="cpu", pin_memory=pin_memory)

def patch_deepspeed_zero_overhead_pinned_memory():
    deepspeed.ops.adam.cpu_adam.zeros_cpu_for_momentums = _zeros_patch_example
    deepspeed.ops.adam.cpu_adam.zeros_cpu_for_variances = _zeros_patch_example
    deepspeed.runtime.zero.stage3.zeros_cpu_for_master_weights = _zeros_patch_example
    deepspeed.runtime.zero.stage3.zeros_cpu_for_master_gradients = _zeros_patch_example
    deepspeed.runtime.zero.stage3.zeros_cpu_for_compute_weights = _zeros_patch_example
    deepspeed.runtime.zero.stage3.zeros_cpu_for_compute_gradients = _zeros_patch_example
```
