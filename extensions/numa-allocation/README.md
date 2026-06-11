# NUMA-Aware Tensor Allocation for PyTorch

This package provides a NUMA-aware CPU tensor allocation function for PyTorch. It allows you to allocate tensors on specific NUMA nodes or interleave them across nodes, with optional page-lock pinning for faster CUDA transfers.

## Installation

1. **Prerequisites:**  
   Ensure you have already follow the prerequisites in the main [README](../../README.md) to set up the NVIDIA driver, CUDA toolkit, and compatible PyTorch version.

2. **Install the Extension from Source:**  
   ```bash
   pip install .
   ```

## Running Tests

A test script is provided to verify the functionality and correctness of the NUMA-aware allocation. It uses `numastat` to print the allocation status on different NUMA nodes.

**Note:** To see the expected NUMA interleaved behavior, run the tests under the NUMA control:

```bash
numactl --interleave=all python test_correctness.py
```

This command runs the tests and prints example outputs for different allocation scenarios, such as default interleaved allocation, allocation on a specific NUMA node, or fallback behavior with invalid nodes.

## API Usage Example

Below is an example of how to use the NUMA-aware allocation API in your code:

```python
import torch
from numa_allocation import (
    zeros_numa_onnode_cpu,
    zeros_numa_on_nodemask_cpu,
)

# Example 1: Allocate a tensor with default NUMA interleaving using zeros_numa_onnode_cpu.
tensor_default = zeros_numa_onnode_cpu(
    shape=(1024, 1024, 1024),
    dtype=torch.float,
    pin_memory=True
)

# Example 2: Allocate a tensor on a preferred NUMA node (e.g., node 0) using zeros_numa_onnode_cpu.
tensor_node0 = zeros_numa_onnode_cpu(
    shape=(1024, 1024, 1024),
    dtype=torch.float,
    pin_memory=False,
    priority_numa_nodes=[0]
)

# Example 3: Allocate a tensor with NUMA interleaved allocation across multiple nodes 
# using zeros_numa_on_nodemask_cpu.
tensor_interleaved = zeros_numa_on_nodemask_cpu(
    shape=(1024, 1024, 1024),
    dtype=torch.float,
    pin_memory=True,
    interleave_numa_nodes=[0, 1]
)

# You can now use these tensors in your application. For example, printing their properties:
print("Default tensor - is_pinned:", tensor_default.is_pinned())
print("Node 0 tensor - dtype:", tensor_node0.dtype)
print("Interleaved tensor - shape:", tensor_interleaved.shape)
```

In the above examples:

- **Example 1:** Allocates a tensor with default NUMA interleaving.
- **Example 2:** Attempts to allocate on a preferred NUMA node (node 0).
- **Example 3:** Uses `zeros_numa_on_nodemask_cpu` to interleave the allocation across nodes 0 and 1.

Setting `pin_memory=True` in these examples registers the tensor’s memory with CUDA for faster host-to-device transfers.