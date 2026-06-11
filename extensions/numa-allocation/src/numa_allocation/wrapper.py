import torch
from .allocation import zeros_numa_onnode as _zeros_numa_onnode
from .allocation import zeros_numa_on_nodemask as _zeros_numa_on_nodemask

def zeros_numa_onnode_cpu(
    shape: tuple[int],
    dtype: torch.dtype,
    pin_memory: bool,
    priority_numa_nodes: list[int] = [],
) -> torch.Tensor:
    """
    Allocates a NUMA-aware CPU tensor.
    
    Args:
        shape (tuple[int]): The shape of the tensor.
        dtype (torch.dtype): Data type (currently supports bfloat16, float16, float32).
        pin_memory (bool): Using the page-lock pinned or not, for faster CUDA device transfer.
        priority_numa_nodes (list[int], optional): Preferred NUMA node(s). If None, the default NUMA policy is used. Otherwise, allocation will first attempt on the leftmost node and will move to the next node only if the current node is invalid or out of free space.
    
    Returns:
        torch.Tensor: A tensor allocated according to the specified NUMA policy.
    """
    assert shape is not None
    assert dtype is not None
    assert pin_memory is not None
    return _zeros_numa_onnode(shape, dtype, pin_memory, priority_numa_nodes)

def zeros_numa_on_nodemask_cpu(
    shape: tuple[int],
    dtype: torch.dtype,
    pin_memory: bool,
    interleave_numa_nodes: list[int],
) -> torch.Tensor:
    """
    Allocates a NUMA-aware CPU tensor using an interleaved nodemask.

    Args:
        shape (tuple[int]): The shape of the tensor.
        dtype (torch.dtype): Data type (currently supports bfloat16, float16, float32).
        pin_memory (bool): If True, the memory will be pinned for faster CUDA transfers.
        interleave_numa_nodes (list[int]): A non-empty list of NUMA node IDs. The allocated memory's pages 
            will be migrated so that future page faults are interleaved across these nodes.

    Returns:
        torch.Tensor: A tensor allocated according to the specified interleaved NUMA policy.
    """
    if not shape:
        raise ValueError("Shape must be non-empty.")
    if dtype is None:
        raise ValueError("dtype must be specified.")
    if not interleave_numa_nodes:
        raise ValueError("interleave_numa_nodes must be provided and non-empty.")
    return _zeros_numa_on_nodemask(shape, dtype, pin_memory, interleave_numa_nodes)