import torch
import psutil
import os
import gc
import multiprocessing

from zero_overhead_pinned_memory import to_posix_memalign_pinned, to_cuda_host_alloc_pinned, zeros_cuda_host_alloc_pinned

def get_memory_usage():
    """Returns the current (RSS) and virtual memory usage in bytes."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss, mem_info.vms

def measure_overhead(tensor_size, dtype, pinned_alloc_func, queue):
    """
    Measures the memory overhead when using pinned memory.
    
    The function first creates an unpinned tensor, measures the memory usage,
    then creates a pinned tensor using the provided allocation function and 
    measures memory usage again.
    
    Results are placed on the queue.
    """
    # --- Measure unpinned allocation ---
    mem_before, _ = get_memory_usage()
    t_unpinned = torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=False)
    mem_after, _ = get_memory_usage()
    unpinned_size = mem_after - mem_before
    del t_unpinned
    gc.collect()

    # --- Measure pinned allocation ---
    mem_before, _ = get_memory_usage()
    t_pinned = pinned_alloc_func(tensor_size, dtype)
    mem_after, _ = get_memory_usage()
    pinned_size = mem_after - mem_before
    del t_pinned
    gc.collect()

    overhead = pinned_size / unpinned_size if unpinned_size > 0 else float('inf')
    queue.put((unpinned_size, pinned_size, overhead))

def get_overhead(tensor_size, dtype, pinned_alloc_func):
    """
    Runs the overhead measurement in a separate process and returns the results.
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=measure_overhead, 
        args=(tensor_size, dtype, pinned_alloc_func, queue)
    )
    process.start()
    process.join()
    return queue.get()

# --- Allocation functions for pinned memory ---

def torch_pinned_func(tensor_size, dtype):
    """
    Allocates pinned memory using PyTorch's built-in support.
    """
    return torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=True)

def to_posix_memalign_pinned_func(tensor_size, dtype):
    """
    Allocates pinned memory using the to_posix_memalign_pinned function.
    """
    unpinned_tensor = torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=False)
    return to_posix_memalign_pinned(unpinned_tensor)

def to_cuda_host_alloc_pinned_func(tensor_size, dtype):
    """
    Allocates pinned memory using the to_cuda_host_alloc_pinned function.
    """
    unpinned_tensor = torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=False)
    return to_cuda_host_alloc_pinned(unpinned_tensor)

def zeros_cuda_host_alloc_pinned_func(tensor_size, dtype):
    """
    Allocates zero-initialized pinned memory using zeros_cuda_host_alloc_pinned.
    """
    return zeros_cuda_host_alloc_pinned((tensor_size,), dtype)

from functools import partial

get_torch_overhead = partial(get_overhead, pinned_alloc_func=torch_pinned_func)
get_posix_memalign_overhead = partial(get_overhead, pinned_alloc_func=to_posix_memalign_pinned_func)
get_cuda_host_alloc_overhead = partial(get_overhead, pinned_alloc_func=to_cuda_host_alloc_pinned_func)
get_zeros_cuda_host_alloc_overhead = partial(get_overhead, pinned_alloc_func=zeros_cuda_host_alloc_pinned_func)