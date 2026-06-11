import torch

# Import both our allocation functions
from zero_overhead_pinned_memory.experiments import get_torch_overhead, get_posix_memalign_overhead, get_cuda_host_alloc_overhead, get_zeros_cuda_host_alloc_overhead

if __name__ == "__main__":
    # Define two test cases: one with a power-of-2 size and one with a non power-of-2 size.
    test_cases = [
        ("Power of 2", int(4 * 1024**3)),
        ("Non Power of 2", int(4.01 * 1024**3))
    ]
    
    # Define the methods to test.
    # Here, we compare PyTorch's built-in pinned memory against our own method.
    methods = {
        "pytorch(baseline)": get_torch_overhead,
        "to_posix_memalign_pinned": get_posix_memalign_overhead,
        "to_cuda_host_alloc_pinned": get_cuda_host_alloc_overhead,
        "zeros_cuda_host_alloc_pinned": get_zeros_cuda_host_alloc_overhead
    }
    
    # Run tests for each case and method.
    for label, size in test_cases:
        print(f"\n--- {label} ---")
        for method_name, get_overhead_func in methods.items():
            unpinned, pinned, overhead = get_overhead_func(size, torch.float16)
            print(f"{method_name}:")
            print(f"  Unpinned memory: {unpinned} bytes")
            print(f"  Pinned memory:   {pinned} bytes")
            print(f"  Overhead:        {overhead:.2f}")
