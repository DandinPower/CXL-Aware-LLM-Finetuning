import time
import torch

from zero_overhead_pinned_memory import to_posix_memalign_pinned, to_cuda_host_alloc_pinned, zeros_cuda_host_alloc_pinned

# --- Allocation Approaches ---

def approach1(tensor_size, dtype):
    """
    Approach 1 (Baseline):
    Create a CPU tensor that is already pinned using PyTorch's built-in support.
    """
    tensor = torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=True)
    return tensor

def approach2(tensor_size, dtype):
    """
    Approach 2:
    Create a CPU tensor without pinned memory and then pin it manually using to_posix_memalign_pinned.
    """
    tensor = to_posix_memalign_pinned(
        torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=False)
    )
    return tensor

def approach3(tensor_size, dtype):
    """
    Approach 3:
    Create a CPU tensor without pinned memory and then pin it manually using to_cuda_host_alloc_pinned.
    """
    tensor = to_cuda_host_alloc_pinned(
        torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=False)
    )
    return tensor

def approach4(tensor_size, dtype):
    """
    Approach 4:
    Create a CPU pinned tensor using zeros_cuda_host_alloc_pinned.
    """
    tensor = zeros_cuda_host_alloc_pinned(tensor_size, dtype)
    return tensor

# --- Benchmark Helpers ---

def run_benchmark(timing_func, method, tensor_size, dtype, num_iters=100):
    """
    Runs the given timing function over multiple iterations and returns the average time.
    
    Parameters:
      timing_func: A callable of the form f(method, tensor_size, dtype) -> float.
      method: The tensor allocation function.
      tensor_size: The shape of the tensor to create.
      dtype: The tensor data type.
      num_iters: The number of iterations to average.
    """
    times = []
    for _ in range(num_iters):
        times.append(timing_func(method, tensor_size, dtype))
    return sum(times) / len(times)

def timing_func_creation(method, tensor_size, dtype):
    """
    Times the creation (including pinning) of a CPU tensor.
    Uses time.time() for wall-clock timing.
    """
    start = time.time()
    _ = method(tensor_size, dtype)
    return time.time() - start

def timing_func_transfer(method, tensor_size, dtype, device='cuda'):
    """
    Times the transfer of a pinned CPU tensor to the GPU.
    Uses torch.cuda.Event for accurate timing (in milliseconds) because the transfer is asynchronous.
    """
    # Create CUDA events for timing.
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    # Create the tensor and ensure any CPU work is finished.
    t = method(tensor_size, dtype)
    torch.cuda.synchronize()
    
    # Record events around the transfer.
    start_evt.record()
    _ = t.to(device)
    end_evt.record()
    torch.cuda.synchronize()  # Wait for the GPU copy to complete.
    
    return start_evt.elapsed_time(end_evt)  # Elapsed time in ms.

def warmup(method, tensor_size, dtype, device='cuda', num_iters=10):
    """
    Runs several warm-up iterations to ensure that any one-time overhead (e.g., CUDA kernel launches)
    does not skew the measured transfer times.
    """
    for _ in range(num_iters):
        t = method(tensor_size, dtype)
        _ = t.to(device)
        torch.cuda.synchronize()

# --- Main Benchmarking Code ---

if __name__ == '__main__':
    # Define tensor parameters.
    tensor_size = (1024, 1024, 1024)  # Adjust this as needed.
    dtype = torch.float32
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit(0)

    num_iters = 5  # Number of iterations for each benchmark.

    # Dictionary mapping method names to allocation functions.
    methods = {
        "pytorch(baseline)": approach1,
        "to_posix_memalign_pinned": approach2,
        "to_cuda_host_alloc_pinned": approach3,
        "zeros_cuda_host_alloc_pinned": approach4
    }

    # Set the baseline method as Approach 1.
    baseline_name = "pytorch(baseline)"
    baseline_method = methods[baseline_name]

    # --- Benchmark: CPU Creation and Pinning ---
    print("CPU creation + pinning time (per iteration):")
    baseline_creation_time = run_benchmark(timing_func_creation, baseline_method, tensor_size, dtype, num_iters)
    print(f"  {baseline_name}: {baseline_creation_time * 1e3:.3f} ms (baseline)")

    for name, method in methods.items():
        if name == baseline_name:
            continue
        creation_time = run_benchmark(timing_func_creation, method, tensor_size, dtype, num_iters)
        ratio = creation_time / baseline_creation_time
        print(f"  {name}: {creation_time * 1e3:.3f} ms, {ratio:.2f}x baseline")

    # --- Benchmark: GPU Transfer ---
    print("\nGPU transfer time (per iteration):")
    # Warm up the baseline method.
    warmup(baseline_method, tensor_size, dtype, device='cuda', num_iters=10)
    baseline_transfer_time = run_benchmark(
        lambda m, ts, dt: timing_func_transfer(m, ts, dt, device='cuda'),
        baseline_method, tensor_size, dtype, num_iters
    )
    print(f"  {baseline_name}: {baseline_transfer_time:.3f} ms (baseline)")

    for name, method in methods.items():
        if name == baseline_name:
            continue
        warmup(method, tensor_size, dtype, device='cuda', num_iters=10)
        transfer_time = run_benchmark(
            lambda m, ts, dt: timing_func_transfer(m, ts, dt, device='cuda'),
            method, tensor_size, dtype, num_iters
        )
        ratio = transfer_time / baseline_transfer_time
        print(f"  {name}: {transfer_time:.3f} ms, {ratio:.2f}x baseline")
