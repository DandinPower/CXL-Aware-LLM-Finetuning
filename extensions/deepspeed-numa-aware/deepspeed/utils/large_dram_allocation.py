import torch

def zeros_cpu(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    return torch.zeros(size=shape, dtype=dtype, pin_memory=pin_memory, device="cpu")

zeros_cpu_for_momentums = zeros_cpu
zeros_cpu_for_variances = zeros_cpu
zeros_cpu_for_master_weights = zeros_cpu
zeros_cpu_for_master_gradients = zeros_cpu
zeros_cpu_for_compute_weights = zeros_cpu
zeros_cpu_for_compute_gradients = zeros_cpu