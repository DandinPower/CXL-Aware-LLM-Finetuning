import os
import subprocess
import torch
import deepspeed
import cpu_lion
from numa_allocation import zeros_numa_on_nodemask_cpu
from .cxl_aware_allocator import latency_first_allocation, convert_interleave_ratio_to_numa_node_mask

# Default
MOMENTUMS_NODEMASK = None
VARIANCES_NODEMASK = None
MASTER_GRADIENTS_NODEMASK = None
MASTER_WEIGHTS_NODEMASK = None
COMPUTE_WEIGHTS_NODEMASK = None
ACTIVATIONS_NODEMASK = None
COMPUTE_GRADIENTS_NODEMASK = None

def assert_numa_allocation_configurations():
    assert MOMENTUMS_NODEMASK is not None, "MOMENTUMS_NODEMASK is not set"
    # assert VARIANCES_NODEMASK is not None, "VARIANCES_NODEMASK is not set" -> LION optimizer doesn't have variances, so it can be None
    assert MASTER_GRADIENTS_NODEMASK is not None, "MASTER_GRADIENTS_NODEMASK is not set"
    assert MASTER_WEIGHTS_NODEMASK is not None, "MASTER_WEIGHTS_NODEMASK is not set"
    assert COMPUTE_WEIGHTS_NODEMASK is not None, "COMPUTE_WEIGHTS_NODEMASK is not set"
    assert ACTIVATIONS_NODEMASK is not None, "ACTIVATIONS_NODEMASK is not set"
    assert COMPUTE_GRADIENTS_NODEMASK is not None, "COMPUTE_GRADIENTS_NODEMASK is not set"
    print("[INIT] All NUMA allocation configurations are set correctly.")
    print(f"[INIT] MOMENTUMS_NODEMASK: {MOMENTUMS_NODEMASK}")
    print(f"[INIT] VARIANCES_NODEMASK: {VARIANCES_NODEMASK}")
    print(f"[INIT] MASTER_GRADIENTS_NODEMASK: {MASTER_GRADIENTS_NODEMASK}")
    print(f"[INIT] MASTER_WEIGHTS_NODEMASK: {MASTER_WEIGHTS_NODEMASK}")
    print(f"[INIT] COMPUTE_WEIGHTS_NODEMASK: {COMPUTE_WEIGHTS_NODEMASK}")
    print(f"[INIT] ACTIVATIONS_NODEMASK: {ACTIVATIONS_NODEMASK}")
    print(f"[INIT] COMPUTE_GRADIENTS_NODEMASK: {COMPUTE_GRADIENTS_NODEMASK}")

def patch_numa_allocation_configurations(node_mask: list[int]):
    """
    node_mask is a list of NUMA node ids, e.g. [0, 1, 2, 3] for 4 NUMA nodes. This function will set all the node masks to the same node mapping, which means all the tensors will be allocated on the interleaved NUMA nodes. This is a naive configuration that doesn't differentiate the importance of different tensors. This function can use for local only strategy or CXL naive interleaving strategy.
    For local only strategy, the node_mask should be all the local NUMA nodes, e.g. [0, 1] for 2 local NUMA nodes.
    For CXL naive interleaving strategy, the node_mask should be all the NUMA nodes including local and CXL memory nodes, e.g. [0, 1] for 1 local NUMA node and 1 CXL memory node.
    """
    global MOMENTUMS_NODEMASK, VARIANCES_NODEMASK, MASTER_GRADIENTS_NODEMASK, MASTER_WEIGHTS_NODEMASK, COMPUTE_WEIGHTS_NODEMASK, ACTIVATIONS_NODEMASK, COMPUTE_GRADIENTS_NODEMASK    
    MOMENTUMS_NODEMASK = node_mask
    VARIANCES_NODEMASK = node_mask
    MASTER_GRADIENTS_NODEMASK = node_mask
    MASTER_WEIGHTS_NODEMASK = node_mask
    COMPUTE_WEIGHTS_NODEMASK = node_mask
    ACTIVATIONS_NODEMASK = node_mask
    COMPUTE_GRADIENTS_NODEMASK = node_mask


def patch_numa_allocation_configurations_for_cxl_aware_strategy(param_size: int, num_hidden_layers: int, hidden_size: int, batch_size: int, seq_length: int, local_budget: int, cxl_budget: int, local_node_mapping: list[int], cxl_node_mapping: list[int], use_lion: bool):
    """
    local_node_mapping is a list of local NUMA node IDs, e.g., [0, 1] for two local NUMA nodes.
    cxl_node_mapping is a list of CXL memory NUMA node IDs, e.g., [2, 3] for two CXL memory nodes.
    local_budget and cxl_budget are the memory budgets for local and CXL memory in bytes, e.g., local_budget=72102410241024 for 72 GB of local memory and cxl_budget=256102410241024 for 256 GB of CXL memory. Remember that the allocator will try to fit this budget rather than the actual remaining space in the system, so the interleaving ratio is based on this budget rather than the actual memory size of the system. However, during the actual allocation, if it uses an interleaving strategy, the Linux kernel will try to allocate memory pages in a round robin manner across all nodes in the node mask until some nodes are out of memory. Therefore, the actual allocated memory size on each NUMA node may be different from the budget if the budget is smaller or larger than the actual memory size of the NUMA node.
    The paper addresses this issue by manually limiting the available physical memory size of local memory using a GRUB_CMDLINE_LINUX_DEFAULT='quiet splash memmap=194G\$64G' style configuration, which can 1. keep the local memory node multichannel speed and 2. also ensure that the interleaving ratio can correctly reflect the actual memory size ratio I want to evaluate.
    """
    global MOMENTUMS_NODEMASK, VARIANCES_NODEMASK, MASTER_GRADIENTS_NODEMASK, MASTER_WEIGHTS_NODEMASK, COMPUTE_WEIGHTS_NODEMASK, ACTIVATIONS_NODEMASK, COMPUTE_GRADIENTS_NODEMASK
    num_cxl_devices = len(cxl_node_mapping)
    if use_lion:
        momentums_size = param_size * 4
        master_gradients_size = param_size * 4
        master_weights_size = param_size * 4
        compute_weights_size = param_size * 2
        activations_size = batch_size * seq_length * hidden_size * num_hidden_layers * 2
        compute_gradients_size = param_size * 2

        group_items_sizes = {
            1: [momentums_size, master_gradients_size, master_weights_size],
            2: [compute_weights_size],
            3: [activations_size],
            4: [compute_gradients_size]
        }

        allocations = latency_first_allocation(local_budget, cxl_budget, num_cxl_devices, group_items_sizes)
        MOMENTUMS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_1_item_0"][1], local_node_mapping, cxl_node_mapping)
        VARIANCES_NODEMASK = None # LION optimizer doesn't have variances
        MASTER_GRADIENTS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_1_item_1"][1], local_node_mapping, cxl_node_mapping)
        MASTER_WEIGHTS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_1_item_2"][1], local_node_mapping, cxl_node_mapping)
        COMPUTE_WEIGHTS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_2_item_0"][1], local_node_mapping, cxl_node_mapping)
        ACTIVATIONS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_3_item_0"][1], local_node_mapping, cxl_node_mapping)
        COMPUTE_GRADIENTS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_4_item_0"][1], local_node_mapping, cxl_node_mapping)
    else:
        momentums_size = param_size * 4
        variances_size = param_size * 4
        master_gradients_size = param_size * 4
        master_weights_size = param_size * 4
        compute_weights_size = param_size * 2
        activations_size = batch_size * seq_length * hidden_size * num_hidden_layers * 2
        compute_gradients_size = param_size * 2

        group_items_sizes = {
            1: [momentums_size, variances_size, master_gradients_size, master_weights_size],
            2: [compute_weights_size],
            3: [activations_size],
            4: [compute_gradients_size]
        }

        allocations = latency_first_allocation(local_budget, cxl_budget, num_cxl_devices, group_items_sizes)
        MOMENTUMS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_1_item_0"][1], local_node_mapping, cxl_node_mapping)
        VARIANCES_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_1_item_1"][1], local_node_mapping, cxl_node_mapping)
        MASTER_GRADIENTS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_1_item_2"][1], local_node_mapping, cxl_node_mapping)
        MASTER_WEIGHTS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_1_item_3"][1], local_node_mapping, cxl_node_mapping)
        COMPUTE_WEIGHTS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_2_item_0"][1], local_node_mapping, cxl_node_mapping)
        ACTIVATIONS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_3_item_0"][1], local_node_mapping, cxl_node_mapping)
        COMPUTE_GRADIENTS_NODEMASK = convert_interleave_ratio_to_numa_node_mask(allocations["level_4_item_0"][1], local_node_mapping, cxl_node_mapping)

def get_numastat_output() -> str:
    """
    Retrieve NUMA statistics for the current process.

    Returns:
        str: Output from the `numastat` command for the current process.
    """
    def run_command(cmd: list[str]) -> str:
        """
        Run a shell command and return its output as a string.

        Args:
            cmd (list[str]): The command to run.

        Returns:
            str: The standard output from the command, or an error message.
        """
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Command failed with exit code {result.returncode}"
        return result.stdout.strip()
    pid = os.getpid()
    return run_command(["numastat", "-c", "-p", str(pid)])

def zeros_cpu_for_checkpointed(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    This one doesn't need to be patched because it is already patched from the offload_grad_checkpoint.
    ngpus * batch * seq * hidden * dtype
    """
    return zeros_numa_on_nodemask_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, interleave_numa_nodes=ACTIVATIONS_NODEMASK)


def _zeros_cpu_for_compute_gradients(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    2 * model_size (if gradient accumulation dtype is bf16)
    """
    return zeros_numa_on_nodemask_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, interleave_numa_nodes=COMPUTE_GRADIENTS_NODEMASK)

def _zeros_cpu_for_compute_weights(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    2 * model_size (if bf16/fp16 mixed precision)
    """
    return zeros_numa_on_nodemask_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, interleave_numa_nodes=COMPUTE_WEIGHTS_NODEMASK)

def _zeros_cpu_for_master_gradients(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    4 * model_size
    """
    return zeros_numa_on_nodemask_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, interleave_numa_nodes=MASTER_GRADIENTS_NODEMASK)

def _zeros_cpu_for_master_weights(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    4 * model_size
    """
    return zeros_numa_on_nodemask_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, interleave_numa_nodes=MASTER_WEIGHTS_NODEMASK)

def _zeros_cpu_for_momentums(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    4 * model_size
    """
    return zeros_numa_on_nodemask_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, interleave_numa_nodes=MOMENTUMS_NODEMASK)

def _zeros_cpu_for_variances(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    4 * model_size
    """
    return zeros_numa_on_nodemask_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, interleave_numa_nodes=VARIANCES_NODEMASK)

def patch_deepspeed_cpu_tensor_allocation(rank: int):
    print("[INIT] Apply numa awareness allocation deepspeed")
    deepspeed.ops.adam.cpu_adam.zeros_cpu_for_momentums = _zeros_cpu_for_momentums
    deepspeed.ops.adam.cpu_adam.zeros_cpu_for_variances = _zeros_cpu_for_variances
    deepspeed.runtime.zero.stage3.zeros_cpu_for_master_weights = _zeros_cpu_for_master_weights
    deepspeed.runtime.zero.stage3.zeros_cpu_for_master_gradients = _zeros_cpu_for_master_gradients
    deepspeed.runtime.zero.stage3.zeros_cpu_for_compute_weights = _zeros_cpu_for_compute_weights
    deepspeed.runtime.zero.stage3.zeros_cpu_for_compute_gradients = _zeros_cpu_for_compute_gradients

def patch_cpu_lion_tensor_allocation():
    print("[INIT] Apply numa awareness allocation for CPU LION optimizer")
    cpu_lion.CPULion.zeros_cpu_for_momentums = _zeros_cpu_for_momentums