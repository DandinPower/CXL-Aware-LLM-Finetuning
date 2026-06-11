import os
import subprocess
import torch
from numa_allocation import zeros_numa_onnode_cpu, zeros_numa_on_nodemask_cpu

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

def get_numastat_output() -> str:
    """
    Retrieve NUMA statistics for the current process.

    Returns:
        str: Output from the `numastat` command for the current process.
    """
    pid = os.getpid()
    return run_command(["numastat", "-c", "-p", str(pid)])

def test_pin_memory() -> None:
    """
    Verify that the `pin_memory` flag sets the tensor’s pinned status correctly.

    When `pin_memory` is False, the tensor should not be pinned.
    When it is True, the tensor should be pinned.
    """
    print("Running test_pin_memory...")
    dtype = torch.float
    shape = (1024, 1024, 1024)
    
    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=False)
    assert not tensor.is_pinned(), "Tensor should not be pinned when pin_memory is False."
    
    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=True)
    assert tensor.is_pinned(), "Tensor should be pinned when pin_memory is True."
    
    tensor = zeros_numa_on_nodemask_cpu(shape=shape, dtype=dtype, pin_memory=False, interleave_numa_nodes=[0])
    assert not tensor.is_pinned(), "Tensor should not be pinned when pin_memory is False."
    
    tensor = zeros_numa_on_nodemask_cpu(shape=shape, dtype=dtype, pin_memory=True, interleave_numa_nodes=[0])
    assert tensor.is_pinned(), "Tensor should be pinned when pin_memory is True."
    
    print("Passed test_pin_memory!\n")

def test_tensor_metadata() -> None:
    """
    Ensure that the allocated tensor has the expected metadata.

    This test verifies:
      - The tensor’s data type (dtype) is correct.
      - The tensor’s shape matches the requested dimensions.
      - The tensor is zero-initialized.
    """
    print("Running test_tensor_metadata...")
    dtypes = [torch.bfloat16, torch.float16, torch.float]
    shape = (1024, 1024, 1024)
    
    for dtype in dtypes:
        tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=False)
        assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
        assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
        assert torch.all(tensor == 0), "Tensor is not zero-initialized."
    
    for dtype in dtypes:
        tensor = zeros_numa_on_nodemask_cpu(shape=shape, dtype=dtype, pin_memory=False, interleave_numa_nodes=[0])
        assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
        assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
        assert torch.all(tensor == 0), "Tensor is not zero-initialized."
 
    print("Passed test_tensor_metadata!\n")

def test_priority_numa_nodes_default() -> None:
    """
    Test default NUMA allocation behavior.

    When no preferred NUMA node is specified, the allocation should follow the system's default
    policy. Under `numactl --interleave=all`, the memory allocation should be interleaved across
    available NUMA nodes.

    The output from `numastat` is printed for manual inspection.

    Example output:
        Per-node process memory usage (in MBs) for PID <pid> (python)
                Node 0 Node 1 Node 2 Node 3 Node 4 Total
                ------ ------ ------ ------ ------ -----
        Huge          0      0      0      0      0     0
        Heap         17     17     17     17     17    86
        Stack         0      0      0      0      0     0
        Private     997    846    847    843    843  4376
        -------  ------ ------ ------ ------ ------ -----
        Total      1014    864    864    860    860  4462
    """
    print("Running test_priority_numa_nodes_default...")
    shape = (1024, 1024, 1024)
    dtype = torch.float
    
    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=False)
    print("numastat output:\n", get_numastat_output())
    del tensor
    print("Passed test_priority_numa_nodes_default!\n")

def test_priority_numa_nodes_with_one_value() -> None:
    """
    Test NUMA allocation when a single preferred NUMA node is specified.

    When `priority_numa_nodes=[0]` is provided, the allocation should attempt to use NUMA node 0 exclusively.
    The output from `numastat` is printed for manual inspection.

    Example output:
        Per-node process memory usage (in MBs) for PID <pid> (python)
                Node 0 Node 1 Node 2 Node 3 Node 4 Total
                ------ ------ ------ ------ ------ -----
        Huge          0      0      0      0      0     0
        Heap         17     17     17     17     17    86
        Stack         0      0      0      0      0     0
        Private    4274     27     27     24     24  4376
        -------  ------ ------ ------ ------ ------ -----
        Total      4291    44     45     41     41  4462
    """
    print("Running test_priority_numa_nodes_with_one_value...")
    shape = (1024, 1024, 1024)
    dtype = torch.float

    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=False, priority_numa_nodes=[0])
    print("numastat output:\n", get_numastat_output())
    del tensor
    print("Passed test_priority_numa_nodes_with_one_value!\n")

def test_priority_numa_nodes_with_multiple_value() -> None:
    """
    Test NUMA allocation when multiple preferred NUMA nodes are specified.

    When `priority_numa_nodes=[0, 1]` is provided, the allocator will attempt to use node 0 first,
    falling back to node 1 if necessary. The output from `numastat` is printed for manual inspection.

    Example output:
        Per-node process memory usage (in MBs) for PID <pid> (python)
                Node 0 Node 1 Node 2 Node 3 Node 4 Total
                ------ ------ ------ ------ ------ -----
        Huge          0      0      0      0      0     0
        Heap         17     17     17     17     17    86
        Stack         0      0      0      0      0     0
        Private    4274     27     28     24     24  4376
        -------  ------ ------ ------ ------ ------ -----
        Total      4291    44     45     41     41  4462
    """
    print("Running test_priority_numa_nodes_with_multiple_value...")
    shape = (1024, 1024, 1024)
    dtype = torch.float

    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=False, priority_numa_nodes=[0, 1])
    print("numastat output:\n", get_numastat_output())
    del tensor
    print("Passed test_priority_numa_nodes_with_multiple_value!\n")

def test_priority_numa_nodes_with_invalid_nodes() -> None:
    """
    Test NUMA allocation behavior when invalid NUMA nodes are specified.

    If invalid nodes (e.g., -1) are provided, the allocation should fall back to the default policy.
    When a mix is provided (e.g., [-1, 1]), it should ignore the invalid node and use the valid one.
    The output from `numastat` is printed for manual inspection.

    Example output for invalid node only:
        Per-node process memory usage (in MBs) for PID <pid> (python)
                Node 0 Node 1 Node 2 Node 3 Node 4 Total
                ------ ------ ------ ------ ------ -----
        Huge          0      0      0      0      0     0
        Heap         17     17     17     17     17    86
        Stack         0      0      0      0      0     0
        Private     997    846    847    843    843  4376
        -------  ------ ------ ------ ------ ------ -----
        Total      1014    864    864    860    860  4462

    Example output for invalid then valid nodes:
        Per-node process memory usage (in MBs) for PID <pid> (python)
                Node 0 Node 1 Node 2 Node 3 Node 4 Total
                ------ ------ ------ ------ ------ -----
        Huge          0      0      0      0      0     0
        Heap         17     17     17     17     17    86
        Stack         0      0      0      0      0     0
        Private     178   4123     28     24     24  4376
        -------  ------ ------ ------ ------ ------ -----
        Total       195   4140     45     41     41  446
    """
    print("Running test_priority_numa_nodes_with_invalid_nodes...")
    shape = (1024, 1024, 1024)
    dtype = torch.float

    # Case 1: Only invalid node provided.
    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=False, priority_numa_nodes=[-1])
    print("numastat output (invalid node only):\n", get_numastat_output())
    del tensor
    
    # Case 2: First node invalid, second node valid.
    tensor = zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=False, priority_numa_nodes=[-1, 1])
    print("numastat output (invalid then valid):\n", get_numastat_output())
    del tensor
    print("Passed test_priority_numa_nodes_with_invalid_nodes!\n")

def test_zeros_numa_on_nodemask_cpu_with_one_nodes() -> None:
    """
    Test zeros_numa_on_nodemask_cpu allocation with a single NUMA node.

    Example output:
        Per-node process memory usage (in MBs) for PID <pid> (python)
                Node 0 Node 1 Node 2 Node 3 Node 4 Total
                ------ ------ ------ ------ ------ -----
        Huge          0      0      0      0      0     0
        Heap         17     17     17     17     17    86
        Stack         0      0      0      0      0     0
        Private    4274     27     28     24     24  4377
        -------  ------ ------ ------ ------ ------ -----
        Total      4291     45     45     41     41  4463
    """
    print("Running test_zeros_numa_on_nodemask_cpu_with_one_node...")
    shape = (1024, 1024, 1024)
    dtype = torch.float

    tensor = zeros_numa_on_nodemask_cpu(
        shape=shape, dtype=dtype, pin_memory=False, interleave_numa_nodes=[0]
    )
    print("numastat output:\n", get_numastat_output())
    del tensor
    print("Passed test_zeros_numa_on_nodemask_cpu_with_one_node!\n")

def test_zeros_numa_on_nodemask_cpu_with_multiple_nodes() -> None:
    """
    Test zeros_numa_on_nodemask_cpu allocation with multiple NUMA nodes.

    Example output:
        Per-node process memory usage (in MBs) for PID 1663157 (python)
                Node 0 Node 1 Node 2 Node 3 Node 4 Total
                ------ ------ ------ ------ ------ -----
        Huge          0      0      0      0      0     0
        Heap         20     20     20     20     20   101
        Stack         0      0      0      0      0     0
        Private    2253   2096     45     42     41  4478
        -------  ------ ------ ------ ------ ------ -----
        Total      2273   2116     66     62     61  4579
    """
    print("Running test_zeros_numa_on_nodemask_cpu_with_multiple_nodes...")
    shape = (1024, 1024, 1024)
    dtype = torch.float

    tensor = zeros_numa_on_nodemask_cpu(
        shape=shape, dtype=dtype, pin_memory=False, interleave_numa_nodes=[0, 1]
    )
    print("numastat output:\n", get_numastat_output())
    del tensor
    print("Passed test_zeros_numa_on_nodemask_cpu_with_multiple_nodes!\n")

def main() -> None:
    """
    Run all test cases sequentially and report the results.
    """
    test_functions = [
        test_tensor_metadata,
        test_zeros_numa_on_nodemask_cpu_with_one_nodes,
        test_zeros_numa_on_nodemask_cpu_with_multiple_nodes,
        test_priority_numa_nodes_default,
        test_priority_numa_nodes_with_one_value,
        test_priority_numa_nodes_with_multiple_value,
        test_priority_numa_nodes_with_invalid_nodes,
    ]
    
    for test in test_functions:
        try:
            test()
        except AssertionError as error:
            print(f"Test {test.__name__} FAILED: {error}")
            raise
    print("All tests passed successfully.")

if __name__ == "__main__":
    main()
