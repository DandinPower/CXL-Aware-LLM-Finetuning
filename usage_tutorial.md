# Usage Tutorial: Running Local, Naive CXL Interleaving, and CXL-Aware Experiments

This tutorial is a lab runbook for running the three NUMA allocation strategies used by this project:

| Strategy | `run.sh` value | What it tests |
| --- | --- | --- |
| Local only | `NUMA_ALLOCATION_STRATEGY="local_only"` | Allocate all patched CPU tensors on local DRAM only. |
| Naive local+CXL interleaving | `NUMA_ALLOCATION_STRATEGY="naive_interleaving"` | Interleave all patched CPU tensors across local DRAM and CXL memory. |
| CXL-aware allocation | `NUMA_ALLOCATION_STRATEGY="cxl_aware"` | Place tensor groups on local DRAM, CXL memory, or both based on access priority and memory budget. |

The main workflow is to edit `run.sh`, run one clean experiment for each strategy, and compare only the final `[RESULT]` metrics printed by `training.py`.

## 1. Before You Start

Follow the root [`README.md`](../README.md) first. This tutorial assumes:

- The Python environment is active.
- PyTorch, Transformers, Liger Kernel, FlashAttention, and the forked DeepSpeed are installed.
- The local project extensions are installed:
  - `extensions/zero-overhead-pinned-memory`
  - `extensions/numa-allocation`
  - `extensions/cpu-lion`
- `numactl` and `libnuma-dev` are installed.
- `python verify_flash_attn.py` passes.
- You have `sudo` permission, because `run.sh` drops Linux page cache before each run.

This tutorial uses the lab setup below as the worked example:

- GPU: one NVIDIA RTX 5090.
- Local DRAM NUMA node: `0`.
- CXL memory NUMA node: `1`.
- Local memory experiment budget: `100 GiB`.
- CXL memory experiment budget: `256 GiB`.
- Framework overhead allocation node: `1`, the CXL node.

## 2. Confirm the Machine Topology

Check the NUMA and GPU topology before running experiments:

```bash
numactl --hardware
lscpu
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
```

On the example machine, `numactl --hardware` shows two NUMA nodes:

```text
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 ... 63
node 0 size: ...
node 1 cpus:
node 1 size: ...
```

In this setup, node `0` is local DRAM because it owns the CPU cores, and node `1` is CXL memory because it has memory but no CPUs. If your topology is different, update `LOCAL_NUMA_MAPPING`, `CXL_NUMA_MAPPING`, and `FRAMEWORK_OVERHEAD_ALLOCATION_NODE` in `run.sh`.

## 3. Fair Comparison Rules

Keep these settings fixed across the three strategy runs:

- Same model.
- Same batch size and sequence length.
- Same number of GPUs.
- Same DeepSpeed config.
- Same optimizer.
- Same gradient checkpointing strategy.
- Same Liger Kernel and FlashAttention settings.
- Same local and CXL memory budgets.
- Same framework overhead placement.

The default `run.sh` already follows this structure. Change only the strategy value between runs unless you are intentionally creating a new experiment.

### Framework Overhead Placement

`run.sh` starts DeepSpeed under:

```bash
numactl --interleave=$FRAMEWORK_OVERHEAD_ALLOCATION_NODE deepspeed ...
```

The example sets:

```bash
FRAMEWORK_OVERHEAD_ALLOCATION_NODE=1
```

This intentionally places framework overhead on the CXL node for all strategies. In Qwen7B testing, around `30 GiB` of reserved framework memory occupied memory but did not represent the tensor allocation strategy being evaluated. Pinning that overhead to node `1` keeps the local DRAM budget comparable across `local_only`, `naive_interleaving`, and `cxl_aware`.

Do not change this value between strategy runs.

### Local DRAM Physical Limit

For fair comparison between naive interleaving and CXL-aware allocation, the physical local DRAM visible to Linux should be limited to match the intended local budget.

The reason is Linux NUMA interleaving behavior. Interleaved allocation is performed in a round robin manner across the selected node mask until one node runs out of available memory. If node `0` exposes much more local DRAM than the expected local budget, the interleaving allocation policy, which both the naive interleaving and CXL aware approaches may use, can consume more local memory than expected, making the comparison unfair.


For the example budget:

```bash
LOCAL_MEMORY_BUDGET_BYTES=$((100 * 1024 * 1024 * 1024))
```

the lab setup limits node `0` to `128 GiB` of visible physical memory. The extra space gives the OS room to run while keeping the experiment close to a `100 GiB` local-memory budget.

The limiting method is machine-specific:

1. Inspect the machine memory map and NUMA layout.
2. Add the appropriate `memmap` reservation or memory limit to the GRUB kernel command line.
3. Run `sudo update-grub`.
4. Reboot.
5. Re-check `numactl --hardware`.

For example, a GRUB setting may contain a machine-specific reservation like:

```text
GRUB_CMDLINE_LINUX_DEFAULT='quiet splash memmap=130G\$128G'
```

Do not copy that value blindly. The address range must match the target machine. After changing the memory limit, run a memory bandwidth test to confirm local DRAM still uses the expected memory-channel performance. In the lab setup, the local limit is checked to preserve the expected four-channel local DRAM speed.

## 4. Configure `run.sh`

Open `run.sh` and keep the shared experiment settings fixed:

```bash
DS_CONFIG_PATH=ds_zero3_cpu_offloading.json
GRADIENT_CHECKPOINTING_STRATEGY="offload"
GRADIENT_ACCUMULATION_STEPS=1
LORA_DIM=0
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
BETA_0=0.9
BETA_1=0.95
NUM_TRAIN_ITERATIONS=7
NUM_GPUS=1
```

`NUM_TRAIN_ITERATIONS` must be greater than `2`. `training.py` removes the first two iterations from the final statistics to avoid initialization and warmup effects.

Use one model configuration at a time. The default example is:

```bash
MODEL_NAME=lfm2.5-8b-a1b
NUM_LAYERS=24
HIDDEN_SIZE=2048
```

Use one or more batch-size and sequence-length configuration:

```bash
BATCH_SIZE_AND_SEQ_LENGTH_CONFIGS=(
    "1 4096"
)
```

Set the lab NUMA mapping and budgets:

```bash
FRAMEWORK_OVERHEAD_ALLOCATION_NODE=1
LOCAL_NUMA_MAPPING="0"
CXL_NUMA_MAPPING="1"
LOCAL_MEMORY_BUDGET_BYTES=$((100 * 1024 * 1024 * 1024))
CXL_MEMORY_BUDGET_BYTES=$((256 * 1024 * 1024 * 1024))
```

## 5. Run the Three Strategies

Run one clean experiment for each strategy. `run.sh` already removes the PyTorch extension cache and drops Linux page cache before launching DeepSpeed.

The output log name does not include the allocation strategy:

```bash
bs${PER_DEVICE_TRAIN_BATCH_SIZE}_seq${MAX_SEQ_LENGTH}.log
```

Rename the log after each run so the next strategy does not overwrite it.

### Strategy 1: Local Only

- Note: When running local_only, there is no need to limit the local DRAM with GRUB because all tensors will be allocated on local DRAM by design. 

Set:

```bash
NUMA_ALLOCATION_STRATEGY="local_only"
```

Run:

```bash
bash run.sh
mv model_lfm2.5-8b-a1b_bs1_seq4096.log local_only_lfm2.5-8b-a1b_bs1_seq4096.log
```

Expected initialization signal:

```text
[INIT] Applying local_only NUMA-aware allocation strategy
[INIT] MOMENTUMS_NODEMASK: [0]
[INIT] VARIANCES_NODEMASK: [0]
[INIT] MASTER_GRADIENTS_NODEMASK: [0]
[INIT] MASTER_WEIGHTS_NODEMASK: [0]
[INIT] COMPUTE_WEIGHTS_NODEMASK: [0]
[INIT] ACTIVATIONS_NODEMASK: [0]
[INIT] COMPUTE_GRADIENTS_NODEMASK: [0]
```

### Strategy 2: Naive Local+CXL Interleaving

Set:

```bash
NUMA_ALLOCATION_STRATEGY="naive_interleaving"
```

Run:

```bash
bash run.sh
mv model_lfm2.5-8b-a1b_bs1_seq4096.log naive_interleaving_lfm2.5-8b-a1b_bs1_seq4096.log
```

Expected initialization signal:

```text
[INIT] Applying naive_interleaving NUMA-aware allocation strategy
[INIT] MOMENTUMS_NODEMASK: [0, 1]
[INIT] VARIANCES_NODEMASK: [0, 1]
[INIT] MASTER_GRADIENTS_NODEMASK: [0, 1]
[INIT] MASTER_WEIGHTS_NODEMASK: [0, 1]
[INIT] COMPUTE_WEIGHTS_NODEMASK: [0, 1]
[INIT] ACTIVATIONS_NODEMASK: [0, 1]
[INIT] COMPUTE_GRADIENTS_NODEMASK: [0, 1]
```

### Strategy 3: CXL-Aware Allocation

Set:

```bash
NUMA_ALLOCATION_STRATEGY="cxl_aware"
```

Run:

```bash
bash run.sh
mv model_lfm2.5-8b-a1b_bs1_seq4096.log cxl_aware_lfm2.5-8b-a1b_bs1_seq4096.log
```

For the default LFM2.5-8B-A1B, batch size `1`, sequence length `4096`, `100 GiB` local budget, and `256 GiB` CXL budget, the CXL-aware allocator prioritizes frequently accessed tensor groups for local DRAM first. Once the local budget is consumed, later tensor groups move to CXL or use a mixed local+CXL node mask.

With DeepSpeed CPU Adam, the expected allocation order is:

| Tensor group | Typical mask in this example |
| --- | --- |
| Momentum | `[0]` |
| Variance | `[0]` |
| Master gradients | `[0]` |
| Master weights | `[0, 1]` |
| Compute weights | `[1]` |
| Activations | `[1]` |
| Compute gradients | `[1]` |

## 6. Compare Results

For each log, use only the final `[RESULT]` lines:

```text
[RESULT] Peak VRAM Usage(per gpu): ...
[RESULT] Each Iteration Latency (rank0): ...
[RESULT] Iteration Latency Stats(rank0): avg=..., min=..., max=..., std=... s
[RESULT] FWD Duration Stats(rank0): ...
[RESULT] BWD Duration Stats(rank0): ...
[RESULT] STEP Duration Stats(rank0): ...
[RESULT] Tokens(total): ...
[RESULT] Throughput(total): ... (token/s)
```

The most important comparison metrics are:

- `Iteration Latency Stats(rank0): avg=...`
- `Throughput(total): ...`
- `FWD Duration Stats(rank0)`
- `BWD Duration Stats(rank0)`
- `STEP Duration Stats(rank0)`
- `Peak VRAM Usage(per gpu)`

The log also print per-node process memory usage near the end:

```text
Per-node process memory usage (in MBs) for PID ...
         Node 0 Node 1  Total
Private   ...
Total     ...
```

Use this as a sanity check that the selected strategy placed CPU memory on the intended nodes.

## 7. Optional Debug Baseline

`training.py` also supports:

```bash
NUMA_ALLOCATION_STRATEGY="none"
```

Use this only for debugging. It bypasses the project NUMA-aware tensor allocation strategy and uses the default PyTorch allocation path with the zero-overhead pinned-memory patch.

## Contact

For any questions about the correct way to run experiments, please feel free to reach out to the project maintainers. The email addresses are `yongchengliaw.cs14@nycu.edu.tw`.
