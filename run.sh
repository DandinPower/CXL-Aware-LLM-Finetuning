# Shared Configurations, should not change for fair comparison
DS_CONFIG_PATH=ds_zero3_cpu_offloading.json
GRADIENT_CHECKPOINTING_STRATEGY="offload" # "none", "hf", "offload"
GRADIENT_ACCUMULATION_STEPS=1
LORA_DIM=0
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
BETA_0=0.9
BETA_1=0.95

# At least larget than 2 iterations, since the first iteration is excluded because of the initialization overhead and the second iteration is also excluded because of the warmup effect. 
NUM_TRAIN_ITERATIONS=7


# Model-specific Configurations, the following are the models we have tested, you can uncomment the one you want to run (MODEL_NAME, NUM_LAYERS, HIDDEN_SIZE).
# 1. Mistral-Nemo-Instruct-2407: https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
# MODEL_NAME=mistralai/Mistral-Nemo-Instruct-2407
# NUM_LAYERS=40
# HIDDEN_SIZE=5120
# 2. Qwen2.5-7B-Instruct-1M: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-1M
NUM_LAYERS=28
HIDDEN_SIZE=3584
# 3. T5GEMMA2-4B-4B: https://huggingface.co/google/t5gemma-2-4b-4b
# MODEL_NAME=t5gemma2-4b-4b
# NUM_LAYERS=68
# HIDDEN_SIZE=2560
# 4. LFM2.5-8B-A1B: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B
# MODEL_NAME=lfm2.5-8b-a1b
# NUM_LAYERS=24
# HIDDEN_SIZE=2048

NUM_GPUS=1
BATCH_SIZE_AND_SEQ_LENGTH_CONFIGS=(
    "1 4096"
    "8 4096"
    "16 4096"
    "1 32768"
    "2 32768"
    "3 32768"
)

FRAMEWORK_OVERHEAD_ALLOCATION_NODE=1    # Pin framework overhead to one NUMA node so local/CXL memory budgets remain
# comparable across NUMA allocation strategies. This mainly covers transient
# CPU memory use such as model loading and page cache effects.

NUMA_ALLOCATION_STRATEGY="cxl_aware"  # "none", "local_only", "naive_interleaving", "cxl_aware"
# none: No special NUMA allocation strategy is applied. 
# local_only: Allocate all memory on the local NUMA nodes, which require local_numa_mapping configuration.
# naive_interleaving: Interleave memory allocation across all NUMA nodes, which require local_numa_mapping and cxl_numa_mapping configurations.
# cxl_aware: Allocate memory on NUMA nodes based on the access pattern of the memory access pattern, which require
# 1. local_numa_mapping and cxl_numa_mapping configurations to specify which NUMA nodes are local and which are CXL-attached, and
# 2. local_memory_budget_bytes and cxl_memory_budget_bytes configurations to specify the memory budget.
LOCAL_NUMA_MAPPING="0"  # If have multiple local NUMA nodes, use spaces to separate, e.g., "0 1"
CXL_NUMA_MAPPING="1"    # If have multiple CXL-attached NUMA nodes, use spaces to separate, e.g., "2 3"
LOCAL_MEMORY_BUDGET_BYTES=$((100 * 1024 * 1024 * 1024)) # 100 GB
CXL_MEMORY_BUDGET_BYTES=$((256 * 1024 * 1024 * 1024)) # 256 GB

for CONFIG in "${BATCH_SIZE_AND_SEQ_LENGTH_CONFIGS[@]}"; do
    read -r PER_DEVICE_TRAIN_BATCH_SIZE MAX_SEQ_LENGTH <<< "$CONFIG"
    echo "Running with batch size ${PER_DEVICE_TRAIN_BATCH_SIZE} and max sequence length ${MAX_SEQ_LENGTH}"

    # ensure the cache is clean
    rm -rf ~/.cache/torch_extensions/
    # If the system is running on NFS path, set a non-NFS path for Triton cache to avoid potential hang. 
    export TRITON_CACHE_DIR=~/.triton_cache

    # ensure the page cache is clean
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

    numactl --interleave=$FRAMEWORK_OVERHEAD_ALLOCATION_NODE deepspeed --num_gpus $NUM_GPUS training.py --model_name $MODEL_NAME --world_size $NUM_GPUS --ds_config_path $DS_CONFIG_PATH \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --num_layers $NUM_LAYERS --hidden_size $HIDDEN_SIZE \
        --num_train_iterations $NUM_TRAIN_ITERATIONS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS --max_seq_len $MAX_SEQ_LENGTH \
        --lora_dim $LORA_DIM \
        --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --beta_0 $BETA_0 --beta_1 $BETA_1 \
        --liger_kernel --flash_attn_2 \
        --gradient_checkpointing_strategy $GRADIENT_CHECKPOINTING_STRATEGY \
        --numa_allocation_strategy $NUMA_ALLOCATION_STRATEGY \
	    --local_numa_mapping $LOCAL_NUMA_MAPPING \
        --cxl_numa_mapping $CXL_NUMA_MAPPING --local_memory_budget_bytes $LOCAL_MEMORY_BUDGET_BYTES --cxl_memory_budget_bytes $CXL_MEMORY_BUDGET_BYTES \
        2>&1 | tee "bs${PER_DEVICE_TRAIN_BATCH_SIZE}_seq${MAX_SEQ_LENGTH}.log"
    # For LION Optimizer ablation, add --use_lion flag and the system will automatically switch to LION optimizer and apply NUMA-aware allocation for LION optimizer.
done
