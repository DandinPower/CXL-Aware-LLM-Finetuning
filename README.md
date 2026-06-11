# CXL-Aware-LLM-Finetuning

This repository is the official implementation of the paper "Analysis and Optimized CXL-Attached Memory Allocation for Long-Context LLM Fine-Tuning" by Yong-Cheng Liaw, Shuo-Han Chen. 

The paper is currently under review and preprint is available at https://arxiv.org/abs/2507.03305.


## Prerequisites

- Ensure an **NVIDIA driver** and **CUDA compiler** are installed.
    1. nvidia driver can install latest version due to the backward compatibility, at 2026-05, i installed 580.142 which supports up to CUDA 13.0.
    2. Since this project rely on Forked DeepSpeed v0.16.0 and Huggingface Transformer v4.47.1, which didn't support torch version with CUDA 13.0, so the recommended compatible torch version is 2.7.0 which supports up to CUDA 12.8, the CUDA 12.8 also are the at least version that support the Blackwell GPU architecture.
    3. With the torch runtime with CUDA 12.8, the DeepSpeed and this project pytorch extensions require to also be compiled with CUDA 12.8, so the CUDA toolkit with version 12.8 are needed to be installed.

- Install the following system dependencies (via `apt`):
    ```bash
    sudo apt-get update && sudo apt-get install build-essential python3-dev python3-venv
    ```

- This project also relies on NUMA libraries for the NUMA-aware allocation extension, so you need to install:
    ```bash
    sudo apt-get update && sudo apt-get install libnuma-dev numactl
    ```

## Install Python dependencies

1. Recommend to use a virtual environment (e.g. python3-venv or `uv`)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install dependencies via pip:
    ```bash
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    pip install transformers==4.47.1 accelerate==1.2.1 liger-kernel==0.5.10
    pip install huggingface_hub torch_tb_profiler matplotlib numpy wheel psutil pytest
    pip install flash-attn
    ``` 

3. (Optional) The flash-attn library is easily encountered miss alignment issue, it is recommended to run following command to make sure the flash-attn is properly installed and working:
    ```bash
    python verify_flash_attn.py
    ```

4. Install the forked DeepSpeed and extensions:
    ```bash
    cd extensions/deepspeed-numa-aware && pip install . && cd ../..
    cd extensions/zero-overhead-pinned-memory && pip install . && cd ../..
    cd extensions/numa-allocation && pip install . && cd ../..
    cd extensions/cpu-lion && pip install . && cd ../..
    ```

## Usage Explanation

Refer to [usage_tutorial.md](usage_tutorial.md) for detailed tutorial on how to run the experiments and reproduce the results in the paper.

## License

Unless otherwise stated in a file or third-party component, this repository is
licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for details.

Important exceptions:

- [utils/offload_grad_checkpoint.py](utils/offload_grad_checkpoint.py) is
  adapted from Unsloth Zoo and is licensed under LGPL-3.0-or-later. See
  [LICENSES/LGPL-3.0-or-later.txt](LICENSES/LGPL-3.0-or-later.txt) and
  [LICENSES/GPL-3.0-or-later.txt](LICENSES/GPL-3.0-or-later.txt).
- Vendored or modified third-party components retain their own copyright
  notices and license terms where stated in their source files or
  subdirectories.
