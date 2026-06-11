import random
import numpy as np
import torch
from transformers import set_seed, AutoTokenizer
from typing import Tuple
from argparse import Namespace

def get_vocab_size(model_name: str) -> int:
    if model_name == "t5gemma2-4b-4b":
        from .model.t5gemma2 import T5Gemma2Config
        config = T5Gemma2Config.for_4b_4b()
        return config.vocab_size
    elif model_name == "lfm2.5-8b-a1b": 
        from .model.lfm2_moe import Lfm2MoeConfig
        config = Lfm2MoeConfig.for_8b_a1b()       
        return config.vocab_size
    elif model_name == "Qwen/Qwen2.5-7B-Instruct-1M" or model_name == "mistralai/Mistral-Nemo-Instruct-2407":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vocab = tokenizer.get_vocab()
        return len(vocab)
    else:
        raise ValueError(f"Unsupported model_name {model_name}. Please contact create issue for supporting the model.")

def get_param_size(model_name: str) -> int:
    if model_name == "t5gemma2-4b-4b":
        return 7760198656
    elif model_name == "lfm2.5-8b-a1b": 
        return 8474147584
    elif model_name == "Qwen/Qwen2.5-7B-Instruct-1M":
        return 7615616512
    elif model_name == "mistralai/Mistral-Nemo-Instruct-2407":
        return 12247782400
    else:
        raise ValueError(f"Unsupported model_name {model_name}. Please contact create issue for supporting the model.")

def get_dummy_inputs_and_labels(batch_size: int, max_seq_length: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be > 1 to avoid pad token for flash-attn tests, got {vocab_size}")
    inputs = torch.randint(1, vocab_size, (batch_size, max_seq_length), dtype=torch.long)
    labels = torch.randint(1, vocab_size, (batch_size, max_seq_length), dtype=torch.long)
    return inputs, labels

def set_random_seed(seed):
    assert seed is not None, "seed must be provided"
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_rank_0(msg: str, rank: int) -> None:
    assert rank is not None, "rank must be provided"
    if rank == 0:
        print(msg)

def print_verbose(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)

def is_offload_optimizer(config_dict: dict) -> bool:
    zero_config_dict = config_dict["zero_optimization"]

    if "offload_optimizer" in zero_config_dict.keys():
        return True
    return False

def get_snap_shot_name(args: Namespace) -> str:
    def get_gradient_checkpointing_type(args: Namespace) -> str:
        if args.gradient_checkpointing:
            if args.offload_gradient_checkpointing:
                return "offload"
            return "normal"
        else:
            return "None"

    return f"{args.system_type}_bs{args.train_batch_size}_seq{args.max_seq_len}_liger{args.liger_kernel}_gradcheck{get_gradient_checkpointing_type(args)}_fa2{args.flash_attn_2}"
