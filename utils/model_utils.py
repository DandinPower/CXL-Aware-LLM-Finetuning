import torch

from accelerate import init_empty_weights
from torch.nn import Module
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from liger_kernel.transformers import AutoLigerKernelForCausalLM

from .model.t5gemma2 import T5Gemma2ForConditionalGeneration, T5Gemma2Config
from .model.lfm2_moe import Lfm2MoeForCausalLM, Lfm2MoeConfig
from .offload_grad_checkpoint import patch_offloaded_gradient_checkpointing
from .lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible

def init_all_parameters_constant_(module: torch.nn.Module, value: float = 0.01) -> None:
    for param in module.parameters():
        if param.requires_grad:
            param.data.fill_(value)

def show_model_param_numel(model: Module):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INIT] Total number of parameters in the model: {total_params}")


def create_model_by_deepspeed(ds_config: dict, model_name: str, lora_dim: int, liger_kernel: bool, gradient_checkpointing: bool, offload_gradient_checkpointing: bool, flash_attn_2: bool) -> Module:
    assert model_name is not None, "model_name must be provided"
    assert liger_kernel is not None, "liger_kernel must be provided"
    assert gradient_checkpointing is not None, "gradient_checkpoint must be provided"
    assert flash_attn_2 is not None, "flash_attn_2 must be provided"
    assert lora_dim is not None, "lora_dim must be provided, if not enable lora, it should be 0"
    
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if model_name.startswith("t5gemma2"):
        if model_name == "t5gemma2-tiny":
            config = T5Gemma2Config.for_tiny()
        elif model_name == "t5gemma2-4b-4b":
            config = T5Gemma2Config.for_4b_4b()
        else:
            raise ValueError(f"Unsupported model_name {model_name} for T5Gemma2")
        
        with init_empty_weights(include_buffers=False):
            model = T5Gemma2ForConditionalGeneration(
                config,
                enable_flash_attn2=flash_attn_2,
                enable_liger_kernel=liger_kernel,
            )
        model = model.to_empty(device="cpu")
        init_all_parameters_constant_(model, value=0.01)
        show_model_param_numel(model)
    elif model_name.startswith("lfm2"):
        if model_name == "lfm2.5-tiny":
            config = Lfm2MoeConfig.for_tiny()
        elif model_name == "lfm2.5-8b-a1b":
            config = Lfm2MoeConfig.for_8b_a1b()
        else:
            raise ValueError(f"Unsupported model_name {model_name} for Lfm2Moe")

        with init_empty_weights(include_buffers=False):
            model = Lfm2MoeForCausalLM(
                config,
                enable_flash_attn2=flash_attn_2,
                enable_liger_kernel=liger_kernel,
            )
        model = model.to_empty(device="cpu")
        init_all_parameters_constant_(model, value=0.01)
        show_model_param_numel(model)
    else:
        model_class = AutoModelForCausalLM
        if liger_kernel:
            model_class = AutoLigerKernelForCausalLM

        if flash_attn_2:
            model = model_class.from_pretrained(model_name, use_cache=False, attn_implementation="flash_attention_2")
        else:
            # model = model_class.from_pretrained(model_name, use_cache=False)
            model = model_class.from_pretrained(model_name, use_cache=False, attn_implementation="eager")

    if lora_dim > 0:
        model = convert_linear_layer_to_lora(model, "layers.", lora_dim)
        model = only_optimize_lora_parameters(model)
        model = make_model_gradient_checkpointing_compatible(model)

    if offload_gradient_checkpointing:
        assert gradient_checkpointing, "Need to enable gradient_checkpointing with offload_gradient_checkpointing"
        patch_offloaded_gradient_checkpointing()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model