import torch

from torch.nn import Module
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from cpu_lion import CPULion

def create_optimizer(model: Module, lr: float, weight_decay: float, betas_0: float, betas_1: float, offload_optimizer: bool, use_lion: bool) -> torch.optim.Optimizer:
    assert model is not None, "model must be provided"
    assert lr is not None, "lr must be provided"
    assert weight_decay is not None, "weight_decay must be provided"
    assert betas_0 is not None, "betas_0 must be provided"
    assert betas_1 is not None, "betas_1 must be provided"
    assert offload_optimizer is not None, "offload_optimizer must be provided"
    assert use_lion is not None, "use_lion must be provided"

    if use_lion:
        assert offload_optimizer, "LION optimizer only works with offload_optimizer=True"
        print("[INIT] Using CPU LION optimizer")
        return CPULion(model.parameters(), lr=lr, betas=(betas_0, betas_1), weight_decay=weight_decay)
    if offload_optimizer:
        print("[INIT] Using DeepSpeed CPU Adam optimizer")
        return DeepSpeedCPUAdam(model.parameters(), lr=lr, betas=(betas_0, betas_1), weight_decay=weight_decay, adamw_mode=True)
    print("[INIT] Using DeepSpeed Fused Adam optimizer")
    return FusedAdam(model.parameters(), lr=lr, betas=(betas_0, betas_1), weight_decay=weight_decay, adam_w_mode=True)
