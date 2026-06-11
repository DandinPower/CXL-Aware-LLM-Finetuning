# Copyright 2026 Yong-Cheng Liaw <yongchengliaw.cs14@nycu.edu.tw>
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Modified gradient checkpointing utility based on Unsloth Zoo:
# https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/gradient_checkpointing.py
#
# Local modifications:
# 1. Adjusted gradient checkpointing to support multiple GPUs.
# 2. Added NUMA-aware pinned buffer management for offloaded activations.
# 3. Added more robust handling of tensor and non-tensor arguments in the
#    checkpointed function.

import torch
from packaging.version import Version
from .zero_overhead_patch import _zeros_cpu_zero_overhead
from .numa_allocation_patch import zeros_cpu_for_checkpointed

torch_version = torch.__version__

if Version(torch_version) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")


class PinnedBufferManager:
    def __init__(self):
        pass
        
    def setup_buffer(self, num_layers: int, batch_size: int, seq_length: int, hidden_size: int, dtype: torch.dtype, rank: int, is_numa: bool) -> None:
        buffer_shape = (batch_size, seq_length, hidden_size)
        if is_numa == False:
            allocator = _zeros_cpu_zero_overhead
        else:
            allocator = zeros_cpu_for_checkpointed
        self.buffers = [allocator(shape=buffer_shape, dtype=dtype, pin_memory=True) for _ in range(num_layers)]
        
    def get_buffer(self) -> torch.Tensor:
        assert len(self.buffers) > 0, "Run out of buffer"
        return self.buffers.pop()
    
    def release_buffer(self, buffer: torch.Tensor) -> None:
        self.buffers.append(buffer)

buffer_manager = PinnedBufferManager()

class Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        device = hidden_states.device
        saved_hidden_states = buffer_manager.get_buffer()
        saved_hidden_states.copy_(hidden_states, non_blocking=True)

        ctx.args = []
        ctx.tensor_arg_indices = []
        tensor_args = []
        for index, arg in enumerate(args):
            if torch.is_tensor(arg):
                ctx.args.append(None)
                ctx.tensor_arg_indices.append(index)
                tensor_args.append(arg)
            else:
                ctx.args.append(arg)

        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states, *tensor_args)
        ctx.forward_function = forward_function
        ctx.device = device
        return output

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, *grad_outputs):
        saved_hidden_states = ctx.saved_tensors[0]
        saved_tensor_args = ctx.saved_tensors[1:]
        hidden_states = saved_hidden_states.to(ctx.device, non_blocking = True).detach()
        buffer_manager.release_buffer(saved_hidden_states)

        detached_hidden_states = hidden_states.detach()
        detached_hidden_states.requires_grad_(ctx.needs_input_grad[1])

        detached_args = list(ctx.args)
        for tensor_idx, arg_idx in enumerate(ctx.tensor_arg_indices):
            detached_arg = saved_tensor_args[tensor_idx].detach()
            detached_arg.requires_grad_(ctx.needs_input_grad[arg_idx + 2])
            detached_args[arg_idx] = detached_arg

        with torch.enable_grad():
            outputs = ctx.forward_function(detached_hidden_states, *detached_args)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs_with_grad = []
        grad_outputs_with_grad = []
        for output, grad_output in zip(outputs, grad_outputs):
            if torch.is_tensor(output) and output.requires_grad:
                outputs_with_grad.append(output)
                grad_outputs_with_grad.append(grad_output)
        if not outputs_with_grad:
            raise RuntimeError("offloaded gradient checkpoint got no differentiable outputs")

        torch.autograd.backward(outputs_with_grad, grad_outputs_with_grad)

        hidden_states_grad = detached_hidden_states.grad
        arg_grads = []
        for arg in detached_args:
            if torch.is_tensor(arg):
                arg_grads.append(arg.grad)
            else:
                arg_grads.append(None)
        return (None, hidden_states_grad, *arg_grads)

@torch._disable_dynamo
def offloaded_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
    return Offloaded_Gradient_Checkpointer.apply(function, *args)

def patch_offloaded_gradient_checkpointing():
    import torch.utils
    torch.utils.checkpoint.checkpoint = offloaded_gradient_checkpoint
    try:
        import transformers.modeling_utils

        transformers.modeling_utils.checkpoint = offloaded_gradient_checkpoint
    except Exception:
        # Keep this utility usable in pure-torch environments.
        pass
