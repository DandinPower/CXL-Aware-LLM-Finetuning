import torch
from torch.optim import Optimizer

from .cpu_lion_interface import create_lion, destroy_lion, lion_update

def zeros_cpu(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    return torch.zeros(size=shape, dtype=dtype, pin_memory=pin_memory, device="cpu")

class CPULion(Optimizer):
    """PyTorch optimizer wrapper over the local C++ CPU Lion kernel."""

    optimizer_id = 0
    zeros_cpu_for_momentums = zeros_cpu

    def __init__(
        self,
        model_params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        fp32_optimizer_states=True,
        should_log=False,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(model_params, defaults)

        self.opt_id = CPULion.optimizer_id
        CPULion.optimizer_id += 1
        self.fp32_optimizer_states = fp32_optimizer_states
        self._destroyed = False

        create_lion(
            self.opt_id,
            lr,
            betas[0],
            betas[1],
            weight_decay,
            should_log,
        )

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def destroy(self):
        if not self._destroyed:
            destroy_lion(self.opt_id)
            self._destroyed = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.device.type != "cpu":
                    raise RuntimeError(f"CPU Lion expects CPU parameters, got {p.device}.")
                if p.grad.device.type != "cpu":
                    raise RuntimeError(f"CPU Lion expects CPU gradients, got {p.grad.device}.")
                if not p.is_contiguous():
                    raise RuntimeError("CPU Lion expects contiguous parameters.")
                if not p.grad.is_contiguous():
                    raise RuntimeError("CPU Lion expects contiguous gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state_dtype = torch.float32 if self.fp32_optimizer_states else p.dtype
                    state["exp_avg"] = CPULion.zeros_cpu_for_momentums(
                        shape=p.data.shape,
                        dtype=state_dtype,
                        pin_memory=True,
                    )


                lion_update(
                    self.opt_id,
                    group["lr"],
                    beta1,
                    beta2,
                    group["weight_decay"],
                    p.data,
                    p.grad.data,
                    state["exp_avg"],
                )

        return loss


__all__ = ["CPULion"]
