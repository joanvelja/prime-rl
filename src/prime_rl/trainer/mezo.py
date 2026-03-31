import random
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer


class MeZO(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.projected_grad = None
        self.zo_random_seed = None

    def _perturb_parameters(self, scaling_factor: float = 1.0):
        torch.manual_seed(self.zo_random_seed)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    z = torch.randn_like(p.data)
                    p.data.add_(z, alpha=scaling_factor * group["eps"])

    def _forward_pass(self, model: nn.Module, inputs: dict, compute_loss: Callable):
        model.eval()
        with torch.inference_mode():
            loss = compute_loss(model, inputs)
            if loss.ndim > 0:
                loss = loss.mean()
        return loss.detach()

    def step(self, closure: Callable = None):
        raise RuntimeError("MeZO requires zo_step and zo_update methods. Use those instead.")

    def zo_step(
        self,
        model: nn.Module,
        inputs: dict,
        compute_loss: Callable,
    ):
        self.zo_random_seed = random.randint(0, 2**31 - 1)

        self._perturb_parameters(scaling_factor=1.0)
        loss1 = self._forward_pass(model, inputs, compute_loss)

        self._perturb_parameters(scaling_factor=-2.0)
        loss2 = self._forward_pass(model, inputs, compute_loss)

        self.projected_grad = ((loss1 - loss2) / (2 * self.defaults["eps"])).item()

        self._perturb_parameters(scaling_factor=1.0)

        return loss1

    def zo_update(self):
        torch.manual_seed(self.zo_random_seed)

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group.get("weight_decay", 0.0)

            for p in group["params"]:
                if p.grad is not None:
                    z = torch.randn_like(p.data)

                    if weight_decay > 0.0:
                        p.data.add_(self.projected_grad * z + weight_decay * p.data, alpha=-lr)
                    else:
                        p.data.add_(self.projected_grad * z, alpha=-lr)
