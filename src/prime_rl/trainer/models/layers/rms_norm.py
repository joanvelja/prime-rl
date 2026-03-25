from dataclasses import dataclass

import torch
from torch import nn

from quack.rmsnorm import rmsnorm


@dataclass
class RMSNormConfig:
    hidden_size: int
    eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, config: RMSNormConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rmsnorm(hidden_states, self.weight, eps=self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
