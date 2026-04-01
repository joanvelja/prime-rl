"""FP8 blockwise linear layer using DeepGEMM.

Drop-in replacement for nn.Linear that performs forward and backward passes
in FP8 using DeepGEMM's fp8_gemm_nt kernel. Requires SM90 (Hopper) GPUs.

Adapted from https://github.com/S1ro1/fp8
"""

from __future__ import annotations
import re

import deep_gemm
import torch
from torch import nn

from prime_rl.trainer.models.kernels.fp8_utils import (
    per_block_cast_to_fp8_triton,
    per_token_cast_to_fp8_triton,
)
from prime_rl.utils.logger import get_logger


class _FP8BlockwiseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, block_size, out_dtype=torch.bfloat16):
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1]).contiguous()
        x_fp8 = per_token_cast_to_fp8_triton(x_2d, False, block_size)
        weight_fp8 = per_block_cast_to_fp8_triton(weight, False, block_size)

        out = torch.empty((x_2d.size(0), weight.size(0)), device=x.device, dtype=out_dtype)
        deep_gemm.fp8_gemm_nt(x_fp8, weight_fp8, out)

        ctx.save_for_backward(x_2d, weight)
        ctx.x_shape = x_shape
        ctx.block_size = block_size
        return out.reshape(*x_shape[:-1], out.size(-1))

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, weight = ctx.saved_tensors
        block_size = ctx.block_size
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        grad_x = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_output_fp8 = per_token_cast_to_fp8_triton(grad_output_2d, False, block_size)
            weight_t = weight.transpose(0, 1).contiguous()
            weight_dx_fp8 = per_block_cast_to_fp8_triton(weight_t, False, block_size)
            grad_x_2d = torch.empty_like(x_2d)
            deep_gemm.fp8_gemm_nt(grad_output_fp8, weight_dx_fp8, grad_x_2d)
            grad_x = grad_x_2d.reshape(ctx.x_shape)

        if ctx.needs_input_grad[1]:
            grad_output_t = grad_output_2d.transpose(0, 1).contiguous()
            x_t = x_2d.transpose(0, 1).contiguous()
            grad_output_t_fp8 = per_token_cast_to_fp8_triton(grad_output_t, False, block_size)
            x_t_fp8 = per_token_cast_to_fp8_triton(x_t, False, block_size)
            grad_weight_fp32 = torch.zeros_like(weight, dtype=torch.float32)
            deep_gemm.fp8_gemm_nt(
                grad_output_t_fp8,
                x_t_fp8,
                grad_weight_fp32,
                c=grad_weight_fp32,
                recipe=(1, 1, 128),
            )
            grad_weight = grad_weight_fp32.to(weight.dtype)

        return grad_x, grad_weight, None, None


class Float8BlockwiseLinear(nn.Linear):
    """nn.Linear replacement that uses FP8 blockwise matmul via DeepGEMM.

    Requires:
    - SM90 (Hopper) GPU
    - bfloat16 inputs/weights
    - No bias
    - in_features and out_features divisible by 128
    """

    def __init__(self, *args, block_size: int = 128, dtype=torch.bfloat16, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FP8BlockwiseMM.apply(x, self.weight, self.block_size, torch.bfloat16)

    @classmethod
    def from_linear(cls, mod: nn.Linear) -> "Float8BlockwiseLinear":
        """Convert an existing nn.Linear to Float8BlockwiseLinear."""
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def replace_linear_with_fp8_blockwise_linear(model: nn.Module, ignore_modules: list[str] = ["lm_head", "model.layers.*.router.gate"]) -> None:
    logger = get_logger()
    logger.info("Replacing linear layers with FP8 blockwise linear layers")
    replaced_modules = []
    named_modules = dict(model.named_modules())
    for name, module in named_modules.items():
        if name in ignore_modules or any(re.match(pattern, name) for pattern in ignore_modules):
            continue
        if isinstance(module, nn.Linear):
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr_name, Float8BlockwiseLinear.from_linear(module))
            replaced_modules.append(name)

    logger.info(f"Replaced {len(replaced_modules)} linear layers with FP8 blockwise linear layers: {replaced_modules[:5]}...")
