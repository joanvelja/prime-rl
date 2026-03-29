import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.lora.base import MultiLoRAModule, get_lora_num_tokens, get_multilora_scaling
from prime_rl.trainer.models.layers.lora.multi_moe import _run_lora_for_loop, _run_lora_grouped_mm
from prime_rl.trainer.models.layers.moe import GroupedExperts


class MultiLoRAPERFTE(MultiLoRAModule):
    """
    GroupedExperts + multi-LoRA as a single bypass path.
    Runs the base MoE unmodified and adds a single LoRA: out = moe(x) + B @ A @ x.
    """

    def __init__(
        self,
        base_layer: GroupedExperts,
        rank: int,
        n_adapters: int,
        alpha: float = 32.0,
        dropout: float = 0.0,
        use_grouped_mm: bool = True,
    ):
        super().__init__(base_layer)
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be > 0")

        self.num_experts = base_layer.num_experts
        self.dim = base_layer.w1.shape[2]

        if rank % 8 != 0 or self.dim % 8 != 0:
            use_grouped_mm = False

        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.use_grouped_mm = use_grouped_mm

        self._lora_num_tokens = get_lora_num_tokens()
        self._scaling_factors = get_multilora_scaling()

        # Single LoRA pair per adapter: A maps dim -> rank, B maps rank -> dim
        self.lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.dim,
                        device=base_layer.w1.device,
                        dtype=base_layer.w1.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.dim,
                        rank,
                        device=base_layer.w1.device,
                        dtype=base_layer.w1.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self, index: int | None = None) -> None:
        if index is None:
            for i in range(self.n_adapters):
                self.reset_parameters(i)
        else:
            nn.init.kaiming_uniform_(self.lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[index])

    def named_parameters_for_adapter(self, idx: int) -> list[tuple[str, nn.Parameter]]:
        return [
            ("lora_A", self.lora_A[idx]),  # [num_experts, rank, dim]
            ("lora_B", self.lora_B[idx]),  # [num_experts, dim, rank]
        ]

    def get_lora_param_counts(self) -> tuple[int, int]:
        adapter_params = self.lora_A[0].numel() + self.lora_B[0].numel()
        adapted_params = self.base_layer.w1.numel() + self.base_layer.w2.numel() + self.base_layer.w3.numel()
        return adapter_params, adapted_params

    def state_dict_for_adapter(self, idx: int) -> dict[str, torch.Tensor]:
        """Get state dict for a specific adapter in per-expert format.

        Returns:
            Dict with keys like "{expert_id}.lora_A.weight" and "{expert_id}.lora_B.weight".
        """
        state_dict = {}

        detached_a = self.lora_A[idx].detach()
        detached_b = self.lora_B[idx].detach()

        if isinstance(detached_a, DTensor):
            detached_a = detached_a.full_tensor()
            detached_b = detached_b.full_tensor()

        for expert_id in range(self.num_experts):
            state_dict[f"{expert_id}.lora_A.weight"] = detached_a[expert_id].clone()
            state_dict[f"{expert_id}.lora_B.weight"] = detached_b[expert_id].clone()

        return state_dict

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        # Base MoE computation (EP handled by @expert_parallel decorator)
        y_moe = self.base_layer(x, num_tokens_per_expert)

        # Select active adapter
        adapter_idx = self._lora_num_tokens.argmax().item()
        lora_a = self.lora_A[adapter_idx]
        lora_b = self.lora_B[adapter_idx]
        scaling = self._scaling_factors[adapter_idx].item()

        # EP handling for LoRA path
        permuted_indices = None
        if isinstance(lora_a, DTensor):
            from torchtitan.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M
            from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

            lora_a = lora_a.to_local()
            lora_b = lora_b.to_local()

            experts_per_ep_rank = lora_a.shape[0]
            num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank

            with torch.no_grad():
                permuted_indices, num_tokens_per_expert, _ = generate_permute_indices(
                    num_tokens_per_expert,
                    experts_per_ep_rank,
                    num_ep_ranks,
                    x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M,
                    TOKEN_GROUP_ALIGN_SIZE_M,
                )

            x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
            input_shape = x.shape
            x = x[permuted_indices, :]

        # LoRA path
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        lora_x = self.lora_dropout(x)

        if self.use_grouped_mm:
            lora_out = _run_lora_grouped_mm(lora_x, lora_a, lora_b, offsets)
        else:
            lora_out = _run_lora_for_loop(lora_x, lora_a, lora_b, num_tokens_per_expert)

        # EP unpermute
        if permuted_indices is not None:
            if lora_out.shape[0] < len(permuted_indices):
                num_padding = len(permuted_indices) - lora_out.shape[0]
                lora_out = torch.vstack((lora_out, lora_out.new_zeros((num_padding, lora_out.shape[-1]))))
            out_unpermuted = lora_out.new_zeros(input_shape)
            out_unpermuted[permuted_indices, :] = lora_out
            lora_out = out_unpermuted[:-1]

        return y_moe + scaling * lora_out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"n_adapters={self.n_adapters}, num_experts={self.num_experts}, "
            f"alpha={self.alpha}, dropout={self.lora_dropout}, "
            f"use_grouped_mm={self.use_grouped_mm})"
        )
