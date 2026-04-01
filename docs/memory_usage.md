# Reducing memory usage

While most of our parallelism techniques in prime-rl are designed to scale training up (FSDP, EP, CP, ...), we also provide many tools to scale training down that allow training large MoE models on a limited amount of GPUs.

These techniques target the trainer part of prime-rl.

## TLDR: config to use for maximum memory usage reduction with correct throughput

```toml
[trainer.model]
impl = "custom"
attn = "flash_attention_2"
fused_lm_head_token_chunk_size = 1024
ep = 8
cp = 2
optim_cpu_offload = true

[trainer.model.compile]

[trainer.model.ac]
mode = "full"
freq = 1

[trainer.model.ac_offloading]
max_inflight_activations = 1
```

## Activation checkpointing

Activation checkpointing discards intermediate activations during the forward pass and recomputes them during the backward pass, trading compute for memory.

If `trainer.model.ac` is unset, supported custom implementations default to selective AC on the cheapest target:

```toml
[trainer.model.ac]
mode = "selective"
targets = ["norm"]
freq = 1
```

HF or other non-custom implementations do not auto-enable activation checkpointing when `trainer.model.ac` is unset.

To explicitly enable full-layer checkpointing, use:

```toml
[trainer.model.ac]
freq = 1
```

This is equivalent to:

```toml
[trainer.model.ac]
mode = "full"
freq = 1
```

## Selective activation checkpointing tuning

Selective AC is only supported with the custom model implementation. It lets you add memory savings incrementally before switching all the way to `mode = "full"`.

The orders below are rough tuning heuristics from best memory-saved/recompute tradeoff to worst. Start on the left and add targets as needed. The runtime treats `targets` as a set, so the order in your config file does not matter.

```toml
[trainer.model]
impl = "custom"

[trainer.model.ac]
mode = "selective"
targets = ["norm", "attn_proj", "moe_act"]
```

Available targets by model family:
- `llama`: `norm`, `attn_proj`, `mlp`
- `minimax_m2`: `norm`, `moe_act`, `attn_proj`, `routed_experts`
- `qwen3_moe` / `glm4_moe`: `norm`, `mlp`, `moe_act`, `attn_proj`, `routed_experts`
- `afmoe`: `norm`, `mlp`, `moe_act`, `attn_proj`, `linear_attn`, `routed_experts`
- `qwen3_5_moe` / `nemotron_h`: `norm`, `moe_act`, `attn_proj`, `linear_attn`, `routed_experts`
- `glm_moe_dsa`: `norm`, `mlp`, `mla_up_proj`, `moe_act`, `attn_proj`, `routed_experts`

Notes:
- These lists are unions across the model. On mixed-architecture families, not every target applies to every layer.
- `qwen3_moe`, `glm4_moe`, and `glm_moe_dsa` contain both dense and MoE layers. `mlp` applies to dense layers, while `moe_act` and `routed_experts` apply to MoE layers.
- `afmoe` only exposes `linear_attn` on its sliding-window attention layers.
- `qwen3_5_moe` only exposes `linear_attn` on its GatedDeltaNet layers.
- `nemotron_h` only exposes `linear_attn` on Mamba layers, while `moe_act` and `routed_experts` only apply to its LatentMoE layers.
- `glm_moe_dsa` only exposes `mla_up_proj` on its sparse MLA attention layers.

When selective tuning is not enough, switch to `mode = "full"`.

## Activation offloading

Activation offloading offloads the activations to CPU to reduce the memory usage of the trainer. It can be used in combination with activation checkpointing.

If you set `trainer.model.ac_offloading` without `trainer.model.ac`, prime-rl also enables explicit full-layer activation checkpointing.

To use activation offloading with the custom-model selective setup, configure both explicitly:

```toml
[trainer.model]
impl = "custom"

[trainer.model.ac]
mode = "selective"
targets = ["norm"]

[trainer.model.ac_offloading]
max_inflight_activations = 5
```

To use activation offloading with full AC, use:

```toml
[trainer.model.ac]
freq = 1

[trainer.model.ac_offloading]
max_inflight_activations = 5
```

## Chunk loss

Chunk loss splits the loss computation into smaller chunks to reduce the memory usage of the trainer.

To enable it, use:

```toml
[trainer.model]
fused_lm_head_token_chunk_size = auto
```

## Expert parallelism

While expert parallelism splits the weights of the experts across all GPUs like FSDP, using EP still reduces memory usage by reducing the communication size and therefore the FSDP buffer.

EP is only available for models with MoE layers using the custom model implementation.

```toml
[trainer.model]
impl = "custom"
ep = 8
```

## Context parallelism

Context parallelism splits the context into smaller chunks to reduce the memory usage of the activations. We don't advise using CP across multiple nodes (i.e., increasing CP beyond 8).

CP is only available for certain models and only with the custom model implementation.

```toml
[trainer.model]
impl = "custom"
cp = 2
```

We recommend CP 2 or CP 4 for most 128K sequence length training runs. Can be pushed to 8.

## torch compile

Enabling `torch.compile` can reduce the memory usage for certain model architectures, especially MoE with the custom model implementation.

```toml
[trainer.model.compile]
```

## CPU Optimizer offloading

Offloading the optimizer states to CPU can reduce the memory usage of the trainer significantly, especially at low GPU counts where the optimizer states take a lot of memory as they won't be sharded enough.

In RL, in contrast with pretraining, we end up with many gradient accumulation steps, so the cost of offloading the optimizer states is not as high as in pretraining, and indeed barely noticeable.

```toml
[trainer.model]
optim_cpu_offload = true
```

## :warning: FSDP CPU offloading

FSDP CPU offloading offloads the parameters, gradients, and optimizer states to CPU to reduce the memory usage of the trainer.

This will make training significantly slower and is not recommended most of the time.

```toml
[trainer.model]
fsdp_cpu_offload = true
```

## :warning: Lora training

LoRA training significantly reduces the memory usage of the trainer at the cost of smaller gradient updates.

```toml
[trainer.model.lora]
rank = 8
```
