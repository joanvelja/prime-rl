from argparse import Namespace
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias
import warnings

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_config import BaseConfig

from prime_rl.configs.shared import BaseModelConfig, SlurmConfig
from prime_rl.utils.utils import rgetattr, rsetattr

# TODO: Set thinking/ solution budget


class WeightBroadcastConfig(BaseConfig):
    """Configures weight broadcast settings."""

    type: Annotated[Literal["nccl", "filesystem"], Field(description="The type of weight broadcast to use.")] = (
        "filesystem"
    )


# Valid vLLM max_lora_rank values (from vllm/config/lora.py)
# TODO: on newer vLLM, can import via `get_args(vllm.config.lora.MaxLoRARanks)`
VALID_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)

# vLLM all2all backend options for expert-parallel deployments.
All2AllBackend = Literal[
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
    "flashinfer_all2allv",
    "naive",
    "pplx",
]


class ServerConfig(BaseConfig):
    """Configures the inference server."""

    host: Annotated[str | None, Field(description="The host to bind to.")] = None
    port: Annotated[int, Field(description="The port to bind to.")] = 8000


class RouterConfig(BaseConfig):
    """Configures the vllm-router."""

    server_config: Annotated[ServerConfig, Field(description="The server configuration.")] = ServerConfig()

    policy: Annotated[Literal["round_robin", "consistent_hash"], Field(description="The routing policy to use.")] = (
        "round_robin"
    )

    extra: Annotated[dict[str, Any], Field(description="Extra arguments to pass to the vllm-router as is.")] = {}


class ParallelConfig(BaseConfig):
    """Configures multi-node and multi-GPU setups through different types of parallelism (TP, DP, PP)."""

    tp: Annotated[
        int,
        Field(
            description="The tensor parallel size. It is passed to vLLM as `--tensor-parallel-size`",
        ),
    ] = 1

    dp: Annotated[
        int,
        Field(
            ge=1,
            description="The data parallel size. It is passed to vLLM as `--data-parallel-size`",
        ),
    ] = 1

    def __str__(self) -> str:
        return f"tp={self.tp} dp={self.dp}"


class ModelConfig(BaseModelConfig):
    """Configures the inference model. Most arguments are passed directly to the vLLM LLM class (https://docs.vllm.ai/en/latest/api/vllm.LLM.html)."""

    dtype: Annotated[
        Literal["auto", "float16", "bfloat16", "float32"],
        Field(
            description="Data type for model weights and activations. If 'auto' will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. Passed to vLLM as `--dtype`",
        ),
    ] = "auto"

    max_model_len: Annotated[
        int | None,
        Field(
            description="Maximum model context length. If None, will use the maximum context length from model config. Passed to vLLM as `--max-model-len`",
        ),
    ] = None

    enforce_eager: Annotated[
        bool,
        Field(
            description="Whether to enforce eager mode. If False, will use PyTorch eager and cuda graphs in hybrid for maximal performance. Passed to vLLM as `--enforce-eager`",
        ),
    ] = False

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code. Passed to vLLM engine init",
        ),
    ] = False

    tool_call_parser: Annotated[
        str | None,
        Field(
            description="The tool call parser to use. Passed to vLLM as `--tool-call-parser`. "
            'Set to "auto" to infer from the model name.',
        ),
    ] = None

    reasoning_parser: Annotated[
        str | None,
        Field(
            description="Parser for extracting reasoning content from model outputs. Passed to vLLM as `--reasoning-parser`. Setting this enables reasoning mode.",
        ),
    ] = None

    rope_scaling: Annotated[
        dict[str, Any] | str | None,
        Field(
            description='RoPE scaling configuration as a dict. For YaRN, use: {rope_type="yarn", factor=4.0, original_max_position_embeddings=32768} or. Passed to vLLM as `--rope-scaling`.',
        ),
    ] = None


class BaseRuntimeInferenceConfig(BaseConfig):
    model: ModelConfig = Field(default_factory=ModelConfig)

    parallel: ParallelConfig = ParallelConfig()

    server_config: Annotated[ServerConfig, Field(description="The server configuration.")] = ServerConfig()

    enable_lora: Annotated[
        bool,
        Field(
            description="Whether to enable LORA. Passed to vLLM as `--enable-lora`",
        ),
    ] = False

    max_loras: Annotated[
        int,
        Field(
            description="The maximum number of LoRAs to use. Passed to vLLM as `--max-loras`",
        ),
    ] = 8

    # TODO: The default value is very high because our areal impl for lora isn't ideal
    # We add a lora with the same name instead of changing weights inplace
    # Because we dont cancel requests that are past max_async, these requests could be using a LoRA that gets unloaded which will crash the inference server
    max_cpu_loras: Annotated[
        int,
        Field(
            description="The maximum number of LoRAs to use on CPU. Passed to vLLM as `--max-cpu-loras`",
        ),
    ] = 100

    max_lora_rank: Annotated[
        int | None,
        Field(
            description="The maximum LoRA rank to use. Passed to vLLM as `--max-lora-rank`",
        ),
    ] = None

    enable_prefix_caching: Annotated[
        bool | None,
        Field(
            description="Whether to enable prefix caching. Passed to vLLM as `--enable-prefix-caching`",
        ),
    ] = None

    gpu_memory_utilization: Annotated[
        float,
        Field(
            description="The GPU memory utilization to use. Passed to vLLM as `--gpu-memory-utilization`",
        ),
    ] = 0.9

    api_server_count: Annotated[
        int,
        Field(
            ge=0,
            description="The number of API servers to use. Passed to vLLM as `--api-server-count`. Set to 0 for headless mode.",
        ),
    ] = 1

    data_parallel_size_local: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of data parallel replicas to run on this node. Passed to vLLM as `--data-parallel-size-local`.",
        ),
    ] = None

    data_parallel_rpc_port: Annotated[
        int,
        Field(
            ge=1,
            le=65535,
            description="RPC port for data parallel communication. Passed to vLLM as `--data-parallel-rpc-port`.",
        ),
    ] = 13345

    seed: Annotated[
        int,
        Field(
            description="Seed the inference components. Passed to vLLM as `--seed`",
        ),
    ] = 0

    enable_expert_parallel: Annotated[
        bool,
        Field(
            description="Enable expert parallelism for MoE models. Passed to vLLM as `--enable-expert-parallel`.",
        ),
    ] = False

    all2all_backend: Annotated[
        All2AllBackend,
        Field(
            description="All-to-all backend for expert parallel communication. Passed to vLLM as `--all2all-backend`.",
        ),
    ] = "allgather_reducescatter"

    enable_eplb: Annotated[
        bool,
        Field(
            description="Enable expert parallel load balancer (EPLB). Passed to vLLM as `--enable-eplb`.",
        ),
    ] = False

    use_deep_gemm: Annotated[
        bool,
        Field(
            description="Force DeepGEMM FP8 kernels via VLLM_USE_DEEP_GEMM=1. Only works with per-tensor FP8 quantization (e.g. GLM-5-FP8).",
        ),
    ] = False

    enable_return_routed_experts: Annotated[
        bool,
        Field(
            description="Whether to enable return routed experts. Passed to vLLM as `--enable-return-routed-experts`",
        ),
    ] = False

    vllm_extra: Annotated[
        dict[str, Any],
        Field(
            description="Extra arguments to pass to vLLM. These are applied as attributes on the vLLM namespace after config translation.",
        ),
    ] = {}

    @model_validator(mode="after")
    def auto_setup_max_lora_rank(self):
        """Auto-setup max_lora_rank by rounding up to the nearest valid vLLM value.

        vLLM only accepts specific values for max_lora_rank: (1, 8, 16, 32, 64, 128, 256, 320, 512).
        This validator ensures that any configured rank is rounded up to the minimum valid value
        that can serve adapters of the requested rank.
        """
        if self.max_lora_rank is not None:
            original_rank = self.max_lora_rank
            for valid_rank in VALID_VLLM_LORA_RANKS:
                if valid_rank >= self.max_lora_rank:
                    self.max_lora_rank = valid_rank
                    break
            else:
                raise ValueError(f"max_lora_rank={original_rank} exceeds vLLM maximum of {VALID_VLLM_LORA_RANKS[-1]}")
        return self

    def to_vllm(self) -> Namespace:
        """Convert InferenceConfig to vLLM-compatible Namespace."""
        namespace = Namespace()
        to_vllm = {
            "server_config.host": "host",
            "server_config.port": "port",
            "model.name": "model",
            "model.dtype": "dtype",
            "model.max_model_len": "max_model_len",
            "model.enforce_eager": "enforce_eager",
            "model.trust_remote_code": "trust_remote_code",
            "model.tool_call_parser": "tool_call_parser",
            "model.reasoning_parser": "reasoning_parser",
            "model.rope_scaling": "rope_scaling",
            "parallel.tp": "tensor_parallel_size",
            "parallel.dp": "data_parallel_size",
            "data_parallel_size_local": "data_parallel_size_local",
            "data_parallel_rpc_port": "data_parallel_rpc_port",
            "enable_lora": "enable_lora",
            "enable_prefix_caching": "enable_prefix_caching",
            "max_loras": "max_loras",
            "max_cpu_loras": "max_cpu_loras",
            "max_lora_rank": "max_lora_rank",
            "gpu_memory_utilization": "gpu_memory_utilization",
            "api_server_count": "api_server_count",
            "enable_return_routed_experts": "enable_return_routed_experts",
            "enable_expert_parallel": "enable_expert_parallel",
            "all2all_backend": "all2all_backend",
            "enable_eplb": "enable_eplb",
            "seed": "seed",
        }

        for config_key, vllm_key in to_vllm.items():
            value = rgetattr(self, config_key.replace("-", "_"))
            rsetattr(namespace, vllm_key, value)

        # Set `logprobs_mode` to `processed_logprobs` by default
        rsetattr(namespace, "logprobs_mode", "processed_logprobs")

        # Remove reasoning_parser if not set (vLLM doesn't accept None)
        if namespace.reasoning_parser is None:
            delattr(namespace, "reasoning_parser")

        # Remove rope_scaling if not set (vLLM doesn't accept None)
        if hasattr(namespace, "rope_scaling"):
            if namespace.rope_scaling is None:
                delattr(namespace, "rope_scaling")

        return namespace

    @model_validator(mode="after")
    def auto_setup_api_server_count(self):
        """
        Ensures that we have at least as many API servers as data parallel
        size. Unless LoRA is enabled, in which case only one API server is
        supported (vLLM limitation).
        """
        if self.vllm_extra.get("headless", False):
            self.api_server_count = 0
            return self

        # User didn't set api_server_count, auto-setup
        min_api_server_count = self.data_parallel_size_local or self.parallel.dp
        if self.api_server_count < min_api_server_count:
            warnings.warn(
                f"api_server_count ({self.api_server_count}) is less than the minimum recommended ({min_api_server_count}). Setting to {min_api_server_count}."
            )
            self.api_server_count = min_api_server_count

        if self.enable_lora:
            warnings.warn("LoRA is enabled, which only supports one API server. Setting to 1.")
            self.api_server_count = 1

        return self

    @model_validator(mode="after")
    def disallow_eplb(self):
        if self.enable_eplb:
            raise ValueError("EPLB is currently not supported due to a bug in weight transfer.")
        return self


class DisaggregatedRuntimeInferenceConfig(BaseRuntimeInferenceConfig):
    num_replicas_per_pod: Annotated[
        int,
        Field(
            description="Pod is a single router instance. Setting this allows you to control the number of prefill/decode replicas per pod, i.e. allowing 2x EP16 prefill replicas to point to 1 EP32 decode replica, all under a single router instance."
        ),
    ] = 1

    @model_validator(mode="after")
    def disallow_headless(self):
        if self.vllm_extra.get("headless", False):
            raise ValueError("Headless mode is not supported for disaggregated deployments.")
        return self


class BaseInferenceDeploymentConfig(BaseModel):
    """Base deployment config for inference."""

    model_config = ConfigDict(extra="forbid")

    gpus_per_node: Annotated[int, Field(description="Number of GPUs per node.")] = 8


class SingleNodeInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    """Configures a single-node inference deployment."""

    type: Literal["single_node"] = "single_node"
    runtime_config: Annotated[BaseRuntimeInferenceConfig, Field(description="Runtime inference configuration.")] = (
        Field(default_factory=BaseRuntimeInferenceConfig)
    )


class MultiNodeInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    """Configures a multi-node inference deployment. Each node runs an independent vLLM replica."""

    type: Literal["multi_node"] = "multi_node"
    runtime_config: Annotated[BaseRuntimeInferenceConfig, Field(description="Runtime inference configuration.")] = (
        Field(default_factory=BaseRuntimeInferenceConfig)
    )

    num_nodes: Annotated[int, Field(ge=1, description="Number of inference nodes.")] = 2
    num_pods: Annotated[int, Field(ge=1, description="Number of router instances.")] = 1

    router_config: Annotated[RouterConfig, Field(description="Configuration for the router instance.")] = RouterConfig()


class DisaggregatedInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    """Configures a disaggregated prefill/decode inference deployment.

    Notation:
    - node = single node with GPUs (single host)
    - P = prefill
    - D = decode
    - replica = single vLLM parallelism unit, i.e. EP32, TP16PP2, etc.
    - pod = single router instance, consists of >= 1 P replica(s) and >= 1 D replica(s)

    Scaling invariants:
    - internal_parallel_size = num_gpus_per_node // (vllm.tp * vllm.pp * vllm.dp * vllm.cp)
    - per_pod_size = (num_decode_nodes_per_pod * decode_runtime_config.internal_parallel_size) + (num_prefill_nodes_per_pod * prefill_runtime_config.internal_parallel_size)
    - num_nodes = num_pods * per_pod_size
    """

    type: Literal["disaggregated"] = "disaggregated"

    num_nodes: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of inference nodes. If omitted, inferred from role configs and num_pods.",
        ),
    ] = None
    num_prefill_nodes_per_pod: Annotated[int, Field(ge=1, description="Number of prefill nodes per pod.")] = 1
    num_decode_nodes_per_pod: Annotated[int, Field(ge=1, description="Number of decode nodes per pod.")] = 1

    num_pods: Annotated[int, Field(ge=1, description="Number of router instances.")] = 1

    prefill_runtime_config: Annotated[
        DisaggregatedRuntimeInferenceConfig, Field(description="Configuration for the prefill nodes.")
    ] = Field(default_factory=lambda: DisaggregatedRuntimeInferenceConfig())

    decode_runtime_config: Annotated[
        DisaggregatedRuntimeInferenceConfig, Field(description="Configuration for the decode nodes.")
    ] = Field(default_factory=lambda: DisaggregatedRuntimeInferenceConfig())

    router_config: Annotated[RouterConfig, Field(description="Configuration for the router instances.")] = (
        RouterConfig()
    )

    @model_validator(mode="after")
    def auto_setup_local_dp(self):
        """Auto-setup data parallel size local for prefill and decode nodes."""
        if self.prefill_runtime_config.data_parallel_size_local is None:
            self.prefill_runtime_config.data_parallel_size_local = (
                self.gpus_per_node // self.prefill_runtime_config.parallel.tp
            )
        if self.decode_runtime_config.data_parallel_size_local is None:
            self.decode_runtime_config.data_parallel_size_local = (
                self.gpus_per_node // self.decode_runtime_config.parallel.tp
            )
        return self

    @model_validator(mode="after")
    def validate_tp_compatibility(self):
        for cfg in [self.prefill_runtime_config, self.decode_runtime_config]:
            if cfg.parallel.tp > self.gpus_per_node:
                raise ValueError(
                    f"parallel.tp ({cfg.parallel.tp}) must currently be less than or equal to gpus_per_node ({self.gpus_per_node})"
                )
            if self.gpus_per_node % cfg.parallel.tp != 0:
                raise ValueError(
                    f"gpus_per_node ({self.gpus_per_node}) must currently be divisible by parallel.tp ({cfg.parallel.tp})"
                )
        return self

    @model_validator(mode="after")
    def validate_local_dp_compatibility(self):
        for cfg in [self.prefill_runtime_config, self.decode_runtime_config]:
            if cfg.data_parallel_size_local > self.gpus_per_node:
                raise ValueError(
                    f"data_parallel_size_local ({cfg.data_parallel_size_local}) must currently be less than or equal to gpus_per_node ({self.gpus_per_node})"
                )
            if cfg.data_parallel_size_local * cfg.parallel.tp != self.gpus_per_node:
                raise ValueError(
                    f"data_parallel_size_local ({cfg.data_parallel_size_local}) * parallel.tp ({cfg.parallel.tp}) must currently equal gpus_per_node ({self.gpus_per_node})"
                )

        return self

    @model_validator(mode="after")
    def auto_setup_global_dp(self):
        for cfg, num_nodes in [
            (self.prefill_runtime_config, self.num_prefill_nodes_per_pod),
            (self.decode_runtime_config, self.num_decode_nodes_per_pod),
        ]:
            expected_dp = num_nodes * self.gpus_per_node // cfg.parallel.tp
            if "dp" not in cfg.parallel.model_fields_set:
                cfg.parallel.dp = expected_dp
        return self

    @model_validator(mode="after")
    def validate_global_dp_compatibility(self):
        for prefix, cfg, num_nodes in [
            ("Prefill", self.prefill_runtime_config, self.num_prefill_nodes_per_pod),
            ("Decode", self.decode_runtime_config, self.num_decode_nodes_per_pod),
        ]:
            if cfg.parallel.dp != num_nodes * self.gpus_per_node // cfg.parallel.tp:
                raise ValueError(
                    f"{prefix}: parallel.dp must be equal to num_nodes * gpus_per_node // parallel.tp => {num_nodes} * {self.gpus_per_node} // {cfg.parallel.tp}"
                )
        return self

    @model_validator(mode="after")
    def auto_setup_disaggregated(self):
        """Auto-configure disaggregated role configs and infer total node count."""
        inferred_num_nodes = self.num_pods * (self.num_prefill_nodes_per_pod + self.num_decode_nodes_per_pod)

        if self.num_nodes is None:
            self.num_nodes = inferred_num_nodes
        elif self.num_nodes != inferred_num_nodes:
            raise ValueError(
                f"deployment.num_nodes ({self.num_nodes}) does not match inferred value "
                f"({inferred_num_nodes}) from num_pods/replicas/parallel settings."
            )
        return self


InferenceDeploymentConfig: TypeAlias = Annotated[
    SingleNodeInferenceDeploymentConfig | MultiNodeInferenceDeploymentConfig | DisaggregatedInferenceDeploymentConfig,
    Field(discriminator="type"),
]


class InferenceConfig(BaseConfig):
    """Configures inference."""

    weight_broadcast: Annotated[WeightBroadcastConfig, Field(description="The weight broadcast config.")] = (
        WeightBroadcastConfig()
    )

    # Launcher-only fields
    deployment: Annotated[
        InferenceDeploymentConfig,
        Field(
            description="Deployment configuration for inference.",
        ),
    ] = SingleNodeInferenceDeploymentConfig()

    slurm: Annotated[
        SlurmConfig | None,
        Field(
            description="SLURM configuration. If set, the run will be submitted as a SLURM job instead of running locally.",
        ),
    ] = None

    output_dir: Annotated[Path, Field(description="Directory for SLURM logs and generated scripts.")] = Path("outputs")

    dry_run: Annotated[bool, Field(description="Only validate and dump resolved configs and exit early.")] = False

    @model_validator(mode="after")
    def validate_multi_node_or_disaggregated_requires_slurm(self):
        if self.deployment.type in ["multi_node", "disaggregated"] and self.slurm is None:
            raise ValueError("Must use SLURM for multi-node or disaggregated deployment.")
        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        if self.slurm is not None and self.slurm.template_path is None:
            import prime_rl

            templates_dir = Path(prime_rl.__file__).parent / "templates"
            self.slurm.template_path = templates_dir / "inference.sbatch.j2"
        return self
