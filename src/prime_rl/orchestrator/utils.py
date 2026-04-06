import asyncio
import time
from itertools import cycle
from pathlib import Path
from typing import Any

import pandas as pd
import verifiers as vf
from openai.types.chat.chat_completion import ChatCompletion
from rich.console import Console
from rich.table import Table
from verifiers.utils.client_utils import setup_openai_client

from prime_rl.configs.orchestrator import EvalSamplingConfig, OrchestratorConfig, SamplingConfig
from prime_rl.transport import TrainingSample
from prime_rl.utils.utils import (
    format_time,
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)


def get_train_sampling_args(sampling_config: SamplingConfig, is_vllm: bool = True) -> dict:
    # Convert SamplingConfig to vLLM OAI sampling args
    # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_2
    sampling_args = dict(sampling_config)
    sampling_args["top_p"] = 1.0
    sampling_args["logprobs"] = True
    extra_body = dict(sampling_config.extra_body)

    min_tokens = sampling_args.pop("min_tokens")
    repetition_penalty = sampling_args.pop("repetition_penalty")

    if min_tokens > 0:
        extra_body["min_tokens"] = min_tokens
    if repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = repetition_penalty

    if is_vllm:
        extra_body["top_k"] = -1
        extra_body["min_p"] = 0.0
        extra_body["return_token_ids"] = True

    if extra_body:
        sampling_args["extra_body"] = extra_body

    return sampling_args


def get_eval_sampling_args(sampling_config: EvalSamplingConfig) -> dict[str, Any]:
    """Get sampling args for evaluation."""
    sampling_args: dict[str, Any] = {}

    if sampling_config.temperature is not None:
        sampling_args["temperature"] = sampling_config.temperature
    if sampling_config.max_completion_tokens is not None:
        sampling_args["max_completion_tokens"] = sampling_config.max_completion_tokens
    if sampling_config.top_p is not None:
        sampling_args["top_p"] = sampling_config.top_p
    if sampling_config.reasoning_effort is not None:
        sampling_args["reasoning_effort"] = sampling_config.reasoning_effort

    extra_body: dict[str, Any] = sampling_config.extra_body.copy()

    if sampling_config.top_k is not None:
        extra_body["top_k"] = sampling_config.top_k
    if sampling_config.min_p is not None:
        extra_body["min_p"] = sampling_config.min_p
    if sampling_config.min_tokens is not None:
        extra_body["min_tokens"] = sampling_config.min_tokens
    if sampling_config.repetition_penalty is not None:
        extra_body["repetition_penalty"] = sampling_config.repetition_penalty

    sampling_args["extra_body"] = extra_body

    return sampling_args


def print_benchmark(history: dict[str, list[Any]]) -> None:
    """
    Print benchmark results as rich table. Shows formatted step time values.
    First N rows show the per-step values, and the last row shows the mean,
    std, min, and max values.
    """
    history.pop("step")
    assert all(len(v) for v in history.values()), "All metrics must have logged the same number of steps"

    # Turn metric history into pd.DataFrame
    df = pd.DataFrame(dict(history.items()))
    columns = {
        "time/step": "Step Time",
    }
    df = df.rename(columns=columns)
    df = df[list(columns.values())]
    df = df.iloc[1:]  # Exclude first row

    # Setup console
    console = Console()
    table = Table(title="Benchmark")

    # Add columns
    table.add_column("Step", justify="right")
    for col in df.columns:
        table.add_column(col, justify="center", style="magenta")

    # Add formatted rows
    formatted_df = pd.DataFrame(columns=df.columns)
    formatted_df["Step Time"] = df["Step Time"].apply(format_time)
    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step)] + [str(x) for x in row]))

    # Separator
    num_table_columns = 1 + len(df.columns)
    table.add_row(*([""] * num_table_columns))

    # Add row for formatted, aggregated statistics
    mean_df = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_mean_df = pd.DataFrame(columns=mean_df.columns)
    formatted_mean_df["Step Time"] = mean_df["Step Time"].apply(format_time)
    mean_row = ["Overall"] + formatted_mean_df.T.apply(
        lambda row: f"{row['mean']} ± {row['std']} [{row['min']}, {row['max']}]", axis=1
    ).tolist()
    table.add_row(*mean_row)

    # Display table
    console.print(table)


async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
) -> list[list[float]]:
    """Compute teacher model logprobs for a batch of training samples via prefill."""

    async def _compute_single(client_config: vf.ClientConfig, sample: TrainingSample) -> list[float]:
        client = setup_openai_client(client_config)

        response = await client.post(
            "/chat/completions/tokens",
            body={
                "model": model_name,
                "messages": [{"role": "user", "content": ""}],
                "tokens": sample.prompt_ids + sample.completion_ids,
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "skip_special_tokens": False,
                "prompt_logprobs": True,
            },
            cast_to=ChatCompletion,
        )
        return [
            0.0 if lp is None else float(next(iter(lp.values()))["logprob"])
            for lp in getattr(response, "prompt_logprobs", [])
        ]

    return await asyncio.gather(*[_compute_single(client, sample) for client, sample in zip(cycle(clients), samples)])


def get_weight_dir(output_dir: Path, step: int, check_exists: bool = True, wait_timeout: int | None = None) -> Path:
    """Get the weight directory for a given checkpoint step.

    Args:
        output_dir: The output directory for the run.
        step: The checkpoint step.
        check_exists: If True, raises FileNotFoundError if no weight directory exists.
            If False, returns the broadcast directory path without checking existence
            (useful for NCCL mode where weights are broadcasted, not stored on disk).
        wait_timeout: Maximum time in seconds to wait for a stable directory to appear.
            If None, no waiting is performed.
    """
    ckpt_weight_dir = get_step_path(get_ckpt_dir(output_dir), step) / "weight"
    broadcast_weight_dir = get_step_path(get_broadcast_dir(output_dir), step)

    def find_stable_dir() -> Path | None:
        # For checkpoint weights, check STABLE file in parent directory (checkpoints/step_{step}/STABLE)
        ckpt_step_dir = get_step_path(get_ckpt_dir(output_dir), step)
        if (ckpt_step_dir / "STABLE").exists() and ckpt_weight_dir.exists():
            return ckpt_weight_dir

        # For broadcast weights, check STABLE file in the broadcast directory itself
        if (broadcast_weight_dir / "STABLE").exists() and broadcast_weight_dir.exists():
            return broadcast_weight_dir

        return None

    # Check immediately, then wait if needed
    result = find_stable_dir()
    if result is None and wait_timeout:
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            time.sleep(1)
            result = find_stable_dir()
            if result:
                break

    if result:
        return result
    if not check_exists:
        return broadcast_weight_dir

    raise FileNotFoundError(f"No weight directory found for checkpoint step {step}")


def setup_external_rollout_model(config: OrchestratorConfig, logger) -> tuple[Any, str, bool]:
    """Resolve rollout client/model and whether policy updates should be enabled."""
    rollout_client_config = config.client
    rollout_model_name = config.model.name
    enable_policy_updates = True

    if config.teacher_rollout_model is not None:
        rollout_client_config = config.teacher_rollout_model.client
        rollout_model_name = config.teacher_rollout_model.model.name
        enable_policy_updates = False
        logger.info(
            f"Using external teacher rollout model (base_url={', '.join(rollout_client_config.base_url)}, "
            f"model={rollout_model_name})"
        )

    return rollout_client_config, rollout_model_name, enable_policy_updates
