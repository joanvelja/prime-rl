from __future__ import annotations

import numpy as np
import pandas as pd
import verifiers as vf

from prime_rl.orchestrator.vf_utils import get_completion_len
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize


def log_eval_metrics(
    env_name: str,
    rollouts: list[vf.RolloutOutput],
    total_rollouts: int,
    rollouts_per_example: int,
    eval_time: float,
    ckpt_step: int,
    step: int,
) -> None:
    """Compute and log eval metrics to the monitor."""
    monitor = get_monitor()
    failed_count = total_rollouts - len(rollouts)

    if failed_count:
        get_logger().warning(
            f"{failed_count}/{total_rollouts} ({failed_count / total_rollouts * 100:.1f}%) rollouts failed"
        )

    if not rollouts:
        get_logger().warning(f"All rollouts failed for {env_name}, skipping logging metrics")
        monitor.log(
            {
                f"eval/{env_name}/failed_rollouts": failed_count / total_rollouts,
                "progress/ckpt_step": ckpt_step,
                "step": step,
            },
            step=step,
        )
        return

    rows = [
        {
            "example_id": o["example_id"],
            "reward": o["reward"],
            "completion_len": get_completion_len(o),
            "is_truncated": o["is_truncated"],
            "has_error": o.get("error") is not None,
            "no_response": not o.get("completion"),
        }
        for o in rollouts
    ]
    results_df = pd.DataFrame(rows)

    unique_rewards = results_df.reward.dropna().unique()
    could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
    if could_be_binary:
        pass_at_k = (
            results_df.groupby("example_id")
            .apply(lambda x: compute_pass_at_k(x.reward.dropna()), include_groups=False)
            .apply(pd.Series)
        )
    else:
        pass_at_k = None
        get_logger().warning("Skipping computing pass@k rates because the task rewards appear to be non-binary")

    message = f"Evaluated {env_name} in {eval_time:.2f}s (Avg@{rollouts_per_example}={results_df.reward.mean():.4f}"
    if could_be_binary:
        assert pass_at_k is not None
        for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
            message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.4f}"

    message += (
        f", No-response: {results_df.no_response.mean() * 100:.1f}%"
        f", Completion Length: {results_df.completion_len.mean():.2f} (±{results_df.completion_len.std():.2f}, ∈[{results_df.completion_len.min():.2f}, {results_df.completion_len.max():.2f}])"
        f", Truncated: {results_df.is_truncated.mean() * 100:.1f}%)"
    )
    get_logger().success(message)

    eval_metrics = {
        f"avg@{rollouts_per_example}": float(results_df.reward.mean()),
        "no_response/mean": float(results_df.no_response.mean()),
        "no_response/count": int(results_df.no_response.sum()),
        "completion_len/mean": results_df.completion_len.mean().item(),
        "completion_len/max": results_df.completion_len.max().item(),
        "completion_len/min": results_df.completion_len.min().item(),
        "is_truncated/mean": results_df.is_truncated.mean().item(),
        "failed_rollouts": failed_count / total_rollouts,
        "time": eval_time,
    }
    if could_be_binary:
        assert pass_at_k is not None
        eval_metrics.update(pd.Series(pass_at_k.mean()).to_dict())
    eval_metrics = {f"eval/{env_name}/{key}": v for key, v in eval_metrics.items()}
    eval_metrics.update({"progress/ckpt_step": ckpt_step, "step": step})
    monitor.log(eval_metrics, step=step)
    monitor.log_eval_samples(rollouts, env_name=env_name, step=step)


def compute_eval_ckpt_step(
    ckpt_step: int,
    prev_ckpt_step: int,
    last_eval_step: int,
    interval: int,
    eval_base_model: bool = True,
) -> int | None:
    """Determine which checkpoint step (if any) should trigger an eval.

    Handles the case where ckpt_step jumps over interval boundaries by finding
    the highest interval-aligned step in (prev_ckpt_step, ckpt_step].

    Returns the interval step to eval at, or None if no eval should run.
    """
    if ckpt_step <= prev_ckpt_step:
        return None
    highest_interval_step = (ckpt_step // interval) * interval
    if highest_interval_step > prev_ckpt_step and highest_interval_step > last_eval_step:
        if highest_interval_step == 0:
            if ckpt_step == 0 and eval_base_model:
                return 0
        else:
            return highest_interval_step
    return None


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    Computes 1 - C(n-c, k) / C(n, k) in a numerically stable way.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k(rewards: list[float]) -> dict[str, float]:
    n = len(rewards)
    c = sum(r == 1.0 for r in rewards)
    ks = [2**i for i in range(n.bit_length())]
    return {f"pass@{k}": _pass_at_k(n, c, k) for k in ks}
