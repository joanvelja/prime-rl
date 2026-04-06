import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import verifiers as vf

from prime_rl.configs.orchestrator import EnvArgsSchedulerConfig
from prime_rl.utils.utils import import_object


@dataclass(frozen=True)
class EnvArgsEvent:
    step: int
    env: str
    event: str
    version: str
    args: dict[str, Any]


@dataclass
class EnvArgsSchedulerInputs:
    step: int
    ckpt_step: int
    env_name: str
    active_args: dict[str, Any]
    desired_args: dict[str, Any]
    reload_in_progress: bool
    global_metrics: dict[str, float]
    env_metrics: dict[str, float]


EnvArgsSchedulerFn = Callable[..., dict[str, Any] | None]


def setup_env_args_scheduler(config: EnvArgsSchedulerConfig) -> EnvArgsSchedulerFn:
    scheduler_fn = import_object(config.import_path)
    kwargs = config.kwargs

    def run_scheduler(inputs: EnvArgsSchedulerInputs) -> dict[str, Any] | None:
        return scheduler_fn(inputs, **kwargs)

    return run_scheduler


def merge_arg_diff(current_args: dict[str, Any], diff: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(current_args)
    if diff is not None:
        merged.update(diff)
    return merged


def env_uses_group_scoring(env: vf.Environment) -> bool:
    rubric = env.rubric
    return any(rubric._is_group_func(func) for func in rubric._get_reward_funcs())


@dataclass
class ManagedTrainEnvVersion:
    version: int
    args: dict[str, Any]
    env: vf.Environment
    address: str
    client: Any
    process: Any | None
    uses_deferred_group_scoring: bool


@dataclass
class EnvArgsState:
    env_name: str
    active_version: int
    desired_version: int
    active_args: dict[str, Any]
    desired_args: dict[str, Any]
    versions: dict[int, ManagedTrainEnvVersion] = field(default_factory=dict)
    reload_task: asyncio.Task | None = None
    pending_version: int | None = None
    reload_started_at: float | None = None
    scheduler_fn: EnvArgsSchedulerFn | None = None
    scheduler_interval: int = 1
    version_refcounts: dict[int, int] = field(default_factory=dict)

    @property
    def reload_in_progress(self) -> bool:
        return self.reload_task is not None and not self.reload_task.done()

    def should_run_scheduler(self, step: int) -> bool:
        return self.scheduler_fn is not None and step % self.scheduler_interval == 0

    def request_change(self, step: int, diff: dict[str, Any] | None) -> list[EnvArgsEvent]:
        if diff is None:
            return []

        next_args = merge_arg_diff(self.desired_args, diff)
        if next_args == self.desired_args:
            return []

        prev_desired_version = self.desired_version
        self.desired_version += 1
        self.desired_args = next_args
        self.pending_version = self.desired_version

        events = [
            EnvArgsEvent(
                step=step,
                env=self.env_name,
                event="request",
                version=f"{prev_desired_version}->{self.desired_version}",
                args=diff,
            )
        ]
        if self.reload_in_progress:
            events.append(
                EnvArgsEvent(
                    step=step,
                    env=self.env_name,
                    event="overlap",
                    version=f"active={self.active_version} desired={self.desired_version}",
                    args=diff,
                )
            )
        return events

    def mark_reload_complete(self, step: int, version: int, args: dict[str, Any]) -> EnvArgsEvent | None:
        if version <= self.active_version:
            return None

        previous_active_version = self.active_version
        self.active_version = version
        self.active_args = dict(args)
        self.pending_version = None if version == self.desired_version else self.desired_version
        self.reload_started_at = None
        return EnvArgsEvent(
            step=step,
            env=self.env_name,
            event="activate",
            version=f"{previous_active_version}->{version}",
            args=dict(args),
        )


class TrainEnvRegistry:
    def __init__(self, states: dict[str, EnvArgsState], env_names: list[str]):
        self.states = states
        self.env_names = env_names

    def get_env_for_task(self, task: str, version: int | None = None) -> vf.Environment:
        state = self.states[task]
        version = state.active_version if version is None else version
        return state.versions[version].env

    def get_active_version(self, task: str) -> int:
        return self.states[task].active_version

    def should_defer_group_scoring(self, task: str, version: int) -> bool:
        return self.states[task].versions[version].uses_deferred_group_scoring

    def add_version(self, task: str, version: ManagedTrainEnvVersion) -> None:
        state = self.states[task]
        state.versions[version.version] = version
        state.version_refcounts.setdefault(version.version, 0)

    def retire_version(self, task: str, version: int) -> None:
        state = self.states[task]
        state.version_refcounts.pop(version, None)
        handle = state.versions.pop(version, None)
        if handle is None:
            return
        if handle.process is not None:
            handle.process.terminate()
            handle.process.join(timeout=25)
            if handle.process.is_alive():
                handle.process.kill()
                handle.process.join(timeout=5)

    def pin_version(self, task: str, version: int) -> None:
        state = self.states[task]
        state.version_refcounts[version] = state.version_refcounts.get(version, 0) + 1

    def release_version(self, task: str, version: int) -> None:
        state = self.states[task]
        if version not in state.version_refcounts:
            return
        state.version_refcounts[version] = max(0, state.version_refcounts[version] - 1)
        if state.version_refcounts[version] != 0:
            return
        if version == state.active_version:
            return
        self.retire_version(task, version)

    def build_active_env_group(self) -> vf.EnvGroup:
        envs = [self.get_env_for_task(name) for name in self.env_names]
        return vf.EnvGroup(
            envs=envs,
            env_names=self.env_names,
            map_kwargs=dict(writer_batch_size=1),
        )


def build_env_args_metrics(states: dict[str, EnvArgsState], events: list[EnvArgsEvent]) -> dict[str, int]:
    return {
        "env_args/changed": sum(event.event == "request" for event in events),
        "env_args/reloaded": sum(event.event == "activate" for event in events),
        "env_args/pending": sum(state.active_version != state.desired_version for state in states.values()),
    }


def build_env_args_table_rows(events: list[EnvArgsEvent]) -> list[dict[str, Any]]:
    return [
        {
            "step": event.step,
            "env": event.env,
            "event": event.event,
            "version": event.version,
            "args": json.dumps(event.args, sort_keys=True),
        }
        for event in events
    ]


@dataclass
class ReloadResult:
    task: str
    version: int
    previous_version: int
    args: dict[str, Any]
    version_handle: ManagedTrainEnvVersion
    duration_s: float


def build_reload_warning_message(env_name: str, desired_version: int, diff: dict[str, Any]) -> str:
    return (
        f"Env args reload already in progress for {env_name}. "
        f"Updated desired version to {desired_version} with diff={diff}."
    )


def now() -> float:
    return time.perf_counter()
