from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from prometheus_client.parser import text_string_to_metric_families

from prime_rl.configs.orchestrator import InferenceObservabilityConfig
from prime_rl.configs.shared import ClientConfig


COUNTER_KEYS = {
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:request_success_total": "request_success_total",
    "vllm:nixl_num_failed_transfers_total": "nixl_failed_transfers_total",
    "vllm:nixl_num_failed_notifications_total": "nixl_failed_notifications_total",
    "vllm:nixl_num_kv_expired_reqs_total": "nixl_kv_expired_requests_total",
}

GAUGE_KEYS = {
    "vllm:num_requests_running": "running_requests",
    "vllm:num_requests_waiting": "waiting_requests",
    "vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
}

HISTOGRAM_SUM_KEYS = {
    "vllm:request_prefill_time_seconds_sum": "request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum": "request_decode_time_seconds_sum",
    "vllm:request_queue_time_seconds_sum": "request_queue_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum": "time_to_first_token_seconds_sum",
    "vllm:inter_token_latency_seconds_sum": "inter_token_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_sum": "e2e_request_latency_seconds_sum",
    "vllm:nixl_xfer_time_seconds_sum": "nixl_xfer_time_seconds_sum",
    "vllm:nixl_bytes_transferred_sum": "nixl_bytes_transferred_sum",
}

HISTOGRAM_COUNT_KEYS = {
    "vllm:request_prefill_time_seconds_count": "request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_count": "request_decode_time_seconds_count",
    "vllm:request_queue_time_seconds_count": "request_queue_time_seconds_count",
    "vllm:time_to_first_token_seconds_count": "time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_count": "inter_token_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_count": "e2e_request_latency_seconds_count",
    "vllm:nixl_xfer_time_seconds_count": "nixl_xfer_time_seconds_count",
    "vllm:nixl_bytes_transferred_count": "nixl_bytes_transferred_count",
}


@dataclass
class EngineRollup:
    running_requests: float = 0.0
    waiting_requests: float = 0.0
    kv_cache_usage_perc: float = 0.0
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    request_success_total: float = 0.0
    nixl_failed_transfers_total: float = 0.0
    nixl_failed_notifications_total: float = 0.0
    nixl_kv_expired_requests_total: float = 0.0
    request_prefill_time_seconds_sum: float = 0.0
    request_prefill_time_seconds_count: float = 0.0
    request_decode_time_seconds_sum: float = 0.0
    request_decode_time_seconds_count: float = 0.0
    request_queue_time_seconds_sum: float = 0.0
    request_queue_time_seconds_count: float = 0.0
    time_to_first_token_seconds_sum: float = 0.0
    time_to_first_token_seconds_count: float = 0.0
    inter_token_latency_seconds_sum: float = 0.0
    inter_token_latency_seconds_count: float = 0.0
    e2e_request_latency_seconds_sum: float = 0.0
    e2e_request_latency_seconds_count: float = 0.0
    nixl_xfer_time_seconds_sum: float = 0.0
    nixl_xfer_time_seconds_count: float = 0.0
    nixl_bytes_transferred_sum: float = 0.0
    nixl_bytes_transferred_count: float = 0.0


@dataclass
class NodeRollup:
    engines: dict[str, EngineRollup] = field(default_factory=dict)

    def summed(self, attribute: str) -> float:
        return sum(getattr(engine, attribute) for engine in self.engines.values())

    def kv_values(self) -> list[float]:
        return [engine.kv_cache_usage_perc for engine in self.engines.values()]


@dataclass
class RateSnapshot:
    prompt_tokens_per_second: float = 0.0
    generation_tokens_per_second: float = 0.0
    requests_finished_per_second: float = 0.0
    avg_prefill_time_seconds: float | None = None
    avg_decode_time_seconds: float | None = None
    avg_queue_time_seconds: float | None = None
    avg_ttft_seconds: float | None = None
    avg_tpot_seconds: float | None = None
    avg_e2e_latency_seconds: float | None = None
    nixl_avg_transfer_time_seconds: float | None = None
    nixl_transfers_per_second: float = 0.0
    nixl_bytes_per_second: float = 0.0
    nixl_avg_bytes_per_transfer: float | None = None


@dataclass
class ObservabilityNode:
    id: str
    hostname: str
    role: str
    pod_index: int
    role_replica_index: int
    global_index: int
    metrics_url: str
    health_url: str
    metric_path: str


@dataclass
class ObservabilityPod:
    id: str
    pod_index: int
    router_url: str
    router_health_url: str
    prefill_nodes: list[ObservabilityNode]
    decode_nodes: list[ObservabilityNode]


@dataclass
class NodeState:
    current_timestamp: float | None = None
    current_rollup: NodeRollup | None = None
    previous_timestamp: float | None = None
    previous_rollup: NodeRollup | None = None
    scrape_error: str | None = None


@dataclass
class NodeSnapshot:
    node: ObservabilityNode
    scrape_ok: bool
    scrape_age_seconds: float
    running_requests: float = 0.0
    waiting_requests: float = 0.0
    kv_cache_usage_mean: float = 0.0
    kv_cache_usage_max: float = 0.0
    prompt_tokens_per_second: float = 0.0
    generation_tokens_per_second: float = 0.0
    requests_finished_per_second: float = 0.0
    avg_prefill_time_seconds: float | None = None
    avg_decode_time_seconds: float | None = None
    avg_queue_time_seconds: float | None = None
    avg_ttft_seconds: float | None = None
    avg_tpot_seconds: float | None = None
    avg_e2e_latency_seconds: float | None = None
    nixl_avg_transfer_time_seconds: float | None = None
    nixl_transfers_per_second: float = 0.0
    nixl_bytes_per_second: float = 0.0
    nixl_avg_bytes_per_transfer: float | None = None
    nixl_failed_transfers_total: float = 0.0
    nixl_failed_notifications_total: float = 0.0
    nixl_kv_expired_requests_total: float = 0.0


def parse_prometheus_text(payload: str) -> NodeRollup:
    engines: dict[str, EngineRollup] = defaultdict(EngineRollup)
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            engine_id = sample.labels.get("engine", "aggregate")
            engine = engines[engine_id]
            if sample.name in GAUGE_KEYS:
                setattr(engine, GAUGE_KEYS[sample.name], float(sample.value))
            elif sample.name in COUNTER_KEYS:
                if sample.name == "vllm:request_success_total":
                    engine.request_success_total += float(sample.value)
                else:
                    setattr(engine, COUNTER_KEYS[sample.name], float(sample.value))
            elif sample.name in HISTOGRAM_SUM_KEYS:
                setattr(engine, HISTOGRAM_SUM_KEYS[sample.name], float(sample.value))
            elif sample.name in HISTOGRAM_COUNT_KEYS:
                setattr(engine, HISTOGRAM_COUNT_KEYS[sample.name], float(sample.value))
    return NodeRollup(engines=dict(engines))


def _counter_rate(current: float, previous: float, dt: float) -> float:
    delta = current - previous
    if dt <= 0 or delta < 0:
        return 0.0
    return delta / dt


def _histogram_average(current_sum: float, current_count: float, previous_sum: float, previous_count: float) -> float | None:
    sum_delta = current_sum - previous_sum
    count_delta = current_count - previous_count
    if sum_delta < 0 or count_delta <= 0:
        return None
    return sum_delta / count_delta


def compute_rates(current: NodeRollup, previous: NodeRollup | None, dt: float) -> RateSnapshot:
    if previous is None:
        return RateSnapshot()

    return RateSnapshot(
        prompt_tokens_per_second=_counter_rate(current.summed("prompt_tokens_total"), previous.summed("prompt_tokens_total"), dt),
        generation_tokens_per_second=_counter_rate(
            current.summed("generation_tokens_total"), previous.summed("generation_tokens_total"), dt
        ),
        requests_finished_per_second=_counter_rate(
            current.summed("request_success_total"), previous.summed("request_success_total"), dt
        ),
        avg_prefill_time_seconds=_histogram_average(
            current.summed("request_prefill_time_seconds_sum"),
            current.summed("request_prefill_time_seconds_count"),
            previous.summed("request_prefill_time_seconds_sum"),
            previous.summed("request_prefill_time_seconds_count"),
        ),
        avg_decode_time_seconds=_histogram_average(
            current.summed("request_decode_time_seconds_sum"),
            current.summed("request_decode_time_seconds_count"),
            previous.summed("request_decode_time_seconds_sum"),
            previous.summed("request_decode_time_seconds_count"),
        ),
        avg_queue_time_seconds=_histogram_average(
            current.summed("request_queue_time_seconds_sum"),
            current.summed("request_queue_time_seconds_count"),
            previous.summed("request_queue_time_seconds_sum"),
            previous.summed("request_queue_time_seconds_count"),
        ),
        avg_ttft_seconds=_histogram_average(
            current.summed("time_to_first_token_seconds_sum"),
            current.summed("time_to_first_token_seconds_count"),
            previous.summed("time_to_first_token_seconds_sum"),
            previous.summed("time_to_first_token_seconds_count"),
        ),
        avg_tpot_seconds=_histogram_average(
            current.summed("inter_token_latency_seconds_sum"),
            current.summed("inter_token_latency_seconds_count"),
            previous.summed("inter_token_latency_seconds_sum"),
            previous.summed("inter_token_latency_seconds_count"),
        ),
        avg_e2e_latency_seconds=_histogram_average(
            current.summed("e2e_request_latency_seconds_sum"),
            current.summed("e2e_request_latency_seconds_count"),
            previous.summed("e2e_request_latency_seconds_sum"),
            previous.summed("e2e_request_latency_seconds_count"),
        ),
        nixl_avg_transfer_time_seconds=_histogram_average(
            current.summed("nixl_xfer_time_seconds_sum"),
            current.summed("nixl_xfer_time_seconds_count"),
            previous.summed("nixl_xfer_time_seconds_sum"),
            previous.summed("nixl_xfer_time_seconds_count"),
        ),
        nixl_transfers_per_second=_counter_rate(
            current.summed("nixl_xfer_time_seconds_count"), previous.summed("nixl_xfer_time_seconds_count"), dt
        ),
        nixl_bytes_per_second=_counter_rate(
            current.summed("nixl_bytes_transferred_sum"), previous.summed("nixl_bytes_transferred_sum"), dt
        ),
        nixl_avg_bytes_per_transfer=_histogram_average(
            current.summed("nixl_bytes_transferred_sum"),
            current.summed("nixl_bytes_transferred_count"),
            previous.summed("nixl_bytes_transferred_sum"),
            previous.summed("nixl_bytes_transferred_count"),
        ),
    )


def _hostname_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname is None:
        raise ValueError(f"could not parse hostname from url {url}")
    return parsed.hostname


class InferenceMetricsCollector:
    def __init__(self, config: InferenceObservabilityConfig, client_config: ClientConfig):
        if not client_config.admin_base_url:
            raise ValueError("orchestrator.client.admin_base_url is required for inference observability")
        self.config = config
        self.client_config = client_config
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=1.0))
        self._node_state: dict[str, NodeState] = {}
        self._last_scrape_at: dict[str, float] = {}
        self._router_status: dict[str, bool] = {}
        self.pods = self._build_topology()

    async def stop(self) -> None:
        await self.http_client.aclose()

    def _build_topology(self) -> list[ObservabilityPod]:
        router_urls = [url.rstrip("/") for url in self.client_config.base_url]
        backend_urls = [url.rstrip("/") for url in self.client_config.admin_base_url or []]
        if len(router_urls) != self.config.num_infer_replicas:
            raise ValueError(
                f"expected {self.config.num_infer_replicas} router urls, found {len(router_urls)} in orchestrator.client.base_url"
            )
        expected_backend_count = self.config.num_infer_replicas * (
            self.config.num_prefill_nodes_per_pod + self.config.num_decode_nodes_per_pod
        )
        if len(backend_urls) != expected_backend_count:
            raise ValueError(
                f"expected {expected_backend_count} backend urls, found {len(backend_urls)} in orchestrator.client.admin_base_url"
            )
        if self.config.num_prefill_nodes_per_pod % self.config.num_prefill_replicas_per_pod != 0:
            raise ValueError("prefill replicas per pod must evenly divide prefill nodes per pod")
        if self.config.num_decode_nodes_per_pod % self.config.num_decode_replicas_per_pod != 0:
            raise ValueError("decode replicas per pod must evenly divide decode nodes per pod")

        prefill_nodes_per_replica = self.config.num_prefill_nodes_per_pod // self.config.num_prefill_replicas_per_pod
        decode_nodes_per_replica = self.config.num_decode_nodes_per_pod // self.config.num_decode_replicas_per_pod
        nodes_per_pod = self.config.num_prefill_nodes_per_pod + self.config.num_decode_nodes_per_pod

        pods: list[ObservabilityPod] = []
        for pod_index in range(self.config.num_infer_replicas):
            pod_base = pod_index * nodes_per_pod
            pod_backend_urls = backend_urls[pod_base : pod_base + nodes_per_pod]
            prefill_nodes: list[ObservabilityNode] = []
            decode_nodes: list[ObservabilityNode] = []

            for role_index in range(self.config.num_prefill_nodes_per_pod):
                url = pod_backend_urls[role_index]
                role_replica_index = role_index // prefill_nodes_per_replica
                role_replica_rank = role_index % prefill_nodes_per_replica
                prefill_nodes.append(
                    ObservabilityNode(
                        id=f"pod_{pod_index:02d}/prefill_replica_{role_replica_index:02d}/node_{role_replica_rank:02d}",
                        hostname=_hostname_from_url(url),
                        role="prefill",
                        pod_index=pod_index,
                        role_replica_index=role_replica_index,
                        global_index=pod_base + role_index,
                        metrics_url=f"{url.removesuffix('/v1')}/metrics",
                        health_url=f"{url.removesuffix('/v1')}/health",
                        metric_path=f"pod_{pod_index:02d}/prefill_replica_{role_replica_index:02d}/node_{role_replica_rank:02d}",
                    )
                )

            for role_index in range(self.config.num_decode_nodes_per_pod):
                url = pod_backend_urls[self.config.num_prefill_nodes_per_pod + role_index]
                role_replica_index = role_index // decode_nodes_per_replica
                role_replica_rank = role_index % decode_nodes_per_replica
                decode_nodes.append(
                    ObservabilityNode(
                        id=f"pod_{pod_index:02d}/decode_replica_{role_replica_index:02d}/node_{role_replica_rank:02d}",
                        hostname=_hostname_from_url(url),
                        role="decode",
                        pod_index=pod_index,
                        role_replica_index=role_replica_index,
                        global_index=pod_base + self.config.num_prefill_nodes_per_pod + role_index,
                        metrics_url=f"{url.removesuffix('/v1')}/metrics",
                        health_url=f"{url.removesuffix('/v1')}/health",
                        metric_path=f"pod_{pod_index:02d}/decode_replica_{role_replica_index:02d}/node_{role_replica_rank:02d}",
                    )
                )

            router_url = router_urls[pod_index].removesuffix("/v1")
            pods.append(
                ObservabilityPod(
                    id=f"pod_{pod_index:02d}",
                    pod_index=pod_index,
                    router_url=router_url,
                    router_health_url=f"{router_url}/health",
                    prefill_nodes=prefill_nodes,
                    decode_nodes=decode_nodes,
                )
            )
        return pods

    async def collect(self) -> dict[str, float]:
        router_tasks = [self._poll_router_health(pod) for pod in self.pods]
        node_tasks = [self._poll_node_metrics(node) for pod in self.pods for node in (*pod.prefill_nodes, *pod.decode_nodes)]
        await asyncio.gather(*router_tasks, return_exceptions=True)
        await asyncio.gather(*node_tasks, return_exceptions=True)

        now = time.time()
        node_snapshots = [self._build_node_snapshot(node, now) for pod in self.pods for node in (*pod.prefill_nodes, *pod.decode_nodes)]
        metrics: dict[str, float] = {}
        metrics.update(self._emit_topology_metrics())
        metrics.update(self._emit_router_metrics())
        metrics.update(self._emit_group_metrics("inference/cluster", node_snapshots))

        prefill_nodes = [node for node in node_snapshots if node.node.role == "prefill"]
        decode_nodes = [node for node in node_snapshots if node.node.role == "decode"]
        metrics.update(self._emit_group_metrics("inference/role/prefill", prefill_nodes))
        metrics.update(self._emit_group_metrics("inference/role/decode", decode_nodes))

        for pod in self.pods:
            pod_nodes = [node for node in node_snapshots if node.node.pod_index == pod.pod_index]
            metrics.update(self._emit_group_metrics(f"inference/pod_{pod.pod_index:02d}", pod_nodes))
            for role in ("prefill", "decode"):
                role_nodes = [node for node in pod_nodes if node.node.role == role]
                metrics.update(self._emit_group_metrics(f"inference/pod_{pod.pod_index:02d}/{role}", role_nodes))
                by_replica: dict[int, list[NodeSnapshot]] = defaultdict(list)
                for node in role_nodes:
                    by_replica[node.node.role_replica_index].append(node)
                for replica_index, replica_nodes in sorted(by_replica.items()):
                    metrics.update(
                        self._emit_group_metrics(
                            f"inference/pod_{pod.pod_index:02d}/{role}_replica_{replica_index:02d}",
                            replica_nodes,
                        )
                    )

        for snapshot in node_snapshots:
            metrics.update(self._emit_node_metrics(snapshot))

        return metrics

    async def _poll_router_health(self, pod: ObservabilityPod) -> None:
        try:
            response = await self.http_client.get(pod.router_health_url)
            response.raise_for_status()
            self._router_status[pod.id] = True
        except Exception:
            self._router_status[pod.id] = False

    async def _poll_node_metrics(self, node: ObservabilityNode) -> None:
        state = self._node_state.setdefault(node.id, NodeState())
        try:
            response = await self.http_client.get(node.metrics_url)
            response.raise_for_status()
            rollup = parse_prometheus_text(response.text)
            state.previous_timestamp = state.current_timestamp
            state.previous_rollup = state.current_rollup
            state.current_timestamp = time.time()
            state.current_rollup = rollup
            state.scrape_error = None
            self._last_scrape_at[node.id] = state.current_timestamp
        except Exception as exc:
            state.scrape_error = str(exc)

    def _build_node_snapshot(self, node: ObservabilityNode, now: float) -> NodeSnapshot:
        state = self._node_state.get(node.id)
        if state is None or state.current_rollup is None or state.current_timestamp is None:
            return NodeSnapshot(node=node, scrape_ok=False, scrape_age_seconds=9999.0)

        dt = 0.0 if state.previous_timestamp is None else state.current_timestamp - state.previous_timestamp
        rates = compute_rates(state.current_rollup, state.previous_rollup, dt)
        kv_values = state.current_rollup.kv_values()
        return NodeSnapshot(
            node=node,
            scrape_ok=state.scrape_error is None,
            scrape_age_seconds=max(now - self._last_scrape_at.get(node.id, state.current_timestamp), 0.0),
            running_requests=state.current_rollup.summed("running_requests"),
            waiting_requests=state.current_rollup.summed("waiting_requests"),
            kv_cache_usage_mean=sum(kv_values) / len(kv_values) if kv_values else 0.0,
            kv_cache_usage_max=max(kv_values, default=0.0),
            prompt_tokens_per_second=rates.prompt_tokens_per_second,
            generation_tokens_per_second=rates.generation_tokens_per_second,
            requests_finished_per_second=rates.requests_finished_per_second,
            avg_prefill_time_seconds=rates.avg_prefill_time_seconds,
            avg_decode_time_seconds=rates.avg_decode_time_seconds,
            avg_queue_time_seconds=rates.avg_queue_time_seconds,
            avg_ttft_seconds=rates.avg_ttft_seconds,
            avg_tpot_seconds=rates.avg_tpot_seconds,
            avg_e2e_latency_seconds=rates.avg_e2e_latency_seconds,
            nixl_avg_transfer_time_seconds=rates.nixl_avg_transfer_time_seconds,
            nixl_transfers_per_second=rates.nixl_transfers_per_second,
            nixl_bytes_per_second=rates.nixl_bytes_per_second,
            nixl_avg_bytes_per_transfer=rates.nixl_avg_bytes_per_transfer,
            nixl_failed_transfers_total=state.current_rollup.summed("nixl_failed_transfers_total"),
            nixl_failed_notifications_total=state.current_rollup.summed("nixl_failed_notifications_total"),
            nixl_kv_expired_requests_total=state.current_rollup.summed("nixl_kv_expired_requests_total"),
        )

    def _emit_topology_metrics(self) -> dict[str, float]:
        return {
            "inference/topology/pods": float(self.config.num_infer_replicas),
            "inference/topology/prefill_nodes_per_pod": float(self.config.num_prefill_nodes_per_pod),
            "inference/topology/decode_nodes_per_pod": float(self.config.num_decode_nodes_per_pod),
            "inference/topology/prefill_replicas_per_pod": float(self.config.num_prefill_replicas_per_pod),
            "inference/topology/decode_replicas_per_pod": float(self.config.num_decode_replicas_per_pod),
        }

    def _emit_router_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        healthy = 0
        for pod in self.pods:
            is_healthy = float(self._router_status.get(pod.id, False))
            metrics[f"inference/{pod.id}/router_healthy"] = is_healthy
            healthy += int(is_healthy)
        metrics["inference/health/routers_healthy"] = float(healthy)
        metrics["inference/health/routers_total"] = float(len(self.pods))
        metrics["inference/health/router_healthy_ratio"] = healthy / max(len(self.pods), 1)
        return metrics

    def _emit_node_metrics(self, snapshot: NodeSnapshot) -> dict[str, float]:
        prefix = f"inference/{snapshot.node.metric_path}"
        metrics = {
            f"{prefix}/scrape_ok": float(snapshot.scrape_ok),
            f"{prefix}/scrape_age_seconds": snapshot.scrape_age_seconds,
            f"{prefix}/requests/running": snapshot.running_requests,
            f"{prefix}/requests/waiting": snapshot.waiting_requests,
            f"{prefix}/requests/completed_per_second": snapshot.requests_finished_per_second,
            f"{prefix}/kv_cache/usage_mean": snapshot.kv_cache_usage_mean,
            f"{prefix}/kv_cache/usage_max": snapshot.kv_cache_usage_max,
            f"{prefix}/throughput/prompt_tokens_per_second": snapshot.prompt_tokens_per_second,
            f"{prefix}/throughput/generation_tokens_per_second": snapshot.generation_tokens_per_second,
            f"{prefix}/nixl/transfers_per_second": snapshot.nixl_transfers_per_second,
            f"{prefix}/nixl/bytes_per_second": snapshot.nixl_bytes_per_second,
            f"{prefix}/nixl/failed_transfers_total": snapshot.nixl_failed_transfers_total,
            f"{prefix}/nixl/failed_notifications_total": snapshot.nixl_failed_notifications_total,
            f"{prefix}/nixl/expired_requests_total": snapshot.nixl_kv_expired_requests_total,
        }
        optional_values = {
            f"{prefix}/latency/prefill_seconds": snapshot.avg_prefill_time_seconds,
            f"{prefix}/latency/decode_seconds": snapshot.avg_decode_time_seconds,
            f"{prefix}/latency/queue_seconds": snapshot.avg_queue_time_seconds,
            f"{prefix}/latency/ttft_seconds": snapshot.avg_ttft_seconds,
            f"{prefix}/latency/tpot_seconds": snapshot.avg_tpot_seconds,
            f"{prefix}/latency/e2e_seconds": snapshot.avg_e2e_latency_seconds,
            f"{prefix}/nixl/transfer_time_seconds": snapshot.nixl_avg_transfer_time_seconds,
            f"{prefix}/nixl/avg_bytes_per_transfer": snapshot.nixl_avg_bytes_per_transfer,
        }
        for key, value in optional_values.items():
            if value is not None:
                metrics[key] = value
        return metrics

    def _emit_group_metrics(self, prefix: str, snapshots: list[NodeSnapshot]) -> dict[str, float]:
        metrics: dict[str, float] = {
            f"{prefix}/nodes": float(len(snapshots)),
            f"{prefix}/health/healthy_nodes": float(sum(1 for snapshot in snapshots if snapshot.scrape_ok)),
            f"{prefix}/health/stale_nodes": float(sum(1 for snapshot in snapshots if snapshot.scrape_age_seconds > 6.0)),
            f"{prefix}/requests/running/all": sum(snapshot.running_requests for snapshot in snapshots),
            f"{prefix}/requests/waiting/all": sum(snapshot.waiting_requests for snapshot in snapshots),
            f"{prefix}/requests/completed_per_second/all": sum(snapshot.requests_finished_per_second for snapshot in snapshots),
            f"{prefix}/throughput/prompt_tokens_per_second/all": sum(snapshot.prompt_tokens_per_second for snapshot in snapshots),
            f"{prefix}/throughput/generation_tokens_per_second/all": sum(
                snapshot.generation_tokens_per_second for snapshot in snapshots
            ),
            f"{prefix}/nixl/transfers_per_second/all": sum(snapshot.nixl_transfers_per_second for snapshot in snapshots),
            f"{prefix}/nixl/bytes_per_second/all": sum(snapshot.nixl_bytes_per_second for snapshot in snapshots),
            f"{prefix}/nixl/failed_transfers_total/all": sum(snapshot.nixl_failed_transfers_total for snapshot in snapshots),
        }
        metrics.update(self._emit_distribution_metrics(f"{prefix}/kv_cache/usage_mean", [snapshot.kv_cache_usage_mean for snapshot in snapshots]))
        metrics.update(self._emit_distribution_metrics(f"{prefix}/kv_cache/usage_max", [snapshot.kv_cache_usage_max for snapshot in snapshots]))
        metrics.update(self._emit_distribution_metrics(f"{prefix}/scrape_age_seconds", [snapshot.scrape_age_seconds for snapshot in snapshots]))
        metrics.update(
            self._emit_distribution_metrics(
                f"{prefix}/latency/prefill_seconds",
                [snapshot.avg_prefill_time_seconds for snapshot in snapshots if snapshot.avg_prefill_time_seconds is not None],
            )
        )
        metrics.update(
            self._emit_distribution_metrics(
                f"{prefix}/latency/decode_seconds",
                [snapshot.avg_decode_time_seconds for snapshot in snapshots if snapshot.avg_decode_time_seconds is not None],
            )
        )
        metrics.update(
            self._emit_distribution_metrics(
                f"{prefix}/latency/queue_seconds",
                [snapshot.avg_queue_time_seconds for snapshot in snapshots if snapshot.avg_queue_time_seconds is not None],
            )
        )
        metrics.update(
            self._emit_distribution_metrics(
                f"{prefix}/latency/ttft_seconds",
                [snapshot.avg_ttft_seconds for snapshot in snapshots if snapshot.avg_ttft_seconds is not None],
            )
        )
        metrics.update(
            self._emit_distribution_metrics(
                f"{prefix}/latency/tpot_seconds",
                [snapshot.avg_tpot_seconds for snapshot in snapshots if snapshot.avg_tpot_seconds is not None],
            )
        )
        metrics.update(
            self._emit_distribution_metrics(
                f"{prefix}/latency/e2e_seconds",
                [snapshot.avg_e2e_latency_seconds for snapshot in snapshots if snapshot.avg_e2e_latency_seconds is not None],
            )
        )
        metrics.update(
            self._emit_distribution_metrics(
                f"{prefix}/nixl/transfer_time_seconds",
                [
                    snapshot.nixl_avg_transfer_time_seconds
                    for snapshot in snapshots
                    if snapshot.nixl_avg_transfer_time_seconds is not None
                ],
            )
        )
        metrics.update(
            self._emit_distribution_metrics(
                f"{prefix}/nixl/avg_bytes_per_transfer",
                [
                    snapshot.nixl_avg_bytes_per_transfer
                    for snapshot in snapshots
                    if snapshot.nixl_avg_bytes_per_transfer is not None
                ],
            )
        )
        return metrics

    @staticmethod
    def _emit_distribution_metrics(prefix: str, values: list[float | None]) -> dict[str, float]:
        filtered = [float(value) for value in values if value is not None]
        if not filtered:
            return {}
        return {
            f"{prefix}/mean": sum(filtered) / len(filtered),
            f"{prefix}/max": max(filtered),
            f"{prefix}/min": min(filtered),
        }
