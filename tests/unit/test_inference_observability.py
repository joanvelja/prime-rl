from prime_rl.configs.orchestrator import InferenceObservabilityConfig
from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.inference_observability import InferenceMetricsCollector


def test_inference_metrics_collector_builds_disaggregated_replica_topology():
    config = InferenceObservabilityConfig(
        num_infer_replicas=2,
        num_prefill_nodes_per_pod=4,
        num_decode_nodes_per_pod=4,
        num_prefill_replicas_per_pod=2,
        num_decode_replicas_per_pod=2,
    )
    client_config = ClientConfig(
        base_url=[
            "http://router-0:8000/v1",
            "http://router-1:8000/v1",
        ],
        admin_base_url=[
            "http://p0-0:8100/v1",
            "http://p0-1:8100/v1",
            "http://p0-2:8100/v1",
            "http://p0-3:8100/v1",
            "http://d0-0:8200/v1",
            "http://d0-1:8200/v1",
            "http://d0-2:8200/v1",
            "http://d0-3:8200/v1",
            "http://p1-0:8100/v1",
            "http://p1-1:8100/v1",
            "http://p1-2:8100/v1",
            "http://p1-3:8100/v1",
            "http://d1-0:8200/v1",
            "http://d1-1:8200/v1",
            "http://d1-2:8200/v1",
            "http://d1-3:8200/v1",
        ],
    )

    collector = InferenceMetricsCollector(config, client_config)

    assert len(collector.pods) == 2
    assert collector.pods[0].router_url == "http://router-0:8000"
    assert [node.hostname for node in collector.pods[0].prefill_nodes] == ["p0-0", "p0-1", "p0-2", "p0-3"]
    assert [node.role_replica_index for node in collector.pods[0].prefill_nodes] == [0, 0, 1, 1]
    assert [node.role_replica_index for node in collector.pods[0].decode_nodes] == [0, 0, 1, 1]
