from __future__ import annotations

import os
from typing import Any

import torch


def process_rss_bytes() -> int:
    try:
        with open("/proc/self/status") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def _cache_keys(cache: Any) -> list[int]:
    if cache is None:
        return []
    if hasattr(cache, "cache"):
        cache = cache.cache
    if isinstance(cache, dict):
        return sorted(int(key) for key in cache.keys())
    return []


def lora_worker_diagnostics(worker: Any) -> dict[str, Any]:
    model_runner = getattr(worker, "model_runner", None)
    lora_manager = getattr(model_runner, "lora_manager", None)
    adapter_manager = getattr(lora_manager, "_adapter_manager", None)

    registered_ids: list[int] = []
    active_ids: list[int] = []
    capacity = None
    lora_slots = None
    if adapter_manager is not None:
        registered_ids = _cache_keys(getattr(adapter_manager, "_registered_adapters", None))
        active_ids = _cache_keys(getattr(adapter_manager, "_active_adapters", None))
        capacity = getattr(adapter_manager, "capacity", None)
        lora_slots = getattr(adapter_manager, "lora_slots", None)

    device = getattr(worker, "device", None)
    gpu_allocated = 0
    gpu_reserved = 0
    if isinstance(device, torch.device) and device.type == "cuda":
        gpu_allocated = int(torch.cuda.memory_allocated(device))
        gpu_reserved = int(torch.cuda.memory_reserved(device))

    return {
        "pid": os.getpid(),
        "rss_bytes": process_rss_bytes(),
        "registered_lora_ids": registered_ids,
        "active_lora_ids": active_ids,
        "capacity": capacity,
        "lora_slots": lora_slots,
        "gpu_allocated_bytes": gpu_allocated,
        "gpu_reserved_bytes": gpu_reserved,
    }
