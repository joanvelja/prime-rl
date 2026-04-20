import os

import pytest
from transformers import AutoTokenizer

from prime_rl.configs.sft import FakeDataConfig
from prime_rl.trainer.sft.data import CatDataset, StatefulIterableDataset, cat_collate, setup_dataloader, setup_dataset
from prime_rl.trainer.world import reset_world

pytestmark = [pytest.mark.gpu]


def test_fake_dataset_single_rank_state():
    # Setup stateful dataloader
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    # Initial state
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 0, "epoch": 0}}

    # Iterate over samples
    micro_batch = next(dataiter)
    print(micro_batch)
    assert micro_batch["input_ids"].unique().item() == 0
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 1, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 1
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 2, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 3, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 3
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 4, "epoch": 0}}


@pytest.mark.parametrize("rank", [0, 1], ids=["rank0", "rank1"])
def test_fake_dataset_multi_rank_state(rank: int):
    # Setup world
    reset_world()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(2)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(2)

    # Setup stateful dataloader
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    # Initial state
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 0, "epoch": 0}}

    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 0 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 1 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 3 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 4 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 5 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 6 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 7 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 8 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 9 + rank, "epoch": 0}}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 10 + rank
    assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": 11 + rank, "epoch": 0}}


def test_fake_dataset_single_rank_resume():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", input_ids="increasing", batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    # First 2 samples
    for step in range(2):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": step + 1, "epoch": 0}}

    # Reload dataloader
    state_dict = dataloader.state_dict()
    dataloader = setup_dataloader(dataset, config)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)

    # Second two samples
    for step in range(2, 4):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert dataloader.state_dict()["dataset_state"] == {"dataset": {"step": step + 1, "epoch": 0}}


def test_fake_dataset_single_rank_state_with_packing():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="variable", input_ids="increasing", batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, config)
    dataiter = iter(dataloader)

    prev_step = 0
    for _ in range(8):
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        dataset_state = dataloader.state_dict()["dataset_state"]["dataset"]
        assert dataset_state["epoch"] == 0
        assert dataset_state["step"] >= prev_step
        prev_step = dataset_state["step"]


def test_cat_dataset_preserves_overflow_and_exposes_cu_seqlens():
    class ToyDataset(StatefulIterableDataset):
        def __init__(self, samples):
            self.samples = samples

        def state_dict(self) -> dict:
            return {}

        def load_state_dict(self, state_dict: dict):
            pass

        def __iter__(self):
            for sample in self.samples:
                yield sample

    samples = [
        {
            "input_ids": [10, 11, 12, 13, 14],
            "target_ids": [11, 12, 13, 14, 15],
            "position_ids": [0, 1, 2, 3, 4],
            "loss_mask": [True] * 5,
            "sequence_lengths": [5],
        },
        {
            "input_ids": [20, 21, 22, 23, 24],
            "target_ids": [21, 22, 23, 24, 25],
            "position_ids": [0, 1, 2, 3, 4],
            "loss_mask": [True] * 5,
            "sequence_lengths": [5],
        },
        {
            "input_ids": [30, 31, 32],
            "target_ids": [31, 32, 33],
            "position_ids": [0, 1, 2],
            "loss_mask": [True] * 3,
            "sequence_lengths": [3],
        },
        {
            "input_ids": [40, 41, 42, 43, 44],
            "target_ids": [41, 42, 43, 44, 45],
            "position_ids": [0, 1, 2, 3, 4],
            "loss_mask": [True] * 5,
            "sequence_lengths": [5],
        },
    ]

    packed = CatDataset(ToyDataset(samples), seq_len=8)
    dataiter = iter(packed)

    first = next(dataiter)
    assert first["input_ids"] == [10, 11, 12, 13, 14, 20, 21, 22]
    assert first["position_ids"] == [0, 1, 2, 3, 4, 0, 1, 2]
    assert first["sequence_lengths"] == [5, 3]

    second = next(dataiter)
    assert second["input_ids"] == [23, 24, 30, 31, 32, 40, 41, 42]
    assert second["position_ids"] == [3, 4, 0, 1, 2, 0, 1, 2]
    assert second["sequence_lengths"] == [2, 3, 3]

    collated = cat_collate([first])
    assert collated["cu_seqlens"].tolist() == [0, 5, 8]
    assert collated["max_seqlen"] == 5
