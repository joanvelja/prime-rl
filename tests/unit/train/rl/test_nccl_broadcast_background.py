from threading import Thread
from unittest.mock import MagicMock

import pytest

from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast


def make_broadcast() -> NCCLWeightBroadcast:
    broadcast = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    broadcast.logger = MagicMock()
    broadcast._bg_thread = None
    broadcast._bg_error = None
    return broadcast


def test_join_background_reraises_send_failure():
    broadcast = make_broadcast()
    failure = RuntimeError("nccl send failed")

    broadcast._wait_for_nccl_ready = MagicMock()
    broadcast.sender = MagicMock()
    broadcast.sender.send_all_layers.side_effect = failure

    broadcast._bg_thread = Thread(
        target=broadcast._background_send,
        args=([], [], 7, 0.0),
        daemon=True,
    )
    broadcast._bg_thread.start()

    with pytest.raises(RuntimeError, match="Background NCCL broadcast failed") as exc_info:
        broadcast._join_background()

    assert exc_info.value.__cause__ is failure
    assert broadcast._bg_thread is None
    assert broadcast._bg_error is None
    broadcast.logger.exception.assert_called_once_with("Pipelined NCCL broadcast failed")
