import pytest
import torch

from prime_rl.configs.trainer import (
    CustomLossConfig,
    DefaultLossConfig,
    SFTLossConfig,
)
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, compute_entropy, compute_loss, setup_loss_fn

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_setup_loss_fn_with_custom_config():
    """Test setup_loss_fn with CustomLossConfig importing a custom loss."""
    loss_config = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss._dummy_custom_loss",
        kwargs={"multiplier": 2.0},
    )
    loss_fn = setup_loss_fn(loss_config)

    inputs = LossInputs(
        trainer_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        inference_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        teacher_logprobs=None,
        advantages=torch.randn(50).cuda(),
        loss_mask=torch.ones(50, dtype=torch.bool).cuda(),
    )

    result = loss_fn(inputs)
    assert isinstance(result, LossOutputs)
    assert result.loss.shape == ()
    assert "custom_metric" in result.metrics


def test_sft_loss_matches_masked_nll():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(SFTLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=2,
    )

    # loss = -sum(masked logprobs) / loss_scale = -(-0.1 - 0.2) / 2 = 0.15
    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics


def test_is_reinforce_loss_uses_importance_ratio():
    trainer_logprobs = [torch.log(torch.tensor([2.0, 4.0], dtype=torch.float32)).cuda()]
    inference_logprobs = [torch.zeros(2, dtype=torch.float32).cuda()]
    advantages = [torch.tensor([1.0, -0.5], dtype=torch.float32).cuda()]
    loss_mask = [torch.ones(2, dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(
        CustomLossConfig(import_path="prime_rl.trainer.rl.loss.is_reinforce_loss_fn", kwargs={"kl_tau": 0.0})
    )
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=1,
    )

    expected = -(
        advantages[0] * torch.exp(trainer_logprobs[0] - inference_logprobs[0]).detach() * trainer_logprobs[0]
    ).sum()
    assert torch.isclose(loss, expected, atol=1e-6)
    assert "importance_ratio" in metrics


def test_reinforce_loss_matches_reward_weighted_nll_without_importance_ratio():
    trainer_logprobs = [torch.log(torch.tensor([2.0, 4.0], dtype=torch.float32)).cuda()]
    inference_logprobs = [torch.zeros(2, dtype=torch.float32).cuda()]
    rewards = [torch.tensor([1.0, 0.0], dtype=torch.float32).cuda()]
    loss_mask = [torch.ones(2, dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(
        CustomLossConfig(import_path="prime_rl.trainer.rl.loss.reinforce_loss_fn", kwargs={"kl_tau": 0.0})
    )
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=rewards,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=1,
    )

    expected = -(rewards[0] * trainer_logprobs[0]).sum()
    assert torch.isclose(loss, expected, atol=1e-6)
    assert "importance_ratio" not in metrics


def test_custom_dppo_loss_can_apply_kl_at_sequence_level():
    trainer_logprobs = [torch.tensor([1.0, 1.0], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(2, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(2, dtype=torch.float32).cuda()]
    loss_mask = [torch.ones(2, dtype=torch.bool).cuda()]

    token_loss_fn = setup_loss_fn(
        CustomLossConfig(
            import_path="prime_rl.trainer.rl.loss.dppo_kl_loss_fn",
            kwargs={"kl_tau": 1.0, "kl_level": "token"},
        )
    )
    sequence_loss_fn = setup_loss_fn(
        CustomLossConfig(
            import_path="prime_rl.trainer.rl.loss.dppo_kl_loss_fn",
            kwargs={"kl_tau": 1.0, "kl_level": "sequence"},
        )
    )
    token_loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fn=token_loss_fn,
        loss_scale=1,
    )
    sequence_loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fn=sequence_loss_fn,
        loss_scale=1,
    )

    assert torch.isclose(token_loss, torch.tensor(2.0, device=token_loss.device), atol=1e-6)
    assert torch.isclose(sequence_loss, torch.tensor(4.0, device=sequence_loss.device), atol=1e-6)
    assert "sequence_kl" in metrics


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )
