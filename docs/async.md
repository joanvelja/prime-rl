# Asynchronous Training

PRIME-RL implements asynchronous off-policy training, instead of the traditional synchronous on-policy training. This means that we allow inference to generate rollouts from a stale policy up to $k$ (in the code we call this `max_async_level`) steps ahead of the trainer. With `k=1` and trainer and inference step timings being equal, this allows overlap between trainer and inference without either side being fully idle.

The current config default is `max_async_level = 1`. NCCL weight broadcast also validates that `max_async_level == 1`; higher async levels require a non-NCCL broadcast path such as filesystem-based adapter/weight loading.

![Two-Step Off-Policy Training](assets/two-step-off-policy.png)

## Loss Objective

We adopt a loss objective capable of handling the natural distribution shift caused by the off-policy nature of the training. The current default trainer loss is implemented in `prime_rl.trainer.rl.loss.dppo_kl_loss_fn`: it uses rollout-time logprobs for an importance ratio, masks tokens whose trainer-vs-rollout probability difference crosses the DPPO thresholds, and optionally adds a squared log-ratio penalty controlled by `kl_tau`.

Older descriptions of this as plain ratio clipping are imprecise for the current implementation: the mask is based on probability difference, while unmasked policy-gradient terms still use the unclipped importance ratio.

At each step, we sample $N$ prompts from our dataset. For
each prompt $x$, we sample a group of rollouts $\{y_i\}^G_{i=1}$
and use a verifier to assign scores $s_i$ to each $y_i$.
Then, the optimization objective is given by

$$
\mathcal{J}_{\text{AIPO}}(\theta)
= \frac{1}{\sum_{j=1}^N \sum_{i=1}^G |y_i^{(j)}|}
\sum_{j=1}^N 
\sum_{i=1}^G 
\sum_{t=1}^{|y_i^{(j)}|}
\min\left(
\frac{\pi(y^{(j)}_{i,t}\mid x_j, y^{(j)}_{i,<t})}{\mu(y^{(j)}_{i,t}\mid x_j, y^{(j)}_{i,<t})},
\delta
\right)\hat{A}^{(j)}_{i,t}
$$

where $\mu$ refers to the policy that generated the rollout, $\pi$ refers to the current policy, $\hat{A}_{i,t}$ is the token-level advantage, and $\delta$ is the importance sampling clipping ratio.


## Step Semantics

PRIME-RL uses a global training step $n=1,2,3,\dots$ that is used to tag artifacts:

- **Trainer**: Produces policy $\pi_n$ with weights $\theta_n$ from rollouts $(x_n, y_n)$
- **Inference**: Produces rollouts $(x_n, y_n)$ from policy $\pi_{max(0, n-k)}$

Here, $k$ is the `max_async_level` parameter, which currently defaults to 1. Note that we use 0-indexed steps to cleanly indicate that at each step, the intended inference/trainer async gap is at most $k$ steps. Rollouts can still be older than this by the time they are consumed if they were already in flight; `max_off_policy_steps` controls the acceptance/drop threshold for those stale rollout groups.
