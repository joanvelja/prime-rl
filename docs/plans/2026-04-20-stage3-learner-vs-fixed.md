# Stage 3 — Learner-vs-Fixed Training (Debate)

**Date:** 2026-04-20
**Prereq branches:** prime-rl `feat/debate-orchestrator` (stacked); verifiers `main @ 42a965e3`
**Target milestone:** first learner-vs-fixed smoke run against `gpqa-debate` + gpt-4.1-mini as fixed opponent/judge

> ⚠ **GPU pre-flight (LoRA-self variant only)** — if you run the LoRA-self-vs-base topology (single vLLM server, learner behind a hot-swapped adapter, opponent on the base model name), **run `scripts/preflight_lora_smoke.py` before the first training step**. Three 5-minute probes catch known upstream vLLM failure modes ([#18372](https://github.com/vllm-project/vllm/issues/18372) silent 3rd+-swap drop, [#33791](https://github.com/vllm-project/vllm/issues/33791) in-place CPU fallback, [#10898](https://github.com/vllm-project/vllm/issues/10898) ~53% base-request tax) that are not covered by any test on our pin. See `skills/preflight-lora-smoke/SKILL.md` for interpretation and the two-instance escape hatch. The external-API-opponent topology (OpenAI / etc.) is not affected by any of this and needs no pre-flight.

## Goal

Train a Qwen3-4B policy by debating against a fixed (frozen) opponent. Each episode:
- one seat is the learner policy (being trained)
- other seat is a fixed opponent (external LLM, e.g. gpt-4.1-mini)
- judge is a fixed external LLM
- reward = `zero_sum_reward(learner_seat, winner)` — +1 if the learner's argument won, −1 if lost, 0 on tie

Learner seat is randomized per episode (Bernoulli 0.5 default) so the learner experiences both sides of every debate over training.

**Open question (not blocking):** whether the current protocol is true free-choice (each debater picks their own answer) or can be flipped to assigned-answer (each debater is given a specific answer to defend). Today only free-choice is wired; see [joanvelja/verifiers#3](https://github.com/joanvelja/verifiers/issues/3) for the gap.

## Architecture (converged)

```
╔══════════════════════════════════════════════════════════════════════════╗
║                  LAYERED SEPARATION (load-bearing)                       ║
║                                                                          ║
║    verifiers-side   → ROUTING      (which client/model runs which seat?) ║
║    dataset-side     → ASSIGNMENT   (which seat is the learner today?)    ║
║    prime-rl-side    → FILTERING    (whose tokens train?)                 ║
║                                                                          ║
║    Single thread linking them:  state["info"]["learner_seat"]            ║
║    Single universal key:        example_id                               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Why it's right

1. **No new machinery.** Every concept already exists:
   - `State.info` is an `INPUT_FIELDS` that forwards input → state → output
   - `agent_overrides` is already a `DebateEnv.__init__` kwarg (we add a state-aware resolver sibling)
   - `MemberRollout` fan-out is already the training-unit path
   - `RAEKey = (task, example_id, member_id)` already naturally partitions learner's baseline from an opponent it won't see again

2. **example_id is the universal key.** Dataset row identity → learner seat assignment → RAE baseline bucket → transcript id → wandb sample filter. Pin example_id, pin everything.

3. **Opt-in / opt-out / composable.**
   - RLVR: resolver=None, info=∅, filter=no-op
   - Self-play: resolver=None, info=∅, filter=no-op
   - Learner-vs-fixed: resolver set, info.learner_seat set, filter keeps-learner-only
   - Future — league play: resolver reads `info.opponent_ckpt_step`, picks from checkpoint clients. Same API.
   - Future — curriculum: dataset builder sets `info.opponent_tier`. Same API.

4. **Loud failure modes.** Every layer's invariant checks at construction or first-rollout time (see §Validation).

### Alternatives considered + rejected

| Alt | Idea | Why rejected |
|-----|------|--------------|
| A | Two envs (`gpqa-debate-learner_a` / `-learner_b`), no resolver | Doubles config surface; dataset splits; breaks league play |
| B | Hardcode learner=debater_a forever, schedule permutes which side argues truth | Hides seat assignment inside schedule; biased training |
| C | Trainer reads both rewards and picks which to train on | That IS self-play |
| D | Dataset row carries full `agent_overrides` dict per episode | Couples dataset to client config; breaks league play |
| E | Verifiers bridge filters MemberRollouts by a training-side flag | Pollutes verifiers' pure projection contract |

The resolver architecture handles Stage 3 (one fixed opponent), league play (checkpoint rotation), and curriculum (dataset-ordered opponent tiers) with the same ~150 LOC.

## Full info package: TOML → gradient step

### STAGE 0 — TOML declaration

```toml
# configs/gpqa_debate/rl_learner_vs_fixed.toml (new, sketched)

output_dir = "outputs/debate-learner-vs-fixed"
max_steps = 200
seq_len = 8192

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"   # ← the LEARNER model

[orchestrator]
batch_size = 128
rollouts_per_example = 4
oversampling_factor = 1.5

[orchestrator.train.sampling]
temperature = 1.0
max_completion_tokens = 1024

[[orchestrator.train.env]]
id = "gpqa-debate"
name = "gpqa-debate-lvf"
args = {
  # pack (free-choice)
  prompts_ref = "selfplay",
  subset = "gpqa_diamond",
  truth_member = "debater_a",       # inert in free-choice; passed because rubric requires it

  # routing (triggers the resolver path)
  opponent_model       = "gpt-4.1-mini",
  opponent_base_url    = "https://api.openai.com/v1",
  opponent_api_key_var = "OPENAI_API_KEY",
  judge_model          = "gpt-4.1-mini",
  judge_base_url       = "https://api.openai.com/v1",
  judge_api_key_var    = "OPENAI_API_KEY",

  # seat assignment (NEW knobs)
  learner_seat_mode = "bernoulli",   # "round_robin" | "hash" | "fixed_a" | "fixed_b"
  learner_seat_seed = 0,
}

[orchestrator.advantage]
type = "ema_per_member"
momentum = 0.9

[orchestrator.multi_agent]
drop_judge = true
filter_by_learner_seat = true       # NEW — opt-in

[orchestrator.buffer]
easy_threshold = 1.0
hard_threshold = -1.0

[ckpt]
interval = 25

[orchestrator.eval]
interval = 25
[[orchestrator.eval.env]]
id = "gpqa-debate"
name = "gpqa-debate-eval"
num_examples = 32
rollouts_per_example = 2
args = {
  prompts_ref = "selfplay", subset = "gpqa_diamond",
  opponent_model = "gpt-4.1-mini", judge_model = "gpt-4.1-mini",
  learner_seat_mode = "bernoulli", learner_seat_seed = 0,
}

[trainer]
[trainer.optim]
lr = 5e-6

[inference]
gpu_memory_utilization = 0.85
[inference.model]
max_model_len = 8192
```

### STAGE 1 — Config instantiation (prime-rl)

New fields on `MultiAgentConfig`:

```python
class MultiAgentConfig(BaseModel):
    drop_judge: bool = True
    filter_by_learner_seat: bool = False   # NEW, opt-in
```

No defensive `learner_uses_top_level_model` flag — if the caller wires the resolver inconsistently, first-rollout log will show it.

### STAGE 2 — `gpqa_debate.load_environment` (verifiers)

```python
def load_environment(
    ...,
    opponent_model:       str | None = None,
    opponent_base_url:    str | None = None,
    opponent_api_key_var: str = "OPENAI_API_KEY",
    judge_model:          str | None = None,
    judge_base_url:       str | None = None,
    judge_api_key_var:    str = "OPENAI_API_KEY",
    learner_seat_mode:    str = "round_robin",
    learner_seat_seed:    int = 0,
    ...,
):
    # Build resolver only if external clients are declared. Otherwise
    # (self-play): resolver=None, existing agent_overrides flow.
    if opponent_model is not None or judge_model is not None:
        resolver = _build_resolver(
            opp_client=OpenAIChatCompletionsClient(
                AsyncOpenAI(base_url=opponent_base_url,
                            api_key=os.getenv(opponent_api_key_var))),
            opp_model=opponent_model,
            judge_client=OpenAIChatCompletionsClient(
                AsyncOpenAI(base_url=judge_base_url,
                            api_key=os.getenv(judge_api_key_var))),
            judge_model=judge_model,
        )
    else:
        resolver = None

    rng = random.Random(learner_seat_seed)
    dataset = _build_dataset(
        subset, n, seed,
        learner_seat_mode=learner_seat_mode,
        learner_seat_rng=rng,
    )
    # _build_dataset calls _assign_learner_seat per row, stashes result
    # in row["info"]["learner_seat"].

    return _debate_load_env(
        ..., agent_overrides_resolver=resolver, eval_dataset=dataset,
    )
```

```python
def _assign_learner_seat(
    example_idx: int, example_id: str, mode: str, rng: random.Random,
) -> str:
    """Return "debater_a" or "debater_b" per the declared mode.

    bernoulli   — coin flip from rng (stateful; needs seed for reproducibility)
    round_robin — example_idx % 2 mapping (deterministic, seed-free)
    hash        — hash(example_id) % 2 (deterministic per example_id,
                  stable under dataset reordering — best reproducibility)
    fixed_a     — always debater_a (ablation: only truth-side when paired
                  with assigned-answer; in free-choice: just always seat a)
    fixed_b     — always debater_b (symmetric ablation)
    """
    if mode == "bernoulli":
        return "debater_a" if rng.random() < 0.5 else "debater_b"
    if mode == "round_robin":
        return ("debater_a", "debater_b")[example_idx % 2]
    if mode == "hash":
        return ("debater_a", "debater_b")[hash(example_id) % 2]
    if mode == "fixed_a":
        return "debater_a"
    if mode == "fixed_b":
        return "debater_b"
    raise ValueError(f"Unknown learner_seat_mode: {mode!r}")
```

```python
def _build_resolver(
    opp_client, opp_model, judge_client, judge_model,
) -> Callable[[State], dict[str, tuple[Client | None, str | None]]]:
    def resolver(state: State) -> dict[str, tuple[Client | None, str | None]]:
        info = state["info"]
        learner_seat = info["learner_seat"]
        opposite_seat = "debater_b" if learner_seat == "debater_a" else "debater_a"
        return {
            learner_seat:  (None, None),   # fall through to orch default = learner
            opposite_seat: (opp_client, opp_model),
            "judge":       (judge_client, judge_model),
        }
    return resolver
```

### STAGE 3 — `DebateEnv.__init__` (verifiers)

Add the kwarg and a new cross-check:

```python
def __init__(
    self, schedule, prompts, members, *,
    agent_overrides=None,
    agent_overrides_resolver: Callable[[State], dict] | None = None,   # NEW
    **kwargs,
):
    super().__init__(..., agent_overrides=agent_overrides, ...)
    self._resolver = agent_overrides_resolver
    # existing cross-checks 1 (members≡rubric.members),
    #                       2 (members≡slot agents),
    #                       3 (schedule×prompts coverage)
    #
    # NEW cross-check 4 (probe the resolver):
    if self._resolver is not None:
        dummy = State()
        dummy["input"] = {"info": {"learner_seat": members[0]}}
        overrides = self._resolver(dummy)
        missing = set(members) - set(overrides)
        if missing:
            raise ValueError(
                f"agent_overrides_resolver returns no override for "
                f"scheduled member(s) {sorted(missing)}; dummy probe "
                f"state = {dummy!r}, returned keys = {sorted(overrides)}"
            )
```

`resolve_agent(member_id, state)` routes through `self._resolver(state)` when set, else falls back to the existing pack-level `agent_overrides` dict.

### STAGE 4 — Scheduler picks example → env.rollout

```
input = RolloutInput(
  prompt=…, answer="B", example_id="gpqa_0042",
  task="gpqa-debate",
  info={"learner_seat": "debater_b"},   # ← from dataset row
)

env.rollout(input, orch_inference_client, learner_model)
```

### STAGE 5 — Rollout loop (per turn)

```
for slot in schedule:
    for member_id in slot.agents:
        client, model = env.resolve_agent(member_id, state)
          # resolver(state) reads state["info"]["learner_seat"]
          # returns per-member (client, model)
          # learner_seat gets (None, None) → use orch defaults
          # opposite gets (opp_client, opp_model) → fixed opponent
          # judge gets (judge_client, judge_model) → fixed judge

        prompt = env.build_prompt(state, member_id, slot)
        response = await client.get_response(prompt, model)
        # trajectory step committed with extras["member_id"] = member_id
        kernel.apply_action(...)

rubric.score_rollout(state)  # writes state["mar_score"]
```

### STAGE 6 — Serialization (state_to_output)

```
RolloutOutput {
  example_id: "gpqa_0042"
  task: "gpqa-debate"
  info: {"learner_seat": "debater_b"}   ← preserved through State
  trajectory: [A propose, B propose, A critique, B critique, judge final]
  sampling_args: {...}
  trajectory_id: "gpqa_0042_ep_000"
  reward: mar.episode_scalar          # ← vestigial in free-choice, see issue #3
  mar_score: MARScore(
    members=[
      MemberScore("debater_a", reward=-1.0),  # opp (fixed) lost
      MemberScore("debater_b", reward=+1.0),  # learner won
      MemberScore("judge",     reward= 0.0),
    ],
    …)
  reward/debater_a: -1.0   # flat projection for wandb
  reward/debater_b: +1.0
  reward/judge:      0.0
  final_correct/debater_a: …   # per-member diagnostic (did their pick match target?)
  final_correct/debater_b: …
}
```

### STAGE 7 — Fan-out + learner-seat filter (prime-rl)

```python
def fan_out_for_multi_agent(
    rollouts, *,
    drop_judge: bool = True,
    learner_seat_from_info: bool = False,   # NEW
) -> tuple[list[MemberRollout], list[list[int]]]:
    training_units, rollout_to_unit_idxs = [], []
    for rollout in rollouts:
        members = rollout_to_member_rollouts(rollout)
        if drop_judge:
            members = [m for m in members if m["member_id"] != "judge"]
        if learner_seat_from_info:
            info = rollout.get("info") or {}
            learner_seat = info.get("learner_seat")
            if learner_seat is None:
                raise ValueError(
                    f"filter_by_learner_seat=True but rollout example_id="
                    f"{rollout['example_id']} has no info.learner_seat — "
                    f"dataset builder must populate this field"
                )
            members = [m for m in members if m["member_id"] == learner_seat]
        rollout_to_unit_idxs.append(
            list(range(len(training_units),
                       len(training_units) + len(members))))
        training_units.extend(members)
    return training_units, rollout_to_unit_idxs
```

Orchestrator invocation passes `learner_seat_from_info=config.multi_agent.filter_by_learner_seat`.

### STAGE 8 — RAE advantage (per learner unit)

```
For each unit:
  key = (task, example_id, member_id) = ("gpqa-debate", "gpqa_0042", "debater_b")

  SPIRAL Alg.1:
    b[key] ← α · b[key] + (1 − α) · reward
    A(unit) ← reward − b[key]

Because learner_seat varies per example_id:
  - Same example_id ALWAYS hits the same (task, ex, learner_seat) bucket
    (Bernoulli + fixed seed ⇒ replayable, hash-mode ⇒ order-invariant)
  - Opponent-seat rollouts NEVER enter RAE state (filtered out at STAGE 7)
  - No bucket pollution, no wasted EMA updates
```

### STAGE 9 — Pretokenize + interleave → TrainingSamples

Only the learner's trajectory steps reach the tokenizer. Opponent + judge tokens are physically absent from the training batch because their MemberRollouts were discarded at fan-out.

### STAGE 10 — Transport → trainer → gradient

```
TrainingBatch(examples=[learner-only samples], step=…)
  → transport
  → trainer (DPPO+KL loss)
  → gradient on learner tokens only
```

## Validation / fail-loud points

| Stage | Invariant | Check |
|-------|-----------|-------|
| Dataset build | Every row has `info.learner_seat ∈ members` | Builder assigns — no silent default |
| DebateEnv init | Resolver covers every scheduled member | NEW cross-check 4 at `__init__`, dummy-state probe |
| DebateEnv init | Resolver + existing cross-checks 1/2/3 compatible | Unchanged |
| resolve_agent | Resolver call succeeds per state | Exception → rubric error boundary → errored MARScore |
| state_to_output | `info` propagates to RolloutOutput | Existing (optional field saved when present) |
| Fan-out filter | Rollout has `info.learner_seat` iff filter enabled | Loud ValueError if filter=True and missing |
| Fan-out filter | `learner_seat` in this rollout's member set | Empty post-filter units warn/error |
| RAE | `(task, ex, member_id)` keys survive resume | Existing cross-batch EMA + checkpoint round-trip |

## Implementation plan

### Verifiers branch: `feat/agent-overrides-resolver` (~120 LOC)

1. `DebateEnv.__init__`: add `agent_overrides_resolver` kwarg; store as `self._resolver`
2. `DebateEnv.resolve_agent(member_id, state)`: route through resolver when set, else fall back to pack-level `agent_overrides`
3. `DebateEnv.__init__`: cross-check 4 — dummy-state probe; assert coverage of scheduled members; loud ValueError with returned keys if missing
4. `gpqa_debate.load_environment`: accept `opponent_*`, `judge_*`, `learner_seat_mode`, `learner_seat_seed` kwargs; build resolver + dataset; fallback to existing pack-level `agent_overrides` when opponent/judge absent
5. `_assign_learner_seat`: five modes (bernoulli/round_robin/hash/fixed_a/fixed_b); seeded RNG for bernoulli
6. Tests (verifiers/tests/):
   - resolver-called-per-turn with state-dependent output
   - resolver-returns-incomplete-coverage → init raises with enumerated missing members
   - dataset-builder produces `info.learner_seat` per each of the 5 modes
   - seed=0 reproducible across two dataset builds (bernoulli mode)
   - hash mode stable under list shuffle
   - resolver=None falls back to pack-level `agent_overrides` (self-play still works)
   - resolver and pack-level `agent_overrides` both None is an error only if a scheduled member has neither

### Prime-rl branch: `feat/learner-seat-filter` (~30 LOC)

1. `MultiAgentConfig`: `filter_by_learner_seat: bool = False`
2. `fan_out_for_multi_agent`: `learner_seat_from_info: bool = False` kwarg; lookup + filter logic; loud on missing
3. `orchestrator.py`: pass `config.multi_agent.filter_by_learner_seat` through; bump verifiers pin to the SHA where the resolver PR lands
4. Tests:
   - filter=True drops opposite-seat units
   - filter=True drops judge (via pre-existing drop_judge) AND opposite-seat
   - filter=True + rollout-missing-info.learner_seat → ValueError
   - filter=False preserves existing self-play behavior
5. Recipe: `configs/gpqa_debate/rl_learner_vs_fixed.toml` (sketched above)
6. Extend `test_ma_fan_out.py` with 2 new cases (keep-only + missing-info-raises)

### Stage 3 smoke run

Launch `configs/gpqa_debate/rl_learner_vs_fixed.toml` for 50 steps. Canary list:

```
seat_balance             ≈ 0.5  per-batch rate of learner==debater_a (Bernoulli check)
learner_reward_mean      ∈ (-1, +1)  sanity (not stuck at extremes)
reward_variance          > 0  RAE baseline is actually doing work
policy_norm_delta        > 0 but bounded — learner is updating, not exploding
judge_picks_truth_rate   trending up? (winner × final_correct post-hoc, see below)
response_length          stable — no "thinking collapse" from SPIRAL
rae_baselines            non-zero, bounded magnitudes
```

Truth-aligned eval metric (post-hoc from per-rollout data, no code change):

```
winner_was_truthful := (winner ∉ {None, "tie"})
                       AND (final_correct/<winner> == 1.0)
```

Track this over training — if it rises, debate is functioning as a truth-seeking protocol on this dataset. If it falls (sycophancy drift), Khan-et-al's consultancy failure mode has arrived; investigate.

## Open issues / deferred work

- **Assigned-answer protocol** ([joanvelja/verifiers#3](https://github.com/joanvelja/verifiers/issues/3)): current packs are effectively free-choice; `truth_member` is vestigial for `episode_scalar`. Real assigned-answer needs a new YAML pack + dynamic `truth_member` + dataset-side per-seat assignment fields. Not blocking Stage 3.
- **Per-episode `truth_member` randomization**: even in free-choice, there may be position bias (e.g. judge prefers first speaker). If observed in smoke data, randomize the schedule order or flip `info.truth_member` per row. Defer until data shows a signal.
- **VLM + MA**: currently rejected at orchestrator startup. No use case yet.
- **League play / curriculum**: resolver architecture supports it; dataset builder extension + a checkpoint client registry ~60 LOC when we want it.

## One-sentence summary

Per-episode `info.learner_seat` (Bernoulli or round-robin or hash) threads through a state-aware resolver on `DebateEnv` to pin the fixed opponent and judge clients, while prime-rl filters post-bridge MemberRollouts to the learner seat before RAE — ~150 LOC across two repos, composable with league play and curriculum, no ornamental pieces.
