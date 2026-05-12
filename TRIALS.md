# Trial Journal

This file is the durable experiment journal for agent-driven iteration. Append to
it whenever a run, code change, config change, topology change, or diagnosis
materially changes what the next agent should try.

## Protocol

For each trial, record:

- **Hypothesis**: what should improve and why.
- **Change**: exact files/configs/commands or run IDs.
- **Evidence**: metrics, logs, wall-clock, failure mode, and sample size.
- **Verdict**: `keep`, `discard`, `needs rerun`, or `promising but not proven`.
- **Next action**: the smallest follow-up that would update beliefs.

Do not keep a change just because it launched. Keep it when it improves a target
metric, removes a real blocker, or creates useful observability. Mark weak or
failed trials explicitly so future agents do not re-run them by accident.

## 2026-05-08: 100-Step Default vs MaxRL Canary

**Hypothesis**: With `14 inference / 2 trainer`, `max_off_policy_steps=8`,
trainer-side importance sampling, and `100x8` eval every 10 steps, MaxRL may
recover learning signal or improve high-k pass rates versus default.

**Change**:

- Default config:
  `tmp/rl_olmo3_dpo_default_14i2t_off8_eval10_100step_20260508_1421.toml`
- MaxRL config:
  `tmp/rl_olmo3_dpo_maxrl_14i2t_off8_eval10_100step_20260508_1421.toml`
- Topology: `14 inference / 2 trainer` on four GH200 nodes.
- Shared shape: `max_steps=100`, `token_batch_size=524288`,
  `max_inflight_rollouts=3072`, `rollouts_per_example=8`,
  `max_off_policy_steps=8`, `importance_ratio_clip=5.0`, eval every 10 steps
  with `100` prompts x `8` rollouts.

**Evidence**:

| row | p@1 | p@2 | p@4 | p@8 | eval time |
|---|---:|---:|---:|---:|---:|
| default best, ckpt60 | 0.3925 | 0.5118 | 0.6121 | 0.6900 | 438.14s |
| MaxRL best, ckpt80 | 0.3750 | 0.4925 | 0.6027 | 0.6900 | 370.19s |
| default final | 0.3762 | 0.4800 | 0.5630 | 0.6200 | 1850.72s |
| MaxRL final | 0.3825 | 0.4839 | 0.5597 | 0.6100 | 1723.97s |

Both curves were high-variance. Default peaked earlier and higher on p@1/p@2/p@4.
MaxRL had a good ckpt80 row, but final lost the high-k gain. Final evals were
much slower than interval evals. During MaxRL final eval all 14 inference GPUs
were at 100% and the 2 trainer GPUs were idle, so the final-eval issue was
long/straggly generation plus final logging, not idle inference.

**Verdict**: `discard as promotion`. MaxRL is not a clear quality or throughput
win as configured. Keep the evidence, not the algorithm change as default.

**Next action**: target the eval/cancel/refill sawtooth and generation tail
before more MaxRL tuning. Separately test `generation_config="auto"` / stop
semantics and paired-bootstrap eval analysis.

## 2026-05-08: 12i/4t vs 14i/2t Topology

**Hypothesis**: More trainer GPUs could improve training throughput enough to
offset fewer inference GPUs.

**Change**:

- Ran sequential 10-step default and MaxRL canaries with `12 inference / 4 trainer`.
- Compared against 14i/2t evidence from later runs and earlier speed work.

**Evidence**:

- `12i/4t` showed about `50%` wait fraction vs `14i/2t` around `38%`.
- `12i/4t` had about `2.7x` lower compute utilization in the observed
  comparison and worse eval direction for both 10-step runs.
- Later 100-step `14i/2t` runs still showed trainer/backpressure limits, but the
  4-trainer topology did not scale well enough in the tested setup to justify
  spending four GPUs there.

**Verdict**: `discard for current canaries`. Do not use `12i/4t` as the default
next topology without a new trainer-scaling change.

**Next action**: keep `14i/2t` for comparable canaries. If revisiting topology,
make it a focused trainer-scaling trial with clear falsifiers, not a mixed
algorithm comparison.

## 2026-05-08: Off-Policy Cap and Importance Sampling

**Hypothesis**: The old `max_off_policy_steps=32` was too stale for OLMo3
Omni-MATH-2 canaries; reducing to `8` and adding trainer-side importance
sampling should keep async speed while limiting stale-policy damage.

**Change**:

- Set the active speed canaries to `max_off_policy_steps=8`.
- Used `importance_ratio_clip=5.0`.
- Verified trainer logs included importance-ratio metrics in MaxRL.

**Evidence**:

- Earlier stale/off-policy runs reached high off-policy levels quickly and did
  not produce reliable 10-step learning evidence.
- 100-step runs with `off8` completed end-to-end and produced interpretable
  checkpoint trajectories.
- Importance-ratio metrics were logged; MaxRL final W&B summary included
  `importance_ratio/mean=0.77347`, `median=0.77879`, `max=1.00087`.

**Verdict**: `keep as safety rail`. This does not prove the learning recipe is
optimal, but `off8 + IS metrics` is better provenance than `off32` for current
canaries.

**Next action**: do not loosen staleness for throughput unless the run is
explicitly labeled as stale-policy throughput exploration. If loosening, compare
paired quality and off-policy metrics, not wall-clock alone.

## 2026-05-08: Eval-Every-10 100-Step Canaries

**Hypothesis**: 10-step canaries were too short and noisy; 100 steps with eval
every 10 should expose trajectory shape and mid-run regressions.

**Change**:

- Promoted from 10/40-step smokes to matched 100-step default and MaxRL canaries.
- Used eval interval `10` rather than only midpoint/final.

**Evidence**:

- Default: ckpt60 spike, ckpt70 collapse, partial ckpt80 recovery, weak final.
- MaxRL: ckpt80 spike, ckpt90 collapse, weak high-k final.
- Without eval every 10, both runs would have hidden the best checkpoint and the
  subsequent regression.

**Verdict**: `keep`. Eval-every-10 is worth the wall-clock when the goal is
learning dynamics, not just launch/perf smoke.

**Next action**: for speed-only smokes, keep shorter runs. For learning claims,
use at least `100` steps or a bigger eval set with paired uncertainty.

## 2026-05-08: Final Eval Tail

**Hypothesis**: The final eval slowdown is an orchestration bubble.

**Change**:

- Monitored final eval, tmux output, Slurm steps, and gpustat during default and
  MaxRL final evals.

**Evidence**:

- Default final eval: `1850.72s`.
- MaxRL final eval: `1723.97s`.
- MaxRL final gpustat showed 14 inference GPUs at 100% and the 2 trainer GPUs
  idle/free.
- Interval evals were typically `~358-447s`.

**Verdict**: `hypothesis rejected`. The final path is slow, but not due to idle
inference. It is dominated by long/straggly generation and final logging.

**Next action**: inspect eval scheduling and completion length/stop behavior.
Test `generation_config="auto"` or explicit stop-token behavior before changing
topology for this symptom.

## 2026-05-09: Upstream Docs Audit

**Hypothesis**: The weak uplift and slow wall-clock may come from running against
upstream PRIME-RL assumptions rather than just normal canary noise.

**Change**:

- Read upstream `origin/main` docs under `docs/`.
- Read `examples/hendrycks_sanity/README.md` and its example TOMLs.
- Parallelized topic reviews across async/off-policy, deployment/topology,
  metrics/logging, config examples, and skeptical run-design audit.

**Evidence**:

- Upstream async training explicitly expects one gradient step per batch while
  inference generates the next batch. Our run shape matches that assumption.
- `max_off_policy_steps=8` matches the upstream config default, but it is not
  the same knob as `max_async_level`; the former controls stale rollout
  retention, while the latter controls trainer/inference checkpoint lag.
- `cancel_inflight_rollouts_on_eval=true` is upstream-supported but explicitly
  documented as slower because the pipeline must refill after eval. Our 100-step
  pair showed exactly this: every eval boundary cancelled about `1.4k-1.9k`
  rollout requests and then paid a long refill step.
- Hendrycks sanity uses `batch_size=512`, `rollouts_per_example=8`, 4 inference
  GPUs, 4 trainer GPUs, and eval interval `50`. Our `token_batch_size=524288`,
  `14 inference / 2 trainer`, eval-every-10 run is a canary/throughput probe, not
  an upstream-like sanity-check replication.
- Upstream recommends `--bench` for trainer/inference MFU and W&B log-extras
  tables/distributions for RL sample-level diagnosis. We mostly used live logs,
  W&B summaries, and gpustat, so we have good evidence but incomplete
  doc-aligned observability.

**Verdict**: `keep diagnosis`. The run is not invalid, but the speed recipe mixed
learning-trajectory diagnosis with a throughput-hostile eval/cancel policy.

**Next action**: run a narrow control with `cancel_inflight_rollouts_on_eval=false`
and the same eval cadence before another algorithm variant. Also add vLLM
`/metrics` sampling and a `--bench` trainer/inference check before making stronger
MFU/topology claims.

## 2026-05-09: Hendrycks Sanity Batch/Runtime Audit

**Hypothesis**: The upstream Hendrycks sanity example may imply that our token-based
batching is hiding too-small independent-prompt batches, and the 5000-step budget
may reveal their expected wall-clock.

**Change**:

- Read `examples/hendrycks_sanity/{README.md,rl.toml,slurm_rl.toml}` from
  `origin/main` and the earlier `joanvelja/hendrycks-sanity` branch.
- Checked Git history for Hendrycks-specific config changes.
- Searched public web/W&B/GitHub evidence for run duration.

**Evidence**:

- Upstream Hendrycks uses fixed `batch_size=512`, `rollouts_per_example=8`, and
  eval interval `50`; it does not use `token_batch_size`.
- Commit `950f99d` removed `oversampling_factor=2.0` with message
  `do not oversample hendrycks sanity (trainer bottlenecks)`.
- Nightly CI runs the same example at `--max-steps 1000` rather than 5000 "to
  finish in time".
- Successful public GitHub Actions nightlies put the 1000-step Hendrycks job at
  about `9.4-9.7h` experiment time on their `research-cluster` H200 runner.
- Public search did not find an indexed W&B run. The nightly uses W&B project
  `nightly-tests` and run name `hendrycks-sanity-{branch}`, likely under the CI
  account from the secret API key.

**Verdict**: `promising but not proven`. Fixed `batch_size` is a better learning
signal control than `token_batch_size` if we care about independent prompt groups
per update. TBS remains useful for throughput but made our "so much data" intuition
misleading.

**Next action**: add a paired fixed-`batch_size` canary axis. Start smaller than
upstream Hendrycks, e.g. `batch_size=128` or `256` with `rollouts_per_example=8`,
`cancel_inflight_rollouts_on_eval=false`, and the same eval cadence, then compare
effective prompt groups/update, trainer MFU, and eval direction.

## 2026-05-09: Runtime Reconciliation vs Hendrycks CI

**Hypothesis**: The Hendrycks 1000-step nightly makes our 100-step OLMo3 run look
too slow unless the difference is mostly model/config/runtime shape.

**Change**:

- Compared our 100-step default/MaxRL wall-clock to public Hendrycks nightly job
  metadata.
- Re-read the active OLMo3 100-step config and Hendrycks example config.

**Evidence**:

- Hendrycks nightly runs `DeepSeek-R1-Distill-Qwen-1.5B`, `batch_size=512`,
  `rollouts_per_example=8`, 4 inference GPUs, 4 trainer GPUs, eval interval `50`,
  and CI overrides to `max_steps=1000`.
- Recent public successful Hendrycks jobs take about `9.4-9.7h` experiment time
  for 1000 steps on their H200 `research-cluster` runner, or about `0.95h` per
  100 steps before accounting for setup.
- Our default 100-step OLMo3 run used a 7B model, `14 inference / 2 trainer`,
  `token_batch_size=524288`, `max_inflight_rollouts=3072`, eval interval `10`,
  and `cancel_inflight_rollouts_on_eval=true`.
- Our default run took about `3.84h` from first orchestrator step to final-eval
  start and `4.36h` including final eval. That is roughly `4x` slower per 100
  steps than Hendrycks CI, but the comparison is not model/config matched.
- The dominant avoidable terms were interval evals (`~6-7m` each), repeated
  `1.4k-1.9k` rollout cancellations, and `~11-13m` post-eval refill steps. The
  dominant unavoidable terms are 7B-vs-1.5B model scale, longer Omni-MATH
  generations, 16K context, and external judge/math-verify overhead.

**Verdict**: `explained enough to stop panicking, not explained enough to stop
optimizing`. The run is slow for coherent reasons, but `eval_every_10 +
cancel_inflight_rollouts_on_eval=true` is self-inflicted wall-clock loss.

**Next action**: measure a fixed-`batch_size` canary and a no-cancel eval canary
separately. For throughput claims, report wall-clock split into steady train
steps, eval wall-clock, post-eval refill, and final-eval tail.

## 2026-05-09: 1000-Step Fixed-Batch Scale Setup

**Hypothesis**: The next learning run should use upstream-style fixed sample
batching rather than token batching, and remove the eval/cancel/refill sawtooth
before judging Default vs MaxRL over 1000 steps.

**Change**:

- Added temporary 1000-step configs:
  - `tmp/rl_olmo3_dpo_default_14i2t_bs256_eval50_1000step_20260509_1825.toml`
  - `tmp/rl_olmo3_dpo_maxrl_14i2t_bs256_eval50_1000step_20260509_1825.toml`
- Added sequential launcher:
  - `tmp/run_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh`
- Shape: `14 inference / 2 trainer`, fixed `batch_size=256`,
  `rollouts_per_example=8`, `max_off_policy_steps=8`, eval `100x8` every `50`,
  `cancel_inflight_rollouts_on_eval=false`, checkpoint every `50`,
  `keep_last=4`, same 15,360 train/eval completion cap, and W&B extras interval
  overridden to `50` via validated CLI override.

**Evidence**:

- Dry-runs passed for both configs against active allocation `4507057` with hosts
  `nid011043,nid011047,nid011074,nid011086`.
- Leaving `max_inflight_rollouts` unset in rollout-batch mode resolves to
  `256`. Do not set `768`: that is effectively 3x oversampling, and upstream
  removed Hendrycks oversampling because the trainer bottlenecked.
- Prior 100-step token-batched rollouts had sample-token mean/p50/p95/max:
  - Default: `5237 / 3578 / 15513 / 15951`, truncation `8.6%`.
  - MaxRL: `5138 / 3492 / 15506 / 15890`, truncation `8.1%`.
- Bootstrapping old rollout lengths to `batch_size=256` suggests typical packed
  microbatches per trainer step around `72-74`, p95 around `79-81`, versus old
  token-batched median around `31`. Worst old-step resampling can reach
  `~144-148` packed microbatches. This should not increase peak activation
  memory because microbatch `seq_len` remains capped at `18432`, but it can make
  individual training steps much longer.
- Active allocation has a 24h wall clock ending `2026-05-10 17:27 UTC`; the full
  Default+MaxRL pair may not fit. Checkpoint interval 50 is therefore important
  for resume.

**Verdict**: `launchable but expensive`. `batch_size=256` is reasonable as the
first fixed-batch scale trial; increasing in-flight capacity before measuring
starvation would recreate the stale/oversampled regime we just diagnosed.

**Next action**: launch the sequential script visibly in tmux, verify all 14
inference servers and the 2-GPU trainer start, then inspect step 0-5 trainer
memory/time before trusting the rest of the allocation.

## 2026-05-09: 1000-Step Fixed-Batch Launch / Early Runtime

**Hypothesis**: Fixed `batch_size=256` with default resolved
`max_inflight_rollouts=256` should preserve the upstream one-step-ahead async
shape while giving larger independent-prompt batches than token batching.

**Change**:

- Launched `tmp/run_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh` in
  tmux window `joanv_cc_4node:omni-bs256-1000`.
- Default run started first, with MaxRL queued only after Default exits
  successfully.

**Evidence**:

- Output dir:
  `outputs/omni_math2_rlvr_canary/20260509_1825/dpo_default_14i2t_bs256_eval50_1000step`.
- W&B run id: `197d96382b0c40d59272ac0fbc94a9e3`.
- Resolved orchestrator config confirms this is the real training run:
  `bench=false`, `batch_size=256`, `max_inflight_rollouts=256`,
  `max_steps=1000`.
- All 14 inference servers reached readiness; orchestrator started step 0 at
  `18:42:06`.
- Step 0 completed at `18:47:12`: `304.19s`, reward `0.4453`, mean sequence
  length `4718.3` tokens/sample, async/off-policy `0/0`, and `1/256`
  repetition-flagged rollout.
- Trainer step 0 completed at `18:49:14`: `427.38s`, peak mem `81.0 GiB`.
  Logged throughput/MFU were `0` because the trainer `PerfCounter` has no prior
  sample on the first step; do not interpret this as real trainer MFU.
- Step 1-8 orchestrator times were `218.65s`, `229.41s`, `222.54s`,
  `186.94s`, `178.45s`, `233.70s`, `229.54s`, and `184.40s`; rewards were
  `0.4492`, `0.4496`, `0.4062`, `0.4258`, `0.4648`, `0.3633`, and
  `0.4254`, and `0.3672`; mean seq lengths were `6361.1`, `5703.8`,
  `5862.6`, `5570.5`, `4594.8`, `5106.3`, `6742.5`, and `5801.5`.
- Trainer step 1 completed at `18:54:08`: `290.79s`,
  `5640 tokens/s`, `20.0%` MFU, peak mem `93.4 GiB`. This is the first
  meaningful MFU point.
- Trainer step 2 completed at `18:57:35`: `204.42s`, `6248 tokens/s`,
  `22.2%` MFU, peak mem `93.4 GiB`.
- Trainer step 3 completed at `19:01:17`: `218.88s`, `6464 tokens/s`,
  `22.9%` MFU, peak mem `93.4 GiB`.
- Trainer step 4 completed at `19:04:16`: `175.78s`, `6774 tokens/s`,
  `24.0%` MFU, peak mem `93.4 GiB`.
- Trainer step 5 completed at `19:06:54`: `155.35s`, `6912 tokens/s`,
  `24.5%` MFU, peak mem `93.4 GiB`.
- Trainer step 6 completed at `19:10:58`: `240.66s`, `6636 tokens/s`,
  `23.6%` MFU, peak mem `93.4 GiB`.
- Trainer step 7 completed at `19:15:43`: `282.78s`, `6555 tokens/s`,
  `23.3%` MFU, peak mem `93.4 GiB`.
- Trainer step 8 completed at `19:18:41`: `174.44s`, `6751 tokens/s`,
  `24.0%` MFU, peak mem `93.4 GiB`.
- Trainer step 9 completed at `19:22:10`: `205.08s`, `6784 tokens/s`,
  `24.1%` MFU, peak mem `93.4 GiB`.
- Trainer step 10 completed at `19:25:50`: `217.80s`, `6851 tokens/s`,
  `24.3%` MFU, peak mem `93.4 GiB`.
- Trainer step 11 completed at `19:29:24`: `211.01s`, `6747 tokens/s`,
  `24.0%` MFU, peak mem `93.4 GiB`.
- Trainer step 12 completed at `19:32:59`: `211.32s`, `6677 tokens/s`,
  `23.7%` MFU, peak mem `93.4 GiB`.
- Trainer step 13 completed at `19:35:29`: `146.47s`, `6643 tokens/s`,
  `23.6%` MFU, peak mem `93.4 GiB`.
- Trainer step 14 completed at `19:39:37`: `244.35s`, `6487 tokens/s`,
  `23.0%` MFU, peak mem `93.4 GiB`.
- Trainer step 15 completed at `19:43:53`: `253.45s`, `6613 tokens/s`,
  `23.5%` MFU, peak mem `93.4 GiB`.
- Trainer step 16 completed at `19:48:14`: `258.51s`, `6545 tokens/s`,
  `23.2%` MFU, peak mem `93.4 GiB`.
- Trainer step 17 completed at `19:51:14`: `176.89s`, `6517 tokens/s`,
  `23.1%` MFU, peak mem `93.4 GiB`.
- Trainer step 18 completed at `19:56:22`: `304.63s`, `6327 tokens/s`,
  `22.5%` MFU, peak mem `93.4 GiB`.
- Trainer step 19 completed at `20:00:01`: `215.72s`, `6458 tokens/s`,
  `22.9%` MFU, peak mem `93.4 GiB`.
- Trainer step 20 completed at `20:03:05`: `181.60s`, `6664 tokens/s`,
  `23.7%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 9 completed at `19:19:18`: `238.55s`, reward `0.4023`,
  mean seq length `5619.8`, async/off-policy `0/2`, and `1/256`
  repetition-flagged.
- Orchestrator step 10 completed at `19:23:17`: `238.43s`, reward `0.3906`,
  mean seq length `4921.7`, async/off-policy `0/1`, and `3/256`
  repetition-flagged.
- Orchestrator step 11 completed at `19:26:53`: `215.80s`, reward `0.3828`,
  mean seq length `5071.5`, async/off-policy `0/2`, and `3/256`
  repetition-flagged.
- Orchestrator step 12 completed at `19:30:21`: `207.28s`, reward `0.3945`,
  mean seq length `5268.7`, async/off-policy `0/2`, and `2/256`
  repetition-flagged.
- Orchestrator step 13 completed at `19:33:12`: `169.92s`, reward `0.4375`,
  mean seq length `4491.9`, async/off-policy `0/2`, and `1/256`
  repetition-flagged.
- Orchestrator step 14 completed at `19:36:39`: `207.49s`, reward `0.3633`,
  mean seq length `5819.7`, async/off-policy `0/2`, and `4/256`
  repetition-flagged.
- Orchestrator step 15 completed at `19:40:42`: `241.63s`, reward `0.3548`,
  mean seq length `6273.6`, async/off-policy `0/1`, and `2/256`
  repetition-flagged.
- Orchestrator step 16 completed at `19:45:21`: `279.30s`, reward `0.3789`,
  mean seq length `5724.9`, async/off-policy `0/1`.
- Orchestrator step 17 completed at `19:48:26`: `184.38s`, reward `0.3789`,
  mean seq length `5675.3`, async/off-policy `0/2`, and `2/256`
  repetition-flagged.
- Orchestrator step 18 completed at `19:53:03`: `276.33s`, reward `0.3906`,
  mean seq length `6667.5`, async/off-policy `0/1`.
- Orchestrator step 19 completed at `19:57:01`: `237.28s`, reward `0.3711`,
  mean seq length `5888.4`, async/off-policy `0/1`, and `1/256`
  repetition-flagged.
- Orchestrator step 20 completed at `19:59:16`: `134.53s`, reward `0.3891`,
  mean seq length `5886.5`, async/off-policy `1/1`, and `2/256`
  repetition-flagged. Step 21 then paused `45.05s` waiting for trainer
  checkpoint 20, so short-tail inference can still outrun the 2-GPU trainer.
- Generated rollout JSONLs for steps 0-20 have exactly `256` rows each, so
  fixed-batch mode is producing `32` prompt groups/update with
  `rollouts_per_example=8`. Truncation counts for steps 0-4 were
  `12, 27, 27, 26, 20`; repetition flags were `1, 1, 1, 4, 0`.
- First small trainer-backpressure event appeared at orchestrator step 9:
  paused `24.03s` waiting for trainer checkpoint 8, then resumed. This is not
  the old severe bubble, but the run is now roughly trainer/inference matched.
- Live `gpustat` at `18:56:49` showed all 16 GPUs at `100%` util and total
  memory `1452143 / 1565936 MiB`. Trainer GPUs were very close to full in the
  snapshot (`96327 / 97871` and `96945 / 97871` MiB), so `batch_size=256` fits
  but has little memory slack.
- Live `gpustat` at `19:25:39` again showed all 16 GPUs at `100%` util and total
  memory `1452855 / 1565936 MiB`; trainer GPUs were around `96.8-97.1 / 97.9`
  GiB. This confirms high device occupancy, but it does not mean high trainer
  MFU.
- Live `gpustat` at `19:29:10` again showed all 16 GPUs at `100%` util and total
  memory `1452535 / 1565936 MiB`; later `87-90%` aggregate snapshots caught the
  trainer ranks between phases, not an inference queue starvation event.
- Live `gpustat` at `19:49:24` showed `98%` aggregate GPU util and total memory
  `1451515 / 1565936 MiB`; trainer GPUs were around `96.2-96.4 / 97.9` GiB.
- Live `gpustat` at `19:56:22` showed all 16 GPUs at `100%` util and total
  memory `1428975 / 1565936 MiB`; one trainer GPU was at `97045 / 97871` MiB.
- Live `gpustat` at `20:00:29` showed `99%` aggregate GPU util and total memory
  `1438677 / 1565936 MiB`; trainer GPUs were at `95865 / 97871` and
  `83943 / 97871` MiB.
- Added a vLLM `/metrics` sampler writing to
  `outputs/omni_math2_rlvr_canary/20260509_1825/dpo_default_14i2t_bs256_eval50_1000step/monitor/vllm_metrics.tsv`.
  Across all 14 inference servers, scrapes showed `num_requests_waiting=0`,
  no preemptions, KV cache roughly `0.17-0.57`, and about `14-21` running
  requests/server. Early rollout latency is decode-volume/tail dominated, not
  vLLM queue-capacity dominated.
- The resolved config has `bench=false`, so this is normal training, not
  `--bench`. Trainer step 1-20 average is `6565 tokens/s` and `23.30%` MFU;
  steps 5-20 average about `23.6%` MFU. Orchestrator steps 1-20 average
  `215.73s` with two backpressure pauses totaling `69.08s`. Allocation-level
  GPU util can read `100%` while trainer MFU remains only mid-20s.
- No OOM/traceback/runtime-error/inference-failure log hits as of the
  20:00 UTC refresh.
- 20:18 UTC refresh: trainer reached step 24; orchestrator reached step 25 and
  is generating step 26. Trainer step 24 was the best live MFU point so far:
  `165.58s`, `7162 tokens/s`, `25.4%` MFU, peak mem `93.4 GiB`.
- Aggregate trainer steps 1-24: mean step time `213.88s`, mean throughput
  `6620 tokens/s`, mean MFU `23.50%`, MFU range `20.0-25.4%`, peak memory
  `93.4 GiB`.
- Aggregate orchestrator steps 0-25: mean step time `218.09s`, mean reward
  `0.3986`, mean sequence length `5639.0` tokens/sample, max async level `1`,
  max off-policy level `2`.
- Rollout JSONLs through step 25 still have exactly `256` rows each. The latest
  vLLM scrape showed all 14 servers OK, `0` queued requests, no preemptions, and
  KV cache around `0.30-0.52` on most servers. A `20:17:33` gpustat snapshot
  showed all 16 GPUs at `100%` util and total memory
  `1451401 / 1565936 MiB`.
- Backpressure through step 25 is still modest but real: 3 pauses totaling
  `115.17s`, max `46.09s`. At the observed trainer mean, a full 1000-step
  Default arm alone is roughly `59.4h` before eval/checkpoint overhead, so the
  current 24h allocation is a stability/eval-collection run, not enough for the
  full sequential Default+MaxRL 1000-step pair unless throughput improves
  dramatically.
- 21:51 UTC refresh: default reached the first checkpoint/eval boundary and
  resumed training afterward. Trainer step 50 completed at `21:43:59`:
  `254.63s`, `7476 tokens/s`, `26.5%` MFU, peak mem `92.9 GiB`. Warm trainer
  aggregate over steps 20-50 is now `196.88s` mean step time,
  `7473 tokens/s`, `26.53%` MFU, MFU range `23.7-28.5%`, peak mem
  `93.4 GiB`.
- Ckpt-50 eval ran `100` prompts x `8` rollouts and completed at `21:47:54` in
  `485.12s`: `Avg@8=0.3638`, `Pass@1=0.3638`, `Pass@2=0.4818`,
  `Pass@4=0.5864`, `Pass@8=0.6700`, no-response `0.0%`, completion length
  `5794.74` tokens, truncated `15.0%`. This is close to the supplied OLMo
  Inst DPO 600-sample reference at p@1/p@8, but it is only 100 eval examples,
  so treat it as a noisy checkpoint row.
- `cancel_inflight_rollouts_on_eval=false` materially changed the post-eval
  behavior. Orchestrator step 51 reported `555.03s`, but that includes the
  `485.12s` eval. After eval finished, training rollout generation resumed
  immediately with `216/256` rollouts already available and completed the
  post-eval train batch in about `66s`. This is not the old
  cancel/refill cliff. The eval wall-clock itself is now the obvious periodic
  tax.
- Follow-up readback after the post-eval batch: trainer step 51 completed at
  `21:52:10` with `253.07s`, `6472 tokens/s`, `23.0%` MFU, peak mem
  `92.9 GiB`; orchestrator step 52 completed at `21:52:18` in `193.49s` with
  async/off-policy `0/1`. So the trainer does resume normally, but the first
  trainer row after eval still has lower useful utilization than the warm
  pre-eval band.
- During eval, allocation GPU util snapshots dropped to `81-87%` because the
  two trainer GPUs were idle or near-idle; during normal overlap, snapshots
  return near full allocation occupancy. Trainer MFU remains mid/high-20s, so
  high allocation util should not be confused with high useful trainer FLOPs.
- vLLM metrics at/after eval showed no sustained request queueing in the latest
  scrapes, but two servers accumulated preemption counters (`29` and `35`) and
  hit high KV usage earlier in eval. This is worth watching if eval tails grow
  or truncation rises, but it did not block post-eval training restart.
- 21:55 UTC utilization refresh: trainer steps 20-52 now average `197.56s`,
  `7413 tokens/s`, and `26.31%` MFU; latest trainer step 52 is lower at
  `163.41s`, `6476 tokens/s`, `23.0%` MFU, peak mem `92.9 GiB`. Orchestrator
  step 53 completed in `130.01s` with async/off-policy `1/1`, then step 54
  paused only `28.03s` waiting for checkpoint 53 before resuming. Gpustat at
  `21:55:52` shows all `16/16` GPUs at `100%` util, memory
  `1451595 / 1565936 MiB`; trainer GPUs are near memory ceiling
  (`95.8-96.9 / 97.9 GiB` on the trainer node) but still only mid-20s MFU.
  PrimeRL top-level `bench` remains disabled; the config default is
  `bench = false`, and enabling it would set trainer/orchestrator benchmark
  mode plus fake trainer data, not merely add telemetry. The current vLLM
  `benchmark_combo_kernel=True` entries are kernel/autotune behavior, not the
  PrimeRL benchmark toggle.
- 22:01 UTC continuation hardening: added
  `tmp/resume_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh`. It runs the
  same Default→MaxRL pair with `--ckpt.resume_step -1`, derives the first four
  hosts from `SLURM_JOB_NODELIST`, and overrides the hardcoded temp-config hosts
  via `--deployment.hosts '["host0","host1","host2","host3"]'`. `bash -n`
  passed. Dry-run validation passed for both arms with `--ckpt.resume_step -1`;
  a separate dry-run confirmed the JSON-list host override shape. The dry-runs
  used `/tmp/...` output dirs, so their "No checkpoints found" warning only
  means the syntax check did not point at the live checkpoint directory. Actual
  live evidence: Default has `run_default/checkpoints/step_50` and
  `weights/step_50/STABLE`.
- 22:02 UTC progress refresh: no OOM/traceback/error matches. Trainer has
  reached step 54 (`173.70s`, `6565 tokens/s`, `23.3%` MFU, peak mem
  `92.9 GiB`); orchestrator has reached step 55 (`169.59s`, reward `0.3246`,
  seq length `6756.7`, async/off-policy `1/1`).
- 22:04 UTC W&B event-file parse: `time/wait_for_batch` is present in the
  trainer `.wandb` event stream. Steps 20-54 average `196.67s` step time,
  `18.84s` wait-for-batch, `177.43s` forward/backward, `26.14%` logged MFU,
  and `7363 tokens/s`; mean wait fraction is only `8.25%`. Pre-eval steps
  20-49 are similar: `8.52%` mean wait fraction, `26.53%` MFU. Post-eval steps
  51-54 have similar wait fraction (`7.87%`) but lower logged MFU (`23.09%`);
  this is likely contaminated by the trainer `PerfCounter` 10-sample rolling
  window including the eval/idle gap, because throughput/MFU are computed from
  wall-clock deltas between `count_tokens()` calls. Starvation is therefore not
  the main current explanation for mid-20s MFU.
- 22:06 UTC progress/wait refresh: trainer has reached step 55 (`197.56s`,
  `6668 tokens/s`, `23.7%` MFU, peak mem `92.9 GiB`); orchestrator has reached
  step 57 (`196.01s`, reward `0.3871`, seq length `6576.9`, async/off-policy
  `1/1`). No error matches. The orchestrator is increasingly waiting for
  trainer checkpoints after finishing generation: waits of `54.10s`, `60.14s`,
  and `103.18s` appeared for checkpoints 54-56. This says the live bottleneck is
  trainer-side wall time under `max_async_level=1`; adding more inference-side
  batch/in-flight headroom alone would mostly create off-policy backlog, not
  reduce step wall-clock.
- 22:08 UTC quantified checkpoint-wait refresh: trainer is now step 56
  (`179.02s`, `6735 tokens/s`, `23.9%` MFU), orchestrator remains ahead at step
  57. Trainer steps 51-56 average `192.84s` and `23.33%` logged MFU; local W&B
  wait-for-batch rows for steps 51-55 average only `15.32s` wait on `195.61s`
  step time (`6.39%`). In contrast, orchestrator checkpoint waits after eval
  average `66.72s` over checkpoints 53-57, with latest waits
  `28.03, 54.10, 60.14, 103.18, 88.15s`. vLLM latest scrape has all 14 servers
  OK, `253` running requests, `0` waiting, mean KV `0.442`; preemption counter
  total remains `64` (`29+35` from earlier). Latest gpustat shows allocation
  util `96%` and memory `1421515 / 1565936 MiB`. Net: inference is not queueing;
  trainer progress is the current pacing item.
- 22:11 UTC progress/disk refresh: trainer reached step 57 (`189.40s`,
  `6875 tokens/s`, `24.4%` MFU, peak mem `92.9 GiB`) and orchestrator reached
  step 58 (`207.32s`, reward `0.4570`, seq length `4811.7`, async/off-policy
  `1/1`), then resumed checkpoint 58 after `73.13s`. No error matches. Disk is
  not an immediate risk: output dir is `60G`; the first full checkpoint is
  `41G`, weights step 50 is `14G`, run_default rollouts are `2.1G` across
  `59` train batches (`~29.6MiB` mean `.bin`, `~5.0MiB` mean `.jsonl`), eval
  JSONL is `15.8MiB`, and the filesystem reports `195T` free. With
  `keep_last=4`, a completed 1000-step arm should be in the hundreds of GiB, not
  a storage-threatening regime on this filesystem.
- Allocation timing: Slurm job `4507057` started `2026-05-09T17:27:21` and
  ends `2026-05-10T17:27:21`; at `22:12 UTC` it had run about `4h45m` with
  roughly `19h15m` remaining. Using post-eval trainer mean `~192s/step`, the
  next ckpt/eval-100 boundary is roughly `135min` away, plus about `8min` for
  eval if the ckpt-50 eval timing repeats. This allocation should reach eval
  100; it still will not finish a 1000-step Default arm, let alone the
  sequential MaxRL arm.
- 22:17 UTC utilization/config refresh: Default is still healthy and running.
  Trainer has reached step 59 (`178.73s`, `7323 tokens/s`, `26.0%` MFU, peak
  mem `92.9 GiB`); orchestrator has reached step 60 (`162.57s`, reward
  `0.3750`, seq length `6770.8`, async/off-policy `1/1`). No error matches.
  Current config has PrimeRL `bench=false`; only vLLM/kernel benchmark/autotune
  knobs are active. Latest gpustat snapshot shows allocation util `89%`
  (`1452255 / 1565936 MiB` used), with 14 GPUs at `100%` SM util and two
  trainer-node GPUs at low instantaneous SM util but `96-97 GiB` memory. Latest
  vLLM scrape: all 14 servers OK, `250` running, `0` waiting, mean KV `0.384`,
  preemptions still `64`. Post-eval checkpoint waits are now `8` samples with
  mean `68.50s`, max `103.18s`, latest `80.14s`. This reinforces the same call:
  useful trainer MFU is mid-20s even while the allocation is mostly busy, and
  inference queue capacity is not the active limiter.
- 22:20 UTC skill sync: updated `skills/config/SKILL.md` so future launches do
  not treat the older `max_inflight_rollouts=3072` / eval-cancel-on probe as the
  default recommendation. The skill now records the checked-in
  `524288/768/off8/cancel=false` token-budget shape, the live fixed-batch
  `batch_size=256` / `max_inflight_rollouts=256` starting point, and the
  `cancel_inflight_rollouts_on_eval` tradeoff.
- 22:23 UTC W&B trainer timing refresh: W&B history rows for trainer steps
  52-61 average `179.72s` step time, only `1.43s` wait-for-batch
  (`0.79%`), `178.02s` forward/backward, `25.36%` MFU, and `7143 tok/s`.
  Recent trainer rows improved to step 60-61 at `30.78-30.80%` MFU and
  `~8675 tok/s`, while wait-for-batch stayed at `~0.93s`. Orchestrator reached
  step 62 and still had checkpoint waits averaging `67.79s` post-eval, with vLLM
  14/14 OK and `0` waiting. This makes the current causal story sharper:
  trainer compute/weight-update/checkpoint pacing is the bottleneck, not
  inference batch supply.
- 22:25 UTC health refresh: trainer step 62 completed (`138.84s`,
  `8677 tok/s`, `30.8%` MFU, peak mem `92.9 GiB`), making steps 60-62 a
  stable `30.8%` MFU streak rather than a single noisy row. Orchestrator reached
  step 63 (`222.74s`, reward `0.3669`, seq length `6148.2`, async/off-policy
  `1/1`). A gpustat snapshot briefly showed one trainer GPU at low memory/util,
  but the next sample during checkpoint 63 weight update had all 16 GPUs at
  `100%` and trainer GPUs back at `97.2/81.2 GiB`, so this looks phase-local
  rather than a crashed rank. The run has logged `45` `Timeout during comparison`
  messages by step 63; prior source tracing says these are
  `math_verify.grader.verify()` per-sample comparison timeouts returning
  verifier-false, so treat them as reward noise/CPU waste unless the rate grows.
- 22:31 UTC steady-state refresh: trainer step 63 completed (`185.59s`,
  `8677 tok/s`, `30.8%` MFU, peak mem `92.9 GiB`), so steps 60-63 are now a
  four-step `30.8%` MFU plateau with mean `177.31s`. Orchestrator reached step
  65 and post-60 checkpoint waits have improved from `80.14,62.15,101.24s` to
  `20.02,28.03s` on the last two boundaries. vLLM still has all 14 servers OK
  with `0` waiting, and gpustat shows all 16 GPUs at `100%` with trainer GPUs at
  `95.9/96.9 GiB`. This is the first clear sign that the run may have warmed
  into a better trainer regime; do not extrapolate too hard until it survives
  more steps and the eval-100 boundary.
- 22:35 UTC sustained-warm refresh: trainer reached step 65. Steps 60-65 now
  average `176.85s`, `8659 tok/s`, and `30.73%` MFU with stable peak memory
  `92.9 GiB`; latest trainer step 65 is `185.03s`, `8618 tok/s`, `30.6%` MFU.
  Orchestrator reached step 66 (`198.96s`, reward `0.4476`, seq length
  `6322.3`, async/off-policy `1/1`). Post-60 checkpoint waits are noisy but
  lower than the immediate post-eval stall: `80.14,62.15,101.24,20.02,28.03,
  55.14,44.05s` (mean `55.82s`). vLLM remains 14/14 OK with `0` waiting and
  preemptions unchanged at `64`. One gpustat snapshot again caught a
  low-util/low-memory trainer GPU phase, so keep memory-watch active, but there
  are still no OOM/Traceback/NCCL/failure signatures.
- 22:37 UTC W&B timing confirmation: complete W&B rows for trainer steps 60-65
  average `176.85s` step time, `0.93s` wait-for-batch (`0.52%`), `175.66s`
  forward/backward, `30.74%` MFU, and `8659 tok/s`. This confirms the warm
  plateau is not hidden trainer starvation; the step time is dominated by model
  compute/weight-update/checkpoint-side pacing. At this warmed rate, trainer
  step/eval 100 is roughly `~100-115min` out from 22:36 UTC once eval overhead
  is included, assuming no regime change.
- 22:49 UTC slower-cadence refresh: trainer reached step 69; steps 60-69 average
  `178.41s`, `8629 tok/s`, `30.62%` MFU, with latest peak mem back at
  `93.4 GiB`. The warm MFU plateau is still real. However, checkpoint waits have
  climbed again: post-66 waits are `44.05,70.08,81.18,129.23,133.09s` (mean
  `91.53s`), while vLLM remains 14/14 OK with `0` waiting and preemptions still
  `64`. Orchestrator has reached step 71 and is ahead of trainer step 69, so the
  current limiter is again trainer/checkpoint pacing, not inference supply. No
  OOM/Traceback/NCCL/failure signatures.
- 22:50 UTC monitor-run status file: created
  `outputs/omni_math2_rlvr_canary/20260509_1825/dpo_default_14i2t_bs256_eval50_1000step/STATUS.md`
  with the current health snapshot. Latest trainer step 70 is `177.52s`,
  `8527 tok/s`, `30.3%` MFU, peak mem `93.4 GiB`; steps 60-70 average
  `178.33s`, `8619 tok/s`, `30.59%` MFU. vLLM remains 14/14 OK with `0`
  waiting.
- 23:27 UTC backpressure refresh: trainer reached step 83; steps 80-83 average
  `173.99s`, `8681 tok/s`, `30.83%` MFU, peak mem `92.9 GiB`, so the warm
  trainer plateau persists. Orchestrator reached step 84; steps 80-84 average
  `168.12s`, reward `0.4131`, seq length `5591.8`. The issue is checkpoint
  backpressure: post-80 checkpoint waits are `84.17,138.22,154.27,156.31,
  127.26s` (mean `132.05s`) while vLLM remains 14/14 OK with `0` waiting and
  preemptions unchanged at `64`. Recent vLLM running count dipped to `61` during
  a drained/waiting phase, then recovered to `247` after checkpoint 84 resumed.
  This is not inference-supply starvation; it is orchestrator running ahead and
  waiting on trainer/checkpoint progress. One recent orchestrator step hit max
  off-policy level `2`.
- 23:39 UTC backpressure follow-up: trainer reached step 87; steps 80-87 average
  `166.82s`, `8683 tok/s`, `30.83%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 89; steps 80-89 average `170.26s`, reward `0.4236`, seq length
  `5759.0`, async/off-policy usually `1/1` with step 82 at max off-policy `2`.
  The post-80 checkpoint-wait spike has partially relaxed: after
  `84.17,138.22,154.27,156.31,127.26s`, the next waits were
  `58.09,60.11,30.03,57.06s`. vLLM remains 14/14 OK with `0` waiting. Net:
  still trainer/checkpoint paced, but not monotonically degrading before eval
  100.
- 23:50 UTC pre-eval-100 refresh: trainer reached step 91; steps 80-91 average
  `172.50s`, `8690 tok/s`, `30.85%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 93; steps 80-93 average `168.90s`, reward `0.4121`, seq length
  `5797.3`. Checkpoint backpressure is high again: post-80 waits now include
  `110.12,168.19,137.25,154.22s`, with post-80 mean `110.41s` and max
  `168.19s`. Recent orchestrator steps 90 and 93 hit max off-policy level `2`.
  vLLM is still 14/14 OK with `0` waiting and preemptions unchanged. Eval-100
  has not started yet. Expect eval-100 interpretation to include this
  trainer/checkpoint lag and policy-staleness context.
- 23:54 UTC utilization / benchmark-mode clarification: instantaneous gpustat
  shows all 16 GPUs at `100%` util and `1,447,875 / 1,565,936 MiB` allocated
  (`~92.5%` allocation-level memory). This is not the same as useful trainer
  MFU. Trainer step 92 reported `8715 tok/s`, `30.9%` MFU, and `92.9 GiB` peak
  memory; trainer steps 80-92 average `171.75s`, `8692 tok/s`, and `30.85%`
  MFU. vLLM shows 14/14 servers OK, `249` running, `0` waiting, mean KV
  `0.350`, and preemptions still `64`. The top-level PrimeRL `bench` toggle is
  not enabled; the generated config omits `bench = true`, and the code default
  is `bench = false`. Do not enable it for this canary because it would switch
  trainer/orchestrator into benchmark/fake-data mode. The run uses normal live
  trainer MFU logging, not isolated benchmark-mode output.
- 00:12 UTC eval-100 boundary: step/eval 100 is now live. Trainer reached step
  99 with post-90 average `169.71s`, `8695 tok/s`, and `30.86%` MFU. Orchestrator
  completed step 100 at `00:09:27` (`157.25s`, reward `0.4012`, seq length
  `6622.9`, async/off-policy `1/2`) after saving ckpt-100 at `00:06:48`.
  Trainer saved checkpoint 100 at `00:12:20`, and orchestrator immediately
  started `Running evals at ckpt_step=100`. This is direct evidence of the
  checkpoint/trainer lag around eval boundaries: orchestrator reached/saved the
  boundary about `5.5min` before trainer checkpoint readiness. vLLM remained
  14/14 OK with `470` running, `0` waiting, mean KV `0.099`, and preemptions
  unchanged at eval start. gpustat during eval start showed the 14 inference
  GPUs at `100%`; the two trainer GPUs were mostly idle but memory-resident.
- 00:19 UTC eval-100 completed: `399.41s`, `Avg@8=0.3563`, `Pass@1=0.3563`,
  `Pass@2=0.4521`, `Pass@4=0.5356`, `Pass@8=0.6000`, no-response `0.0%`,
  completion length `5959.01 ± 5384.23`, truncated `16.9%`. Versus eval-50,
  this is worse on every pass metric: Avg@8/Pass@1 `-0.75 pp`, Pass@2
  `-2.97 pp`, Pass@4 `-5.08 pp`, Pass@8 `-7.00 pp`; truncation rose `+1.9 pp`.
  Orchestrator step 101 completed immediately after eval and charged the
  boundary/eval time into the step wall clock: `575.64s`, reward `0.3359`,
  async/off-policy `0/2`. Trainer steps 100-101 also cooled to `28.8%` and
  `27.3%` MFU (`8121` then `7679 tok/s`). vLLM remained 14/14 OK and `0`
  waiting, but preemptions rose from `64` to `91` after eval. Watch steps
  102-105 before deciding whether this is merely eval-time accounting or a
  true post-eval refill/staleness bubble.
- 00:31 UTC post-eval-100 recovery check: the orchestrator largely recovered
  after two inflated boundary steps, but the trainer did not recover to the
  pre-eval MFU plateau. Trainer steps 100-104 average `186.74s`, `7636 tok/s`,
  and `27.12%` MFU; steps 102-104 are flat around `26.5%` MFU and
  `~7460 tok/s`, down from the pre-eval `~30.9%` / `~8700 tok/s` plateau.
  Orchestrator steps 102-106 are `241.7,104.9,130.7,155.8,115.8s`, so the
  step-101 wall-clock spike was mostly boundary/eval accounting, not a lasting
  rollout refill stall. vLLM is still 14/14 OK, `251` running, `0` waiting,
  mean KV `0.163`, preemptions stable at `91`. Checkpoint waits post-100 are
  `170.32,19.06,72.14,105.24s`; the current limiter remains trainer/checkpoint
  pacing, now at a worse trainer MFU regime.
- 00:44 UTC MFU recovery check: trainer still has not recovered. Trainer steps
  100-108 average `181.66s`, `7574 tok/s`, `26.90%` MFU; steps 106-108 average
  `170.61s`, `7507 tok/s`, `26.67%` MFU. Orchestrator reached step 109; steps
  106-109 average `171.17s`, reward `0.4241`, seq length `5692.4`, with recent
  async/off-policy back at `1/1`. Checkpoint waits remain high post-100:
  `170.32,19.06,72.14,105.24,182.22,139.23,180.29,132.27s` (mean
  `125.10s`). vLLM remains 14/14 OK with `252` running, `0` waiting, mean KV
  `0.370`, and preemptions stable at `91`; gpustat shows 16-GPU aggregate
  `99%` util and `~92.6%` memory. This is a persistent post-eval trainer-side
  throughput regression, not an inference starvation symptom.
- 00:56 UTC correction on MFU recovery: the post-eval trainer slowdown was
  transient, not permanent. Trainer steps 110-112 recovered to `164.90s`,
  `8520 tok/s`, and `30.23%` MFU, with step 112 at `8619 tok/s` / `30.6%`.
  Orchestrator reached step 114; steps 110-114 average `166.48s`, reward
  `0.4335`, seq length `5721.0`, with recent async/off-policy mostly `1/1`
  except step 113 at off-policy `2`. Post-100 checkpoint waits are still high
  (`110.79s` mean, max `182.22s`), so the lasting performance issue is
  checkpoint/backpressure, while the MFU dip after eval lasted roughly 10
  trainer steps. vLLM remains 14/14 OK with `0` waiting and preemptions stable
  at `91`.
- 00:58 UTC resume artifact check: `checkpoints/step_50/trainer` and
  `checkpoints/step_100/trainer` both exist and contain `.metadata`,
  `__0_0.distcp`, and `__1_0.distcp`. The resume launcher
  `tmp/resume_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh` uses
  `--ckpt.resume_step -1`, derives four hosts from `SLURM_JOB_NODELIST`, and
  passes `--deployment.hosts` as JSON. Current live scrape: trainer reached
  step 113; steps 110-113 average `166.52s`, `8550 tok/s`, `30.35%` MFU.
  Orchestrator reached step 115; steps 110-115 average `163.81s`, reward
  `0.4394`, seq length `5816.8`. vLLM remains 14/14 OK with `118` running,
  `0` waiting, mean KV `0.247`, preemptions `91`.
- 01:14 UTC step-120 band: trainer reached step 119; steps 110-119 average
  `156.24s`, `8598 tok/s`, `30.52%` MFU, and steps 115-119 average `143.53s`,
  `8624 tok/s`, `30.62%` MFU. Orchestrator reached step 121; steps 115-121
  average `162.48s`, reward `0.4185`, seq length `5026.6`, latest async/off
  `1/2`. Checkpoint waits remain the main drag: post-110 waits are
  `66.10,53.07,82.11,127.40,143.23,174.38,167.27,159.38,85.14,76.19,45.04s`
  (mean `107.21s`). vLLM remains 14/14 OK with `186` running, `0` waiting,
  mean KV `0.376`, preemptions stable at `91`. Eval-150 has not started.
- 01:35 UTC step-128 band: trainer reached step 127; steps 120-127 average
  `170.04s`, `8660 tok/s`, `30.75%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 128; steps 120-128 average `160.22s`, reward `0.4066`, seq
  length `5621.2`, async/off-policy mostly `1/1` with steps 121/122/125 at
  off-policy `2`. Checkpoint waits post-120 are
  `45.04,97.10,134.23,123.20,139.21,167.41,177.38,141.38,136.17s`
  (mean `129.01s`). vLLM remains 14/14 OK with `250` running, `0` waiting,
  mean KV `0.244`, preemptions stable at `91`. Eval-150 has not started.
- 02:02 UTC step-140 approach: trainer reached step 137; steps 130-137 average
  `160.88s`, `8633 tok/s`, `30.66%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 139; steps 130-139 average `156.13s`, reward `0.4111`, seq
  length `5375.2`; async/off-policy remains mostly `1/1` with occasional
  off-policy `2`. Checkpoint waits are still bad: post-130 waits are
  `94.17,110.21,149.38,168.25,162.28,92.17,119.14,139.28,128.24s` (mean
  `129.24s`). vLLM remains 14/14 OK with `207` running, `0` waiting, mean KV
  `0.232`, preemptions stable at `91`. Eval-150 has not started.
- 02:28 UTC pre-eval-150: trainer reached step 147; steps 140-147 average
  `158.14s`, `8632 tok/s`, `30.65%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 148; steps 140-148 average `163.53s`, reward `0.4160`, seq
  length `5423.8`, with steps 143/144/145 at off-policy `2` and step 146
  off-policy `0`. Checkpoint waits remain high: post-140 waits are
  `94.10,59.06,83.14,143.21,165.27,189.24,127.22,126.32,126.18s` (mean
  `123.75s`). vLLM remains 14/14 OK with `252` running, `0` waiting, mean KV
  `0.147`, preemptions stable at `91`. Checkpoint/eval 150 has not started and
  no `step_150` checkpoint files exist yet.
- 02:45 UTC eval-150 completed: orchestrator completed step 150 at `02:32:46`
  (`154.63s`, reward `0.4180`, seq length `5900.6`, async/off-policy `1/1`),
  waited for checkpoint 150 for `138.23s`, started eval at `02:35:06`, and
  completed eval at `02:41:52` in `405.22s`. Eval-150 metrics:
  `Avg@8=0.3850`, `Pass@1=0.3850`, `Pass@2=0.4889`, `Pass@4=0.5781`,
  `Pass@8=0.6400`, no-response `0.0%`, completion length
  `5716.65 ± 5249.57`, truncated `15.2%`. Versus eval-100 this rebounds:
  Avg@8/Pass@1 `+2.87 pp`, Pass@2 `+3.68 pp`, Pass@4 `+4.25 pp`, Pass@8
  `+4.00 pp`, truncation `-1.7 pp`. Versus eval-50 it is mixed:
  Avg@8/Pass@1 `+2.12 pp`, Pass@2 `+0.71 pp`, Pass@4 `-0.83 pp`, Pass@8
  `-3.00 pp`, truncation `+0.2 pp`. Post-eval bubble recurred: orchestrator
  step 151 took `549.5s`, reward `0.346`, async/off-policy `0/3`; trainer step
  151 dropped to `7562 tok/s` / `26.8%` MFU. vLLM remained 14/14 OK with
  `0` waiting, but preemptions rose from `91` to `96`. `step_150/trainer`
  checkpoint files now exist (`.metadata`, `__0_0.distcp`, `__1_0.distcp`).
- 03:02 UTC post-eval-150 recovery check: MFU has not recovered yet by trainer
  step 157. Trainer steps 150-157 average `174.64s`, `7677 tok/s`, `27.24%`
  MFU, and steps 156-157 average `187.50s`, `7601 tok/s`, `27.00%` MFU. This
  is a longer-lived post-eval slowdown than eval-100 showed at the same
  relative age. Orchestrator reached step 158; steps 150-158 average `200.80s`,
  reward `0.4107`, seq length `5912.1`, with step 151 at `549.5s` and
  off-policy `3`, and step 157 also at off-policy `3`. Checkpoint waits
  post-150 are `138.23,37.11,47.05,82.11,120.26,175.29,196.34,129.29s`
  (mean `115.71s`). vLLM remains 14/14 OK, `253` running, `0` waiting, mean KV
  `0.180`, preemptions stable at `96`.
- 03:18 UTC post-eval-150 recovery confirmed: trainer MFU recovered at step
  160. Steps 160-162 average `172.39s`, `8653 tok/s`, `30.73%` MFU, after
  steps 151-159 sat around `26.7-27.3%` MFU. So eval-150 produced the same
  rough pattern as eval-100: one large boundary/eval wall-clock step plus a
  ~9-10 trainer-step MFU cooldown, then recovery. Orchestrator reached step
  164; steps 160-164 average `171.45s`, reward `0.4142`, seq length `5766.0`,
  with steps 160/161 at off-policy `2`. Post-150 checkpoint waits remain
  severe: mean `130.23s`, tail `147.32,151.24,164.28,154.25,150.19s`. vLLM
  remains 14/14 OK with `0` waiting and preemptions stable at `96`.
- 03:50 UTC step-175 approach: trainer reached step 173; steps 170-173 average
  `177.61s`, `8648 tok/s`, `30.70%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 175; steps 170-175 average `178.05s`, reward `0.3900`, seq
  length `5900.8`, with step 173 at off-policy `2` and step 174 off-policy
  `0`. Checkpoint waits remain severe: post-170 waits are
  `124.23,135.23,172.22,177.26,140.30s` (mean `149.85s`). vLLM remains 14/14
  OK with `54` running, `0` waiting, mean KV `0.141`, preemptions stable at
  `96`. Eval-200 has not started.
- 04:21 UTC step-185 band: trainer reached step 184; steps 180-184 average
  `156.60s`, `8579 tok/s`, `30.46%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 186; steps 180-186 average `161.70s`, reward `0.4452`, seq
  length `5538.8`, all recent async/off-policy `1/1`. Post-180 checkpoint
  waits remain high: `132.23,134.26,148.24,107.24,109.23,170.28s` (mean
  `133.58s`). vLLM remains 14/14 OK with `13` running, `0` waiting, mean KV
  `0.043`, preemptions stable at `96`. Eval-200 has not started.
- 04:53 UTC pre-eval-200: trainer reached step 196; steps 190-196 average
  `168.61s`, `8609 tok/s`, `30.59%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 198; steps 190-198 average `163.74s`, reward `0.4257`, seq
  length `5911.7`, mostly async/off-policy `1/1` with step 197 at off-policy
  `2`. Checkpoint waits post-190 are
  `133.26,140.29,145.24,145.24,136.23,129.15,145.34,165.29s` (mean
  `142.50s`). vLLM remains 14/14 OK with `212` running, `0` waiting, mean KV
  `0.283`, preemptions stable at `96`. Eval-200 has not started and no
  `step_200` checkpoint files exist yet.
- 05:14 UTC eval-200 completed: orchestrator completed step 200 at `04:59:54`
  (`205.10s`, reward `0.3789`, seq length `6258.6`, async/off-policy `1/1`),
  waited for checkpoint 200 for `171.30s`, started eval at `05:02:48`, and
  completed eval at `05:08:43` in `355.34s`. Eval-200 metrics:
  `Avg@8=0.3613`, `Pass@1=0.3613`, `Pass@2=0.4686`, `Pass@4=0.5607`,
  `Pass@8=0.6200`, no-response `0.0%`, completion length
  `5675.15 ± 5212.35`, truncated `14.5%`. Versus eval-150, all pass metrics
  fell: Avg@8/Pass@1 `-2.37 pp`, Pass@2 `-2.03 pp`, Pass@4 `-1.74 pp`,
  Pass@8 `-2.00 pp`, while truncation improved `-0.7 pp`. Versus eval-50:
  Avg@8/Pass@1 `-0.25 pp`, Pass@2 `-1.32 pp`, Pass@4 `-2.57 pp`, Pass@8
  `-5.00 pp`, truncation `-0.5 pp`. Post-eval bubble recurred: orchestrator
  step 201 took `561.3s`; trainer steps 201-202 are `27.4%` and `27.3%` MFU.
  vLLM remained 14/14 OK with `0` waiting and preemptions stable at `96`.
  `step_200/trainer` checkpoint files now exist (`.metadata`, `__0_0.distcp`,
  `__1_0.distcp`), so latest real resume point is 200.
- 05:31 UTC post-eval-200 recovery check: trainer MFU has not recovered by
  step 208. Trainer steps 200-208 average `164.76s`, `7737 tok/s`, `27.48%`
  MFU; steps 201-208 are stuck around `26.8-27.4%` MFU. Orchestrator reached
  step 210; steps 200-210 average `190.72s`, reward `0.4170`, seq length
  `5386.9`; step 201 was the boundary/eval bubble (`561.3s`), and steps
  204-206 hit off-policy `2`. Post-200 checkpoint waits average `119.29s`,
  with tail `171.30,41.06,45.06,154.16,141.28,191.23,138.16,118.23,73.10s`.
  vLLM remains 14/14 OK, `253` running, `0` waiting, mean KV `0.363`,
  preemptions stable at `96`. This is a longer post-eval trainer cooldown than
  the eval-100/eval-150 recoveries.

- 05:40 UTC utilization recheck: trainer MFU recovered at steps 210-211. Steps
  210-211 average `168.06s`, `8550 tok/s`, and `30.35%` MFU, with peak memory
  still `92.9 GiB`. Orchestrator reached step 213; steps 210-213 average
  `151.53s`, reward `0.4154`, seq length `5893.7`, with off-policy levels
  `1,2,2,1`. Instantaneous `gpustat` briefly looked low at 05:38 UTC (`78%`
  aggregate, three idle-looking GPUs), but the 05:40 UTC snapshot showed all
  16 GPUs at `100%`, using `1451921/1565936 MiB` total memory (~`92.7%`) and
  `8311W`. Treat instantaneous `gpustat` as noisy; trainer MFU is the better
  run-level signal. PrimeRL `bench = true` remains off, so these are live
  training MFU/throughput numbers, not fake-data benchmark numbers.
- 05:55 UTC post-210 stability check: trainer reached step 216. Steps 210-216
  average `169.37s`, `8584 tok/s`, and `30.47%` MFU; steps 215-216 average
  `162.13s`, `8612 tok/s`, and `30.60%` MFU. Peak memory remains `92.9 GiB`.
  Orchestrator reached step 218; steps 210-218 average `162.10s`, reward
  `0.4090`, seq length `5823.5`, with recent off-policy mostly `1` after the
  step-211/212 `2`s. Checkpoint waits are still material but softened after
  step 215: post-210 mean `150.53s`, post-215 mean `129.57s`, tail
  `210:129.17,211:144.26,212:184.36,213:189.39,214:168.34,215:104.16,216:135.31,217:149.23`.
  vLLM remains 14/14 OK with `0` waiting, mean KV `0.173`, and preemptions
  stable at `96`. A 05:55 `gpustat` snapshot showed only `70%` instantaneous
  aggregate GPU util despite healthy trainer MFU, reinforcing that single
  gpustat frames are too noisy for steering this run. Eval-250 has not started;
  latest real checkpoint remains step 200.
- 06:27 UTC pre-eval-250 status: trainer reached step 228, orchestrator step
  230. Trainer steps 215-228 average `164.16s`, `8639 tok/s`, and `30.68%`
  MFU; steps 220-228 average `161.12s`, `8647 tok/s`, and `30.70%` MFU. Peak
  memory remains stable at `92.9 GiB`. Orchestrator steps 215-230 average
  `166.44s`, reward `0.4118`, seq length `5534.1`; steps 220-230 average
  `165.66s`, reward `0.4130`, seq length `5449.9`. Recent async/off-policy is
  mostly `1/1`, with steps 226 and 230 at off-policy `2`. Checkpoint waits
  remain the visible drag: post-215 mean `138.49s`, post-220 mean `131.60s`,
  tail
  `215:104.16,216:135.31,217:149.23,218:177.35,219:195.33,220:142.19,221:91.09,222:92.18,223:108.13,224:127.13,225:207.23,226:159.31,227:138.24,228:122.27,229:128.23`.
  vLLM remains 14/14 OK with `156` running, `0` waiting, mean KV `0.167`,
  preemptions stable at `96`. `gpustat` at 06:27 showed all 16 GPUs at `100%`,
  using `1449145/1565936 MiB` total memory. Eval-250 has not started; latest
  real checkpoint remains step 200.
- 06:59 UTC pre-eval-250 approach: trainer reached step 239, orchestrator step
  241. Trainer steps 230-239 average `160.57s`, `8694 tok/s`, and `30.87%`
  MFU, with peak memory still `92.9 GiB`. Orchestrator steps 230-241 average
  `161.02s`, reward `0.4165`, seq length `5471.7`; recent off-policy includes
  steps 230, 234, and 241 at `2`, with step 236 at `0`. Checkpoint waits
  remain high but steady: post-230 mean `139.86s`, tail
  `230:168.27,231:157.20,232:92.15,233:129.27,234:128.28,235:165.28,236:151.23,237:117.13,238:134.22,239:146.23,240:149.24`.
  vLLM remains 14/14 OK with `44` running, `0` waiting, mean KV `0.123`,
  preemptions stable at `96`. `gpustat` again caught a low instantaneous
  aggregate frame (`63%`) despite stable trainer MFU, so keep treating gpustat
  snapshots as noisy. Eval-250 has not started; latest real checkpoint remains
  step 200.
- 07:33 UTC eval-250 completed: orchestrator saved checkpoint 250 at
  `07:19:33`, completed step 250 at `07:22:00` (`146.17s`, reward `0.4315`,
  seq length `5920.8`, async/off-policy `1/1`), waited `163.24s` for
  checkpoint 250, then ran eval from `07:24:46` to `07:30:55` (`369.37s`).
  Eval-250 metrics: `Avg@8=0.3837`, `Pass@1=0.3837`, `Pass@2=0.5004`,
  `Pass@4=0.6111`, `Pass@8=0.7000`, no-response `0.0%`, completion length
  `5599.56 ± 5102.68`, truncated `13.2%`. Versus eval-200, Pass@1/2/4/8 moved
  `+2.24/+3.18/+5.04/+8.00 pp`; versus eval-50, `+1.99/+1.86/+2.47/+3.00 pp`.
  This is the first row with a clear best-so-far Pass@2/4/8 and best-so-far
  truncation, though Pass@1 remains just below eval-150 (`0.3837` vs `0.3850`).
  Boundary bubble recurred: step 251 completed immediately after eval at
  `07:30:59` with `538.58s`, reward `0.4023`, seq length `6684.9`,
  async/off-policy `0/2`. Trainer step 250 completed at `07:28:02` with
  `178.35s`, `8501 tok/s`, `30.2%` MFU, peak memory `92.9 GiB`. vLLM remained
  14/14 OK with `250` running, `0` waiting, mean KV `0.403`, preemptions stable
  at `96`. `checkpoints/step_250/trainer` contains `.metadata`,
  `__0_0.distcp`, and `__1_0.distcp`, so latest real resume point is now step
  250.
- 07:51 UTC post-eval-250 cooldown: trainer reached step 257. Trainer steps
  251-257 average `167.05s`, `7590 tok/s`, and `26.94%` MFU; including step
  250, steps 250-257 average `168.46s`, `7704 tok/s`, and `27.35%` MFU. This
  repeats the eval-boundary cooldown pattern from prior evals, despite eval-250
  itself being positive. Orchestrator reached step 259. Steps 251-259 average
  `194.09s`, reward `0.4249`, seq length `5587.7`; step 251 was the boundary
  bubble (`538.58s`), and off-policy levels rose to `3` at step 257 and `4` at
  step 258 before returning to `1` at step 259. Checkpoint waits post-250
  average `130.22s`, tail
  `250:163.24,252:64.10,253:39.04,254:78.14,255:164.36,256:185.35,257:179.24,258:168.25`.
  vLLM remains 14/14 OK with `255` running, `0` waiting, mean KV `0.234`,
  preemptions stable at `96`. Peak trainer memory remains `92.9 GiB`.
- 08:03 UTC post-eval-250 recovery confirmed: trainer recovered at step 260.
  Trainer steps 251-259 stayed in the cooldown band (`~26.7-27.4%` MFU), then
  steps 260-261 averaged `162.02s`, `8585 tok/s`, and `30.45%` MFU. Including
  the cooldown, trainer steps 251-261 average `166.12s`, `7790 tok/s`, and
  `27.65%` MFU. Orchestrator reached step 263. Steps 260-263 average
  `161.43s`, reward `0.4075`, seq length `5367.6`; off-policy returned to
  `1` for steps 260-262, then `2` at step 263. vLLM remains 14/14 OK with
  `130` running, `0` waiting, mean KV `0.280`, preemptions stable at `96`.
  This locks in the recurring eval-boundary shape: one large post-eval
  orchestrator step plus about 9 low-MFU trainer steps, then recovery.
- 08:36 UTC stable mid-band toward eval-300: trainer reached step 273,
  orchestrator step 275. Trainer steps 260-273 average `162.36s`, `8585 tok/s`,
  and `30.48%` MFU; steps 270-273 average `158.99s`, `8617 tok/s`, and
  `30.60%` MFU. Peak memory remains `92.9 GiB`. Orchestrator steps 260-275
  average `166.73s`, reward `0.4181`, seq length `5479.7`; recent off-policy
  is mostly `1`, with step 272 at `2`. Checkpoint waits post-260 average
  `147.40s`, post-270 average `140.49s`. vLLM remains 14/14 OK with `252`
  running, `0` waiting, mean KV `0.337`, preemptions stable at `96`. Latest
  real checkpoint remains step 250.
- 09:07 UTC utilization/bench-mode check: trainer reached step 284; steps
  270-284 average `161.54s`, `8610 tok/s`, and `30.57%` MFU, with latest step
  `174.16s`, `8550 tok/s`, `30.4%` MFU. Peak trainer memory remains
  `92.9 GiB`. Orchestrator reached step 286; steps 270-286 average `166.14s`,
  reward `0.4130`, seq length `5441.3`. vLLM remains 14/14 OK with `37`
  running, `0` waiting, mean KV `0.100`, preemptions stable at `96`.
  `gpustat` snapshot showed `56%` instantaneous aggregate GPU compute
  utilization and `1413851/1565936 MiB` allocated memory; 9/16 GPUs were at
  `100%` and 7/16 at `0%`, so use this as a bursty instantaneous frame rather
  than a stable MFU substitute. Top-level PrimeRL `bench` is not enabled in the
  active TOML or launcher. Do not enable `bench=true` for this canary:
  PrimeRL's RL bench path switches the trainer to fake data, sets orchestrator
  benchmark mode, changes max-step/eval behavior, and would no longer be the
  real 1000-step RL comparison. Normal trainer logs already provide `perf/mfu`.
- 09:54 UTC eval-300 completed: orchestrator saved checkpoint 300 at
  `09:40:43`, completed step 300 at `09:43:28` (`164.10s`, reward `0.4133`,
  seq length `5574.8`, async/off-policy `1/1`), waited `148.21s` for trainer
  checkpoint 300, then ran eval from `09:45:59` to `09:51:48` (`349.04s`).
  Checkpoint files exist under `checkpoints/step_300/trainer` (`.metadata`,
  `__0_0.distcp`, `__1_0.distcp`), so latest real resume point is now step
  300. Eval-300 metrics: `Avg@8=0.3912`, `Pass@1=0.3912`, `Pass@2=0.5039`,
  `Pass@4=0.6076`, `Pass@8=0.7000`, no-response `0.0%`, completion length
  `5511.45 ± 5086.46`, truncated `13.2%`. Versus eval-250, Pass@1/2/4/8 moved
  `+0.75/+0.35/-0.35/+0.00 pp`; versus eval-50, `+2.74/+2.21/+2.12/+3.00 pp`.
  Pass@1 and Pass@2 are best-so-far; Pass@8 ties best-so-far; Pass@4 is just
  below eval-250. Trainer steps 290-299 averaged `163.34s`, `8580 tok/s`, and
  `30.46%` MFU, with peak memory still `92.9 GiB`. Trainer step 300 completed
  at `09:49:07` with `167.62s`, `8428 tok/s`, `29.9%` MFU. The eval-boundary
  bubble recurred: step 301 completed at `09:51:52` with `503.31s`, reward
  `0.4336`, seq length `6403.0`, async/off-policy `0/2`. vLLM was 14/14 OK at
  `09:53:59` with `256` running, `0` waiting, mean KV `0.422`; eval increased
  preemptions from `96` to `116`. `gpustat` at `09:54` caught all 16 GPUs at
  `100%` and `1451113/1565936 MiB` allocated.
- 10:21 UTC post-eval-300 recovery confirmed: trainer steps 301-309 were the
  cooldown band, averaging about `27.1%` MFU and `~7.6k tok/s`; step 309 was
  still low at `27.5%` MFU, then step 310 recovered to `142.65s`,
  `8574 tok/s`, and `30.4%` MFU. Including step 300, trainer steps 300-310
  averaged `164.83s`, `7793 tok/s`, and `27.66%` MFU, with peak memory still
  `92.9 GiB`. Orchestrator reached step 312; steps 300-312 average `182.21s`,
  reward `0.4199`, seq length `5473.2`, and off-policy returned to `1`.
  vLLM remained 14/14 OK with `65` running, `0` waiting, mean KV `0.171`, and
  preemptions stable at `116`. Eval-boundary cooldown estimate is now
  consistently ~9 low-MFU trainer steps, with recovery at the tenth post-eval
  trainer step.

**Verdict**: `keep running, memory-watch active`. The setup corrected the `768`
in-flight mistake without introducing an immediate warmup/OOM failure, and
overlap begins after step 0. The run is not in benchmark mode, so use these as
live end-to-end datapoints, not isolated `--bench` throughput claims. Pre-eval
100 trainer MFU warmed to `~30.9%`; the later stable band is `~30.5%` MFU, and
eval-300 is the strongest eval row so far on Pass@1/2. The recurring weakness
is still eval-boundary refill/cooldown, not memory pressure. Do not assume this
allocation can finish both 1000-step arms; resume planning matters.

**Next action**: continue Default toward eval-350 unless the run errors or
memory spikes. Eval-300 supports keeping the Default arm running; the resume
script remains important because the allocation probably will not complete both
1000-step arms.

## 2026-05-10 12:14 UTC

**Step**: trainer 350 / 1000; orchestrator 352 / 1000; eval-350 complete.
**Health**: Healthy, but eval quality regressed from the eval-300 high-water
mark and the eval-boundary bubble recurred.

**Eval-350**: The orchestrator saved checkpoint 350 at `11:58:19`, completed
step 350 at `12:01:48` (`207.35s`, reward `0.4133`, seq length `6472.6`,
async/off-policy `1/1`), waited `161.32s` for the trainer checkpoint, then ran
eval from `12:04:32` to `12:10:46` in `374.34s`. Metrics:
`Avg@8=0.3650`, `Pass@1=0.3650`, `Pass@2=0.4789`, `Pass@4=0.5821`,
`Pass@8=0.6800`, no-response `0.0%`, completion length
`5850.65 ± 5218.24`, truncated `15.4%`.

**Comparison**: Versus eval-300, Pass@1/2/4/8 moved
`-2.62/-2.50/-2.55/-2.00 pp`, and truncation worsened by `+2.2 pp`. Versus
eval-50, Pass@1/2/4/8 moved only `+0.12/-0.29/-0.43/+1.00 pp`, with truncation
`+0.4 pp`. This weakens the "monotone learning by step 350" story; eval-250
and eval-300 still look like a local uplift, but eval-350 falls back toward the
initial eval row.

**Trainer**: Step 350 completed at `12:07:59` with `189.57s`, `8453 tok/s`,
`30.0%` MFU, peak memory `92.9 GiB`. Steps 340-350 averaged `164.04s`,
`8533 tok/s`, and `30.27%` MFU. No OOM/traceback/error matches.

**Boundary**: Step 351 completed immediately after eval at `12:10:50` with a
`541.61s` wall-clock time, reward `0.4531`, seq length `6494.6`,
async/off-policy `0/2`. Step 352 completed normally at `12:13:16` in
`145.33s`, async/off-policy `1/0`. This repeats the now-standard eval-boundary
pattern: checkpoint wait + eval wall time + one oversized post-eval
orchestrator step, followed by trainer cooldown that still needs monitoring.

**vLLM/GPU**: vLLM remained 14/14 OK with `255` running, `0` waiting, mean KV
`0.415`, and preemptions stable at `116`. `gpustat` at `12:13` showed `94%`
instantaneous aggregate GPU util and `1448598/1565936 MiB` allocated; one
trainer-node GPU was at low SM util in that frame, so keep treating gpustat as
phase-sensitive and use trainer MFU for useful-FLOP claims.

**Resume**: `checkpoints/step_350/trainer` contains `.metadata`,
`__0_0.distcp`, and `__1_0.distcp`; latest real checkpoint is step 350.

## 2026-05-10 12:42 UTC

**Step**: trainer 361 / 1000; orchestrator 363 / 1000.
**Health**: Healthy; post-eval-350 cooldown has recovered.

**Trainer**: Steps 351-359 were the cooldown band, rising only from `27.2%` to
`27.6%` MFU and `7649` to `7784 tok/s`. Steps 360-361 recovered to
`30.6%` MFU and `8628 tok/s` average. Including step 350, steps 350-361 average
`163.51s`, `7896 tok/s`, and `28.03%` MFU. Peak memory remains `92.9 GiB`.

**Orchestrator**: Steps 350-363 average `183.49s`, reward `0.4187`, and seq
length `5525.2`. The post-eval off-policy spike peaked at `5` on step 357 and
returned to `1` by steps 358-359; step 360 had off-policy `0`.

**vLLM/GPU**: vLLM remains 14/14 OK with `196` running, `0` waiting, mean KV
`0.352`, and preemptions stable at `116`. `gpustat` at `12:41` showed `93%`
instantaneous aggregate GPU util and `1416921/1565936 MiB` allocated; one
trainer-node GPU was phase-idle in that snapshot. No error/OOM matches.

**Interpretation**: Eval-350 exactly repeats the earlier throughput shape:
one large post-eval orchestrator step plus about 9 low-MFU trainer steps, then
recovery at trainer step 360. This is now a stable tax of eval/checkpoint
boundaries under the current pipeline settings, not a vLLM queue-capacity
problem.

## 2026-05-10 13:14 UTC

**Step**: trainer 372 / 1000; orchestrator 374 / 1000.
**Health**: Healthy; recovered steady-state band is stable toward eval-400.

**Trainer**: Steps 360-372 average `164.50s`, `8563 tok/s`, and `30.40%`
MFU. Steps 370-372 average `179.59s`, `8528 tok/s`, and `30.27%` MFU. Peak
memory remains `92.9 GiB`.

**Orchestrator**: Steps 360-374 average `169.27s`, reward `0.4308`, seq length
`5487.0`, with max off-policy `2`; steps 370-374 average `172.66s`, reward
`0.4162`, seq length `5807.8`, max off-policy `2`.

**Backpressure**: Checkpoint waits since 360 average `148.47s`, max `184.33s`;
since 370 they average `162.53s`. This is still the visible pacing tax even
when trainer MFU has recovered.

**vLLM**: 14/14 OK with `144` running, `0` waiting, mean KV `0.301`,
preemptions stable at `116`. No eval-400 yet and no error/OOM matches.

**Allocation**: Slurm job `4507057` is at `19:46:03 / 24:00:00`, ending
`2026-05-10T17:27:21`. At current cadence, this allocation should reach
eval-400 and maybe eval-450, but not the full 1000-step Default arm.

## 2026-05-10 13:55 UTC

**Step**: trainer 387 / 1000; orchestrator 389 / 1000.
**Health**: Healthy; eval-400 still pending.

**Trainer**: Steps 370-387 average `166.03s`, `8571 tok/s`, and `30.42%`
MFU. Steps 380-387 average `161.58s`, `8612 tok/s`, and `30.56%` MFU. Peak
memory remains `92.9 GiB`.

**Orchestrator**: Steps 370-389 average `166.19s`, reward `0.4159`, seq length
`5614.6`, max off-policy `2`. Steps 380-389 average `164.36s`, reward
`0.4088`, seq length `5548.9`, max off-policy `2`.

**Backpressure/vLLM**: Checkpoint waits since 370 average `151.37s`, max
`196.38s`. vLLM is 14/14 OK with `125` running, `0` waiting, mean KV `0.281`,
preemptions stable at `116`.

**GPU snapshot**: `gpustat` showed `83%` instantaneous aggregate GPU util and
`1452241/1565936 MiB` allocated. Several GPUs were phase-low in this frame, so
the stable trainer MFU remains the better utilization signal.

## 2026-05-10 14:36 UTC

**Step**: trainer 400 / 1000; orchestrator 402 / 1000; eval-400 complete.
**Health**: Healthy runtime, but eval quality worsened again.

**Eval-400**: The orchestrator saved checkpoint 400 at `14:21:53`, completed
step 400 at `14:24:51` (`176.21s`, reward `0.4009`, seq length `5513.0`,
async/off-policy `1/1`), waited `153.26s` for the trainer checkpoint, then ran
eval from `14:27:27` to `14:33:20` in `353.47s`. Metrics:
`Avg@8=0.3525`, `Pass@1=0.3525`, `Pass@2=0.4571`, `Pass@4=0.5621`,
`Pass@8=0.6600`, no-response `0.0%`, completion length
`5479.82 ± 5109.73`, truncated `13.4%`.

**Comparison**: Versus eval-350, Pass@1/2/4/8 moved
`-1.25/-2.18/-2.00/-2.00 pp` while truncation improved by `-2.0 pp`. Versus
eval-300, Pass@1/2/4/8 moved `-3.87/-4.68/-4.55/-4.00 pp`. Versus eval-50,
Pass@1/2/4/8 moved `-1.13/-2.47/-2.43/-1.00 pp`. This is no longer just
"no monotone learning"; by ckpt-400, eval is below the initial eval-50 row on
the main pass@k metrics.

**Trainer**: Steps 380-400 average `164.07s`, `8558 tok/s`, and `30.38%`
MFU. Steps 390-400 average `164.20s`, `8516 tok/s`, and `30.23%` MFU. Step 400
completed at `14:30:32` with `167.45s`, `8364 tok/s`, `29.7%` MFU, peak memory
`92.8 GiB`.

**Boundary**: Step 401 completed immediately after eval at `14:33:24` with
`512.97s`, reward `0.4267`, seq length `5850.7`, async/off-policy `0/3`.
Step 402 completed at `14:35:43` in `137.85s`, async/off-policy `1/0`. The
post-eval boundary tax recurred again.

**vLLM/Resume**: vLLM was 14/14 OK with `206` running, `0` waiting, mean KV
`0.428`, preemptions stable at `116`. `checkpoints/step_400/trainer` contains
`.metadata`, `__0_0.distcp`, and `__1_0.distcp`; latest real checkpoint is
step 400.

**Interpretation**: Throughput remains stable, but quality is drifting the
wrong way after the eval-250/300 local high. This strengthens the hypothesis
that the run is not simply undertrained; something about policy update/noise/
data mix/objective may be hurting the eval distribution, despite healthy
infrastructure metrics.

## 2026-05-10 15:08 UTC

**Step**: trainer 412 / 1000; orchestrator 414 / 1000.
**Health**: Healthy; post-eval-400 cooldown has recovered.

**Trainer**: Steps 401-409 were the low-MFU cooldown band (`26.7-27.6%` MFU,
`7517-7765 tok/s`). Steps 410-412 recovered to `30.67%` MFU and `8638 tok/s`
average. Including step 400, steps 400-412 average `163.22s`, `7903 tok/s`,
and `28.05%` MFU. Peak memory remains `92.9 GiB`.

**Orchestrator**: Steps 400-414 average `179.26s`, reward `0.4064`, seq length
`5545.1`, max off-policy `3`. Steps 410-414 average `152.00s`, reward
`0.3966`, seq length `5265.7`, max off-policy `2`.

**vLLM**: 14/14 OK with `78` running, `0` waiting, mean KV `0.194`,
preemptions stable at `116`. No error/OOM matches.

**Interpretation**: Eval-400 repeats the exact boundary pattern again:
post-eval step bubble plus about 9 low-MFU trainer steps, then recovery at
step 410. The infrastructure is behaving consistently; the alarming part is
quality degradation, not launch instability.

## 2026-05-10 15:50 UTC

**Step**: trainer 428 / 1000; orchestrator 430 / 1000.
**Health**: Healthy; recovered steady-state band is stable toward eval-450.

**Trainer**: Steps 410-428 average `154.11s`, `8612 tok/s`, and `30.57%`
MFU. Steps 420-428 average `151.28s`, `8611 tok/s`, and `30.58%` MFU. Peak
memory remains `92.9 GiB`.

**Orchestrator**: Steps 410-430 average `157.63s`, reward `0.4124`, seq length
`5119.1`, max off-policy `2`; steps 420-430 average `155.98s`, reward
`0.4206`, seq length `4970.2`, max off-policy `1`.

**Backpressure/vLLM**: Checkpoint waits since 410 average `141.80s`, max
`170.35s`. vLLM is 14/14 OK with `160` running, `0` waiting, mean KV `0.332`,
preemptions stable at `116`.

**Allocation**: Slurm job is at `22:22:39 / 24:00:00`, ending
`2026-05-10T17:27:21`. Eval-450 still looks feasible before walltime if the
current cadence holds; eval-500 does not.

## 2026-05-10 16:57 UTC

**Step**: trainer 452 / 1000; orchestrator 454 / 1000; eval-450 complete.
**Health**: Healthy runtime; eval bounced back from ckpt-400 but did not regain
the ckpt-300 high-water mark.

**Eval-450**: The orchestrator saved checkpoint 450 at `16:39:28`, completed
step 450 at `16:41:31` (`122.26s`, reward `0.4569`, seq length `5382.6`,
async/off-policy `1/1`), waited `132.17s` for the trainer checkpoint, then ran
eval from `16:43:46` to `16:49:57` in `370.41s`. Metrics:
`Avg@8=0.3837`, `Pass@1=0.3837`, `Pass@2=0.4904`, `Pass@4=0.5844`,
`Pass@8=0.6700`, no-response `0.0%`, completion length
`5592.09 ± 5159.52`, truncated `14.0%`.

**Comparison**: Versus eval-400, Pass@1/2/4/8 moved
`+3.12/+3.33/+2.23/+1.00 pp`, but versus eval-300 it remains
`-0.75/-1.35/-2.32/-3.00 pp`. Versus eval-50, Pass@1/2/4/8 moved
`+1.99/+0.86/-0.20/+0.00 pp`. So ckpt-450 recovers from the ckpt-400 dip, but
does not support a clean scaling/learning claim.

**Trainer**: Steps 430-448 averaged `154.16s`, `8590 tok/s`, and `30.49%`
MFU; steps 440-448 averaged `151.28s`, `8581 tok/s`, and `30.46%` MFU. Trainer
step 450 completed at `16:46:50` with `165.54s`, `8412 tok/s`, `29.9%` MFU,
peak memory `92.8 GiB`.

**Boundary**: Step 451 completed immediately after eval at `16:50:01` with
`508.94s`, reward `0.3875`, seq length `5867.5`, async/off-policy `0/3`.
Steps 452-454 then resumed normally. The eval-boundary tax repeated again.

**vLLM/GPU**: vLLM was 14/14 OK with `24` running, `0` waiting, mean KV
`0.063`, preemptions stable at `116`. `gpustat` at `16:57` showed only `56%`
instantaneous aggregate GPU util because the run was in a phase-local lull;
this is not an error.

**Resume/allocation**: `checkpoints/step_450/trainer` contains `.metadata`,
`__0_0.distcp`, and `__1_0.distcp`; latest real checkpoint is step 450. Slurm
job `4507057` has about 30 minutes left, so the useful outcome from this
allocation is a resumable Default arm through eval-450, not completion of the
1000-step Default arm or any MaxRL steps.

## 2026-05-11 11:18 UTC

**Previous allocation final scrape**: Slurm job `4507057` ended after reaching
trainer step 464 and orchestrator step 466, but no checkpoint beyond step 450
was written. The last trainer row is step 464 at `17:26:40` (`175.33s`,
`8533 tok/s`, `30.3%` MFU, peak memory `92.8 GiB`). The last orchestrator row
is step 466 at `17:26:43` (`166.21s`, reward `0.3685`, seq length `5469.0`,
async/off-policy `1/1`). Treat `checkpoints/step_450/trainer` as the only safe
resume point.

**Current allocation**: New 4-node job `4542540` is running on
`nid011162,nid011164,nid011183,nid011184`, from `2026-05-11T11:14:24` to
`2026-05-12T11:14:24`.

## 2026-05-11 11:50 UTC

**Eval-side intervention**: Updated the active sequential 1000-step tmp configs
for both Default and MaxRL:

- Keep `omni-math2-baseline100` as the online heartbeat: 100 examples × 8
  rollouts, interval 50.
- Add `omni-math2-full600-p1`: 600 examples × 1 rollout, interval 250. This is
  the broad p@1 decision sentinel without adding a 4,800-rollout online bubble.
- Raise eval-only `math_verify_timeout_seconds` from 5s to 10s for both eval
  envs. Training timeout remains 5s to avoid slowing the rollout pipeline.
- Keep `cancel_inflight_rollouts_on_eval = false`, `max_concurrent_rollouts_per_client = 32`,
  and `num_workers = 96`.

**Why timeouts still matter**: The Omni-MATH rubric runs
`math_verify -> aliases -> judge -> correct_answer`. The LLM judge fallback
runs after symbolic verification fails/times out, not instead of it. Therefore
`Timeout during comparison` still burns scorer latency and can still become a
zero if parsed-answer extraction or judge fallback fails/returns negative.

**Code/tools added**:

- `src/prime_rl/orchestrator/envs.py` now annotates eval rollouts with
  `env_name` and logs eval scorer/source diagnostics: generation/scoring ms,
  stop-condition rates, and math-verify/alias/judge metric means.
- `scripts/evals/analyze_online_eval_rollouts.py` summarizes persisted online
  eval rollouts and computes paired bootstrap CIs by example id.
- `scripts/evals/make_omni_math2_perfectible_subset.py` creates an upstream-style
  "perfectible" Omni-MATH subset from baseline rollouts, e.g. solve rate in
  `[0.2, 0.8]`.
- `skills/config/SKILL.md` documents the eval recipe and timeout/judge-fallback
  caveat.

**Validation**:

- `uv run --no-sync python -m scripts.evals.analyze_online_eval_rollouts ...`
  succeeded on the existing Default eval artifacts and wrote
  `/tmp/prime-rl-online-eval-summary.json`.
- `uv run --no-sync python -m scripts.evals.make_omni_math2_perfectible_subset ...`
  succeeded on a probe using step-51 eval rollouts; it selected 37/100 probe
  examples. Do not treat that probe as the real base-model sanity subset.
- Dry runs succeeded for active tmp Default/MaxRL configs and the two repo-local
  `*_pipelinerl_speed.toml` configs; generated env logs include both
  `omni-math2-baseline100` and `omni-math2-full600-p1`.

**Current eval diagnosis from the new analyzer**: Existing eval rows are under
`env_name = unknown` because older artifacts did not save env names. Bootstrap
against ckpt-50 still says the checkpoint deltas are not decisive: ckpt-450
vs ckpt-50 p@8 mean is about flat, with CI roughly `[-0.06, +0.07]` in the
200-sample validation run. Future artifacts will be separated by eval env.

## 2026-05-11 11:59 UTC

**Default resume launch**: Started `tmp/resume_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh`
inside tmux window `joanv_cc_4node:olmo3-resume` on Slurm job `4542540`.
Resume picked `run_default/checkpoints/step_450/trainer`, deleted stale future
rollout/broadcast dirs, initialized both eval envs, skipped eval-450 as
intended, completed NCCL weight update, and entered orchestrator step 450.

**Live health**: No `*FAILED` marker. `logs/inference.log` shows successful
`/init_broadcaster`, `/pause`, `/update_weights`, `/resume`, and live
`/v1/chat/completions` traffic. At last check, train rollout generation was
advancing through step 450 (`40/256` rollouts). Do not count a post-resume
trainer step until a new `SUCCESS Step ...` line appears after the `11:56`
training-loop restart.

**Launcher noise**: The tmux pane printed one early `STOP_INFERENCE observed on
nid011162` line, but the orchestrator subsequently reported inference ready,
resumed all engines, and inference logs show live generation. Treat it as
non-fatal unless the run later writes a failure marker or serving traffic stops.

## 2026-05-11 12:00 UTC

**First post-resume step**: Verified. Orchestrator step 450 completed at
`11:59:43` in `215.97s`, reward `0.3915`, seq length `5156.5`, async/off-policy
`0/0`, then started step 451. Trainer step 450 completed at `12:00:23` in
`258.61s`, peak memory `93.4 GiB`.

**Performance caveat**: Trainer step 450 reports `0 tok/s` and `0.0% MFU`; this
is a resume/warmup-accounting artifact, not a steady-state datapoint. Use step
451 onward for speed claims.

## 2026-05-11 13:13 UTC

**Stopped fragmented resume**: The live Default resume in tmux window
`joanv_cc_4node:olmo3-resume` was stopped after reaching trainer step 468 and
orchestrator step 470. No stable checkpoint beyond step 450 existed, so step
450 remains the latest safe model checkpoint.

**W&B reattach fix**: Future restarts of the Default arm now pin W&B identity to
the original run id, not just the run name:

- run name: `olmo3-dpo-default-14i2t-bs256-eval50-1000step-20260509-1825`
- original W&B id: `197d96382b0c40d59272ac0fbc94a9e3`
- accidental fragmented id observed on the interrupted resume:
  `66363fe840af4a9cb308118fddd02ea3`

Both `tmp/resume_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh` and
`tmp/run_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh` now export
`WANDB_RESUME=allow`, `WANDB_SHARED_RUN_ID`, and `WANDB_RUN_ID` for the Default
arm. MaxRL is left unset unless `MAXRL_WANDB_RUN_ID` is provided, because no
original MaxRL W&B id exists yet for this 20260509 sequential pair.

**Checkpoint preservation**: Since the already-running process did not inherit
the new `ckpt.keep_interval = 50` config, stable Default checkpoints were
hardlink-archived manually under
`outputs/omni_math2_rlvr_canary/20260509_1825/checkpoint_archive_50/default`:
weights and trainer checkpoints for steps 300, 350, 400, and 450.

**Offline eval recovery**: Added `scripts/evals/offline_omni_math2_ckpt_eval.py`
and launcher scripts in `tmp/` for checkpoint hot-swap eval. The first sharded
implementation used threads and failed because the verifiers env installs
signal handlers; it has been patched to use spawned subprocesses per backend.
The logged 600×8 Default offline eval is currently running in tmux window
`joanv_cc_4node:offline-eval`, log:
`outputs/omni_math2_rlvr_canary/20260509_1825/offline_eval_600x8/logs/launcher_20260511T130924Z.log`.

Current target: recover 600 examples × 8 rollouts for archived Default
checkpoints 300, 350, 400, and 450, using the four per-node vLLM backends
directly because `vllm-router` is not installed in this environment.

**Logging follow-up**: The active run's tqdm display is faithful in totals but
not legible: each of four shards prints its own 1,200-rollout progress bars to
the same pane/log. After this eval completes, update the offline evaluator so
future sharded evals write per-shard logs/artifacts plus a single parent
aggregate progress record for the full 4,800 rollouts per checkpoint.

## 2026-05-11 14:46 UTC

**Offline eval recovery simplified**: Step 300 parent artifacts are now
recovered from the four shard outputs. Summary: 4,800 rollouts, mean sample
accuracy `0.390625`, single-shot / prefix p@1 `0.3883333333333333`, pass@8
`0.6433333333333333`, error rate `0.0`, truncation `0.134583`.

The recovery process is now explicit:

- `tmp/recover_olmo3_offline_eval_600x8_20260511.sh summarize` merges complete
  shard artifacts and recomputes summaries without touching GPUs.
- `OFFLINE_EVAL_MIN_STEP=350 tmp/recover_olmo3_offline_eval_600x8_20260511.sh continue-fresh`
  resumes generation from the next missing checkpoint with a fresh server.
- `continue-existing` is available but should only be used when admin endpoints
  are verified healthy; reused/orphaned 9100 backends failed during step-350
  weight update.

Detailed runbook: `tmp/olmo3_offline_eval_recovery_20260511.md`.

## 2026-05-11 15:04 UTC

**Pass@k estimator alignment**: Offline baseline summaries now route their
`pass_at_k` calculation through the shared online-eval estimator in
`prime_rl.orchestrator.eval_utils`. The reported `pass_at_k` fields use the
Chen et al. / HumanEval unbiased estimator over all sampled attempts; ordered
first-k diagnostics remain separately available as `prefix_pass_at_k`.

For Default step 300, recomputing through the shared helper leaves values
unchanged: p@1 `0.390625`, p@2 `0.4870833333333338`, p@3
`0.5386309523809528`, p@4 `0.5728571428571426`, p@5
`0.5976785714285717`, p@6 `0.6166071428571432`, p@8
`0.6433333333333333`.

**Truncation diagnostic**: Step 300 still has material truncation:
`646 / 4800 = 13.46%` rollouts hit the `15360` completion-token cap. This is a
quality issue, not just noisy accounting: truncated rollouts were correct at
`2.0%`, while non-truncated rollouts were correct at `44.8%`. Truncation spans
244 examples, with only 4 examples truncating all 8 samples.

**Stopped compromised step-350 continuation**: The fresh `offline-eval5`
continuation on ports `9200/9300` was stopped after `nid011162:9300` became
unhealthy and its Slurm step disappeared. The pane was emitting repeated
API-connection-error rollouts, so partial step-350 artifacts from that attempt
should not be used as eval signal.

**Missing-checkpoint offline eval relaunch**: A second fresh launch on
`9400/9500` exposed the real race: the evaluator waited for the head endpoint
only, then attempted the step-350 weight update before the other three backend
APIs were listening. Patched `scripts/evals/offline_omni_math2_ckpt_eval.py` to
wait for every derived generation/admin endpoint before the first update.

Relaunched in tmux window `joanv_cc_4node:offline-eval6` with:

```bash
OFFLINE_EVAL_STEPS=350,400,450 OFFLINE_EVAL_PORT=9600 OFFLINE_EVAL_BACKEND_PORT=9700 OFFLINE_EVAL_MAX_RETRIES=3 tmp/recover_olmo3_offline_eval_600x8_20260511.sh continue-fresh
```

As of `2026-05-11 15:29 UTC`, all four endpoints report healthy, step 350
paused/resumed cleanly, and generation/scoring has started. Log:
`outputs/omni_math2_rlvr_canary/20260509_1825/offline_eval_600x8/logs/launcher_20260511T152642Z.log`.

## 2026-05-11 22:00 UTC: Default Live Monitoring + Queued MaxRL Watchers

**Hypothesis**: If the user is asleep, the useful move is not to hand-wave
monitoring; it is to leave durable, low-ambiguity live monitors that make the
Default to MaxRL handoff recoverable without guessing which W&B/local artifacts
belong to which arm.

**Change**:

- Patched `tmp/watch_wandb_metrics_20260511.py` to accept explicit
  `--run-path` and `--out-dir`, so the same watcher can attach to Default now
  and MaxRL later.
- Added `tmp/watch_eval_strata_20260511.py` for local W&B eval-table monitoring
  with Chen/HumanEval unbiased pass@k, held-out early-pool strata, and
  difficulty/domain/source breakdowns.
- Added `tmp/start_maxrl_wandb_watch_20260511.sh`; it waits for the MaxRL local
  W&B run directory and then starts a `maxrl-wandb-watch` tmux window.
- Started tmux watchers: `wandb-watch`, `eval-strata`,
  `maxrl-eval-strata`, and `maxrl-wandb-starter`.

**Evidence**:

- Default is still running in `joanv_cc_4node:train-resume` under Slurm job
  `4542540`, around trainer step 520 / orchestrator step 522 at the last
  documented check.
- Default W&B watcher latest file:
  `outputs/omni_math2_rlvr_canary/20260509_1825/wandb_metrics/live/latest.md`.
  At `2026-05-11T21:56Z`, W&B state was `running`, current ckpt step `521`,
  train reward `0.4259`, off-policy max `2`, importance mean about `0.786`,
  and latest eval rows were baseline100/full600 at step 500.
- Step-500 evals: baseline100 p@8 `0.6300`; full600-p8 p@8 `0.6450`.
  Full600-p8 p@1 was `0.3902`, close to the OLMo Inst DPO reference p@1
  `0.367` the user provided, but this is one checkpoint and not a monotone
  learning claim.
- Eval-strata watch found held-out early-hard p@8 improved by `+0.1212` at
  step 500, while early-easy p@1 fell by `-0.0592` and early-normal p@8 fell by
  `-0.1667`.
- `maxrl-eval-strata` correctly reports `tables=0` before MaxRL exists.
  `maxrl-wandb-starter` is waiting for the MaxRL local W&B run directory.
- `logs/orchestrator.log` contains repeated `Timeout during comparison` lines,
  including during train generation and full600 eval. No fatal traceback, OOM,
  inference failure marker, trainer error, or inference-server error was found
  at this check.

**Verdict**: `keep as observability`. The watcher set is useful and should
stay. The eval-strata result argues against a naive "more steps are uniformly
helping" story; current improvement looks selection- and stratum-dependent.

**Next action**: let Default continue under the existing sequential launcher.
When Default exits, confirm MaxRL starts in `train-resume`, confirm
`maxrl-wandb-starter` spawns `maxrl-wandb-watch`, then compare MaxRL against
Default on the same W&B watcher and eval-strata outputs. If allocation ends
first, resume from the latest complete checkpoint and keep the same W&B/run-dir
identity.

## 2026-05-11 22:45 UTC: Stopped Default, Pivoted to Filtering/Refill

**Verdict change**: `discard as long-run recipe unless filtering is fixed`.
The live Default run was stopped intentionally. Do not continue it or launch the
queued MaxRL arm unchanged.

**Evidence**:

- Latest W&B watcher snapshot reached around step `528`.
- Pool composition had collapsed to `pool_easy=0.4018`, `pool_hard=0.5645`,
  `pool_normal=0.0337`; the current 50-step mean normal pool was `0.0375`.
- The current recipe makes the easy/hard pools absorbing inside a live run:
  examples are moved out of normal after one empirical group average crosses
  `easy_threshold=0.875` or `hard_threshold=0.0625`, sampling draws only from
  normal, and `online_difficulty_filtering=true` drops non-normal groups from
  the train buffer.
- `easy_fraction` and `hard_fraction` only reintroduce saved examples on
  checkpoint load. They do not provide per-step hard/easy sampling in this
  code.

**Parallel research result**:

- Upstream Hendrycks sanity uses a prefiltered "perfectible" dataset: base model
  solve rate 20-80% over 40 rollouts. It does not use the online hard/easy
  buffer filter.
- Public math recipes mostly use strict extremes:
  `easy_threshold=1.0`, `hard_threshold=0.0`, often without online filtering.
- Relevant old branches/forks either implement older difficulty-pool sampling
  fractions, solve-rate dataset filters, or experimental replay buffers.
- Broader DAPO/OpenRLHF/verl-style recipes use dynamic filtering with refill:
  reject all-0/all-1 groups and generate replacements until enough accepted
  groups exist, with a finite generated-batch cap.

**Next action**:

1. Do not resume Default or start MaxRL unchanged.
2. Patch a solved-only filter canary: `easy_threshold=1.0`, remove
   `hard_threshold`, keep `online_difficulty_filtering=true`, then dry-run.
   This is a TOML-only stopgap: all-zero groups stay eligible for future
   sampling instead of being permanently quarantined.
3. Implement optional DAPO-style refill so filtered groups do not shrink the
   accepted training batch; log candidate/accepted/refill counts and
   prompts-consumed-per-accepted-group.
4. Build an OmniMath2 perfectible subset with
   `scripts/evals/make_omni_math2_perfectible_subset.py` and use it as a sanity
   run before spending another 1000-step allocation.
5. If an existing checkpoint must be resumed for salvage, pass nonzero
   `--orchestrator.buffer.hard_fraction` and `--orchestrator.buffer.easy_fraction`
   on resume, but treat that as a recovery hack, not a live-run fix.

## 2026-05-11 22:49 UTC: TOML Patch for Solved-Only Filtering

**Verdict**: `keep as stopgap, but still needs refill code`.

Changed all `configs/omni_math2/rl_olmo3*.toml` from the old absorbing
`0.875/0.0625` rule to solved-only filtering: `easy_threshold=1.0`,
`online_difficulty_filtering=true`, and no `hard_threshold`. This intentionally
does not permanently quarantine step-0 all-zero prompts, because they may become
solvable later.

Changed the two PipelineRL-speed recipes to rollout batching:
`batch_size=256`, `max_inflight_rollouts=768`, `rollouts_per_example=8`,
`max_off_policy_steps=8`.

Dry-runs passed for representative configs:

- `/tmp/prime-rl-dryrun-nohard-token-default`
- `/tmp/prime-rl-dryrun-bs256-nohard-pipe-default`
- `/tmp/prime-rl-dryrun-bs256-nohard-pipe-maxrl`

Remaining issue: all-zero rollout groups are now kept in the live prompt pool,
but they can still enter the train rollout buffer with no useful advantage until
we implement DAPO-style drop-without-evict refill. That code change is the real
fix.

## 2026-05-11 23:02 UTC: Independent Buffer-Filtering Verification

**Verdict**: `keep solved-only TOML; implement refill next`.

An independent agent verified the filtering behavior against
`src/prime_rl/orchestrator/buffer.py`,
`src/prime_rl/configs/orchestrator.py`,
`tests/unit/orchestrator/test_buffer.py`, and a focused repro in
`tmp/buffer_verify/`. Its verdict was `CORRECT`.

Key confirmations:

- `online_difficulty_filtering=true` does not itself filter reward-0 groups.
  It only skips groups classified non-normal by thresholds.
- `easy_fraction` and `hard_fraction` are checkpoint-load resurfacing
  fractions, not live sampling proportions.
- With `easy_threshold=1.0`, no `hard_threshold`, and online filtering:
  avg reward `0.0` and `0.5` stay normal, train, and remain sampleable; avg
  reward `1.0` becomes easy, is skipped, and stops being sampled.
- With `easy_threshold=1.0`, `hard_threshold=0.0`, and online filtering:
  avg reward `0.0` becomes hard, is skipped, and stops being sampled. This is
  rejected for OmniMath2 because step-0 unsolved prompts may become solvable.

New gotchas:

- Inflight duplicates can leak training data from evicted prompts. Pool
  membership is sticky, but each arriving group recomputes its own pool name.
  A late mixed-reward group for an already-evicted prompt can still enter the
  train buffer. This is async leakage, not a real reintroduction mechanism.
- `rollout_buffer` is unbounded and sampled from the tail. Fresh rollouts are
  preferred, but transient producer overrun can accumulate stale rollouts and
  memory.
- `online_difficulty_filtering=true` with no thresholds set is a silent no-op.

Next action:

1. Keep `easy_threshold=1.0`, no `hard_threshold`.
2. Implement drop-without-evict refill for all-zero/all-one or filtered groups.
3. Log `example_was_already_evicted`, `previous_pool`, `current_group_pool`,
   `late_normal_rollouts_from_evicted_examples`, rollout-buffer length, and
   rollout age/staleness.
4. Add or consider validator warning when `online_difficulty_filtering=true`
   and both thresholds are unset.

## 2026-05-11 23:18 UTC: Corrected 8-Node Topology BOTEC

**Verdict**: `prefer 24i/8t if multi_node smoke passes; keep 28i/4t as fallback`.

Important correction: `orchestrator.batch_size` is rollouts/samples, not
problem groups. With `rollouts_per_example=8`, `batch_size=256` means 32
problems per step. The earlier 15M-token-per-step estimate was for
`batch_size=2048`, i.e. 256 problems.

Corrected rough sizing at `batch_size=256`, avg ~7.5k tokens/rollout:

- Tokens/step: ~1.9M.
- 4-node `14i/2t`: trainer-bound, roughly ~3.2 min/step if rates extrapolate.
- 8-node `28i/4t`: trainer-bound, roughly ~1.6 min/step.
- 8-node `24i/8t`: roughly balanced, ~50 sec/step if multi-node semantics and
  scaling hold.

Local code check:

- `gpu_layout` cannot express `24i/8t` because it supports exactly one trainer
  node.
- `multi_node` can express the desired shape:
  `num_train_nodes=2`, `num_infer_nodes=1`, `num_infer_replicas=6`,
  `nodes_per_fsdp_group=2`.
- The multi-node template starts one router per inference replica, and the
  config path auto-expands standard TP=1 inference to 4 local DP/api workers per
  inference node. This should mean 6 independent one-node replicas, 24 vLLM
  workers total, but it needs a short smoke before a real run.

Next action:

1. On an 8-node allocation, create temp Default/MaxRL multi-node configs rather
   than rewriting the checked-in gpu_layout TOMLs.
2. Dry-run and smoke 24i/8t for 3-5 steps at `batch_size=256`,
   `max_inflight_rollouts=768`.
3. If clean, run 50-100 step canary at either 256/768 or 512/1536.
4. Keep 28i/4t gpu_layout as fallback if multi-node worker/routing/broadcast
   semantics fail.
5. Do not use `batch_size=2048` until 512/1024 have real telemetry.

## 2026-05-12 10:49 UTC: 24i/8t Multi-Node Runtime Recheck

**Verdict**: `router and launcher preflight clean; launch Default smoke next`.

Verified against `https://github.com/PrimeIntellect-ai/router` tag `v0.1.22`:

- The repo dependency points at `PrimeIntellect-ai/router` `v0.1.22`, not the
  different `vllm-project/router` repo.
- The vendored aarch64 wheel metadata is `vllm-router==0.1.22`; its Python
  files match the `v0.1.22` tag.
- `vllm_router.version.__version__` reports `0.1.12`, but this is a stale
  upstream constant present in the `v0.1.22` tag and is not evidence of the
  wrong wheel.
- `uv run --no-sync vllm-router --help` exposes the flags used by
  `multi_node_rl.sbatch.j2`.
- `parse_router_args(...)` accepts the exact non-PD launch shape:
  `--worker-urls`, `--policy consistent_hash`, `--host 0.0.0.0`,
  `--port`, `--intra-node-data-parallel-size 4`, and
  `--worker-startup-timeout-secs 4200`.
- `uv sync --extra disagg --locked --dry-run` would replace the ad hoc wheel
  from `tmp/vllm-router-aarch64/dist` with the vendored
  `vendor/wheels/aarch64/vllm_router-0.1.22-cp38-abi3-linux_aarch64.whl`.

Verified launcher/runtime corrections:

- Multi-node `srun` is usable in allocation `4555723` with
  `--network=no_vni,disable_rdzv_get`. The earlier "multi-node impossible"
  note is superseded.
- Template now sets node-local compile caches with the correct variable names:
  `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`,
  `XDG_CACHE_HOME`, and `VLLM_RPC_BASE_PATH`.
- Template sets Isambard NCCL/FI defaults explicitly and keeps them overridable.
  This includes `NCCL_NET="AWS Libfabric"` and `NCCL_DEBUG=INFO` for the next
  smoke.
- Repo-env NCCL probes loaded `NCCL version 2.30.4+cuda13.2` and reached
  ~139 GB/s busbw on 2 nodes and ~85 GB/s busbw on 8 nodes at 1 GiB.
- Fresh `bash -n` passed for the generated Default 24i/8t launcher.
- Generated Default 24i/8t configs still have solved-only filtering and the
  expected batch/topology values: `batch_size=256`,
  `max_inflight_rollouts=768`, `rollouts_per_example=8`,
  `max_off_policy_steps=8`, `num_train_workers=8`, 6 one-node inference
  replicas, 4 API/data-parallel workers per inference node, and no
  `hard_threshold`.

Next action: launch the Default 24i/8t smoke in a fresh visible tmux window.
Do not launch MaxRL in parallel on the same 8-node allocation.

## 2026-05-12 11:13 UTC: 24i/8t Multi-Node Smoke — First 8-Node Success

**Hypothesis**: 24 inference + 8 trainer on 8 GH200 nodes will scale better
than the 4-node 14i/2t baseline because the 4-node baseline trainer was
rollout-starved (`time/wait_for_ckpt ≈ 94%` of step). Doubling cluster size
with most of the bump going to inference should yield super-linear speedup
*on rollout-bound configs*.

**Change**:

- Config: `tmp/rl_olmo3_dpo_default_24i8t_bs256_smoke_20260512.toml` (multi-node
  deployment, NOT `gpu_layout`).
- Topology: 6 inference nodes (24 GPUs, DP=4 per node, intra-node API server
  count 4) + 2 trainer nodes (8 GPUs, FSDP shard, `dp_replicate=1`).
- Shape: `batch_size=256`, `rollouts_per_example=8`,
  `max_inflight_rollouts=768`, `max_off_policy_steps=8`, `max_async_level=1`,
  `max_steps=5`.
- Same Olmo-3-7B-Instruct-DPO model, same DPPO default loss, same
  `max_completion_tokens=15360`.
- Two prior runtime fixes shipped together: vllm-router built for aarch64 and
  vendored at `vendor/wheels/aarch64/`, and node-local Triton compile caches
  via `TRITON_CACHE_DIR=/tmp/prime-rl-compile-cache-...`.
- Output dir: `outputs/omni_math2_rlvr_canary/default_24i8t_bs256_smoke_20260512/`.
- Slurm step: `4555723.208`.
- W&B: `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/5d3ee20513174bd9b97fb6a4ce1fc0c3`.

**Evidence (5-step smoke, cold)**:

| step | time (s) | seq len | async | off-policy |
|---:|---:|---:|---:|---:|
| 0 | 252.6 | 2,684 | 0 | 0 (includes baseline100 eval) |
| 1 | 12.2 | 7,611 | 1 | 0 (pipelined behind step 0 eval) |
| 2 | 65.5 | 7,061 | 0 | 2 |
| 3 | 97.1 | 4,653 | 0 | 3 |
| 4 | 73.3 | 6,964 | 1 | 3 |

Steps 2-4 mean: **78.6 s/step** (cold, no warm steady state observed at
max_steps=5).

vs 4-node 14i/2t warm baseline `20260509_1825/dpo_default_14i2t_bs256_eval50_1000step`
(steps 60+): 170 s/step, ~8.6k tok/s aggregate, ~30% trainer MFU, with
`time/wait_for_ckpt = 62s` of 66s step time (trainer idle 94%).

Throughput uplift: ~2.2× faster at 2× total GPUs → **~109% scaling efficiency**
(super-linear), consistent with removing trainer starvation by adding inference
capacity. The trainer-side 4× GPU bump (2 → 8) is over-provisioned for this
batch size; effective batch did not scale 4× so the trainer is now under-utilized
in a different direction.

**Apples-to-apples scope**: Both the 4-node 14i/2t baseline and the 8-node
24i/8t smoke run with `max_async_level=1` and `weight_broadcast.type=nccl`.
The 2.2× speedup is attributable to topology, not to the orthogonal
broadcast/async axis. Switching to `weight_broadcast.type=filesystem` and
`max_async_level=4` is an independent, unpulled lever expected to add another
30-50% throughput by overlapping weight sync with inference (NCCL broadcast
currently idles inference every step). That should be tested in isolation on
24i/8t before introducing further topology changes, so the gain is attributable.

Cold trainer MFU 18-19% is in family with the 4-node cold-start
(also ~20% at step 1, warming to 30% by step 60). Not interpretable as a
warm-state datapoint.

`time/wait_for_ckpt` for this smoke is the load-bearing follow-up metric; if
still >40% of step time at warm steady-state, the conclusion is that 24i/8t is
still rollout-bound and the next sweep should push inference even higher
(26i6t / 28i4t / 30i2t).

**Verdict**: `keep`. 24i/8t multi-node runtime works end-to-end on Isambard,
is faster than 4-node 14i/2t at apples-to-apples config shape, and removes the
trainer-starvation pathology. Two open questions for the next trial.

**Next action**:

1. Promote 24i/8t Default to a 100-step run with eval every 10 steps to read
   warm `time/wait_for_ckpt` and warm MFU. Same config shape, raise `max_steps`
   only.
2. In parallel with (1), `/botec` the inference/trainer ratio at 32 GPUs to
   pick the next topology candidate (26i6t / 28i4t / 30i2t). Ceteris paribus:
   no quantization, no shorter sequences.
3. Defer MaxRL until 24i/8t Default has a warm-state baseline. MaxRL on this
   topology was queued but not launched.

## 2026-05-12 11:38 UTC: 24i/8t Default bs512/fsasync4 Probe

**Hypothesis**: `batch_size=512`, `max_inflight_rollouts=1536`,
`weight_broadcast.type="filesystem"`, and `max_async_level=4` would improve
32-GPU utilization over the clean 24i/8t bs256 NCCL smoke by giving the
trainer more work and overlapping more rollout/weight-sync work.

**Change**:

- Config: `tmp/rl_olmo3_dpo_default_24i8t_bs512_fsasync4_25step_20260512.toml`.
- Output:
  `outputs/omni_math2_rlvr_canary/default_24i8t_bs512_fsasync4_25step_20260512`.
- W&B:
  `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/2aaaf6116f8c4b1b8cf85c5f547b7ccf`.
- Topology unchanged: 24 inference GPUs / 8 trainer GPUs.
- Batch/concurrency: `batch_size=512`, `rollouts_per_example=8`,
  `max_inflight_rollouts=1536`.
- Async/broadcast: `max_async_level=4`,
  `weight_broadcast.type="filesystem"`.
- Inference memory: `gpu_memory_utilization=0.95`.
- Checkpoints every 5 steps.

**Evidence**:

- Reached step 5 and wrote checkpoint/weights for `step_5`.
- No stale file handle crash and no fatal vLLM/router errors.
- `vllm-router` was active.
- `FileSystemWeightUpdateWorker` was active.
- Inference still pauses during filesystem weight updates, typically about
  3-4 seconds.
- Trainer MFU did not become healthy: step 3/4/5 were roughly
  `12.7%`, `12.6%`, `12.0%`.
- vLLM metrics showed KV cache saturation:
  - KV avg often `0.96-0.998`.
  - KV max often `0.999+`.
  - Per-node preemptions climbed into the thousands by step 6.
  - Waiting queues were nonzero/high.

**Verdict**: `discard for now`. This is not a crash, but it is not a good
utilization point. At bf16 KV, `512/1536` is over the useful concurrency knee.
Do not spend a 25-step smoke here unless testing `kv_cache_dtype="fp8_e5m2"` or
another capacity/sequence-length change.

## 2026-05-12 11:43 UTC: 24i/8t Default bs256/fsasync4 25-Step Smoke

**Hypothesis**: `batch_size=256`, `max_inflight_rollouts=768` with filesystem
broadcast and `max_async_level=4` should preserve the clean 24i/8t topology
while avoiding the bs512 KV-preemption cliff.

**Change**:

- Config: `tmp/rl_olmo3_dpo_default_24i8t_bs256_fsasync4_25step_20260512.toml`.
- Output:
  `outputs/omni_math2_rlvr_canary/default_24i8t_bs256_fsasync4_25step_20260512`.
- W&B:
  `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/5181b162216b4cd7817ed64eb3371442`.
- GPU telemetry:
  `outputs/omni_math2_rlvr_canary/gpu_telemetry_bs256_fsasync4_25step_20260512`.
- vLLM metrics:
  `outputs/omni_math2_rlvr_canary/default_24i8t_bs256_fsasync4_25step_20260512/monitor/vllm_metrics_8node.tsv`.
- Eval: low-sample online eval, 32 examples x 4 rollouts every 10 steps.
- Checkpoint: every 5 steps, keep last 2 and interval 25.

**Current evidence while running**:

- Resolved topology is correct: 6 one-node inference replicas, 4 vLLM DP/API
  workers each, 24 inference workers total; trainer world size 8.
- `vllm-router` is active.
- `FileSystemWeightUpdateWorker` is active.
- Filtering is the intended solved-only policy:
  `easy_threshold=1.0`, no `hard_threshold`,
  `online_difficulty_filtering=true`.
- Early step-0 vLLM metrics are much healthier than bs512:
  - running requests roughly 70-135 per inference node;
  - waiting requests 0;
  - KV avg roughly `0.07 -> 0.61`;
  - KV max below roughly `0.67`;
  - preemptions 0.

**Status**: running in visible tmux window `joanv_cc_8node:5` as
`fs256x25`. GPU CSV telemetry is in window `4:gpu-csv`; vLLM metrics are in
window `6:vllm-metrics`.

**Interim verdict**: `keep running`. Do not increase concurrency inside this
run. If this establishes a stable 25-step baseline, the next measured
concurrency point should be either `384/1152` or `512/1536` with fp8 KV.

## 2026-05-12 12:05 UTC: 24i/8t Default bs256/fsasync4 Result

**Verdict**: `partial/discard as utilization answer`. The runtime stack works,
but `max_async_level=4` plus filesystem broadcast did not produce the predicted
trainer-utilization jump. The run hit the off-policy cap and eval produced a
large stale-backlog sawtooth.

**Run**:

- Config: `tmp/rl_olmo3_dpo_default_24i8t_bs256_fsasync4_25step_20260512.toml`
- Output:
  `outputs/omni_math2_rlvr_canary/default_24i8t_bs256_fsasync4_25step_20260512`
- W&B:
  `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/5181b162216b4cd7817ed64eb3371442`
- Topology: 24 inference GPUs / 8 trainer GPUs.
- Shape: `batch_size=256`, `max_inflight_rollouts=768`,
  `rollouts_per_example=8`, `max_async_level=4`,
  `weight_broadcast.type="filesystem"`, `gpu_memory_utilization=0.95`.

**Evidence**:

- Steps 1-10 trainer MFU rose only from `4.8%` to `12.9%`.
- Orchestrator max off-policy reached `8` by step 9.
- Eval at ckpt step 10 was `32x4` and took `363.88s`, with `Avg@4=0.4062`,
  mean completion length `6845.92`, and truncation `23.4%`.
- After eval, step 11 took `370.50s`, then stale-buffer flush steps 12-15
  completed in about `0.5s` each. This is not healthy steady-state throughput;
  it is eval/backlog interaction.

**Decision**:

- Do not read the 24i/8t fsasync4 run as evidence that async alone reaches
  ~30% trainer MFU. It falsifies that simple claim.
- Keep filesystem broadcast as mechanically working, but not sufficient.
- Do not raise `batch_size` above 256 at bf16 KV without either fp8 KV or a
  separate capacity test. The `512/1536` probe already saturated KV.

## 2026-05-12 12:12 UTC: 28i/4t Default bs256/fsasync4 Prefix-Off Probe

**Hypothesis**: If inference is still the bottleneck, shifting from `24i/8t`
to `28i/4t` should improve cluster utilization by increasing decode capacity
and reducing trainer starvation. Also test `enable_prefix_caching=false` at the
same time because rollout groups are pinned/shared enough that APC overhead may
not pay for short math prompts.

**Run**:

- Config:
  `tmp/rl_olmo3_dpo_default_28i4t_bs256_fsasync4_prefixoff_25step_20260512.toml`
- Output:
  `outputs/omni_math2_rlvr_canary/default_28i4t_bs256_fsasync4_prefixoff_25step_20260512`
- W&B:
  `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/eea5e138b12341ff94c6a326e52adf20`
- GPU CSV:
  `outputs/omni_math2_rlvr_canary/gpu_telemetry_28i4t_bs256_fsasync4_prefixoff_25step_20260512`
- vLLM metrics:
  `outputs/omni_math2_rlvr_canary/default_28i4t_bs256_fsasync4_prefixoff_25step_20260512/monitor/vllm_metrics_8node.tsv`
- Topology: 7 inference nodes / 1 trainer node, i.e. 28 inference GPUs and
  FSDP world size 4.
- Shape: `batch_size=256`, `rollouts_per_example=8`,
  `max_inflight_rollouts=768`, `max_async_level=4`,
  `max_off_policy_steps=8`, filesystem broadcast,
  `gpu_memory_utilization=0.95`, `enable_prefix_caching=false`.
- Eval: `32x4` every 10 steps.

**Evidence**:

| window | readout |
|---|---|
| trainer memory | peak `80.8 GiB`, so FSDP4 fits |
| trainer MFU | warmed to `26.4%` at steps 11-12; later mostly `20-23%`, peak `28.0%` at step 22 |
| interval evals | step-10 `32x4` took `345.34s`; step-20 `32x4` took `318.00s` |
| staleness | max off-policy still hit `8` after both evals |
| post-eval behavior | step 13 took `349.96s`; steps 14-16 were stale flushes near `0.5-0.9s`; same pattern after step 20 |
| final eval | final `32x4` hung/wedged after progress output reached about `87/128`; vLLM metrics showed zero running/waiting everywhere and the orchestrator log stopped updating at `12:48:42`; cancelled only the stuck run step |

**Interpretation**:

- 28i/4t improves per-trainer-GPU MFU versus 24i/8t, but it halves trainer GPU
  count. A rough trainer-side cluster contribution is similar:
  `4 * 26% / 32 ~= 3.25%` vs `8 * 13% / 32 ~= 3.25%`.
- Any 28i/4t win must come from inference-side throughput/latency, not from
  trainer FLOPs alone. Current evidence does not crown 28i/4t.
- Eval/backlog interaction remains the main bad shape. The run reached 25
  training steps, trainer finished, and final checkpoint was written, but final
  eval did not complete cleanly.

**Decision**:

- Keep `28i/4t` as a viable topology to retest, not as the default main run.
- For utilization canaries, either disable final eval or use an eval path that
  cancels training rollouts cleanly and does not leave stale buffer/backlog
  state.
- Before another topology sweep, instrument rollout buffer length/age and
  evaluate with a clean saturated `100x8` measurement rather than extrapolating
  from `32x4`.

## 2026-05-12 13:02 UTC: Eval Timing Correction

The earlier claim that `100x8` eval might take 30+ minutes was a bad
extrapolation from the current `32x4` timings. Do not use that estimate.

Observed eval regimes:

- Current 24i/8t `32x4`: `363.88s` for 128 rollouts.
- Current 28i/4t `32x4`: `345.34s` and `318.00s` for 128 rollouts.
- Earlier clean 14i/2t interval `100x8` evals were usually around
  `360-450s`; examples in `HANDOFF.md` include `357.77s`, `361.62s`,
  `370.84s`, `373.52s`, `374.06s`, `381.88s`, `403.02s`, and `447.17s`.
- Earlier final evals were pathological: default final `1850.72s`, MaxRL final
  `1723.97s`. Treat final eval wall-clock separately from interval eval.

Correct eval law:

```text
eval_time ~= fixed_startup + generated_tokens / effective_decode_tok_s + tail_penalty
```

The `32x4` probe underfills 24-28 inference GPUs and is dominated by long-tail
stragglers; linear rollout-count scaling from it is wrong. A clean saturated
`100x8` on the 8-node topology should be measured directly. Prior evidence says
central planning range is more like `6-12 min`; `30+ min` is a bad-case/final
eval number, not the central interval-eval estimate.

## 2026-05-12 13:14 UTC: 28i/4t vs 24i/8t Training-Throughput Comparison

**Question**: ignoring eval wall-clock, which 32-GPU topology is better for
the current `batch_size=256`, `max_inflight_rollouts=768`, fs+async4 shape?

**Fair pre-eval comparison**:

| run | comparison window | orchestrator time | trainer time | trainer MFU | trainer tok/s | max off-policy |
|---|---:|---:|---:|---:|---:|---:|
| 24i/8t fs+async4 bs256 | steps 3-10 | mean `60.80s`, median `61.22s` | mean `50.75s`, median `51.02s` | mean `11.07%`, median `11.25%` | mean `12,480` | `8` |
| 28i/4t fs+async4 prefix-off bs256 | steps 3-12 | mean `48.89s`, median `47.70s` | mean `51.51s`, median `51.05s` | mean `21.57%`, median `21.10%` | mean `12,166` | `7` |
| 28i/4t fs+async4 prefix-off bs256 | steps 3-12 and 17-22, excluding eval/stale flush | mean `55.48s`, median `51.82s` | mean `49.91s`, median `49.60s` | mean `21.83%`, median `21.20%` | mean `12,312` | `7` |

Important correction to the Claude table: the `28i/4t` window `3-22 excluding
13` includes stale-flush steps `14-16` at roughly `0.5-0.9s`. Including those
is not a clean steady-state estimate. Excluding them, 28i/4t still wins
orchestrator wall-clock versus 24i/8t, but by about `9-20%`, not by a fake
near-zero-step boost.

**GPU telemetry over pre-eval windows**:

| run | inference util | trainer util | aggregate util | trainer mem |
|---|---:|---:|---:|---:|
| 24i/8t steps 3-10 | mean `93.4%` | mean `39.9%`, median `3.0%` | mean `79.7%` | mean `75.5 GiB` |
| 28i/4t steps 3-12 | mean `93.2%` | mean `63.0%`, median `100.0%` | mean `89.3%` | mean `83.1 GiB` |

**Verdict**:

- For training-loop throughput at `bs=256`, `28i/4t + fs+async4 +
  prefix_caching=false` is the best-known tested topology.
- The strongest measured point is not trainer token throughput, which is
  roughly tied with 24i/8t. The win is lower orchestrator wall-clock with a
  much busier single-node trainer.
- The claim that FSDP4 trainer time would roughly double versus FSDP8 is
  falsified in this regime: measured trainer time is about the same. The
  likely explanation is single-node FSDP4 avoiding cross-node trainer
  collectives, but this mechanism is inferred; we did not capture a comm
  breakdown.
- Do not yet promote checked-in TOMLs blindly. Use this as the next temp
  default for canaries, then make a controlled run without final-eval wedging
  before editing the main checked-in recipes.

**Claim triage**:

- Correct: `28i/4t` is the best-known tested training topology at `bs=256`.
- Correct: `24i/8t bs512` is overloaded at bf16 KV.
- Overconfident: the exact cluster MFU calculation using inference MFU
  `~1%`; we did not measure inference FLOP MFU, only high GPU util and vLLM
  queue/KV metrics.
- Overconfident: attributing all the win to intra-node FSDP. Plausible, but
  not directly measured.
- Wrong/currently unavailable: `vllm_extra.num_scheduler_steps`; the installed
  local vLLM did not expose that knob when checked.

## 2026-05-12 13:56 UTC: 28i/4t `bs512`, Inflight 1024 Probe

**Hypothesis**: The `24i/8t bs512 max_inflight=1536` run failed because it
overfilled bf16 KV. On the better `28i/4t` topology, lowering inflight to 1024
might make `batch_size=512` usable while improving trainer MFU.

**Change**:

- Temp config:
  `tmp/rl_olmo3_dpo_default_28i4t_bs512_inflight1024_fsasync4_prefixoff_12step_20260512.toml`
- Output:
  `outputs/omni_math2_rlvr_canary/default_28i4t_bs512_inflight1024_fsasync4_prefixoff_12step_20260512`
- W&B:
  `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/c9373ff8a0ae42dba4833f21abbd9dad`
- Shape: `28i/4t`, `batch_size=512`, `max_inflight_rollouts=1024`,
  `rollouts_per_example=8`, `max_async_level=4`, filesystem broadcast,
  prefix caching off, no online/final eval, checkpoint interval `100`,
  solved-only filtering (`easy_threshold=1.0`, no `hard_threshold`).
- Caveat: the temp TOML still had `dry_run = true` because it was used to
  generate the launcher. The generated launcher did run. Clean future temp TOMLs
  should remove that flag after generation to avoid confusion.

**Evidence**:

The run completed 12 train/orchestrator steps and wrote final checkpoints.
After completion, telemetry step `.224` and run step `.226` were cancelled
because vLLM/metrics shells were still alive after useful work had finished.

Orchestrator timing:

| window | mean step time | median step time | max off-policy | mean seq len |
|---|---:|---:|---:|---:|
| steps 2-11 | `95.31s` | `98.38s` | `5` | `6576.5` |
| steps 4-11 | `100.55s` | `101.97s` | `5` | `6264.3` |

Trainer timing:

| window | mean step time | median step time | mean MFU | median MFU | mean tok/s |
|---|---:|---:|---:|---:|---:|
| steps 2-11 | `94.16s` | `94.53s` | `27.22%` | `27.75%` | `15,314` |
| steps 4-11 | `92.39s` | `94.53s` | `28.62%` | `28.15%` | `16,110` |

Late trainer steps reached higher MFU than the `bs256` run: step 10 was
`98.74s`, `18,302 tok/s`, `32.5%` MFU; step 11 was `79.61s`,
`18,195 tok/s`, `32.3%` MFU. Peak trainer memory was `80.8 GiB`.

vLLM pressure from the 7 inference-node metrics sampler:

- mean running requests `836.5`, max running `1021` out of
  `max_inflight_rollouts=1024`.
- mean waiting requests `34.9`, max waiting `130`.
- mean KV average `0.744`, p95 KV average `0.938`, max KV `1.000`.
- preemptions accumulated from `0` to `5242`.

Historical GPU CSV telemetry, with pre-start idle rows dropped:

| role | util mean | util median | p90 util | mean mem | max mem |
|---|---:|---:|---:|---:|---:|
| inference | `75.3%` | `100.0%` | `100.0%` | `85.4 GiB` | `94.9 GiB` |
| trainer | `47.3%` | `15.0%` | `100.0%` | `66.5 GiB` | `83.5 GiB` |
| all GPUs | `72.0%` | `100.0%` | `100.0%` | `83.2 GiB` | `94.9 GiB` |

W&B sample logging dirtied timing: step 4 logged 512 long samples and emitted
many 100k+ string serialization warnings. Future perf probes should set sample
logging off or a low sample ratio.

**Verdict**: `promising but not default`. `28i/4t bs512 inflight1024` is viable
and did not collapse like `24i/8t bs512 inflight1536`, but it is slower in
wall-clock than `28i/4t bs256`: roughly `95-101s` orchestrator steps versus
`49-55s` for bs256 windows. The trainer gets busier (`~28-32%` MFU late), but
inference/KV pressure becomes the limiter.

**Next action**:

1. Keep `28i/4t bs256 fs+async4 prefix-off` as the current default utilization
   shape.
2. If testing bs512 again, use a clean compiled-trainer config and disable W&B
   sample tables; do not raise inflight above 1024 with bf16 KV.
3. Add `[trainer.model.compile]` in the next controlled probe; the bs512 run
   confirmed `compile=None` in trainer logs, so this run is an uncompiled
   baseline, not a max-performance recipe.

### 2026-05-12 14:16 UTC — First-class 28i/4t compiled config dry-run

**Change**:

- Added first-class config:
  `configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4.toml`
- Promoted telemetry helpers to:
  `scripts/monitoring/gpu_telemetry_loop.sh` and
  `scripts/monitoring/sample_vllm_metrics.sh`
- Updated `skills/config/SKILL.md`, `GPUS.md`, and `HANDOFF.md` to make the
  2026-05-12 28i/4t result the current default instead of stale 24i/8t notes.

**Dry-run command**:

```bash
uv run --no-sync rl @ configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4.toml \
  --dry-run --output-dir /tmp/prime_rl_dryrun_28i4t_compile_fsasync4_20260512_b
```

**Evidence**:

- Dry-run completed and wrote `/tmp/prime_rl_dryrun_28i4t_compile_fsasync4_20260512_b/rl.sbatch`.
- `bash -n` passed on the generated launcher.
- Resolved orchestrator config has `batch_size = 256`,
  `max_inflight_rollouts = 768`, `rollouts_per_example = 8`,
  `max_off_policy_steps = 8`, `max_async_level = 4`,
  `easy_threshold = 1.0`, `online_difficulty_filtering = true`, and no
  `hard_threshold`.
- Resolved trainer config contains `[model.compile]`.
- Resolved inference config has `gpu_memory_utilization = 0.95`,
  `enable_prefix_caching = false`, `max_num_seqs = 192`, and
  `max_num_batched_tokens = 65536`.
- Generated launcher resolves `NUM_TRAIN_NODES=1` and
  `NUM_INFER_REPLICAS=7`, sets node-local TorchInductor/Triton cache dirs, and
  detects `vllm-router`.

**Correction**: Do not add `vllm_extra.num_scheduler_steps` in this environment.
`rg` over `.venv/lib/python3.12/site-packages/vllm` found no such flag in the
installed local vLLM, so the committed config leaves it out.

**Tests**:

```bash
uv run --no-sync pytest tests/unit/test_configs.py -k 'not test_load_configs' \
  tests/unit/orchestrator/test_envs.py \
  tests/unit/orchestrator/test_scheduler.py \
  tests/unit/train/rl/test_loss.py \
  tests/unit/train/rl/test_nccl_broadcast_coordination.py \
  tests/unit/utils/test_client.py
```

Result: `49 passed, 125 deselected`.

The broader command including `test_load_configs` failed on seven existing
baseline/eval-suite TOMLs (`configs/baselines/*_local.toml` and
`configs/evals/rung6_suite.toml`) because that test only tries RL/trainer/SFT/
orchestrator/inference config classes, not the baseline/eval-suite loaders.
The OmniMath2 run-coupled tests passed.
