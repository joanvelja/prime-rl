# Handoff: Omni-MATH-2 OLMo3 RLVR Canaries

> Last updated: 2026-05-13 11:40 UTC
> Session focus: 8-node Isambard OLMo3 RLVR config refresh for fp32 lm-head
> inference, private perfectible dataset, LR `3e-6`, and 512/16 rollout groups.
> Trial journal: append to `TRIALS.md` after each material run/config/code trial.

## 2026-05-13 11:40 UTC Checked-In Next RLVR Recipe

Active config:
`configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4_refill.toml`.

Recipe now targets:

- 28 inference GPUs / 4 trainer GPUs via `multi_node`.
- `inference.enable_fp32_lm_head = true`, ported locally from merged upstream
  PR `PrimeIntellect-ai/prime-rl#2441`.
- Dataset: private HF repo
  `joanvelja/omni-math2-olmo3-perfectible-seed42`.
- `trainer.optim.lr = 3e-6`.
- `batch_size = 512`, `rollouts_per_example = 16`, so still 32 prompt
  groups/update.
- `max_inflight_rollouts = 1024` for 2x in-flight rollout capacity.
- DAPO-like refill/filtering is preserved:
  `train_batch_refill.enabled=true`, `candidate_groups_per_round=32`,
  `max_candidate_groups=128`, `easy_threshold=1.0`,
  `online_difficulty_filtering=true`, and no `hard_threshold`.

Verification just run:

- `uv run --no-sync ruff check src/prime_rl/configs/inference.py src/prime_rl/inference/patches.py src/prime_rl/inference/vllm/worker/__init__.py tests/unit/test_configs.py`
- `uv run --no-sync pytest tests/unit/test_configs.py::test_inference_fp32_lm_head_threads_through_vllm_additional_config`
- `uv run --no-sync rl @ configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4_refill.toml --dry-run --output-dir /tmp/olmo3_fp32lmhead_perfectible_lr3e6_bs512_gs16_dryrun_20260513T1208`
- `uv run --no-sync python -m prime_rl.entrypoints.launch rlvr --config configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4_refill.toml --dry-run --output-dir /tmp/prime_launch_fp32lmhead_perfectible_lr3e6_bs512_gs16_dryrun_20260513T1208`
- `bash -n /tmp/prime_launch_fp32lmhead_perfectible_lr3e6_bs512_gs16_dryrun_20260513T1208/rl.sbatch`
- HF smoke passed only after sourcing `.env` and redirecting caches to writable
  `/tmp`: the dataset has 340 train rows and columns
  `answer,difficulty,domain,id,problem,solution,source,tags`.

Launch caveat: the dataset repo is private. A raw shell without `.env`/HF token
gets `DatasetNotFoundError`; the Slurm script sources `.env`, so make sure the
allocation environment still has the Hugging Face token and that cache paths are
writable on compute nodes.

## 2026-05-12 21:56 UTC Current State

Do not touch the GPUs in the live allocation unless the user explicitly asks.
The user is also working in parallel there.

Live `1e-6` refill run:

- Slurm job: `4570549`
- Nodes: `nid[010571,010577,010597-010601,010603]`
- Run dir:
  `outputs/omni_math2_rlvr_canary/default_8node_28i4t_compile_fsasync4_refill_20260512_1745`
- W&B:
  `jvelja-private/omni-math2-rlvr/9dd054d2dd2e46de924a027c9ef20be4`
- Config shape: 28 inference GPUs / 4 trainer GPUs, `batch_size=256`,
  `rollouts_per_example=8`, `max_inflight_rollouts=768`,
  `max_async_level=4`, filesystem broadcast, prefix caching off,
  solved-only filtering (`easy_threshold=1.0`, no `hard_threshold`).
- Latest read-only check: trainer step 53 completed at `21:39:54`, MFU `10.4%`,
  peak trainer memory `84.0 GiB`; orchestrator was working on step 55.
- Last 15 completed orchestrator steps averaged `238.21s`, reward `0.4077`,
  and mean sequence length `5829` tokens/sample.
- Last 15 completed trainer steps averaged `227.58s`, `5940 tok/s`, and
  `10.51%` MFU.
- This live run is pre-`c8a5b8307` adaptive refill patch. Its refill path still
  draws full candidate batches for each retry.

Parallel LR arm status:

- Failed first submission: Slurm job `4572407`, state `FAILED`, elapsed `44s`,
  exit `15:0`. `sacct` showed `.0` completed and `.1` failed, so the failure
  was in the main multi-node `srun` phase, before meaningful RL runtime. The
  submit dir was under node-local `/tmp`, so durable logs were not available;
  do not reuse that path:
  `/tmp/olmo3_refill_patch_lr3e6_100step_submit_20260512T2029`
- Active retry: Slurm job `4574276`, job name
  `olmo3-28i4t-refill-lr3e6b`, pending on priority at `21:54 UTC`.
- Durable submit dir:
  `/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/outputs/omni_math2_rlvr_canary/lr3e6_28i4t_refill_shared_submit_20260512_2155`
- Temp config:
  `tmp/olmo3_28i4t_refill_patch_lr3e6_100step_shared_20260512_2150.toml`
- Tmux watcher: `joanv_cc_8node:6 lr3e6-watch`
- Shape: same as the live run except `trainer.optim.lr = 3e-6` and it uses
  the patched adaptive refill code.
- This is a separate `sbatch`; it does not use the live allocation.

Code state:

- Adaptive refill code is present in the current working tree. Current HEAD at
  the last check was `e5c993ede feat(baselines): re-add FP8 KV + prefix caching
  + stream_interval to omni-math2 olmo3 config`.
- Unrelated dirty files remain and were intentionally not included in this LR
  work: `src/prime_rl/baselines/provision.py` and
  `docs/plans/2026-05-07-fused-mlp-kernels.md`.

Verification for `c8a5b8307`:

- Focused refill/config/scheduler/buffer tests: `6 passed`.
- Ruff on touched files: passed.
- Dry-run:
  `/tmp/olmo3_refill_bottleneck_patch_explicit_dryrun_20260512T2021`.
- Failed LR-arm dry-run used `/tmp`:
  `/tmp/olmo3_refill_patch_lr3e6_100step_submit_20260512T2029`.
- Durable LR-arm dry-run:
  `/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/outputs/omni_math2_rlvr_canary/lr3e6_28i4t_refill_shared_submit_20260512_2155`.
- Generated LR-arm configs confirmed `lr=3e-06`, `batch_size=256`,
  `max_inflight_rollouts=768`, `max_async_level=4`, `num_workers=28`,
  `gpu_memory_utilization=0.95`, `online_difficulty_filtering=true`,
  `easy_threshold=1.0`, no `hard_threshold`, `candidate_groups_per_round=32`,
  `max_candidate_groups=128`.
- Generated durable LR-arm `rl.sbatch` uses absolute shared output paths and
  passed `bash -n`.

Next checks:

1. Wait for `4574276` to start; verify worker count, router/backend URLs,
   first trainer step, and W&B run id.
2. If `4574276` fails before startup, read the durable job log first:
   `/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/outputs/omni_math2_rlvr_canary/lr3e6_28i4t_refill_shared_submit_20260512_2155/job_4574276.log`.
   Prediction: if `.0` succeeds and `.1` fails, this is main `srun` launch/env,
   not LR/refill behavior.
3. Compare `4574276` vs `4570549` on:
   `train_batch_refill/reward_unconditioned_on_filtering/mean`,
   `train_batch_refill/reward_conditioned_on_filtering/mean`,
   `train_batch_refill/prompts_consumed_per_accepted_group`,
   `train_batch_refill/overflow_groups`, step time, trainer MFU, and eventual
   offline eval.
4. Treat `4570549` vs `4574276` as a bundled comparison
   (`1e-6 + old refill scheduler` vs `3e-6 + adaptive refill scheduler`).
   If this looks good or ambiguous, run a patched `1e-6` control before making
   a clean LR-only claim.

## 2026-05-12 8-Node Status

Allocation `4555723` is still alive on
`nid010685,nid010752,nid010753,nid010756,nid010757,nid010758,nid010765,nid010768`.
The 28i/4t bs512 probe and its telemetry panes have been stopped; tmux windows
`4`, `5`, and `6` were closed. Remaining tmux windows in `joanv_cc_8node` are
`claude`, `work`, and `gpustat`. Slurm still shows only base allocation/batch
steps plus long-lived shell/gpustat steps; leave those alone unless
intentionally releasing the allocation.

Current verified filtering policy remains:

- `online_difficulty_filtering=true`
- `easy_threshold=1.0`
- no `hard_threshold`

Do not add `hard_threshold=0.0`; all-zero prompt groups should not be
permanently evicted.

Runtime/template fixes made during the 8-node work:

- `src/prime_rl/templates/multi_node_rl.sbatch.j2` now uses node-local compile
  caches under `/tmp` for vLLM/TorchInductor/Triton.
- Isambard NCCL/FI envs are explicit in `configure_prime_rl_runtime`, with
  `NCCL_DEBUG=INFO` for smoke visibility.
- `PRIME_RL_SRUN_NETWORK` defaults to `no_vni,disable_rdzv_get`.
- `vllm-router` now exists on aarch64 via the vendored wheel
  `vendor/wheels/aarch64/vllm_router-0.1.22-cp38-abi3-linux_aarch64.whl`;
  the multi-node launcher detects and uses it.
- First-class 28i/4t compiled-trainer config:
  `configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4.toml`.
- First-class monitoring helpers:
  `scripts/monitoring/gpu_telemetry_loop.sh` and
  `scripts/monitoring/sample_vllm_metrics.sh`.
- Targeted run-coupled unit tests passed with `49 passed, 125 deselected`.
  The broader `test_load_configs` collection still fails on baseline/eval-suite
  TOMLs that are not parsed by the RL config classes; do not interpret that as
  an OmniMath2 config failure.

8-node run conclusions:

- Live proper run is currently active:
  `outputs/omni_math2_rlvr_canary/default_8node_28i4t_compile_fsasync4_20260512_1430`.
  It uses the first-class 28i/4t compiled config, no online/final eval,
  checkpoint interval `25`, and W&B run
  `d42fd686c81044f1a52341633bb3331d`. Visible tmux windows are
  `vllm-metrics`, `gpu-telemetry`, and `run-28i4t` in `joanv_cc_8node`.
  Early checks: inference pool ready, routers active, trainer compiled 32
  layers, step 1 trainer logged `80.19s`, `8381 tok/s`, `14.9%` MFU, and
  latest GPU telemetry showed all 32 GPUs at `100%`.

- 24i/8t multi-node launches correctly: six one-node inference replicas, four
  vLLM workers per replica, 24 inference workers total, and FSDP world size 8.
- 24i/8t `batch_size=512`, `max_inflight_rollouts=1536` at bf16 KV is over the
  useful concurrency knee: KV avg/max approached saturation and per-node
  preemptions climbed into the thousands. Discard this point unless fp8 KV is
  explicitly tested.
- 24i/8t `batch_size=256`, `max_inflight_rollouts=768`,
  `max_async_level=4`, filesystem broadcast works mechanically but still hit
  max off-policy `8` and only reached about `12-13%` trainer MFU by step 10.
  Eval caused a stale-backlog sawtooth.
- 28i/4t `batch_size=256`, `max_inflight_rollouts=768`,
  `max_async_level=4`, filesystem broadcast, prefix caching off completed 25
  train steps. FSDP4 fits with peak trainer memory `80.8 GiB`. A follow-up
  apples-to-apples parse of pre-eval training windows says 28i/4t is the
  current best-known tested training topology: mean orchestrator time `48.89s`
  on steps 3-12, versus `60.80s` for 24i/8t fs+async4 steps 3-10. Trainer
  wall-clock is effectively tied (`51.51s` vs `50.75s`), but trainer MFU is
  much higher (`21.57%` vs `11.07%`).
- The 28i/4t final `32x4` eval wedged after progress output reached about
  `87/128`; vLLM endpoints were healthy but had zero running/waiting requests
  and the orchestrator log stopped updating. The stuck run step was cancelled
  after trainer completion/final checkpoint.
- 28i/4t `batch_size=512`, `max_inflight_rollouts=1024`,
  `max_async_level=4`, filesystem broadcast, prefix caching off, and no eval
  completed 12 train/orchestrator steps. It did not collapse like the earlier
  24i/8t bs512/inflight1536 run, but it is not the current default: steps 2-11
  had mean/median orchestrator time `95.31s` / `98.38s`, while trainer MFU
  averaged `27.22%` and reached `32.5%` late. vLLM was under real pressure:
  mean/max running requests `836.5` / `1021`, waiting max `130`, p95 KV average
  `0.938`, max KV `1.000`, and `5242` cumulative preemptions. This is a viable
  uncompiled baseline, not a max-performance recipe.
- The bs512 temp TOML had `dry_run = true` because it was used to generate the
  launcher; the generated launcher did run. Clean future launch TOMLs should
  remove that flag after generation to avoid confusion.
- Trainer compile was missing in the bs512 probe: trainer log showed
  `compile=None`. The upstream `examples/Intellect-3.1/rl.toml` uses an empty
  `[trainer.model.compile]` table. Add that to the next controlled perf probe
  before making a stronger throughput call.

Eval timing correction:

- Do not extrapolate `100x8` wall-clock linearly from current `32x4` evals.
  Current `32x4` underfills 24-28 inference GPUs and is tail dominated.
- Earlier clean interval `100x8` evals on 14i/2t were often `~360-450s`; final
  evals were pathological at `~1700-1850s`.
- For planning, central clean/saturated `100x8` is closer to `6-12 min` than
  `30+ min`. The `30+ min` number is a final-eval/bad-path estimate, not the
  interval-eval baseline.

Recommended next experiments:

1. Use 28i/4t fs+async4 prefix-off `batch_size=256` as the current temp
   default for training-loop utilization. The first-class config above encodes
   that shape with `[trainer.model.compile]` and W&B sample-table logging off.
2. Run the clean compiled-trainer probe from that config before changing batch
   size again.
3. Run a clean saturated eval-only or interval eval `100x8` on the 8-node
   topology to fit eval timing directly.
4. For utilization canaries, disable final eval and either disable online eval
   or make eval cancellation/backlog behavior explicit. Otherwise the run
   measures eval/backlog sawtooth, not steady training throughput.
5. Add rollout-buffer length/age/staleness metrics and DAPO-style
   drop-without-evict refill before spending a long canary.
6. Do not re-add `vllm_extra.num_scheduler_steps` unless a local vLLM package
   exposes it. The 2026-05-12 environment did not.

Claim triage from the 28i/4t vs 24i/8t comparison:

- Correct: 28i/4t is the best-known tested training topology at `bs=256`.
- Correct: FSDP4 did not double trainer step time versus cross-node FSDP8.
- Plausible but unproven: the win comes from single-node FSDP4 avoiding
  cross-node trainer collectives. We did not capture a comm breakdown.
- Not verified: exact cluster MFU using inference MFU `~1%`; we measured high
  GPU utilization, not inference FLOP MFU.
- Not currently actionable: `vllm_extra.num_scheduler_steps`; the installed
  local vLLM did not expose that knob when checked.

## Objective

Continue Omni-MATH-2 OLMo3 RLVR without GIGO: preserve solved-only filtering,
measure 8-node utilization honestly, and do not relaunch long Default/MaxRL
runs until rollout/refill/eval-backlog pathologies are instrumented. The older
4-node runs diagnosed weak/unstable eval uplift and absorbing difficulty-filter
collapse; the 2026-05-12 8-node work verified multi-node launch mechanics but
did not yet find a clean high-utilization recipe.

## Current Status

**State**: Slurm allocation `4542540` is still preserved on `nid011162,nid011164,nid011183,nid011184` in tmux session `joanv_cc_4node`. The RL/vLLM/trainer/offline-eval panes were stopped and closed. Current tmux windows are only `node` and `work`; do **not** kill them while Codex is attached. Slurm still shows base allocation/batch steps plus interactive `bash` steps on `nid011162`; leave those alone unless intentionally releasing or reattaching the allocation.
**Branch**: `feat/omni-math2-olmo3-rlvr-canary`
**Important user instruction**: when launching a run, attach/use a tmux pane. Do not fire long runs from an invisible shell.

## Live 2026-05-11 Monitoring Notes

The earlier run/watch windows listed here were stopped and closed after the
absorbing-filter diagnosis. Current live windows are only `node` and `work`.
Do not infer that Default or MaxRL is running from the historical notes below.

Monitoring/code artifacts added or updated:

- `tmp/watch_wandb_metrics_20260511.py`: live W&B watcher; now accepts `--run-path` and `--out-dir`.
- `tmp/watch_eval_strata_20260511.py`: local W&B eval-table watcher with Chen/HumanEval unbiased pass@k, held-out early-pool strata, difficulty/domain/source strata, and `latest.md`/CSV/JSONL outputs.
- `tmp/start_maxrl_wandb_watch_20260511.sh`: waits for MaxRL local W&B metadata and starts the MaxRL W&B watcher.

Latest Default online eval at step 500:

- Baseline100: p@1 `0.3725`, p@2 `0.4804`, p@4 `0.5627`, p@8 `0.6300`, truncation `0.1187`, eval time `841.2s`.
- Full600-p8: p@1 `0.3902`, p@2 `0.4874`, p@4 `0.5730`, p@8 `0.6450`, truncation `0.1288`, eval time `1818.8s`.
- Held-out early-pool watch says step-500 improvement is concentrated in `early_hard` p@8 (`+0.1212` vs first baseline100 eval), while `early_easy` p@1 dropped (`-0.0592`) and `early_normal` p@8 dropped (`-0.1667`). This is not monotone global improvement.

Important caveats:

- The online eval set is held out from train: `omni_math2_train_excluding_baseline600_seed42.jsonl` excludes `omni_math2_stratified_600_seed42.jsonl`. Do not join eval IDs to train easy/normal/hard pools. Use held-out early-pool labels from the first baseline100 eval instead.
- `filtered_rollouts/easy|hard` in W&B means difficulty-filtered out of the train pool, not literal repetition/gibberish filtering.
- `Timeout during comparison` appears repeatedly in `logs/orchestrator.log`, including during train generation and full600 eval, but there is no matching fatal traceback, OOM, inference failure marker, or trainer/inference log error at the last check. Treat it as a scorer/verification timeout symptom to monitor, not as proof the run is broken.

## Live 2026-05-09 Notes

Active Slurm allocation:

- Job: `4507057`
- Nodes: `nid011043,nid011047,nid011074,nid011086`
- Time limit: 24h, ending `2026-05-10 17:27 UTC`
- tmux session: `joanv_cc_4node`
- Pre-launch steps showed only allocation/batch/bash steps alive; no RL child steps.

Prepared fixed-batch 1000-step sequential trial:

- Default config: `tmp/rl_olmo3_dpo_default_14i2t_bs256_eval50_1000step_20260509_1825.toml`
- MaxRL config: `tmp/rl_olmo3_dpo_maxrl_14i2t_bs256_eval50_1000step_20260509_1825.toml`
- Launcher: `tmp/run_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh`
- Shape: `max_steps=1000`, `14 inference / 2 trainer`, fixed `batch_size=256`, resolved `max_inflight_rollouts=256`, train `rollouts_per_example=8`, `max_off_policy_steps=8`, `importance_ratio_clip=5.0`, eval `100x8`, eval interval `50`, `eval_base_model=false`, `cancel_inflight_rollouts_on_eval=false`, checkpoint interval `50`, `keep_last=4`, W&B extras interval overridden to `50` in the launcher, vLLM caps unchanged (`gpu_memory_utilization=0.93`, `max_num_seqs=192`, `max_num_batched_tokens=65536`, `generation_config="vllm"`).
- Dry-runs passed at `2026-05-09 18:37 UTC` with the active host list. Resolved orchestrator configs confirmed `batch_size=256`, `max_inflight_rollouts=256`, `max_steps=1000`, eval/checkpoint/W&B-extra intervals `50`, eval cancellation off, and `bench=false`.
- Do not silently change `max_inflight_rollouts` to `768`: that is 3x oversampling. Upstream removed Hendrycks oversampling for trainer-bottleneck reasons, and the user explicitly pushed back on this.
- Runtime risk: old token-batched `524288` steps had median packed microbatches around `31`; bootstrapping old rollout lengths to fixed `batch_size=256` gives typical `72-74` packed microbatches and p95 `79-81`, with worst old-step resampling around `144-148`. Peak activation memory should still be bounded by the existing `18432` trainer seq_len, but step wall time may be much longer. The full Default+MaxRL pair may not fit the remaining allocation; rely on ckpt-50 resume if needed.
- Launch: started `tmp/run_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh` in tmux window `joanv_cc_4node:omni-bs256-1000` at `2026-05-09 18:39 UTC`. Default runs first; MaxRL starts only if Default exits successfully.
- Default output dir: `outputs/omni_math2_rlvr_canary/20260509_1825/dpo_default_14i2t_bs256_eval50_1000step`
- Default W&B run id: `197d96382b0c40d59272ac0fbc94a9e3`.
- Startup: all 14 inference servers came up; orchestrator logged `Inference pool ready` at `18:42:03`, then started step 0 at `18:42:06`. The startup `Inference server was not reached after N seconds` warnings resolved and were not fatal.
- Step 0 completed cleanly at `18:47:12`: `304.19s`, reward `0.4453`, mean seq length `4718.3` tokens/sample, async/off-policy `0/0`, and only `1/256` rollouts flagged by repetition monitoring. No OOM or traceback observed.
- Trainer step 0 completed at `18:49:14`: `427.38s`, loss `-0.0001`, mismatch KL `0.7097`, peak mem `81.0 GiB`. The logged `0 tokens/s` / `0.0%` MFU is startup PerfCounter warmup, not meaningful trainer MFU.
- Orchestrator step 1 completed at `18:50:51`: `218.65s`, reward `0.4492`, mean seq length `6361.1`, async/off-policy `0/1`, and `1/256` repetition-flagged.
- Trainer step 1 completed at `18:54:08`: `290.79s`, `5640 tokens/s`, `20.0%` MFU, peak mem `93.4 GiB`. This is the first meaningful trainer MFU point. It fits but is close to the device limit.
- Orchestrator step 2 completed at `18:54:41`: `229.41s`, reward `0.4496`, mean seq length `5703.8`, async/off-policy `0/1`, and `1/256` repetition-flagged.
- Trainer step 2 completed at `18:57:35`: `204.42s`, `6248 tokens/s`, `22.2%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 3 completed at `18:58:24`: `222.54s`, reward `0.4062`, mean seq length `5862.6`, async/off-policy `0/2`, and `4/256` repetition-flagged.
- Trainer step 3 completed at `19:01:17`: `218.88s`, `6464 tokens/s`, `22.9%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 4 completed at `19:01:32`: `186.94s`, reward `0.4258`, mean seq length `5570.5`, async/off-policy `0/2`.
- Trainer step 4 completed at `19:04:16`: `175.78s`, `6774 tokens/s`, `24.0%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 5 completed at `19:04:30`: `178.45s`, reward `0.4648`, mean seq length `4594.8`, async/off-policy `0/2`.
- Trainer step 5 completed at `19:06:54`: `155.35s`, `6912 tokens/s`, `24.5%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 6 completed at `19:08:25`: `233.70s`, reward `0.3633`, mean seq length `5106.3`, async/off-policy `0/2`, and `4/256` repetition-flagged.
- Trainer step 6 completed at `19:10:58`: `240.66s`, `6636 tokens/s`, `23.6%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 7 completed at `19:12:14`: `229.54s`, reward `0.4254`, mean seq length `6742.5`, async/off-policy `0/2`.
- Trainer step 7 completed at `19:15:43`: `282.78s`, `6555 tokens/s`, `23.3%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 8 completed at `19:15:19`: `184.40s`, reward `0.3672`, mean seq length `5801.5`, async/off-policy `1/1`, and `4/256` repetition-flagged. It then hit the first small backpressure event: step 9 paused at `19:15:20` waiting for trainer checkpoint 8 and resumed after `24.03s`.
- Trainer step 8 completed at `19:18:41`: `174.44s`, `6751 tokens/s`, `24.0%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 9 completed at `19:19:18`: `238.55s`, reward `0.4023`, mean seq length `5619.8`, async/off-policy `0/2`, and `1/256` repetition-flagged.
- Trainer step 9 completed at `19:22:10`: `205.08s`, `6784 tokens/s`, `24.1%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 10 completed at `19:23:17`: `238.43s`, reward `0.3906`, mean seq length `4921.7`, async/off-policy `0/1`, and `3/256` repetition-flagged.
- Trainer step 10 completed at `19:25:50`: `217.80s`, `6851 tokens/s`, `24.3%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 11 completed at `19:26:53`: `215.80s`, reward `0.3828`, mean seq length `5071.5`, async/off-policy `0/2`, and `3/256` repetition-flagged.
- Trainer step 11 completed at `19:29:24`: `211.01s`, `6747 tokens/s`, `24.0%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 12 completed at `19:30:21`: `207.28s`, reward `0.3945`, mean seq length `5268.7`, async/off-policy `0/2`, and `2/256` repetition-flagged.
- Trainer step 12 completed at `19:32:59`: `211.32s`, `6677 tokens/s`, `23.7%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 13 completed at `19:33:12`: `169.92s`, reward `0.4375`, mean seq length `4491.9`, async/off-policy `0/2`, and `1/256` repetition-flagged.
- Trainer step 13 completed at `19:35:29`: `146.47s`, `6643 tokens/s`, `23.6%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 14 completed at `19:36:39`: `207.49s`, reward `0.3633`, mean seq length `5819.7`, async/off-policy `0/2`, and `4/256` repetition-flagged.
- Trainer step 14 completed at `19:39:37`: `244.35s`, `6487 tokens/s`, `23.0%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 15 completed at `19:40:42`: `241.63s`, reward `0.3548`, mean seq length `6273.6`, async/off-policy `0/1`, and `2/256` repetition-flagged.
- Trainer step 15 completed at `19:43:53`: `253.45s`, `6613 tokens/s`, `23.5%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 16 completed at `19:45:21`: `279.30s`, reward `0.3789`, mean seq length `5724.9`, async/off-policy `0/1`.
- Trainer step 16 completed at `19:48:14`: `258.51s`, `6545 tokens/s`, `23.2%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 17 completed at `19:48:26`: `184.38s`, reward `0.3789`, mean seq length `5675.3`, async/off-policy `0/2`, and `2/256` repetition-flagged.
- Trainer step 17 completed at `19:51:14`: `176.89s`, `6517 tokens/s`, `23.1%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 18 completed at `19:53:03`: `276.33s`, reward `0.3906`, mean seq length `6667.5`, async/off-policy `0/1`.
- Trainer step 18 completed at `19:56:22`: `304.63s`, `6327 tokens/s`, `22.5%` MFU, peak mem `93.4 GiB`. This is the current slowest non-startup trainer step; it coincided with a long/high-token rollout step.
- Orchestrator step 19 completed at `19:57:01`: `237.28s`, reward `0.3711`, mean seq length `5888.4`, async/off-policy `0/1`, and `1/256` repetition-flagged.
- Orchestrator step 20 completed at `19:59:16`: `134.53s`, reward `0.3891`, mean seq length `5886.5`, async/off-policy `1/1`, and `2/256` repetition-flagged. It then paused at step 21 waiting for trainer checkpoint 20 and resumed after `45.05s`. This is a real small bubble: short-tail inference outran the 2-GPU trainer.
- Trainer step 19 completed at `20:00:01`: `215.72s`, `6458 tokens/s`, `22.9%` MFU, peak mem `93.4 GiB`.
- Trainer step 20 completed at `20:03:05`: `181.60s`, `6664 tokens/s`, `23.7%` MFU, peak mem `93.4 GiB`.
- Step 21 started rollouts after the pause at `20:00:04`. No eval is expected until step 50.
- Generated rollout JSONLs for steps 0-20 have exactly `256` rows each, i.e. `32` prompt groups/update with `rollouts_per_example=8`. Step 0-4 truncation counts were `12, 27, 27, 26, 20`; repetition flags were `1, 1, 1, 4, 0`; `is_filtered` was `0` in these JSONLs.
- Live GPU checks at `18:56-20:00`: allocation-level util stayed mostly `97-100%`, with occasional `87-92%` snapshots while trainer ranks were between phases. The `20:00:29` gpustat snapshot showed `99%` aggregate GPU util, total memory `1438677 / 1565936 MiB`, and trainer GPUs at `95865 / 97871` and `83943 / 97871` MiB. `batch_size=256` fits, but the trainer memory margin is narrow; keep watching for OOM as sequence lengths vary.
- Added vLLM `/metrics` sampler in tmux window `joanv_cc_4node:vllm-metrics`, writing `outputs/omni_math2_rlvr_canary/20260509_1825/dpo_default_14i2t_bs256_eval50_1000step/monitor/vllm_metrics.tsv`. Latest scrapes across all 14 servers showed `num_requests_waiting=0`, no preemptions, KV cache roughly `0.17-0.57`, and about `14-21` running requests/server. Early bottleneck is decode volume/tails plus trainer step cost, not vLLM queue capacity.
- Benchmark toggle is off: the resolved run config has `bench=false`. Current live trainer MFU has warmed into the `~22.5-24.5%` band after startup, with step 1-20 average `23.30%` MFU and `6565 tokens/s`; steps 5-20 average about `23.6%`. Orchestrator steps 1-20 average `215.73s` with two backpressure pauses totaling `69.08s`. Treat this as end-to-end live MFU, not isolated `--bench` MFU.
- No `OutOfMemory`, traceback, runtime error, inference failure, eval start, or checkpoint save observed in the default run logs as of `20:00 UTC`.
- 20:18 UTC refresh: trainer reached step 24; orchestrator reached step 25 and
  is generating step 26. Trainer step 24 was the best live MFU point so far:
  `165.58s`, `7162 tokens/s`, `25.4%` MFU, peak mem `93.4 GiB`.
- Aggregate trainer steps 1-24: mean step time `213.88s`, mean throughput
  `6620 tokens/s`, mean MFU `23.50%`, MFU range `20.0-25.4%`, peak memory
  `93.4 GiB`.
- Aggregate orchestrator steps 0-25: mean step time `218.09s`, mean reward
  `0.3986`, mean sequence length `5639.0` tokens/sample, max async level `1`,
  max off-policy level `2`.
- Rollout JSONLs through step 25 have exactly `256` rows each. The latest vLLM
  scrapes show all 14 servers OK, `0` queued requests, no preemptions, and no
  inference starvation signal. A `20:17:33` gpustat snapshot showed all 16 GPUs
  at `100%` util and total memory `1451401 / 1565936 MiB`.
- Backpressure through step 25: 3 pauses totaling `115.17s`, max `46.09s`.
  At the observed trainer mean, a full 1000-step Default arm alone is roughly
  `59.4h` before eval/checkpoint overhead. This active 24h allocation can
  validate the recipe and collect evals/checkpoints, but it should not be
  expected to finish both 1000-step sequential arms unless throughput changes a
  lot.
- 21:51 UTC refresh: Default reached ckpt/eval 50 and resumed training. Trainer
  step 50 completed at `21:43:59`: `254.63s`, `7476 tokens/s`, `26.5%` MFU,
  peak mem `92.9 GiB`. Trainer steps 20-50 averaged `196.88s`, `7473 tokens/s`,
  and `26.53%` MFU with peak mem `93.4 GiB`.
- Ckpt-50 eval completed at `21:47:54` in `485.12s`: `Avg@8=0.3638`,
  `Pass@1=0.3638`, `Pass@2=0.4818`, `Pass@4=0.5864`, `Pass@8=0.6700`,
  no-response `0.0%`, completion length `5794.74`, truncated `15.0%`.
- Post-eval restart did **not** look like the old cancel/refill cliff.
  Orchestrator step 51 reports `555.03s`, but that includes the `485.12s` eval;
  after eval, the training batch completed in about `66s` and had `216/256`
  rollouts immediately available. Therefore `cancel_inflight_rollouts_on_eval=false`
  appears to be doing the useful thing: it removes the huge refill bubble, while
  eval itself remains a periodic wall-clock tax.
- Trainer step 51 then completed at `21:52:10`: `253.07s`, `6472 tokens/s`,
  `23.0%` MFU, peak mem `92.9 GiB`; orchestrator step 52 completed at
  `21:52:18` in `193.49s`, async/off-policy `0/1`. The first trainer row after
  eval is lower-utilization than the warm pre-eval band, but the run did resume
  normally.
- GPU/MFU note: during eval, aggregate GPU util dropped to `81-87%` because the
  two trainer GPUs were idle/near-idle. During normal overlap, allocation util
  returns near full occupancy, but trainer MFU is only mid/high-20s. Do not
  conflate `100%` gpustat with high useful trainer FLOPs.
- Benchmark note: PrimeRL `bench=false`; do not turn on top-level `bench` for
  this run because it switches to fake/short benchmark semantics. vLLM logs show
  `benchmark_combo_kernel=True`, which is just inductor/kernel autotuning, not
  PrimeRL benchmark mode.
- 21:55 UTC utilization refresh: trainer steps 20-52 average `197.56s`,
  `7413 tokens/s`, `26.31%` MFU; latest trainer row is step 52 at `163.41s`,
  `6476 tokens/s`, `23.0%` MFU, peak mem `92.9 GiB`. Orchestrator step 53
  completed in `130.01s` with async/off-policy `1/1`; step 54 only paused
  `28.03s` waiting for checkpoint 53. Gpustat at `21:55:52` shows all 16 GPUs
  at `100%`, total memory `1451595 / 1565936 MiB`; trainer-node GPUs are at
  `95.8-96.9 / 97.9 GiB`, so memory is tight while useful trainer MFU remains
  mid-20s.
- 22:01 UTC continuation hardening: added
  `tmp/resume_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh`. Use this in
  a future 4-node allocation to continue the same Default→MaxRL pair. It adds
  `--ckpt.resume_step -1` and derives current allocation hosts from
  `SLURM_JOB_NODELIST`, overriding the temp configs' hardcoded launch hosts.
  `bash -n` passed; dry-runs validated the resume flag and JSON-list host
  override. Do not use top-level `bench` for this continuation.
- 22:02 UTC progress refresh: no OOM/traceback/error matches. Trainer has
  reached step 54 (`173.70s`, `6565 tokens/s`, `23.3%` MFU, peak mem
  `92.9 GiB`); orchestrator has reached step 55 (`169.59s`, reward `0.3246`,
  seq length `6756.7`, async/off-policy `1/1`).
- 22:04 UTC W&B event-file parse: trainer `time/wait_for_batch` is in the
  local `.wandb` event stream. Steps 20-54 average `196.67s` step time,
  `18.84s` wait-for-batch, `177.43s` forward/backward, `26.14%` logged MFU,
  `7363 tokens/s`; mean wait fraction is `8.25%`. Pre-eval steps 20-49 are
  `8.52%` wait fraction and `26.53%` MFU. Post-eval steps 51-54 are `7.87%`
  wait fraction but only `23.09%` logged MFU; this is probably partly
  `PerfCounter` rolling-window contamination from the eval/idle gap. Do not
  diagnose current mid-20s MFU as simple chronic batch starvation.
- 22:06 UTC progress/wait refresh: trainer has reached step 55 (`197.56s`,
  `6668 tokens/s`, `23.7%` MFU, peak mem `92.9 GiB`); orchestrator has reached
  step 57 (`196.01s`, reward `0.3871`, seq length `6576.9`, async/off-policy
  `1/1`). No error matches. Orchestrator is now waiting for trainer checkpoints
  after generation (`54.10s`, `60.14s`, `103.18s` for checkpoints 54-56), so
  the live bottleneck is trainer-side wall time under `max_async_level=1`.
- 22:08 UTC quantified refresh: trainer is now step 56 (`179.02s`,
  `6735 tokens/s`, `23.9%` MFU), orchestrator remains ahead at step 57. Trainer
  steps 51-56 average `192.84s` and `23.33%` logged MFU. W&B wait-for-batch
  rows for steps 51-55 average only `15.32s` wait on `195.61s` step time
  (`6.39%`), while orchestrator checkpoint waits after eval average `66.72s`
  over checkpoints 53-57. Latest vLLM scrape: all 14 servers OK, `253` running,
  `0` waiting, mean KV `0.442`, preemptions still `64`. Inference is not
  queueing; trainer progress is pacing the run.
- 22:11 UTC progress/disk refresh: trainer reached step 57 (`189.40s`,
  `6875 tokens/s`, `24.4%` MFU, peak mem `92.9 GiB`) and orchestrator reached
  step 58 (`207.32s`, reward `0.4570`, seq length `4811.7`, async/off-policy
  `1/1`), then resumed checkpoint 58 after `73.13s`. No error matches. Disk is
  fine: output dir `60G`, first full checkpoint `41G`, weights step 50 `14G`,
  run_default rollouts `2.1G` across `59` train batches, filesystem `195T`
  free. With `keep_last=4`, storage should not block this 1000-step arm.
- Allocation timing: job `4507057` runs from `2026-05-09T17:27:21` to
  `2026-05-10T17:27:21`. At `22:12 UTC`, eval-100 is estimated around
  `135min` away plus about `8min` eval overhead if ckpt-50 timing repeats, so
  this allocation should reach eval 100 but not complete the 1000-step
  Default→MaxRL sequence.
- 22:17 UTC utilization/config refresh: Default remains live. Trainer reached
  step 59 (`178.73s`, `7323 tokens/s`, `26.0%` MFU, peak mem `92.9 GiB`);
  orchestrator reached step 60 (`162.57s`, reward `0.3750`, seq length
  `6770.8`, async/off-policy `1/1`). No error matches. Config still has
  PrimeRL `bench=false`; do not enable it for this RLVR run. Latest gpustat:
  allocation util `89%`, memory `1452255 / 1565936 MiB`, 14 GPUs at `100%`,
  two trainer-node GPUs low-SM but memory-full (`96-97 GiB`). Latest vLLM scrape
  has 14/14 OK, `250` running, `0` waiting, mean KV `0.384`, preemptions still
  `64`. Post-eval checkpoint waits average `68.50s` over 8 samples, latest
  `80.14s`; inference queue capacity is not the active limiter.
- 22:20 UTC skill sync: `skills/config/SKILL.md` was updated to stop presenting
  the older `max_inflight_rollouts=3072` / eval-cancel-on speed probe as the
  default recommendation. It now records the checked-in `524288/768/off8` shape
  with eval cancellation off, the live fixed-batch `256/256` starting point, and
  the eval-cancel tradeoff.
- 22:23 UTC W&B trainer timing refresh: steps 52-61 average `179.72s` step time,
  only `1.43s` wait-for-batch (`0.79%`), `178.02s` forward/backward, `25.36%`
  MFU, and `7143 tok/s`. Steps 60-61 improved to `30.78-30.80%` MFU and
  `~8675 tok/s` while wait-for-batch stayed `~0.93s`. Orchestrator reached step
  62, post-eval checkpoint waits still average `67.79s`, and vLLM remains 14/14
  OK with `0` waiting. Current bottleneck is trainer compute/weight-update/
  checkpoint pacing, not inference batch supply.
- 22:25 UTC health refresh: trainer step 62 completed (`138.84s`,
  `8677 tok/s`, `30.8%` MFU, peak mem `92.9 GiB`), so steps 60-62 are a
  `30.8%` MFU streak. Orchestrator reached step 63 (`222.74s`, reward `0.3669`,
  seq length `6148.2`, async/off-policy `1/1`). One gpustat snapshot briefly
  showed a trainer GPU at low memory/util, but the next sample had all 16 GPUs
  at `100%` and trainer GPUs back to `97.2/81.2 GiB`; likely phase-local, not a
  crashed rank. Current run has `45` `Timeout during comparison` messages by
  step 63; prior tracing says these are math-verify per-sample comparison
  timeouts that return verifier-false. Treat as reward noise unless rate grows.
- 22:31 UTC steady-state refresh: trainer step 63 completed (`185.59s`,
  `8677 tok/s`, `30.8%` MFU, peak mem `92.9 GiB`), making steps 60-63 a
  four-step `30.8%` MFU plateau with mean `177.31s`. Orchestrator reached step
  65. Later live notes supersede this partial 22:31 refresh.

## Live 2026-05-10 Notes

- Default is still running in tmux window `joanv_cc_4node:omni-bs256-1000`.
  MaxRL has not started.
- At `12:14 UTC`, Default had reached trainer step 350 and orchestrator step
  352. Latest real checkpoint is `checkpoints/step_350/trainer` with
  `.metadata`, `__0_0.distcp`, and `__1_0.distcp`.
- Eval-350 completed in `374.34s`: `Avg@8=0.3650`, `Pass@1=0.3650`,
  `Pass@2=0.4789`, `Pass@4=0.5821`, `Pass@8=0.6800`, no-response `0.0%`,
  completion length `5850.65 ± 5218.24`, truncated `15.4%`. Versus eval-300,
  Pass@1/2/4/8 moved `-2.62/-2.50/-2.55/-2.00 pp`; versus eval-50 it moved
  `+0.12/-0.29/-0.43/+1.00 pp`.
- Trainer step 350 was `189.57s`, `8453 tok/s`, `30.0%` MFU, peak memory
  `92.9 GiB`. Steps 340-350 averaged `164.04s`, `8533 tok/s`, and `30.27%`
  MFU.
- The recurring eval-boundary bubble remains: checkpoint 350 wait was
  `161.32s`, eval took `374.34s`, and step 351 then reported `541.61s`
  wall-clock time with async/off-policy `0/2`.
- vLLM stayed healthy at the scrape after eval: 14/14 servers OK, `255`
  running, `0` waiting, mean KV `0.415`, preemptions stable at `116`.
  `gpustat` showed `94%` instantaneous aggregate GPU util and
  `1448598/1565936 MiB` allocated, but single gpustat frames remain
  phase-sensitive. Use logged trainer MFU for useful-FLOP claims.
- PrimeRL `bench` is still off. Do not turn on top-level `bench=true` for this
  canary; it changes run semantics. Normal trainer logs already emit real-run
  MFU.
- At `12:42 UTC`, post-eval-350 recovery was confirmed. Trainer steps 351-359
  were low-MFU cooldown (`27.0-27.6%`, `~7.6-7.8k tok/s`), then steps 360-361
  recovered to `30.6%` MFU and `8628 tok/s` average. Orchestrator off-policy
  peaked at `5` on step 357, then returned to `1` by steps 358-359 and `0` at
  step 360. vLLM still had `0` waiting and preemptions stable at `116`, so the
  repeated boundary tax is not a vLLM queue-capacity problem.
- At `13:14 UTC`, Default was trainer step 372 / orchestrator step 374.
  Recovered steady-state held: steps 360-372 averaged `30.40%` MFU and
  `8563 tok/s`; steps 370-372 averaged `30.27%` MFU. Checkpoint waits since
  360 averaged `148.47s` and vLLM still had `0` waiting. No eval-400 yet, no
  error/OOM matches. With the Slurm job ending `2026-05-10T17:27:21`, expect
  eval-400 and maybe eval-450 in this allocation, not the full 1000-step
  Default arm.
- At `13:55 UTC`, Default was trainer step 387 / orchestrator step 389.
  Recovered band remained stable: steps 380-387 averaged `30.56%` MFU and
  `8612 tok/s`; vLLM still had `0` waiting and preemptions stayed at `116`.
  Eval-400 had not started yet.
- At `14:36 UTC`, eval-400 completed and worsened again:
  `Avg@8=0.3525`, `Pass@1=0.3525`, `Pass@2=0.4571`, `Pass@4=0.5621`,
  `Pass@8=0.6600`, truncated `13.4%`. Versus eval-350, Pass@1/2/4/8 moved
  `-1.25/-2.18/-2.00/-2.00 pp`; versus eval-50, it moved
  `-1.13/-2.47/-2.43/-1.00 pp`. Runtime stayed healthy: steps 390-400 averaged
  `30.23%` MFU and `8516 tok/s`, vLLM had `0` waiting, and latest checkpoint is
  `checkpoints/step_400/trainer`. This strengthens the "quality/objective/noise
  problem, not infra failure" read.
- At `15:08 UTC`, post-eval-400 recovery was confirmed: trainer steps 401-409
  were the low-MFU band (`26.7-27.6%`, `7517-7765 tok/s`), then steps 410-412
  recovered to `30.67%` MFU and `8638 tok/s`. vLLM still had `0` waiting and no
  errors/OOMs matched.
- At `15:50 UTC`, Default was trainer step 428 / orchestrator step 430. Steps
  410-428 averaged `30.57%` MFU and `8612 tok/s`; steps 420-428 averaged
  `30.58%`. vLLM still had `0` waiting. Job ends `2026-05-10T17:27:21`;
  eval-450 still looks feasible, eval-500 does not.
- At `16:57 UTC`, eval-450 completed: `Avg@8=0.3837`, `Pass@1=0.3837`,
  `Pass@2=0.4904`, `Pass@4=0.5844`, `Pass@8=0.6700`, truncated `14.0%`.
  It bounced back from eval-400 (`+3.12/+3.33/+2.23/+1.00 pp` on
  Pass@1/2/4/8) but remained below eval-300 on Pass@2/4/8. Latest real resume
  point is `checkpoints/step_450/trainer` with `.metadata`, `__0_0.distcp`,
  and `__1_0.distcp`. Allocation has about 30 minutes left, so expect a
  resumable Default arm through eval-450, not Default completion or MaxRL start.
- Final scrape on 2026-05-11 showed the previous allocation reached trainer
  step 464 and orchestrator step 466 before walltime, but no checkpoint beyond
  step 450 exists. Last trainer row: step 464 at `17:26:40`, `8533 tok/s`,
  `30.3%` MFU. Last orchestrator row: step 466 at `17:26:43`, reward `0.3685`,
  async/off-policy `1/1`. Resume from `checkpoints/step_450/trainer`, not from
  the uncheckpointed final steps.
- Current allocation: Slurm job `4542540`, nodes
  `nid011162,nid011164,nid011183,nid011184`, ending `2026-05-12T11:14:24`.
  `20.02,28.03s` on the last two boundaries. vLLM is still 14/14 OK with `0`
  waiting; gpustat shows all 16 GPUs at `100%` and trainer GPUs at
  `95.9/96.9 GiB`. This may be a better warmed trainer regime, but do not
  extrapolate until it survives more steps and eval-100.
- 22:35 UTC sustained-warm refresh: trainer reached step 65. Steps 60-65 average
  `176.85s`, `8659 tok/s`, `30.73%` MFU, peak mem `92.9 GiB`; latest trainer
  step 65 is `185.03s`, `8618 tok/s`, `30.6%` MFU. Orchestrator reached step 66
  (`198.96s`, reward `0.4476`, seq length `6322.3`, async/off-policy `1/1`).
  Post-60 checkpoint waits remain noisy but lower than the immediate post-eval
  stall: `80.14,62.15,101.24,20.02,28.03,55.14,44.05s` (mean `55.82s`). vLLM is
  14/14 OK with `0` waiting. Gpustat again caught one low-util/low-memory
  trainer-GPU phase, but no OOM/Traceback/NCCL/failure signatures are present.
- 22:37 UTC W&B timing confirmation: complete W&B rows for trainer steps 60-65
  average `176.85s` step time, `0.93s` wait-for-batch (`0.52%`), `175.66s`
  forward/backward, `30.74%` MFU, and `8659 tok/s`. The warm plateau is not
  hidden trainer starvation. At this rate, trainer step/eval 100 is roughly
  `~100-115min` out from 22:36 UTC including expected eval overhead, assuming no
  regime change.
- 22:49 UTC slower-cadence refresh: trainer reached step 69; steps 60-69 average
  `178.41s`, `8629 tok/s`, `30.62%` MFU, with latest peak mem `93.4 GiB`. Warm
  MFU plateau persists. Checkpoint waits have climbed again: post-66 waits are
  `44.05,70.08,81.18,129.23,133.09s` (mean `91.53s`), while vLLM is still 14/14
  OK with `0` waiting and preemptions unchanged at `64`. Orchestrator reached
  step 71 and is ahead of trainer step 69, so the limiter is again trainer/
  checkpoint pacing, not inference supply. No OOM/Traceback/NCCL/failure
  signatures.
- 22:50 UTC monitor-run status file: created
  `outputs/omni_math2_rlvr_canary/20260509_1825/dpo_default_14i2t_bs256_eval50_1000step/STATUS.md`
  with a compact health snapshot. Latest trainer step 70 is `177.52s`,
  `8527 tok/s`, `30.3%` MFU, peak mem `93.4 GiB`; steps 60-70 average
  `178.33s`, `8619 tok/s`, `30.59%` MFU. vLLM remains 14/14 OK with `0`
  waiting.
- 23:27 UTC backpressure refresh: trainer reached step 83; steps 80-83 average
  `173.99s`, `8681 tok/s`, `30.83%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 84; steps 80-84 average `168.12s`, reward `0.4131`, seq length
  `5591.8`. Post-80 checkpoint waits are bad again:
  `84.17,138.22,154.27,156.31,127.26s` (mean `132.05s`), while vLLM remains
  14/14 OK with `0` waiting and preemptions unchanged at `64`. vLLM running
  dipped to `61` during the drained/waiting phase, then recovered to `247` after
  checkpoint 84 resumed. This is not inference-supply starvation; orchestrator is
  ahead and waiting on trainer/checkpoint progress. One recent orchestrator step
  hit max off-policy level `2`.
- 23:39 UTC backpressure follow-up: trainer reached step 87; steps 80-87 average
  `166.82s`, `8683 tok/s`, `30.83%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 89; steps 80-89 average `170.26s`, reward `0.4236`, seq length
  `5759.0`, async/off-policy usually `1/1` with step 82 at max off-policy `2`.
  The post-80 checkpoint-wait spike partially relaxed: after
  `84.17,138.22,154.27,156.31,127.26s`, the next waits were
  `58.09,60.11,30.03,57.06s`. vLLM remains 14/14 OK with `0` waiting. Still
  trainer/checkpoint paced, but not monotonically degrading before eval 100.
- 23:50 UTC pre-eval-100 refresh: trainer reached step 91; steps 80-91 average
  `172.50s`, `8690 tok/s`, `30.85%` MFU, peak mem `92.9 GiB`. Orchestrator
  reached step 93; steps 80-93 average `168.90s`, reward `0.4121`, seq length
  `5797.3`. Checkpoint backpressure is high again: post-80 waits now include
  `110.12,168.19,137.25,154.22s`, with post-80 mean `110.41s` and max
  `168.19s`. Recent orchestrator steps 90 and 93 hit max off-policy level `2`.
  vLLM is still 14/14 OK with `0` waiting and preemptions unchanged. Eval-100 has
  not started yet; interpret it with this trainer/checkpoint lag and staleness
  context.

## Live 2026-05-08 Notes

Active Slurm allocation:

- Job: `4481823`
- Nodes: `nid010815,nid010819,nid010840,nid010870`
- tmux session: `joanv_cc_4node`
- Default 10-step pane: `joanv_cc_4node:omni-12i4t-def` (finished)
- MaxRL 10-step pane: `joanv_cc_4node:omni-12i4t-max` (finished)
- Stopped stale 100-step pane: `joanv_cc_4node:omni-14i2t-def100b`
- Stopped 40-step default pane: `joanv_cc_4node:omni-14i2t-def40`
- Finished 100-step default pane: `joanv_cc_4node:omni-14i2t-def100c`
- Finished 100-step MaxRL pane: `joanv_cc_4node:omni-14i2t-max100`
- gpustat pane: `joanv_cc_4node:gpustat`

100-step sequential pair:

- Default config: `tmp/rl_olmo3_dpo_default_14i2t_off8_eval10_100step_20260508_1421.toml`
- MaxRL config: `tmp/rl_olmo3_dpo_maxrl_14i2t_off8_eval10_100step_20260508_1421.toml`
- Both dry-ran cleanly at `2026-05-08 14:24 UTC`.
- Shape: `max_steps=100`, `14 inference / 2 trainer`, `token_batch_size=524288`, `max_inflight_rollouts=3072`, train `rollouts_per_example=8`, `max_off_policy_steps=8`, `importance_ratio_clip=5.0`, eval `100x8`, eval interval `10`, `eval_base_model=false`, `cancel_inflight_rollouts_on_eval=true`, vLLM caps `gpu_memory_utilization=0.93`, `max_num_seqs=192`, `max_num_batched_tokens=65536`, `generation_config="vllm"`.
- Default output dir: `outputs/omni_math2_rlvr_canary/20260508_1421/dpo_default_14i2t_off8_eval10_100step`
- Default launch: `2026-05-08 14:26 UTC` in `joanv_cc_4node:omni-14i2t-def100c`.
- Default Slurm child steps at launch: `.29-.32`.
- Default W&B run id: `7868fdea82384313a6aad594ce61db92`.
- Startup check: all 14 inference servers reached `Application startup complete`; orchestrator logged `Inference pool ready` at `14:28:08` and started orchestrator step 0 at `14:28:11`.
- Early runtime: orchestrator step 0 `333.68s`, step 1 `149.55s`, step 2 `4.93s`, then expected `>1 step ahead` backpressure at step 3. Trainer warmed from step 1 `3429 tok/s`, `12.2%` MFU to step 9 `7172 tok/s`, `25.5%` MFU, peak mem `93.4 GiB`. This recovers to roughly the best partial 40-step default steady band before the eval boundary.
- Ckpt-10 boundary: orchestrator step 10 took `400.95s`, trainer step 10 took `340.00s`, `5606 tok/s`, `19.9%` MFU. The run cancelled `1485` old rollout requests at `14:45:52` and another `131` at `14:47:07`, so this is a boundary/refill tax rather than steady-state slowdown.
- Ckpt-10 eval: started at `14:51:19` on orchestrator step 11 after trainer checkpoint 10 was ready; completed at `14:58:47` in `447.17s` with `Avg@8=0.3625`, `Pass@1=0.3625`, `Pass@2=0.4786`, `Pass@4=0.5754`, `Pass@8=0.6400`, no-response `0.0%`, truncated `15.2%`. Compared to the partial 40-step default ckpt-10 (`0.3713/0.4807/0.5630/0.6300` for p@1/2/4/8), this is mixed and not convincing learning: p@4 is higher, p@8 is `+0.01` versus that 40-step run, but p@1/p@2 are lower.
- Eval scheduling note: in this codepath, ckpt-10 eval logged under `Starting orchestrator step 11`; step 10 had already generated a training batch while checkpoint 10 was becoming ready. Eval itself used ckpt_step 10, but the surrounding cancellation/refill behavior is ugly and should be considered when interpreting wall-clock throughput.
- Post-eval perf accounting note: trainer steps 13-16 returned to normal step wall time (`~66-71s`) but logged throughput/MFU stayed around `3.2k tok/s` / `11.5%`. This is because `PerfCounter` uses a rolling `time.perf_counter()` window and includes the trainer idle time during eval/refill. Treat low post-eval MFU as real end-to-end utilization loss, not evidence that one trainer rank died.
- Ckpt-20 eval: before eval, orchestrator step 20 reached max off-policy `8`, then `1396` old rollout requests were cancelled. Eval started at `15:15:27` and completed at `15:21:42` in `374.06s` with `Avg@8=0.3550`, `Pass@1=0.3550`, `Pass@2=0.4611`, `Pass@4=0.5520`, `Pass@8=0.6200`, no-response `0.0%`, truncated `16.5%`. This is worse than ckpt-10 across all pass@k and still not learning-looking.
- Ckpt-30 eval: before eval, orchestrator step 30 reached max off-policy `8`, then `1517` old rollout requests were cancelled. Eval started at `15:38:19` and completed at `15:45:10` in `410.22s` with `Avg@8=0.3650`, `Pass@1=0.3650`, `Pass@2=0.4718`, `Pass@4=0.5629`, `Pass@8=0.6300`, no-response `0.0%`, truncated `15.9%`. This rebounds from ckpt-20 but is still below ckpt-10 on p@2/p@4/p@8 (`0.4786/0.5754/0.6400`) and only barely above ckpt-10 on p@1 (`0.3650` vs `0.3625`). Decision at `15:47 UTC`: continue default 100 because allocation time is ample and this was explicitly promoted to a 100-step canary, but do not describe it as learning yet.
- Ckpt-30 speed note: the boundary paid another large cancellation/refill tax and eval hit `Timeout during comparison` twice while still completing. A `15:47` gpustat snapshot showed aggregate `87%` GPU util with trainer GPUs on `nid010870:g2,g3` at `0%` during the post-eval/refill region. The throughput problem is still the eval/cancel/refill rhythm, not failure of the 14 inference clients to stay busy during steady rollout generation.
- Ckpt-40 eval: before eval, orchestrator step 40 reached max off-policy `8`, then `1758` old rollout requests were cancelled. Eval started at `16:03:04` and completed at `16:09:16` in `371.90s` with `Avg@8=0.3738`, `Pass@1=0.3738`, `Pass@2=0.4854`, `Pass@4=0.5734`, `Pass@8=0.6400`, no-response `0.0%`, truncated `15.9%`. This is the first better-looking row: above ckpt-10 on p@1/p@2, tied on p@8, and just below on p@4. Continue default 100.
- Ckpt-40 speed note: trainer step 40 recovered to `7004 tok/s`, `24.9%` MFU, peak mem `93.4 GiB` right before eval. Immediately after eval, training rollout generation emitted repeated `Timeout during comparison` warnings during refill. Keep treating those as per-sample reward/CPU noise unless frequency grows enough to bias reward, but note the post-eval refill remains ugly.
- Ckpt-50 eval: before eval, orchestrator step 50 reached max off-policy `8`, then `1696` old rollout requests were cancelled. Eval started at `16:25:33` and completed at `16:31:55` in `381.88s` with `Avg@8=0.3825`, `Pass@1=0.3825`, `Pass@2=0.4925`, `Pass@4=0.5777`, `Pass@8=0.6500`, no-response `0.0%`, truncated `14.2%`. This is now above ckpt-10 on all four pass@k and back to the earlier base-reference p@8 level. Continue default 100; quality signal is no longer merely flat.
- Ckpt-50 speed note: trainer step 50 recovered to `6567 tok/s`, `23.3%` MFU, peak mem `93.4 GiB` right before eval. The eval/cancel/refill sawtooth remains the dominant wall-clock problem despite the improving eval metric.
- Ckpt-60 eval: before eval, orchestrator step 60 reached max off-policy `8`, then `1569` old rollout requests were cancelled. Eval started at `16:49:15` and completed at `16:56:34` in `438.14s` with `Avg@8=0.3925`, `Pass@1=0.3925`, `Pass@2=0.5118`, `Pass@4=0.6121`, `Pass@8=0.6900`, no-response `0.0%`, truncated `13.9%`. This is a clear upward row versus ckpt-10/20/30/40/50. Default 100 is canary-positive so far.
- Ckpt-60 speed note: trainer step 60 logged `5191 tok/s`, `18.4%` MFU, peak mem `93.4 GiB`, lower than the ckpt-40/50 boundary recoveries. The eval itself was slower (`438.14s`) but completed cleanly.
- Ckpt-70 eval: before eval, orchestrator step 70 reached max off-policy `8`, then `1881` old rollout requests were cancelled. Eval started at `17:12:51` and completed at `17:19:43` in `411.10s` with `Avg@8=0.3563`, `Pass@1=0.3563`, `Pass@2=0.4536`, `Pass@4=0.5379`, `Pass@8=0.6100`, no-response `0.0%`, truncated `17.4%`. This sharply regresses from ckpt-60 and is worse than ckpt-10 on all pass@k. Continue default 100 anyway because this is a 100-step canary and 100-example stochastic evals can swing, but stop calling the curve monotone.
- Ckpt-70 speed note: trainer step 70 recovered to `7121 tok/s`, `25.3%` MFU, peak mem `93.4 GiB` right before eval. Runtime remains stable; the issue is quality variance plus the recurring eval/cancel/refill tax.
- Ckpt-80 eval: before eval, orchestrator step 80 reached max off-policy `8`, then `1888` old rollout requests were cancelled. Eval started at `17:36:07` and completed at `17:42:21` in `373.52s` with `Avg@8=0.3713`, `Pass@1=0.3713`, `Pass@2=0.4886`, `Pass@4=0.5884`, `Pass@8=0.6700`, no-response `0.0%`, truncated `17.0%`. This partially rebounds from ckpt-70 and is above ckpt-10 on p@1/p@2/p@4/p@8, but not as strong as ckpt-60. Continue default 100.
- Ckpt-80 speed note: trainer step 80 recovered to `6856 tok/s`, `24.4%` MFU, peak mem `93.4 GiB` before eval. Eval hit one `Timeout during comparison` near 92% and completed cleanly.
- Ckpt-90 eval: before eval, orchestrator step 90 reached max off-policy `8`, then `1794` old rollout requests were cancelled. Eval started at `17:58:11` and completed at `18:04:13` in `361.62s` with `Avg@8=0.3613`, `Pass@1=0.3613`, `Pass@2=0.4729`, `Pass@4=0.5647`, `Pass@8=0.6400`, no-response `0.0%`, truncated `14.4%`. This fades from ckpt-80 and is far below ckpt-60; finish default 100 and report the full trajectory rather than best checkpoint only.
- Ckpt-90 speed note: trainer step 90 recovered to `6648 tok/s`, `23.6%` MFU, peak mem `93.4 GiB`. The eval completed quickly, but the run immediately paid another post-eval refill bubble: orchestrator step 91 took `673.97s` with async/off-policy reset to `0`.
- Default final eval: final eval started at `18:18:52` and completed at `18:49:42` in `1850.72s` with `Avg@8=0.3762`, `Pass@1=0.3762`, `Pass@2=0.4800`, `Pass@4=0.5630`, `Pass@8=0.6200`, no-response `0.0%`, truncated `14.4%`. This is a weak final row: above ckpt-90 on p@1/p@2, tied with ckpt-20 p@8, below ckpt-60/80 on p@4/p@8, and nowhere near the ckpt-60 spike.
- Default final speed note: final eval was pathological versus interval evals (`1850.72s` vs `~362-447s`). It had a multi-minute first-result stall while trainer final checkpoint/W&B shutdown was still happening, then long tail plus `Timeout during comparison` warnings. Treat final-eval wall-clock separately from steady interval eval throughput.
- Default cleanup: after orchestrator finished at `18:49:53`, inference child step `.31` exited but `.29`, `.30`, and `.32` hung after `STOP_INFERENCE`; those three only were cancelled with `scancel 4481823.29 4481823.30 4481823.32`. Allocation `.0`, `.batch`, and gpustat were left alive.
- MaxRL launch: launched `tmp/rl_olmo3_dpo_maxrl_14i2t_off8_eval10_100step_20260508_1421.toml` from tmux pane `joanv_cc_4node:omni-14i2t-max100`. W&B run id `35daa0e7deb24883b28c232a3ab4918e`; local W&B path `outputs/omni_math2_rlvr_canary/20260508_1421/dpo_maxrl_14i2t_off8_eval10_100step/run_default/wandb/run-20260508_185506-35daa0e7deb24883b28c232a3ab4918e`.
- MaxRL startup: child steps `.33-.36` launched. All 14 inference servers reached `Application startup complete`; orchestrator logged `Inference pool ready` at `18:56:06` and started step 0 at `18:56:09`. No `INFERENCE_FAILED` file at launch check.
- MaxRL early progress through ckpt-10: orchestrator completed step 0 at `19:01:38` (`327.52s`, reward `0.5000`, async/off-policy `0/0`), step 1 at `19:03:49` (`129.68s`, `0/1`), step 2 at `19:04:18` (`28.81s`, `1/1`), step 3 at `19:05:02` (`43.76s`, `1/2`), step 4 at `19:06:15` (`71.67s`, `1/3`), step 5 at `19:07:22` (`66.72s`, `1/4`), step 6 at `19:09:16` (`113.37s`, `1/5`), step 7 at `19:10:45` (`88.25s`, `1/6`), step 8 at `19:11:33` (`47.70s`, `1/7`), step 9 at `19:12:47` (`73.70s`, `1/8`), and step 10 at `19:17:01` (`252.74s`, `0/8`). Trainer warmed from step 1 `3813 tok/s`, `13.5%` MFU to step 9 `7265 tok/s`, `25.8%` MFU, peak mem `93.4 GiB`; trainer step 10 took `237.82s`, `6293 tok/s`, `22.3%` MFU.
- MaxRL ckpt-10 boundary/eval: before eval, checkpoint 10 triggered cancellations of `1145` and `135` old rollout requests. Eval started at `19:17:02` and completed at `19:23:45` in `403.02s` with `Avg@8=0.3675`, `Pass@1=0.3675`, `Pass@2=0.4807`, `Pass@4=0.5764`, `Pass@8=0.6400`, no-response `0.0%`, truncated `15.6%`. This is slightly above default 100 ckpt-10 on p@1/p@2/p@4 and tied on p@8, but still well inside the noisy 100-prompt band. After eval, the run resumed training and printed `Timeout during comparison` during refill; no `INFERENCE_FAILED` file was present and Slurm steps `.33-.36` remained alive.
- MaxRL post-ckpt-10 refill: orchestrator step 11 took `669.76s` with async/off-policy reset to `0/0`, then steps 12-20 completed as `122.37s`, `74.14s`, `28.01s`, `39.57s`, `67.67s`, `75.05s`, `65.81s`, `73.77s`, `66.84s`. This repeats the eval/cancel/refill bubble seen in default, not a launch failure.
- MaxRL trainer steps 11-20: trainer step 11 took `327.80s`, `3756 tok/s`, `13.3%` MFU. Steps 12-18 logged only `~3550-3603 tok/s`, `12.6-12.8%` MFU despite normal step wall times, then step 19 rose to `4092 tok/s`, `14.5%` and step 20 recovered to `7035 tok/s`, `25.0%` MFU. Do not overstate this as permanent MaxRL trainer slowdown; it is a long post-eval utilization/accounting recovery window, with full recovery by the ckpt-20 boundary.
- MaxRL ckpt-20 boundary/eval: before eval, orchestrator step 20 reached max off-policy `8`, then `1637` old rollout requests were cancelled. Eval started at `19:39:39` and completed at `19:45:53` in `373.17s`, with `Timeout during comparison` at about 68% and 94%, and metrics `Avg@8=0.3550`, `Pass@1=0.3550`, `Pass@2=0.4668`, `Pass@4=0.5697`, `Pass@8=0.6500`, no-response `0.0%`, truncated `16.4%`. Versus default 100 ckpt-20, this is tied on p@1, better on p@2/p@4/p@8 (`0.4668/0.5697/0.6500` vs `0.4611/0.5520/0.6200`). Versus MaxRL ckpt-10, p@1/p@2/p@4 moved down and p@8 moved up. The run resumed after eval and is generating step-21 training rollouts.
- MaxRL post-ckpt-20 refill: orchestrator step 21 took `744.61s` with async/off-policy reset to `0/0`; steps 22-30 then completed as `143.36s`, `38.70s`, `28.66s`, `70.82s`, `73.70s`, `69.83s`, `67.85s`, `81.25s`, `61.20s`. Trainer was low for almost the entire interval: step 21 `4100 tok/s`, `14.6%` MFU; steps 22-29 stayed around `3880-4026 tok/s`, `13.8-14.3%` MFU; only step 30 recovered to `7108 tok/s`, `25.2%` MFU. Accurate framing: MaxRL has a long low-utilization post-eval stretch and recovers only near the next checkpoint boundary.
- MaxRL ckpt-30 boundary/eval: before eval, orchestrator step 30 reached max off-policy `8`, then `1698` old rollout requests were cancelled. Eval started at `20:02:52` and completed at `20:09:05` in `372.36s` with `Avg@8=0.3613`, `Pass@1=0.3613`, `Pass@2=0.4579`, `Pass@4=0.5413`, `Pass@8=0.6100`, no-response `0.0%`, truncated `14.6%`. This is worse than default 100 ckpt-30 on every pass@k (`0.3650/0.4718/0.5629/0.6300`) and worse than MaxRL ckpt-20 except p@1. Current evidence is against MaxRL quality and wall-clock, but keep running to 100 for the matched trajectory unless the user decides to stop early.
- MaxRL post-ckpt-30 refill: orchestrator step 31 took `768.28s`; steps 32-40 then completed as `109.20s`, `48.59s`, `23.70s`, `68.79s`, `69.96s`, `81.92s`, `72.76s`, `99.91s`, `37.37s`. Trainer repeated the same pattern: step 31 `4118 tok/s`, `14.6%` MFU; steps 32-39 stayed around `3957-4096 tok/s`, `14.1-14.5%` MFU; step 40 recovered to `7568 tok/s`, `26.9%` MFU. This regularizes the diagnosis: each eval boundary resets utilization badly, most of the following interval is low-MFU, and recovery occurs only at the next checkpoint boundary.
- MaxRL ckpt-40 boundary/eval: before eval, orchestrator step 40 reached max off-policy `8`, then `1673` old rollout requests were cancelled. Eval started at `20:25:55` and completed at `20:31:53` in `357.77s` with `Avg@8=0.3688`, `Pass@1=0.3688`, `Pass@2=0.4764`, `Pass@4=0.5737`, `Pass@8=0.6500`, no-response `0.0%`, truncated `15.6%`. Versus default 100 ckpt-40 (`0.3738/0.4854/0.5734/0.6400`), MaxRL is lower on p@1/p@2, effectively tied on p@4, and `+0.01` on p@8. Quality is mixed at ckpt-40 rather than dominated, but wall-clock/utilization remains worse.
- MaxRL post-ckpt-40 refill: orchestrator step 41 took `699.40s`; steps 42-50 then completed as `150.08s`, `62.78s`, `12.93s`, `70.99s`, `67.81s`, `67.07s`, `74.09s`, `68.89s`, `71.91s`. Trainer step 41 was `4402 tok/s`, `15.6%` MFU; steps 42-49 stayed around `4094-4146 tok/s`, `14.5-14.7%` MFU; step 50 recovered only partially to `6740 tok/s`, `23.9%` MFU. Same eval-boundary utilization tax, with a weaker checkpoint recovery than ckpt40.
- MaxRL ckpt-50 boundary/eval: before eval, orchestrator step 50 reached max off-policy `8`, then `1566` old rollout requests were cancelled. Eval started at `20:48:32` and completed at `20:54:50` in `377.73s` with `Avg@8=0.3700`, `Pass@1=0.3700`, `Pass@2=0.4764`, `Pass@4=0.5766`, `Pass@8=0.6600`, no-response `0.0%`, truncated `17.5%`. Versus default 100 ckpt-50 (`0.3825/0.4925/0.5777/0.6500`), MaxRL is lower on p@1/p@2/p@4 and `+0.01` on p@8. Quality is not collapsed, but MaxRL still loses the low-k eval and runtime story at the midpoint.
- MaxRL post-ckpt-50 refill: orchestrator step 51 took `733.37s`; steps 52-60 then completed as `172.99s`, `30.32s`, `37.77s`, `68.01s`, `74.88s`, `74.05s`, `72.94s`, `71.76s`, `67.04s`. Trainer step 51 was `4199 tok/s`, `14.9%` MFU; steps 52-59 stayed around `3857-4054 tok/s`, `13.7-14.4%` MFU; step 60 recovered to `6641 tok/s`, `23.6%` MFU. Same sawtooth again: most of the interval is trainer-limited and the checkpoint boundary recovers too late to help end-to-end utilization.
- MaxRL ckpt-60 boundary/eval: before eval, orchestrator step 60 reached max off-policy `8`, then `1461` old rollout requests were cancelled. Eval started at `21:12:07` and completed at `21:18:21` in `373.68s` with `Avg@8=0.3750`, `Pass@1=0.3750`, `Pass@2=0.4825`, `Pass@4=0.5783`, `Pass@8=0.6500`, no-response `0.0%`, truncated `15.8%`. Versus default 100 ckpt-60 (`0.3925/0.5118/0.6121/0.6900`), MaxRL is worse across pass@k. This is not catching the default ckpt-60 spike.
- MaxRL post-ckpt-60 refill: orchestrator step 61 took `682.87s`; steps 62-70 then completed as `153.59s`, `76.77s`, `25.73s`, `46.75s`, `67.87s`, `69.65s`, `70.96s`, `72.31s`, `75.00s`. Trainer step 61 was `4485 tok/s`, `15.9%` MFU; steps 62-69 stayed around `4111-4280 tok/s`, `14.6-15.2%` MFU; step 70 recovered to `6951 tok/s`, `24.7%` MFU. This is the cleanest sawtooth evidence so far: a long refill step, low interval MFU, and a checkpoint-boundary recovery that looks good only if cherry-picked.
- MaxRL ckpt-70 boundary/eval: before eval, orchestrator step 70 reached max off-policy `8`, then `1693` old rollout requests were cancelled. Eval started at `21:34:30` and completed at `21:40:35` in `365.04s` with `Avg@8=0.3588`, `Pass@1=0.3588`, `Pass@2=0.4654`, `Pass@4=0.5694`, `Pass@8=0.6600`, no-response `0.0%`, truncated `14.9%`. Versus default 100 ckpt-70 (`0.3563/0.4536/0.5379/0.6100`), MaxRL is better at this particular weak default checkpoint, especially p@8. Do not overread this: it is still far below default ckpt-60 and below default ckpt-80 on p@2/p@4/p@8.
- MaxRL post-ckpt-70 refill: orchestrator step 71 took `701.75s`; steps 72-80 then completed as `78.36s`, `104.25s`, `40.24s`, `29.07s`, `65.77s`, `69.62s`, `68.73s`, `65.70s`, `76.74s`. Trainer step 71 was `4286 tok/s`, `15.2%` MFU; steps 72-79 stayed around `4083-4292 tok/s`, `14.5-15.2%` MFU; step 80 recovered to `7427 tok/s`, `26.4%` MFU. Same sawtooth, with the strongest checkpoint-boundary MFU so far.
- MaxRL ckpt-80 boundary/eval: before eval, orchestrator step 80 reached max off-policy `8`, then `1828` old rollout requests were cancelled. Eval started at `21:56:16` and completed at `22:02:27` in `370.19s` with `Avg@8=0.3750`, `Pass@1=0.3750`, `Pass@2=0.4925`, `Pass@4=0.6027`, `Pass@8=0.6900`, no-response `0.0%`, truncated `15.5%`. This is MaxRL's best row so far. It beats default 100 ckpt-80 (`0.3713/0.4886/0.5884/0.6700`) and matches default ckpt-60 p@8, but remains below default ckpt-60 on p@1/p@2/p@4 (`0.3925/0.5118/0.6121`).
- MaxRL post-ckpt-80 refill: orchestrator step 81 took `695.62s`; steps 82-90 then completed as `145.43s`, `82.66s`, `5.77s`, `66.04s`, `70.86s`, `68.08s`, `66.07s`, `73.04s`, `72.01s`. Trainer step 81 was `4183 tok/s`, `14.9%` MFU; steps 82-89 stayed around `3996-4120 tok/s`, `14.2-14.6%` MFU; step 90 recovered to `6781 tok/s`, `24.1%` MFU. The eval/refill sawtooth persisted even after the strong ckpt-80 quality row.
- MaxRL ckpt-90 boundary/eval: before eval, orchestrator step 90 reached max off-policy `8`, then `1893` old rollout requests were cancelled. Eval started at `22:18:51` and completed at `22:25:02` in `370.84s` with `Avg@8=0.3650`, `Pass@1=0.3650`, `Pass@2=0.4664`, `Pass@4=0.5600`, `Pass@8=0.6400`, no-response `0.0%`, truncated `16.6%`. This falls back sharply from ckpt-80. Versus default 100 ckpt-90 (`0.3613/0.4729/0.5647/0.6400`), MaxRL is slightly higher on p@1, tied on p@8, and lower on p@2/p@4.
- MaxRL post-ckpt-90 refill/final train steps: orchestrator step 91 took `689.16s`; steps 92-99 then completed as `152.84s`, `83.80s`, `3.95s`, `68.91s`, `66.79s`, `64.95s`, `71.88s`, `68.78s`. Trainer step 91 was `4366 tok/s`, `15.5%` MFU; steps 92-99 stayed around `3998-4255 tok/s`, `14.2-15.1%` MFU. Trainer finished at `22:42:03` with peak memory `93.4 GiB`.
- MaxRL final eval: final eval started at `22:39:00` and completed at `23:07:44` in `1723.97s` with `Avg@8=0.3825`, `Pass@1=0.3825`, `Pass@2=0.4839`, `Pass@4=0.5597`, `Pass@8=0.6100`, no-response `0.0%`, truncated `12.4%`. Versus default final (`0.3762/0.4800/0.5630/0.6200`), MaxRL is slightly better on p@1/p@2, worse on p@4/p@8. Versus MaxRL ckpt-80 (`0.3750/0.4925/0.6027/0.6900`), final loses the high-k gain badly.
- MaxRL final speed note: final eval was also pathological versus interval evals (`1723.97s` vs `~358-403s`), but this was not an idle-GPU bubble. A gpustat snapshot during final eval showed all 14 inference GPUs at `100%` while the two trainer GPUs were idle/free. The slow path is long/straggly eval generation plus W&B/final logging, not failed inference utilization.
- MaxRL cleanup: after orchestrator finished at `23:07:53`, child step `.34` exited cleanly, but `.33`, `.35`, and `.36` hung after `STOP_INFERENCE`; those three only were cancelled with `scancel 4481823.33 4481823.35 4481823.36`. Verified afterward that only allocation `.0` and `.batch` remained.
- 100-step pair conclusion: default has the single best row at ckpt-60 on p@1/p@2/p@4 and ties MaxRL's best p@8 (`0.6900`). MaxRL has a stronger ckpt-80 than default ckpt-80 and a slightly better final p@1/p@2, but it does not retain high-k performance and is not faster. The evidence does not justify promoting MaxRL as-is; the next speed work should target the eval/cancel/refill sawtooth, final-eval tail, `generation_config`/stop behavior, and trainer utilization rather than simply increasing inference pressure.

Stopped partial 40-step default:

- Default config: `tmp/rl_olmo3_dpo_default_14i2t_off8_eval10_40step_20260508_1254.toml`
- MaxRL config: `tmp/rl_olmo3_dpo_maxrl_14i2t_off8_eval10_40step_20260508_1254.toml`
- Both dry-ran cleanly at `2026-05-08 12:55 UTC`.
- Shape: `max_steps=40`, `14 inference / 2 trainer`, `token_batch_size=524288`, `max_inflight_rollouts=3072`, train `rollouts_per_example=8`, `max_off_policy_steps=8`, `importance_ratio_clip=5.0`, eval `100x8`, eval interval `10`, `eval_base_model=false`, `cancel_inflight_rollouts_on_eval=true`, vLLM caps `gpu_memory_utilization=0.93`, `max_num_seqs=192`, `max_num_batched_tokens=65536`, `generation_config="vllm"`.
- Default output dir: `outputs/omni_math2_rlvr_canary/20260508_1254/dpo_default_14i2t_off8_eval10_40step`
- Default launch: `2026-05-08 12:56 UTC` in `joanv_cc_4node:omni-14i2t-def40`.
- Default Slurm child steps at launch: `.25-.28`.
- Default W&B run id: `21f7f3992771411398c39c6e2312a493`.
- Default ckpt-10 eval completed at `13:29:43`: `452.86s`, `Avg@8=0.3713`, `Pass@1=0.3713`, `Pass@2=0.4807`, `Pass@4=0.5630`, `Pass@8=0.6300`, no-response `0.0%`, truncated `15.5%`. This is below the earlier default base-eval reference (`0.3787/0.4868/0.5834/0.6500` for p@1/2/4/8), so do not call it learning yet.
- Default ckpt-20 eval completed at `13:53:25`: `366.63s`, `Avg@8=0.3625`, `Pass@1=0.3625`, `Pass@2=0.4625`, `Pass@4=0.5434`, `Pass@8=0.6000`, no-response `0.0%`, truncated `16.9%`. This is worse than ckpt-10 across all pass@k and below the earlier base-eval reference; the default 40-step trajectory is negative through 20 steps.
- Default ckpt-30 eval completed at `14:16:40`: `381.07s`, `Avg@8=0.3650`, `Pass@1=0.3650`, `Pass@2=0.4743`, `Pass@4=0.5711`, `Pass@8=0.6500`, no-response `0.0%`, truncated `16.2%`. This partially rebounds from ckpt-20, especially at p@8, but low-k remains below ckpt-10 and below the earlier base-eval reference. Call this "no convincing learning yet", not monotone collapse.
- Post-ckpt-20 refill was expensive: orchestrator step 21 took `730.03s` after eval cancellation/resetting off-policy to 0; trainer step 21 took `371.09s` at `4238 tok/s`, `15.1%` MFU. Orchestrator step 22 partly recovered to `107.31s`. This supports treating `cancel_inflight_rollouts_on_eval=true` plus `max_off_policy_steps=8` as a real throughput cost, not just log noise.
- Ckpt-30 boundary also paid a cancellation/refill cost: `1556` old rollout requests were cancelled before eval. Trainer step 30 itself was fast (`7383 tok/s`, `26.2%` MFU), so trainer efficiency can recover right before eval; the biggest throughput hit is the refill after cancellation, not a permanent trainer slowdown.
- Stopped intentionally at step 32 around `14:24 UTC` after the user asked to graduate to 100-step canaries. The final ckpt-40 eval was not run.
- Eval semantics note: trainer steps are zero-indexed. Trainer step 9 completed and was broadcast before the ckpt-10 eval; trainer step 10 completed during eval, and its weight update was held until after eval. So ckpt-10 was post-10-updates and not mid-eval contaminated.

Topology conclusion:

- Treat `14 inference / 2 trainer` as the active topology unless new trainer-scaling evidence appears. The topology data now favors `14i/2t` over `12i/4t` more strongly than the earlier BOTEC: `12i/4t` showed about `50%` wait fraction vs `14i/2t` at about `38%`, about `2.7x` lower compute utilization, slower eval, and worse eval direction for both `12i/4t` runs. The conclusion is not that 4 trainer GPUs can never work; it is that the current trainer path does not scale well enough to justify spending four GPUs on it for these canaries.

Stopped stale 100-step default:

- Config: `tmp/rl_olmo3_dpo_default_14i2t_off32_finaleval_100step_20260508_1228.toml`
- Output dir: `outputs/omni_math2_rlvr_canary/20260508_1228/dpo_default_14i2t_off32_finaleval_100step`
- W&B run id: `7e1098c4c8024784943b8b7d112f4d67`
- It skipped base eval and reached orchestrator step 9 / trainer step 7 before intentional stop.
- Observed trainer-side bottleneck: post-warmup trainer steps climbed to `~7.4k tok/s`, `~26%` MFU, peak mem `93.4 GiB`; orchestrator repeatedly paused waiting for trainer checkpoints. This argues against pushing inference harder without improving learner throughput or allowing more staleness.
- Observed max off-policy reached `8` by orchestrator step 9. That is exactly why the new canaries use `max_off_policy_steps=8`; the old `32` cap was not needed for the observed early pipeline and is out-of-family with upstream recipes.

Default 10-step canary:

- Config: `tmp/rl_olmo3_dpo_default_12i4t_off16_10step_20260508_1015.toml`
- Output dir: `outputs/omni_math2_rlvr_canary/20260508_1015/dpo_default_12i4t_off16_10step`
- W&B run id: `e8c5fb289e9541cc9e4fcb794391292a`
- Topology: 12 inference GPUs on `nid010815,nid010819,nid010840`; 4 trainer GPUs on `nid010870`.
- Train shape: `max_steps=10`, `token_batch_size=524288`, `max_inflight_rollouts=1536`, `max_off_policy_steps=16`, train `rollouts_per_example=8`, eval `100x8`, vLLM caps `gpu_memory_utilization=0.93`, `max_num_seqs=192`, `max_num_batched_tokens=65536`.
- Step-0 eval: `525.32s`, `Avg@8=0.3787`, `Pass@1=0.3787`, `Pass@2=0.4868`, `Pass@4=0.5834`, `Pass@8=0.6500`, truncated `16.0%`.
- Final eval after 10 training steps: `1080.33s`, `Avg@8=0.3800`, `Pass@1=0.3800`, `Pass@2=0.4886`, `Pass@4=0.5724`, `Pass@8=0.6400`, truncated `16.1%`.
- Eval interpretation: no meaningful uplift. Avg@8/Pass@1/Pass@2 rose by about `0.001-0.002`, while Pass@4/Pass@8 fell by about `0.01`. On only 100 eval prompts, this is noise, not evidence of learning.
- Trainer throughput steps 3-9: `7860, 8619, 7381, 7317, 7119, 7266, 7489 tok/s`; MFU mostly `12.6-15.3%` across 4 trainer GPUs.
- Topology interpretation: `12i/4t` did not produce a useful learner scaling win over prior `14i/2t` total throughput, and it made eval slower by using fewer inference GPUs. Larger trainer allocation is not currently justified unless the trainer path is fixed.
- Operational note: after default finished, inference Slurm steps `.4-.6` hung after `STOP_INFERENCE`; they were terminated with `scancel 4481823.4 4481823.5 4481823.6` before launching MaxRL.

MaxRL 10-step canary:

- Config: `tmp/rl_olmo3_dpo_maxrl_12i4t_off16_10step_20260508_1015.toml`
- Output dir: `outputs/omni_math2_rlvr_canary/20260508_1015/dpo_maxrl_12i4t_off16_10step`
- W&B run id: `42921698b2a548fcb08af66db954e8d4`
- Topology: 12 inference GPUs on `nid010815,nid010819,nid010840`; 4 trainer GPUs on `nid010870`.
- Step-0 eval: `583.68s`, `Avg@8=0.3688`, `Pass@1=0.3688`, `Pass@2=0.4857`, `Pass@4=0.5910`, `Pass@8=0.6800`, truncated `17.2%`.
- Final eval after 10 training steps: `1311.52s`, `Avg@8=0.3525`, `Pass@1=0.3525`, `Pass@2=0.4564`, `Pass@4=0.5391`, `Pass@8=0.6100`, truncated `15.9%`.
- Eval interpretation: worse than init; paired deltas from `tmp/analyze_20260508_12i4t_canaries.py` were `p1=-0.0163±0.0303`, `p2=-0.0293±0.0390`, `p4=-0.0519±0.0517`, `p8=-0.0700±0.0796`.
- Trainer throughput post-step0: mean `6424 tok/s`, mean MFU `11.4%`, last `6360 tok/s`; slower than default `12i/4t`.
- Topology interpretation: MaxRL did not rescue the `12i/4t` topology. It was slower and eval got worse.
- Operational note: after MaxRL finished, inference Slurm steps `.8-.10` hung after shutdown; they were terminated with `scancel 4481823.8 4481823.9 4481823.10`.

Obsolete off32 100-step sequential pair:

- Superseded by the live `off8/eval10/100step` configs above. Keep this as provenance only.
- Chosen topology: `14 inference / 2 trainer`, because `12i/4t` did not improve learner throughput enough and made eval slower.
- Default config: `tmp/rl_olmo3_dpo_default_14i2t_off32_100step_20260508_1205.toml`
- MaxRL config: `tmp/rl_olmo3_dpo_maxrl_14i2t_off32_100step_20260508_1205.toml`
- Both dry-runed cleanly with explicit hosts at `2026-05-08 12:08 UTC`.
- Default output dir: `outputs/omni_math2_rlvr_canary/20260508_1205/dpo_default_14i2t_off32_100step`
- Default W&B run id: `79b482b6a91b4b09a890610db8d6c134`
- Default launched at about `2026-05-08 12:09 UTC` in `joanv_cc_4node:omni-14i2t-def100`.
- Default Slurm child steps at launch: `.11-.14`.
- Shape: `max_steps=100`, `token_batch_size=524288`, `max_inflight_rollouts=3072`, `max_off_policy_steps=32`, train `rollouts_per_example=8`, eval `100x8`, `cancel_inflight_rollouts_on_eval=true`, vLLM caps `gpu_memory_utilization=0.93`, `max_num_seqs=192`, `max_num_batched_tokens=65536`, `generation_config="vllm"`.
- Next checks: initial eval time/metrics, step 1-10 trainer throughput and off-policy levels, whether eval shutdown again leaves inference Slurm steps alive, and whether final/interval evals show any movement beyond noise.

## Live 2026-05-07 Notes

Active Slurm allocation:

- Job: `4473657`
- Nodes: `nid011107,nid011114,nid011131,nid011132`
- tmux session: `joanv_cc_4node`
- Active run pane: none. `joanv_cc_4node:omni-default` and `joanv_cc_4node:omni-maxrl` are idle shells after intentional Ctrl-C stops.
- gpustat pane: `joanv_cc_4node:gpustat`

15:06 UTC update:

- MaxRL OOM-guard eval-refill retry output dir: `outputs/omni_math2_rlvr_canary/20260507_1449/dpo_maxrl_pipelinerl_speed_evalrefill_oomguard_retry`.
- W&B run id: `e1fefe2478ed4194963058aeed955ad9`.
- Config shape confirmed before launch: `importance_ratio_clip=5.0`, `token_batch_size=524288`, `max_inflight_rollouts=3072`, train `rollouts_per_example=8`, `max_off_policy_steps=32`, eval `num_examples=100`, eval `rollouts_per_example=8`, eval `max_concurrent_rollouts_per_client=48`, vLLM caps `gpu_memory_utilization=0.93`, `max_num_seqs=192`, `max_num_batched_tokens=65536`, `generation_config="vllm"`.
- Initial eval completed at `15:00:53`: `467.30s`, `Avg@8=0.3812`, `Pass@8=0.6400`, `No-response=0.0%`, completion length `5930.34`, truncated `16.6%`.
- This initial eval is faster than earlier MaxRL initial evals on the same 14 inference / 2 trainer topology: fixedcaps `506.19s`, genauto `480.99s`. It is slower than the default OOM-guard retry initial eval `449.24s`.
- Orchestrator step 0 completed at `15:05:33`: `747.29s`, reward `0.5000`, seq length `2451.5`, async `0`, max off-policy `0`. This includes initial eval plus first rollout fill; do not compare it to steady-state train steps.
- Step 0 rollout fill generated `529514` tokens against target `524288` in about `4:36` and logged one `Timeout during comparison`.
- Latest GPU snapshot at `15:06:06`: all 16 GPUs at `100%`, aggregate `100%`; overlap is active after startup fill.
- Slurm child steps `.67-.70` are alive on the four nodes; allocation `4473657` remains active.

15:10 UTC update:

- Trainer step 0 completed at `15:06:31`: `805.68s`, loss `-0.0041`, mismatch KL `0.4380`, peak mem `81.2 GiB`. Throughput/MFU logged as `0 tokens/s` / `0.0%`; this is startup accounting noise.
- Normal weight update after trainer step 0 succeeded: orchestrator paused inference at `15:06:31`, all engines paused immediately, and all engines resumed at `15:06:35`.
- Orchestrator step 1 completed at `15:08:00`: `146.22s`, reward `0.4231`, seq length `5192.0`, async `0`, max off-policy `1`.
- Orchestrator step 2 completed at `15:09:04`: `63.55s`, reward `0.4000`, seq length `7133.1`, async `1`, max off-policy `1`.
- Trainer step 1 completed at `15:09:09`: `154.85s`, `3471 tok/s`, `12.3%` MFU, peak mem `93.4 GiB`. This is slow versus the default OOM-guard retry's later `~7.3k tok/s` / `~26%` MFU, but earlier MaxRL runs also climbed after step 1.
- Orchestrator briefly backpressured at step 3 waiting for trainer checkpoint 2, then resumed after `5.00s`.
- Admin retry patch was exercised and recovered: pause requests to `nid011107:8001`, `8002`, `8003`, and `nid011114:8004` hit `RemoteProtocolError('Server disconnected without sending a response.')` on attempt 1/5 at `15:09:10`; all engines paused by `15:09:13` and resumed by `15:09:17`.
- Orchestrator step 3 completed at `15:09:17`: `11.77s`, reward `0.2812`, seq length `8954.2`, async `1`, max off-policy `2`; this mostly consumed already-inflight rollouts after the weight update.
- Latest GPU snapshot at `15:10:13`: all 16 GPUs at `100%`, aggregate `100%`.

15:12 UTC update:

- Trainer step 2 completed at `15:10:28`: `71.89s`, `4777 tok/s`, `17.0%` MFU, peak mem `93.4 GiB`.
- Trainer step 3 completed at `15:11:45`: `69.20s`, `5577 tok/s`, `19.8%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 4 completed at `15:10:37`: `79.13s`, reward `0.4167`, seq length `5522.5`, async `1`, max off-policy `3`.
- Orchestrator step 5 completed at `15:12:12`: `94.42s`, reward `0.3125`, seq length `6572.1`, async `1`, max off-policy `4`.
- Backpressure remains visible: orchestrator waited `72.31s` for checkpoint 3 and `69.28s` for checkpoint 4, then paused again at `15:12:12` waiting for checkpoint 5. Inference can outrun trainer; trainer throughput is the current speed limiter.
- Later admin updates after the retried `15:09:10` update were clean: checkpoint 3 update paused at `15:10:30` and resumed `15:10:36`; checkpoint 4 update paused `15:11:47` and resumed `15:11:50`.
- Latest GPU snapshot at `15:12:37`: all 16 GPUs at `100%`, aggregate `100%`.

15:14 UTC update:

- Trainer step 4 completed at `15:12:56`: `65.84s`, `5986 tok/s`, `21.3%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 6 completed at `15:13:55`: `102.18s`, reward `0.4375`, seq length `4791.5`, async `1`, max off-policy `5`.
- Orchestrator waited `46.43s` for checkpoint 5, then paused again at `15:13:55` waiting for checkpoint 6. The system remains trainer/backpressure limited.
- Latest GPU snapshot at `15:14:00`: aggregate `96%`; all inference GPUs `100%`, trainer GPUs `98%` and `46%`.

15:15 UTC update:

- Trainer step 5 completed at `15:14:07`: `64.65s`, `6260 tok/s`, `22.2%` MFU, peak mem `93.4 GiB`.
- Trainer step 6 completed at `15:15:16`: `65.77s`, `6478 tok/s`, `23.0%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 7 completed at `15:15:14`: `78.43s`, reward `0.4083`, seq length `4435.6`, async `1`, max off-policy `6`.
- Normal updates stayed healthy: checkpoint 6 update paused `15:14:07` and resumed `15:14:11`; checkpoint 7 update paused `15:15:17` and resumed `15:15:21`.
- Trainer throughput is improving but still below the default OOM-guard retry late steps (`7352-7666 tok/s`, `26.1-27.2%` MFU on steps 9-10). Next critical gate is step-10 eval survival and tail latency.
- Latest GPU snapshot at `15:15:22`: aggregate `97%`, all inference GPUs `100%`, trainer GPUs `100%` and `62%`.

15:16 UTC update:

- Trainer step 7 completed at `15:16:30`: `68.79s`, `6608 tok/s`, `23.5%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 8 completed at `15:16:07`: `53.16s`, reward `0.4000`, seq length `7367.5`, async `1`, max off-policy `7`.
- Checkpoint 8 update paused at `15:16:30` and resumed at `15:16:34`; no retry warning.
- Latest GPU snapshot at `15:16:40`: all 16 GPUs at `100%`, aggregate `100%`.
- Step-10 eval should trigger soon after checkpoint 10 is ready; key checks are whether trainer-side NCCL coordination avoids the old rank timeout and whether eval tail latency is still ~28min.

15:18 UTC update:

- Trainer step 8 completed at `15:17:47`: `72.42s`, `6780 tok/s`, `24.1%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 9 completed at `15:16:49`: `41.18s`, reward `0.3611`, seq length `8015.3`, async `1`, max off-policy `8`.
- Orchestrator step 10 completed at `15:17:56`: `62.85s`, reward `0.2750`, seq length `6657.4`, async `1`, max off-policy `9`.
- Step-10 eval has not started yet; after orchestrator step 10 the run paused at `15:17:56` waiting for trainer checkpoint 10. The critical eval boundary is still pending.
- Latest GPU snapshot at `15:18:13`: all 16 GPUs at `100%`, aggregate `100%`.

Prior default run:

- Config: `configs/omni_math2/rl_olmo3_dpo_default_pipelinerl_speed.toml`
- Output dir: `outputs/omni_math2_rlvr_canary/20260507_0809/dpo_default_pipelinerl_speed_fixedcaps`
- W&B run id: `179d153e48ee49bc9370f9734511356d`
- Topology: node0-node2 inference GPUs `[0,1,2,3]`; node3 inference GPUs `[0,1]`; node3 trainer GPUs `[2,3]`. Total: 14 inference + 2 trainer.
- vLLM caps that booted: `gpu_memory_utilization=0.93`, `max_num_seqs=192`, `max_num_batched_tokens=131072`. The more aggressive `0.95/256/262144` shape OOMed during EngineCore warmup.

Code/config changes made for this phase:

- Added `importance_ratio_clip` support for `default` and `is_reinforce` trainer losses, with raw/clipped ratio metrics.
- Set PipelineRL-speed configs to `importance_ratio_clip = 5.0`.
- Added a narrow logging filter in `src/prime_rl/__init__.py` to suppress the exact Transformers hub-kernels warning. Current live run started before that patch, so it still prints the warning; future launches should not.
- Updated `skills/config/SKILL.md` with the PipelineRL/IS, vLLM caps, and warning-filter lessons.

Current speed evidence:

- Step-0 eval completed: `483.24s` for `100x8`, versus old 2-node initial eval around `938.94s`. Eval is about `1.9x` faster.
- Orchestrator step 1: `113.78s`, seq length `5111.2`, async `0`, max off-policy `1`.
- Orchestrator step 2: `42.88s`, seq length `7205.9`, async `1`, max off-policy `1`.
- Orchestrator step 3: `90.49s`, seq length `5120.0`, async `1`, max off-policy `2`; includes a `24.03s` pause waiting for trainer checkpoint 2.
- Orchestrator step 4: `109.39s`, seq length `5346.6`, async `0`, max off-policy `4`.
- Orchestrator step 5: `95.61s`, seq length `5055.5`, async `0`, max off-policy `5`.
- Orchestrator step 6: `60.97s`, seq length `5115.5`, async `1`, max off-policy `5`.
- Orchestrator step 7: `70.86s`, seq length `5917.7`, async `1`, max off-policy `6`.
- Orchestrator step 8: `42.80s`, seq length `6860.6`, async `1`, max off-policy `7`.
- Orchestrator step 9: `62.38s`, seq length `8542.0`, async `1`, max off-policy `8`.
- At `08:35:10`, step 10 started and paused waiting for trainer checkpoint 9.
- Trainer step 1: `121.89s`, `4307 tok/s`, `15.3%` MFU.
- Trainer step 2: `67.32s`, `5651 tok/s`, `20.0%` MFU.
- Trainer step 3: `62.01s`, `6316 tok/s`, `22.4%` MFU.
- Trainer step 4: `111.33s`, `6139 tok/s`, `21.8%` MFU.
- Trainer step 5: `76.93s`, `6240 tok/s`, `22.1%` MFU.
- Trainer step 6: `68.81s`, `6488 tok/s`, `23.0%` MFU.
- Trainer step 7: `67.71s`, `6679 tok/s`, `23.7%` MFU.
- gpustat snapshot at `08:30:04` showed all 16 GPUs at `100%` utilization.
- gpustat snapshot at `08:34:16` showed `98%` aggregate GPU utilization.

Interpretation as of 08:35 UTC: eval/inference got much faster, and post-warmup trainer MFU is improving. Old 2-node default orchestrator steps 2-10 had rough median `~151s`; current 4-node default steps 1-9 are `114,43,90,109,96,61,71,43,62s`, so there is a real step-time win. It is not close to a `2x` training wall-clock win because the orchestrator is now capable of outrunning the trainer and hits `max_async_level=1` backpressure. More inference GPUs alone no longer helps unless we loosen staleness, increase/amortize training token batch size, or speed the trainer path.

More precise early-run comparison from logs:

- Current 4-node default orchestrator mean, steps 1-4: `89.13s`.
- Prior 2-node default orchestrator mean, steps 1-10: `153.77s`; current is `1.73x` faster.
- Prior 2-node MaxRL orchestrator mean, steps 1-10: `172.56s`; current is `1.94x` faster.
- Current trainer mean, steps 1-3: `83.74s`, `5425 tok/s`, `19.2%` MFU.
- Prior 2-node default trainer mean, steps 1-10: `150.00s`, `3905 tok/s`, `13.8%` MFU.
- Prior 2-node MaxRL trainer mean, steps 1-10: `167.70s`, `3744 tok/s`, `13.2%` MFU.
- Caveat: current sample is still early and small; treat speedup as real but preliminary.

Known live noise:

- `Timeout during comparison` comes from `.venv/lib/python3.12/site-packages/math_verify/grader.py` inside `verify()`. It catches `TimeoutException`, logs the warning, returns `False`, and the affected candidate receives a failed symbolic-verifier path. Treat as per-sample reward noise / CPU waste, not run-fatal, unless frequency gets high enough to bias reward.
- Current live run still emits `kernels hub usage is disabled...` because it predates the suppressor patch.

08:40 UTC update:

- Step 10 did complete: orchestrator step 10 took `88.96s`; trainer step 10 took `68.01s`, `7389 tok/s`, `26.2%` MFU.
- The step-10 eval was not skipped. It fired at `08:38:26` as `Running evals at ckpt_step=10`, after trainer checkpoint 10 became ready and after orchestrator step 11 completed. This is because eval scheduling uses `ckpt_step`, not the just-logged orchestrator step number.
- Live eval is `100x8` as intended. Do not reduce sample count or group size.
- During blocking eval, the 14 inference GPUs are hot while the 2 trainer GPUs on `nid011132:g2,g3` are idle. Snapshot at `08:40:00`: 14/16 GPUs at `100%`, aggregate `87%`. This is not an inference bubble; it is blocking eval intentionally pausing training.
- Caveat: this default step-10 eval is likely weight-contaminated. Trainer step 11 completed at `08:39:45`, and the live old-code orchestrator logged `Pausing inference engines for weight update` inside the `ckpt_step=10` eval at `08:39:46`. Treat the resulting eval metric as "post-10-ish", not a clean ckpt-10 eval.
- Fix applied for future launches: `Scheduler.pause_policy_updates()` now cancels the background policy-update poller and waits out any already-started update before blocking online eval. The orchestrator calls it before eval. Verified with `uv run --no-sync pytest tests/unit/orchestrator/test_scheduler.py::test_pause_policy_updates_cancels_poller_and_waits_for_inflight_update tests/unit/orchestrator/test_scheduler.py::test_stop_cancels_inflight_policy_update_task -q` (`2 passed`).
- After this eval completes, kill only the live default run steps/pane, keep the Slurm allocation, then launch the MaxRL PipelineRL-speed config sequentially in tmux.

08:45 UTC update:

- Default run did not finish the step-10 eval. It reached about `197/800` eval completions, then stopped with `Inference failure observed; stopping orchestrator and trainer`.
- Pane/trace showed `Inference server manager on nid011107 observed pid 81620 exit with status 0`, `STOP_INFERENCE` on the other nodes, and the launcher raised `RuntimeError: gpu_layout run failed with exit codes [0, 137, 137, 1]`.
- Slurm allocation stayed alive. Old default run steps `.26-.29` exited; only allocation shell plus gpustat step remained.
- Launched MaxRL sequentially in tmux pane `joanv_cc_4node:omni-maxrl` using patched code:
  - Config: `configs/omni_math2/rl_olmo3_dpo_maxrl_pipelinerl_speed.toml`
  - Output dir: `outputs/omni_math2_rlvr_canary/20260507_0846/dpo_maxrl_pipelinerl_speed_fixedcaps`
  - W&B name: `olmo3-dpo-maxrl-pipelinerl-speed-20260507-0846`
  - W&B run id: `9e8690371a0143bca01eb0a8b271ea25`
  - Slurm steps up: `.31-.34` on the four nodes.
- MaxRL reached initial eval at `08:47:05`: `Running evals at ckpt_step=0 for omni-math2-baseline100`, still `100x8`.
- All 14 inference GPUs were hot at `08:48:50` while trainer GPUs idled during blocking eval. This is expected for initial eval. The patched code has not yet reached the step-10 eval boundary where the mid-eval weight-update fix matters.
- Exact hub-kernels warning is absent from the new MaxRL logs so far.

08:56 UTC update:

- MaxRL initial eval completed cleanly at `08:55:31`: `Evaluated omni-math2-baseline100 in 506.19s`.
- Metrics: `Avg@8=0.3600`, `Pass@1=0.3600`, `Pass@2=0.4696`, `Pass@4=0.5593`, `Pass@8=0.6400`, `No-response=0.0%`, completion length `5842.09`, truncated `15.5%`.
- The run then began train rollout generation. The next important gate is the first few trainer/orchestrator steps, then the patched step-10 eval boundary.
- GPU snapshot at `08:55:41` showed the expected eval shape: 14 inference GPUs at `100%`, trainer GPUs idle, aggregate `87%`.

09:02 UTC update:

- MaxRL completed orchestrator step 0 at `09:01:17`: `Step 0 | Time: 850.78s | Reward: 0.5400 | Seq. Length: 2864.3 tokens/sample | Async Level: 0 | Max. Off-Policy Level: 0`.
- Step 0 includes initial eval plus the first full rollout-fill, so it is not comparable to steady-state train steps.
- The first training rollout fill took about `5m42s` after eval, generated `572851` tokens against the `524288` token target, and logged one `Timeout during comparison`.
- Step 1 began immediately at `09:01:17`. A gpustat snapshot at `09:01:56` showed all 16 GPUs at `100%`: inference was already generating the next batch while the trainer GPUs were active. That means the pipeline overlap is working after the first-batch startup fill.

09:04 UTC update:

- Trainer step 0 completed at `09:02:17`: `Time: 912.01s`, `Loss: -0.0078`, `Mismatch KL: 0.4886`, `Grad. Norm: 2.0312`, `Peak Mem.: 81.1 GiB`. The reported `Throughput: 0 tokens/s | MFU: 0.0%` is startup-accounting noise, not representative.
- During orchestrator step 1, inference paused for a weight update at `09:02:17` and resumed at `09:02:21`; this was a normal training update, not the eval-contamination case.
- Orchestrator step 1 completed at `09:02:54`: `Time: 96.42s`, reward `0.5179`, seq length `4816.7`, async `0`, max off-policy `1`.
- Orchestrator step 2 completed at `09:03:20`: `Time: 25.36s`, reward `0.5000`, seq length `6776.9`, async `1`, max off-policy `1`.
- At `09:03:20`, orchestrator step 3 paused waiting for trainer checkpoint 2: `Orchestrator paused: waiting for trainer process to complete checkpoint 2 (>1 step(s) ahead). Training is progressing normally.` This is the expected `max_async_level=1` backpressure after inference outruns trainer.
- GPU snapshot at `09:03:45` showed all inference GPUs at `100%` and trainer GPUs at `14-16%`, aggregate `89%`. The system is no longer in the startup fill bubble, but trainer backpressure is visible.

09:06 UTC update:

- Trainer post-warmup-ish steps:
  - Step 1 at `09:04:03`: `101.85s`, `5211 tokens/s`, `18.5%` MFU, peak mem `93.4 GiB`.
  - Step 2 at `09:05:10`: `63.17s`, `6536 tokens/s`, `23.2%` MFU, peak mem `93.4 GiB`.
- Orchestrator step 3 completed at `09:04:19`: `58.69s`, reward `0.2727`, seq length `6001.2`, async `1`, max off-policy `2`.
- Orchestrator step 4 then paused waiting for trainer checkpoint 3, and a normal weight update started at `09:05:10`.
- Current interpretation: eval throughput is clearly improved by the 14/2 topology, and overlap works after startup. Trainer throughput is not yet obviously better than the default run's later `~7.3k tok/s` peak; more data needed before claiming a MaxRL speed win.

09:10 UTC update:

- Trainer steps now show steady-state-ish throughput:
  - Step 3 at `09:06:18`: `64.24s`, `6963 tokens/s`, `24.7%` MFU.
  - Step 4 at `09:07:29`: `67.19s`, `7259 tokens/s`, `25.7%` MFU.
  - Step 5 at `09:08:53`: `78.98s`, `7122 tokens/s`, `25.2%` MFU.
- Orchestrator:
  - Step 4 at `09:06:04`: `104.48s`, seq length `5719.2`, async `1`, max off-policy `3`.
  - Step 5 at `09:07:48`: `103.19s`, seq length `3728.9`, async `0`, max off-policy `5`.
  - Step 6 at `09:09:28`: `100.18s`, seq length `5520.4`, async `0`, max off-policy `6`.
- The run is healthy and still trainer/backpressure limited. MaxRL trainer throughput is roughly matching the better default late-step range so far, not clearly exceeding it.

09:18 UTC update:

- Trainer:
  - Step 6 at `09:10:39`: `101.96s`, `6790 tokens/s`, `24.1%` MFU.
  - Step 7 at `09:11:47`: `64.01s`, `6943 tokens/s`, `24.6%` MFU.
  - Step 8 at `09:12:56`: `63.77s`, `7062 tokens/s`, `25.0%` MFU.
  - Step 9 at `09:14:03`: `63.64s`, `7168 tokens/s`, `25.4%` MFU.
  - Step 10 at `09:15:37`: `63.06s`, `7297 tokens/s`, `25.9%` MFU.
- Orchestrator:
  - Step 7 at `09:10:33`: `63.93s`, async `1`, max off-policy `6`.
  - Step 8 at `09:11:37`: `63.69s`, async `1`, max off-policy `7`.
  - Step 9 at `09:12:57`: `79.11s`, async `1`, max off-policy `8`.
  - Step 10 at `09:13:40`: `41.97s`, async `1`, max off-policy `9`.
  - Step 11 at `09:15:33`: `111.82s`, async `1`, max off-policy `10`.
- Step-10 eval started at `09:15:34`: `Running evals at ckpt_step=10 for omni-math2-baseline100`, `Evaluating ... (num_examples=100, rollouts_per_example=8)`.
- As of this note, no `Pausing inference engines for weight update` has appeared after the `09:15:34` eval start. Keep checking until eval completion.

09:21 UTC update:

- Step-10 eval is still running and is slow/straggly: latest log progress reached `5/800` after `4:41`.
- This does not look like an idle bubble at the GPU level. Snapshot at `09:20:09`: all 14 inference GPUs were at `100%`; trainer GPU `nid011132:g2` was idle and `g3` showed `100%` util at low power. Aggregate GPU util was `93%`.
- Still no `Pausing inference engines for weight update` after the `09:15:34` eval start. Last observed pause was the normal pre-eval update at `09:14:04`, followed by resume at `09:14:08`.
- Interpretation: current blocker is eval tail latency / long completions, not the old mid-eval policy-update contamination. Keep monitoring for completion or inference failure before killing anything.

09:24 UTC update:

- Step-10 eval remains much slower than the initial eval. At `09:23:56`, progress was only `38/800` after `8:24`; the initial MaxRL eval had completed `800/800` in `8:26`.
- Two `Timeout during comparison` messages appeared inside this eval tail. That is verifier noise after completions arrive, not proof of an inference crash.
- GPU snapshot at `09:23:56` still showed all 14 inference GPUs at `100%`, with trainer `g2` idle and `g3` at `100%` low power; aggregate `93%`.
- Working hypothesis: checkpoint-10 MaxRL is producing much longer eval completions / worse tail latency. The topology is not bubbling, but eval wall-clock is bad.

09:26 UTC update:

- Step-10 eval was `58/800` at `9:44` elapsed. A naive linear extrapolation is about `2h15m`, though this is noisy because eval progress is straggler-dominated.
- GPU snapshot at `09:25:24` was unchanged: 14 inference GPUs at `100%`; trainer `g2` idle, trainer `g3` at `100%` low power; aggregate `93%`.
- If this keeps going at the same rate, the likely next judgment call is whether to abort the eval and treat this as a failed/too-expensive MaxRL step-10 eval, not as a scheduling bubble.

09:32 UTC update:

- MaxRL run failed before the step-10 eval completed. Eval reached about `97/800` after `11:11`, then inference was stopped.
- Trainer root cause: rank 1 hit NCCL watchdog timeout at `09:25:37` in `NCCLWeightBroadcastSender._resolve_dtensors` / `DTensor.full_tensor()`: `WorkNCCL(SeqNum=15703, OpType=COALESCED, NumelIn=205369344, NumelOut=410738688, Timeout(ms)=600000)`.
- Mechanism: the orchestrator correctly paused policy updates for blocking eval, so rank 0 of the trainer waited for inference `NCCL_READY` for the next broadcast. Non-master trainer rank 1 did not wait and entered DTensor/FSDP collectives; rank 0 was not participating, so rank 1 timed out after 600s and aborted.
- This validates the earlier no-contamination patch only partially: no post-`09:15:34` orchestrator weight-update log appeared, but the trainer still attempted the next broadcast and died because rank coordination was incomplete.
- Patch applied: `src/prime_rl/trainer/rl/broadcast/nccl.py` now uses a trainer-side `TRAINER_NCCL_READY` filesystem marker. Master trainer rank touches it only after inference `NCCL_READY`; non-master ranks wait for it before entering `nccl_broadcast_sender.broadcast_weights()`.
- Added `tests/unit/train/rl/test_nccl_broadcast_coordination.py`; verified with `uv run --no-sync pytest tests/unit/train/rl/test_nccl_broadcast_coordination.py -q` (`2 passed`).
- Updated `skills/config/SKILL.md` with the blocking-eval/NCCL-rank coordination lesson.
- Slurm allocation remains alive: only allocation shell, gpustat step, and batch step remain. GPU snapshot after failure showed `0%` util and near-zero memory on all 16 GPUs.
- Do not rerun MaxRL/default until the user chooses whether to spend allocation time on another full `100x8` step-10 eval despite the observed checkpoint-10 MaxRL eval tail-latency blowup.

09:42 UTC update:

- Created temp smoke config `tmp/rl_olmo3_dpo_default_genauto_smoke_20260507.toml` to test only `generation_config = "auto"` on a short, explicitly labeled run. It keeps eval at `100x8` and train rollout group size `8`; it is not a default-vs-MaxRL comparison run.
- First genauto smoke launch used output dir `outputs/omni_math2_rlvr_canary/20260507_0935/dpo_default_genauto_smoke` and failed during vLLM startup before eval/training evidence. The actual root cause was not the genauto axis: multiple inference servers failed in Torch/Inductor compile/autotune with `torch._inductor.exc.InductorError: RuntimeError: Failed to run autotuning code block: [Errno 116] Stale file handle`. Example: `logs/inference/server_10.log`.
- Patched `src/prime_rl/templates/gpu_layout_rl.sh.j2` so each one-GPU inference server gets an isolated node-local compile cache root under `/tmp`, exporting per-server `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `XDG_CACHE_HOME`.
- Updated `skills/config/SKILL.md` with the shared-cache stale-file-handle failure mode.
- Verified render/syntax with:
  `source .env && PYTHONPATH=/lus/lfs1aip2/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/verifiers:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/environments/omni_math2_singleturn uv run --no-sync rl @ tmp/rl_olmo3_dpo_default_genauto_smoke_20260507.toml --dry-run --output-dir tmp/dryrun_olmo3_genauto_smoke_cachefix_20260507`
  and `bash -n tmp/dryrun_olmo3_genauto_smoke_cachefix_20260507/gpu_layout_rl.sh`.
- Relaunched the patched smoke in tmux pane `joanv_cc_4node:omni-default`:
  - Output dir: `outputs/omni_math2_rlvr_canary/20260507_0941/dpo_default_genauto_smoke_cachefix`
  - W&B run id: `e908816d91f34af5b5271dbd0e3f7d05`
  - Slurm steps: `.39-.42` on `nid011131,nid011107,nid011132,nid011114`
  - At `09:41:33`, all four steps were still alive and GPUs had model memory loaded; the run had passed the prior quick stale-file-handle failure point but had not yet reached eval.
  - At `09:42:32`, all 14 inference servers reached `Application startup complete` and the orchestrator logged `Inference pool ready`. Logs confirmed per-server vLLM compile caches under `/tmp/prime-rl-vllm-cache-joanv.a6r-4473657-<server_idx>/...`.
  - The vLLM warning `Default vLLM sampling parameters have been overridden by the model's generation_config.json` is expected in this smoke because `generation_config = "auto"` is the tested axis.
  - At `09:43:05`, initial `100x8` eval started. GPU snapshot at `09:43:28` showed the expected blocking-eval shape: 14 inference GPUs at `100%`, trainer GPUs idle, aggregate `87%`.
  - At `09:44:38`, inference remained busy with no waiting queue on sampled servers. Server logs showed many completed `/v1/chat/completions` requests and active stragglers; the orchestrator file progress line had not advanced visibly, likely because tqdm carriage returns were not flushing cleanly to the log.
  - At `09:46:01`, eval progress was `168/800` after `2:52`. Linear extrapolation is roughly `13.7m`, slower than the earlier `generation_config="vllm"` initial evals (`483s` default, `506s` MaxRL), though this may still change with the tail.
  - At `09:46:36`, eval progress was `202/800` after `3:31` and all 14 inference GPUs were still at `100%`. This is not a scheduling bubble; `generation_config="auto"` is producing slower eval traffic so far.
  - Correction at `09:47:38`: the middle bulk caught up sharply. Eval progress was `644/800` after `4:31`, so the early linear extrapolation was too pessimistic; final wall-clock depends on the last-tail behavior.

09:50 UTC update:

- Patched genauto smoke completed initial eval cleanly at `09:50:13`: `Evaluated omni-math2-baseline100 in 427.50s`.
- Metrics: `Avg@8=0.3738`, `Pass@1=0.3738`, `Pass@2=0.4789`, `Pass@4=0.5639`, `Pass@8=0.6200`, `No-response=0.0%`, completion length `5484.69` (range `[104.00, 15360.00]`), truncated `14.0%`.
- This is faster than both earlier `generation_config="vllm"` initial evals on the same 14i/2t topology: default `483.24s`, MaxRL `506.19s`. Do not overread one eval; this smoke tests finish behavior and truncation, not learning.
- After eval, the run entered train rollout fill: latest observed progress was `23968/262144` tokens at about `00:25` elapsed. Continue monitoring for first trainer steps and for recurrence of the blocking-eval/NCCL issue at the step-10 boundary.

09:55 UTC update:

- Orchestrator step 0 completed at `09:52:39`: `573.88s`, reward `0.5114`, seq length `3022.9`, async `0`, max off-policy `0`. This includes initial eval plus first rollout fill, so do not compare it to steady-state train steps.
- Trainer step 0 completed at `09:53:10`: `604.80s`, loss `-0.0021`, mismatch KL `0.4739`, peak mem `80.9 GiB`; throughput/MFU were logged as `0 tokens/s` / `0.0%`, which is startup accounting noise.
- A normal weight update during orchestrator step 1 paused inference at `09:53:11` and resumed at `09:53:14`.
- Orchestrator step 1 completed at `09:55:05`: `145.81s`, reward `0.4479`, seq length `3025.5`, async `0`, max off-policy `1`.
- Step 2 began immediately and was already at `83666/262144` rollout tokens after about `9s`. GPU snapshot at `09:55:18` showed 14 inference GPUs at `100%` and trainer GPUs at `16%`, aggregate `89%`.
- Need the trainer step 1 timing before making any statement about the genauto smoke's training throughput.

09:56 UTC update:

- Trainer step 1 completed at `09:55:41`: `147.01s`, `1938 tokens/s`, `6.9%` MFU, peak mem `93.4 GiB`. This is the first non-startup trainer point but still includes heavy first-overlap/fill effects; do not treat as steady state.
- Orchestrator step 2 completed at `09:55:41`: `35.83s`, reward `0.4750`, seq length `6700.1`, async `1`, max off-policy `1`.
- Orchestrator step 3 completed at `09:56:12`: `30.03s`, reward `0.5500`, seq length `7410.0`, async `1`, max off-policy `2`.
- Trainer step 2 completed at `09:56:19`: `33.23s`, `3111 tokens/s`, `11.0%` MFU, peak mem `93.3 GiB`.
- Orchestrator step 4 briefly paused waiting for trainer checkpoint 3, then resumed after `7.01s`; a normal weight update paused inference at `09:56:19` and resumed at `09:56:23`.
- GPU snapshot at `09:56:48` showed all 16 GPUs at `100%`, aggregate `100%`. Pipeline overlap is now real; remaining question is whether trainer throughput climbs toward the prior `~7k tok/s` range.

09:59 UTC update:

- Trainer continued climbing but is still below the earlier PipelineRL-speed configs:
  - Step 3 at `09:56:58`: `35.62s`, `4019 tokens/s`, `14.2%` MFU.
  - Step 4 at `09:57:45`: `43.32s`, `4650 tokens/s`, `16.5%` MFU.
  - Step 5 at `09:58:23`: `34.17s`, `5011 tokens/s`, `17.7%` MFU.
- Orchestrator:
  - Step 4 at `09:56:48`: `35.89s`, seq length `8024.4`, async `1`, max off-policy `3`.
  - Step 5 at `09:57:02`: `13.56s`, seq length `7100.6`, async `1`, max off-policy `4`.
  - Step 6 at `09:59:46`: `163.31s`, seq length `2383.4`, async `0`, max off-policy `4`.
- Step 6 slowdown was driven by cancellation/refill under the smoke's tight old off-policy cap: `Cancelled 64 old rollout requests` at `09:57:49`, then `Cancelled 97 old rollout requests` at `09:58:28`.
- Interpretation: this smoke proves genauto can start/eval/train and that overlap becomes real, but its training speed is not comparable to the main PipelineRL-speed configs because it uses the old small `token_batch_size=262144`, `max_inflight_rollouts=384`, and `max_off_policy_steps=4`.

10:13 UTC update:

- Patched `generation_config="auto"` smoke completed cleanly:
  - Output dir: `outputs/omni_math2_rlvr_canary/20260507_0941/dpo_default_genauto_smoke_cachefix`
  - W&B run id: `e908816d91f34af5b5271dbd0e3f7d05`
  - Orchestrator finished at `10:13:14`; `STOP_INFERENCE` was observed and inference servers shut down normally.
- Final `100x8` eval completed at `10:13:05`: `572.30s`, `Avg@8=0.3663`, `Pass@1=0.3663`, `Pass@2=0.4861`, `Pass@4=0.5929`, `Pass@8=0.6700`, `No-response=0.0%`, completion length `5771.65` (range `[104.00, 15360.00]`), truncated `14.5%`.
- Initial eval for the same run was `427.50s`, `Avg@8=0.3738`, `Pass@8=0.6200`, truncated `14.0%`. So genauto did not create an obvious truncation/no-response problem in this smoke; it also did not prove a learning improvement.
- Trainer finished cleanly at `10:04:36` and wrote final checkpoint before eval finished. No recurrence of the rank-1 NCCL timeout that killed the MaxRL run. The trainer-side `TRAINER_NCCL_READY` coordination patch passed this smoke's step-10/final-eval boundary.
- Trainer step timings after startup:
  - Step 1: `147.01s`, `1938 tok/s`, `6.9%` MFU.
  - Step 2: `33.23s`, `3111 tok/s`, `11.0%` MFU.
  - Step 3: `35.62s`, `4019 tok/s`, `14.2%` MFU.
  - Step 4: `43.32s`, `4650 tok/s`, `16.5%` MFU.
  - Step 5: `34.17s`, `5011 tok/s`, `17.7%` MFU.
  - Step 6: `111.69s`, `4333 tok/s`, `15.3%` MFU.
  - Step 7: `100.67s`, `4032 tok/s`, `14.3%` MFU.
  - Step 8: `92.79s`, `3879 tok/s`, `13.7%` MFU.
  - Step 9: `34.53s`, `4166 tok/s`, `14.7%` MFU.
- The slowdown after step 5 is not evidence against genauto by itself. This smoke used the old small shape: `token_batch_size=262144`, `max_inflight_rollouts=384`, `max_off_policy_steps=4`; cancellations at steps 6-8 forced refill churn (`64`, `97`, `9`, `7` old rollout requests cancelled).
- Correct comparison status:
  - `generation_config="auto"` is now smoke-tested for startup, eval, training, checkpoint, and shutdown behavior.
  - It is not yet tested on the main PipelineRL-speed shape (`token_batch_size=524288`, `max_inflight_rollouts=3072`, `max_off_policy_steps=32`).
  - If we want to keep spending this allocation, the useful next canary is a main-shape `generation_config="auto"` default run, not another small smoke.

10:21 UTC update:

- Cleared stale child Slurm steps from the completed smoke with `scancel 4473657.39 4473657.40 4473657.42`; allocation `4473657` stayed alive. `srun --overlap --jobid=4473657 --nodes=1 --ntasks=1 -w nid011107 hostname` succeeded afterward.
- Created main-shape temp config:
  - `tmp/rl_olmo3_dpo_default_pipelinerl_speed_genauto_20260507_1015.toml`
  - Source: `configs/omni_math2/rl_olmo3_dpo_default_pipelinerl_speed.toml`
  - Changes only: output dir, W&B name, `vllm_extra.generation_config = "auto"`.
  - Preserved main shape: `max_steps=100`, eval `100x8`, train `rollouts_per_example=8`, `token_batch_size=524288`, `max_inflight_rollouts=3072`, `max_off_policy_steps=32`.
- Dry-run passed:
  `source .env && PYTHONPATH=/lus/lfs1aip2/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/verifiers:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/environments/omni_math2_singleturn uv run --no-sync rl @ tmp/rl_olmo3_dpo_default_pipelinerl_speed_genauto_20260507_1015.toml --dry-run --output-dir tmp/dryrun_olmo3_pipelinerl_speed_genauto_20260507_1015`
  plus `bash -n tmp/dryrun_olmo3_pipelinerl_speed_genauto_20260507_1015/gpu_layout_rl.sh`.
- Launched from visible tmux pane `joanv_cc_4node:omni-default`:
  - Output dir: `outputs/omni_math2_rlvr_canary/20260507_1015/dpo_default_pipelinerl_speed_genauto`
  - W&B run id: `b8c791049bb048d28231e1ce11c89cf9`
  - Slurm steps: `.47-.50` on all four nodes.
- Startup status:
  - All 14 inference servers reached `Application startup complete`.
  - Orchestrator logged `Inference pool ready` at `10:19:34`.
  - Initial `100x8` eval started at `10:20:10`.
  - Logs confirm per-server node-local compile caches, e.g. `/tmp/prime-rl-vllm-cache-joanv.a6r-4473657-0/...` and `/tmp/prime-rl-vllm-cache-joanv.a6r-4473657-12/...`.
  - No stale-file-handle, OOM, or inference failure signature so far.
- Early GPU snapshot at `10:21:46` showed 9/14 inference GPUs at `100%` and 5 inference GPUs at `0%`; trainer GPUs idle as expected during blocking eval. Do not call this a bubble yet; recheck after eval has progressed beyond startup/dispatch skew.

10:29 UTC update:

- Main-shape default `generation_config="auto"` initial eval completed at `10:28:38`: `508.01s`, `Avg@8=0.3725`, `Pass@1=0.3725`, `Pass@2=0.4793`, `Pass@4=0.5703`, `Pass@8=0.6500`, `No-response=0.0%`, completion length `5933.69` (range `[129.00, 15360.00]`), truncated `15.5%`.
- Relative to prior initial evals on 14i/2t:
  - Default `generation_config="vllm"`: `483.24s`, `Pass@8=0.6100`, truncated `15.4%`.
  - MaxRL `generation_config="vllm"`: `506.19s`, `Pass@8=0.6400`, truncated `15.5%`.
  - Small-shape genauto smoke: initial `427.50s`, final `572.30s`.
- Interpretation: main-shape genauto is not obviously faster on initial eval; it is roughly equal to the MaxRL-vllm initial eval and slower than the earlier default-vllm initial eval. It also does not show a no-response or truncation regression.
- GPU behavior:
  - At `10:24:27`, all 14 inference GPUs were at `100%` and trainer GPUs were idle during blocking eval.
  - Late-tail snapshot at `10:27:48` showed only 6/14 inference GPUs busy because a few eval requests dominated the tail.
  - After eval, at `10:29:39`, 13/14 inference GPUs were hot again during rollout fill; trainer GPUs still idle before the first batch reaches the trainer.
- One `Timeout during comparison` occurred inside the initial eval. Treat as verifier noise unless frequency rises.
- The run is now in first train rollout fill for orchestrator step 0, target `524288` rollout tokens.

10:33 UTC update:

- Main-shape genauto orchestrator step 0 completed at `10:33:29`: `798.63s`, reward `0.4115`, seq length `2765.2`, async `0`, max off-policy `0`.
- First train rollout fill generated `530909` tokens in `4:48` (`1840.62 tok/s` effective including startup/tail effects) against the `524288` target.
- GPU snapshot at `10:33:30`: all 14 inference GPUs were at `100%`; trainer GPUs idle while waiting for the first batch to reach the trainer. This confirms no inference-side bubble at the first-fill boundary.
- Orchestrator step 1 started at `10:33:30`. Need trainer step 0/1 timings before judging training speed.

10:41 UTC update:

- Trainer is climbing but has not reached the earlier default/MaxRL late-step `~7.3k tok/s` band:
  - Step 0 at `10:34:27`: `857.18s`, startup-accounted `0 tok/s`, `0.0%` MFU, peak mem `81.1 GiB`.
  - Step 1 at `10:37:30`: `178.65s`, `3188 tok/s`, `11.3%` MFU, peak mem `93.4 GiB`.
  - Step 2 at `10:38:40`: `65.14s`, `4624 tok/s`, `16.4%` MFU.
  - Step 3 at `10:39:47`: `62.61s`, `5365 tok/s`, `19.0%` MFU.
  - Step 4 at `10:40:54`: `61.86s`, `5858 tok/s`, `20.7%` MFU.
- Orchestrator:
  - Step 1 at `10:36:14`: `164.11s`, reward `0.4659`, seq length `6303.4`, async `0`, max off-policy `1`.
  - Step 2 at `10:36:21`: `6.66s`, reward `0.3889`, seq length `7812.1`, async `1`, max off-policy `1`.
  - Step 3 paused waiting for trainer checkpoint 2, then completed at `10:37:35`: `73.07s`, async `1`, max off-policy `2`.
  - Step 4 paused waiting for trainer checkpoint 3, then completed at `10:38:45`: `69.23s`, async `1`, max off-policy `3`.
  - Step 5 paused waiting for trainer checkpoint 4, then completed at `10:40:30`: `105.01s`, async `1`, max off-policy `4`.
  - Step 6 started at `10:40:31` and paused waiting for trainer checkpoint 5.
- There have been three `Timeout during comparison` lines in this run so far: one during initial eval and two during step-1 rollout fill. No `ERROR`, `Traceback`, `Inference failure`, or `STOP_INFERENCE` signatures yet.
- GPU snapshot at `10:41:14` showed all 16 GPUs at `100%`, including trainer GPUs. Current bottleneck is not idle inference; the orchestrator is producing enough work to hit the trainer backpressure guard.

10:46 UTC update:

- Trainer continued improving but still has not beaten the earlier late-step `~7.3k tok/s` default/MaxRL band:
  - Step 5 at `10:42:13`: `73.73s`, `6212 tok/s`, `22.0%` MFU.
  - Step 6 at `10:43:25`: `68.30s`, `6470 tok/s`, `22.9%` MFU.
  - Step 7 at `10:44:32`: `62.31s`, `6668 tok/s`, `23.6%` MFU.
  - Step 8 at `10:45:47`: `70.45s`, `6800 tok/s`, `24.1%` MFU.
- Orchestrator evidence around the first checkpoint boundary:
  - Step 6 at `10:42:09`: `97.94s`, async `1`, max off-policy `5`.
  - Step 7 at `10:43:14`: `64.25s`, async `1`, max off-policy `6`.
  - Step 8 at `10:43:44`: `30.17s`, async `1`, max off-policy `7`.
  - Step 9 at `10:45:49`: `124.22s`, async `1`, max off-policy `8`.
  - Checkpoint save at step 10 logged at `10:45:50`; orchestrator step 10 then started immediately.
- Interpretation: the pipeline can build enough backlog for very fast orchestrator steps, but the `>1 step ahead` guard repeatedly pauses the orchestrator while the trainer catches up. This is the correct guard if we want bounded staleness; it also means larger train batches by themselves will not help unless trainer MFU/step time improves or we deliberately relax the staleness guard.
- `Running evals at ckpt_step=10` has not appeared yet. Eval seems likely to trigger after orchestrator step 10 rather than immediately at checkpoint-save time.

10:48 UTC update:

- Trainer step 9 completed at `10:47:05`: `71.73s`, `6874 tok/s`, `24.3%` MFU. This is the best trainer throughput so far in the run, still below the earlier default/MaxRL late-step `~7.3k tok/s` band.
- Trainer logged `Saving checkpoint at step 10` at `10:47:09` and `Saving weight checkpoint at step 10` at `10:47:23`.
- Orchestrator step 10 completed at `10:47:24`: `93.20s`, reward `0.5000`, seq length `6302.8`, async `0`, max off-policy `10`.
- Step-10 eval started at `10:47:26`: `Running evals at ckpt_step=10 for omni-math2-baseline100`, `num_examples=100`, `rollouts_per_example=8`.
- As of `10:48:11`, there has been no `Pausing inference engines for weight update` after the eval start, and gpustat showed all 16 GPUs at `100%`. This is the desired behavior for the blocking-eval policy-update patch; keep watching until eval completes.

10:50 UTC update:

- Step-10 eval made its first progress: `1/800` at `1:24`, `16/800` by `1:55`.
- Trainer step 10 completed during the blocking eval at `10:48:41`: `67.50s`, `7764 tok/s`, `27.5%` MFU. Correction to the earlier intuition: blocking eval does not stop trainer progress; the important invariant is that the orchestrator does not push trainer weight updates into inference while eval is running.
- GPU snapshot at `10:50:15`: all 14 inference GPUs at `100%`; trainer GPUs (`nid011132 g2/g3`) idle at `0%` after trainer step 10 completed.
- Still no post-eval-start `Pausing inference engines for weight update`, no `ERROR`, no `Traceback`, no `RuntimeError`, no `Inference failure`, and no `STOP_INFERENCE` signature.
- Operationally, this is the first clean crossing of the previous step-10 danger zone so far. Do not call it fully cleared until eval completes and the run resumes training afterward.

10:56 UTC update:

- Step-10 eval is alive but much slower than the initial eval: progress reached `32/800` at `7:33` elapsed. The earlier `584/800` line in an `rg` result belonged to the initial eval, not this step-10 eval.
- No post-eval-start inference weight-update pause has appeared after `10:47:26`. The last pause/resume pair was the normal pre-eval update at `10:47:05`/`10:47:09`.
- Inference is saturated rather than bubbling. GPU snapshot at `10:54:56`: all 14 inference GPUs at `100%`, trainer GPUs idle after trainer step 10. Sample server logs around `10:54:55-10:54:58` showed KV cache near `99-100%`, e.g. `Running: 42 reqs, Waiting: 136 reqs` on server 11 and `Running: 23 reqs, Waiting: 159 reqs` on server 5.
- Inference log sweep did not find new crash signatures; matches were startup NCCL initialization and configured arguments, not errors. Current bottleneck looks like eval tail latency plus high KV pressure, not an unused-inference bubble.

10:59 UTC update:

- Step-10 eval progressed through the previous MaxRL failure region: `198/800` at `11:08` elapsed. MaxRL had failed around `97/800` after `11:11`.
- One `Timeout during comparison` appeared inside this eval around `73/800`; this is verifier/sample noise, not yet a run-fatal signal.
- No new trainer step after step 10, no NCCL timeout in trainer `node_3.log`, and no orchestrator `ERROR`/`Traceback`/`RuntimeError`/`Inference failure`/`STOP_INFERENCE` signature.
- GPU snapshot at `10:58:33`: all 14 inference GPUs still `100%`, trainer GPUs idle, aggregate `87%`. Slurm steps `.47-.50` remain alive.
- Current read: the trainer-rank NCCL coordination patch has likely fixed the previous eval-boundary crash, but step-10 eval wall time remains bad.

11:02 UTC update:

- Step-10 eval reached `403/800` at `14:00` elapsed. This confirms the previous MaxRL NCCL failure mode is not recurring in the same place.
- No new post-eval-start weight update, trainer NCCL timeout, or orchestrator failure signature. Trainer remains idle after step 10 while eval uses inference.
- GPU snapshot at `11:01:27`: all 14 inference GPUs still `100%`; trainer GPUs idle; Slurm steps `.47-.50` remain alive.
- Current read is unchanged: eval is slow because the checkpoint-10 workload is decode-tail/KV-pressure heavy, not because inference is underused.

11:05 UTC update:

- Step-10 eval reached `667/800` at `17:43` elapsed. Still no eval completion yet.
- No new post-eval-start update, trainer NCCL timeout, or orchestrator failure signature.
- GPU snapshot at `11:05:10`: 13/14 inference GPUs at `100%`; `nid011131:g1` briefly showed `0%` while model memory remained allocated. Trainer GPUs remained idle. Treat this as likely eval-tail drain/straggler shape unless errors appear.

11:15 UTC update:

- Step-10 eval reached `789/800` at `27:07` elapsed. It is not crashed, but the tail is severe: examples `776`, `787`, etc. took tens of seconds each.
- GPU snapshots now show straggler drain rather than full saturation. At `11:13:02`, 9/14 inference GPUs were hot; at `11:14:34`, only about 6/14 inference GPUs were hot while model memory remained allocated everywhere. Trainer GPUs are still idle after trainer step 10.
- Trainer still has no step after `Step 10 | Time: 67.50s | Throughput: 7764 tokens/s | MFU: 27.5%`; no trainer NCCL timeout appeared in `logs/trainer/node_3.log`.
- Slurm steps `.47-.50` remain alive. Current read: the NCCL/eval-boundary crash is probably fixed, but the eval pipeline has a real tail-latency bubble. Need eval completion plus at least one post-eval trainer step before judging the training-speed win.

11:23 UTC update:

- Step-10 eval completed at `11:16:11`: `1725.14s` / `28m45s`, `Avg@8=0.3875`, `Pass@8=0.6300`, no-response `0.0%`, completion length `5603.11`, truncation `14.5%`.
- The blocking-eval policy worked: there was no mid-eval weight update. The next `Pausing inference engines for weight update` happened at `11:16:12`, after eval success.
- Training resumed and crossed several post-eval steps. Orchestrator steps 13-17 were `~67-76s` wall time and repeatedly hit the `>1 step ahead` guard while waiting for trainer checkpoints.
- Important correction: post-eval trainer log throughput/MFU around `2230 tokens/s` / `8%` is polluted by the 10-point rolling `PerfCounter` including the 28-minute eval gap in its denominator. Actual token files remained full-sized: step 11 `567862`, step 12 `619287`, step 13 `541241`, step 14 `525728`, step 15 `534387`, step 16 `534197` rollout tokens. Dividing by trainer step time gives roughly `7.7k-8.7k tokens/s`, so trainer speed likely recovered.
- Stopped the default `generation_config="auto"` run with `tmux send-keys -t joanv_cc_4node:omni-default C-c` before it reached the next eval. `squeue --steps -j 4473657` now shows only allocation shell `.0` and `.batch`; Slurm allocation remains alive.
- Net read: NCCL/eval-boundary crash fixed, trainer speed probably improved/healthy, but eval has a massive tail-latency bubble. Throughput metrics after long eval pauses need either a reset or a non-rolling per-step companion metric.

11:33 UTC update:

- Created main-shape MaxRL temp config:
  - `tmp/rl_olmo3_dpo_maxrl_pipelinerl_speed_genauto_20260507_1125.toml`
  - Source: `configs/omni_math2/rl_olmo3_dpo_maxrl_pipelinerl_speed.toml`
  - Changes only: output dir, W&B name, `vllm_extra.generation_config = "auto"`.
  - Preserved main shape: eval `100x8`, train `rollouts_per_example=8`, `token_batch_size=524288`, `max_inflight_rollouts=3072`, `max_off_policy_steps=32`, `importance_ratio_clip=5.0`.
- Dry-run and generated launcher syntax passed:
  `source .env && PYTHONPATH=/lus/lfs1aip2/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/verifiers:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/environments/omni_math2_singleturn uv run --no-sync rl @ tmp/rl_olmo3_dpo_maxrl_pipelinerl_speed_genauto_20260507_1125.toml --dry-run --output-dir tmp/dryrun_olmo3_maxrl_pipelinerl_speed_genauto_20260507_1125`
  plus `bash -n tmp/dryrun_olmo3_maxrl_pipelinerl_speed_genauto_20260507_1125/gpu_layout_rl.sh`.
- Launched from visible tmux pane `joanv_cc_4node:omni-maxrl`:
  - Output dir: `outputs/omni_math2_rlvr_canary/20260507_1125/dpo_maxrl_pipelinerl_speed_genauto`
  - W&B run id: `95817e2aa2884143b32e97bef529c86c`
  - Slurm steps: `.51-.54` on all four nodes.
- Startup cleared the previous cache/stale-handle failure point:
  - All inference servers reached `Application startup complete`; the slowest was server 2, which loaded in `140.87s`.
  - Orchestrator logged `Inference pool ready` at `11:31:31`.
  - Logs confirm node-local compile caches under `/tmp/prime-rl-vllm-cache-joanv.a6r-4473657-<server_idx>/...`.
  - No stale-file-handle, OOM, `Inference failure`, `Traceback`, or `RuntimeError` signature so far.
- Initial eval started at `11:31:34`: `Evaluating omni-math2-baseline100 (num_examples=100, rollouts_per_example=8)`.
- GPU snapshot at `11:33:22`: all 14 inference GPUs were at `100%`; trainer GPUs were idle, as expected during blocking eval; aggregate GPU util `87%`.
- Initial eval completed at `11:39:35`: `480.99s`, `Avg@8=0.3525`, `Pass@1=0.3525`, `Pass@2=0.4618`, `Pass@4=0.5647`, `Pass@8=0.6400`, no-response `0.0%`, completion length `6219.38`, truncated `17.5%`.
- One `Timeout during comparison` appeared near the eval tail. Treat it as the known verifier-noise class unless frequency rises.
- Eval tail drained to one hot inference GPU near `775/800`, then completed; this is normal straggler shape, not a crash.
- After eval, the run moved into train rollout fill. GPU snapshot at `11:40:43` showed all 14 inference GPUs at `100%`; trainer GPUs were idle while waiting for the first batch.
- Orchestrator step 0 completed at `11:44:43`: `788.13s`, reward `0.5694`, seq length `2446.5`, async `0`, max off-policy `0`.
- First train rollout fill generated `528434` tokens in `5:04` against target `524288`. This is first-fill/startup plus eval overhead context, so do not use it as steady-state throughput.
- Trainer step 0 completed at `11:45:42`: `847.65s`, loss `-0.0033`, mismatch KL `0.4303`, peak mem `81.1 GiB`. Reported `0 tokens/s` / `0.0%` MFU is startup-accounting noise, same class as prior runs.
- Normal post-step weight update paused inference at `11:45:42` and resumed at `11:45:48`.
- Step 1 rollout fill is active. GPU snapshot at `11:46:17` showed all 14 inference GPUs at `100%`; trainer GPUs had model memory loaded but were idle between trainer steps.

11:52 UTC update:

- Orchestrator:
  - Step 1 completed at `11:46:54`: `131.16s`, reward `0.3750`, seq length `6129.1`, async `0`, max off-policy `1`.
  - Step 2 completed at `11:47:26`: `31.36s`, reward `0.3194`, seq length `8025.5`, async `1`, max off-policy `1`; detected `2/72` repetition rollouts.
  - Step 3 paused waiting for trainer checkpoint 2, resumed after `39.05s`, then completed at `11:48:12`: `45.01s`, reward `0.2361`, seq length `7697.8`, async `1`, max off-policy `2`.
  - Step 4 paused waiting for trainer checkpoint 3, resumed after `71.64s`, then completed at `11:49:28`: `75.86s`, reward `0.4917`, seq length `4470.2`, async `1`, max off-policy `3`.
  - Step 5 paused waiting for trainer checkpoint 4, resumed after `64.15s`, and completed at `11:51:30`: `120.88s`, reward `0.4531`, seq length `4867.0`, async `1`, max off-policy `4`.
  - Step 6 paused waiting for trainer checkpoint 5, resumed after only `13.02s`, and is generating after the normal checkpoint-5 weight update.
- Trainer:
  - Step 1 completed at `11:48:05`: `136.83s`, `3849 tok/s`, `13.7%` MFU, peak mem `93.4 GiB`.
  - Step 2 completed at `11:49:20`: `70.62s`, `5362 tok/s`, `19.0%` MFU.
  - Step 3 completed at `11:50:33`: `64.70s`, `6050 tok/s`, `21.5%` MFU.
  - Step 4 completed at `11:51:43`: `65.64s`, `6396 tok/s`, `22.7%` MFU.
- GPU telemetry at `11:52:11`: all 16 GPUs at `100%`, aggregate utilization `100%`. Snapshot at `11:52:32` showed 14 inference GPUs still `100%`, while trainer GPUs had already dropped to `21%`/`0%` during the next generation phase, aggregate `88%`.
- No `ERROR`, `Traceback`, `RuntimeError`, `Inference failure`, `STOP_INFERENCE`, or trainer NCCL timeout signatures in the latest scan.
- Interpretation: the new `Orchestrator paused: waiting for trainer process to complete checkpoint ... (>1 step(s) ahead)` messages are the bounded-staleness backpressure guard firing because inference can fill the next batch faster than the trainer can consume checkpoints. That is not the old eval deadlock. Trainer throughput is warming but not yet back to the default genauto post-eval actual-token band of roughly `7.7k-8.7k tok/s`; keep watching through steps 5-10 and the step-10 eval boundary.

12:00 UTC update:

- Trainer speed recovered into the prior good band:
  - Step 5 at `11:53:05`: `77.73s`, `6702 tok/s`, `23.8%` MFU.
  - Step 6 at `11:54:17`: `67.41s`, `6840 tok/s`, `24.3%` MFU.
  - Step 7 at `11:55:24`: `63.33s`, `6992 tok/s`, `24.8%` MFU.
  - Step 8 at `11:56:30`: `61.49s`, `7127 tok/s`, `25.3%` MFU.
  - Step 9 at `11:57:47`: `73.12s`, `7174 tok/s`, `25.5%` MFU.
  - Step 10 at `11:59:27`: `69.27s`, `7634 tok/s`, `27.1%` MFU.
- Orchestrator:
  - Step 6 completed at `11:52:56`: `84.60s`, seq length `5116.3`, async `1`, max off-policy `5`.
  - Step 7 completed at `11:54:07`: `70.60s`, seq length `6714.0`, async `1`, max off-policy `6`.
  - Step 8 completed at `11:54:36`: `28.62s`, seq length `8245.1`, async `1`, max off-policy `7`.
  - Step 9 completed at `11:55:51`: `73.89s`, seq length `7871.0`, async `1`, max off-policy `8`.
  - Step 10 completed at `11:57:48`: `115.80s`, seq length `5634.3`, async `1`, max off-policy `9`.
  - Step 11 completed at `11:59:15`: `86.85s`, seq length `5407.6`, async `1`, max off-policy `10`.
- Step-10 eval started at `11:59:16` as `Running evals at ckpt_step=10 for omni-math2-baseline100`, with `num_examples=100`, `rollouts_per_example=8`.
- As of `11:59:39`, there has been no post-eval-start `Pausing inference engines for weight update`, no `ERROR`, no `Traceback`, no `RuntimeError`, no `Inference failure`, no `STOP_INFERENCE`, and no trainer NCCL timeout. Trainer step 10 completed during eval, then trainer GPUs idled.
- GPU telemetry at `11:59:39`: all 14 inference GPUs were at `100%`; trainer GPUs were idle after trainer step 10. Aggregate utilization was `87%`, the expected blocking-eval shape.
- Current read: training speed is now comparable to the best default genauto point (`~7.6k tok/s`), not obviously faster. The important unresolved risk is whether the MaxRL checkpoint-10 eval again develops the severe long-tail behavior or hits the old NCCL/eval-boundary failure.

12:04 UTC update:

- Step-10 eval made early progress to `6/800` by `1:15` elapsed, which is better than the prior failed MaxRL `generation_config="vllm"` step-10 eval (`5/800` after `4:41`). But progress then stalled in the orchestrator log at `6/800` through at least `12:03`.
- No new orchestrator failure signature, no post-eval-start `Pausing inference engines for weight update`, and no trainer NCCL timeout as of the latest scan.
- Inference servers are alive and busy. Sampled server logs show many successful `/v1/chat/completions` responses and high KV pressure:
  - Server 0 around `12:02`: `Running: 27-70`, `Waiting: 139-185`, KV cache `97.6-99.9%`, generation throughput roughly `0.9k-2.0k tok/s`.
  - Server 7 around `12:02`: `Running: 28-45`, `Waiting: 151-176`, KV cache `97.9-99.9%`, generation throughput roughly `1.0k-1.5k tok/s`.
- GPU telemetry at `12:03:30`: all 14 inference GPUs at `100%`; trainer GPUs idle after trainer step 10. Aggregate utilization `87%`.
- Current read: this is not a dead server or idle bubble. It is long-completion / KV-pressure behavior at eval start. Keep monitoring; the key comparison is whether the bulk catches up or this becomes another multi-hour tail.

12:09 UTC update:

- The eval bulk did catch up after the ugly first few completions:
  - `7/800` at `5:08`, with one `Timeout during comparison`.
  - `13/800` at `6:14`.
  - `26/800` at `~7:00`.
  - `61/800` at `8:29`.
  - `90/800` at `9:11`.
- No new trainer output after step 10, no trainer NCCL timeout, no post-eval-start inference weight-update pause, and no orchestrator failure signature.
- GPU telemetry continues to show the expected blocking-eval shape: 14 inference GPUs saturated, trainer GPUs idle.
- Current read: this is still much slower than the initial eval, but it is not extrapolating to hours anymore. It resembles the default genauto step-10 eval behavior: slow, KV-saturated, and tail-heavy, but alive.

12:15 UTC update:

- Step-10 eval is still alive and has passed the prior failed MaxRL eval region by a lot:
  - `198/800` at `12:16` elapsed.
  - `273/800` at `13:44` elapsed.
  - `401/800` at `15:39` elapsed.
- Multiple `Timeout during comparison` messages appeared during the eval, but the orchestrator log continues to advance and there is still no `Inference failure`, `STOP_INFERENCE`, `Traceback`, `RuntimeError`, or trainer NCCL timeout observed.
- Slurm still shows active child steps `.51-.54`; the run has not collapsed back to only the allocation shell.
- GPU telemetry at `12:14:56`: all 14 inference GPUs at `100%`; trainer GPUs `nid011132:g2,g3` idle; aggregate utilization `87%`.
- Current read: the NCCL/eval-boundary failure is fixed. The remaining problem is eval tail latency under saturated inference/KV pressure, not idle inference bubbles.

12:31 UTC update:

- MaxRL step-10 eval completed cleanly:
  - `Evaluated omni-math2-baseline100 in 1677.14s`.
  - `Avg@8=0.4000`, `Pass@1=0.4000`, `Pass@2=0.5071`, `Pass@4=0.5886`, `Pass@8=0.6700`.
  - No-response `0.0%`, completion length `5379.03 ±4896.63`, range `[88,15360]`, truncated `11.9%`.
- The eval did **not** hit the old NCCL/eval-boundary failure. No mid-eval policy-update pause appeared after eval start; the next weight update happened after eval success at `12:27:14`.
- The eval did have a real late-stage bubble:
  - Around `12:17`, inference servers `server_6` (`nid011114:g2`) and `server_12` (`nid011132:g0`) had drained to `Running=0, Waiting=0` while other servers still had queued/running requests.
  - GPU telemetry at `12:20:59` showed only 12/14 inference GPUs hot, aggregate `75%`.
  - GPU telemetry at `12:26:20` showed only 4/14 inference GPUs hot, aggregate `25%`, while the final stragglers ran.
  - This is static/uneven eval-request distribution plus long-tail samples; more total inference GPUs alone will not fix that tail.
- Training resumed after eval and completed extra steps before stop:
  - Orchestrator step 12 included the eval and logged `1682.00s`, reward `0.5167`, max off-policy `11`.
  - Post-eval orchestrator steps 13 and 14 took `73.42s` and `70.19s`, with max off-policy `12` and `13`.
  - Trainer steps 11-13 logged `~2200 tok/s` / `7.8%` MFU, but treat that as polluted by the same rolling-PerfCounter-after-long-eval issue observed in the default run.
- Stopped the MaxRL run via `tmux send-keys -t joanv_cc_4node:omni-maxrl C-c` at `12:30:48`, after the step-10 eval result was captured and before the next eval cycle.
- Confirmed `squeue --steps -j 4473657` now shows only allocation shell `.0` and `.batch`. Slurm allocation remains alive.

## Key Files Touched

- `configs/omni_math2/rl_olmo3_*.toml` - set `max_steps = 100`, `gpu_layout`, 6i/2t topology, eval every 10, eval seed, train temp `1.0`, eval temp `0.6`.
- `src/prime_rl/configs/rl.py` - added/extended `gpu_layout` deployment model and validation.
- `src/prime_rl/entrypoints/rl.py` - added `gpu_layout` allocation launcher path.
- `src/prime_rl/templates/gpu_layout_rl.sh.j2` - new generated launcher for sub-node inference/trainer placement; includes inference-failure sentinel handling.
- `src/prime_rl/templates/gpu_layout_rl.sh.j2` - now also isolates vLLM/Torch/Triton compile caches per inference server on node-local `/tmp`.
- `src/prime_rl/configs/orchestrator.py`, `src/prime_rl/orchestrator/envs.py` - added eval seed support.
- `src/prime_rl/utils/validation.py` - fixed shared weight-broadcast validation so inference mismatches cannot slip through.
- `src/prime_rl/trainer/models/*/modeling_*.py` - added `cache_position` docstring entries to stop Transformers docstring-signature errors.
- `src/prime_rl/trainer/rl/broadcast/nccl.py` - added trainer-side filesystem marker coordination so non-master trainer ranks do not enter DTensor collectives while master waits for inference `NCCL_READY`.
- `docs/async.md`, `docs/metrics.md`, `skills/config/SKILL.md`, `skills/monitor-run/SKILL.md` - updated async/staleness/config monitoring docs.
- `tests/unit/test_configs.py` - added config regression tests for eval seed, gpu layout, and weight-broadcast mismatch.
- `tests/unit/train/rl/test_nccl_broadcast_coordination.py` - added regression coverage for the trainer-side NCCL coordination marker.

## Validated State

These passed after the latest temp/config changes:

```bash
uv run --no-sync pytest \
  tests/unit/test_configs.py::test_eval_seed_inherits_to_env \
  tests/unit/test_configs.py::test_eval_env_seed_override_wins \
  tests/unit/test_configs.py::test_validate_shared_weight_broadcast_rejects_inference_mismatch \
  tests/unit/test_configs.py::test_gpu_layout_deployment_sets_one_gpu_inference_pool \
  tests/unit/test_configs.py::test_gpu_layout_rejects_overlapping_gpu_roles \
  tests/unit/test_configs.py::test_gpu_layout_rejects_layout_without_inference_gpus \
  -q
```

Result: `6 passed`.

All seven Omni TOMLs dry-ran successfully after the latest main-recipe edit:

```bash
set -e
for cfg in configs/omni_math2/rl_olmo3_*.toml; do
  source .env
  PYTHONPATH=/lus/lfs1aip2/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/verifiers:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/environments/omni_math2_singleturn \
    uv run --no-sync rl @ "$cfg" --dry-run --output-dir /tmp/prime-rl-dryrun-main-recipe
done
```

Also validated `bash -n /tmp/prime-rl-dryrun-final/gpu_layout_rl.sh`.

## Current TOML Shape

The two DPO configs are currently the main next-run pair:

- `configs/omni_math2/rl_olmo3_dpo_default_20step.toml`
- `configs/omni_math2/rl_olmo3_dpo_maxrl_20step.toml`

Despite the filename, both now have `max_steps = 100`.

Current common shape:

```toml
max_steps = 100
max_async_level = 1

[deployment]
type = "gpu_layout"
gpus_per_node = 4
inference_port_start = 8000

[[deployment.nodes]]
inference = [0, 1, 2, 3]

[[deployment.nodes]]
inference = [0, 1]
trainer = [2, 3]

[orchestrator]
token_batch_size = 524288
max_inflight_rollouts = 768
rollouts_per_example = 8
max_off_policy_steps = 8

[trainer.optim]
lr = 1e-6
max_norm = 1.0

[orchestrator.buffer]
easy_threshold = 1.0
online_difficulty_filtering = true
seed = 42

[orchestrator.train.sampling]
temperature = 1.0
top_p = 0.95
max_completion_tokens = 15360

[orchestrator.eval]
interval = 10
num_examples = 100
seed = 42
rollouts_per_example = 8
num_workers = 128
cancel_inflight_rollouts_on_eval = false

[orchestrator.eval.sampling]
temperature = 0.6
top_p = 0.95
max_completion_tokens = 15360

[inference.parallel]
tp = 1
dp = 1
```

## Next Run Recommendation

For the next allocation, do **not** run the old 1000-step recipe unchanged.
Use the PipelineRL-speed Default/MaxRL TOMLs for the current stopgap recipe:

```toml
[trainer.optim]
lr = 1e-6

[orchestrator]
batch_size = 256
max_inflight_rollouts = 768
rollouts_per_example = 8
max_off_policy_steps = 8

[orchestrator.buffer]
easy_threshold = 1.0
online_difficulty_filtering = true

[orchestrator.eval]
cancel_inflight_rollouts_on_eval = false
```

Keep:

```toml
[orchestrator.train.sampling]
temperature = 1.0

[orchestrator.eval.sampling]
temperature = 0.6
```

Do **not** shorten `max_completion_tokens`; the user explicitly rejected that.

These LR/filter thresholds are model-specific to OLMo3 Omni-MATH RLVR. Do not automatically port them to other models.

## Short Smoke Canaries

Before the 100-step pair, consider two small apples-to-apples smokes using temporary config copies and the older small shape (`token_batch_size = 262144`, `max_inflight_rollouts = 384`, `max_off_policy_steps = 4`, original LR/filtering unless testing that axis). Keep the smoke labels/output dirs explicit.

1. `generation_config = "auto"` smoke:
   - Change only `vllm_extra.generation_config` from `"vllm"` to `"auto"`.
   - Use a short run, e.g. `max_steps = 10`, eval interval 10, eval temp 0.6.
   - Judge only truncation, completion length, and finish/stop behavior. Do not interpret reward movement from this smoke.
2. Frozen-reference KL or loss rewrite smoke:
   - This is not currently a config-only change. It requires an implemented reference-logprob/loss path or a deliberate loss variant.
   - Keep it separate from the main recipe and run a short labeled canary after the code path exists.
   - Compare against a same-shape control; do not mix it into the default-vs-MaxRL pair.

## New Allocation Launch Procedure

Inside the new 4-node allocation, attach/create a tmux pane first.

```bash
tmux new-window -n omni-canary
# or split an existing allocation tmux pane; the important bit is visible/attached launch.
```

From the tmux pane:

```bash
cd /lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl
source .env
export PYTHONPATH=/lus/lfs1aip2/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/verifiers:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/environments/omni_math2_singleturn
scontrol show hostnames "$SLURM_JOB_NODELIST"
```

Critical: `gpu_layout` defaults to the first `len(nodes)` hosts. For two parallel runs, split the 4 hosts explicitly:

- Run A: hosts 1 and 2 from the allocation.
- Run B: hosts 3 and 4 from the allocation.

Use `[deployment].hosts = ["nid...", "nid..."]` in temporary config copies or an equivalent CLI override. Do not let both runs default to the same first two nodes.

Suggested temporary-copy workflow:

1. Copy `rl_olmo3_dpo_default_20step.toml` and `rl_olmo3_dpo_maxrl_20step.toml` to `/tmp` or `tmp/`.
2. In each copy:
   - set `output_dir` to a unique run root;
   - set `[deployment].hosts` to the intended two-node slice;
   - keep the main recipe values unless intentionally making a short smoke/control.
3. Dry-run both copied configs before launching.

Dry-run command:

```bash
uv run --no-sync rl @ /path/to/temp_config.toml --dry-run --output-dir /tmp/prime-rl-dryrun-new-allocation
```

Launch command from tmux pane:

```bash
uv run --no-sync rl @ /path/to/temp_config.toml
```

The launcher itself prints the per-host `srun --overlap --jobid=...` commands and then runs them unless `--dry-run` is set.

## Monitoring Commands

Check jobs:

```bash
squeue -u "$USER"
squeue --steps -u "$USER"
```

Watch logs:

```bash
tail -F outputs/omni_math2_rlvr_canary/dpo_default/logs/trainer.log
tail -F outputs/omni_math2_rlvr_canary/dpo_default/logs/orchestrator.log
tail -F outputs/omni_math2_rlvr_canary/dpo_default/logs/inference/server_*.log
```

For GPU snapshots, prefer the allocation’s `GPUSTAT_DIR` if present:

```bash
find "$GPUSTAT_DIR" -maxdepth 1 -type f -print -exec tail -n 5 {} \;
```

Do not use gpustat latest-snapshot files after a run as historical utilization evidence.

Important W&B/log metrics:

- `time/wait_for_batch`
- `time/forward_backward`
- `time/generate_completions`
- `scheduler/async_level`
- `scheduler/inflight_rollouts`
- `scheduler/cancelled_rollouts`
- `off_policy_level/*`
- `is_truncated/all/mean`
- `mismatch_kl/mean`
- `is_masked_low/mean`, `is_masked_high/mean`
- eval `avg@8`, `pass@8`, and reward components.

## Diagnostic Conclusions So Far

Current best read:

- The scary eval “regression” is not real model damage. It is mostly eval-composition mismatch: first-100 eval subset is hard/skewed and omits easy Pascal/Fermat/Cayley strata.
- RL signal looked inert/noisy so far. Train reward and eval did not show detectable coupling at current power.
- Inference/topology speedup worked: 6i/2t reduced eval/rollout wall time substantially. But trainer still starves on rollout generation.
- Bigger token batch is the next cheap lever; activation checkpointing is not the main unlock for merely accumulating more packed microbatches.
- Train temperature should be `1.0`; eval should stay `0.6`.
- `generation_config = "auto"` is worth a smoke test, but the EOS bug is not proven just because returned text lacks `<|im_end|>`; stop tokens may be stripped.
- DPPO mask analysis: current mask is prob-diff based while policy-gradient still uses the unclipped importance ratio. But the claim that masking is random w.r.t. KL is probably wrong; masked tokens have higher KL than unmasked tokens.
- There is no frozen-reference KL anchor. Current `kl_tau` is trainer-vs-rollout, not base-vs-policy.
- Judge fallback did not show increasing reward hacking, but stable judge leniency/noise still deserves audit.

## What Not To Do

- Do not compare the 100-subset eval directly to published full-600 baseline numbers.
- Do not shorten `max_completion_tokens`.
- Do not launch two runs without explicit `deployment.hosts`; both will otherwise grab the first two hosts.
- Do not run repeated `srun` polling loops inside an allocation; they can burn Slurm step IDs.
- Do not assume `max_async_level = 1` means consumed batches are on-policy. In-flight rollout groups can age; inspect `off_policy_level/*`.
- Do not treat `generation_config="auto"` as already proven. Run a smoke and compare truncation/length/finish behavior.

## Recommended Next Steps

1. Get a fresh 4-node GH200 allocation and attach a tmux pane.
2. Create two temporary config copies for default and MaxRL with explicit disjoint `[deployment].hosts`.
3. Verify the current main recipe is present: `lr=1e-6`, thresholds `0.875/0.0625`, batch shape `524288 / 768 / 8`, and eval cancellation off.
4. Dry-run both temporary configs.
5. Launch both from tmux.
6. Monitor throughput and staleness. The falsifier is: if decode throughput does not improve and vLLM waiting queue is low, bottleneck moved to orchestration/env concurrency, not GPU count.
7. After the pair finishes, analyze:
   - eval composition-corrected metrics;
   - reward trajectories;
   - `time/wait_for_batch` vs `time/forward_backward`;
   - cancellation/off-policy levels;
   - truncation and decode length;
   - DPPO mask/KL metrics.

## Useful Analysis Artifacts

- `tmp/omni_diag/SYNTHESIS_FINAL.md` - first synthesis; useful but overclaims EOS and some variance claims.
- `tmp/omni_diag/INTERACTION_FINDINGS.md` - newer interaction analysis; useful but read critically.
- `tmp/omni_diag/findings_E_interactions.md` - analyst details for correlations and eval subset.
- `tmp/omni_diag/plots/interactions_*.png` - new plots.

## 2026-05-07 12:44 UTC Notes

- After the main-shape MaxRL `generation_config="auto"` run completed step-10
  eval cleanly, the remaining speed pathology was not NCCL. It was eval tail
  imbalance: some 1-GPU vLLM servers drained to `Running=0, Waiting=0` while
  others were still stuck on long completions.
- Implemented bounded dynamic refill for online evals:
  - `src/prime_rl/configs/orchestrator.py` adds
    `max_concurrent_rollouts_per_client` on `[orchestrator.eval]` and per-env
    eval config.
  - `src/prime_rl/orchestrator/envs.py` keeps legacy eager gather when unset,
    but when set uses an eval-local worker queue and least-current-eval-load
    assignment across `eval_clients`.
  - `src/prime_rl/orchestrator/orchestrator.py` passes
    `inference_pool.eval_clients` into evals.
  - `src/prime_rl/utils/client.py` exposes `eval_clients` on the inference-pool
    protocol.
- Set both PipelineRL-speed Omni configs to
  `[orchestrator.eval].max_concurrent_rollouts_per_client = 48`. This does not
  change eval sample size (`100`) or `rollouts_per_example` (`8`); it changes
  prequeue geometry from eager 800-way assignment to a 14 × 48 rollout window
  with refill.
- Added `tests/unit/orchestrator/test_envs.py` to verify fast eval clients pull
  extra work without exceeding the per-client window. Also repaired an existing
  brittle scheduler test fixture that lacked `train_envs` and old `*_by_env`
  counters.
- Validation:
  - `uv run pytest tests/unit/orchestrator/test_envs.py tests/unit/orchestrator/test_scheduler.py -q`
    → `6 passed`.
  - `uv run pytest tests/unit/orchestrator/test_eval_scheduling.py -q`
    → `13 passed`.
  - Dry-runs passed for
    `configs/omni_math2/rl_olmo3_dpo_default_pipelinerl_speed.toml` and
    `configs/omni_math2/rl_olmo3_dpo_maxrl_pipelinerl_speed.toml` into
    `tmp/dryrun_default_eval_refill_20260507` and
    `tmp/dryrun_maxrl_eval_refill_20260507`; both rendered scripts passed
    `bash -n`.

## 2026-05-07 12:49 UTC Live Eval-Refill Run

- Allocation `4473657` is still alive on
  `nid011107,nid011114,nid011131,nid011132`.
- Created exact temp configs for the eval-refill canaries:
  - `tmp/rl_olmo3_dpo_default_pipelinerl_speed_evalrefill_20260507_1245.toml`
    → `outputs/omni_math2_rlvr_canary/20260507_1245/dpo_default_pipelinerl_speed_evalrefill`
  - `tmp/rl_olmo3_dpo_maxrl_pipelinerl_speed_evalrefill_20260507_1245.toml`
    → `outputs/omni_math2_rlvr_canary/20260507_1245/dpo_maxrl_pipelinerl_speed_evalrefill`
- Both exact temp configs dry-ran successfully into
  `tmp/dryrun_default_evalrefill_exact_20260507_1245` and
  `tmp/dryrun_maxrl_evalrefill_exact_20260507_1245`; both rendered
  `gpu_layout_rl.sh` scripts passed `bash -n`.
- Launched default first from tmux pane `joanv_cc_4node:omni-default`:
  `source .env && PYTHONPATH=/lus/lfs1aip2/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/verifiers:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/environments/omni_math2_singleturn uv run --no-sync rl @ tmp/rl_olmo3_dpo_default_pipelinerl_speed_evalrefill_20260507_1245.toml`
- W&B run id for default eval-refill: `727ef370ba9c4516b529e4d6d8c32827`.
- Four child Slurm steps are running: `4473657.55-.58`.

## 2026-05-07 13:05 UTC Live Eval-Refill Update

- Default eval-refill startup was clean enough for the canary: all 14 inference
  servers reached `Application startup complete`; no OOM, traceback, or stale
  file-handle failure was found in the startup sweep.
- Initial `100x8` eval used the new dynamic-refill path:
  `Eval dynamic refill enabled for omni-math2-baseline100: 672 workers across
  14 inference clients`.
- Initial eval completed in `470.85s`:
  `Avg@8=0.3575`, `Pass@8=0.6400`, no-response `0.0%`, mean completion length
  `5935.47`, truncation `16.8%`.
- GPU snapshots during initial eval and train rollout generation had all 14
  inference GPUs hot and the 2 trainer GPUs idle; during the start of step 1
  both trainer GPUs also became active, with aggregate allocation utilization
  around `94%` at `13:04:08 UTC`.
- Step 0 completed at `13:03:34`:
  `Time=788.66s`, reward `0.5195`, sequence length `2165.8 tokens/sample`,
  async level `0`, max off-policy level `0`. Train rollout generation for step
  0 overshot the `524288` token target to `557140` tokens in `315s`.
- `Timeout during comparison` appeared both during eval and train scoring. It
  did not stop the run; treat it as verifier/judge timeout noise unless it
  starts correlating with failures or metric drift.
- Default is currently on orchestrator step 1. Do not stop it until the step-10
  eval completes or a hard failure occurs.

## 2026-05-07 13:10 UTC Eval-Refill Failure

- The default eval-refill run did hit a hard failure before step-10 eval:
  `outputs/omni_math2_rlvr_canary/20260507_1245/dpo_default_pipelinerl_speed_evalrefill`.
- This was not the earlier trainer NCCL issue and not caused by verifier
  comparison timeouts. The failing process was inference server 3 on
  `nid011107`; its log is
  `outputs/omni_math2_rlvr_canary/20260507_1245/dpo_default_pipelinerl_speed_evalrefill/logs/inference/server_3.log`.
- Root cause in that log: vLLM raised `torch.OutOfMemoryError` while trying to
  allocate `4.98 GiB` with only `3.69 GiB` free. The server had roughly
  `91.30 GiB` in use, KV cache was near full, and the config was running
  `max_num_seqs = 192`, `max_num_batched_tokens = 131072`,
  `gpu_memory_utilization = 0.93`.
- The queue was already large around the failure (`Running` around `18-20`
  requests and `Waiting` around `196-204` requests), so the likely bad knob is
  the inference-side token microbatch cap, not train batch size, eval sample
  size, or rollout group size.
- The run got through initial `100x8` eval in `470.85s`; step 1 consumed a
  queued batch quickly, and step 2 was extremely fast because it was eating
  in-flight results. Do not count step-2 wall time as fresh generation
  throughput.
- Cleanup: sent Ctrl-C to tmux pane `joanv_cc_4node:omni-default`; child Slurm
  steps `.55-.58` disappeared. Allocation `4473657` remains alive with only
  `4473657.0` and `4473657.batch`.
- Next launch should keep train/eval shape fixed (`100x8`,
  `rollouts_per_example = 8`, `token_batch_size = 524288`) and cut
  `max_num_batched_tokens` to `65536` for both PipelineRL-speed configs before
  retrying. If that still OOMs, lower `gpu_memory_utilization` or
  `max_num_seqs`; do not silently shrink the scientific comparison.

## 2026-05-07 13:14 UTC OOM-Guard Relaunch

- Patched both checked PipelineRL-speed configs and new temp configs to use
  `max_num_batched_tokens = 65536`; kept `max_num_seqs = 192`,
  `gpu_memory_utilization = 0.93`, eval `100x8`,
  `rollouts_per_example = 8`, `token_batch_size = 524288`, and
  `max_inflight_rollouts = 3072`.
- New temp configs:
  - `tmp/rl_olmo3_dpo_default_pipelinerl_speed_evalrefill_oomguard_20260507_1310.toml`
  - `tmp/rl_olmo3_dpo_maxrl_pipelinerl_speed_evalrefill_oomguard_20260507_1310.toml`
- Both temp configs dry-ran successfully into
  `tmp/dryrun_default_evalrefill_oomguard_20260507_1310` and
  `tmp/dryrun_maxrl_evalrefill_oomguard_20260507_1310`; both rendered
  `gpu_layout_rl.sh` scripts passed `bash -n`.
- Relaunched default from tmux pane `joanv_cc_4node:omni-default`.
  Output dir:
  `outputs/omni_math2_rlvr_canary/20260507_1310/dpo_default_pipelinerl_speed_evalrefill_oomguard`.
  W&B run id: `ea0b0b0fc5704230a006c3c147606e41`.
- Child Slurm steps after relaunch: `4473657.59-.62`.

## 2026-05-07 13:24 UTC OOM-Guard Default Initial Eval

- All 14 inference servers reached `Application startup complete`; the two
  `nid011132` servers were slow because CUDA graph capture took about 40s, not
  because they were wedged.
- Verified server logs show the intended vLLM cap:
  `max_num_batched_tokens = 65536`.
- Initial eval started at `13:15:58` with dynamic refill:
  `672 workers across 14 inference clients`.
- Initial `100x8` eval completed at `13:23:29` in `451.26s`:
  `Avg@8=0.3625`, `Pass@8=0.6400`, no-response `0.0%`, mean completion length
  `5887.65`, truncation `15.4%`.
- This is faster than the previous eval-refill initial eval (`470.85s`) despite
  halving `max_num_batched_tokens`; the OOM guard does not appear to hurt
  initial eval throughput. Continue monitoring through train steps and the
  step-10 eval, where the previous run failed after queue pressure and weight
  update.

## 2026-05-07 13:35 UTC OOM-Guard Default Mid-Run

- Step 0:
  - Orchestrator generated `567092` tokens in `5:22` rollout-generation time.
  - Orchestrator step 0 finished at `13:28:55`: `Time=776.62s`,
    reward `0.5192`, seq length `2726.4`, async `0`, max off-policy `0`.
  - Trainer step 0 finished at `13:29:54`: `Time=835.85s`, loss `-0.0013`,
    mismatch KL `0.4902`, peak memory `81.2 GiB`. Throughput/MFU are still
    useless for step 0 because it includes initial wait-for-batch.
- Step 1 survived the first post-update OOM-risk point:
  - Weight update pause/resume `13:29:54-13:29:58`.
  - Orchestrator step 1 finished at `13:31:37`: `Time=161.85s`, reward
    `0.4423`, seq length `5459.6`, async `0`, max off-policy `1`.
  - Trainer step 1 finished at `13:32:49`: `Time=170.93s`, throughput
    `3360 tokens/s`, MFU `11.9%`, peak memory `93.4 GiB`.
- Step 2 was mostly queued/in-flight consumption, not fresh generation:
  - Orchestrator step 2 finished in `5.59s`, async `1`, max off-policy `1`.
  - Trainer step 2 finished at `13:33:57`: `Time=64.06s`, throughput
    `4923 tokens/s`, MFU `17.5%`, peak memory `93.3 GiB`.
- Steps 3-4 show the current topology is trainer-limited:
  - Step 3 waited `65.17s` for checkpoint 2, then finished with max
    off-policy `2`.
  - Step 4 waited `63.15s` for checkpoint 3, then finished with max
    off-policy `3`.
- No OOM, `STOP_INFERENCE`, or hard inference failure signatures have appeared
  so far. The bubbles now are trainer-checkpoint backpressure, not inference
  idleness.

## 2026-05-07 13:43 UTC OOM-Guard Default Pause Failure

- The OOM-guard default run failed at step 5 before reaching step-10 eval:
  `outputs/omni_math2_rlvr_canary/20260507_1310/dpo_default_pipelinerl_speed_evalrefill_oomguard`.
- This was not the previous vLLM OOM. Server 0 stayed alive through the failure
  window and continued logging metrics at `13:36:25` and `13:36:35` with
  `Running=0`, `Waiting=0`, `GPU KV cache usage=0.0%` after the orchestrator
  exited. No `OutOfMemoryError`, CUDA OOM, traceback, or server-side fatal log
  was found in `logs/inference/server_0.log`.
- The fatal orchestrator error was an admin transport failure during weight
  update pause:
  `httpx.RemoteProtocolError: Server disconnected without sending a response`
  for `POST http://nid011107:8000/pause?mode=keep&clear_cache=false`.
- Right before the failed pause, server 0 was heavily saturated but alive:
  around `13:36:15`, `Running=26`, `Waiting=187`, `GPU KV cache usage=98.6%`,
  and generation throughput around `850.9 tokens/s`.
- Previous pause/update/resume cycles on server 0 succeeded at `13:32:49`,
  `13:33:57`, and `13:35:10`; the fifth pause did not produce a logged `200
  OK`. Some other servers did log a fifth successful pause, which means this
  can leave the fleet partially paused if the client hard-fails immediately.
- Child Slurm steps `.59-.62` were cleaned up by the launcher after failure;
  allocation `4473657` remains alive with only `4473657.0` and
  `4473657.batch`.
- Patch applied:
  - `src/prime_rl/utils/client.py` adds bounded retry/backoff for admin
    `/pause`, `/resume`, and `/update_weights` calls on retryable transport
    errors and 408/409/425/429/5xx statuses.
  - If `_pause_engines()` fails after a partial pause, `update_weights()` now
    attempts `_resume_engines()` before re-raising.
  - Retry budget is bounded: 5 attempts, exponential backoff up to 8s, admin
    request read timeout aligned with the 1200s NCCL window.
- Focused validation passed:
  `uv run pytest tests/unit/utils/test_client.py -q` → `11 passed`.
- Next step: rerun the default OOM-guard temp config from tmux. If it reaches
  and completes the step-10 eval, launch the MaxRL OOM-guard temp config with
  the same scientific shape.

## 2026-05-07 14:09 UTC OOM-Guard Default Retry Past Step-5

- Live retry:
  `outputs/omni_math2_rlvr_canary/20260507_1345/dpo_default_pipelinerl_speed_evalrefill_oomguard_retry`.
- W&B run id: `042695832a314268bb809b581ab66255`.
- Initial `100x8` eval completed at `13:57:10` in `449.24s`:
  `Avg@8=0.3638`, `Pass@8=0.6300`, no-response `0.0%`, truncation `15.8%`.
  This is slightly faster than the prior OOM-guard initial eval (`451.26s`)
  and faster than the failed `max_num_batched_tokens=131072` eval-refill run
  (`470.85s`).
- Step 0 finished at `14:03:17`: orchestrator time `815.90s`, reward
  `0.4850`, async `0`, max off-policy `0`; trainer step 0 finished at
  `14:04:13` with peak memory `81.0 GiB`.
- Step 1 pause/resume succeeded at `14:04:14-14:04:19`; orchestrator step 1
  finished in `124.08s`; trainer step 1 finished in `132.42s`, `4240 tok/s`,
  `15.1%` MFU.
- Steps 2-5 were mostly queue consumption plus trainer-checkpoint waits:
  - Step 2 orchestrator `0.76s`, async `1`, max off-policy `1`.
  - Step 3 waited `68.15s` for checkpoint 2, then completed after a clean
    pause/resume.
  - Step 4 waited `61.10s` for checkpoint 3, then completed after a clean
    pause/resume.
  - Step 5 waited `65.12s` for checkpoint 4, then completed after a clean
    pause/resume at `14:08:48-14:08:53`.
- This clears the previous failure point. No `RemoteProtocolError`, CUDA OOM,
  `OutOfMemoryError`, traceback, or `STOP_INFERENCE` signature was present in
  the retry logs as of `14:09 UTC`.
- The current bubble is not inference idleness. With queued rollouts available,
  orchestrator steps 2-5 complete generation almost immediately and then wait
  roughly a trainer step for fresh checkpoints. Live gpustat at `14:07:57`
  showed all 16 GPUs at `100%` utilization, with about `1.50 TiB / 1.56 TiB`
  aggregate GPU memory in use.

## 2026-05-07 14:47 UTC Default Retry Step-10 Eval Result

- The default OOM-guard eval-refill retry survived the previous failure modes:
  no vLLM OOM, no admin `RemoteProtocolError`, and no NCCL broadcast failure
  before or during the step-10 eval.
- Trainer settled around `65-72s` per step after startup, with throughput rising
  from `4240 tok/s` / `15.1%` MFU at trainer step 1 to `7666 tok/s` / `27.2%`
  MFU at trainer step 10. Peak memory stayed around `93.4 GiB`.
- Orchestrator reached the interval-10 eval at `14:16:11` for `ckpt_step=10`.
  The eval completed at `14:45:06` in `1734.51s`:
  `Avg@8=0.3775`, `Pass@8=0.6200`, no-response `0.0%`, mean completion length
  `5487.58`, truncation `12.4%`.
- This is not a speed win versus the earlier main-shape `generation_config=auto`
  default step-10 eval (`1725.14s`). Dynamic refill kept the fleet busy during
  the bulk of eval, but the final straggler tail was severe: around `14:43 UTC`,
  only 3 inference GPUs were still active while most inference GPUs plus the 2
  trainer GPUs were idle.
- After recording the eval, the default retry was stopped with Ctrl-C from
  `joanv_cc_4node:omni-default`. Slurm child steps `.63-.66` were cleaned up;
  allocation `4473657` remains alive.
- Next step: dry-run and launch the matching MaxRL OOM-guard eval-refill retry
  in `joanv_cc_4node:omni-maxrl`.

## 2026-05-07 14:57 UTC MaxRL Retry Launch

- Launched matching MaxRL OOM-guard eval-refill retry from
  `joanv_cc_4node:omni-maxrl`:
  `outputs/omni_math2_rlvr_canary/20260507_1449/dpo_maxrl_pipelinerl_speed_evalrefill_oomguard_retry`.
- W&B run id: `e1fefe2478ed4194963058aeed955ad9`.
- Dry-run and `bash -n` passed before launch:
  `tmp/dryrun_maxrl_evalrefill_oomguard_retry_20260507_1448`.
- Slurm child steps: `.67-.70`.
- All 14 inference servers reached `Application startup complete`; vLLM cap
  verified as `gpu_memory_utilization=0.93`, `max_num_seqs=192`,
  `max_num_batched_tokens=65536`, `generation_config="vllm"`.
- Trainer and orchestrator are live; initial `100x8` eval is running as of
  `14:57 UTC`.

## 2026-05-07 15:21 UTC MaxRL Retry Step-10 Eval Started

- Initial eval completed at `15:00:53` in `467.30s`:
  `Avg@8=0.3812`, `Pass@8=0.6400`, no-response `0.0%`, truncation `16.6%`.
- Trainer reached step 9 at `15:19:04` with `6904 tok/s` and `24.5%` MFU.
  This is improving, but still below the default OOM-guard retry late steady
  state (`7352-7666 tok/s`, `26.1-27.2%` MFU at steps 9-10).
- The admin retry patch recovered a real pause failure at `15:09:10`:
  four pause requests hit `RemoteProtocolError` on attempt 1/5, then all
  inference engines paused and resumed successfully by `15:09:17`; the run
  continued.
- Orchestrator step 11 finished at `15:19:48` and then launched the interval
  eval at `15:19:49` for `ckpt_step=10`.
- Eval dynamic refill reported `672 workers across 14 inference clients`.
  Early eval progress was slow (`1/800` at about 74s, `2/800` at about 95s).
- `Timeout during comparison` appears in the orchestrator log, but it also
  appears embedded in an earlier train rollout progress line, so do not treat
  it as a fatal launcher error without more evidence.
- Live gpustat at `15:21:26` showed aggregate GPU utilization `81%`: 13 GPUs at
  `100%`, while `nid011132` GPUs `g0-g2` were idle and `g3` was active. This is
  during the step-10 eval window, so keep checking for eval tail/idleness.
- The temporary idleness was not because the last-node inference servers had no
  work. `server_12` and `server_13` showed large `Running`/`Waiting` queues but
  near-zero generation throughput around `15:20-15:21`, then resumed by
  `15:24` with about `2300-2560 tok/s` generation throughput.
- Live gpustat at `15:24:01` showed all 16 GPUs at `100%`, but eval progress was
  still only around `21/800` after roughly 4m12s. Current read: not a fatal
  stall, but the step-10 eval is much slower than the initial eval so far.

## 2026-05-07 15:30 UTC Eval Backlog Fix + Relaunch

- Judgement call: stopped the MaxRL run during the slow step-10 eval instead of
  waiting for completion. It had reached only `24/800` after `6:54`, so the
  eval was measuring backlog contention more than MaxRL.
- Root cause found: interval eval ran only after orchestrator step 11 had
  already queued a full train batch at `ckpt_step=10`. Because
  `cancel_inflight_rollouts_on_eval = false`, eval requests sat behind stale
  train rollout work in vLLM queues. The `Running+Waiting` server counts were
  far above the eval cap alone, confirming this.
- Changed both speed configs:
  - `configs/omni_math2/rl_olmo3_dpo_default_pipelinerl_speed.toml`
  - `configs/omni_math2/rl_olmo3_dpo_maxrl_pipelinerl_speed.toml`
  now set `[orchestrator.eval].cancel_inflight_rollouts_on_eval = true`.
- Updated `skills/config/SKILL.md` with the same lesson so future pipelined
  blocking evals do not inherit stale train queue congestion.
- Dry-runs passed:
  - `tmp/dryrun_default_cancel_eval_20260507_1528`
  - `tmp/dryrun_maxrl_cancel_eval_20260507_1528`
- Relaunched default first, sequential-smoke style, from `joanv_cc_4node:omni-default`:
  `outputs/omni_math2_rlvr_canary/20260507_1530/dpo_default_pipelinerl_speed_cancel_eval`.
  Same topology: 14 inference GPUs + 2 trainer GPUs.

## 2026-05-07 15:36 UTC Default Relaunch Startup Failure

- The 15:30 default relaunch failed before eval/training with launcher return
  codes `[0, 137, 137, 1]`; allocation `4473657` itself stayed alive.
- Failure path: node `nid011107` inference manager reported
  `Inference server manager on nid011107 observed pid 280865 exit with status 0`,
  wrote `INFERENCE_FAILED`, and triggered `STOP_INFERENCE` on the other nodes.
- Server logs on `nid011107` (`server_0.log` through `server_3.log`) do not show
  a CUDA/OOM/traceback root cause before `EngineDeadError`; they show
  `EngineCore ... Shutdown initiated (timeout=0)` first, then the API server
  reports `AsyncLLM output_handler failed`. Current read: the engine was told to
  shut down or otherwise exited cleanly; `EngineDeadError` is fallout, not root
  cause.
- Startup was badly staggered because the generated launcher deleted each
  per-server compile cache on every launch. Trainer-node inference servers
  `8012/8013` were still compiling around `15:33:50`, roughly 90s after the
  orchestrator started waiting for the inference pool.
- Patched `src/prime_rl/templates/gpu_layout_rl.sh.j2` again to preserve
  `compile_cache_root` across relaunches inside the same allocation. It still
  clears the vLLM RPC dir, but no longer `rm -rf`s the compile cache.
- Updated `skills/config/SKILL.md` to record both the corrected OLMo3 serving
  cap (`max_num_batched_tokens = 65536`, not `131072`) and the rule to preserve
  compile caches across relaunches.
- Validation after patch:
  - `uv run --no-sync rl @ configs/omni_math2/rl_olmo3_dpo_default_pipelinerl_speed.toml --dry-run --output-dir tmp/dryrun_default_cancel_eval_cachepreserve_20260507_1540`
  - `bash -n tmp/dryrun_default_cancel_eval_cachepreserve_20260507_1540/gpu_layout_rl.sh`
  - Rendered launcher has no `rm -rf "$compile_cache_root"`.

## 2026-05-07 15:41 UTC Default Cache-Preserve Relaunch

- Relaunched default from `joanv_cc_4node:omni-default` into
  `outputs/omni_math2_rlvr_canary/20260507_1541/dpo_default_pipelinerl_speed_cancel_eval_cachepreserve`.
- W&B run id: `562fd7c8a27140e1937da7330acbd272`; name:
  `olmo3-dpo-default-pipelinerl-speed-cancel-eval-cachepreserve-20260507-1541`.
- Four child Slurm steps are running: `4473657.77` through `4473657.80`.
- This relaunch cleared the startup failure: orchestrator logged
  `Inference pool ready` at `15:43:00`, then initialized NCCL broadcast and
  started initial eval at `15:43:03`.
- Initial eval is running with dynamic refill:
  `Eval dynamic refill enabled for omni-math2-baseline100: 672 workers across 14 inference clients`.
- Important measured bubble: although eval was logged at `15:43:03`, the eval
  env workers were still lazy-starting/loading. Sample worker logs first appear
  around `15:46:41-15:46:45`, and the first vLLM generation requests/metrics
  appear around `15:46:48-15:46:56`.
- `gpustat` pane at `15:46:38` showed all inference GPUs resident but idle
  (`0%` util). This was an env-worker startup bubble, not vLLM queue idleness.
- By `15:48:08`, generation was active: 14 inference GPUs were at `100%` util,
  trainer GPUs `nid011132:g2-g3` were idle as expected during blocking eval.
  Server logs showed `POST /v1/chat/completions` and per-server generation
  throughput roughly in the `1.2k-3.3k tok/s` range with `Waiting: 0`.
- Follow-up speed target: make `ZMQEnvServer` readiness or orchestrator env
  startup wait for env worker readiness, or otherwise prewarm eval workers before
  eval is allowed to claim the inference pool. Current `wait_for_server_startup`
  only proves the ZMQ frontend is healthy; it does not prove 128 worker
  processes have loaded the environment.

## 2026-05-07 15:54 UTC Default Cache-Preserve Initial Eval Complete

- Initial eval completed successfully at `15:53:41`:
  `638.28s`, `Avg@8=0.3725`, `Pass@8=0.6400`, no-response `0.0%`,
  completion length `5751.02 ± 5293.02`, truncation `16.4%`.
- This run is past startup failure and into train rollout generation. Live
  `gpustat` at `15:54:18` showed all 14 inference GPUs active (`95-100%` util)
  and trainer GPUs `nid011132:g2-g3` idle, which is expected during rollout
  generation before a train step.
- The eval wall time decomposes into two different bubbles:
  - Env-worker startup bubble: eval logged at `15:43:03`, but first completion
    did not land until about `15:47:18` (`1/800 [04:15]`). vLLM was ready, but
    the 128 eval env workers were still loading.
  - Heavy-tail bubble: after dynamic refill produced a large burst around
    `53-72%` progress, utilization collapsed to a few long rollouts near the
    end. Example: `gpustat` at `15:52:59` showed only 4 inference GPUs active,
    and `server_12.log` had `Running: 0` while `server_8.log` still had
    `Running: 3`.
- Current read: dynamic refill improved the middle of eval, but cannot remove
  the long-tail idle tail for fixed `100x8` blocking evals. The remaining eval
  speed work is env prewarm/readiness plus either accepting the tail, changing
  eval semantics, or reducing tail length via generation caps/timeouts.
- Correction/refinement: a readiness wait by itself would mostly make the eval
  timer honest; it would not save wall time unless the eval workers are started
  earlier or fewer workers are spawned. The concrete observed problem is
  `orchestrator.eval.num_workers = 128`: all 128 worker logs show initialization
  around `15:47:00-15:47:01`, despite orchestrator claiming eval env readiness
  at `15:42:57`. For async rollout I/O this is probably far too many processes.
  Do not stealth-change this mid-comparison, but consider `num_workers=32` or
  `auto` in the next speed iteration.
- Step 0 train rollout generation completed at `15:58:12`. Orchestrator logged:
  `Step 0 | Time: 908.46s | Reward: 0.5491 | Seq. Length: 2386.5 tokens/sample | Async Level: 0 | Max. Off-Policy Level: 0`.
  This time includes initial eval plus train rollout generation; do not read it
  as trainer throughput. Live `gpustat` at `15:58:19` showed all 14 inference
  GPUs still saturated and trainer GPUs beginning to work (`nid011132:g2-g3`
  around `11-12%` util), so trainer-step timing/MFU still needs the trainer log
  after the batch is consumed.
- Trainer step 0 completed at `15:59:11`:
  `Step 0 | Time: 968.94s | Loss: -0.0012 | Entropy: 1.1132 | Mismatch KL: 0.4226 | Grad. Norm: 0.2988 | LR: 1.00e-06 | Throughput: 0 tokens/s | MFU: 0.0% | Peak Mem.: 81.1 GiB`.
  Do not treat the zero throughput/MFU as the steady-state result; the time base
  still includes startup and initial eval. Weight update pause/resume worked:
  orchestrator paused inference at `15:59:12`, all engines paused immediately,
  and resumed by `15:59:16` while step 1 rollouts were already in flight.

## 2026-05-07 16:04 UTC Default Cache-Preserve Backpressure Check

- The `Timeout during comparison` / pause message is not currently a crash.
  Orchestrator step 3 hit the intended backpressure gate at `16:01:59`:
  `waiting for trainer process to complete checkpoint 2 (>1 step(s) ahead)`.
  It resumed at `16:02:59` after `60.13s` when checkpoint 2 was ready.
- Same pattern repeated for step 4: paused at `16:03:04` waiting for checkpoint
  3, resumed at `16:04:09` after `65.10s`, then paused inference for weight
  update. This is the async pipeline preventing unlimited off-policy drift.
- Trainer steady-state is still ramping but no longer zero:
  - Step 1 at `16:02:58`: `222.88s`, `2418 tokens/s`, `8.6%` MFU, peak
    memory `93.4 GiB`.
  - Step 2 at `16:04:08`: `65.28s`, `3927 tokens/s`, `13.9%` MFU, peak
    memory `93.4 GiB`.
- Live `gpustat` at `16:04:01` showed all 16 GPUs at `100%` utilization,
  including trainer GPUs `nid011132:g2-g3`. So the current bottleneck is not an
  idle-inference bubble; it is the orchestrator running ahead and correctly
  waiting for trainer checkpoints.
- Important interpretation: the very fast orchestrator steps 2/3 are mostly
  consuming already-buffered/inflight rollouts, not generating fresh 524k-token
  batches from scratch. That is expected after raising inflight capacity and
  allowing off-policy. It also means the pause messages are evidence that the
  pipeline has enough inference slack to outrun training.
- Follow-up at `16:07`: run is still healthy. Orchestrator reached step 7 and
  is repeatedly pausing/resuming on the one-step-ahead checkpoint gate:
  checkpoint waits after steps 5/6 were `49.62s` and `19.02s`.
- The literal `Timeout during comparison` appeared in console/tail output near
  pause/resume events, but `rg` over persisted `logs/` did not find the string.
  Current read: it is launcher/stdout noise or a transient comparison timeout,
  not a recorded orchestrator exception. It did not stop the run.
- Trainer ramp:
  - Step 3 at `16:05:16`: `63.53s`, `4913 tokens/s`, `17.4%` MFU.
  - Step 4 at `16:06:28`: `68.35s`, `5439 tokens/s`, `19.3%` MFU.
  - Step 5 at `16:07:37`: `64.05s`, `5786 tokens/s`, `20.5%` MFU.
- Live `gpustat` at `16:07:53`: 14 inference GPUs at `100%`; trainer GPUs were
  low (`11-12%`) because they were between/just after a train step. Earlier
  `16:05:39` snapshot had all 16 GPUs at `100%`.

## 2026-05-07 16:13 UTC Default Step-10 Eval Started Cleanly

- Trainer ramp through the interval boundary:
  - Step 6 at `16:08:45`: `63.79s`, `6060 tokens/s`, `21.5%` MFU.
  - Step 7 at `16:09:59`: `70.22s`, `6280 tokens/s`, `22.3%` MFU.
  - Step 8 at `16:11:13`: `69.07s`, `6453 tokens/s`, `22.9%` MFU.
  - Step 9 at `16:12:25`: `66.16s`, `6563 tokens/s`, `23.3%` MFU.
- Scheduler/eval-boundary fix worked for the important edge case. Orchestrator
  finished step 10 at `16:11:28`, then waited for trainer checkpoint 10 at step
  11. Once checkpoint 10 was ready, it paused/resumed inference for the weight
  update and then immediately logged:
  `Running evals at ckpt_step=10 for omni-math2-baseline100` at `16:12:30`.
- The eval cancellation guard also fired at the right time:
  `Cancelling in-flight training rollouts before starting evals to avoid congestion.`
  This means step-10 eval should not sit behind a stale step-11 train backlog
  like the previous MaxRL attempt.
- Step-10 eval has no 4-minute worker-start bubble this time. At `16:13:21`,
  roughly 51s after eval start, progress was already around `330/800`. Live
  `gpustat` at `16:13:30` showed all 16 GPUs at `100%`.

## 2026-05-07 16:24 UTC Default Step-10 Eval Complete + MaxRL Relaunch

- Default step-10 eval completed cleanly:
  `Evaluated omni-math2-baseline100 in 362.37s (Avg@8=0.3750, Pass@1: 0.3750, Pass@2: 0.4950, Pass@4: 0.5996, Pass@8: 0.6800, No-response: 0.0%, Completion Length: 6221.16 (±5384.11, ∈[104.00, 15360.00]), Truncated: 16.5%)`.
- This is the first clean evidence that the eval-boundary/cancel-inflight patch
  fixed the pathological step-10 eval congestion: previous polluted interval
  evals were around 1677-1735s, while this run's interval eval was 362.37s.
  Initial eval in the same default run was 638.28s, mostly hurt by lazy eval
  worker startup.
- Default trainer steady-state reached `7699 tokens/s`, `27.3%` MFU at trainer
  step 10 (`16:14:20`). The run was intentionally stopped after the interval
  eval, at `16:22:27`, by sending Ctrl-C to the `omni-default` tmux pane.
  `squeue --steps -j 4473657` afterwards showed only the allocation shell and
  batch step, so the Slurm allocation is still alive.
- Relaunched the sequential MaxRL smoke in pane `joanv_cc_4node:omni-maxrl`
  using the same ceteris-paribus speed topology/config:
  `configs/omni_math2/rl_olmo3_dpo_maxrl_pipelinerl_speed.toml`.
  Output dir:
  `outputs/omni_math2_rlvr_canary/20260507_1624/dpo_maxrl_pipelinerl_speed_cancel_eval_cachepreserve`.
  W&B name:
  `olmo3-dpo-maxrl-pipelinerl-speed-cancel-eval-cachepreserve-20260507-1624`.

## 2026-05-07 16:27 UTC MaxRL Relaunch Fix

- The `16:24` MaxRL launch failed before useful work. Root cause: the tmux
  launch command clobbered `PYTHONPATH`, so the orchestrator could import the
  shared `verifiers` package but not the repo-local
  `omni_math2_singleturn` task environment:
  `ValueError: Could not import 'omni_math2_singleturn' environment`.
- No child steps were left running after the failure; `squeue --steps -j 4473657`
  showed only the allocation shell and batch step.
- Relaunched MaxRL at `16:27` in fresh output dir
  `outputs/omni_math2_rlvr_canary/20260507_1627/dpo_maxrl_pipelinerl_speed_cancel_eval_cachepreserve`
  with `PYTHONPATH` ordered as documented in `skills/config/SKILL.md`:
  HF-task verifiers checkout, shared verifiers checkout, then
  repo-local `environments/omni_math2_singleturn`, preserving any existing path.
- Corrected MaxRL run passed the prior failure point: train and eval envs
  loaded, W&B run id is `80a5b1374ee145eba15055b9c87b510c`, inference pool
  became ready at `16:28:49`, and initial eval started at `16:28:52` with the
  intended `100x8` eval and dynamic refill `672 workers across 14 inference
  clients`.
- Initial eval had the same early visible-progress bubble as the default run.
  `gpustat` stayed at 0% through `16:31:57`, then by `16:32:38` all 14
  inference GPUs were at 100% while the two trainer GPUs remained idle, as
  expected during eval. Server logs confirm completions are flowing despite the
  pane's tqdm line still showing `0/800`: `server_0` and `server_13` began
  logging `POST /v1/chat/completions` around `16:32:38`, with per-server
  generation throughput roughly `1.4k-3.2k tokens/s`, running `31-46` requests,
  and `Waiting: 0 reqs`.

## 2026-05-07 16:41 UTC MaxRL Initial Eval Complete

- Corrected MaxRL initial eval completed at `16:39:50`:
  `Evaluated omni-math2-baseline100 in 657.41s (Avg@8=0.3650, Pass@1: 0.3650, Pass@2: 0.4782, Pass@4: 0.5797, Pass@8: 0.6600, No-response: 0.0%, Completion Length: 5897.53 (±5376.99, ∈[133.00, 15360.00]), Truncated: 17.8%)`.
- This is ceteris-paribus comparable to the default initial eval from the same
  speed config family (`638.28s`, `Avg@8=0.3725`, `Pass@8=0.6400`). It is not a
  meaningful initial-eval speedup; both runs pay the lazy eval-worker startup
  cost and both have the long-completion tail.
- The eval tail is visible in tqdm: progress reached `797/800` at `10:15`, then
  the final three completions stretched total eval time to `10:57`. This is the
  same fixed-batch tail problem seen in the default run.
- After eval, MaxRL moved into train rollout generation. At `16:40:50`,
  `gpustat` showed the intended topology alive: 14 inference GPUs at `100%`
  util with ~87 GiB each, and the 2 trainer GPUs allocated but idle-ish while
  waiting for the first batch. Total cluster GPU util was `87%`.
- Current run state: MaxRL is healthy and generating the first 524288-token
  train batch in
  `outputs/omni_math2_rlvr_canary/20260507_1627/dpo_maxrl_pipelinerl_speed_cancel_eval_cachepreserve`.
  Next useful evidence is trainer step timing/MFU after the first batch lands;
  step 0 will still include startup/eval in its wall clock, so steady-state
  interpretation should start from later trainer steps.

## 2026-05-07 16:46 UTC MaxRL Step 0 Complete

- MaxRL step 0 rollout generation completed at `16:44:28`:
  `Step 0 | Time: 935.70s | Reward: 0.5046 | Seq. Length: 2555.1 tokens/sample | Async Level: 0 | Max. Off-Policy Level: 0`.
  As with default, this includes startup plus initial eval plus first train
  batch generation; do not treat it as steady-state training speed.
- Trainer step 0 completed at `16:45:25`:
  `Step 0 | Time: 992.40s | Loss: -0.0065 | Entropy: 1.3464 | Mismatch KL: 0.5094 | Grad. Norm: 1.5547 | LR: 1.00e-06 | Throughput: 0 tokens/s | MFU: 0.0% | Peak Mem.: 81.2 GiB`.
  The zero throughput/MFU is the same startup-timebase artifact as the default
  run; wait for trainer steps 1+ before comparing speed.
- Weight-update coordination worked after trainer step 0: orchestrator paused
  inference at `16:45:25`, all engines paused, and all engines resumed at
  `16:45:29`.
- At `16:45:54`, MaxRL was generating step 1 and `gpustat` showed all 14
  inference GPUs at `100%`; trainer GPUs had the model resident
  (`~84-85 GiB`) but were not actively training at that instant. No Slurm
  child-step leak: only allocation shell/batch plus the four active run steps.

## 2026-05-07 16:49 UTC MaxRL Early Ramp

- First non-startup-polluted MaxRL trainer point:
  `Step 1 | Time: 130.83s | Loss: -0.0065 | Entropy: 1.8535 | Mismatch KL: 0.8779 | Grad. Norm: 2.4531 | LR: 1.00e-06 | Throughput: 4068 tokens/s | MFU: 14.5% | Peak Mem.: 93.4 GiB`.
  This is faster than default trainer step 1 (`222.88s`, `2418 tok/s`,
  `8.6%` MFU), but default ramped quickly after step 1, so do not call the
  comparison from one point.
- Orchestrator/train rollout side:
  - Step 1 completed at `16:46:32`: `122.67s`, reward `0.4792`, sequence
    length `5614.7`, max off-policy level `1`.
  - Step 2 completed at `16:48:05`: `92.45s`, reward `0.5227`, sequence
    length `6023.6`, max off-policy level `2`.
  - Step 3 completed at `16:48:14`: `8.66s`, reward `0.3036`, sequence length
    `9695.7`, async level `1`, max off-policy level `2`. This very short step
    is buffered/off-policy consumption, not fresh generation speed.
- The one-step-ahead gate is active again: at `16:48:14`, orchestrator step 4
  paused waiting for trainer checkpoint 3. Same interpretation as default:
  inference/orchestrator has enough slack to outrun the trainer, and the
  backpressure gate is preventing unbounded policy lag.
- `gpustat` at `16:48:30` showed all 14 inference GPUs at `100%`; trainer GPUs
  had high memory and power draw but `0%` instantaneous util, likely between
  trainer kernels / waiting for checkpoint boundary at the sample instant.

## 2026-05-07 16:52 UTC MaxRL Trainer Ramp Through Step 4

- MaxRL trainer ramp:
  - Step 2 at `16:49:10`: `85.91s`, `4880 tokens/s`, `17.3%` MFU.
  - Step 3 at `16:50:18`: `63.27s`, `5869 tokens/s`, `20.9%` MFU.
  - Step 4 at `16:51:25`: `62.85s`, `6310 tokens/s`, `22.4%` MFU.
- Default comparison points from the earlier same-allocation run:
  - Step 2: `65.28s`, `3927 tokens/s`, `13.9%` MFU.
  - Step 3: `63.53s`, `4913 tokens/s`, `17.4%` MFU.
  - Step 4: `68.35s`, `5439 tokens/s`, `19.3%` MFU.
  Current read: MaxRL is not faster on every wall-clock step, but by steps 3-4
  it is producing higher trainer throughput/MFU than default. Continue to step
  10 before making the final call.
- Orchestrator remains ahead of trainer:
  - Step 4 paused `56.13s` waiting for checkpoint 3.
  - Step 5 paused `50.05s` waiting for checkpoint 4.
  - Step 6 paused `26.03s` waiting for checkpoint 5.
  This is again backpressure doing its job, not a crash.
- The literal `Timeout during comparison` appeared again around the
  `16:49:10` pause/resume sequence, but the run immediately resumed and
  progressed through trainer steps 3 and 4. Treat it as noisy live-stream /
  wrapper output unless it starts correlating with aborted steps.
- Live `gpustat` at `16:52:07` showed all 16 GPUs at `100%`, including trainer
  GPUs, with total GPU util `100%`. This is the first clean all-GPU-busy MaxRL
  snapshot after the ramp.

## 2026-05-07 16:57 UTC MaxRL Through Trainer Step 8

- MaxRL trainer ramp continued:
  - Step 5 at `16:52:32`: `61.73s`, `6615 tokens/s`, `23.5%` MFU.
  - Step 6 at `16:53:44`: `67.95s`, `6831 tokens/s`, `24.3%` MFU.
  - Step 7 at `16:55:34`: `105.27s`, `6569 tokens/s`, `23.3%` MFU.
  - Step 8 at `16:56:47`: `69.54s`, `6722 tokens/s`, `23.9%` MFU.
- Default same-run comparison:
  - Step 5: `64.05s`, `5786 tokens/s`, `20.5%` MFU.
  - Step 6: `63.79s`, `6060 tokens/s`, `21.5%` MFU.
  - Step 7: `70.22s`, `6280 tokens/s`, `22.3%` MFU.
  - Step 8: `69.07s`, `6453 tokens/s`, `22.9%` MFU.
  Current read: MaxRL is ahead on throughput/MFU but not by a huge margin, and
  step 7 was slower wall-clock than default. It is definitely not a dramatic
  training-speed bulldozer under this topology/config.
- Orchestrator reached step 9 at `16:55:30`, generated it quickly, saved
  checkpoint at step 10 at `16:56:13`, and then paused at orchestrator step 10
  waiting for trainer checkpoint 9. It resumed at `16:56:48`.
- `gpustat` at `16:56:46` again showed all 16 GPUs at `100%`, total GPU util
  `100%`. Utilization is not the obvious issue; the remaining question is
  trainer MFU and eval-boundary behavior.

## 2026-05-07 17:00 UTC MaxRL Step-10 Eval Started Cleanly

- MaxRL trainer step 9 completed at `16:58:01`:
  `Step 9 | Time: 70.07s | Loss: -0.0034 | Entropy: 2.0779 | Mismatch KL: 0.7126 | Grad. Norm: 1.8906 | LR: 1.00e-06 | Throughput: 6834 tokens/s | MFU: 24.3% | Peak Mem.: 93.4 GiB`.
  This is ahead of default step 9 (`6563 tok/s`, `23.3%` MFU) but still below
  default step 10 (`7699 tok/s`, `27.3%` MFU).
- Orchestrator step 10 finished rollout generation at `16:58:22`:
  `Step 10 | Time: 127.57s | Reward: 0.4231 | Seq. Length: 5517.2 tokens/sample | Async Level: 0 | Max. Off-Policy Level: 10`.
  This is very off-policy, exactly as expected after the speed config changes
  and the user's stated preference to accept more lag with IS enabled.
- Interval eval boundary behaved correctly under MaxRL:
  - Trainer saved step-10 checkpoint at `16:58:05`, then weight checkpoint at
    `16:58:19`.
  - Orchestrator started step 11 and immediately logged
    `Running evals at ckpt_step=10 for omni-math2-baseline100` at `16:58:22`.
  - The cancellation guard fired:
    `Cancelling in-flight training rollouts before starting evals to avoid congestion.`
  - Eval started at `16:58:23` with dynamic refill `672 workers across 14
    inference clients`.
- At `16:59:31`, the interval eval was already around `383/800` at `1:06`.
  This confirms no repeat of the old multi-minute eval-worker startup bubble
  on interval eval.

## 2026-05-07 17:07 UTC MaxRL Step-10 Eval Completed

- MaxRL step-10 interval eval completed at `17:05:29`:
  `425.86s`, `Avg@8=0.3762`, `Pass@1=0.3762`, `Pass@2=0.4807`,
  `Pass@4=0.5669`, `Pass@8=0.6400`, no-response `0.0%`, completion length
  `5929.53 ± 5306.44`, range `[104,15360]`, truncated `16.2%`.
- Default step-10 interval eval was faster: `362.37s` with comparable shape
  (`Avg@8=0.3750`, `Pass@8=0.6800`, truncated `16.5%`). So MaxRL did not buy
  eval speed here. The eval cancellation guard did work, but the MaxRL interval
  eval still had a long tail.
- MaxRL trainer step 10 completed during the eval at `16:59:39`:
  `70.79s`, `7247 tokens/s`, `25.8%` MFU, peak memory `93.4 GiB`.
  This is better than MaxRL step 9 but still below default step 10
  (`7699 tokens/s`, `27.3%` MFU).
- The literal `Timeout during comparison` appeared during the interval-eval
  tqdm stream again, but the eval completed successfully. Current read: noisy
  stream/wrapper output, not a run-stopping failure.
- Live `gpustat` at `17:06:43` showed 14/16 GPUs at `100%`; the two trainer
  GPUs were idle while the orchestrator had resumed rollouts after eval. Total
  instantaneous util was `87%`.
- Speed verdict from the same-allocation ceteris-paribus pair: the pipeline
  changes fixed the eval boundary/backlog problem and kept GPUs much busier
  during training, but MaxRL is only mildly ahead on trainer throughput/MFU and
  is not clearly faster end-to-end. Next worthwhile speed work is not "more
  concurrency"; it is training-side batch/compute efficiency, eval long-tail
  reduction, and measuring whether larger train batches lift MFU without
  making policy lag/importance weights useless.
- Stopped MaxRL with Ctrl-C after the step-10 eval. The resulting
  `KeyboardInterrupt`/`srun ... Killed` lines are expected shutdown noise. The
  Slurm allocation was left alive; `squeue --steps -j 4473657` showed only
  `4473657.0` and `4473657.batch` afterward.

## 2026-05-08 10:00 UTC Step-10 Diagnosis

Added analysis artifact:

- `tmp/analyze_olmo3_step10_canaries.py`
- `tmp/olmo3_step10_diag.json`
- `tmp/wandb_step10_selected_history.json`

Scope analyzed:

- Default run:
  `outputs/omni_math2_rlvr_canary/20260507_1541/dpo_default_pipelinerl_speed_cancel_eval_cachepreserve`
- MaxRL run:
  `outputs/omni_math2_rlvr_canary/20260507_1627/dpo_maxrl_pipelinerl_speed_cancel_eval_cachepreserve`
- W&B:
  - default:
    `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/562fd7c8a27140e1937da7330acbd272`
  - MaxRL:
    `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/80a5b1374ee145eba15055b9c87b510c`

Main conclusions:

- The 10-step eval did show tiny positive `Avg@8` movement, but not enough to
  be statistically meaningful on `100 × 8`.
  - default: `0.3725 -> 0.3750` (`+0.25 pp`)
  - MaxRL: `0.3650 -> 0.3762` (`+1.125 pp`)
  - Paired prompt bootstrap 95% CIs cross zero for every pass@k metric.
    Approximate MDE for an 80%-power 100-prompt paired eval is about `5 pp`
    on p@1 and about `10 pp` on p@8, so a 10-step run would need a large
    effect to be visible.
- There is hidden reward-component movement:
  - default `math_verify_score`: `0.2150 -> 0.3025` (`+8.75 pp`)
  - MaxRL `math_verify_score`: `0.2338 -> 0.3063` (`+7.25 pp`)
  - final `correct_answer` barely moved because judge-only wins fell:
    default judge-only correct `126 -> 58`, MaxRL `105 -> 56`.
  - This looks like the model became more symbolically-verifiable without
    becoming much more finally-correct under the hybrid judge metric. Audit
    the `not math_verify && correct_answer` cases before declaring no learning.
- The learner did not actually update on "so much data" in independent-prompt
  terms. Over steps 0-10:
  - default trained on `1144` rollouts / `143` prompt groups, about `13.0`
    groups per step.
  - MaxRL trained on `1184` rollouts / `148` prompt groups, about `13.5`
    groups per step.
  - W&B filtering shows about `60%` of generated rollout groups were evicted
    as easy/hard before the gradient saw them. This is intended online
    difficulty filtering, but it means token volume is not equivalent to many
    independent training prompts.
- The speed config was very stale by step 10:
  - default last off-policy mean/max: `5.64 / 9`
  - MaxRL last off-policy mean/max: `6.08 / 10`
  - mean trainer importance ratio was about `0.69`; mean mismatch KL about
    `0.77-0.78`; DPPO masked about `23-25%` of trainable tokens, mostly
    low-side masks.
  - Importance-sampling clipping itself barely fired (`~0.2%`), so the main
    issue is not top-end ratio explosions. It is stale-policy mismatch and
    mask/drop of useful gradient mass.
- Data split is not perfectly clean:
  - ID overlap between train and eval-600 is `0`.
  - Exact problem-text overlap is `7`.
  - One duplicate problem is in the active eval-100 subset: eval id `3077`,
    train id `2933`.
  - This probably does not explain weak learning, but it is a real data hygiene
    bug and should be fixed before using tiny eval deltas as evidence.
- Topology verdict:
  - `14 inference / 2 trainer` is good for inference/eval stress and proved
    the dynamic-refill/eval-boundary fixes, but steady training is now
    trainer/backpressure limited.
  - The next 4-node single-run topology to try is `12 inference / 4 trainer`.
    Keep eval dynamic refill and measure whether trainer step time drops
    enough to compensate for fewer inference GPUs.
  - If `12i/4t` still spends most time waiting on trainer, try `10i/6t`.
    `8i/8t` is likely too inference-poor for this long-completion workload.
- Batch-size verdict:
  - Bigger `token_batch_size` can reduce gradient noise and amortize trainer
    overhead, but it will also reduce update cadence and may worsen stale
    queue age if inference stays ahead.
  - Do not jump straight to `1048576`. First test `786432` after changing to
    `12i/4t`, and add ESS-style logging for importance weights.
- PipelineRL paper read:
  - The relevant lesson is not "make a huge stale queue". It is balancing
    trainer accelerator count, generation window size, and max lag so that
    throughput improves without destroying learning effectiveness.
  - The paper explicitly frames learning speed as throughput times learning
    effectiveness; our current config improved throughput but likely paid too
    much in off-policy/masking/noisy eval terms.

Next diagnosis/fix checklist:

1. Add W&B/local metrics for normalized ESS of token-level importance weights,
   ideally by off-policy age bucket.
2. Add a raw pre-filter difficulty histogram so `effective_batch_size=1` is not
   misleading after online difficulty filtering.
3. Extract and inspect eval transitions where `math_verify_score` improved but
   `correct_answer` did not, and where judge-only correctness disappeared.
4. Repair or exclude the 7 train/eval prompt duplicates, especially eval id
   `3077`.
5. Run a `12i/4t` default canary with `max_inflight_rollouts` closer to
   `1536` and `max_off_policy_steps = 16` before increasing
   `token_batch_size`.
6. Only after `12i/4t` is measured, test `token_batch_size = 786432`.

## 2026-05-09 23:54 UTC Utilization / Benchmark Clarification

Current gpustat snapshot shows all 16 GPUs at `100%` util with
`1,447,875 / 1,565,936 MiB` allocated (`~92.5%` of visible GPU memory).
That should not be confused with useful trainer MFU. The trainer is at step
92 with last-step `8715 tok/s`, `30.9%` MFU, and `92.9 GiB` peak memory;
trainer steps 80-92 average `171.75s`, `8692 tok/s`, and `30.85%` MFU.
vLLM remains 14/14 OK with `249` running, `0` waiting, mean KV `0.350`, and
preemptions unchanged at `64`.

The top-level PrimeRL `bench` flag is not enabled. The temp configs omit
`bench = true`, and `src/prime_rl/configs/rl.py` defaults it to `False`.
Leave it off for this real canary: enabling it would set trainer benchmark
mode, orchestrator benchmark mode, and fake trainer data. The current MFU
numbers are normal live trainer log metrics, not isolated benchmark-mode
results.

## 2026-05-10 00:12 UTC Eval-100 Started

Default reached the eval-100 boundary. Trainer reached step 99 with post-90
average `169.71s`, `8695 tok/s`, and `30.86%` MFU. Orchestrator completed
step 100 at `00:09:27` (`157.25s`, reward `0.4012`, seq length `6622.9`,
async/off-policy `1/2`) after saving ckpt-100 at `00:06:48`. Trainer saved
checkpoint 100 at `00:12:20`, and orchestrator then started
`Running evals at ckpt_step=100 for omni-math2-baseline100`.

Interpret this as checkpoint/trainer-boundary lag, not inference starvation:
at eval start vLLM was 14/14 OK with `470` running, `0` waiting, mean KV
`0.099`, and preemptions still `64`. A gpustat snapshot at `00:12:24` showed
the 14 inference GPUs at `100%` while the two trainer GPUs were low-util but
memory-resident. Next important evidence is eval-100 completion metrics and
whether post-eval refill resumes without the old hard bubble.

## 2026-05-10 00:19 UTC Eval-100 Completed

Eval-100 completed in `399.41s`:

- `Avg@8=0.3563`
- `Pass@1=0.3563`
- `Pass@2=0.4521`
- `Pass@4=0.5356`
- `Pass@8=0.6000`
- no-response `0.0%`
- completion length `5959.01 ± 5384.23`
- truncated `16.9%`

Versus eval-50 (`Avg@8=0.3638`, `Pass@8=0.6700`), this is worse across all
pass metrics: Avg@8/Pass@1 `-0.75 pp`, Pass@2 `-2.97 pp`, Pass@4 `-5.08 pp`,
Pass@8 `-7.00 pp`; truncation increased `+1.9 pp`. Do not overclaim from
100 prompts, but this is a negative direction-of-travel signal.

Step 101 completed right after eval and charged the boundary/eval wall time
into one orchestrator step: `575.64s`, reward `0.3359`, seq length `6429.9`,
async/off-policy `0/2`. Trainer steps 100-101 cooled to `28.8%` and `27.3%`
MFU (`8121` then `7679 tok/s`). vLLM still had 14/14 OK with `0` waiting, but
preemptions rose from `64` to `91`. The next check should focus on steps
102-105 to distinguish harmless eval-time accounting from a real post-eval
refill/staleness bubble.

## 2026-05-10 00:31 UTC Post-Eval-100 Recovery Check

The orchestrator mostly recovered after the boundary/eval step, but trainer
MFU stayed degraded. Trainer steps 100-104 average `186.74s`, `7636 tok/s`,
and `27.12%` MFU; steps 102-104 are flat around `26.5%` MFU and
`~7460 tok/s`, down from the pre-eval `~30.9%` / `~8700 tok/s` plateau.

Orchestrator steps 102-106 were `241.7,104.9,130.7,155.8,115.8s`, so step 101
looks mostly like eval/checkpoint wall-time accounting rather than a persistent
rollout refill stall. vLLM remains 14/14 OK, `251` running, `0` waiting, mean
KV `0.163`, and preemptions stable at `91` after the eval jump. Post-100
checkpoint waits are `170.32,19.06,72.14,105.24s`.

Current read: still trainer/checkpoint paced, and now in a worse trainer MFU
regime. Keep monitoring; do not increase inference-side queueing in response
to this.

## 2026-05-10 00:44 UTC MFU Recovery Check

Trainer MFU has not recovered by step 108. Trainer steps 100-108 average
`181.66s`, `7574 tok/s`, `26.90%` MFU; steps 106-108 average `170.61s`,
`7507 tok/s`, `26.67%` MFU. This is still materially below the pre-eval
`~30.9%` / `~8700 tok/s` plateau.

Orchestrator reached step 109. Steps 106-109 average `171.17s`, reward
`0.4241`, seq length `5692.4`, and recent async/off-policy is back at `1/1`.
Post-100 checkpoint waits are high:
`170.32,19.06,72.14,105.24,182.22,139.23,180.29,132.27s` (mean `125.10s`).
vLLM remains 14/14 OK with `252` running, `0` waiting, mean KV `0.370`, and
preemptions stable at `91`; gpustat shows aggregate `99%` util and `~92.6%`
memory.

Current read: persistent post-eval trainer-side throughput regression. Do not
treat this as a reason to add inference queue depth; the inference side is not
waiting. At current pace this allocation cannot finish Default 1000, so MaxRL
will require a later resume/allocation unless Default is intentionally stopped.

## 2026-05-10 00:56 UTC MFU Recovery Correction

The post-eval trainer slowdown recovered by steps 110-112. Trainer steps
110-112 average `164.90s`, `8520 tok/s`, and `30.23%` MFU; step 112 reached
`8619 tok/s` / `30.6%`. So the post-eval MFU drop was a roughly 10-step
transient, not a permanent throughput regime.

Orchestrator reached step 114. Steps 110-114 average `166.48s`, reward
`0.4335`, seq length `5721.0`; recent async/off-policy is mostly `1/1`, with
step 113 at off-policy `2`. Post-100 checkpoint waits remain high
(`110.79s` mean, max `182.22s`), so checkpoint/backpressure is still the
lasting performance problem. vLLM remains 14/14 OK with `0` waiting and
preemptions stable at `91`.

Next useful milestone is eval-150: check whether pass metrics recover or the
negative eval-100 direction repeats, and whether the same ~10-step post-eval
MFU dip recurs.

## 2026-05-10 00:58 UTC Resume Artifact Check

Real checkpoint files exist for resume:

- `outputs/omni_math2_rlvr_canary/20260509_1825/dpo_default_14i2t_bs256_eval50_1000step/checkpoints/step_50/trainer/.metadata`
- `.../step_50/trainer/__0_0.distcp`
- `.../step_50/trainer/__1_0.distcp`
- `.../step_100/trainer/.metadata`
- `.../step_100/trainer/__0_0.distcp`
- `.../step_100/trainer/__1_0.distcp`

The resume launcher is
`tmp/resume_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh`; it uses
`--ckpt.resume_step -1`, derives four hosts from `SLURM_JOB_NODELIST`, and
passes `--deployment.hosts` as JSON. Latest live scrape: trainer reached step
113, with steps 110-113 averaging `166.52s`, `8550 tok/s`, `30.35%` MFU.
Orchestrator reached step 115, with steps 110-115 averaging `163.81s`, reward
`0.4394`, seq length `5816.8`. vLLM remains 14/14 OK, `0` waiting.

## 2026-05-10 01:14 UTC Step-120 Band

Trainer reached step 119. Steps 110-119 average `156.24s`, `8598 tok/s`,
`30.52%` MFU; steps 115-119 average `143.53s`, `8624 tok/s`, `30.62%` MFU.
This confirms recovery to the warm plateau after the eval-100 transient.

Orchestrator reached step 121. Steps 115-121 average `162.48s`, reward
`0.4185`, seq length `5026.6`, latest async/off-policy `1/2`. Checkpoint waits
remain the main drag: post-110 waits are
`66.10,53.07,82.11,127.40,143.23,174.38,167.27,159.38,85.14,76.19,45.04s`
(mean `107.21s`). vLLM remains 14/14 OK with `186` running, `0` waiting,
mean KV `0.376`, preemptions stable at `91`. Eval-150 has not started.

## 2026-05-10 01:35 UTC Step-128 Band

Trainer reached step 127. Steps 120-127 average `170.04s`, `8660 tok/s`,
`30.75%` MFU, peak memory `92.9 GiB`. Orchestrator reached step 128. Steps
120-128 average `160.22s`, reward `0.4066`, seq length `5621.2`; async/off
policy is mostly `1/1`, with steps 121/122/125 at off-policy `2`.

Checkpoint waits are still the main drag. Post-120 waits are
`45.04,97.10,134.23,123.20,139.21,167.41,177.38,141.38,136.17s`, mean
`129.01s`. vLLM remains 14/14 OK with `250` running, `0` waiting, mean KV
`0.244`, preemptions stable at `91`. Eval-150 has not started.

## 2026-05-10 02:02 UTC Step-140 Approach

Trainer reached step 137. Steps 130-137 average `160.88s`, `8633 tok/s`,
`30.66%` MFU, peak memory `92.9 GiB`. Orchestrator reached step 139. Steps
130-139 average `156.13s`, reward `0.4111`, seq length `5375.2`; async/off
policy remains mostly `1/1` with occasional off-policy `2`.

Checkpoint waits are still the drag: post-130 waits are
`94.17,110.21,149.38,168.25,162.28,92.17,119.14,139.28,128.24s`, mean
`129.24s`. vLLM remains 14/14 OK with `207` running, `0` waiting, mean KV
`0.232`, and preemptions stable at `91`. Eval-150 has not started.

## 2026-05-10 02:28 UTC Pre-Eval-150

Trainer reached step 147. Steps 140-147 average `158.14s`, `8632 tok/s`,
`30.65%` MFU, peak memory `92.9 GiB`. Orchestrator reached step 148. Steps
140-148 average `163.53s`, reward `0.4160`, seq length `5423.8`; steps
143/144/145 hit off-policy `2`, step 146 was off-policy `0`, otherwise recent
steps are `1`.

Checkpoint waits remain high: post-140 waits are
`94.10,59.06,83.14,143.21,165.27,189.24,127.22,126.32,126.18s`, mean
`123.75s`. vLLM remains 14/14 OK with `252` running, `0` waiting, mean KV
`0.147`, and preemptions stable at `91`. Checkpoint/eval 150 has not started,
and no `step_150` checkpoint files exist yet.

## 2026-05-10 02:45 UTC Eval-150 Completed

Boundary timing: orchestrator completed step 150 at `02:32:46` (`154.63s`,
reward `0.4180`, seq length `5900.6`, async/off-policy `1/1`), then paused
for checkpoint 150. It resumed after `138.23s`, started eval at `02:35:06`,
and completed eval at `02:41:52` in `405.22s`.

Eval-150 metrics:

- `Avg@8=0.3850`
- `Pass@1=0.3850`
- `Pass@2=0.4889`
- `Pass@4=0.5781`
- `Pass@8=0.6400`
- no-response `0.0%`
- completion length `5716.65 ± 5249.57`
- truncated `15.2%`

Versus eval-100, this rebounded: Avg@8/Pass@1 `+2.87 pp`, Pass@2 `+3.68 pp`,
Pass@4 `+4.25 pp`, Pass@8 `+4.00 pp`, truncation `-1.7 pp`. Versus eval-50,
it is mixed: Avg@8/Pass@1 `+2.12 pp`, Pass@2 `+0.71 pp`, Pass@4 `-0.83 pp`,
Pass@8 `-3.00 pp`, truncation `+0.2 pp`.

Post-eval bubble recurred. Orchestrator step 151 took `549.5s`, reward
`0.346`, async/off-policy `0/3`. Trainer step 151 dropped to `7562 tok/s` /
`26.8%` MFU. vLLM remained 14/14 OK with `0` waiting, but preemptions rose
from `91` to `96`. Checkpoint files for `step_150/trainer` now exist:
`.metadata`, `__0_0.distcp`, and `__1_0.distcp`; latest real resume point is
step 150.

## 2026-05-10 03:02 UTC Post-Eval-150 Recovery Check

Trainer MFU has not recovered yet by step 157. Trainer steps 150-157 average
`174.64s`, `7677 tok/s`, `27.24%` MFU; steps 156-157 average `187.50s`,
`7601 tok/s`, `27.00%` MFU. This is longer-lived than the eval-100 MFU dip at
the same relative age.

Orchestrator reached step 158. Steps 150-158 average `200.80s`, reward
`0.4107`, seq length `5912.1`. Step 151 was the boundary/eval bubble
(`549.5s`, off-policy `3`), and step 157 also hit off-policy `3`. Post-150
checkpoint waits are `138.23,37.11,47.05,82.11,120.26,175.29,196.34,129.29s`
(mean `115.71s`). vLLM remains 14/14 OK, `253` running, `0` waiting, mean KV
`0.180`, preemptions stable at `96`.

## 2026-05-10 03:18 UTC Post-Eval-150 Recovery Confirmed

Trainer MFU recovered at step 160. Steps 160-162 average `172.39s`,
`8653 tok/s`, `30.73%` MFU, after steps 151-159 sat around `26.7-27.3%` MFU.
So eval-150 produced the same rough pattern as eval-100: one large
boundary/eval wall-clock step plus about a 9-10 trainer-step MFU cooldown, then
recovery.

Orchestrator reached step 164. Steps 160-164 average `171.45s`, reward
`0.4142`, seq length `5766.0`, with steps 160/161 at off-policy `2`.
Post-150 checkpoint waits remain severe: mean `130.23s`, tail
`147.32,151.24,164.28,154.25,150.19s`. vLLM remains 14/14 OK with `0`
waiting and preemptions stable at `96`.

## 2026-05-10 03:50 UTC Step-175 Approach

Trainer reached step 173. Steps 170-173 average `177.61s`, `8648 tok/s`,
`30.70%` MFU, peak memory `92.9 GiB`. Orchestrator reached step 175. Steps
170-175 average `178.05s`, reward `0.3900`, seq length `5900.8`; step 173 hit
off-policy `2`, step 174 was off-policy `0`, otherwise recent steps were `1`.

Checkpoint waits remain severe: post-170 waits are
`124.23,135.23,172.22,177.26,140.30s`, mean `149.85s`. vLLM remains 14/14 OK
with `54` running, `0` waiting, mean KV `0.141`, preemptions stable at `96`.
Eval-200 has not started.

## 2026-05-10 04:21 UTC Step-185 Band

Trainer reached step 184. Steps 180-184 average `156.60s`, `8579 tok/s`,
`30.46%` MFU, peak memory `92.9 GiB`. Orchestrator reached step 186. Steps
180-186 average `161.70s`, reward `0.4452`, seq length `5538.8`; recent
async/off-policy is all `1/1`.

Post-180 checkpoint waits remain high:
`132.23,134.26,148.24,107.24,109.23,170.28s`, mean `133.58s`. vLLM remains
14/14 OK with `13` running, `0` waiting, mean KV `0.043`, preemptions stable
at `96`. Eval-200 has not started.

## 2026-05-10 04:53 UTC Pre-Eval-200

Trainer reached step 196. Steps 190-196 average `168.61s`, `8609 tok/s`,
`30.59%` MFU, peak memory `92.9 GiB`. Orchestrator reached step 198. Steps
190-198 average `163.74s`, reward `0.4257`, seq length `5911.7`; mostly
async/off-policy `1/1`, with step 197 at off-policy `2`.

Checkpoint waits post-190 are
`133.26,140.29,145.24,145.24,136.23,129.15,145.34,165.29s`, mean `142.50s`.
vLLM remains 14/14 OK with `212` running, `0` waiting, mean KV `0.283`,
preemptions stable at `96`. Eval-200 has not started, and no `step_200`
checkpoint files exist yet.

## 2026-05-10 05:14 UTC Eval-200 Completed

Boundary timing: orchestrator completed step 200 at `04:59:54` (`205.10s`,
reward `0.3789`, seq length `6258.6`, async/off-policy `1/1`), then waited
`171.30s` for checkpoint 200. Eval started at `05:02:48` and completed at
`05:08:43` in `355.34s`.

Eval-200 metrics:

- `Avg@8=0.3613`
- `Pass@1=0.3613`
- `Pass@2=0.4686`
- `Pass@4=0.5607`
- `Pass@8=0.6200`
- no-response `0.0%`
- completion length `5675.15 ± 5212.35`
- truncated `14.5%`

Versus eval-150, all pass metrics fell: Avg@8/Pass@1 `-2.37 pp`, Pass@2
`-2.03 pp`, Pass@4 `-1.74 pp`, Pass@8 `-2.00 pp`, while truncation improved
`-0.7 pp`. Versus eval-50: Avg@8/Pass@1 `-0.25 pp`, Pass@2 `-1.32 pp`,
Pass@4 `-2.57 pp`, Pass@8 `-5.00 pp`, truncation `-0.5 pp`.

Post-eval bubble recurred. Orchestrator step 201 took `561.3s`; trainer steps
201-202 dropped to `27.4%` and `27.3%` MFU. vLLM remained 14/14 OK with
`0` waiting and preemptions stable at `96`. Checkpoint files for
`step_200/trainer` now exist (`.metadata`, `__0_0.distcp`, `__1_0.distcp`),
so latest real resume point is step 200.

## 2026-05-10 05:31 UTC Post-Eval-200 Recovery Check

Trainer MFU has not recovered by step 208. Trainer steps 200-208 average
`164.76s`, `7737 tok/s`, `27.48%` MFU; steps 201-208 are stuck around
`26.8-27.4%` MFU. This is a longer cooldown than the eval-100/eval-150
recoveries.

Orchestrator reached step 210. Steps 200-210 average `190.72s`, reward
`0.4170`, seq length `5386.9`. Step 201 was the boundary/eval bubble
(`561.3s`); steps 204-206 hit off-policy `2`. Post-200 checkpoint waits
average `119.29s`, with tail
`171.30,41.06,45.06,154.16,141.28,191.23,138.16,118.23,73.10s`. vLLM remains
14/14 OK, `253` running, `0` waiting, mean KV `0.363`, preemptions stable at
`96`.

## 2026-05-10 05:40 UTC

**Step**: trainer 211 / 1000; orchestrator 213 / 1000.
**Health**: Healthy; post-eval-200 MFU has recovered.

**Trainer**: Steps 210-211 average `168.06s`, `8550 tok/s`, and `30.35%` MFU.
Peak memory remains `92.9 GiB`.

**Orchestrator**: Steps 210-213 average `151.53s`, reward `0.4154`, seq length
`5893.7`, with off-policy levels `1,2,2,1`.

**GPU utilization**: A 05:38 UTC `gpustat` snapshot showed only `78%` aggregate
GPU util, but 05:40 UTC showed all 16 GPUs at `100%`, using
`1451921/1565936 MiB` total memory. Treat instantaneous `gpustat` as noisy;
trainer MFU is the better run-level signal.

**Benchmark mode**: PrimeRL `bench = true` is not enabled. These are live
training MFU/throughput numbers, not fake-data benchmark numbers.

## 2026-05-10 05:55 UTC

**Step**: trainer 216 / 1000; orchestrator 218 / 1000.
**Health**: Healthy; eval-250 has not started.

**Trainer**: Steps 210-216 average `169.37s`, `8584 tok/s`, and `30.47%` MFU.
Steps 215-216 average `162.13s`, `8612 tok/s`, and `30.60%` MFU. Peak memory
remains `92.9 GiB`.

**Orchestrator**: Steps 210-218 average `162.10s`, reward `0.4090`, seq length
`5823.5`; recent off-policy is mostly `1` after the step-211/212 `2`s.

**Backpressure**: Checkpoint waits remain material but softened after step 215:
post-210 mean `150.53s`, post-215 mean `129.57s`, tail
`210:129.17,211:144.26,212:184.36,213:189.39,214:168.34,215:104.16,216:135.31,217:149.23`.

**vLLM/GPU**: vLLM is 14/14 OK with `0` waiting, mean KV `0.173`, and
preemptions stable at `96`. A 05:55 `gpustat` snapshot showed only `70%`
instantaneous aggregate GPU util despite healthy trainer MFU; treat single
gpustat frames as noisy.

**Resume**: Latest real checkpoint remains step 200.

## 2026-05-10 06:27 UTC

**Step**: trainer 228 / 1000; orchestrator 230 / 1000.
**Health**: Healthy; eval-250 has not started.

**Trainer**: Steps 215-228 average `164.16s`, `8639 tok/s`, and `30.68%` MFU.
Steps 220-228 average `161.12s`, `8647 tok/s`, and `30.70%` MFU. Peak memory
remains `92.9 GiB`.

**Orchestrator**: Steps 215-230 average `166.44s`, reward `0.4118`, seq length
`5534.1`. Steps 220-230 average `165.66s`, reward `0.4130`, seq length
`5449.9`. Recent async/off-policy is mostly `1/1`, with steps 226 and 230 at
off-policy `2`.

**Backpressure**: Checkpoint waits remain the visible drag: post-215 mean
`138.49s`, post-220 mean `131.60s`, tail
`215:104.16,216:135.31,217:149.23,218:177.35,219:195.33,220:142.19,221:91.09,222:92.18,223:108.13,224:127.13,225:207.23,226:159.31,227:138.24,228:122.27,229:128.23`.

**vLLM/GPU**: vLLM is 14/14 OK with `156` running, `0` waiting, mean KV
`0.167`, and preemptions stable at `96`. `gpustat` showed all 16 GPUs at
`100%`, using `1449145/1565936 MiB` total memory.

**Resume**: Latest real checkpoint remains step 200.

## 2026-05-10 06:59 UTC

**Step**: trainer 239 / 1000; orchestrator 241 / 1000.
**Health**: Healthy; eval-250 has not started.

**Trainer**: Steps 230-239 average `160.57s`, `8694 tok/s`, and `30.87%` MFU.
Peak memory remains `92.9 GiB`.

**Orchestrator**: Steps 230-241 average `161.02s`, reward `0.4165`, seq length
`5471.7`; recent off-policy includes steps 230, 234, and 241 at `2`, with
step 236 at `0`.

**Backpressure**: Checkpoint waits remain high but steady: post-230 mean
`139.86s`, tail
`230:168.27,231:157.20,232:92.15,233:129.27,234:128.28,235:165.28,236:151.23,237:117.13,238:134.22,239:146.23,240:149.24`.

**vLLM/GPU**: vLLM is 14/14 OK with `44` running, `0` waiting, mean KV `0.123`,
and preemptions stable at `96`. `gpustat` showed `63%` instantaneous aggregate
GPU util despite stable trainer MFU; keep treating single gpustat frames as
noisy.

**Resume**: Latest real checkpoint remains step 200.

## 2026-05-10 07:33 UTC

**Step**: trainer 250 / 1000; orchestrator 251 / 1000; eval-250 complete.
**Health**: Healthy; first clearly positive eval row, with boundary bubble
recurring.

**Eval**: Eval-250 ran from `07:24:46` to `07:30:55` in `369.37s` with
`Avg@8=0.3837`, `Pass@1=0.3837`, `Pass@2=0.5004`, `Pass@4=0.6111`,
`Pass@8=0.7000`, no-response `0.0%`, completion length
`5599.56 ± 5102.68`, truncated `13.2%`.

**Comparison**: Versus eval-200, Pass@1/2/4/8 moved
`+2.24/+3.18/+5.04/+8.00 pp`. Versus eval-50, Pass@1/2/4/8 moved
`+1.99/+1.86/+2.47/+3.00 pp`. Pass@2/4/8 and truncation are best-so-far;
Pass@1 is still slightly below eval-150 (`0.3837` vs `0.3850`).

**Boundary**: Orchestrator saved checkpoint 250 at `07:19:33`, completed step
250 at `07:22:00`, waited `163.24s` for checkpoint 250, then started eval.
Step 251 completed immediately after eval with a `538.58s` wall-clock time,
reward `0.4023`, seq length `6684.9`, async/off-policy `0/2`.

**Trainer**: Step 250 completed at `07:28:02` with `178.35s`, `8501 tok/s`,
`30.2%` MFU, peak memory `92.9 GiB`.

**vLLM**: 14/14 OK with `250` running, `0` waiting, mean KV `0.403`,
preemptions stable at `96`.

**Resume**: `checkpoints/step_250/trainer` now contains `.metadata`,
`__0_0.distcp`, and `__1_0.distcp`; latest real checkpoint is step 250.

## 2026-05-10 07:51 UTC

**Step**: trainer 257 / 1000; orchestrator 259 / 1000.
**Health**: Healthy, but post-eval-250 cooldown is active.

**Trainer**: Steps 251-257 average `167.05s`, `7590 tok/s`, and `26.94%` MFU.
Including step 250, steps 250-257 average `168.46s`, `7704 tok/s`, and
`27.35%` MFU. Peak memory remains `92.9 GiB`.

**Orchestrator**: Steps 251-259 average `194.09s`, reward `0.4249`, seq length
`5587.7`. Step 251 was the boundary bubble (`538.58s`). Off-policy levels rose
to `3` at step 257 and `4` at step 258 before returning to `1` at step 259.

**Backpressure**: Post-250 checkpoint waits average `130.22s`, tail
`250:163.24,252:64.10,253:39.04,254:78.14,255:164.36,256:185.35,257:179.24,258:168.25`.

**vLLM**: 14/14 OK with `255` running, `0` waiting, mean KV `0.234`,
preemptions stable at `96`.

## 2026-05-10 08:03 UTC

**Step**: trainer 261 / 1000; orchestrator 263 / 1000.
**Health**: Healthy; post-eval-250 MFU has recovered.

**Trainer**: Steps 251-259 stayed in the cooldown band (`~26.7-27.4%` MFU).
Steps 260-261 recovered to `162.02s`, `8585 tok/s`, and `30.45%` MFU. Including
the cooldown, steps 251-261 average `166.12s`, `7790 tok/s`, and `27.65%`
MFU. Peak memory remains `92.9 GiB`.

**Orchestrator**: Steps 260-263 average `161.43s`, reward `0.4075`, seq length
`5367.6`; off-policy returned to `1` for steps 260-262, then `2` at step 263.

**vLLM**: 14/14 OK with `130` running, `0` waiting, mean KV `0.280`,
preemptions stable at `96`.

**Interpretation**: Eval-250 repeated the same boundary pattern: one very large
post-eval orchestrator step, about 9 low-MFU trainer steps, then recovery.

## 2026-05-10 08:36 UTC

**Step**: trainer 273 / 1000; orchestrator 275 / 1000.
**Health**: Healthy; stable recovered band toward eval-300.

**Trainer**: Steps 260-273 average `162.36s`, `8585 tok/s`, and `30.48%` MFU.
Steps 270-273 average `158.99s`, `8617 tok/s`, and `30.60%` MFU. Peak memory
remains `92.9 GiB`.

**Orchestrator**: Steps 260-275 average `166.73s`, reward `0.4181`, seq length
`5479.7`; recent off-policy is mostly `1`, with step 272 at `2`.

**Backpressure**: Checkpoint waits post-260 average `147.40s`; post-270 average
`140.49s`.

**vLLM**: 14/14 OK with `252` running, `0` waiting, mean KV `0.337`,
preemptions stable at `96`.

**Resume**: Latest real checkpoint remains step 250.

## 2026-05-11 11:50 UTC Current State

The active 1000-step sequential configs were updated on the eval side before
resuming:

- `tmp/rl_olmo3_dpo_default_14i2t_bs256_eval50_1000step_20260509_1825.toml`
- `tmp/rl_olmo3_dpo_maxrl_14i2t_bs256_eval50_1000step_20260509_1825.toml`

They now keep `omni-math2-baseline100` as 100×8 every 50 steps and add
`omni-math2-full600-p1` as 600×1 every 250 steps. Eval `math_verify_timeout_seconds`
is 10s; train timeout remains 5s. Keep `cancel_inflight_rollouts_on_eval = false`.

Why: symbolic `math_verify` timeouts happen before the judge fallback, so the
fallback does not make them free. They still cost scorer time and can still
land as zero if parsing or the judge path fails/returns negative.

New tools:

- `scripts/evals/analyze_online_eval_rollouts.py` for pass@k summaries and
  paired bootstrap CIs over saved `eval_rollouts.jsonl`.
- `scripts/evals/make_omni_math2_perfectible_subset.py` for building a
  Hendrycks-style sensitivity subset from baseline rollouts.

Validation already run:

- Active tmp Default dry-run: ok, both eval envs resolved.
- Active tmp MaxRL dry-run: ok, both eval envs resolved.
- Repo-local `*_pipelinerl_speed.toml` Default/MaxRL dry-runs: ok.
- Analyzer succeeded on existing Default artifacts and confirmed ckpt-450 vs
  ckpt-50 is effectively flat within prompt-level uncertainty.

Resume point remains
`outputs/omni_math2_rlvr_canary/20260509_1825/dpo_default_14i2t_bs256_eval50_1000step/run_default/checkpoints/step_450/trainer`.
Use `tmp/resume_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh` inside
the current allocation. The default `skip_eval_on_resume = true` avoids
rerunning eval-450 immediately; the next eval boundary at ckpt-500 should run
both baseline100 and full600-p1.

## 2026-05-11 11:59 UTC Live Resume Status

Default has been relaunched in attached tmux window `joanv_cc_4node:olmo3-resume`
on Slurm job `4542540`.

Current verified state:

- Resume loaded `run_default/checkpoints/step_450/trainer`.
- Both eval envs initialized: `omni-math2-baseline100` and
  `omni-math2-full600-p1`.
- Eval-450 was skipped on resume, as intended.
- NCCL weight update completed; inference engines resumed.
- Orchestrator entered step 450 and train rollout generation is live. Last
  observed progress was `40/256` rollouts.
- No `*FAILED` marker was present.

Important caveat: the tmux pane printed an early
`STOP_INFERENCE observed on nid011162` line, but inference logs later show
successful `/init_broadcaster`, `/pause`, `/update_weights`, `/resume`, and
live `/v1/chat/completions`. Treat that line as non-fatal launcher noise unless
serving traffic stops or a failure marker appears.

Next check:

- Wait for the first new trainer `SUCCESS Step ...` after the `11:56` training
  loop restart.
- Then update this handoff, `TRIALS.md`, and the run `STATUS.md` with the
  actual post-resume step timing.

## 2026-05-11 12:00 UTC First Post-Resume Step

The first post-resume step completed:

- Orchestrator step 450: `11:59:43`, `215.97s`, reward `0.3915`, seq length
  `5156.5`, async/off-policy `0/0`.
- Trainer step 450: `12:00:23`, `258.61s`, peak memory `93.4 GiB`.

Do not use trainer step 450 for speed/MFU claims: it reports `0 tok/s` and
`0.0% MFU`, which is resume/warmup accounting. Use step 451 onward.

## 2026-05-11 13:13 UTC W&B Reattach + Offline Recovery

The online Default resume was stopped. It had reached trainer step 468 and
orchestrator step 470, but no stable checkpoint beyond step 450 existed. Treat
step 450 as the only safe resume/eval checkpoint from the stopped online run.

Future Default restarts must attach to the original W&B run id:

- run name: `olmo3-dpo-default-14i2t-bs256-eval50-1000step-20260509-1825`
- original W&B id: `197d96382b0c40d59272ac0fbc94a9e3`
- fragmented resume id to ignore: `66363fe840af4a9cb308118fddd02ea3`

This is implemented in:

- `tmp/resume_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh`
- `tmp/run_olmo3_bs256_eval50_1000_sequential_20260509_1825.sh`

Both scripts export `WANDB_RESUME=allow`, `WANDB_SHARED_RUN_ID`, and
`WANDB_RUN_ID` for the Default arm. Do not rely on the display name alone;
that fragments W&B logs. MaxRL remains unset unless `MAXRL_WANDB_RUN_ID` is
provided.

Checkpoint archive:

- `outputs/omni_math2_rlvr_canary/20260509_1825/checkpoint_archive_50/default/weights/step_{300,350,400,450}`
- `outputs/omni_math2_rlvr_canary/20260509_1825/checkpoint_archive_50/default/checkpoints/step_{300,350,400,450}`

Offline eval:

- Script: `scripts/evals/offline_omni_math2_ckpt_eval.py`
- Launcher: `tmp/run_olmo3_offline_eval_600x8_logged_20260511.sh`
- Active tmux window: `joanv_cc_4node:offline-eval`
- Current log:
  `outputs/omni_math2_rlvr_canary/20260509_1825/offline_eval_600x8/logs/launcher_20260511T130924Z.log`

The offline evaluator shards across the four per-node vLLM backends because
`vllm-router` is not installed here. It is currently targeting Default steps
300, 350, 400, and 450 with 600 examples × 8 rollouts each. The first threaded
shard attempt failed on `signal only works in main thread`; the current code
uses spawned subprocesses per backend and was live/advancing after relaunch.

Next checks:

- `tmux capture-pane -pt joanv_cc_4node:offline-eval -S -120`
- `tail -n 160 outputs/omni_math2_rlvr_canary/20260509_1825/offline_eval_600x8/logs/launcher_20260511T130924Z.log`
- When a checkpoint finishes, inspect
  `outputs/omni_math2_rlvr_canary/20260509_1825/offline_eval_600x8/default/step_*/summary.json`.

Post-completion logging fix: do not perturb the active offline eval, but update
`scripts/evals/offline_omni_math2_ckpt_eval.py` afterward so sharded evals no
longer interleave tqdm bars. Desired behavior is per-shard logs/artifacts and
one parent aggregate progress line/counter over the full 600×8 = 4,800 rollouts
per checkpoint.

## 2026-05-11 14:46 UTC Offline Eval Recovery Runbook

Default step 300 is recovered and summarized:

- Parent output: `outputs/omni_math2_rlvr_canary/20260509_1825/offline_eval_600x8/default/step_000300`
- Rollouts: `4800`
- Mean sample accuracy: `0.390625`
- Single-shot / prefix p@1: `0.3883333333333333`
- Pass@8: `0.6433333333333333`
- Error rate: `0.0`

Use the new direct recovery entrypoint:

```bash
tmp/recover_olmo3_offline_eval_600x8_20260511.sh summarize
OFFLINE_EVAL_MIN_STEP=350 tmp/recover_olmo3_offline_eval_600x8_20260511.sh continue-fresh
```

The detailed notes are in `tmp/olmo3_offline_eval_recovery_20260511.md`.
`summarize` is CPU/local only and merges completed shard artifacts before
recomputing parent summaries. `continue-fresh` unsets inherited endpoint
variables and defaults to ports `9200/9300`, so orphaned 9100 backends do not
silently contaminate recovery. Use `continue-existing` only after explicit
health and admin endpoint checks.

## 2026-05-11 15:04 UTC Metric Clarification

Offline `summary["pass"][k]["pass_at_k"]` is now explicitly routed through the
same Chen et al. / HumanEval unbiased estimator used by online eval utilities:
`1 - C(n-c,k) / C(n,k)`. Ordered first-k diagnostics are still present as
`prefix_pass_at_k`.

Default step 300 shared-estimator values: p@1 `0.390625`, p@2
`0.4870833333333338`, p@3 `0.5386309523809528`, p@4
`0.5728571428571426`, p@5 `0.5976785714285717`, p@6
`0.6166071428571432`, p@8 `0.6433333333333333`.

Truncation is still a real quality problem at step 300: `646 / 4800 = 13.46%`
rollouts hit the `15360` completion-token cap, truncated rollouts were correct
at only `2.0%`, and non-truncated rollouts were correct at `44.8%`.

The `offline-eval5` continuation was stopped at `2026-05-11 15:10 UTC` because
`nid011162:9300` was down, its Slurm step had disappeared, and the live shard was
emitting repeated API connection-error rollouts. Do not treat partial step-350
artifacts from that attempt as valid eval signal.

Follow-up relaunch: cancelled orphan Slurm step `4542540.18`, patched
`scripts/evals/offline_omni_math2_ckpt_eval.py` to wait for every generation and
admin endpoint before checkpoint weight update, then relaunched missing steps
`350,400,450` in `joanv_cc_4node:offline-eval6`:

```bash
OFFLINE_EVAL_STEPS=350,400,450 OFFLINE_EVAL_PORT=9600 OFFLINE_EVAL_BACKEND_PORT=9700 OFFLINE_EVAL_MAX_RETRIES=3 tmp/recover_olmo3_offline_eval_600x8_20260511.sh continue-fresh
```

Current log:
`outputs/omni_math2_rlvr_canary/20260509_1825/offline_eval_600x8/logs/launcher_20260511T152642Z.log`.
At `2026-05-11 15:29 UTC`, all four endpoints were healthy and step-350
generation/scoring was underway after a clean pause/resume.

## 2026-05-11 22:45 UTC Stop + Difficulty-Filter Pivot

The online Default run was stopped on purpose. Do not continue it or launch
MaxRL unchanged.

Why it was stopped:

- Latest W&B watcher snapshot scanned the original Default run
  `197d96382b0c40d59272ac0fbc94a9e3` through trainer/orchestrator step
  `528`.
- The normal pool had collapsed: `pool_easy=0.4018`, `pool_hard=0.5645`,
  `pool_normal=0.0337` at step 528; the current 50-step window mean for normal
  pool was `0.0375`.
- That means train reward was heavily selection-conditioned on the tiny
  remaining normal pool. Reward/reward-max traces are not comparable to the
  original dataset distribution after this point.
- Eval at step 500 was not a clean upward story: baseline100 p@8 `0.6300`,
  full600-p8 p@8 `0.6450`, with held-out strata showing gains concentrated in
  early-hard and regressions elsewhere.

Cleanup status:

- Allocation `4542540` was preserved; do not `scancel 4542540` unless the user
  explicitly wants to release the allocation.
- Run/eval/monitor tmux panes were closed. Current panes are only `node` and
  `work`.
- `squeue --steps -j 4542540` may still show base/batch and one-node
  interactive `bash` steps on `nid011162`; those are control-path steps, not
  evidence the RL job is running.
- Local `pgrep` only sees Codex sandbox processes for the old RL search terms.
  `nvidia-smi` is not reliable from the current sandbox, and `srun --overlap`
  hit the known interconnect setup error.

What the code actually does:

- `src/prime_rl/orchestrator/buffer.py` moves an example from normal to
  easy/hard after a single observed group average crosses
  `easy_threshold`/`hard_threshold`.
- Sampling draws only from normal examples.
- With `online_difficulty_filtering=true`, rollouts from non-normal groups are
  not added to the training buffer.
- `easy_fraction` and `hard_fraction` only reintroduce saved easy/hard examples
  when loading a checkpoint. They are not per-step sampling fractions in the
  current code.
- Therefore the current OLMo3 recipe
  `easy_threshold=0.875`, `hard_threshold=0.0625`,
  `online_difficulty_filtering=true`, `easy_fraction=hard_fraction=0.0` creates
  one-shot absorbing quarantine during a live run.

Upstream/fork findings:

- Upstream `PrimeIntellect-ai/prime-rl` main now has the same modern
  hard/easy mechanism plus resume-time reintroduction fractions.
- The Hendrycks sanity example does **not** use this online hard/easy filter;
  it uses a prefiltered dataset of problems the base model solves 20-80% of the
  time across 40 rollouts.
- The repeated public math recipe is stricter than ours:
  `easy_threshold=1.0`, `hard_threshold=0.0`, usually with no explicit
  `online_difficulty_filtering`, or with filtering only for all-0/all-1 groups.
- Branches/forks with different behavior:
  - `mika/feat/buffer-ckpt+offline-filter`: older `difficulty-pool` and
    `online-difficulty` buffers where `easy_fraction`/`hard_fraction` meant
    batch sampling proportions, plus oversampling to compensate for filtering.
  - `replay` / `replay-buffer-feature`: experimental replay buffers, but with
    stale/incomplete knobs in at least one branch.
  - `sami/refactor/difficulty_filtering`,
    `sami/intellect_math_1b_2k_config`,
    `justus/difficulty-filtering-config-qwen-32b`: dataset-level solve-rate
    filters such as `min_solve_rate=0.4`, `max_solve_rate=0.9` or
    `max_solve_rate=0.7`.
- Broader primary-source RLVR recipes point to DAPO-style dynamic sampling:
  keep generating replacement prompts until enough mixed-outcome groups are
  accepted, with an explicit max generated-batches cap. OpenRLHF and verl both
  use the all-0/all-1 dynamic filtering shape; NeMo exposes a max generated
  batch cap and trajectory-age accounting for async/replay settings.

Current recommendation:

1. Do **not** disable filtering as the main fix. The issue is not filtering per
   se; it is absorbing one-shot filtering plus no batch refill.
2. TOML-only stopgap: switch OmniMath OLMo3 configs to `easy_threshold=1.0`
   and remove `hard_threshold`, keeping `online_difficulty_filtering=true`.
   This still removes solved prompts, but it does **not** permanently quarantine
   step-0 all-zero prompts that may become solvable later.
3. Implement a DAPO-style refill path before any 1000-step relaunch:
   filtered groups should trigger replacement sampling until `batch_size`
   accepted rollouts are available or a finite cap is hit. Log
   `candidate_groups`, `accepted_groups`, `filtered_easy_groups`,
   `filtered_hard_groups`, `refill_rounds`, and
   `prompts_consumed_per_accepted_group`.
4. Build/try an OmniMath2 perfectible subset using
   `scripts/evals/make_omni_math2_perfectible_subset.py`, analogous to
   Hendrycks sanity: base-policy solve rate in roughly `[0.2, 0.8]` over many
   rollouts. This is the cleanest way to test whether RL can learn when support
   is known to contain gradient signal.
5. If resuming an existing checkpoint only for recovery, set nonzero
   `--orchestrator.buffer.hard_fraction` and
   `--orchestrator.buffer.easy_fraction` on resume to reintroduce quarantined
   examples. This is a recovery hack, not a full live-run fix, because the same
   examples can be absorbed again after one new group.
6. Delay prompt-level replay until it has age limits and accounting tied to
   `max_off_policy_steps`. Otherwise it is easy to build stale-data bias under
   a different name.

2026-05-11 22:49 UTC config update:

- All `configs/omni_math2/rl_olmo3*.toml` now use solved-only online
  difficulty filtering: `easy_threshold=1.0`, no `hard_threshold`,
  `online_difficulty_filtering=true`.
- The two PipelineRL-speed recipes now use rollout batching again:
  `batch_size=256`, `max_inflight_rollouts=768`, `rollouts_per_example=8`,
  `max_off_policy_steps=8`. This matches the user's "training batch size"
  correction while keeping three batches of inflight slack.
- Representative dry-runs passed after the change:
  - `/tmp/prime-rl-dryrun-nohard-token-default`
  - `/tmp/prime-rl-dryrun-bs256-nohard-pipe-default`
  - `/tmp/prime-rl-dryrun-bs256-nohard-pipe-maxrl`

2026-05-11 23:02 UTC independent verification:

- A separate agent verified the filtering semantics against
  `src/prime_rl/orchestrator/buffer.py`,
  `src/prime_rl/configs/orchestrator.py`,
  `tests/unit/orchestrator/test_buffer.py`, and a focused repro under
  `tmp/buffer_verify/`.
- Verdict: the claim is correct. `online_difficulty_filtering=true` does not
  independently filter reward-0 groups; it only skips groups classified
  non-normal by the thresholds.
- Truth table for current stopgap (`easy_threshold=1.0`, no
  `hard_threshold`, online filtering on):
  - avg reward `0.0` -> normal, rollouts enter train buffer, still sampleable.
  - avg reward `0.5` -> normal, rollouts enter train buffer, still sampleable.
  - avg reward `1.0` -> easy, rollouts skipped, prompt removed from normal.
- Truth table for the rejected strict-hard variant (`easy_threshold=1.0`,
  `hard_threshold=0.0`, online filtering on):
  - avg reward `0.0` -> hard, rollouts skipped, prompt removed from normal.
  - avg reward `0.5` -> normal, rollouts enter train buffer, still sampleable.
  - avg reward `1.0` -> easy, rollouts skipped, prompt removed from normal.
- Important gotcha: pool membership is sticky, but `update_pools()` recomputes
  a pool name for every arriving group. If duplicate groups for the same prompt
  were already in flight, a late mixed-reward group for an already-evicted
  prompt can still enter the training buffer. Treat this as async leakage, not
  a real reintroduction mechanism.
- Important gotcha: `rollout_buffer` is an unbounded Python list and
  `sample_rollouts()` takes from the tail. Freshness preference exists, but
  producer-over-consumer bursts can accumulate stale rollouts and memory.
- Important gotcha: `online_difficulty_filtering=true` with both thresholds
  unset is a silent no-op. Our current configs set `easy_threshold`, so this is
  not currently a problem.

Suggested next concrete work:

- Patch the temp Default/MaxRL continuation configs or checked-in OmniMath
  configs to run the solved-only filter canary (`easy_threshold=1.0`, no
  `hard_threshold`).
- Add an optional drop-without-evict refill cap to the buffer/orchestrator
  path. Do not implement hard-pool quarantine as the refill mechanism.
- Add the logging above to W&B and local rollout metadata, plus
  `example_was_already_evicted`, `previous_pool`, `current_group_pool`,
  `late_normal_rollouts_from_evicted_examples`, rollout-buffer length, and
  rollout age/staleness.
- Consider adding a validator warning for `online_difficulty_filtering=true`
  with no thresholds set.
- Dry-run both Default and MaxRL after the config/code patch.
- Launch only a short attached tmux canary first; do not spend another long
  allocation until `pool/normal`, accepted batch size, and eval direction look
  sane.

2026-05-11 23:18 UTC corrected 8-node topology update:

- Corrected unit: `orchestrator.batch_size` is **rollouts/samples**, not
  problems. With `rollouts_per_example=8`, `batch_size=256` means 32 problems
  per step, not 256. A 256-problem step would be `batch_size=2048`.
- The earlier 15M-token BOTEC applies to `batch_size=2048`, not to the current
  `batch_size=256`. Current `batch_size=256` at ~7.5k tokens/rollout is roughly
  1.9M tokens/step.
- This correction moves the 8-node recommendation:
  - Preferred target if the launcher works: `24 inference / 8 trainer` using
    `[deployment] type = "multi_node"`, `num_train_nodes=2`,
    `num_infer_nodes=1`, `num_infer_replicas=6`,
    `nodes_per_fsdp_group=2`.
  - Fallback if multi-node smoke fails: current `gpu_layout` can do
    `28 inference / 4 trainer` because it supports exactly one trainer node.
- Why: corrected BOTEC puts 24i/8t near balanced at current `batch_size=256`,
  while 28i/4t remains trainer-bound. 24i/8t also gives headroom for
  `batch_size=512` or `1024` later without re-laying out.
- Local code check:
  - `gpu_layout` validator still rejects more than one trainer node.
  - `multi_node` supports `num_infer_replicas`; the template passes
    `total_infer_nodes = num_infer_nodes * num_infer_replicas` to Slurm and
    starts one router per replica.
  - For standard TP=1 inference, config auto-sets per-node DP/api servers to
    `gpus_per_node / tp`, so `num_infer_nodes=1`, `num_infer_replicas=6`,
    `tp=1` should produce 6 one-node replicas with 4 vLLM workers each.
  - This is still a semantics claim to smoke-test, not something to assume for
    a 100-step burn.
- Multi-node is a Slurm/sbatch path. For a new allocation, create temporary
  configs and dry-run; do not blindly rewrite the checked-in gpu_layout TOMLs.

Updated launch plan for 8 nodes:

1. Create temp Default and MaxRL multi-node configs from the PipelineRL-speed
   TOMLs.
2. Use:
   ```toml
   [deployment]
   type = "multi_node"
   num_train_nodes = 2
   num_infer_nodes = 1
   num_infer_replicas = 6
   nodes_per_fsdp_group = 2
   ```
3. Keep solved-only filtering: `easy_threshold=1.0`, no `hard_threshold`.
4. Smoke first at `batch_size=256`, `max_inflight_rollouts=768` for 3-5 steps
   to verify worker count, routing, weight broadcast, trainer throughput, and
   eval/log plumbing.
5. If clean, run a 50-100 step canary at either:
   - conservative: `batch_size=256`, `max_inflight_rollouts=768`; or
   - utilization compromise: `batch_size=512`, `max_inflight_rollouts=1536`.
6. Do not try `batch_size=2048` until 512/1024 are measured.

2026-05-12 15:42 UTC eval continuation:

- Live compiled 28i/4t run:
  `outputs/omni_math2_rlvr_canary/default_8node_28i4t_compile_fsasync4_20260512_1430`.
- It has no online eval stanza; quality should be read through offline eval.
- User requested offline eval over all real checkpoints. For this run, that
  means scheduled checkpoints `25,50,75,100` (`ckpt.interval=25`), not every
  per-step filesystem broadcast snapshot.
- Added and syntax-checked:
  `tmp/run_olmo3_offline_eval_28i4t_all_ckpts_20260512.sh`.
- The script waits for
  `run_default/broadcasts/step_100/STABLE`, then runs `600x8` Omni-MATH-2
  offline eval over `25,50,75,100` using `--weights-root run_default/broadcasts`.
  This is required because `scripts/evals/offline_omni_math2_ckpt_eval.py`
  discovers stable HF-style weight directories, while `run_default/checkpoints`
  contains trainer state.
- Queued visibly in tmux:
  `joanv_cc_8node:7 eval-all`.
- At queue time it was just waiting and not consuming GPUs. Results/logs will
  go under
  `outputs/omni_math2_rlvr_canary/default_8node_28i4t_compile_fsasync4_20260512_1430/offline_eval_600x8_all_ckpts`.

2026-05-12 15:58 UTC DAPO-style refill implementation:

- Added opt-in config:
  ```toml
  [orchestrator.train_batch_refill]
  enabled = true
  max_refill_rounds = 4
  ```
- Default remains disabled, so the live 28i/4t run is unaffected.
- Implementation files:
  - `src/prime_rl/configs/orchestrator.py`
  - `src/prime_rl/orchestrator/orchestrator.py`
  - `src/prime_rl/orchestrator/refill.py`
  - `src/prime_rl/orchestrator/train_batch_refill.py`
  - `tests/unit/orchestrator/test_buffer.py`
  - `tests/unit/test_configs.py`
- Semantics: after advantages and filters are applied, whole groups with no
  trainable units are dropped from the current train batch and replacement
  candidate batches are drawn up to `max_refill_rounds`. This does **not**
  hard-evict all-zero prompts from future sampling. Easy/all-one groups can
  still be evicted by the existing solved-only buffer filter.
- Scope: rollout batching only. Validator rejects `token_batch_size` with
  refill enabled. Multi-agent envs also error if enabled.
- Metrics emitted under `train_batch_refill/*`: candidate/accepted groups,
  filtered easy/hard/zero-advantage groups, candidate/accepted rollouts,
  refill rounds, prompts consumed per accepted group, and reward means
  conditioned/unconditioned on filtering:
  `train_batch_refill/reward_conditioned_on_filtering/mean`,
  `train_batch_refill/reward_unconditioned_on_filtering/mean`,
  `train_batch_refill/reward_filtered_out/mean`.
- Prepared refill config:
  `configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4_refill.toml`.
  It preserves the non-refill 28i/4t shape (`batch_size=256`,
  `max_inflight_rollouts=768`, filesystem broadcast, compile, solved-only
  filter with no `hard_threshold`) and enables:
  ```toml
  [orchestrator.train_batch_refill]
  enabled = true
  max_refill_rounds = 4
  ```
- CPU checks passed:
  ```bash
  uv run pytest tests/unit/orchestrator/test_buffer.py \
    tests/unit/orchestrator/test_filters.py \
    tests/unit/orchestrator/test_advantage.py \
    tests/unit/test_configs.py::test_train_batch_refill_requires_rollout_batching
  # 48 passed, 2 warnings

  uv run ruff check src/prime_rl/orchestrator/refill.py \
    src/prime_rl/orchestrator/train_batch_refill.py \
    src/prime_rl/orchestrator/orchestrator.py \
    src/prime_rl/configs/orchestrator.py \
    tests/unit/orchestrator/test_buffer.py \
    tests/unit/test_configs.py
  # All checks passed
  ```
- Next live canary after current eval/run finishes: enable this on a short
  28i/4t `batch_size=256` canary and compare accepted reward/eval against raw
  candidate reward. Do not treat accepted-batch reward alone as quality.

2026-05-12 16:08 UTC interruption/update:

- The live 28i/4t run did not finish to step 100.
- Slurm step `4555723.236` ended as `CANCELLED by 1483805060`, exit `0:15`,
  at `2026-05-12T15:51:59`, while generating orchestrator step 80.
- Allocation `4555723` stayed running.
- Last completed trainer step in `trainer/node_0.log`: step 78.
- Last completed orchestrator step in `orchestrator.log`: step 79.
- Quick log grep found no Python traceback/runtime error; pane showed
  `srun: forcing job termination`, then all 8 tasks were terminated/killed.
- Stable trainer checkpoints exist for `25,50,75`; filesystem broadcast
  snapshots additionally exist for `76,77,78,79`.
- The old eval waiter was waiting for nonexistent `step_100`. It was retargeted
  and relaunched visibly in `joanv_cc_8node:6 eval-all` with:
  ```bash
  OFFLINE_EVAL_WAIT_STEP=75 OFFLINE_EVAL_STEPS=25,50,75 \
    bash tmp/run_olmo3_offline_eval_28i4t_all_ckpts_20260512.sh
  ```
- Treat the termination as externally signalled unless later evidence shows
  otherwise. Do not repeat the mistaken claim that this was a model/vLLM crash
  without a traceback.

2026-05-12 16:36 UTC eval route correction:

- The first direct-backend offline eval on all 8 nodes is invalid for quality:
  `nid010685` was both the eval driver node and a vLLM shard. Its vLLM DP
  coordinator died around `16:29 UTC` with `RuntimeError: cancelled`; shard 00
  then repeatedly logged `APIConnectionError('Connection error.')` while the
  aggregate tqdm kept moving. Do not use
  `offline_eval_600x8_all_ckpts` for quality conclusions.
- Added durable support for the clean route:
  - `src/prime_rl/baselines/provision.py` now honors
    `PRIME_RL_MULTINODE_HOSTS` for srun-multinode eval launches.
  - `scripts/evals/offline_omni_math2_ckpt_eval.py` derives admin/generation
    URLs from that same host override.
  - `src/prime_rl/templates/multi_node_rl.sbatch.j2` now honors
    `PRIME_RL_DISABLE_VLLM_ROUTER=1` when direct backends are valid.
  - New reusable wrapper:
    `scripts/evals/run_omni_math2_offline_eval_28i4t.sh`.
- Clean non-refill eval is now running visibly in `joanv_cc_8node:6 eval-all`
  using 7 vLLM nodes and excluding the driver node:
  `nid010752,nid010753,nid010756,nid010757,nid010758,nid010765,nid010768`.
  Output path:
  `outputs/omni_math2_rlvr_canary/default_8node_28i4t_compile_fsasync4_20260512_1430/offline_eval_600x8_7node_clean`.
- Observed clean-launch GPU state: `nid010685` driver-only with no vLLM GPU
  residency; all 28 vLLM GPUs at 100% util / about 88.5 GiB used after
  generation started.

2026-05-12 22:22 UTC post-run offline eval automation:

- Added durable comparator:
  `scripts/evals/compare_omni_math2_offline_evals.py`.
  It writes
  `outputs/omni_math2_rlvr_canary/offline_eval_comparison_20260512.md`,
  pulling the user-provided raw OLMo Inst DPO reference plus existing
  `600x8` offline eval summaries for non-filtering 14i/2t and 28i/4t runs.
  It will pick up DAPO refill eval summaries as soon as they appear.
- Added reusable Slurm submitter:
  `scripts/evals/submit_omni_math2_postrun_offline_eval.sh`.
  It submits an 8-node dependent eval job, keeps the batch driver off the 7
  vLLM nodes via `PRIME_RL_MULTINODE_HOSTS`, disables `vllm-router`, evaluates
  stable checkpoints on the 25-step grid up to 100, then reruns the comparator.
- Live `1e-6` DAPO refill run:
  - training job: `4570549`
  - post-run eval job: `4574647`
  - dependency: `afterany:4570549`
  - eval output:
    `outputs/omni_math2_rlvr_canary/default_8node_28i4t_compile_fsasync4_refill_20260512_1745/offline_eval_600x8_7node_clean`
  - submission record:
    `.../offline_eval_600x8_7node_clean/POSTRUN_EVAL_SUBMISSION.md`
- LR `3e-6` arm:
  - current job: `4574276`, still pending on priority at setup time.
  - supervisor pane: `joanv_cc_8node:6 lr3e6-supervise`.
  - supervisor status:
    `outputs/omni_math2_rlvr_canary/lr3e6_supervisor_20260512/status.md`.
  - The supervisor now submits the matching post-run offline eval for whatever
    final LR Slurm job ID it is actually supervising, so retry job IDs do not
    strand an `afterok` dependency on the wrong job.
- Codex-side read-only monitor session `13539` polls the two training jobs,
  live post-run eval job, LR supervisor status, and comparison markdown every
  120 seconds.

2026-05-13 09:30 UTC morning update:

- Active goal was recreated after context transition: continue OmniMath2 OLMo3
  RLVR without GIGO, finish DAPO refill evals, compare `1e-6` vs `3e-6` and
  non-filter baselines, keep docs current.
- `1e-6` refill training job `4570549` timed out after 6h. Usable stable
  checkpoints: `25,50,75`; no `step_100`.
- `3e-6` refill training job `4574276` timed out after 8h at Slurm level, but
  logs and stable broadcasts reached `step_100`.
- First `1e-6` post-run eval job `4574647` failed due a vLLM worker port bind
  collision (`EADDRINUSE`), not a model quality failure.
- Retried `1e-6` eval: Slurm job `4582655`, running.
- Submitted missing `3e-6` eval: Slurm job `4582691`, running.
- Visible monitor pane: `tmux attach -t joanv_cc_8node`, window
  `4:eval-watch`.
- Monitor status file:
  `outputs/omni_math2_rlvr_canary/postrun_eval_monitor_20260513.md`
- Comparison report:
  `outputs/omni_math2_rlvr_canary/offline_eval_comparison_20260512.md`
- New reusable monitor script:
  `scripts/evals/monitor_omni_math2_postrun_offline_evals.sh`
- Stale item: the old `lr3e6_supervisor_20260512/status.md` stopped updating
  around `2026-05-13T04:59Z`; ignore it unless cross-checking historical state.

2026-05-13 09:46 UTC refill/MFU correction:

- Corrected an earlier bad inference: `1e-6` job `4570549` was **not**
  non-refill. It used the first DAPO-style refill implementation from commit
  `0470684cb`, with full `batch_size=256` candidate batches per refill round.
  Its old warning string
  `Attempt ... filtered out all 256 rollouts - retrying batch generation` is
  misleading because the refill branch could still accept partial groups before
  falling through to that warning path.
- `3e-6` job `4574276` used the later optimized candidate-batched refill from
  commit `c8a5b8307`, with `candidate_groups_per_round=32` and
  `max_candidate_groups=128`.
- Generated config evidence:
  - `1e-6`: `[train_batch_refill] enabled = true`, `max_refill_rounds = 4`,
    no candidate-group budget fields.
  - `3e-6`: same refill enabled plus `candidate_groups_per_round = 32` and
    `max_candidate_groups = 128`.
- Parsed trainer MFU:
  - non-refill `28i/4t bs256`, steps `25-74`: `49.6s` mean step time,
    `28.70%` mean trainer MFU, `16,156` tok/s.
  - `1e-6` refill v1, steps `25-74`: `230.0s`, `10.82%`, `6,119` tok/s.
  - `3e-6` refill v2, steps `25-74`: `191.6s`, `12.40%`, `6,990` tok/s.
- Parsed refill metrics:
  - `1e-6` refill v1, steps `25-74`: `65.28` candidate groups, `32.00`
    accepted groups, `25.40` filtered groups, `2.04` prompts per accepted
    group, unconditioned reward `0.248`, conditioned reward `0.408`.
  - `3e-6` refill v2, steps `25-74`: `55.26` candidate groups, `32.00`
    accepted groups, `21.80` filtered groups, `1.73` prompts per accepted
    group, unconditioned reward `0.258`, conditioned reward `0.418`.
- Interpretation: optimized refill reduced candidate waste relative to v1, but
  it still did **not** improve MFU versus the non-refill 28i/4t baseline.
  Current open question is quality/pass@k from the running offline evals.

2026-05-13 10:45 UTC launch-command cleanup:

- Added a canonical launch entrypoint:
  `uv run --no-sync python -m prime_rl.entrypoints.launch`.
- Subcommands:
  - `rlvr`: one command for PrimeRL RLVR launches/dry-runs.
  - `offline-eval`: one command for in-allocation or dependent-sbatch
    OmniMath2 checkpoint evals.
  - `data`: one command for baseline rollout generation and/or perfectible
    dataset filtering.
- Added `docs/launch.md` with the current recipes.
- Updated `skills/entrypoints/SKILL.md` so future agents stop recreating the
  call shape.
- `scripts/evals/run_omni_math2_offline_eval_28i4t.sh` and
  `scripts/evals/submit_omni_math2_postrun_offline_eval.sh` remain for
  compatibility, but now delegate to `prime_rl.entrypoints.launch`.
- Verified:
  ```bash
  uv run --no-sync ruff check src/prime_rl/entrypoints/launch.py \
    tests/unit/test_launch_entrypoint.py scripts/evals/offline_omni_math2_ckpt_eval.py \
    scripts/evals/compare_omni_math2_offline_evals.py tests/unit/baselines/test_provision.py
  uv run --no-sync pytest tests/unit/test_launch_entrypoint.py tests/unit/baselines/test_provision.py
  bash -n scripts/evals/run_omni_math2_offline_eval_28i4t.sh \
    scripts/evals/submit_omni_math2_postrun_offline_eval.sh
  uv run --no-sync python -m prime_rl.entrypoints.launch rlvr \
    --config configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4_refill.toml \
    --dry-run --output-dir /tmp/prime-launch-rlvr-real
  bash -n /tmp/prime-launch-rlvr-real/rl.sbatch
  ```
- Active routed eval relaunches:
  - `4583877`: `1e-6` DAPO refill offline eval, true 8-node routed route.
  - `4583883`: `3e-6` DAPO refill offline eval, true 8-node routed route.
  - Both routers reached 8 healthy hosts and expanded to 32 DP-aware workers.
  - Both are currently on `step_25` after successful backend
    pause/update/resume.

2026-05-13 10:50 UTC routed eval correction:

- The routed eval jobs `4583877` and `4583883` failed after reaching
  `step_25`; no summaries landed.
- Failure mode: `run_baseline()` re-entered `InferenceProvisioner` in external
  mode and checked router `/v1/models`; current `vllm-router` returns 500
  there, while `/health` works.
- Patch applied:
  - `BaselineConfig.launch.external_health_check` now supports
    `"models"` (default) and `"router_health"`.
  - External `InferenceProvisioner` can wait on router `/health`.
  - Offline checkpoint eval marks the generation endpoint as
    `router_health` when it is the provisioned router and admin URLs are the
    direct backends.
- Verified:
  ```bash
  uv run --no-sync ruff check src/prime_rl/baselines/config.py \
    src/prime_rl/baselines/provision.py scripts/evals/offline_omni_math2_ckpt_eval.py \
    tests/unit/baselines/test_config.py tests/unit/baselines/test_provision.py
  uv run --no-sync pytest tests/unit/baselines/test_config.py \
    tests/unit/baselines/test_provision.py tests/unit/test_launch_entrypoint.py
  ```
- Replacement eval jobs:
  - `4584396`: `1e-6` refill, pending on priority at submission.
  - `4584395`: `3e-6` refill, pending on priority at submission.

2026-05-13 11:15 UTC upstream check and routed eval state:

- Upstream `PrimeIntellect-ai/prime-rl` `main` is currently
  `e4330c2d2ca4fc4af46b1dfed2e6541489f52cbb`.
- There is no exact upstream cherry-pick for the local offline-eval
  router-readiness bug: upstream does not have our `src/prime_rl/baselines/*`
  provisioner path. The local `external_health_check="router_health"` patch is
  still the right narrow fix.
- Upstream does have directly relevant stale-node cleanup in `c3a24c3`
  (`feat(slurm): cleanup stale node-local state before launch`). The new
  `3e-6` replacement eval `4584395` hit that class of failure on `nid010069`:
  `RuntimeError: DP Coordinator process failed to report ZMQ addresses during startup`.
- Applied a narrow backport to `src/prime_rl/baselines/provision.py`: the
  offline-eval `srun_multinode` driver now runs a per-node cleanup step before
  starting vLLM/router.
- Verified:
  ```bash
  uv run --no-sync ruff check src/prime_rl/baselines/provision.py \
    tests/unit/baselines/test_provision.py
  uv run --no-sync pytest tests/unit/baselines/test_provision.py \
    tests/unit/test_launch_entrypoint.py
  bash -n /tmp/provision-driver-t2k0y0jc/inference/launch_multinode_driver.sh
  ```
- Current eval state at the time of the check:
  - `4584396` (`1e-6`) is running and has reached `step_25` eval after
    successful pause/update/resume.
  - `4584395` (`3e-6`) is still listed as running but has one failed backend
    task (`4584395.7`) and is likely doomed unless vLLM/router tolerates the
    missing node. If it fails, retry with the cleanup patch above.

2026-05-13 11:31 UTC eval restart:

- Cancelled doomed `3e-6` eval job `4584395`; it was stuck before `step_25`
  after backend task `4584395.7` failed on `nid010069`.
- Re-submitted the same routed `3e-6` eval sbatch after the cleanup patch.
  New job id: `4584655`.
- Live evals to monitor:
  - `4584396`: `1e-6` routed refill eval, running.
  - `4584655`: `3e-6` routed refill eval retry, pending/running depending on
    queue state.
- Check with:
  ```bash
  squeue -j 4584396,4584655 -o '%.18i %.9P %.40j %.10T %.10M %.9l %.6D %R'
  sacct -j 4584396,4584655 --format=JobID,JobName%40,State,ExitCode,Elapsed,NodeList%80 -P
  ```

2026-05-13 11:35 UTC eval split:

- Cancelled serial routed evals `4584396` and `4584655`. The `1e-6` job had
  only `228/4800` rollout rows for `step_25` after about 25 minutes; at the
  observed `~2.1k` generated tok/s across 32 eval GPUs, four checkpoints in
  one 6h allocation was not viable.
- Submitted one `600x8` routed job per checkpoint with `08:00:00` wall time:
  - `1e-6`: `4584726` step 25, `4584727` step 50, `4584733` step 75,
    `4584739` step 100.
  - `3e-6`: `4584740` step 25, `4584741` step 50, `4584743` step 75,
    `4584744` step 100.
- Output dirs are `offline_eval_600x8_8node_router_step{25,50,75,100}` under
  each arm's run root.
- `scripts/evals/compare_omni_math2_offline_evals.py` now globs
  `offline_eval_600x8_8node_router*`, so the step-split summaries are included.
- Monitor with:
  ```bash
  squeue -j 4584726,4584727,4584733,4584739,4584740,4584741,4584743,4584744 \
    -o '%.18i %.9P %.40j %.10T %.10M %.9l %.6D %R'
  ```

2026-05-13 11:42 UTC correction:

- `4584726` started and failed in `00:00:43` with exit `1:0` because the
  stale-cleanup backport killed its own remote cleanup `srun` task
  (`srun: error: nid010645: task 0: Killed`; endpoint wait saw code `137`).
- Cancelled the seven pending jobs from that broken batch:
  `4584727`, `4584733`, `4584739`, `4584740`, `4584741`, `4584743`,
  `4584744`.
- Patched `src/prime_rl/baselines/provision.py`: remote cleanup now uses
  `ps`/`awk` to collect stale PIDs while excluding the cleanup task's own
  process group, instead of `pkill -f` patterns that can match the cleanup
  command line.
- Verified:
  ```bash
  uv run --no-sync ruff check src/prime_rl/baselines/provision.py \
    tests/unit/baselines/test_provision.py
  uv run --no-sync pytest tests/unit/baselines/test_provision.py \
    tests/unit/test_launch_entrypoint.py
  ```
- Resubmitted split routed evals:
  - `1e-6`: `4585067` step 100, `4585068` step 25, `4585069` step 50,
    `4585070` step 75.
  - `3e-6`: `4585071` step 100, `4585072` step 25, `4585073` step 50,
    `4585074` step 75.
- At `11:40 UTC`, `4585067`, `4585068`, and `4585069` were running and had
  passed the previous failure point: remote cleanup completed, router launched,
  8 backend hosts were listed, and logs showed `nccl_net=AWS Libfabric`.
- Persistent monitor was restarted outside the sandbox:
  - pid file:
    `outputs/omni_math2_rlvr_canary/monitors/postrun_eval_monitor_20260513_stepsplit.pid`
  - status:
    `outputs/omni_math2_rlvr_canary/postrun_eval_monitor_20260513_stepsplit.md`
  - comparison:
    `outputs/omni_math2_rlvr_canary/offline_eval_comparison_20260512.md`

2026-05-13 11:50 UTC correction:

- `1e-6` step `100` is invalid: that run stopped at step `85`. Job `4585067`
  reached readiness and failed with `No matching stable weight checkpoints
  found`.
- `1e-6` step `25` job `4585068` also failed after readiness with the same
  discovery error, but local inspection shows
  `run_default/broadcasts/step_25/{STABLE,model-00001..00003-of-00003.safetensors}`.
  Local `_discover_weight_steps(..., steps={25})` returns the checkpoint, so
  this was treated as a transient compute-side visibility failure and retried.
- `1e-6` step `50` job `4585069` is the first valid split eval actively
  generating; it reached pause/update/resume and has partial rollouts under
  `offline_eval_600x8_8node_router_step50/refill_lr1e6_28i4t/step_000050/`.
- Valid stable `1e-6` broadcast steps are `25`, `50`, `75`, and final `85`
  for the comparison; there are also per-step final broadcasts `81-84`.
- Submitted:
  - `4585323`: `1e-6` step `25` retry.
  - `4585324`: `1e-6` final step `85`.
- Monitor PID was replaced with `279784`; it now tracks:
  `4585067`, `4585068`, `4585069`, `4585070`, `4585071`, `4585072`,
  `4585073`, `4585074`, `4585323`, `4585324`.

2026-05-13 12:07 UTC status update:

- Added a generated-sbatch checkpoint preflight in
  `src/prime_rl/entrypoints/launch.py`: explicit `--steps` are checked for
  `STABLE` plus safetensors manifest/shards before vLLM starts. Verified with
  `uv run --no-sync ruff check src/prime_rl/entrypoints/launch.py
  tests/unit/test_launch_entrypoint.py` and
  `uv run --no-sync pytest tests/unit/test_launch_entrypoint.py`.
- `4585070` (`1e-6` step `75`) and `4585072` (`3e-6` step `25`) failed after
  readiness with no matching stable checkpoint even though local discovery sees
  the requested broadcasts.
- `4585074` (`3e-6` step `75`) was cancelled; `node_7.log` showed a vLLM
  engine-core initialization failure on `nid010501`, and the router kept
  health-checking unhealthy DP ranks.
- Submitted retries:
  - `4585649`: `1e-6` step `75`.
  - `4585647`: `3e-6` step `25`.
  - `4585648`: `3e-6` step `75`.
- Persistent monitor is now:
  - PID: `24242`
  - script:
    `outputs/omni_math2_rlvr_canary/monitors/postrun_eval_monitor_20260513_stepsplit.sh`
  - status:
    `outputs/omni_math2_rlvr_canary/postrun_eval_monitor_20260513_stepsplit.md`
- At the `12:07 UTC` refresh, active evals were `4585069`, `4585071`,
  `4585073`, `4585323`, `4585324`, and `4585647`; `4585648` and `4585649`
  were queued. Partial rows were `1e-6 step50=429`,
  `3e-6 step100=289`, and `3e-6 step50=230`.

2026-05-13 12:16 UTC correction:

- The checkpoint discovery failures were not compute-side filesystem
  visibility. Actual bug: `scripts/evals/offline_omni_math2_ckpt_eval.py`
  still applied default `--step-interval 50` when `--steps` was explicit. That
  filtered out explicit steps `25`, `75`, and `85` after vLLM startup.
- Patched the eval script so explicit `--steps` disables interval/min/max
  filters. Added `tests/unit/test_offline_omni_math2_ckpt_eval.py`.
- Verified:
  - `uv run --no-sync ruff check scripts/evals/offline_omni_math2_ckpt_eval.py
    tests/unit/test_offline_omni_math2_ckpt_eval.py
    src/prime_rl/entrypoints/launch.py tests/unit/test_launch_entrypoint.py`
  - `uv run --no-sync pytest tests/unit/test_offline_omni_math2_ckpt_eval.py
    tests/unit/test_launch_entrypoint.py`
- Cancelled pre-patch doomed jobs: `4585323`, `4585648`, `4585649`.
- Already failed pre-patch jobs from this same bug: `4585324`, `4585647`.
- Submitted corrected retries:
  - `4586007`: `1e-6` step `25`.
  - `4586010`: `1e-6` step `75`.
  - `4585994`: `1e-6` step `85`.
  - `4586008`: `3e-6` step `25`.
  - `4586009`: `3e-6` step `75`.
- Monitor PID is now `44199`; status file remains
  `outputs/omni_math2_rlvr_canary/postrun_eval_monitor_20260513_stepsplit.md`.

2026-05-13 12:24 UTC live confirmation:

- Corrected retry jobs all reached checkpoint evaluation in their Slurm logs:
  - `4586007`: `1e-6` step `25`.
  - `4586010`: `1e-6` step `75`.
  - `4585994`: `1e-6` step `85`.
  - `4586008`: `3e-6` step `25`.
  - `4586009`: `3e-6` step `75`.
- Partial rollout files have appeared for the corrected retry outputs, so the
  explicit-step fix is validated in the live Slurm path.
