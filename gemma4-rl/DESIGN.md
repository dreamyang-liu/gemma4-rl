# GRPO for Gemma 4 31B on swegemma shell_v2 trajectories

Goal: RL trajectories identical in format to the collection logs
`gemma4-agent/runs/cand_logs/g4-glmcproxy/col_glm{5s,v}_*.jsonl`, and training tokens identical to the sampled tokens.

## Decisions (user, 2026-10-25)
- Agent: `gemma4-agent/agents/shell_v2` (28 calls, 8 min, 240 s per command, 80 turns).
- ADK compaction stays on. Compaction summarizer calls are trained too.
- Plain GRPO. No SPROUT, no review/branching.
- All 8 H200 are available.
- Open: init weights (recommended: LoRA on `models/gemma-4-31b-it-qat-dequant-bf16`).

## Components
1. **Harness (unchanged):** `gemma4-agent/rebench_run.sh agents/shell_v2 <name> --tasks <iid>
   --external-endpoint http://<proxy>/w/<name>/v1`. This is the code that made the collection logs. Reward is the
   `resolved` field in `results.jsonl` (swegemma Phase 2).
2. **TITO proxy (new):** an OpenAI Chat Completions server. It runs vLLM 0.19.1's own `OpenAIServingChat`
   (render, validation, sampling defaults, gemma4 reasoning and tool parsers, ids, finish_reason, usage) on CPU,
   with a custom `EngineClient` that sends the generation to SGLang's native `/generate` (token ids in, token ids
   and logprobs out).
   - The prompt token ids are vLLM's. Each request is rendered from scratch, as on Kaggle, so the template drops
     earlier thoughts after a user message exactly as vLLM does.
   - Sampling defaults follow vLLM: generation_config (top_k 64), eos ids {1, 106, 50}, max_tokens rules.
   - The thinking budget follows vLLM's `ThinkingTokenBudgetLogitsProcessor` state machine. SGLang generates in
     chunks between the processor's check points, and the proxy inserts the forced `<channel|>` tokens. Forced tokens
     are marked so that they get loss_mask 0.
   - Logs keep the collection-log record format (`ts, dt, sim_sleep, status, request, bedrock_response,
     chat_response`) and add a `tito` field: prompt_ids, output_ids, output_logprobs, forced mask.
   - No resampling of bad tool calls or summaries (the teacher proxy did that; RL must not).
3. **Agent server (new, host side):** `POST /run {sample_id, instance_id}` starts the harness for one
   trajectory, waits, then returns the reward and the TITO records of that trajectory.
4. **Miles rollout function (new):** for each GRPO group (one task, n samples), it calls the agent server and
   builds Samples:
   - It merges consecutive calls into one Sample while `prompt_{t+1}` starts with `prompt_t + output_t` (token
     level). Otherwise it starts a new Sample. Summarizer calls are their own Samples.
   - loss_mask is 1 on sampled output tokens and 0 on prompt, tool/user tokens and forced tokens.
   - Every Sample of a trajectory gets the trajectory reward. The advantage is normalized per trajectory within the
     task group, then broadcast to its Samples.
   - A trajectory that fails for infrastructure reasons (no records, harness crash) is dropped, not given reward 0.

## Findings (2026-10-25)
- **The thinking budget depends on prefill chunking.** vLLM updates the budget processor on every engine step,
  including prompt-only prefill chunk steps. A prompt that needs k prefill steps therefore reaches the forced
  `<channel|>` k-1 steps earlier. Kaggle runs one request at a time with max_num_batched_tokens 2048. The proxy
  models this as prefill steps = ceil((P - cached) / 2048), where `cached` is the block-aligned longest common
  prefix with recent sequences, at most P-1. Against a Kaggle-configured vLLM (60 sequential calls, budget 24, 42 of
  them forced, 1 to 5 prefill steps), every forced position matched.
- **Re-rendered history is not the sampled tokens.** The chat template renders a past thought as
  `'<|channel>thought\n' + text + '\n<channel|>'`. The model samples `text<channel|>` without that newline, so
  P_{t+1} never extends P_t + O_t. Exact TITO therefore gives one segment per model call. In the first e2e trajectory
  this was 27 calls -> 27 segments, about 287k tokens for 16.8k trained tokens. Merging by re-rendering (as
  traj_to_sft.py does for SFT) would be about 60k tokens, but it would train one extra "\n" per turn that the model
  never sampled.
- **First e2e trajectory** (3yourmind__django-migration-linter-186, SGLang + proxy): resolved, 23 tool calls,
  371 s, with one compaction. format_diff.py passed on every check: system prompt, tools and the task prompt are
  byte-identical to the collection log.

- **Sliding-window FlashAttention-3 failed verification** (replay of 48 saved segments, SGD lr 1): each kernel is
  reproducible (unfused bit-identical, FA3 gradient cosine 0.999 run to run), but FA3 vs unfused gives gradient cosine
  0.075 and log-prob |diff| 8.8e-3. Per the user, training uses the original unfused attention.
- **DAPO run dapo_e15** (2026-10-25): dynamic sampling + token-level loss + clip-higher, 16 tasks x 8 samples per step,
  15 epochs (490 steps), checkpoint every 5 steps, wandb project gemma4-rl. Estimated ~10 h of training per step at the
  measured ~1.05k tokens/s (per-call segments: ~38M tokens per step).

- **Switched to one sample per compaction segment** (user, 2026-10-25; tito_samples.segments_merged, agent server
  --sample-mode merged). Consecutive calls share a sample while P_{t+1} extends P_t, call t ended with a tool call, and
  the re-render of O_t in P_{t+1} matches >= 95% of O_t; compaction, summarizer calls and nudges start new samples.
  The sample is P_last + O_last. Earlier outputs appear re-rendered and are aligned token by token with what was
  sampled; matched tokens are trained with their sampling logprob, the re-render-only tokens (the extra newline
  before <channel|>, re-tokenized pieces) are not. On 32 trajectories: 4.6 samples and 50.2k tokens per trajectory
  (per call: 27.7 and 266.6k), 99.84% of sampled tokens trained. An output that the history drops (e.g. after a forced
  thought end) fails the 95% rule, so its call ends the segment and is trained exactly as sampled.

## Attention: the real finding (2026-10-25 night)
- **Megatron's unfused attention ignored the sliding window for sequences <= 4096 tokens.** `FusedScaleMaskSoftmax`
  picks the apex fused causal kernel when `sk <= 4096`; that kernel has no sliding-window support, and the window mask
  is only built on the torch fallback path. So every sliding layer attended to the full history on short sequences
  and to a 1024-token window on long ones. Verified with bench/layer_probe.py against an fp64 reference: at seq 4096
  the unfused output is 43% away from the windowed reference and 0.37% from the unwindowed one; at 4100 it flips.
  Upstream Megatron-LM fixed this (`and self.window_size is None` in `is_kernel_available`, and `mask | window`);
  miles_patches.py item 6 applies the same fix.
- **FlashAttention-3 was right all along.** It is 0.2% from the fp64 reference at 2k/4k/8k; the earlier "FA3 fails
  gradient verification (cosine 0.075)" compared it against the buggy unfused on a mixed-length batch. The correct
  window is `(1023, 0)`: a token sees itself and the 1023 before it (`(1024, 0)` is 1.9% off).
- Fixed unfused: 0.37-0.40% from fp64 at all lengths. FA3: 0.20%. Global layers (head_dim 512): unfused was always
  correct (0.37%); TE falls back to its own unfused there (no fused backend for d=512), 4k 2.4 ms, 16k 76 ms.

- **Real-activation accuracy (bench/capture_probe.py, loaded weights, seq 3000, softmax_scale 1.0 as the model
  sets it).** Fixed unfused: out 0.26-0.73%, grads 0.6-1.7% vs fp64. FA3 (sliding layers): out 0.17%, grads
  0.2-1.0%. Triton d=512 (global layers, kernels/attn_d512.py): out 0.17%, grads 0.2-0.65%. Both new kernels are
  more accurate than unfused on real activations (fp32 accumulation of P@V; unfused multiplies a bf16 P).
  Pitfall: `DotProductAttention(softmax_scale=...)` is a constructor argument, not read from config; Gemma 4 passes
  1.0 (Q is pre-scaled). A probe that omits it builds a 1/sqrt(d) instance and is 40-90% off.
- Triton d=512 kernel: no shipped FlashAttention takes head_dim 512 on SM90 (FA2-Triton 128, FA3/FA4 256; Flex and
  SDPA-efficient are 3x slower than the fp32 score matrix). The kernel splits d into two 256 halves with one fp32
  score accumulator. Shared memory limits tiles to 64x32 (1 stage) or 64x16 (2 stages); 8k 18 ms, 16k 67 ms, 20k
  105 ms per TP rank vs the math path 21/83/OOM ms. Accuracy equals FA3's.

- **Full-model gradient check (bench/fullmodel_grad_diff.py, TP=1, loaded weights, 1024 real tokens).** FA3 vs
  fixed unfused: loss rel diff 7.4e-4; o_proj weight-gradient cosine 0.996 at layer 59 falling to 0.30 at layer 0.
  Control, the same unfused model with a 1-ulp bf16 jitter on the embedding output: loss rel diff 1.7e-3 and
  cosine 0.14 at layer 0 with gradient norms 2.7x off. The 60-layer bf16 backward is chaotic in the bottom layers for
  any ~0.4% perturbation; FA3's deviation is inside the baseline's own noise floor, so FA3 is precision-neutral, and
  on forward accuracy it is better than unfused (0.17% vs 0.5-0.7% vs fp64). The Triton d=512 kernel, though accurate
  per layer, drives layers 15-26 to negative cosine with norms 0.35x (a different signature from the control), so it
  stays off (G4RL_GLOBAL_D512=0) until that is understood (suspect: the K=V tied tensor; .contiguous() copies).
- Apples-to-apples control: perturbing each sliding layer's attention output by 0.5% gaussian noise (the unfused
  kernel's own error vs fp64) inside the unfused model gives layer-0 cosine 0.28 and norms 0.34x. FA3 vs unfused is
  0.38 and 1.08x: FA3 is closer to the baseline than the baseline is to itself under its own error.
- Profile (TP=1, seq 1024, FA3 sliding + unfused global): attention is 4% of GPU time, GEMMs and elementwise ops 96%.
  The remaining attention cost is the 10 global layers at long sequences only.
- **Gradient precision against an fp32 ground truth (bench/fullmodel_grad_diff.py --truth; same model, attention in
  fp32 math with the layer's window and scale).** Per-layer o_proj weight gradient: the FA3 config is closer to the
  truth than fixed unfused in 51/60 layers (e.g. layer 40: rel 0.035 vs 0.130; layer 50: 0.106 vs 0.642; layer 0:
  0.91 vs 1.08). Both are far from truth in layers 0-25 (bf16 depth chaos), FA3 less so. Loss: unfused 8.3e-4,
  FA3 1.6e-3 from truth, both within bf16 rounding. So "precision not worse" holds at the gradient level, scored
  against truth rather than against the baseline. (The d=512 config's aggregate looked worse, but see the next item:
  below layer ~30 the aggregate cannot rank two accurate kernels.)
- **Per-layer accuracy on real in-model inputs vs fp32 truth (same q, k, v, dout; `--truth` mode).** Global layers,
  unfused: out 4.0-5.8e-3, dq/dk/dv 4.5-11e-3. Triton d=512: out 0.8-1.5e-3, dq/dk/dv 1.9-3.7e-3, i.e. 3-5x more
  accurate on every quantity in every global layer. Sliding layers, FA3 vs unfused: 1.0e-3 vs 6-11e-3 (out).
  **What the full-model aggregate can and cannot say:** below layer ~30 the 60-layer bf16 backward is chaotic, so two
  implementations that are both more accurate than the baseline still end orthogonal to each other and to the fp32
  reference; the aggregate cannot rank them there. In the uncontaminated top layers (40-59, cos >= 0.99 for all),
  the d512 config is closer to truth than unfused in every layer (59: 0.031 vs 0.038; 50: 0.076 vs 0.642; 40: 0.042
  vs 0.130), as is FA3. The precision criterion for a kernel change is therefore per-layer error on real inputs, plus
  the top-layer composition; by both, FA3 and d512 are improvements over unfused.
- **Triton d=512 kernel: resolved, kept off on speed.** Accuracy on real activations at 3k and 12k tokens: better than
  unfused on every quantity in every global layer (out 1.7e-3 vs 2.4e-3; dk 3-5e-3 vs 9e-3; dv 2.3e-3 vs 8-10e-3),
  with the model's K=V aliasing and GQA 32/4, and in linear->attn->linear wiring (gradient into the residual 1.9e-2
  vs 4.2e-2 for bf16 math). Speed in the real Miles path (TP=4, 48 merged segments): log_probs 75.1 s vs 72.1 s,
  train step 245.6 s vs 231.1 s with unfused global layers. The kernel only beats the fp32 score matrix above ~8k
  tokens... (first explanation; superseded below).
  Follow-up: Triton recompiled the four kernels for every new sequence-length class (~3-5 s each; integer args are
  specialized on divisibility by 16). Fixed by excluding seqlen/n_rep/the lse row stride from specialization and by
  dropping the batch dimension from the kernels (batch strides scaled with seqlen): now one 5.3 s compile per
  process, then <1 ms per new length, at full speed (103 ms at 19.6k) and unchanged accuracy. With that fixed, the
  replay still measures d512 slower: log_probs 75.4 s vs 72.1 s, train step 248.8 s vs 231.1 s. In-model at TP=1
  (32 heads/rank) d512 halves global-layer attention (393 vs 756 ms forward at 12k, whole forward 15% faster), but
  Miles trains at TP=4 with 8 query heads and 1 KV head per rank: the kernel's grid (cdiv(s,64) x 8 programs) is too
  small to fill 132 SMs over the causal triangle, while cuBLAS bmm is indifferent to head count. So on the actual
  training configuration the generic path is faster, and GLOBAL_D512=0 stays the default. The kernel is correct,
  more accurate than unfused, compile-stable, and the right choice at TP<=2 or for long-context work.
- **Decision (2026-10-25 night): train with FA3 on the sliding layers + the fused-softmax window fix; global layers
  keep the fixed unfused.** Measured on the 48-segment replay (TP=4): log_probs 72 s vs 129 s, train step 235 s vs
  437 s, 1.86x overall.

## Parity tests (before any training)
- Prompt ids: proxy render vs real vLLM 0.19.1 `prompt_token_ids` (`return_token_ids`) on collection-log requests.
- Post-processing: feed the output token ids that a real vLLM server produced into the proxy's fake engine. The
  chat responses must be byte-identical.
- Thinking budget: with a small budget on real vLLM, the proxy's state machine must mark the same forced positions.
- Format: diff RL rollout logs against the collection logs (system prompt, tools, tool-result shapes, nudges,
  budget_warning, compaction structure).

## Colocate (8 GPUs) made to work — Oct 25, 2026 evening

Production 4+4 run `dapo_4x4_e15` (2 steps, 58 min/step) showed the real rollout problem: trajectories were
**time-starved**, not format-different. format_diff over 584 trajectories / 6995 requests: system prompt, tools,
params, tool shapes and all 61 task prompts byte-identical to the collection logs; compaction (29% of trajectories,
up to 3), nudges and get_status come from the same harness code. But 85% of trajectories hit the 8-minute session
budget with ~12.6 tool calls (vLLM W4A16 eval: 24 calls, 4% hit the budget), 51% ended without a patch, so 47% of
groups were uninformative. Causes: (1) one TP1 SGLang engine per GPU has an 85k-token KV pool (bf16 weights take
62 GB of the 119 GB budget) and reserves max_tokens=8192 per running request, so only 10-12 requests ran per engine
while 50-64 queued; (2) bf16 decode on one H200 is ~30-45 tok/s per sequence with the triton backend, and a call
averages ~340 output tokens, so 28 calls need ~8 min of model time even unloaded (model time was 75% of the session).
The harness warns about remaining tool calls (budget_warning from 18/28 on) but never about time.

Fix: colocate with **two TP4 engines** (`ENGINE_TP=4`, Miles' Gemma 4 recipe layout): 487k-token KV per engine,
per-sequence decode 56-76 tok/s, request latency median 2 s (TP1: 8.4 s), model time per trajectory 145-155 s
(TP1: 368 s), 0/16 trajectories hit the 8-minute budget, 44% reach 28 calls, 6% empty patches (smoke 4/5, 16
concurrent). Training on 8 GPUs (DP2) halves log_probs/train time. Per-step overhead of the switch ~1.2 min.

Miles patches needed (all in miles_patches.py, applied at container start):
- 7: sleep()/wake_up() gated on is_lora_enabled (not lora_rollout_enabled) so --lora-train-only keeps the
  no-backup LoRA param buffers resident; otherwise pause(tag=None) discards them and update_weights reads unmapped
  memory ("cpu_backup does not exist", actor dies).
- 8: update_weights maps the frozen base back for the merged export: with it asleep, the bridge export still reads
  Gemma 4 buffer-like params (layer_scalar/scale) and module state through the live model -> Xid 31 MMU fault on
  every rank, engines hang in the IPC copy. Likely the crash behind the Miles recipe's "keep resident" note.
- 9: model built in its own memory-saver region ("g4rl_model", CPU backup) so 8 maps back only 16 GB. Without it
  the export peaked at 133 GB: after a training step the trainer keeps ~58 GB live per rank: 15.5 GB weights, a
  ~6 GB IPC bucket the weight updater never frees (allocated outside the memory saver, never offloaded), and
  ~36 GB of grad-carrying activations of the last micro-batch that Miles/Megatron still reference (bounded, not
  cumulative; open optimization). Export peak is now 49.9 GB with TP4 engines.
- 10: G4RL_DUMP_LIVE=1 prints the largest live CUDA storages before offload (diagnostic).
Also: Miles' static micro-batch schedule needs the step's sample count to be a multiple of dp_size (2 in colocate);
gemma4_rollout.py pads with advantage-0 fillers that **borrow an existing rollout_id** (global_batch_size counts
rollout_ids and drops leftovers; a pad with its own id was dropped and the count stayed odd).

Production colocate run `dapo_colo_e15` (launched 21:57 UTC): COLOCATE=1 ENGINE_TP=4 DAPO=1 OVERSAMPLE=2.0
NUM_EPOCH=15 SAVE_INTERVAL=5 WANDB=1 MAX_CONCURRENT=128, 16 tasks x 8, --sglang-schedule-conservativeness 0.3,
separate log-prob pass kept, 8-minute budget and agents/shell_v2 unchanged. `agents/shell_v2_15m` (15-minute
budget in eval_config.yaml and the system prompt HARD LIMITS line) exists but is unused; the submitted agent would
have to change with it. Monitor pipelines must not end in `cut` (fully buffered here); use `sed -u` to truncate.

Update 22:20 UTC: `dapo_colo_e15` stopped after its first rollout (at MAX_CONCURRENT=128 each TP4 engine decoded ~41
sequences at once, per-sequence speed fell to ~33 tok/s and 25% of trajectories hit the 8-minute budget again).
Relaunched as `dapo_colo12_e15` per the user's decision: AGENT=agents/shell_v2_12m (eval_config.yaml
max_time_minutes 12, **system prompt unchanged at "8 minutes"** so prompts stay byte-identical to the collection logs;
only get_status reports the 12-minute budget), OVERSAMPLE=1.6, MAX_CONCURRENT=96 (~30 sequences per engine, ~42 tok/s).

### Patched image (Oct 25, 23:40 UTC)
`docker/build_miles_image.sh` builds `gemma4-miles:g4rl` (also tagged `g4rl-MMDD`) = `gemma4-miles:patched` +
`miles_patches.py` applied at build time + `kernels/` copied to /root/g4rl/kernels. The marker `/root/.g4rl_patched`
holds the sha256 of the applied script; `run_experiment.sh`/`replay_experiment.sh` default to this image, skip the
runtime patch when the marker matches the mounted script, refuse to start (exit 2) when it differs (rebuild), and still
patch at start on an image without the marker (old `gemma4-miles:patched`, via MILES_IMAGE=...). Verified: marker check
passes, re-applying fails (targets gone), a stale marker aborts. The running `dapo_colo12_e15` still uses the old image
with runtime patching (same code).

### Solved-task skip (user-approved Oct 26, 00:05 UTC) — run `dapo_colo12s_e15`
At a ~70% resolve rate 40-55% of the sampled groups are solved by all 8 trajectories (no DAPO signal), so steps needed
2-4 waves (35-45 min of rollout). `gemma4_rollout.py` now keeps per-task stats in runs/<name>/task_stats.json
(attempts, trajectories, passes, last_all_pass) and, with G4RL_SKIP_SOLVED=1, skips drawing a task whose last group was
all-pass except with probability G4RL_SOLVED_RETRY_P (0.2, a regression check); all-fail tasks are never skipped. The
Miles data source cycles epochs by itself, so a skip only consumes the task's slot in the epoch order. Offline test:
tests/test_rollout_filter.py (synthetic easy/mixed/hard tasks; waves, fillers, DP padding, skip, persistence).
`dapo_colo12s_e15` (00:05 UTC) = dapo_colo12_e15 config + SKIP_SOLVED=1, task_stats.json seeded from the 77 complete
groups the base policy had already run (45 tasks, 22 flagged solved; every fresh launch draws the same task order
because Miles shuffles with the epoch id). First image-based launch (gemma4-miles:g4rl). The machine rebooted at
~23:15 UTC Oct 25 during step 1 of dapo_colo12_e15 (no checkpoint); its logs are runs/dapo_colo12_e15_reboot1/ and
_reboot2/.
Update 00:12 UTC Oct 26: relaunched `dapo_colo12s_e15` with OVERSAMPLE=2.0 (user's choice; first wave 32 groups = 256
trajectories, with the solved-task skip expected to yield >= 16 usable groups in one wave). Seeded task_stats.json kept.
Note (Oct 26, 03:05 UTC): with agents/shell_v2_12m the harness writes "Time allowance: 12.0 minutes" into the task
message's "## Task Budget" section while the system prompt still says 8 minutes; the task prompt therefore differs from
the collection logs by that line (get_status also reports 12). User decision: accept, keep the run going.
tools/render_trajectory.py renders a proxy log as a markdown transcript (example: runs/dapo_colo12s_e15/samples/).

### Orphan generations killed the run at step 5 (Oct 26, 04:02 UTC) — patch 11
`dapo_colo12s_e15` ran 4 full steps (49-58 min each, one rollout wave each, pg_loss -0.016/+0.003/-0.021/+0.040) and
died right after the 5th rollout, before its first checkpoint: Miles' `release_memory_occupation` calls
`/flush_cache` (60 x 1 s) and one engine kept answering "Cache not flushed because there are pending requests,
#running-req: 1". A trajectory killed by the harness at its session budget leaves its generation running on SGLang
(the TITO proxy posts a plain /generate and does not cancel on client disconnect), so the flush never succeeds.
Fix (patch 11): `abort_all_requests()` before `flush_cache()` in `sglang_api_client.release_memory_occupation` — every
rollout has returned by then, so anything running is an orphan. Image rebuilt (`gemma4-miles:g4rl`, g4rl-1026).
Relaunched 04:15 UTC with the same config, task_stats.json carried over (180 tasks, 57 flagged solved) and
SAVE_INTERVAL=2 (a failure now costs at most ~2 h instead of ~4; LoRA checkpoints take 1 s). Open follow-up: make the
proxy abort the SGLang request when its client disconnects, which would also stop orphans from occupying engines.

### Replaying the lost rollouts (user-approved, Oct 26 05:00 UTC)
`tools/export_replay.py` rebuilds a run's rollouts as replay steps (`runs/<dst>/replay_steps/step<k>.json`: per wave
the groups' instance_ids and the 8 agent-server-style results, segmented with the same `tito_samples.segments_merged`;
plus the data-buffer draws the step consumed = groups run + solved-task draws skipped). `gemma4_rollout.generate_rollout`
replays such a step when G4RL_REPLAY_DIR is set (REPLAY_STEPS=1): recorded groups/results replace the harness, the buffer
is advanced by the recorded draws so later live steps draw what the source run would have drawn, task stats are not
re-counted. tests/test_rollout_replay.py reproduced the five lost steps exactly (16 kept groups, 531/499/474/499/492
segments, dp-pads, draws). Relaunched `dapo_colo12s_e15` 05:00 UTC: steps 0-4 replayed (~31 min each, no rollout),
step 5 on live. Caveat: steps 1-4 were sampled by the source run's theta_1..4; the replay's theta_k' differ only by
training nondeterminism, so the data is very slightly off-policy for those steps (watch train_rollout_logprob_abs_diff).

### Resuming a LoRA run (Oct 26, 06:15 UTC) — patch 12, RESUME=1
Checkpoints are LoRA-only: `megatron/iter_NNNNNNN/adapter/` (adapter_megatron_rank*.pt, training_state_rank*.pt with the
DistributedOptimizer state and the iteration, adapter_model.safetensors, adapter_config.json; ~3 GB) plus the rollout-side
state `megatron/rollout/<iteration>/` (data-source position, output snapshotter). No `latest_checkpointed_iteration.txt`
is written, so a plain relaunch with `--load /root/out/megatron` silently starts from the HF base (Miles rewrites
args.load). Resume = `RESUME=1 ./run_experiment.sh <same name> ...` -> `--lora-adapter-path .../iter_X/adapter`:
restore_lora_adapter loads adapter + optimizer, start_rollout_id = X + 1, and rollout_executor.load(X) reads
`megatron/rollout/X` — which needed patch 12 (it used the rewritten args.load = HF base). Untested on GPUs so far
(the production run occupies them); fallback if it fails: export + replay (tools/export_replay.py), which is proven.

### Replay divergence at step 3 and "phantom" records (Oct 26, 06:40 UTC)
Replayed steps 0-2 reproduced the lost run's pg_loss to the last digit (-0.01550032174386781, 0.0031049185753939594,
-0.020811326504171213): training is deterministic here. Step 3 did not (0.0493 vs 0.0395, grad_norm 19.5 vs 3.1):
the export included 4 proxy-log records that completed 4-118 s AFTER their harness run had ended (generations still
running on the engine when the harness killed the trajectory at its session budget; the live agent server read the log
before they landed). 14.7k extra tokens, but with the trajectory's reward attached they dominated that step's gradient
(clipped to norm 1 like every step, so one noisy update). tools/export_replay.py now drops records with
ts + dt > mtime(results.jsonl) + 2 s; the corrected export reproduces the live sample statistics exactly (step 3 mean
response length 4598.62 incl. the pad). Steps 1/2 also had late records but in groups that were not kept; step 4 none.
Decision: continue the run (theta_4' differs from the lost theta_4 by that one update); corrected step files are in
runs/dapo_colo12s_e15/replay_steps/ for any future RESUME. Root cause to fix later: the TITO proxy should abort the
SGLang generation when its client disconnects (patch 11 only cleans up before offload).

### Adaptive oversampling (code ready, not yet active) — Oct 26, 10:50 UTC
Live steps 6-8 after the replay: 3 / 1 / 3 waves, 57-65% of groups uninformative (the policy solves more tasks: 62%
resolve rate, 58% of trajectories use all 28 calls, trajectories mean 380 s with a pytest tail up to 1468 s), rollouts
38-42 min, training 35-39 min (560-624 segments) -> 72-80 min per step. The solved-task skip cannot help in epoch 1
(every draw is a new task; 97 of 298 seen tasks flagged solved so far). `gemma4_rollout.py` now sizes the first wave from
an EMA of the usable-group rate kept in task_stats.json under `__usable_rate__` when G4RL_ADAPTIVE_OVERSAMPLE=1
(ADAPTIVE_OVERSAMPLE=1 in run_experiment.sh): groups = ceil(deficit / rate), clamped to [OVERSAMPLE, 3.5] x deficit;
offline tests pass. To activate: RESUME=1 ADAPTIVE_OVERSAMPLE=1 ... at a checkpoint boundary (also the first real test of
RESUME). Other levers noted: stop surplus trajectories once 16 usable groups are complete; lower the verification timeout
(900 s -> 300-400 s) to cut the wave tail.

### Checkpoint mirror on Hugging Face (Oct 26, 18:25 UTC)
Private repo `mliu444/gemma4-rl-dapo-colo12s` (README describes the layout). `tools/hf_sync.py <run dir> <repo> --loop 15`
runs in the background (log: runs/<run>/hf_sync.log) and uploads every complete checkpoint
(`megatron/iter_N/adapter` incl. per-rank optimizer state, `megatron/rollout/N/data_source`), task_stats.json and
DESIGN.md; checkpoints already on the hub are skipped. Also `tools/task_progress.py <run prefixes>` compares pass counts of
tasks attempted at several steps (epoch 2 started at step 14; the solved-task skip fired for the first time there).

### OOM at step 19 (Oct 26, 23:53 UTC) — global-layer d512 kernel switched on
Training of rollout 18 (742 segments) died with CUDA OOM on rank 3 during the recompute backward of a layer: "tried to
allocate 11.82 GiB, 10 GiB free, 125 GiB in use (103 GiB allocated by PyTorch)". Training-phase memory had grown with the
segment lengths (78-85 GB at step 1 -> 98-108 GB by step 11 -> 119 GB at step 18) because the unfused global layers
materialise [8, s, s] fp32 scores (+ bf16 probs) and segments now approach the 32k context; the trainer also carries
~42 GB of residue (6 GB IPC bucket + ~36 GB stale activations of the previous step). Job died after 18 completed steps;
last checkpoint iter_0000017 (after step 18); the completed rollout of step 19 was exported as replay_steps/step18.json
(exporter now has --steps; draws [41, 2] = 32 + 9 skipped and the 2-group second wave).
Fix: GLOBAL_D512=1 — kernels/attn_d512.py (FlashAttention-2-style, O(s) memory) for the 10 global layers; verified more
accurate than the unfused path on real activations, ~8% slower at TP4 in the replay. Relaunch = first real test of
RESUME=1 (restores iter_0000017 + rollout/17 data-source state, start_rollout_id 18) + replay of step 18 + live from 19.
Resume attempt 1 (00:00 UTC Oct 27) restored the adapter + optimizer ("Loaded 360 adapter tensors ... Restored optimizer
state") but Miles logged "restored rollout 18, and --start-rollout-id 0 was asked for, so it starts at 0": in bridge mode
resolve_args_checkpoint_load sets args.start_rollout_id = 0 whenever --load has no Megatron tracker (always, for LoRA
checkpoints), and create_training_model treats any non-None value as an explicit request. The run began replaying step 0
on theta_18; killed within 2 minutes, before any optimizer step or save (checkpoints intact). run_experiment.sh RESUME now
passes --start-rollout-id <iter+1> itself (iter from the adapter dir name).
Resume attempt 2 (00:06 UTC) with --start-rollout-id 18 still showed start_rollout_id 0 in the args dump: the override in
resolve_args_checkpoint_load runs after argument parsing and clobbers the explicit value too. Killed during model load
(no training, checkpoints intact). Patch 13 keeps args.start_rollout_id untouched when a LoRA adapter is being restored
or when --start-rollout-id was given; the fresh-run path (no adapter) still gets 0. Image rebuilt (13 patches).
Resume attempt 3 (00:11 UTC Oct 27) worked end to end: adapter + optimizer restored, rollout/17 data-source state loaded,
"rollout 18: REPLAY of ... step 18" with the live run's counts, training started. With GLOBAL_D512=1 on every global-layer
call the log-prob pass took 830 s for 708 segments (596 s for 742 segments without it): at 8 heads per TP rank the
Triton kernel underfills the GPU and the gap grows with the sequence length. Training-phase memory 104-112 GB (119 GB
before). New knob G4RL_D512_MIN_SEQ / D512_MIN_SEQ: the d512 kernel only from that length on (16384 keeps the fp32 score
matrix <= 8 GB), shorter segments stay on cuBLAS + fp32 softmax. Image rebuilt; to be applied with RESUME at the next
checkpoint. Next memory target: the ~36 GB of stale activations the trainer keeps between steps.
Resume 3 died at 01:11 UTC after ~34 min of actor_train: ranks 1-7 timed out (600 s) waiting for rank 0 to publish an
NCCL unique id (c10d store key /default_pg/0//112//cuda//0) — rank 0 stalled or died first; its log vanished with the
container (--rm), no OOM/Xid/Python error in the driver log. Suspect but unproven: the d512 kernel on every global-layer
call including ~32k-token segments in backward (the forward and 34 min of backward had run). Resume 4 (01:14 UTC):
same checkpoint/replay, GLOBAL_D512=1 with D512_MIN_SEQ=16384 (d512 only for >= 16k tokens). If the same step stalls
again, drop d512 and protect memory by capping segment length instead. Follow-up: keep Ray worker logs on the host
(mount /tmp/ray or stop using --rm) so a dead rank's log survives.
Diagnosis of the 01:11 "NCCL timeout" (Oct 27, 02:20 UTC): not a crash but a **DP straggler**. During actor_train the
two DP groups (TP ranks 0-3 and 4-7) get 354 micro-batches each (equal counts, lengths vary 1k-28k); the group with
the longer segments finishes later, and if it lags more than Megatron's distributed timeout (10 min, the c10d store
wait when the end-of-step collective's NCCL communicator is created) the waiting group raises DistBackendError and the
job dies. Resume 4 showed the same pattern (GPUs 4-7 idle at 100% for GPUs 0-3 near the step end). d512 on the long
segments makes the slow group slower. Fix: `--distributed-timeout-minutes 120` (Megatron arg, accepted by Miles) on the
next launch; a better DP balance (by estimated attention FLOPs) would remove the idle time itself.
