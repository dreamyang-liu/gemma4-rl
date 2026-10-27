"""Full-weight SFT of Gemma 4 (Gemma4ForConditionalGeneration, text only) on pre-tokenized rows (input_ids, labels).

torchrun --nproc_per_node 8 sft_train.py --model DIR --data DIR/tokenized --out DIR [--steps 250 --rows-per-step 16 --save-every 100]

- FSDP2 per decoder layer + root; bf16 compute, fp32 master weights / grad reduction; activation checkpointing.
- Vision tower is frozen; loss is computed only at labeled positions, in chunks (262k vocab), with the model's
  final logit softcap. Loss is normalized by the number of labeled tokens in the global batch.
- Saves a bf16 HF checkpoint (same format as the base) at the end; vLLM can serve it directly.
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--data', required=True, help='dir of tokenized/*.parquet')
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=500)
ap.add_argument('--rows-per-step', type=int, default=16, help='fill each step with whole trajectories until >= this many rows')
ap.add_argument('--save-every', type=int, default=100)
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--min-lr-ratio', type=float, default=0.1)
ap.add_argument('--warmup', type=int, default=25)
ap.add_argument('--weight-decay', type=float, default=0.0)
ap.add_argument('--max-len', type=int, default=18432)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--loss-chunk', type=int, default=2048)
ap.add_argument('--attn', default='sdpa', help='sdpa | gemma4_fast (flex for sliding layers, causal sdpa for global) | eager')
ap.add_argument('--no-final-save', action='store_true')
ap.add_argument('--debug-grad-norms', default='', help='write per-layer grad norms after step 1 to this json')
ap.add_argument('--cpu-test', action='store_true', help='gloo/CPU smoke test with a tiny random model')
args = ap.parse_args()

dist.init_process_group('gloo' if args.cpu_test else 'nccl')
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get('LOCAL_RANK', 0))
dev = torch.device('cpu') if args.cpu_test else torch.device('cuda', local)
if not args.cpu_test:
    torch.cuda.set_device(dev)
log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

# ---------------- data: whole trajectories per step ----------------
# Trajectories are shuffled (seed) and packed into steps: a step takes whole trajectories (all agent segments and all
# compaction summaries) until it has >= --rows-per-step rows; a trajectory is never split across steps. Each step's rows
# are spread over ranks by token count (longest first, least-loaded rank); ranks with fewer rows run a fully masked
# dummy row so every rank does the same number of forward/backward passes (FSDP collectives).
files = sorted(glob.glob(os.path.join(args.data, '*.parquet')))
meta = []
for fi, f in enumerate(files):
    t = pq.read_table(f, columns=['trajectory_id', 'n_tokens'])
    meta += [(tid, fi, ri, n) for ri, (tid, n) in enumerate(zip(t['trajectory_id'].to_pylist(), t['n_tokens'].to_pylist()))]
by_traj = {}
for tid, fi, ri, n in meta:
    by_traj.setdefault(tid, []).append((fi, ri, min(n, args.max_len)))
traj_ids = sorted(by_traj)
random.Random(args.seed).shuffle(traj_ids)
plan, cur, used = [], [], 0
for tid in traj_ids:
    cur.append(tid)
    if sum(len(by_traj[t]) for t in cur) >= args.rows_per_step:
        plan.append(cur)
        cur = []
        if len(plan) == args.steps:
            break
used_traj = sum(len(p) for p in plan)
assignment = []  # per step: list (per rank) of row keys
for step_trajs in plan:
    rows_s = sorted((r for t in step_trajs for r in by_traj[t]), key=lambda r: -r[2])
    load, buckets = [0] * world, [[] for _ in range(world)]
    for r in rows_s:
        k = min(range(world), key=lambda i: load[i])
        buckets[k].append(r)
        load[k] += r[2]
    assignment.append(buckets)
accums = [max(len(b) for b in buckets) for buckets in assignment]
mine = [(s, (b[rank][k][0], b[rank][k][1]) if k < len(b[rank]) else None)
        for s, b in enumerate(assignment) for k in range(accums[s])]
by_file = {}
for s, key in mine:
    if key is not None:
        by_file.setdefault(key[0], set()).add(key[1])
rows = {}
for fi, want in by_file.items():
    t = pq.read_table(files[fi], columns=['input_ids', 'labels'])
    for ri in want:
        ids = np.asarray(t['input_ids'][ri].as_py(), dtype=np.int64)[:args.max_len]
        lab = np.asarray(t['labels'][ri].as_py(), dtype=np.int64)[:args.max_len]
        rows[(fi, ri)] = (ids, lab)
DUMMY = (np.zeros(16, dtype=np.int64) + 2, np.full(16, -100, dtype=np.int64))  # fully masked filler row
local_tok = torch.zeros(args.steps, dtype=torch.float64)
for s, key in mine:
    if key is not None:
        local_tok[s] += float((rows[key][1][1:] != -100).sum())
step_tok = local_tok.to(dev)
dist.all_reduce(step_tok)
log(f'world={world} steps={len(plan)} trajectories={used_traj} rows={sum(len(r) for t in plan for r in [by_traj[x] for x in t])} '
    f'micro-steps/rank mean={sum(accums) / len(accums):.2f} labeled tokens/step mean={step_tok.mean().item():.0f}')
if rank == 0:
    os.makedirs(args.out, exist_ok=True)
    json.dump({'seed': args.seed, 'steps': [list(p) for p in plan]}, open(os.path.join(args.out, 'train_plan.json'), 'w'))
assert len(plan) == args.steps, f'not enough trajectories for {args.steps} steps'

# ---------------- model ----------------
from transformers import AutoConfig, Gemma4ForConditionalGeneration  # noqa: E402
if args.attn == 'gemma4_fast':
    import fast_attention  # env/fast_attention.py, next to this script
    fast_attention.register()

if args.cpu_test:
    cfg = AutoConfig.from_pretrained(args.model)
    tc = cfg.text_config
    tc.num_hidden_layers, tc.hidden_size, tc.intermediate_size = 2, 64, 128
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 2, 1, 32
    if hasattr(tc, 'global_head_dim'):
        tc.global_head_dim = 32
    for k in ('layer_types',):
        if getattr(tc, k, None):
            setattr(tc, k, getattr(tc, k)[:2])
    vc = cfg.vision_config
    vc.num_hidden_layers, vc.hidden_size, vc.intermediate_size, vc.num_attention_heads = 1, 32, 64, 2
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(cfg).to(torch.bfloat16)
else:
    model = Gemma4ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
model.config.use_cache = False
text_cfg = model.config.get_text_config()
softcap = getattr(text_cfg, 'final_logit_softcapping', None)
for n, p in model.named_parameters():
    p.requires_grad = not any(x in n for x in ('vision', 'multi_modal', 'embed_vision', 'audio'))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f'params: total {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B trainable {n_train / 1e9:.2f}B softcap={softcap}')


class LossModel(nn.Module):
    """Root FSDP module: hidden states -> chunked logits at labeled positions -> summed CE."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, labels):
        hs = self.m.model(input_ids=input_ids, use_cache=False).last_hidden_state[0, :-1]
        tgt = labels[0, 1:]
        sel = tgt != -100
        h, tgt = hs[sel], tgt[sel]
        w = self.m.get_output_embeddings().weight
        # keep the graph alive when a row has no labeled tokens (every rank must run backward under FSDP)
        total = (hs.sum() * 0 + w.sum() * 0).float()
        for i in range(0, h.shape[0], args.loss_chunk):
            lg = (h[i:i + args.loss_chunk] @ w.t()).float()
            if softcap:
                lg = torch.tanh(lg / softcap) * softcap
            total = total + F.cross_entropy(lg, tgt[i:i + args.loss_chunk], reduction='sum')
        return total


from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard  # noqa: E402

lm = LossModel(model)
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
layers = model.model.language_model.layers
for layer in layers:
    fully_shard(layer, mp_policy=mp)
fully_shard(lm, mp_policy=mp)
lm.to(torch.float32)  # fp32 master weights (each rank converts only its own shards)
if not args.cpu_test:
    torch.cuda.empty_cache()

lm.train()  # HF only applies gradient checkpointing in training mode (from_pretrained returns eval mode)
params = [p for p in lm.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)


def lr_at(s):
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    p = (s - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


# ---------------- save bf16 HF checkpoint ----------------
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict  # noqa: E402


def save_checkpoint(out_dir):
    sd = get_model_state_dict(lm, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        ts = time.time()
        sd = {k[len('m.'):] if k.startswith('m.') else k: v.to(torch.bfloat16) for k, v in sd.items()}
        with torch.device('meta'):
            out_model = Gemma4ForConditionalGeneration(model.config)
        out_model.load_state_dict(sd, strict=False, assign=True)
        out_model.tie_weights()
        out_model.save_pretrained(out_dir, safe_serialization=True, max_shard_size='5GB')
        if not args.cpu_test:
            for f in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'processor_config.json',
                      'generation_config.json'):
                if os.path.exists(os.path.join(args.model, f)):
                    shutil.copy(os.path.join(args.model, f), os.path.join(out_dir, f))
        log(f'saved {out_dir} in {time.time() - ts:.0f}s')
    del sd
    dist.barrier()


os.makedirs(args.out, exist_ok=True)
logf = open(os.path.join(args.out, 'train_log.jsonl'), 'a') if rank == 0 else None
t0 = time.time()
for s in range(args.steps):
    for g in opt.param_groups:
        g['lr'] = lr_at(s)
    loss_acc = torch.zeros((), device=dev, dtype=torch.float64)
    tok_step = 0
    my_rows = [key for st, key in mine if st == s]
    for key in my_rows:
        ids, lab = rows[key] if key is not None else DUMMY
        ids_t = torch.from_numpy(ids)[None].to(dev)
        lab_t = torch.from_numpy(lab)[None].to(dev)
        tok_step += len(ids)
        loss_sum = lm(ids_t, lab_t)
        # grads are averaged over ranks by FSDP; scale so the result is sum / global labeled tokens
        (loss_sum * world / step_tok[s]).backward()
        loss_acc += loss_sum.detach().double()
    if args.debug_grad_norms and s == 0:
        import collections, re as _re
        sq = collections.defaultdict(float)
        for n, p in lm.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.to_local() if hasattr(p.grad, 'to_local') else p.grad
            m_ = _re.search(r'layers\.(\d+)\.', n)
            key = f'layer{int(m_.group(1)):02d}' if m_ else n.split('.')[-2] + '.' + n.split('.')[-1]
            sq[key] += float(g.float().pow(2).sum())
        keys = sorted(sq)
        v = torch.tensor([sq[k] for k in keys], device=dev, dtype=torch.float64)
        dist.all_reduce(v)
        if rank == 0:
            json.dump({k: math.sqrt(x) for k, x in zip(keys, v.tolist())}, open(args.debug_grad_norms, 'w'), indent=1)
    gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    gnorm = gnorm.full_tensor() if hasattr(gnorm, 'full_tensor') else gnorm
    opt.step()
    opt.zero_grad(set_to_none=True)
    dist.all_reduce(loss_acc)
    tt = torch.tensor([tok_step], device=dev, dtype=torch.float64)
    dist.all_reduce(tt)
    if rank == 0:
        el = time.time() - t0
        rec = {'step': s + 1, 'loss': (loss_acc / step_tok[s]).item(), 'grad_norm': float(gnorm), 'lr': lr_at(s),
               'tokens': int(tt.item()), 'elapsed_s': round(el, 1), 'eta_min': round(el / (s + 1) * (args.steps - s - 1) / 60, 1)}
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        if s < 5 or (s + 1) % 10 == 0:
            log(json.dumps(rec))
    if args.save_every and (s + 1) % args.save_every == 0 and s + 1 < args.steps:
        save_checkpoint(os.path.join(args.out, f'step{s + 1}'))

if not args.no_final_save:
    save_checkpoint(args.out)
log(f'done in {time.time() - t0:.0f}s total')
dist.barrier()
dist.destroy_process_group()
