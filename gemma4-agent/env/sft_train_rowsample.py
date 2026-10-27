"""Full-weight SFT of Gemma 4 (Gemma4ForConditionalGeneration, text only) on pre-tokenized rows (input_ids, labels).

torchrun --nproc_per_node 8 sft_train.py --model DIR --data DIR/tokenized --out DIR [--steps 500 --global-batch 16]

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
ap.add_argument('--global-batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--min-lr-ratio', type=float, default=0.1)
ap.add_argument('--warmup', type=int, default=25)
ap.add_argument('--weight-decay', type=float, default=0.0)
ap.add_argument('--max-len', type=int, default=18432)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--loss-chunk', type=int, default=2048)
ap.add_argument('--attn', default='sdpa', help='sdpa | flex_attention | eager')
ap.add_argument('--cpu-test', action='store_true', help='gloo/CPU smoke test with a tiny random model')
args = ap.parse_args()

dist.init_process_group('gloo' if args.cpu_test else 'nccl')
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get('LOCAL_RANK', 0))
dev = torch.device('cpu') if args.cpu_test else torch.device('cuda', local)
if not args.cpu_test:
    torch.cuda.set_device(dev)
log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
assert args.global_batch % world == 0
accum = args.global_batch // world

# ---------------- data: fixed random sample of rows, rank r takes every world-th row of each step ----------------
files = sorted(glob.glob(os.path.join(args.data, '*.parquet')))
counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
index = [(fi, ri) for fi, n in enumerate(counts) for ri in range(n)]
rng = random.Random(args.seed)
rng.shuffle(index)
need = args.steps * args.global_batch
picked = index[:need]
mine = [(s, picked[s * args.global_batch + rank * accum + k]) for s in range(args.steps) for k in range(accum)]
by_file = {}
for s, (fi, ri) in mine:
    by_file.setdefault(fi, set()).add(ri)
rows = {}
for fi, want in by_file.items():
    t = pq.read_table(files[fi], columns=['input_ids', 'labels'])
    for ri in want:
        ids = np.asarray(t['input_ids'][ri].as_py(), dtype=np.int64)[:args.max_len]
        lab = np.asarray(t['labels'][ri].as_py(), dtype=np.int64)[:args.max_len]
        rows[(fi, ri)] = (ids, lab)
# labeled-token count per step over the whole global batch (for normalization)
local_tok = torch.zeros(args.steps, dtype=torch.float64)
for s, key in mine:
    local_tok[s] += float((rows[key][1][1:] != -100).sum())
step_tok = local_tok.to(dev)
dist.all_reduce(step_tok)
log(f'world={world} accum={accum} rows/rank={len(mine)} labeled tokens/step mean={step_tok.mean().item():.0f}')

# ---------------- model ----------------
from transformers import AutoConfig, Gemma4ForConditionalGeneration  # noqa: E402

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


os.makedirs(args.out, exist_ok=True)
logf = open(os.path.join(args.out, 'train_log.jsonl'), 'a') if rank == 0 else None
t0 = time.time()
for s in range(args.steps):
    for g in opt.param_groups:
        g['lr'] = lr_at(s)
    loss_acc = torch.zeros((), device=dev, dtype=torch.float64)
    tok_step = 0
    for k in range(accum):
        ids, lab = rows[mine[s * accum + k][1]]
        ids_t = torch.from_numpy(ids)[None].to(dev)
        lab_t = torch.from_numpy(lab)[None].to(dev)
        tok_step += len(ids)
        loss_sum = lm(ids_t, lab_t)
        # grads are averaged over ranks by FSDP; scale so the result is sum / global labeled tokens
        (loss_sum * world / step_tok[s]).backward()
        loss_acc += loss_sum.detach().double()
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

# ---------------- save bf16 HF checkpoint ----------------
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict  # noqa: E402

sd = get_model_state_dict(lm, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
if rank == 0:
    sd = {k[len('m.'):] if k.startswith('m.') else k: v.to(torch.bfloat16) for k, v in sd.items()}
    with torch.device('meta'):
        out_model = Gemma4ForConditionalGeneration(model.config)
    out_model.load_state_dict(sd, strict=False, assign=True)
    out_model.tie_weights()
    out_model.save_pretrained(args.out, safe_serialization=True, max_shard_size='5GB')
    if not args.cpu_test:
        for f in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'processor_config.json',
                  'generation_config.json'):
            if os.path.exists(os.path.join(args.model, f)):
                shutil.copy(os.path.join(args.model, f), os.path.join(args.out, f))
    log(f'saved to {args.out} in {time.time() - t0:.0f}s total')
dist.barrier()
dist.destroy_process_group()
