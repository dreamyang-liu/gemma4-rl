"""Compare two replay runs (replay_experiment.sh) that trained one step on identical inputs.

  - per-token log-probs of the sampled tokens (loss_mask 1): run A vs run B, and each vs the rollout (SGLang) values;
  - gradients: with SGD, lr 1, no clipping and zero-initialized LoRA B, each saved lora B matrix is minus the gradient.
    Reports cosine similarity and relative difference per tensor and over all tensors.

usage (gemma4-miles image): python compare_replay.py RUN_DIR_A RUN_DIR_B
"""
import glob
import math
import os
import sys

import torch


def load_train_data(run):
    files = sorted(glob.glob(os.path.join(run, 'train_data_*.pt')))
    assert files, f'no train_data_*.pt in {run}'
    return torch.load(files[0], weights_only=False, map_location='cpu')['rollout_data']


def masked_tokens(data, key):
    out = []
    for lp, m in zip(data[key], data['loss_masks']):
        lp = torch.as_tensor(lp, dtype=torch.float64)
        m = torch.as_tensor(m, dtype=torch.bool)
        out.append(lp[-len(m):][m] if len(lp) >= len(m) else lp[m[:len(lp)]])
    return torch.cat(out)


def load_adapter(run):
    files = sorted(glob.glob(os.path.join(run, 'megatron', 'iter_*', 'adapter', '**', '*.safetensors'), recursive=True))
    if files:
        from safetensors.torch import load_file
        t = {}
        for f in files:
            t.update(load_file(f))
        return t
    files = sorted(glob.glob(os.path.join(run, 'megatron', 'iter_*', 'adapter', '**', '*.pt'), recursive=True))
    assert files, f'no adapter weights under {run}/megatron'
    t = {}
    for f in files:
        obj = torch.load(f, weights_only=False, map_location='cpu')
        t.update({k: v for k, v in (obj.items() if isinstance(obj, dict) else []) if torch.is_tensor(v)})
    return t


a_dir, b_dir = sys.argv[1:3]
da, db = load_train_data(a_dir), load_train_data(b_dir)
print('train data keys:', sorted(da.keys()))
la, lb = masked_tokens(da, 'log_probs'), masked_tokens(db, 'log_probs')
ra = masked_tokens(da, 'rollout_log_probs')
assert la.shape == lb.shape == ra.shape, (la.shape, lb.shape, ra.shape)
print(f'{la.numel()} sampled tokens')
for name, x, y in (('A vs B', la, lb), ('A vs rollout', la, ra), ('B vs rollout', lb, ra)):
    d = (x - y).abs()
    print(f'  log-prob |{name}|: mean {d.mean():.2e}  p99 {d.quantile(0.99):.2e}  max {d.max():.2e}')

ta, tb = load_adapter(a_dir), load_adapter(b_dir)
keys = sorted(set(ta) & set(tb))
print(f'adapter tensors: {len(ta)} in A, {len(tb)} in B, {len(keys)} shared')
dots = na = nb = nd = 0.0
worst = []
for k in keys:
    x, y = ta[k].double().flatten(), tb[k].double().flatten()
    if 'lora_A' in k or 'linear_in' in k:
        continue  # A is unchanged after one step from B = 0 (its gradient is zero)
    dots += float(x @ y); na += float(x @ x); nb += float(y @ y); nd += float((x - y) @ (x - y))
    if x.norm() > 0:
        worst.append((float(torch.nn.functional.cosine_similarity(x, y, dim=0)), float((x - y).norm() / x.norm()), k))
cos = dots / math.sqrt(na * nb) if na and nb else float('nan')
print(f'gradient over all lora B: cosine {cos:.6f}  relative diff {math.sqrt(nd / na):.2e}  |A| {math.sqrt(na):.3e} |B| {math.sqrt(nb):.3e}')
worst.sort()
for c, r, k in worst[:5]:
    print(f'  lowest cosine: {c:.5f}  rel diff {r:.2e}  {k}')
