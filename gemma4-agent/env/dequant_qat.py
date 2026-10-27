"""Dequantize google/gemma-4-31B-it-qat-w4a16-ct (compressed-tensors pack-quantized int4, symmetric, group 32) to a
plain bf16 HF checkpoint with the same tensor names as the bf16 release.

W = q * scale exactly (q in [-8, 7], one bf16 scale per 32 input columns): these are the weights the W4A16 kernel
multiplies by at inference, so a LoRA trained on this checkpoint sees the same base as vLLM serving the QAT model.
Unquantized tensors (vision tower, embeddings, norms, ...) are copied unchanged.

usage (gemma4-kaggle image): dequant_qat.py --src QAT_DIR --out OUT_DIR [--check-unpack]
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--shard-gb', type=float, default=5.0)
ap.add_argument('--check-unpack', action='store_true', help='cross-check unpacking against compressed_tensors')
args = ap.parse_args()

cfg = json.load(open(os.path.join(args.src, 'config.json')))
q = cfg.get('quantization_config') or {}
w = q['config_groups']['group_0']['weights']
assert q.get('format') == 'pack-quantized' and w['num_bits'] == 4 and w['symmetric'] and w['strategy'] == 'group', w
GROUP = w['group_size']


def unpack_int4(packed, shape):
    """int32 [out, in/8] -> int8 [out, in]; value j of a word sits in bits 4j..4j+3, stored with +8 offset."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    u = (packed.unsqueeze(-1) >> shifts) & 0xF
    return (u.reshape(packed.shape[0], -1)[:, :shape[1]] - 8).to(torch.int8)


f = safe_open(os.path.join(args.src, 'model.safetensors'), 'pt')
keys = list(f.keys())
quantized = sorted(k[:-len('.weight_packed')] for k in keys if k.endswith('.weight_packed'))
if args.check_unpack:
    os.environ.setdefault('USER', 'user')
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    for base in quantized[:3] + quantized[-3:]:
        packed, shape = f.get_tensor(base + '.weight_packed'), f.get_tensor(base + '.weight_shape').tolist()
        ref = unpack_from_int32(packed, 4, torch.Size(shape))
        assert torch.equal(unpack_int4(packed, shape), ref.to(torch.int8)), base
    print('unpack matches compressed_tensors on', 6, 'tensors')

os.makedirs(args.out, exist_ok=True)
out_tensors, shard, shard_bytes, weight_map, n_shard = {}, {}, 0, {}, 0
shards = []


def flush():
    global shard, shard_bytes, n_shard
    if not shard:
        return
    n_shard += 1
    name = f'model-{n_shard:05d}.safetensors'
    save_file(shard, os.path.join(args.out, name), metadata={'format': 'pt'})
    for k in shard:
        weight_map[k] = name
    shards.append(name)
    shard, shard_bytes = {}, 0


def add(name, t):
    global shard_bytes
    shard[name] = t.contiguous()
    shard_bytes += t.numel() * t.element_size()
    if shard_bytes >= args.shard_gb * 1e9:
        flush()


qset = set(quantized)
stats = {'quantized': 0, 'copied': 0, 'q_min': 0, 'q_max': 0}
for k in keys:
    if k.endswith(('.weight_scale', '.weight_shape')):
        continue
    if k.endswith('.weight_packed'):
        base = k[:-len('.weight_packed')]
        shape = f.get_tensor(base + '.weight_shape').tolist()
        qv = unpack_int4(f.get_tensor(k), shape)
        scale = f.get_tensor(base + '.weight_scale')
        assert scale.shape == (shape[0], shape[1] // GROUP), (base, scale.shape, shape)
        wt = (qv.to(torch.float32).reshape(shape[0], -1, GROUP) * scale.to(torch.float32).unsqueeze(-1)).reshape(shape)
        add(base + '.weight', wt.to(torch.bfloat16))
        stats['quantized'] += 1
        stats['q_min'] = min(stats['q_min'], int(qv.min()))
        stats['q_max'] = max(stats['q_max'], int(qv.max()))
    elif k == 'lm_head.weight' and cfg.get('tie_word_embeddings'):
        # the QAT file stores a copy of the tied embedding; drop it like the bf16 release does
        assert torch.equal(f.get_tensor(k), f.get_tensor('model.language_model.embed_tokens.weight'))
        stats['dropped_tied_lm_head'] = 1
    else:
        add(k, f.get_tensor(k))
        stats['copied'] += 1
flush()
# rename shards to the HF "model-0000i-of-0000n" convention
total = len(shards)
final_map = {}
for i, name in enumerate(shards, 1):
    new = f'model-{i:05d}-of-{total:05d}.safetensors'
    os.rename(os.path.join(args.out, name), os.path.join(args.out, new))
    for k, v in weight_map.items():
        if v == name:
            final_map[k] = new
size = sum(os.path.getsize(os.path.join(args.out, v)) for v in set(final_map.values()))
json.dump({'metadata': {'total_size': size}, 'weight_map': dict(sorted(final_map.items()))},
          open(os.path.join(args.out, 'model.safetensors.index.json'), 'w'), indent=2)

cfg.pop('quantization_config', None)
for sub in ('text_config', 'vision_config'):
    if isinstance(cfg.get(sub), dict):
        cfg[sub].pop('quantization_config', None)
json.dump(cfg, open(os.path.join(args.out, 'config.json'), 'w'), indent=2)
for fn in ('chat_template.jinja', 'generation_config.json', 'processor_config.json', 'tokenizer.json',
           'tokenizer_config.json'):
    if os.path.exists(os.path.join(args.src, fn)):
        shutil.copy(os.path.join(args.src, fn), os.path.join(args.out, fn))
print(json.dumps({**stats, 'shards': total, 'bytes': size}))
