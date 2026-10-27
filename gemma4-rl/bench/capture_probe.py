"""Capture the real inputs of one sliding layer's core attention inside the Gemma 4 model (TP=1, bf16, no LoRA), then
replay the fixed-unfused parent and the FA3 subclass on exactly those tensors and compare output and gradients.

This closes the gap between the synthetic layer probe (which agreed to 0.5%) and the full-model replay (whose LoRA
gradients disagreed): same class, same shapes/dtypes/mask/strides as the model produces, real activations.

usage (gemma4-miles image, one GPU, ~70 GB): python capture_probe.py --layer 1 --seq 3000
"""
import argparse
import copy
import math
import os
import runpy

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/root/models/base')
ap.add_argument('--seq', type=int, default=3000)
ap.add_argument('--layers', default='1,2,30,59', help='1-indexed layer numbers to capture (sliding layers)')
args = ap.parse_args()

os.environ['G4RL_SLIDING_FLASH'] = '0'
runpy.run_path('/root/g4rl/miles_patches.py')  # fixed unfused everywhere; FA3 subclass is built separately below
from megatron.core import parallel_state  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.enums import AttnMaskType  # noqa: E402

torch.distributed.init_process_group('nccl', init_method='tcp://127.0.0.1:29513', world_size=1, rank=0)
parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
torch.cuda.set_device(0)
model_parallel_cuda_manual_seed(0)
from megatron.bridge import AutoBridge  # noqa: E402

bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=True)
provider.attention_backend = 'unfused'
provider.attention_softmax_in_fp32 = True
provider.bf16 = True
provider.params_dtype = torch.bfloat16
provider.finalize()
model = provider.provide().cuda().bfloat16()
bridge.load_hf_weights(model)  # provide() builds; the weights are loaded here (provide_distributed_model does it via a hook)
model.eval()
import megatron.bridge.models.gemma.modeling_gemma4 as mg  # noqa: E402

layers = [int(x) for x in args.layers.split(',')]
captured = {}


def make_hook(L):
    def hook(mod, inputs, output):
        q, k, v, mask = inputs[0], inputs[1], inputs[2], inputs[3]
        captured[L] = dict(q=q.detach().clone(), k=k.detach().clone(), v=v.detach().clone(),
                           mask=None if mask is None else (mask.detach().clone() if torch.is_tensor(mask) else mask),
                           mask_type=type(mask).__name__, q_stride=q.stride(), k_stride=k.stride(),
                           kwargs_keys=sorted(getattr(hook, 'kw', {}).keys()))
    return hook


cores = {}
for name, m in model.named_modules():
    if type(m).__name__ in ('DotProductAttention',) and '.self_attention.core_attention' in name:
        L = int(name.split('.layers.')[1].split('.')[0]) + 1
        if L in layers:
            cores[L] = m
            m.register_forward_hook(make_hook(L), with_kwargs=False)
print('hooked layers:', sorted(cores))
tok_ids = torch.randint(10, 200000, (1, args.seq), device='cuda')
with torch.no_grad():
    out = model(input_ids=tok_ids, position_ids=None, attention_mask=None, labels=None, loss_mask=None)
logits = out[0] if isinstance(out, tuple) else out
print('forward done; logits', logits.shape, 'std', logits.float().std().item(), 'argmax[:8]', logits[0, :8].argmax(-1).tolist())
for L, c in captured.items():
    assert c['q'].float().abs().max() > 0 and c['k'].float().abs().max() > 0, f'layer {L}: zero q/k, weights not loaded'
# the core-attention modules hold no parameters; free the 62 GB model so long-sequence references fit
del out, logits
keep = {L: cores[L] for L in cores}
del model
import gc
gc.collect(); torch.cuda.empty_cache()
cores = keep

# Build the FA3 subclass of the same base class with the same config as the model's layer.
os.environ['G4RL_SLIDING_FLASH'] = '1'
from megatron.core.models.backends import LocalSpecProvider  # noqa: E402

FlashCls = mg._gemma4_core_attention(LocalSpecProvider())


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


for L in sorted(captured):
    c = captured[L]
    base = cores[L]
    cfg = base.config
    flash = FlashCls(config=cfg, layer_number=L, attn_mask_type=base.attn_mask_type, attention_type='self', softmax_scale=base.softmax_scale).cuda()
    # Construction diff: every scalar/str/tuple attribute that differs between the model's instance and the fresh one.
    def scalars(m):
        d = {}
        for k2, v2 in vars(m).items():
            if isinstance(v2, (int, float, str, bool, tuple, type(None))) or hasattr(v2, 'name'):
                d[k2] = v2
        sm = m.scale_mask_softmax
        for k2 in ('scale', 'window_size', 'attn_mask_type', 'scaled_masked_softmax_fusion', 'softmax_in_fp32', 'input_in_fp16', 'input_in_bf16'):
            d['scale_mask_softmax.' + k2] = getattr(sm, k2, 'MISSING')
        return d
    db, df = scalars(base), scalars(flash)
    diffs = {k2: (db.get(k2, 'MISSING'), df.get(k2, 'MISSING')) for k2 in set(db) | set(df) if db.get(k2, 'MISSING') != df.get(k2, 'MISSING')}
    print(f'   attribute diffs (model instance vs fresh): {diffs if diffs else "none"}')
    print(f'   model instance: class {type(base).__name__}, softmax_scale {base.softmax_scale}, fresh softmax_scale {flash.softmax_scale}, '
          f'config.softmax_scale {cfg.softmax_scale}, config.kv_channels {cfg.kv_channels}, hidden/head base {base.hidden_size_per_attention_head} fresh {flash.hidden_size_per_attention_head}')
    print(f'layer {L}: q {tuple(c["q"].shape)} stride {c["q_stride"]} k {tuple(c["k"].shape)} stride {c["k_stride"]} '
          f'mask {c["mask_type"]}{"" if c["mask"] is None else tuple(c["mask"].shape)} dtype {c["q"].dtype} '
          f'softmax_scale {base.softmax_scale} window {getattr(base.scale_mask_softmax, "window_size", None)} '
          f'flash.g4_flash {flash.g4_flash}')
    q, k, v = (t.clone().requires_grad_() for t in (c['q'], c['k'], c['v']))
    g = torch.randn(args.seq, 1, q.shape[2] * q.shape[3], device='cuda', dtype=q.dtype)
    base.train(); flash.train()
    outs, grads = {}, {}
    flash_off = FlashCls(config=cfg, layer_number=L, attn_mask_type=base.attn_mask_type, attention_type='self', softmax_scale=base.softmax_scale).cuda()
    flash_off.g4_flash = False; flash_off.g4_d512 = False; flash_off.train()
    for name, mod in (('unfused', base), ('fa3', flash), ('fresh-parent', flash_off)):
        for t in (q, k, v):
            t.grad = None
        o = mod(q, k, v, c['mask'])
        o.backward(g)
        outs[name] = o.detach().clone()
        grads[name] = [t.grad.clone() for t in (q, k, v)]
    # fp64 reference with the window
    rep = q.shape[2] // k.shape[2]
    q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
    s = args.seq
    i = torch.arange(s, device='cuda')
    m = i[None, :] > i[:, None]
    ws = getattr(base.scale_mask_softmax, 'window_size', None)
    if ws:
        m |= (i[:, None] - i[None, :]) > ws[0]
    nh, d = q.shape[2], q.shape[3]
    g64 = g.double().view(s, 1, nh, d)
    outs64, dq64 = [], torch.zeros_like(q64)
    dk64, dv64 = torch.zeros_like(k64), torch.zeros_like(v64)
    for h in range(nh):  # one query head at a time, forward and backward: nothing s^2-sized survives the iteration
        Qh = q64[:, 0, h].detach().requires_grad_()
        Kh = k64[:, 0, h // rep].detach().requires_grad_()
        Vh = v64[:, 0, h // rep].detach().requires_grad_()
        sc = (Qh @ Kh.T) * base.softmax_scale
        sc = sc.masked_fill(m, float('-inf'))
        sc = sc - sc.amax(-1, keepdim=True).detach()
        oh = sc.softmax(-1) @ Vh
        gq, gk, gv = torch.autograd.grad(oh, (Qh, Kh, Vh), g64[:, 0, h])
        outs64.append(oh.detach()); dq64[:, 0, h] = gq; dk64[:, 0, h // rep] += gk; dv64[:, 0, h // rep] += gv
        del sc, oh
    o64 = torch.stack(outs64, 1).reshape(s, 1, -1)
    ref = [o64, dq64, dk64, dv64]
    for name in ('unfused', 'fa3', 'fresh-parent'):
        e = [rel(outs[name], ref[0])] + [rel(a, b) for a, b in zip(grads[name], ref[1:])]
        print(f'   {name:8s} vs fp64: out {e[0]:.2e} dq {e[1]:.2e} dk {e[2]:.2e} dv {e[3]:.2e}')
    e = [rel(outs['fa3'], outs['unfused'])] + [rel(a, b) for a, b in zip(grads['fa3'], grads['unfused'])]
    print(f'   fa3 vs unfused:  out {e[0]:.2e} dq {e[1]:.2e} dk {e[2]:.2e} dv {e[3]:.2e}')
    if q.shape[3] == 512:
        import sys
        sys.path.insert(0, '/root/g4rl/kernels')
        from attn_d512 import attn_d512_causal
        for t in (q, k, v):
            t.grad = None
        o = attn_d512_causal(q.transpose(0, 1).contiguous(), k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous(),
                             base.softmax_scale).transpose(0, 1).reshape(s, 1, -1)
        o.backward(g)
        e = [rel(o.detach(), ref[0])] + [rel(t.grad, r) for t, r in zip((q, k, v), ref[1:])]
        print(f'   d512-triton vs fp64: out {e[0]:.2e} dq {e[1]:.2e} dk {e[2]:.2e} dv {e[3]:.2e}')
        e = [rel(o.detach(), outs['unfused'])] + [rel(t.grad, b) for t, b in zip((q, k, v), grads['unfused'])]
        print(f'   d512-triton vs unfused: out {e[0]:.2e} dq {e[1]:.2e} dk {e[2]:.2e} dv {e[3]:.2e}')
