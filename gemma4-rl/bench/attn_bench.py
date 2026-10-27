"""Attention kernels for Gemma 4 31B training, per TP4 rank: forward + backward time and peak memory.

Global layers: 8 query heads, 1 KV head (K = V), head_dim 512, causal.
Local layers: 8 query heads, 4 KV heads, head_dim 256, causal sliding window of 1024.
Megatron currently runs every layer with the unfused (math) attention. Candidates: torch SDPA backends, FlexAttention,
flash-attn (head_dim <= 256 only). Each candidate is checked against the math result (forward and gradients).

usage (gemma4-miles image, one GPU): python attn_bench.py [--seqs 8192,16384]
"""
import argparse
import math
import time

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--seqs', default='8192,16384')
ap.add_argument('--iters', type=int, default=3)
ap.add_argument('--fp64-ref', action='store_true', help='compare every candidate (bf16 math included) to an fp64 reference')
args = ap.parse_args()
dev, dt = 'cuda', torch.bfloat16
WINDOW = 1024


def make(s, hq, hkv, d):
    q = torch.randn(1, hq, s, d, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(1, hkv, s, d, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(1, hkv, s, d, device=dev, dtype=dt, requires_grad=True)
    return q, k, v


def math_attn(q, k, v, window=None):
    """Megatron-style unfused attention: full score matrix, fp32 softmax."""
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = q.shape[2]
    scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1])
    i = torch.arange(s, device=dev)
    mask = i[None, :] > i[:, None]
    if window:
        mask |= (i[:, None] - i[None, :]) >= window
    scores.masked_fill_(mask, float('-inf'))
    return (scores.softmax(-1).to(dt) @ v)


def sdpa(backend):
    from torch.nn.attention import sdpa_kernel

    def f(q, k, v, window=None):
        assert window is None, 'SDPA fused kernels take no sliding window'
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    return f


_flex_cache = {}


def flex(q, k, v, window=None):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    s = q.shape[2]
    key = (s, window)
    if key not in _flex_cache:
        def causal(b, h, qi, ki):
            return qi >= ki

        def sliding(b, h, qi, ki):
            return (qi >= ki) & (qi - ki < window)
        _flex_cache[key] = (create_block_mask(sliding if window else causal, None, None, s, s, device=dev),
                            torch.compile(flex_attention, dynamic=False))
    bm, fn = _flex_cache[key]
    return fn(q, k, v, block_mask=bm, enable_gqa=True)


def flash(q, k, v, window=None):
    from flash_attn import flash_attn_func
    o = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True,
                        window_size=(window - 1, 0) if window else (-1, -1))
    return o.transpose(1, 2)


def flash3(q, k, v, window=None):
    from flash_attn_interface import flash_attn_func
    o = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True,
                        window_size=(window - 1, 0) if window else (-1, -1))
    o = o[0] if isinstance(o, tuple) else o
    return o.transpose(1, 2)


_te_cache = {}


def te_attn(q, k, v, window=None):
    """Transformer Engine DotProductAttention (picks cuDNN fused or flash, whichever supports the shape)."""
    import transformer_engine.pytorch as tep
    hq, hkv, d = q.shape[1], k.shape[1], q.shape[-1]
    key = (hq, hkv, d, window)
    if key not in _te_cache:
        _te_cache[key] = tep.DotProductAttention(
            num_attention_heads=hq, kv_channels=d, num_gqa_groups=hkv, attn_mask_type='causal', qkv_format='bshd',
            window_size=(window - 1, 0) if window else (-1, 0), attention_dropout=0.0)
    o = _te_cache[key](q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())
    return o.view(q.shape[0], q.shape[2], hq, d).transpose(1, 2)


def bench(name, fn, q, k, v, window, ref, g):
    for t in (q, k, v):
        t.grad = None
    try:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        out = fn(q, k, v, window)
        out.backward(g)
        torch.cuda.synchronize()
        def rel(a, b):
            return ((a.double() - b.double()).norm() / b.double().norm()).item()
        err = rel(out, ref[0])
        gerr = max(rel(t.grad, r) for t, r in zip((q, k, v), ref[1]))
        t0 = time.time()
        for _ in range(args.iters):
            for t in (q, k, v):
                t.grad = None
            fn(q, k, v, window).backward(g)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / args.iters * 1000
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f'    {name:16s} {ms:9.1f} ms  peak {mem:6.1f} GiB  rel|out-ref| {err:.2e}  max rel|grad-ref| {gerr:.2e}')
    except Exception as e:
        print(f'    {name:16s} FAILED: {type(e).__name__}: {str(e)[:150]}')


from torch.nn.attention import SDPBackend  # noqa: E402

for s in [int(x) for x in args.seqs.split(',')]:
    for label, (hq, hkv, d, window) in {'global d512': (8, 1, 512, None), 'local d256 w1024': (8, 4, 256, WINDOW)}.items():
        print(f'seq {s} {label}:')
        q, k, v = make(s, hq, hkv, d)
        g = torch.randn(q.shape, device=dev, dtype=dt)
        if args.fp64_ref:
            q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
            rep = q.shape[1] // k.shape[1]
            sc = (q64 @ k64.repeat_interleave(rep, 1).transpose(-1, -2)) / math.sqrt(d)
            i = torch.arange(s, device=dev)
            m = i[None, :] > i[:, None]
            if window:
                m |= (i[:, None] - i[None, :]) >= window
            out = sc.masked_fill(m, float('-inf')).softmax(-1) @ v64.repeat_interleave(rep, 1)
            out.backward(g.double())
            ref = (out.detach(), [t.grad.detach() for t in (q64, k64, v64)])
            del q64, k64, v64, sc
        else:
            out = math_attn(q, k, v, window)
            out.backward(g)
            ref = (out.detach(), [t.grad.detach().clone() for t in (q, k, v)])
        cands = {'math (current)': math_attn, 'flex': flex}
        if window is None:
            cands.update({'sdpa efficient': sdpa(SDPBackend.EFFICIENT_ATTENTION), 'sdpa cudnn': sdpa(SDPBackend.CUDNN_ATTENTION),
                          'sdpa flash': sdpa(SDPBackend.FLASH_ATTENTION)})
        cands['te (cudnn/flash)'] = te_attn
        cands['flash-attn-3'] = flash3
        if d <= 256:
            cands['flash-attn-2'] = flash
        for name, fn in cands.items():
            bench(name, fn, q, k, v, window, ref, g)  # the same upstream gradient as the reference
        del q, k, v, ref
        torch.cuda.empty_cache()
