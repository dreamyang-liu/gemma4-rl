"""Accuracy (vs fp64) and speed (vs the fp32-score math path) of kernels/attn_d512.py at Gemma 4's global-layer shape.

usage (gemma4-miles image, one GPU): python test_attn_d512.py
"""
import math
import sys
import time

import torch

sys.path.insert(0, '/root/g4rl/kernels')
from attn_d512 import attn_d512_causal  # noqa: E402

dev, dt = 'cuda', torch.bfloat16
torch.manual_seed(0)


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def fp64_ref(q, k, v, scale):
    rep = q.shape[2] // k.shape[2]
    Q, K, V = (t.double().transpose(1, 2) for t in (q, k, v))
    K, V = K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1)
    s = q.shape[1]
    sc = (Q @ K.transpose(-1, -2)) * scale
    i = torch.arange(s, device=dev)
    return (sc.masked_fill(i[None, :] > i[:, None], float('-inf')).softmax(-1) @ V).transpose(1, 2)


def math_bf16(q, k, v, scale):
    rep = q.shape[2] // k.shape[2]
    Q, K, V = (t.transpose(1, 2) for t in (q, k, v))
    K, V = K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1)
    s = q.shape[1]
    sc = (Q @ K.transpose(-1, -2)).float() * scale
    i = torch.arange(s, device=dev)
    return (sc.masked_fill(i[None, :] > i[:, None], float('-inf')).softmax(-1).to(dt) @ V).transpose(1, 2)


def timeit(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


d = 512
print('shape (b=1, hq=8, hkv, d=512), causal.  rel err vs fp64: out / dq / dk / dv;  time fwd+bwd ms: kernel vs math')
for hkv, k_eq_v in ((1, True), (2, False), (8, False)):
    for s in (1, 77, 1000, 2048, 4096, 8192, 16384):
        hq = 8
        q = torch.randn(1, s, hq, d, device=dev, dtype=dt, requires_grad=True)
        k = torch.randn(1, s, hkv, d, device=dev, dtype=dt, requires_grad=True)
        v = (k.detach().clone() if k_eq_v else torch.randn(1, s, hkv, d, device=dev, dtype=dt)).requires_grad_()
        g = torch.randn(1, s, hq, d, device=dev, dtype=dt)
        scale = 1.0 / math.sqrt(d)
        try:
            o = attn_d512_causal(q, k, v, scale)
            o.backward(g)
            got = (o.detach(), q.grad.clone(), k.grad.clone(), v.grad.clone())
            q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
            r = fp64_ref(q64, k64, v64, scale)
            r.backward(g.double())
            ref = (r.detach(), q64.grad, k64.grad, v64.grad)
            errs = [rel(a, b) for a, b in zip(got, ref)]
            # the math path's own error, as the bar to meet
            for t in (q, k, v):
                t.grad = None
            om = math_bf16(q, k, v, scale)
            om.backward(g)
            merrs = [rel(a, b) for a, b in zip((om.detach(), q.grad, k.grad, v.grad), ref)]
            line = (f'hkv={hkv} K=V={k_eq_v!s:5} s={s:6d}: kernel {errs[0]:.2e}/{errs[1]:.2e}/{errs[2]:.2e}/{errs[3]:.2e}'
                    f'   math {merrs[0]:.2e}/{merrs[1]:.2e}/{merrs[2]:.2e}/{merrs[3]:.2e}')
            if s >= 2048:
                def run_k():
                    for t in (q, k, v):
                        t.grad = None
                    attn_d512_causal(q, k, v, scale).backward(g)

                def run_m():
                    for t in (q, k, v):
                        t.grad = None
                    math_bf16(q, k, v, scale).backward(g)
                tk = timeit(run_k)
                tm = timeit(run_m) if s <= 16384 else float('nan')
                line += f'   {tk:7.1f} ms vs {tm:7.1f} ms ({tm / tk:.1f}x)'
            print(line, flush=True)
        except Exception as e:
            print(f'hkv={hkv} K=V={k_eq_v!s:5} s={s:6d}: FAILED {type(e).__name__}: {str(e)[:200]}', flush=True)
        del q, k, v, g
        torch.cuda.empty_cache()
