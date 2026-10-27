"""Causal attention for head_dim 512 (Gemma 4 global layers), forward + backward, in Triton. EXPERIMENTAL: accurate per
layer (more accurate than the unfused path on real activations at 3k and 12k tokens) but not faster end to end on this
workload (it beats the fp32 score matrix only above ~8k tokens); kept off by default (G4RL_GLOBAL_D512=0). See DESIGN.md.

No shipped FlashAttention kernel takes head_dim 512 on H100/H200 (FA2 Triton: 128, FA3/FA4: 256), and the generic
kernels (SDPA efficient, FlexAttention) are slower than the plain fp32 score matrix at this width. This is the
FlashAttention-2 algorithm with the head dimension processed as two 256-wide halves:
  scores = Q[:, :256] K[:, :256]^T + Q[:, 256:] K[:, 256:]^T          (one fp32 accumulator, no d=512 tile)
  out    = P V, as two 256-wide output accumulators.
Online softmax in fp32 over key blocks; the backward recomputes P from the saved log-sum-exp, as FA2 does. GQA is
handled by mapping each query head to its KV head; K=V (Gemma 4 global layers) is simply the same tensor passed twice.

Layout: q [b, s, hq, d], k/v [b, s, hkv, d], bf16, d == 512. Returns o [b, s, hq, d] (and lse in fp32 for tests).
Accuracy target: bf16 output, fp32 accumulation, within bf16 rounding of an fp64 reference (verified by test_attn_d512).
"""
import math

import torch
import triton
import triton.language as tl

HALF = tl.constexpr(256)  # half of head_dim 512; a module constant read inside @triton.jit must be constexpr


@triton.jit(do_not_specialize=['seqlen', 'n_rep', 'stride_lh'])
def _fwd_kernel(Q, K, V, O, LSE, sm_scale, stride_qs, stride_qh, stride_ks, stride_kh,
                stride_os, stride_oh, stride_lh, seqlen, n_rep,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_kvh = off_h // n_rep
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HALF)
    q_base = Q + off_h * stride_qh
    k_base = K + off_kvh * stride_kh
    v_base = V + off_kvh * stride_kh
    m_mask = offs_m < seqlen
    q1 = tl.load(q_base + offs_m[:, None] * stride_qs + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    q2 = tl.load(q_base + offs_m[:, None] * stride_qs + (offs_d[None, :] + HALF), mask=m_mask[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc1 = tl.zeros([BLOCK_M, HALF], tl.float32)
    acc2 = tl.zeros([BLOCK_M, HALF], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634  # exp2 domain
    hi = (start_m + 1) * BLOCK_M
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < seqlen
        k1 = tl.load(k_base + n_idx[:, None] * stride_ks + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        k2 = tl.load(k_base + n_idx[:, None] * stride_ks + (offs_d[None, :] + HALF), mask=n_mask[:, None], other=0.0)
        s = tl.dot(q1, tl.trans(k1)) + tl.dot(q2, tl.trans(k2))
        s = s * qk_scale
        causal = offs_m[:, None] >= n_idx[None, :]
        s = tl.where(causal & n_mask[None, :], s, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc1 = acc1 * alpha[:, None]
        acc2 = acc2 * alpha[:, None]
        v1 = tl.load(v_base + n_idx[:, None] * stride_ks + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        v2 = tl.load(v_base + n_idx[:, None] * stride_ks + (offs_d[None, :] + HALF), mask=n_mask[:, None], other=0.0)
        p16 = p.to(v1.dtype)
        acc1 += tl.dot(p16, v1)
        acc2 += tl.dot(p16, v2)
        m_i = m_new
    acc1 = acc1 / l_i[:, None]
    acc2 = acc2 / l_i[:, None]
    o_base = O + off_h * stride_oh
    tl.store(o_base + offs_m[:, None] * stride_os + offs_d[None, :], acc1.to(O.dtype.element_ty), mask=m_mask[:, None])
    tl.store(o_base + offs_m[:, None] * stride_os + (offs_d[None, :] + HALF), acc2.to(O.dtype.element_ty), mask=m_mask[:, None])
    lse = m_i + tl.math.log2(l_i)  # in exp2 domain; log-sum-exp of scaled scores = lse * ln2
    tl.store(LSE + off_h * stride_lh + offs_m, lse, mask=m_mask)


@triton.jit(do_not_specialize=['seqlen', 'stride_dh'])
def _bwd_preprocess(O, DO, Delta, stride_os, stride_oh, stride_dh, seqlen, BLOCK_M: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HALF)
    m_mask = offs_m < seqlen
    base = off_h * stride_oh + offs_m[:, None] * stride_os
    o1 = tl.load(O + base + offs_d[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
    o2 = tl.load(O + base + (offs_d[None, :] + HALF), mask=m_mask[:, None], other=0.0).to(tl.float32)
    do1 = tl.load(DO + base + offs_d[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
    do2 = tl.load(DO + base + (offs_d[None, :] + HALF), mask=m_mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o1 * do1, 1) + tl.sum(o2 * do2, 1)
    tl.store(Delta + off_h * stride_dh + offs_m, delta, mask=m_mask)


@triton.jit(do_not_specialize=['seqlen', 'n_rep', 'stride_lh'])
def _bwd_kv_kernel(Q, K, V, DO, LSE, Delta, DK, DV, sm_scale, stride_qs, stride_qh, stride_ks,
                   stride_kh, stride_lh, stride_dks, stride_dkh, seqlen, n_rep,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """dK, dV for one key block, looping over the query blocks that see it; accumulates over the n_rep query heads
    of this KV head (GQA), so each KV head is owned by exactly one program and needs no atomics."""
    start_n = tl.program_id(0)
    off_kvh = tl.program_id(1)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HALF)
    n_mask = offs_n < seqlen
    k_base = K + off_kvh * stride_kh
    v_base = V + off_kvh * stride_kh
    k1 = tl.load(k_base + offs_n[:, None] * stride_ks + offs_d[None, :], mask=n_mask[:, None], other=0.0)
    k2 = tl.load(k_base + offs_n[:, None] * stride_ks + (offs_d[None, :] + HALF), mask=n_mask[:, None], other=0.0)
    v1 = tl.load(v_base + offs_n[:, None] * stride_ks + offs_d[None, :], mask=n_mask[:, None], other=0.0)
    v2 = tl.load(v_base + offs_n[:, None] * stride_ks + (offs_d[None, :] + HALF), mask=n_mask[:, None], other=0.0)
    dk1 = tl.zeros([BLOCK_N, HALF], tl.float32)
    dk2 = tl.zeros([BLOCK_N, HALF], tl.float32)
    dv1 = tl.zeros([BLOCK_N, HALF], tl.float32)
    dv2 = tl.zeros([BLOCK_N, HALF], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    for rep in range(0, n_rep):
        off_h = off_kvh * n_rep + rep
        q_base = Q + off_h * stride_qh
        do_base = DO + off_h * stride_qh
        lse_base = LSE + off_h * stride_lh
        d_base = Delta + off_h * stride_lh
        lo = start_n * BLOCK_N
        lo = (lo // BLOCK_M) * BLOCK_M
        for start_m in range(lo, seqlen, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + offs_m_base
            m_mask = offs_m < seqlen
            q1 = tl.load(q_base + offs_m[:, None] * stride_qs + offs_d[None, :], mask=m_mask[:, None], other=0.0)
            q2 = tl.load(q_base + offs_m[:, None] * stride_qs + (offs_d[None, :] + HALF), mask=m_mask[:, None], other=0.0)
            # s^T: [BLOCK_N, BLOCK_M]
            sT = tl.dot(k1, tl.trans(q1)) + tl.dot(k2, tl.trans(q2))
            sT = sT * qk_scale
            lse = tl.load(lse_base + offs_m, mask=m_mask, other=float('inf'))
            causal = offs_m[None, :] >= offs_n[:, None]
            pT = tl.math.exp2(sT - lse[None, :])
            pT = tl.where(causal & m_mask[None, :], pT, 0.0)
            do1 = tl.load(do_base + offs_m[:, None] * stride_qs + offs_d[None, :], mask=m_mask[:, None], other=0.0)
            do2 = tl.load(do_base + offs_m[:, None] * stride_qs + (offs_d[None, :] + HALF), mask=m_mask[:, None], other=0.0)
            pT16 = pT.to(do1.dtype)
            dv1 += tl.dot(pT16, do1)
            dv2 += tl.dot(pT16, do2)
            delta = tl.load(d_base + offs_m, mask=m_mask, other=0.0)
            dpT = tl.dot(v1, tl.trans(do1)) + tl.dot(v2, tl.trans(do2))
            dsT = pT * (dpT - delta[None, :]) * sm_scale
            dsT16 = dsT.to(q1.dtype)
            dk1 += tl.dot(dsT16, q1)
            dk2 += tl.dot(dsT16, q2)
    dk_base = DK + off_kvh * stride_dkh
    dv_base = DV + off_kvh * stride_dkh
    tl.store(dk_base + offs_n[:, None] * stride_dks + offs_d[None, :], dk1.to(DK.dtype.element_ty), mask=n_mask[:, None])
    tl.store(dk_base + offs_n[:, None] * stride_dks + (offs_d[None, :] + HALF), dk2.to(DK.dtype.element_ty), mask=n_mask[:, None])
    tl.store(dv_base + offs_n[:, None] * stride_dks + offs_d[None, :], dv1.to(DV.dtype.element_ty), mask=n_mask[:, None])
    tl.store(dv_base + offs_n[:, None] * stride_dks + (offs_d[None, :] + HALF), dv2.to(DV.dtype.element_ty), mask=n_mask[:, None])


@triton.jit(do_not_specialize=['seqlen', 'n_rep', 'stride_lh'])
def _bwd_q_kernel(Q, K, V, DO, LSE, Delta, DQ, sm_scale, stride_qs, stride_qh, stride_ks,
                  stride_kh, stride_lh, seqlen, n_rep, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """dQ for one query block, looping over its key blocks."""
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_kvh = off_h // n_rep
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HALF)
    m_mask = offs_m < seqlen
    q_base = Q + off_h * stride_qh
    do_base = DO + off_h * stride_qh
    k_base = K + off_kvh * stride_kh
    v_base = V + off_kvh * stride_kh
    q1 = tl.load(q_base + offs_m[:, None] * stride_qs + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    q2 = tl.load(q_base + offs_m[:, None] * stride_qs + (offs_d[None, :] + HALF), mask=m_mask[:, None], other=0.0)
    do1 = tl.load(do_base + offs_m[:, None] * stride_qs + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    do2 = tl.load(do_base + offs_m[:, None] * stride_qs + (offs_d[None, :] + HALF), mask=m_mask[:, None], other=0.0)
    lse = tl.load(LSE + off_h * stride_lh + offs_m, mask=m_mask, other=float('inf'))
    delta = tl.load(Delta + off_h * stride_lh + offs_m, mask=m_mask, other=0.0)
    dq1 = tl.zeros([BLOCK_M, HALF], tl.float32)
    dq2 = tl.zeros([BLOCK_M, HALF], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    hi = (start_m + 1) * BLOCK_M
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        n_mask = offs_n < seqlen
        k1 = tl.load(k_base + offs_n[:, None] * stride_ks + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        k2 = tl.load(k_base + offs_n[:, None] * stride_ks + (offs_d[None, :] + HALF), mask=n_mask[:, None], other=0.0)
        s = (tl.dot(q1, tl.trans(k1)) + tl.dot(q2, tl.trans(k2))) * qk_scale
        causal = offs_m[:, None] >= offs_n[None, :]
        p = tl.math.exp2(s - lse[:, None])
        p = tl.where(causal & n_mask[None, :], p, 0.0)
        v1 = tl.load(v_base + offs_n[:, None] * stride_ks + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        v2 = tl.load(v_base + offs_n[:, None] * stride_ks + (offs_d[None, :] + HALF), mask=n_mask[:, None], other=0.0)
        dp = tl.dot(do1, tl.trans(v1)) + tl.dot(do2, tl.trans(v2))
        ds = p * (dp - delta[:, None]) * sm_scale
        ds16 = ds.to(k1.dtype)
        dq1 += tl.dot(ds16, k1)
        dq2 += tl.dot(ds16, k2)
    dq_base = DQ + off_h * stride_qh
    tl.store(dq_base + offs_m[:, None] * stride_qs + offs_d[None, :], dq1.to(DQ.dtype.element_ty), mask=m_mask[:, None])
    tl.store(dq_base + offs_m[:, None] * stride_qs + (offs_d[None, :] + HALF), dq2.to(DQ.dtype.element_ty), mask=m_mask[:, None])


# Triton specializes integer arguments on divisibility by 16: anything that scales with the sequence length (seqlen
# itself, the lse/delta row stride) must be excluded, or every new length class recompiles the four kernels (~3-5 s
# each). Token/head strides are constant per layer shape and stay specialized (they drive vectorized access).
# Tile configuration. A 64x512 bf16 K or V tile is 64 KB, so with two pipeline stages the 64x64 FA2 defaults need
# 320 KB of shared memory (H100/H200 allow 227 KB). CONFIG is set by configure() / the test sweep.
CONFIG = {'BLOCK_M': 64, 'BLOCK_N': 32, 'num_warps': 8, 'num_stages': 1}  # best of the sweep at 8k-20k (H200)


def configure(**kw):
    CONFIG.update(kw)


def _launch_cfg():
    return dict(BLOCK_M=CONFIG['BLOCK_M'], BLOCK_N=CONFIG['BLOCK_N'], num_warps=CONFIG['num_warps'],
                num_stages=CONFIG['num_stages'])


def _fwd(q, k, v, sm_scale):
    b, s, hq, d = q.shape
    hkv = k.shape[2]
    assert d == 512 and k.shape[3] == d and v.shape == k.shape and hq % hkv == 0
    o = torch.empty_like(q)
    lse = torch.empty((b, hq, s), device=q.device, dtype=torch.float32)
    c = _launch_cfg()
    grid = (triton.cdiv(s, c['BLOCK_M']), hq)
    for bi in range(b):  # one launch per batch element: no batch strides in the kernel, so no per-length recompilation
        qb, kb, vb, ob, lb = q[bi], k[bi], v[bi], o[bi], lse[bi]
        _fwd_kernel[grid](qb, kb, vb, ob, lb, sm_scale, qb.stride(0), qb.stride(1), kb.stride(0), kb.stride(1),
                          ob.stride(0), ob.stride(1), lb.stride(0), s, hq // hkv, **c)
    return o, lse


def _bwd(q, k, v, o, do, lse, sm_scale):
    b, s, hq, d = q.shape
    hkv = k.shape[2]
    do = do.contiguous()
    delta = torch.empty_like(lse)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    c = _launch_cfg()
    for bi in range(b):
        qb, kb, vb, ob, dob, lb, deb = q[bi], k[bi], v[bi], o[bi], do[bi], lse[bi], delta[bi]
        dqb, dkb, dvb = dq[bi], dk[bi], dv[bi]
        _bwd_preprocess[(triton.cdiv(s, c['BLOCK_M']), hq)](ob, dob, deb, ob.stride(0), ob.stride(1), deb.stride(0), s,
                                                             BLOCK_M=c['BLOCK_M'], num_warps=4)
        _bwd_kv_kernel[(triton.cdiv(s, c['BLOCK_N']), hkv)](
            qb, kb, vb, dob, lb, deb, dkb, dvb, sm_scale, qb.stride(0), qb.stride(1), kb.stride(0), kb.stride(1),
            lb.stride(0), dkb.stride(0), dkb.stride(1), s, hq // hkv, **c)
        _bwd_q_kernel[(triton.cdiv(s, c['BLOCK_M']), hq)](
            qb, kb, vb, dob, lb, deb, dqb, sm_scale, qb.stride(0), qb.stride(1), kb.stride(0), kb.stride(1),
            lb.stride(0), s, hq // hkv, **c)
    return dq, dk, dv


class _AttnD512(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o, lse = _fwd(q, k, v, sm_scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _bwd(q, k, v, o, do, lse, ctx.sm_scale)
        return dq, dk, dv, None


def attn_d512_causal(q, k, v, softmax_scale=None):
    """Causal attention, q [b, s, hq, 512], k/v [b, s, hkv, 512] (hq % hkv == 0), bf16/fp16. Returns [b, s, hq, 512]."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    return _AttnD512.apply(q, k, v, softmax_scale)
