"""Full-model, same-process, per-layer gradient comparison: fixed unfused vs the G4RL kernels (FA3 on sliding layers,
Triton d=512 on global layers), TP=1, bf16, loaded weights, one real token sequence.

The layer probes agree to <1% on real activations, yet a 4-GPU Miles replay gave orthogonal LoRA gradients. This
script removes Miles, TP and LoRA from the picture: it runs forward+backward on the same model twice (swapping the
core-attention class in place), and reports per layer the relative difference and cosine of (a) the attention
output's gradient w.r.t. the residual stream entering the layer, and (b) the o_proj weight gradient. The first layer
where they diverge is where the problem is.

usage (gemma4-miles image, one GPU): python fullmodel_grad_diff.py --seq 2048 [--replay /root/g4rl/data/replay_smoke1_s0_merged.json]
"""
import argparse
import json
import os
import runpy

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/root/models/base')
ap.add_argument('--seq', type=int, default=2048)
ap.add_argument('--replay', default='', help='take the first real token sequence from this replay json instead of random ids')
ap.add_argument('--d512', action='store_true', help='also swap in the Triton d=512 kernel on global layers')
ap.add_argument('--control', action='store_true',
                help='noise floor: both runs use the fixed unfused path; the second gets a 1-ulp bf16 jitter on the embedding output')
ap.add_argument('--truth', action='store_true',
                help='ground truth: compare fixed unfused and the new kernels against an fp32-math attention (score, softmax, PV in '
                     'fp32, same window/scale) computed in the same model; reports which is closer')
ap.add_argument('--control-attn', action='store_true',
                help='noise floor, apples to apples: both runs use the fixed unfused path; the second perturbs each sliding '
                     "layer's attention output by 0.5% relative gaussian noise (the unfused kernel's own error vs fp64)")
args = ap.parse_args()

os.environ['G4RL_SLIDING_FLASH'] = '1'
os.environ['G4RL_GLOBAL_D512'] = '1' if args.d512 else '0'
runpy.run_path('/root/g4rl/miles_patches.py')
from megatron.core import parallel_state  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402

torch.distributed.init_process_group('nccl', init_method='tcp://127.0.0.1:29514', world_size=1, rank=0)
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
provider.gradient_accumulation_fusion = False  # no DDP here: Megatron's fused path needs weight.main_grad
provider.finalize()
model = provider.provide().cuda().bfloat16()
bridge.load_hf_weights(model)
model.train()
for p in model.parameters():
    p.requires_grad_(False)


# Parameters are frozen (no 62 GB of grad buffers), so make the embedding output require grad: then every activation,
# including the input of each linear_proj, is in the autograd graph and its gradient is defined.
JITTER = {'on': False}


def _embed_hook(mod, inp, out):
    if torch.is_tensor(out):
        o = out.detach()
        if JITTER['on']:
            # flip the last mantissa bit of every element: a 1-ulp bf16 perturbation (~0.4% relative)
            o = (o.view(torch.int16) ^ 1).view(torch.bfloat16)
        return o.requires_grad_(True)
    return out


emb = model.language_model.embedding if hasattr(model, 'language_model') else model.embedding
emb.register_forward_hook(_embed_hook)

if args.replay:
    seg = json.load(open(args.replay))[0]['segments'][0]
    ids = torch.tensor(seg['tokens'][:args.seq], device='cuda')[None]
    print('real tokens from replay, len', ids.shape[1])
else:
    ids = torch.randint(10, 200000, (1, args.seq), device='cuda')
s = ids.shape[1]
labels_mask = torch.zeros(1, s, dtype=torch.bool, device='cuda')
labels_mask[:, s // 2:] = True  # "response" tokens: the last half

# Collect the core-attention modules and their o_proj (linear_proj) per layer.
cores, projs, layer_mods = {}, {}, {}
for name, m in model.named_modules():
    if name.endswith('.self_attention.core_attention') is False and '.layers.' in name and name.split('.layers.')[1].isdigit():
        layer_mods[int(name.split('.layers.')[1])] = m
    if name.endswith('.self_attention.core_attention'):
        L = int(name.split('.layers.')[1].split('.')[0])
        cores[L] = m
    if name.endswith('.self_attention.linear_proj'):
        L = int(name.split('.layers.')[1].split('.')[0])
        projs[L] = m
print(f'{len(cores)} layers; core attention class now: {type(cores[0]).__name__} g4_flash L0 {getattr(cores[0], "g4_flash", None)} '
      f'L5 {getattr(cores[5], "g4_flash", None)} g4_d512 L5 {getattr(cores[5], "g4_d512", None)}')


def run(use_new):
    """Forward+backward; returns per layer: grad of the layer's attention output (input to linear_proj) and
    the linear_proj weight gradient."""
    for L, c in cores.items():
        if use_new:
            c.g4_flash = c._g4_flash_saved
            c.g4_d512 = c._g4_d512_saved
        else:
            c.g4_flash = False
            c.g4_d512 = False
    grads_in, grads_layer_in = {}, {}
    hooks = []
    for L, lm in layer_mods.items():
        def mk2(L):
            def h(mod, gin, gout):
                g0 = next((x for x in gin if x is not None), None)
                if g0 is not None:
                    grads_layer_in[L] = g0.detach().float().clone()
            return h
        hooks.append(lm.register_full_backward_hook(mk2(L)))
    for L, pm in projs.items():
        pm.weight.requires_grad_(True)
        pm.weight.grad = None

        def mk(L):
            def h(mod, gin, gout):
                g0 = gin[0] if gin and gin[0] is not None else gout[0]
                grads_in[L] = g0.detach().float().clone()
            return h
        hooks.append(pm.register_full_backward_hook(mk(L)))
    out = model(input_ids=ids, position_ids=None, attention_mask=None, labels=None, loss_mask=None)
    logits = out[0] if isinstance(out, tuple) else out
    # response tokens only, cross_entropy in fp32 on the selected rows (no full-vocab fp32 log-softmax copy)
    sel = labels_mask[:, 1:][0]
    rows = logits[0, :-1][sel].float()
    tgt = ids[0, 1:][sel]
    loss = torch.nn.functional.cross_entropy(rows, tgt)
    del logits, rows
    loss.backward()
    for h in hooks:
        h.remove()
    wg = {L: pm.weight.grad.detach().float().clone() for L, pm in projs.items()}
    return loss.item(), grads_in, wg, grads_layer_in


for c in cores.values():
    c._g4_flash_saved = getattr(c, 'g4_flash', False)
    c._g4_d512_saved = getattr(c, 'g4_d512', False)

TRUTH = {'on': False}
import math as _math
from megatron.core.transformer.utils import is_layer_window_attention as _iswin


def _truth_forward(self, query, key, value, attention_mask, attn_mask_type=None, attention_bias=None, packed_seq_params=None):
    """fp32 attention with the layer's own window and softmax_scale: the precision reference."""
    sq, b, nh, hn = query.shape
    rep = nh // key.shape[2]
    q = query.float().permute(1, 2, 0, 3)
    k = key.float().permute(1, 2, 0, 3).repeat_interleave(rep, 1)
    v = value.float().permute(1, 2, 0, 3).repeat_interleave(rep, 1)
    sc = (q @ k.transpose(-1, -2)) * self.softmax_scale
    i = torch.arange(sq, device=q.device)
    m = i[None, :] > i[:, None]
    ws = getattr(self.scale_mask_softmax, 'window_size', None)
    if ws:
        m |= (i[:, None] - i[None, :]) > ws[0]
    o = (sc.masked_fill(m, float('-inf')).softmax(-1) @ v).permute(2, 0, 1, 3).reshape(sq, b, nh * hn)
    return o.to(query.dtype)


_orig_forward = type(next(iter(cores.values()))).forward


def _dispatch_forward(self, *a, **k):
    if TRUTH['on']:
        return _truth_forward(self, *a, **k)
    return _orig_forward(self, *a, **k)


type(next(iter(cores.values()))).forward = _dispatch_forward

PERTURB = {'on': False}


def _attn_out_hook(mod, inp, out):
    if PERTURB['on'] and getattr(mod, '_g4_flash_saved', False):
        return out + out.detach().norm() / out.numel() ** 0.5 * 0.005 * torch.randn_like(out)
    return out


for c in cores.values():
    c.register_forward_hook(_attn_out_hook)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


CAP = {}


def _cap_fwd(mod, inp, out):
    if TRUTH['on'] and getattr(mod, '_cap_id', None) is not None:
        CAP[mod._cap_id] = {'q': inp[0].detach(), 'k': inp[1].detach(), 'v': inp[2].detach(), 'mask': inp[3]}


def _cap_bwd(mod, gin, gout):
    if TRUTH['on'] and getattr(mod, '_cap_id', None) is not None and mod._cap_id in CAP:
        CAP[mod._cap_id]['dout'] = gout[0].detach()


for L, c in cores.items():
    c._cap_id = L
    c.register_forward_hook(_cap_fwd)
    c.register_full_backward_hook(_cap_bwd)

if args.truth:
    torch.manual_seed(0)
    TRUTH['on'] = True
    loss_t, gin_t, wg_t, _ = run(False)
    TRUTH['on'] = False
    # per-layer, in-model inputs: how far is each kernel's own fwd/bwd from the fp32 truth on the SAME inputs?
    print('per-layer kernel error on real in-model inputs (vs fp32 truth on identical q,k,v,dout): layer type | unfused out dq dk dv | new out dq dk dv')
    for L in sorted(CAP):
        c = cores[L]; cap = CAP[L]
        if 'dout' not in cap:
            continue
        res = {}
        for mode in ('truth', 'unfused', 'new'):
            q, k, v = (t.clone().requires_grad_() for t in (cap['q'], cap['k'], cap['v']))
            TRUTH['on'] = mode == 'truth'
            c.g4_flash = c._g4_flash_saved if mode == 'new' else False
            c.g4_d512 = c._g4_d512_saved if mode == 'new' else False
            o = c(q, k, v, cap['mask'])
            o.backward(cap['dout'])
            res[mode] = (o.detach().float(), q.grad.float(), k.grad.float(), v.grad.float())
            TRUTH['on'] = False
        t = 'sliding' if c._g4_flash_saved else 'global '
        eu = [rel(a, b) for a, b in zip(res['unfused'], res['truth'])]; en = [rel(a, b) for a, b in zip(res['new'], res['truth'])]
        if L % 6 == 5 or L in (0, 1, 58, 59):
            print(f'  {L:3d} {t} | ' + ' '.join(f'{e:.1e}' for e in eu) + ' | ' + ' '.join(f'{e:.1e}' for e in en))
    for c in cores.values():
        c.g4_flash = False; c.g4_d512 = False
    torch.manual_seed(0)
    loss_u, gin_u, wg_u, _ = run(False)
    torch.manual_seed(0)
    loss_n, gin_n, wg_n, _ = run(True)
    print(f'loss: fp32-truth {loss_t:.6f}  unfused {loss_u:.6f} (rel {abs(loss_u-loss_t)/abs(loss_t):.2e})  new kernels {loss_n:.6f} (rel {abs(loss_n-loss_t)/abs(loss_t):.2e})')
    import math as _m
    def agg(wg):
        num = sum(float((wg[L].flatten() @ wg_t[L].flatten())) for L in wg); na = _m.sqrt(sum(float(wg[L].norm() ** 2) for L in wg)); nt = _m.sqrt(sum(float(wg_t[L].norm() ** 2) for L in wg))
        nd = _m.sqrt(sum(float((wg[L] - wg_t[L]).norm() ** 2) for L in wg))
        return num / (na * nt), nd / nt
    cu, ru = agg(wg_u); cn, rn = agg(wg_n)
    print(f'o_proj.W gradient over all 60 layers vs fp32 truth:  unfused cos {cu:.4f} rel {ru:.3f}  |  new kernels cos {cn:.4f} rel {rn:.3f}')
    print('layer  type    unfused-vs-truth: dW rel  cos   |  new-vs-truth: dW rel  cos   |  closer')
    better = 0
    for L in sorted(cores):
        t = 'sliding' if cores[L]._g4_flash_saved else 'global '
        eu, cu_ = rel(wg_u[L], wg_t[L]), cos(wg_u[L], wg_t[L]); en, cn_ = rel(wg_n[L], wg_t[L]), cos(wg_n[L], wg_t[L])
        better += en <= eu
        print(f'{L:3d}   {t}  {eu:9.2e} {cu_:7.4f}   |  {en:9.2e} {cn_:7.4f}   |  {"new" if en <= eu else "unfused"}')
    print(f'new kernels closer to fp32 truth in {better}/60 layers')
    raise SystemExit(0)

torch.manual_seed(0)
loss_u, gin_u, wg_u, lin_u = run(False)
torch.manual_seed(0)
if args.control:
    JITTER['on'] = True
    loss_n, gin_n, wg_n, lin_n = run(False)
    JITTER['on'] = False
    label = 'unfused+1ulp-jitter (control)'
elif args.control_attn:
    PERTURB['on'] = True
    loss_n, gin_n, wg_n, lin_n = run(False)
    PERTURB['on'] = False
    label = 'unfused+0.5%-attn-noise (control)'
else:
    loss_n, gin_n, wg_n, lin_n = run(True)
    label = 'new kernels'
print(f'loss unfused {loss_u:.6f}  {label} {loss_n:.6f}  rel diff {abs(loss_n - loss_u) / abs(loss_u):.2e}')
print(f'layer  type    d(layer_in) rel  cos     d(attn_out) rel  cos     d(o_proj.W) rel  cos    |W_b|/|W_a|   [{label}]')
for L in sorted(cores):
    t = 'sliding' if cores[L]._g4_flash_saved else 'global '
    li = (f'{rel(lin_n[L], lin_u[L]):9.2e} {cos(lin_n[L], lin_u[L]):7.4f}' if L in lin_n and L in lin_u else '      n/a        ')
    print(f'{L:3d}   {t}  {li}  {rel(gin_n[L], gin_u[L]):9.2e} {cos(gin_n[L], gin_u[L]):7.4f}  {rel(wg_n[L], wg_u[L]):9.2e} '
          f'{cos(wg_n[L], wg_u[L]):7.4f}  {(wg_n[L].norm() / wg_u[L].norm()).item():7.3f}')
