"""Single-layer probe: Megatron's DotProductAttention (as built by the Gemma 4 bridge spec) vs candidate kernels,
on the exact tensors the layer receives.

Builds one Gemma4 sliding layer's core attention from the real TransformerConfig of the bridge provider, feeds it
[s, b, h, d] q/k/v and the attention mask the Gemma4VLModel would build, and compares forward output and dq/dk/dv
with FlashAttention-3 called the way miles_patches.py calls it. Every intermediate (softmax scale, mask type, window,
head counts) is printed, so a mismatch can be located instead of guessed.

usage (gemma4-miles image, one GPU): python layer_probe.py --model /root/models/base --seq 4096
"""
import argparse
import math

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/root/models/base')
ap.add_argument('--seq', type=int, default=4096)
ap.add_argument('--layer', type=int, default=1, help='1-indexed layer number (layer 1 is sliding, 6 is global)')
ap.add_argument('--patched', action='store_true', help='apply /root/g4rl/miles_patches.py first (fixed unfused)')
args = ap.parse_args()
if args.patched:
    import runpy

    runpy.run_path('/root/g4rl/miles_patches.py')
dev, dt = 'cuda', torch.bfloat16

from megatron.bridge import AutoBridge  # noqa: E402
from megatron.core import parallel_state  # noqa: E402
from megatron.core.transformer.enums import AttnMaskType  # noqa: E402
from megatron.core.transformer.utils import is_layer_window_attention  # noqa: E402

torch.distributed.init_process_group('nccl', init_method='tcp://127.0.0.1:29511', world_size=1, rank=0)
parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
torch.cuda.set_device(0)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402

model_parallel_cuda_manual_seed(0)

bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=False)
provider.attention_backend = 'unfused'
provider.attention_softmax_in_fp32 = True
provider.bf16 = True
provider.params_dtype = dt
provider.finalize()
cfg = provider
print('config: window_size', cfg.window_size, 'skip_freq type', type(cfg.window_attn_skip_freq).__name__,
      'kv_channels', cfg.kv_channels, 'global_kv_channels', getattr(cfg, 'global_kv_channels', None),
      'heads', cfg.num_attention_heads, 'query_groups', cfg.num_query_groups,
      'global_query_groups', getattr(cfg, 'num_global_query_groups', None),
      'softmax_scale', cfg.softmax_scale, 'apply_qk_layer_scaling', cfg.apply_query_key_layer_scaling,
      'softmax_type', getattr(cfg, 'softmax_type', None), 'masked_softmax_fusion', cfg.masked_softmax_fusion)

from megatron.bridge.models.gemma.modeling_gemma4 import _is_gemma4_sliding_layer, get_gemma4_layer_spec  # noqa: E402

L = args.layer
sliding = _is_gemma4_sliding_layer(cfg, L)
print(f'layer {L}: sliding={sliding}, is_layer_window_attention={is_layer_window_attention(cfg.window_size, cfg.window_attn_skip_freq, L)}')
spec = get_gemma4_layer_spec(cfg)
attn_spec = spec.submodules.self_attention
# The attention module copies the config and sets softmax_scale = 1.0 and kv_channels for global layers; mirror that.
import copy  # noqa: E402

acfg = copy.copy(cfg)
acfg.softmax_scale = 1.0 if cfg.softmax_scale is None else cfg.softmax_scale
if not sliding:
    acfg.kv_channels = cfg.global_kv_channels
    acfg.num_query_groups = cfg.num_global_query_groups
core_cls = attn_spec.submodules.core_attention
core = core_cls(config=acfg, layer_number=L, attn_mask_type=AttnMaskType.causal, attention_type='self').to(dev)
print('core attention class:', type(core).__name__, 'softmax_scale used:', core.softmax_scale,
      'heads/partition', core.num_attention_heads_per_partition, 'groups/partition', core.num_query_groups_per_partition,
      'hidden/head', core.hidden_size_per_attention_head)
fsm = core.scale_mask_softmax
print('FusedScaleMaskSoftmax: window_size', getattr(fsm, 'window_size', None), 'mask type', fsm.attn_mask_type,
      'fusion', fsm.scaled_masked_softmax_fusion, 'softmax_in_fp32', fsm.softmax_in_fp32, 'scale', fsm.scale)

s, b = args.seq, 1
nh, ng, d = core.num_attention_heads_per_partition, core.num_query_groups_per_partition, core.hidden_size_per_attention_head
torch.manual_seed(0)
q = torch.randn(s, b, nh, d, device=dev, dtype=dt, requires_grad=True)
k = torch.randn(s, b, ng, d, device=dev, dtype=dt, requires_grad=True)
v = torch.randn(s, b, ng, d, device=dev, dtype=dt, requires_grad=True)
# the mask Gemma4VLModel._compute_attention_mask builds: True = blocked, [b, 1, s, s]
causal = torch.tril(torch.ones((b, 1, s, s), dtype=torch.bool, device=dev))
mask = ~causal
g = torch.randn(s, b, nh * d, device=dev, dtype=dt)

core.eval()
out_ref = core(q, k, v, mask)
out_ref.backward(g)
ref = (out_ref.detach().float(), q.grad.detach().float().clone(), k.grad.detach().float().clone(), v.grad.detach().float().clone())
for t in (q, k, v):
    t.grad = None

# fp64 reference built from the printed semantics: scale = core.softmax_scale, causal + window (ws[0] back, inclusive)
rep = nh // ng
q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
Q = q64.permute(1, 2, 0, 3); K = k64.permute(1, 2, 0, 3).repeat_interleave(rep, 1); V = v64.permute(1, 2, 0, 3).repeat_interleave(rep, 1)
sc = (Q @ K.transpose(-1, -2)) * core.softmax_scale
i = torch.arange(s, device=dev)
m = i[None, :] > i[:, None]
ws = getattr(fsm, 'window_size', None)
if ws:
    m |= (i[:, None] - i[None, :]) > ws[0]
sc = sc.masked_fill(m, float('-inf'))
o64 = (sc.softmax(-1) @ V).permute(2, 0, 1, 3).reshape(s, b, nh * d)
o64.backward(g.double())
ref64 = (o64.detach(), q64.grad.detach(), k64.grad.detach(), v64.grad.detach())


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


print('\nMegatron unfused vs fp64 model of the printed semantics (window edge: tokens up to ws[0] back):')
print('  out %.2e  dq %.2e  dk %.2e  dv %.2e' % tuple(rel(a, b) for a, b in zip(ref, ref64)))
if ws:
    # alternative edge: strictly fewer than ws[0] back (flash window_size=(ws[0]-1, 0))
    q64b, k64b, v64b = (t.detach().double().requires_grad_() for t in (q, k, v))
    Q = q64b.permute(1, 2, 0, 3); K = k64b.permute(1, 2, 0, 3).repeat_interleave(rep, 1); V = v64b.permute(1, 2, 0, 3).repeat_interleave(rep, 1)
    sc = (Q @ K.transpose(-1, -2)) * core.softmax_scale
    m2 = (i[None, :] > i[:, None]) | ((i[:, None] - i[None, :]) >= ws[0])
    o64b = (sc.masked_fill(m2, float('-inf')).softmax(-1) @ V).permute(2, 0, 1, 3).reshape(s, b, nh * d)
    print('  with edge ">= ws[0]" instead: out %.2e' % rel(ref[0], o64b.detach()))

from flash_attn_interface import flash_attn_func  # noqa: E402

for name, window in (('flash window=(ws0,0)', (ws[0], 0) if ws else (-1, -1)), ('flash window=(ws0-1,0)', (ws[0] - 1, 0) if ws else (-1, -1))):
    for t in (q, k, v):
        t.grad = None
    o = flash_attn_func(q.transpose(0, 1).contiguous(), k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous(),
                        softmax_scale=core.softmax_scale, causal=True, window_size=window)
    o = o[0] if isinstance(o, tuple) else o
    o = o.transpose(0, 1).reshape(s, b, nh * d)
    o.backward(g)
    got = (o.detach().float(), q.grad.detach().float().clone(), k.grad.detach().float().clone(), v.grad.detach().float().clone())
    print(f'{name}: vs unfused  out %.2e dq %.2e dk %.2e dv %.2e | vs fp64 out %.2e dq %.2e dk %.2e dv %.2e' % (
        *[rel(a, b) for a, b in zip(got, ref)], *[rel(a, b) for a, b in zip(got, ref64)]))
    if not ws:
        break

# ---- what does unfused actually compute? compare against fp64 variants of the semantics ----
def fp64_variant(win, invert_mask=False):
    q64, k64, v64 = (t.detach().double() for t in (q, k, v))
    Q = q64.permute(1, 2, 0, 3); K = k64.permute(1, 2, 0, 3).repeat_interleave(rep, 1); V = v64.permute(1, 2, 0, 3).repeat_interleave(rep, 1)
    sc = (Q @ K.transpose(-1, -2)) * core.softmax_scale
    mm = i[None, :] > i[:, None]
    if invert_mask:
        mm = ~mm
    if win is not None:
        mm = mm | ((i[:, None] - i[None, :]) > win)
    return (sc.masked_fill(mm, float('-inf')).softmax(-1) @ V).permute(2, 0, 1, 3).reshape(s, b, nh * d)

print('\nunfused output vs fp64 variants:')
for label, win, inv in (('causal only, no window', None, False), ('causal + window 1023', 1023, False),
                        ('causal + window 1024', 1024, False), ('inverted causal (future only)', None, True)):
    print(f'  {label:32s} rel err {rel(ref[0], fp64_variant(win, inv)):.2e}')
