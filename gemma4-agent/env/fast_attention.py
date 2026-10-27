"""Fast attention for Gemma 4 training (batch 1, no padding).

Sliding-window layers (head_dim 256, window 1024): flex_attention with a causal+window block mask, which skips
blocks outside the window (sdpa with an explicit mask computes the full LxL). Global layers (head_dim 512, no flash
kernel): sdpa with is_causal=True and no explicit mask (efficient kernel, ~2x faster than with a mask).
Register with `register()` and load the model with attn_implementation='gemma4_fast'.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_flex = torch.compile(flex_attention, dynamic=True)
_masks = {}


def _sliding_block_mask(L, window, device):
    key = (L, window, str(device))
    if key not in _masks:
        _masks[key] = create_block_mask(lambda b, h, q, k: (q >= k) & (q - k < window), None, None, L, L, device=device)
    return _masks[key]


def gemma4_fast_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, **kwargs):
    assert query.shape[0] == 1, 'gemma4_fast assumes batch size 1 without padding'
    L = query.shape[2]
    gqa = query.shape[1] != key.shape[1]
    if sliding_window:
        out = _flex(query, key, value, block_mask=_sliding_block_mask(L, sliding_window, query.device), scale=scaling,
                    enable_gqa=gqa)
    else:
        if gqa:  # the efficient kernel has no GQA support: expand KV heads
            rep = query.shape[1] // key.shape[1]
            key, value = key.repeat_interleave(rep, dim=1), value.repeat_interleave(rep, dim=1)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def register():
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    AttentionInterface.register('gemma4_fast', gemma4_fast_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register('gemma4_fast', lambda *a, **k: None)  # masks are implicit (causal/window)
