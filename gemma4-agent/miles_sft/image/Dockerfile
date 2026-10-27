# radixark/miles + fix: Megatron-Bridge 0.7.0 Gemma4DenseSelfAttention.get_query_key_value_tensors lacks the
# `head_wise_gate` kwarg that Megatron-LM 0.19.0 SelfAttention.forward passes (Gemma 4 never enables it), and its
# positional super() calls bind split_qkv to the new head_wise_gate parameter.
FROM radixark/miles:latest
RUN set -e; f=/opt/sglang/lib/python3.12/site-packages/megatron/bridge/models/gemma/modeling_gemma4.py; \
    python3 - "$f" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = """        output_gate: bool = False,
        split_qkv: bool = True,
    ):
        if self.is_kv_shared_layer:"""
new = """        output_gate: bool = False,
        split_qkv: bool = True,
        head_wise_gate: bool = False,
    ):
        assert not head_wise_gate, "Gemma 4 dense has no head-wise attention gate"
        if self.is_kv_shared_layer:"""
assert s.count(old) == 1, "patch target not found"
s = s.replace(old, new)
# Megatron-LM 0.19 inserted head_wise_gate before split_qkv: pass the flags by keyword, not by position.
for a, b in (
    ("super().get_query_key_value_tensors(hidden_states, key_value_states, output_gate, split_qkv)",
     "super().get_query_key_value_tensors(hidden_states, key_value_states, output_gate=output_gate, split_qkv=split_qkv)"),
    ("super().get_query_key_value_tensors(hidden_states, key_value_states, False, True)",
     "super().get_query_key_value_tensors(hidden_states, key_value_states, output_gate=False, split_qkv=True)"),
):
    assert a in s, a
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched", p)
PY
