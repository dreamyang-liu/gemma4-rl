"""Bedrock vs vLLM token accounting for the same requests (Gemma 4 31B, budget_v2 sampling logs).

For a sample of logged calls: prompt tokens as vLLM would count them (Gemma chat template with tools and
enable_thinking, Gemma tokenizer) vs Bedrock's usage.prompt_tokens; and how many responses have more thinking than
the 4096-token budget vLLM enforces on Kaggle (Bedrock has no budget control).
usage: python -I bedrock_vs_vllm_tokens.py ROOT [N_FILES]"""
import glob
import json
import os
import random
import sys

import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

root = sys.argv[1]
n_files = int(sys.argv[2]) if len(sys.argv) > 2 else 24
mdir = os.path.join(root, 'models/gemma-4-31b-it-qat-w4a16-ct')
tok = Tokenizer.from_file(os.path.join(mdir, 'tokenizer.json'))
tpl = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
    open(os.path.join(mdir, 'chat_template.jinja')).read())


def ntok(text):
    return len(tok.encode(text, add_special_tokens=False).ids)


def render(msgs, tools):
    kw = {'tools': tools} if tools else {}
    return tpl.render(messages=msgs, bos_token='<bos>', enable_thinking=True, add_generation_prompt=True, **kw)


def clean(m):
    m = {k: v for k, v in m.items() if k in ('role', 'content', 'tool_calls', 'tool_call_id', 'reasoning', 'name')}
    for tc in m.get('tool_calls') or []:
        a = tc['function'].get('arguments')
        if isinstance(a, str):
            try:
                tc['function']['arguments'] = json.loads(a)
            except Exception:
                pass
    return m


files = sorted(glob.glob(os.path.join(root, 'runs/bedrock_logs/bs_r[0-3]_s*.jsonl')))
random.Random(0).shuffle(files)
P_b, P_v, R_tok, out_tok, length_cut = [], [], [], [], 0
over_b = over_v = 0
for f in files[:n_files]:
    for line in open(f):
        r = json.loads(line)
        if r['status'] != 200 or not r['request'].get('tools'):
            continue
        req, u = r['request'], r['chat_response']['usage']
        msgs = []
        for m in req['messages']:
            m = dict(m)
            m['reasoning'] = m.get('reasoning') or m.get('reasoning_content')  # what vLLM's template reads
            msgs.append(clean(m))
        pv = ntok(render(msgs, req['tools']))
        P_b.append(u['prompt_tokens']); P_v.append(pv)
        over_b += u['prompt_tokens'] >= 14336; over_v += pv >= 14336
        msg = r['chat_response']['choices'][0]['message']
        R_tok.append(ntok(msg.get('reasoning_content') or ''))
        out_tok.append(u['completion_tokens'])
        length_cut += r['chat_response']['choices'][0]['finish_reason'] == 'length'
P_b, P_v, R_tok, out_tok = map(np.array, (P_b, P_v, R_tok, out_tok))
ratio = P_v / np.maximum(P_b, 1)
print(f'{len(P_b)} calls from {min(n_files, len(files))} worker logs')
print(f'prompt tokens vLLM-render / Bedrock: total {P_v.sum() / P_b.sum():.3f}; per call p10/p50/p90 '
      f'{np.percentile(ratio, 10):.3f} / {np.median(ratio):.3f} / {np.percentile(ratio, 90):.3f}; '
      f'median diff {np.median(P_v - P_b):+.0f} tokens')
print(f'calls at/over the 14336 compaction threshold: Bedrock {over_b}, vLLM-render {over_v}')
print(f'thinking tokens per call: p50 {np.median(R_tok):.0f}, p90 {np.percentile(R_tok, 90):.0f}, max {R_tok.max()}; '
      f'> 4096 (cut by the Kaggle thinking budget): {(R_tok > 4096).sum()} calls ({(R_tok > 4096).mean():.1%})')
print(f'responses that hit max_output_tokens 8192: {length_cut} ({length_cut / len(P_b):.1%})')
