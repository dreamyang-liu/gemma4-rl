"""Re-tokenize Kimi K3 trajectories (proxy logs) with the Gemma 4 chat template + tokenizer and compare with the
token counts Kimi reported. Per call: prompt (render(messages, tools, gen) vs usage.prompt_tokens) and completion
(rendered assistant turn delta vs usage.completion_tokens). Only successful attempts.
usage: python -I kimi_to_gemma_tokens.py ROOT"""
import glob, json, os, sys
import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

root = sys.argv[1]
mdir = os.path.join(root, 'models/gemma-4-31b-it-qat-w4a16-ct')
tok = Tokenizer.from_file(os.path.join(mdir, 'tokenizer.json'))
tpl = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
    open(os.path.join(mdir, 'chat_template.jinja')).read())
THRESHOLD = 14336


def n(msgs, tools, gen):
    kw = {'tools': tools} if tools else {}
    return len(tok.encode(tpl.render(messages=msgs, bos_token='<bos>', enable_thinking=True,
                                     add_generation_prompt=gen, **kw), add_special_tokens=False).ids)


def clean(m):
    m = {k: v for k, v in m.items() if k in ('role', 'content', 'tool_calls', 'tool_call_id', 'reasoning_content', 'name')}
    for tc in m.get('tool_calls') or []:
        a = tc['function'].get('arguments')
        if isinstance(a, str):
            try:
                tc['function']['arguments'] = json.loads(a)
            except Exception:
                pass
    return m


succ = []
for d in glob.glob(os.path.join(root, 'runs/k_*')):
    f = os.path.join(d, 'results.jsonl')
    if os.path.exists(f) and any(json.loads(l)['resolved'] for l in open(f)):
        succ.append(os.path.basename(d))
P_k, P_g, C_k, C_g, R_k, R_g, per_traj = [], [], [], [], [], [], []
for w in sorted(succ):
    rows = [json.loads(l) for l in open(os.path.join(root, 'runs/kimi_logs', w + '.jsonl'))]
    tk = tg = 0
    over_k = over_g = 0
    for r in rows:
        if r['status'] != 200:
            continue
        req, u = r['request'], r['chat_response']['usage']
        tools = req.get('tools')
        msgs = [clean(m) for m in req['messages']]
        resp = clean(r['chat_response']['choices'][0]['message'])
        pg = n(msgs, tools, True)
        cg = n(msgs + [resp], tools, False) - pg
        P_k.append(u['prompt_tokens']); P_g.append(pg); C_k.append(u['completion_tokens']); C_g.append(cg)
        rk = (u.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
        if resp.get('reasoning_content') and rk:
            R_k.append(rk); R_g.append(len(tok.encode(resp['reasoning_content'], add_special_tokens=False).ids))
        tk += u['prompt_tokens'] + u['completion_tokens']; tg += pg + cg
        over_k += u['prompt_tokens'] >= THRESHOLD; over_g += pg >= THRESHOLD
    per_traj.append((w, len(rows), tk, tg, over_k, over_g))

P_k, P_g, C_k, C_g, R_k, R_g = map(np.array, (P_k, P_g, C_k, C_g, R_k, R_g))
q = lambda v: ' '.join(f'{x:6.3f}' for x in np.percentile(v, [10, 50, 90]))
print(f'{len(succ)} successful Kimi trajectories, {len(P_k)} calls')
print(f'prompt     gemma/kimi ratio: total {P_g.sum() / P_k.sum():.3f}   per-call p10/p50/p90 {q(P_g / P_k)}')
print(f'completion gemma/kimi ratio: total {C_g.sum() / C_k.sum():.3f}   per-call p10/p50/p90 {q(C_g / np.maximum(C_k, 1))}')
print(f'reasoning  gemma/kimi ratio: total {R_g.sum() / R_k.sum():.3f}   per-call p10/p50/p90 {q(R_g / R_k)}')
print(f'max prompt per call: kimi {P_k.max()}  gemma {P_g.max()};  median kimi {np.median(P_k):.0f} gemma {np.median(P_g):.0f}')
print(f'completion per call median: kimi {np.median(C_k):.0f} gemma {np.median(C_g):.0f};  >4096 gemma tokens: {(C_g > 4096).sum()} calls')
print(f'\n{"trajectory":<24}{"calls":>6}{"kimi_tok":>10}{"gemma_tok":>10}{"ratio":>7}{"kimi>=14336":>12}{"gemma>=14336":>13}')
for w, c, tk, tg, ok, og in per_traj:
    print(f'{w:<24}{c:>6}{tk:>10}{tg:>10}{tg / tk:>7.3f}{ok:>12}{og:>13}')
