"""Scan agent trajectories for signs that the model recalls the upstream fix instead of deriving it.

For each model call (thinking + visible text), look for memory/upstream phrases and for PR numbers or commit hashes
that do not occur in the task prompt. Rates are per call and per trajectory; examples are printed for review.
usage: python3 analysis/leak_scan.py [ROOT] [--examples N]
"""
import glob
import json
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else '.'
n_ex = int(sys.argv[sys.argv.index('--examples') + 1]) if '--examples' in sys.argv else 4
os.chdir(root)
SOURCES = {
    'gemma bedrock adapted': 'runs/bedrock_logs_v/col_gv_*.jsonl',
    'gemma local vllm': 'runs/local_logs/lsh_*.jsonl',
    'glm-5.3 converse': 'runs/cand_logs/g4-glm53proxy/col_g53s_*.jsonl',
    'kimi-k2.5 converse': 'runs/cand_logs/g4-k25cproxy/col_k25s_*.jsonl',
    'glm-5 converse': 'runs/cand_logs/g4-glmcproxy/col_glmv_*.jsonl',
    'kimi-k3 (known recall)': 'runs/kimi_logs/k_*.jsonl',
}
MEMORY = re.compile(
    r"(?i)\b(upstream|I (recall|remember)|from memory|if I remember|as I remember|the (actual|real|official|original) "
    r"(upstream )?(fix|patch|implementation|PR|commit)|(fixed|was fixed|resolved) in (version|v?\d+\.\d+|PR|pull request|commit)|"
    r"in (later|newer|recent) versions|the (merged|upstream) (PR|pull request)|known fix|the maintainers (fixed|changed))\b")
PR = re.compile(r'(?:PR|pull request|pull/)\s*#?(\d{2,6})', re.I)
SHA = re.compile(r'\b[0-9a-f]{7,40}\b')


def attempts(pattern):
    for f in sorted(glob.glob(pattern)):
        cur, prompt = None, ''
        for line in open(f):
            r = json.loads(line)
            msgs = r['request'].get('messages', [])
            if len(msgs) == 2 and msgs[1]['role'] == 'user':
                if cur:
                    yield prompt, cur
                c = msgs[1]['content']
                prompt, cur = (c if isinstance(c, str) else json.dumps(c)), []
            if cur is not None and r.get('status') == 200 and r['request'].get('tools'):
                m = r['chat_response']['choices'][0]['message']
                cur.append((m.get('reasoning') or m.get('reasoning_content') or '') + '\n' + (m.get('content') or ''))
        if cur:
            yield prompt, cur


for name, pat in SOURCES.items():
    calls = hit_calls = trajs = hit_trajs = 0
    examples = []
    for prompt, texts in attempts(pat):
        trajs += 1
        prompt_prs = set(PR.findall(prompt)) | set(re.findall(r'#(\d{2,6})', prompt))
        prompt_shas = set(SHA.findall(prompt))
        hit_t = False
        for t in texts:
            calls += 1
            reasons = []
            m = MEMORY.search(t)
            if m:
                reasons.append(m.group(0))
            prs = [p for p in PR.findall(t) if p not in prompt_prs]
            if prs:
                reasons.append(f'PR {prs[0]} (not in prompt)')
            shas = [s for s in SHA.findall(t) if s not in prompt_shas and re.search(r'commit', t[max(0, t.find(s) - 40):t.find(s)], re.I)]
            if shas:
                reasons.append(f'commit {shas[0]}')
            if reasons:
                hit_calls += 1
                hit_t = True
                if len(examples) < n_ex:
                    i = t.find(reasons[0].split(' (')[0].split(' ')[-1]) if reasons else 0
                    examples.append((reasons[0], t[max(0, i - 160):i + 200].replace('\n', ' ')))
        hit_trajs += hit_t
    if not trajs:
        continue
    print(f'== {name}: {hit_calls}/{calls} calls ({hit_calls / max(1, calls):.1%}), {hit_trajs}/{trajs} trajectories ({hit_trajs / trajs:.1%})')
    for why, ctx in examples:
        print(f'   [{why}] ...{ctx}...')
