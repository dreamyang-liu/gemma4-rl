"""Parity of the TITO proxy with a real vLLM 0.19.1 server, independent of sampling.

1. `collect`: send collection-log requests (every kind: first turn, later turns, after a nudge, after compaction,
   summarizer calls, plus copies with a small thinking_token_budget) to the real vLLM server with
   return_token_ids, and store prompt ids, output ids and the response.
2. Start the proxy with --fake-outputs on the stored output ids: its engine replays vLLM's sampled tokens.
3. `compare`: send the same requests to the proxy. Prompt ids, output ids and the whole response must match,
   except the random response id, `created` and the random tool-call ids.

usage: parity_test.py collect --logs 'GLOB' --vllm URL --out CASES.jsonl [--n 80]
       parity_test.py compare --cases CASES.jsonl --proxy URL
"""
import argparse
import concurrent.futures as cf
import glob
import json
import random
import re
import sys

import httpx

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['collect', 'compare'])
ap.add_argument('--logs')
ap.add_argument('--vllm')
ap.add_argument('--proxy')
ap.add_argument('--cases')
ap.add_argument('--out')
ap.add_argument('--n', type=int, default=80)
ap.add_argument('--model', default='gemma-4-31b-it')
ap.add_argument('--budgets', default='16,64,300', help='extra copies with these thinking_token_budget values')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--seq', default='', help='collect: take the first --n records of this one worker log, in order')
ap.add_argument('--budget', type=int, default=None, help='collect --seq: set this thinking_token_budget on every request')
args = ap.parse_args()


def kind(req):
    msgs = req['messages']
    if not req.get('tools'):
        return 'summarizer'
    if any(m['role'] == 'user' and not isinstance(m.get('content'), str) for m in msgs):
        return 'compacted'
    if any(m['role'] == 'user' for m in msgs[2:]):
        return 'after_nudge'
    return 'first' if len(msgs) == 2 else 'later'


def pick():
    by_kind = {}
    rng = random.Random(0)
    files = sorted(glob.glob(args.logs))
    rng.shuffle(files)
    for f in files[:30]:
        for line in open(f):
            r = json.loads(line)
            if r.get('status') == 200:
                by_kind.setdefault(kind(r['request']), []).append(r['request'])
    per = max(1, args.n // len(by_kind))
    reqs = []
    for k, v in sorted(by_kind.items()):
        rng.shuffle(v)
        reqs += [(k, r) for r in v[:per]]
    for b in [int(x) for x in args.budgets.split(',') if x]:
        for k, r in reqs[:: max(1, len(reqs) // 6)][:6]:
            reqs.append((f'{k}+budget{b}', {**r, 'thinking_token_budget': b}))
    return reqs


def post(url, body):
    r = httpx.post(url, json=body, timeout=1800)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {'raw': r.text[:500]}


def pick_seq():
    reqs = []
    for line in open(args.seq):
        r = json.loads(line)
        if r.get('status') == 200:
            req = r['request'] if args.budget is None else {**r['request'], 'thinking_token_budget': args.budget}
            reqs.append((kind(req), req))
        if len(reqs) >= args.n:
            break
    return reqs


def collect():
    reqs = pick_seq() if args.seq else pick()
    print(f'{len(reqs)} requests:', {k: sum(1 for kk, _ in reqs if kk == k) for k in sorted({k for k, _ in reqs})})

    def one(item):
        k, req = item
        body = {**req, 'model': args.model, 'return_token_ids': True}
        code, d = post(args.vllm.rstrip('/') + '/v1/chat/completions', body)
        return {'kind': k, 'request': {**req, 'model': args.model}, 'status': code, 'response': d,
                'prompt_ids': d.get('prompt_token_ids') if code == 200 else None,
                'output_ids': d['choices'][0].get('token_ids') if code == 200 else None}

    with cf.ThreadPoolExecutor(args.workers) as ex, open(args.out, 'w') as f, \
            open(args.out + '.fake.jsonl', 'w') as fake:
        for i, c in enumerate(ex.map(one, reqs)):
            f.write(json.dumps(c) + '\n')
            if c['status'] == 200:
                fake.write(json.dumps({'prompt_ids': c['prompt_ids'], 'output_ids': c['output_ids'],
                                       'thinking_token_budget': c['request'].get('thinking_token_budget')}) + '\n')
            print(i, c['kind'], c['status'], len(c['output_ids'] or []), flush=True)


ID = re.compile(r'chatcmpl-tool-[0-9a-f]+')


def norm(d):
    d = json.loads(ID.sub('TOOLID', json.dumps(d)))
    for k in ('id', 'created'):
        d.pop(k, None)
    return d


def compare():
    cases = [json.loads(l) for l in open(args.cases)]
    bad = 0

    def one(c):
        body = {**c['request'], 'return_token_ids': True}
        return c, post(args.proxy.rstrip('/') + '/v1/chat/completions', body)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for c, (code, d) in ex.map(one, cases):
            if c['status'] != 200:
                ok = code == c['status']
                diff = '' if ok else f'status {code} vs {c["status"]}'
            else:
                a, b = norm(c['response']), norm(d) if code == 200 else d
                ok = code == 200 and a == b
                diff = ''
                if not ok:
                    if code != 200:
                        diff = f'proxy HTTP {code}: {json.dumps(d)[:300]}'
                    else:
                        for k in sorted(set(a) | set(b)):
                            if a.get(k) != b.get(k):
                                diff += f'\n    {k}: vllm={json.dumps(a.get(k))[:400]}\n    {k}: prxy={json.dumps(b.get(k))[:400]}'
            bad += not ok
            print(f'{"ok  " if ok else "DIFF"} {c["kind"]:24s} out={len(c["output_ids"] or [])}{diff}', flush=True)
    print(f'{len(cases) - bad}/{len(cases)} identical')
    sys.exit(1 if bad else 0)


collect() if args.mode == 'collect' else compare()
