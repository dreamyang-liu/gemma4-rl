"""Structural diff of RL rollout logs against the collection logs (swegemma shell_v2 via the GLM converse proxy).

Checks that each RL request has what the collection requests have:
  - the same system prompt and tool declarations, byte for byte;
  - the same task prompt as the collection log for the same task (first user message, byte for byte);
  - request parameters with the same keys and values (two variants: agent calls and summarizer calls);
  - tool-result JSON shapes, mid-conversation user messages (nudges), the compaction message format and
    assistant-message key sets that also occur in the collection logs.
Prints every shape that occurs in the RL logs but never in the collection logs.

usage: format_diff.py --rl 'RL_GLOB' --ref 'COLLECTION_GLOB'
"""
import argparse
import collections
import glob
import json

ap = argparse.ArgumentParser()
ap.add_argument('--rl', required=True)
ap.add_argument('--ref', required=True)
ap.add_argument('--ref-files', type=int, default=40)
args = ap.parse_args()


def shapes(path, out, task_prompts=None):
    for line in open(path):
        r = json.loads(line)
        if r.get('status') != 200:
            continue
        req = r['request']
        msgs = req['messages']
        summ = not req.get('tools')
        params = {k: v for k, v in req.items() if k not in ('messages', 'tools', 'model')}
        out['params'][('summarizer' if summ else 'agent', json.dumps(params, sort_keys=True))] += 1
        if summ:
            out['summarizer_prefix'][msgs[0]['content'][:200] if isinstance(msgs[0]['content'], str) else 'LIST'] += 1
            continue
        out['system'][msgs[0]['content']] += 1
        out['tools'][json.dumps(req['tools'], sort_keys=True)] += 1
        calls = {}
        for i, m in enumerate(msgs):
            if m['role'] == 'assistant':
                out['assistant_keys'][tuple(sorted(k for k, v in m.items() if v is not None))] += 1
                for c in m.get('tool_calls') or []:
                    calls[c['id']] = c['function']['name']
            elif m['role'] == 'tool':
                try:
                    d = json.loads(m['content'])
                    key = (calls.get(m.get('tool_call_id'), '?'),) + (tuple(sorted(d)) if isinstance(d, dict) else
                                                                        (type(d).__name__,))
                except ValueError:
                    key = ('NON-JSON',)
                out['tool_shapes'][key] += 1
                out['tool_msg_keys'][tuple(sorted(m))] += 1
            elif m['role'] == 'user' and i == 1:
                c = m['content']
                if isinstance(c, str):
                    if task_prompts is not None:
                        task_prompts.setdefault(c.split('\n', 1)[0] + '|' + c[:400], c)
                    out['first_user'][('str', c.split('\n', 1)[0][:60])] += 1
                else:
                    out['first_user'][('parts', tuple(p.get('type') for p in c[:1]), c[0].get('text'),
                                       (c[1].get('text') or '')[:13] if len(c) > 1 else None)] += 1
            elif m['role'] == 'user':
                c = m['content']
                out['mid_user'][c if isinstance(c, str) else json.dumps(c)[:120]] += 1


def new():
    return collections.defaultdict(collections.Counter)


ref, rl = new(), new()
ref_prompts, rl_prompts = {}, {}
for f in sorted(glob.glob(args.ref))[:args.ref_files]:
    shapes(f, ref, ref_prompts)
rl_files = sorted(glob.glob(args.rl))
for f in rl_files:
    shapes(f, rl, rl_prompts)
print(f'RL files: {len(rl_files)}, requests: {sum(rl["params"].values())}')
bad = 0
for cat in ('system', 'tools', 'params', 'tool_shapes', 'tool_msg_keys', 'assistant_keys', 'first_user', 'mid_user',
            'summarizer_prefix'):
    unseen = [(k, v) for k, v in rl[cat].items() if k not in ref[cat]]
    status = 'ok ' if not unseen else 'NEW'
    print(f'{status} {cat}: {len(rl[cat])} distinct in RL, {len(unseen)} not in collection logs')
    for k, v in unseen[:6]:
        bad += 1
        print(f'      {v:5d} x {str(k)[:300]!r}')
same = sum(1 for k, v in rl_prompts.items() if ref_prompts.get(k) == v)
known = sum(1 for k in rl_prompts if k in ref_prompts)
print(f'task prompts: {len(rl_prompts)} in RL, {known} also in collection logs, {same} of those byte-identical')
