"""Render an adk-eval-core ATIF trace (results/traces/trace_<id>.json) to readable markdown.

Compaction events are not stored in the trace; they are inferred from drops in prompt_tokens.
usage: python render_trace.py <trace.json> <out.md> [--max-obs N]
"""
import json, sys

src, out = sys.argv[1], sys.argv[2]
max_obs = int(sys.argv[sys.argv.index('--max-obs') + 1]) if '--max-obs' in sys.argv else 0
t = json.load(open(src))


def unq(x):
    if isinstance(x, str):
        try:
            return unq(json.loads(x))
        except (ValueError, TypeError):
            return x
    return x


def obs_text(o):
    o = unq(o)
    if isinstance(o, dict) and 'content' in o:
        o = unq(o['content'])
    if isinstance(o, dict):
        parts = [f"status={o.get('status')}" + (f" exit_code={o['exit_code']}" if 'exit_code' in o else '')]
        for k in ('stdout', 'stderr', 'content', 'diff', 'error_message', 'patch', 'message', 'result'):
            if o.get(k):
                parts.append(f'[{k}]\n{o[k]}')
        rest = {k: v for k, v in o.items() if k not in ('status', 'exit_code', 'stdout', 'stderr', 'content', 'diff',
                                                        'error_message', 'patch', 'message', 'result')}
        if rest:
            parts.append(json.dumps(rest, ensure_ascii=False))
        o = '\n'.join(parts)
    o = o if isinstance(o, str) else json.dumps(o, ensure_ascii=False, indent=1)
    if max_obs and len(o) > max_obs:
        o = o[:max_obs] + f'\n... [{len(o) - max_obs} more chars]'
    return o


def fence(s):
    s = str(s)
    return '````text\n' + s + '\n````\n'


L = [f'# Trajectory `{src}`\n', f"final_metrics: `{json.dumps(t.get('final_metrics'))}`\n"]
prev_pt, n_comp, n_tool = 0, 0, 0
for s in t['steps']:
    ex = s.get('extra') or {}
    ev, el = ex.get('event_type'), ex.get('elapsed_s')
    pt = (s.get('metrics') or {}).get('prompt_tokens')
    if pt and prev_pt and pt < prev_pt - 3000:
        n_comp += 1
        L.append(f'\n> ### 🗜️ COMPACTION #{n_comp} (inferred): prompt {prev_pt} → {pt} tokens. '
                 f'Earlier events were replaced by an LLM-written summary (summary text is not stored in the trace).\n')
    if pt:
        prev_pt = pt
    head = f"\n## step {s['step_id']} · {s['source']}" + (f' · {ev}' if ev else '') + (f' · t={el:.1f}s' if el else '')
    if pt:
        head += f" · prompt={pt} completion={s['metrics'].get('completion_tokens')}"
    L.append(head + '\n')
    if s.get('message'):
        label = {'thinking': '**model output / reasoning**', 'final': '**final message**'}.get(ev, '**message**')
        L.append(label + '\n' + fence(unq(s['message'])))
    for tc in unq(s.get('tool_calls')) or []:
        n_tool += 1
        args = tc.get('arguments')
        L.append(f"**tool call #{n_tool}: `{tc.get('function_name')}`**\n" +
                 fence(json.dumps(args, ensure_ascii=False, indent=1) if not isinstance(args, str) else args))
    if s.get('observation'):
        L.append('**observation**\n' + fence(obs_text(s['observation'])))
open(out, 'w').write('\n'.join(L))
print(f'wrote {out}: {len(t["steps"])} steps, {n_tool} tool calls, {n_comp} inferred compactions')
