"""Render one RL trajectory (a TITO proxy log, one request per line) as a readable markdown transcript.

Shows what the model saw and produced, in order: system prompt and task once; then per request the new messages
since the previous request (tool results, nudges), the model's thinking, its tool call or text; compaction summarizer
calls (the history being compressed -> the summary); the harness result from the run dir's results.jsonl.

usage: python3 tools/render_trajectory.py runs/<run>/proxy_logs/<sample>.jsonl [out.md]
"""
import json
import os
import sys

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + '.md'
name = os.path.splitext(os.path.basename(src))[0]
recs = [json.loads(l) for l in open(src)]
recs = [r for r in recs if r.get('status') == 200]
res = None
G4A = os.environ.get('G4A') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gemma4-agent')
rf = os.path.join(G4A, 'runs', name, 'results.jsonl')
if os.path.exists(rf):
    res = json.loads(open(rf).readline())


def text(c, limit=None):
    if c is None:
        return ''
    if not isinstance(c, str):
        c = json.dumps(c, ensure_ascii=False, indent=1)
    if limit and len(c) > limit:
        return c[:limit] + f'\n… [{len(c) - limit} more chars]'
    return c


def fence(s, lang=''):
    s = s.replace('```', 'ʼʼʼ')
    return f'```{lang}\n{s}\n```'


lines = [f'# Trajectory `{name}`', '']
if res:
    lines += [f"task **{res['id']}** — resolved: **{res['resolved']}**, exit_code {res['exit_code']}, tool_calls {res['tool_calls']}, "
              f"duration {res['duration_s']:.0f}s, patch {res['patch_chars']} chars" + (f", error: {res['error']}" if res.get('error') else ''), '']
n_agent = sum(1 for r in recs if r['request'].get('tools'))
n_summ = len(recs) - n_agent
lines += [f'{len(recs)} model requests: {n_agent} agent turns, {n_summ} compaction summaries. '
          f'Generated tokens: {sum(len(r["tito"].get("output_ids", [])) for r in recs)}.', '']

first = recs[0]['request']['messages']
lines += ['## System prompt', fence(text(first[0]['content'])), '', '## Task (first user message)', fence(text(first[1]['content'])), '']

prev = None
turn = 0
for r in recs:
    req, t = r['request'], r['tito']
    msgs = req['messages']
    out_tokens = len(t.get('output_ids', []))
    choice = r['chat_response']['choices'][0]
    msg = choice['message']
    if not req.get('tools'):
        # compaction summarizer call: its prompt is the history to compress + the instruction; output is the summary
        lines += [f'## ⟲ Compaction summary request ({len(msgs)} messages in, {out_tokens} tokens out, {r.get("dt", 0):.1f}s)']
        instr = msgs[-1]['content'] if isinstance(msgs[-1].get('content'), str) else text(msgs[-1].get('content'))
        lines += ['summarization instruction (last message):', fence(text(instr, 1200)), '**summary produced (trained as its own sample):**',
                  fence(text(msg.get('content') or msg.get('reasoning') or '')), '']
        prev = None  # the next agent request has a compacted history
        continue
    turn += 1
    # new messages since the previous agent request
    if prev is not None and len(msgs) >= len(prev) and msgs[:len(prev)] == prev:
        new = msgs[len(prev):]
        note = ''
    else:
        new = msgs[2:] if prev is None and turn > 1 else msgs[2:] if turn == 1 else msgs
        note = ' — history was COMPACTED; the conversation now reads:' if turn > 1 else ''
        if turn == 1:
            new = msgs[2:]
    lines += [f'## Turn {turn}  (prompt {len(t.get("prompt_ids", []))} tokens, output {out_tokens} tokens, {r.get("dt", 0):.1f}s, finish={choice.get("finish_reason")}{", forced thinking close" if t.get("forced") else ""}){note}']
    for m in new:
        role = m.get('role')
        if role == 'assistant':
            if m.get('tool_calls'):
                for tc in m['tool_calls']:
                    lines += [f'*(earlier assistant tool call, as re-rendered)* `{tc["function"]["name"]}` {text(tc["function"].get("arguments"), 300)}']
            elif m.get('content'):
                lines += ['*(earlier assistant text, as re-rendered)*', fence(text(m['content'], 600))]
        elif role == 'tool':
            lines += ['**tool result:**', fence(text(m.get('content'), 2500), 'json')]
        elif role == 'user':
            lines += ['**user message (harness):**', fence(text(m.get('content'), 1500))]
        elif role == 'system':
            lines += ['**system message:**', fence(text(m.get('content'), 1500))]
    if msg.get('reasoning'):
        lines += ['**thinking:**', fence(text(msg['reasoning']))]
    if msg.get('content'):
        lines += ['**assistant text:**', fence(text(msg['content']))]
    for tc in msg.get('tool_calls') or []:
        fn = tc['function']
        try:
            args = json.loads(fn.get('arguments') or '{}')
            shown = args.get('command', json.dumps(args, ensure_ascii=False)) if isinstance(args, dict) else fn.get('arguments')
        except Exception:
            shown = fn.get('arguments')
        lines += [f'**tool call → `{fn["name"]}`**', fence(text(shown), 'bash' if fn['name'] == 'run_command' else '')]
    lines += ['']
    # the assistant message of this turn as the harness will store it, for the next delta
    prev = msgs
if res and res.get('patch'):
    lines += ['## Final patch (graded)', fence(res['patch'], 'diff'), '']
open(out, 'w').write('\n'.join(lines))
print(f'wrote {out} ({sum(len(l) for l in lines) // 1024} KB, {turn} turns, {n_summ} compactions)')
