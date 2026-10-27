"""Render adk-eval-core ATIF traces to self-contained, offline HTML pages.

  python env/trace_html.py runs/qwen_s12/results/traces/trace_rich_4079.json -o trajectories/html
  python env/trace_html.py 'runs/qwen_s*' 'runs/out8k_s*' -o trajectories/html     # many runs + index.html

Run dirs are scanned for results/traces/*.json; results.jsonl / submission.parquet next to them supply
the verdict and the final agent patch. --tasks adds the problem's gold patch for comparison.
Compaction is not recorded in traces; it is inferred from prompt_tokens drops (> --drop tokens).
"""

import argparse
import glob
import html
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('inputs', nargs='+', help='trace .json files, run dirs, or globs of either')
ap.add_argument('-o', '--out', default='trajectories/html')
ap.add_argument('--tasks', default=None, help='tasks.jsonl for gold patches (default: data/tasks.jsonl if present)')
ap.add_argument('--drop', type=int, default=3000, help='prompt-token drop treated as a compaction')
ap.add_argument('--threshold', type=int, default=14336, help='compaction token_threshold to draw')
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent.parent
tasks_path = Path(args.tasks) if args.tasks else ROOT / 'data' / 'tasks.jsonl'
GOLD = {}
if tasks_path.exists():
    for line in open(tasks_path):
        t = json.loads(line)
        GOLD[t['instance_id']] = t


def unq(x):
    while isinstance(x, str):
        try:
            x = json.loads(x)
        except (ValueError, TypeError):
            break
    return x


ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07')
OUTSIDE = re.compile(r'/wheels|\.whl\b|site-packages|dist-packages|pip (show|download)|/kaggle/|swegemma_sandbox_|/usr/local/lib/python|/usr/lib/python')


def esc(s):
    return html.escape(ANSI.sub('', str(s)), quote=True)


# ---------- collect traces ----------
traces = []  # (run_name, trace_path, run_dir)
for pat in args.inputs:
    for p in sorted(glob.glob(pat)) or [pat]:
        p = Path(p)
        if p.is_file() and p.suffix == '.json':
            run_dir = p.parent.parent.parent
            traces.append((run_dir.name, p, run_dir))
        elif p.is_dir():
            for f in sorted((p / 'results' / 'traces').glob('trace_*.json')):
                traces.append((p.name, f, p))


def run_meta(run_dir):
    res, patches = {}, {}
    rj = run_dir / 'results.jsonl'
    if rj.exists():
        for x in map(json.loads, open(rj)):
            res[x['id']] = x
    pq = run_dir / 'submission.parquet'
    if pq.exists():
        try:
            import pandas as pd
            for r in pd.read_parquet(pq).itertuples():
                patches[r.id] = r.prediction
        except Exception:
            pass
    return res, patches


_meta_cache = {}

# ---------- styling ----------
CSS = r"""
:root{color-scheme:light;
 --page:#f9f9f7;--surface:#fcfcfb;--surface-2:#f3f2ee;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series:#2a78d6;
 --ok:#006300;--ok-bg:#e6f2e6;--bad:#b42318;--bad-bg:#fbeaea;--warn:#8a5a00;--warn-bg:#fdf3dc;
 --think:#6250d6;--think-bg:#f1effc;--tool:#2a78d6;--tool-bg:#eaf2fc;--add:#e6f4ea;--del:#fdecec;
 --code:#f5f4f0;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--surface-2:#222220;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series:#3987e5;
 --ok:#0ca30c;--ok-bg:#132413;--bad:#f07070;--bad-bg:#2c1616;--warn:#e0a82e;--warn-bg:#2a2210;
 --think:#9085e9;--think-bg:#1f1c33;--tool:#3987e5;--tool-bg:#14202e;--add:#132a18;--del:#2e1515;--code:#141413;}}
:root[data-theme="dark"]{color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--surface-2:#222220;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series:#3987e5;
 --ok:#0ca30c;--ok-bg:#132413;--bad:#f07070;--bad-bg:#2c1616;--warn:#e0a82e;--warn-bg:#2a2210;
 --think:#9085e9;--think-bg:#1f1c33;--tool:#3987e5;--tool-bg:#14202e;--add:#132a18;--del:#2e1515;--code:#141413;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--series)}
code,pre,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
.layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:20px;max-width:1500px;margin:0 auto;padding:20px}
@media (max-width:900px){.layout{grid-template-columns:1fr}.side{position:static!important;max-height:none!important}}
.side{position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);overflow:auto;
 background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px}
.side h4{margin:4px 6px 8px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.nav a{display:flex;gap:6px;align-items:baseline;padding:3px 6px;border-radius:6px;color:var(--ink-2);
 text-decoration:none;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nav a:hover{background:var(--surface-2);color:var(--ink)}
.nav .t{color:var(--muted);font-variant-numeric:tabular-nums;min-width:38px;text-align:right}
.nav .comp{color:var(--warn);font-weight:600}
.nav .err{color:var(--bad)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 18px;margin-bottom:14px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--ink-2);margin:0 0 12px}
.badge{display:inline-flex;align-items:center;gap:6px;font-weight:600;border-radius:999px;padding:2px 10px;font-size:13px}
.badge.ok{color:var(--ok);background:var(--ok-bg)}.badge.bad{color:var(--bad);background:var(--bad-bg)}
.badge.neutral{color:var(--ink-2);background:var(--surface-2)}
.stats{display:flex;flex-wrap:wrap;gap:22px;margin-top:12px}
.stat .v{font-size:20px;font-weight:600}.stat .k{color:var(--muted);font-size:12px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 14px}
.toolbar button,.toolbar input{font:inherit;font-size:13px;color:var(--ink);background:var(--surface);
 border:1px solid var(--border);border-radius:8px;padding:5px 10px;cursor:pointer}
.toolbar input{cursor:text;min-width:220px}
.toolbar button:hover{background:var(--surface-2)}
.step{position:relative;margin:0 0 10px 18px;padding-left:18px;border-left:2px solid var(--grid)}
.step::before{content:"";position:absolute;left:-7px;top:16px;width:12px;height:12px;border-radius:50%;
 background:var(--surface);border:2px solid var(--axis)}
.step.think::before{border-color:var(--think)}.step.tool::before{border-color:var(--tool)}
.step.final::before{border-color:var(--ok);background:var(--ok)}
.step .box{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 14px}
.meta{display:flex;flex-wrap:wrap;gap:10px;color:var(--muted);font-size:12px;margin-bottom:6px;font-variant-numeric:tabular-nums}
.meta b{color:var(--ink-2);font-weight:600}
.thinking{border-left:3px solid var(--think);background:var(--think-bg);border-radius:6px;padding:6px 10px;margin:6px 0}
.thinking summary{cursor:pointer;color:var(--think);font-weight:600;font-size:12.5px}
.thinking .body{white-space:pre-wrap;word-break:break-word;margin-top:6px;color:var(--ink)}
.prose{white-space:pre-wrap;word-break:break-word}
.call{border:1px solid var(--border);border-radius:8px;margin:8px 0;overflow:hidden}
.call .hd{display:flex;gap:8px;align-items:center;padding:6px 10px;background:var(--tool-bg);font-size:13px}
.call .hd .fn{font-weight:700;color:var(--tool)}.call .hd .n{color:var(--muted);font-size:12px}
.call .hd .st{margin-left:auto}
pre{margin:0;padding:8px 10px;background:var(--code);white-space:pre-wrap;word-break:break-word;max-height:none}
pre.clip{max-height:340px;overflow:auto}
.obs{border-top:1px dashed var(--border)}
.obs>summary{cursor:pointer;padding:5px 10px;font-size:12.5px;color:var(--ink-2);list-style:none}
.obs>summary::-webkit-details-marker{display:none}
.obs>summary::before{content:"▸ ";color:var(--muted)}.obs[open]>summary::before{content:"▾ "}
.chip{white-space:nowrap;font-size:11.5px;font-weight:600;border-radius:999px;padding:1px 8px}
.chip.ok{color:var(--ok);background:var(--ok-bg)}.chip.bad{color:var(--bad);background:var(--bad-bg)}
.chip.warn{color:var(--warn);background:var(--warn-bg)}
.lbl{color:var(--muted);font-size:11.5px;padding:4px 10px 0;background:var(--code)}
.diff .a{background:var(--add);display:block}.diff .d{background:var(--del);display:block}
.diff .h{color:var(--muted);display:block}
.compaction{margin:14px 0 14px 18px;padding:10px 14px;border:1px dashed var(--warn);background:var(--warn-bg);
 color:var(--warn);border-radius:10px;font-weight:600}
.compaction .why{font-weight:400;color:var(--ink-2);font-size:12.5px}
mark{background:#ffe58a;color:#000}
.chart-wrap{position:relative}
.chart-wrap svg{display:block;width:100%;height:auto}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);
 border-radius:8px;padding:6px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;white-space:nowrap}
.tip .v{font-weight:700;font-size:13px}
.tip .k{color:var(--ink-2)}
table.data{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
table.data td,table.data th{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:left}
details.plain>summary{cursor:pointer;color:var(--ink-2);font-size:13px}
.hidden{display:none!important}
.idx td a{font-weight:600}
"""

JS = r"""
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
function setAll(sel,open){$$(sel).forEach(d=>d.open=open)}
function toggleTheme(){const r=document.documentElement;const cur=r.dataset.theme||
 (matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');r.dataset.theme=cur==='dark'?'light':'dark'}
function search(q){q=q.trim().toLowerCase();$$('.step').forEach(s=>{s.classList.toggle('hidden',!!q&&!s.textContent.toLowerCase().includes(q))})}
function chart(){
 const el=$('#chart'); if(!el) return; const D=JSON.parse($('#chart-data').textContent);
 const pts=D.points; if(!pts.length){el.textContent='no LLM calls with token counts';return}
 const W=900,H=220,m={l:52,r:16,t:14,b:30};
 const xmax=Math.max(...pts.map(p=>p.t),1), ymax=Math.max(D.threshold*1.15,...pts.map(p=>p.y))*1.05;
 const X=t=>m.l+(W-m.l-m.r)*t/xmax, Y=v=>H-m.b-(H-m.t-m.b)*v/ymax;
 const NS='http://www.w3.org/2000/svg', svg=document.createElementNS(NS,'svg');
 svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('role','img');
 svg.setAttribute('aria-label','Prompt tokens per LLM call over elapsed time');
 const add=(tag,at,txt)=>{const e=document.createElementNS(NS,tag);for(const k in at)e.setAttribute(k,at[k]);if(txt!=null)e.textContent=txt;svg.appendChild(e);return e};
 const step=ymax>30000?10000:ymax>12000?4000:2000;
 for(let v=0;v<=ymax;v+=step){add('line',{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:'var(--grid)','stroke-width':1});
  add('text',{x:m.l-8,y:Y(v)+4,'text-anchor':'end','font-size':11,fill:'var(--muted)'},v>=1000?(v/1000)+'k':v)}
 const xs=xmax>600?120:xmax>240?60:30;
 for(let t=0;t<=xmax;t+=xs){add('text',{x:X(t),y:H-m.b+16,'text-anchor':'middle','font-size':11,fill:'var(--muted)'},t+'s')}
 add('line',{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:'var(--axis)','stroke-width':1});
 add('line',{x1:m.l,x2:W-m.r,y1:Y(D.threshold),y2:Y(D.threshold),stroke:'var(--muted)','stroke-width':1.5,'stroke-dasharray':'5 4'});
 add('text',{x:m.l+6,y:Y(D.threshold)-6,'text-anchor':'start','font-size':11,fill:'var(--ink-2)'},'compaction threshold '+D.threshold.toLocaleString());
 D.compactions.forEach((c,i)=>{add('line',{x1:X(c.t),x2:X(c.t),y1:m.t,y2:H-m.b,stroke:'var(--warn)','stroke-width':1,'stroke-dasharray':'2 3'});
  add('text',{x:X(c.t)+4,y:m.t+10,'font-size':11,fill:'var(--warn)','font-weight':600},'C'+(i+1))});
 add('path',{d:pts.map((p,i)=>(i?'L':'M')+X(p.t).toFixed(1)+','+Y(p.y).toFixed(1)).join(''),fill:'none',
  stroke:'var(--series)','stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'});
 pts.forEach(p=>add('circle',{cx:X(p.t),cy:Y(p.y),r:3,fill:'var(--series)',stroke:'var(--surface)','stroke-width':2}));
 const hair=add('line',{y1:m.t,y2:H-m.b,stroke:'var(--axis)','stroke-width':1,visibility:'hidden'});
 const dot=add('circle',{r:5,fill:'var(--series)',stroke:'var(--surface)','stroke-width':2,visibility:'hidden'});
 const hit=add('rect',{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:'transparent',style:'cursor:pointer'});
 el.appendChild(svg); const tip=$('#tip'); let cur=null;
 hit.addEventListener('pointermove',e=>{const r=svg.getBoundingClientRect();const sx=(e.clientX-r.left)*W/r.width;
  let best=pts[0];for(const p of pts)if(Math.abs(X(p.t)-sx)<Math.abs(X(best.t)-sx))best=p;cur=best;
  hair.setAttribute('x1',X(best.t));hair.setAttribute('x2',X(best.t));hair.setAttribute('visibility','visible');
  dot.setAttribute('cx',X(best.t));dot.setAttribute('cy',Y(best.y));dot.setAttribute('visibility','visible');
  tip.replaceChildren();const v=document.createElement('div');v.className='v';v.textContent=best.y.toLocaleString()+' prompt tokens';
  const k=document.createElement('div');k.className='k';k.textContent=`step ${best.step} · t=${best.t.toFixed(0)}s · ${best.label}`;
  tip.append(v,k);tip.style.display='block';
  const px=(X(best.t)/W)*r.width, py=(Y(best.y)/H)*r.height;
  tip.style.left=Math.min(px+12,r.width-tip.offsetWidth-4)+'px';tip.style.top=Math.max(py-48,0)+'px'});
 hit.addEventListener('pointerleave',()=>{tip.style.display='none';hair.setAttribute('visibility','hidden');dot.setAttribute('visibility','hidden')});
 hit.addEventListener('click',()=>{if(cur)location.hash='s'+cur.step});
}
document.addEventListener('DOMContentLoaded',chart);
"""


# ---------- rendering helpers ----------
def pre(text, clip=True, cls=''):
    return f'<pre class="{"clip " if clip else ""}{cls}">{esc(text)}</pre>'


def diff_block(text):
    out = []
    for ln in str(text).splitlines():
        c = 'a' if ln.startswith('+') and not ln.startswith('+++') else \
            'd' if ln.startswith('-') and not ln.startswith('---') else \
            'h' if ln.startswith(('@@', '---', '+++', 'diff ', 'index ')) else ''
        out.append(f'<span class="{c}">{esc(ln) or " "}</span>' if c else esc(ln))
    return '<pre class="clip diff">' + '\n'.join(out) + '</pre>'


def render_args(fn, a):
    if not isinstance(a, dict):
        return pre(a)
    parts = []
    if 'command' in a:
        parts.append('<div class="lbl">$ command</div>' + pre(a['command']))
    for k in ('filepath', 'path', 'query', 'node', 'nodes', 'start_line', 'end_line'):
        if k in a:
            parts.append(f'<div class="lbl">{esc(k)}: <span class="mono">{esc(json.dumps(a[k], ensure_ascii=False))}</span></div>')
    if fn == 'edit_file' and ('old_string' in a or 'new_string' in a):
        old = ''.join('-' + l + '\n' for l in str(a.get('old_string', '')).splitlines())
        new = ''.join('+' + l + '\n' for l in str(a.get('new_string', '')).splitlines())
        parts.append('<div class="lbl">edit</div>' + diff_block(old + new))
    if 'content' in a:
        parts.append('<div class="lbl">content</div>' + pre(a['content']))
    rest = {k: v for k, v in a.items() if k not in ('command', 'filepath', 'path', 'query', 'node', 'nodes', 'start_line',
                                                    'end_line', 'old_string', 'new_string', 'content')}
    if rest:
        parts.append('<div class="lbl">args</div>' + pre(json.dumps(rest, ensure_ascii=False, indent=1)))
    return ''.join(parts) or '<div class="lbl">(no arguments)</div>'


def obs_parts(o):
    """-> (status_chip_html, summary_text, body_html, n_chars)"""
    o = unq(o)
    if isinstance(o, dict) and 'content' in o:
        o = unq(o['content'])
    if not isinstance(o, dict):
        s = str(o)
        return '', s[:90], pre(s), len(s)
    st = o.get('status')
    chip = f'<span class="chip {"ok" if st == "ok" else "bad"}">{"✓ ok" if st == "ok" else "✗ " + esc(st or "?")}</span>'
    if 'exit_code' in o and o['exit_code'] not in (0, None):
        chip += f' <span class="chip warn">exit {esc(o["exit_code"])}</span>'
    if o.get('budget_warning'):
        chip += ' <span class="chip warn">budget warning</span>'
    body, total = [], 0
    for k in ('error_type', 'error_message', 'details'):
        if o.get(k):
            body.append(f'<div class="lbl">{k}</div>' + pre(o[k] if isinstance(o[k], str) else json.dumps(o[k], ensure_ascii=False, indent=1)))
            total += len(str(o[k]))
    for k in ('stdout', 'stderr', 'content', 'results'):
        v = o.get(k)
        if v:
            v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1)
            body.append(f'<div class="lbl">{k}' + (f' · lines {o.get("start_line")}–{o.get("end_line")} of {o.get("total_lines")}'
                                                  if k == 'content' and o.get('total_lines') else '') + '</div>' + pre(v))
            total += len(v)
    if o.get('diff'):
        body.append('<div class="lbl">diff</div>' + diff_block(o['diff']))
        total += len(o['diff'])
    if o.get('budget_warning'):
        body.append('<div class="lbl">budget_warning</div>' + pre(o['budget_warning']))
    rest = {k: v for k, v in o.items() if k not in ('status', 'stdout', 'stderr', 'content', 'results', 'diff', 'exit_code',
                                                    'error_type', 'error_message', 'details', 'budget_warning')}
    if rest:
        body.append(pre(json.dumps(rest, ensure_ascii=False, indent=1)))
    first = (o.get('error_message') or o.get('stdout') or o.get('content') or o.get('diff') or json.dumps(rest)[:200] or '')
    first = ' '.join(str(first).split())[:110]
    return chip, first, ''.join(body), total


ICON = {'run_command': '⌘', 'read_file': '📄', 'edit_file': '✏️', 'write_file': '📝', 'submit_patch': '📤',
        'get_status': '⏱', 'search_similar_code': '🔎', 'get_code_neighbors': '🕸', 'get_code_subgraph': '🕸'}


def render(run_name, trace_path, run_dir):
    if run_dir not in _meta_cache:
        _meta_cache[run_dir] = run_meta(run_dir)
    res, patches = _meta_cache[run_dir]
    t = json.load(open(trace_path))
    iid = trace_path.stem.removeprefix('trace_')
    r = res.get(iid, {})
    steps = t['steps']
    model = next((s.get('model_name') for s in steps if s.get('model_name')), '?')
    if '/' in model:  # local vLLM path like .../other/gemma-4-31b-it-qat-w4a16-ct/2
        model = next((p for p in reversed(model.split('/')) if p and not p.isdigit()), model)
    task = GOLD.get(iid, {})

    # pair observations with tool calls (FIFO per tool name, falling back to FIFO overall)
    pending, obs_for = [], {}
    calls = []  # (step_id, n, tool_call)
    for s in steps:
        for tc in unq(s.get('tool_calls')) or []:
            calls.append((s['step_id'], len(calls) + 1, tc))
            pending.append(len(calls))
        if s.get('observation') and pending:
            o = unq(s['observation'])
            name = ((o.get('extra') or {}).get('tool_name')) if isinstance(o, dict) else None
            idx = next((i for i in pending if calls[i - 1][2].get('function_name') == name), pending[0])
            pending.remove(idx)
            obs_for[idx] = s['observation']

    body, nav, points, comps = [], [], [], []
    prev_pt, n_think, n_outside = 0, 0, 0
    call_by_step = {}
    for sid, n, tc in calls:
        call_by_step.setdefault(sid, []).append((n, tc))
    for s in steps:
        ex = s.get('extra') or {}
        ev, el = ex.get('event_type'), ex.get('elapsed_s') or 0
        m = s.get('metrics') or {}
        pt = m.get('prompt_tokens')
        if pt and prev_pt and pt < prev_pt - args.drop:
            comps.append({'t': el, 'from': prev_pt, 'to': pt})
            k = len(comps)
            body.append(f'<div class="compaction" id="c{k}">🗜️ Compaction #{k} — prompt {prev_pt:,} → {pt:,} tokens'
                        f'<div class="why">Older events were replaced by an LLM-written summary (ADK token-threshold compaction; '
                        f'last 5 raw events kept). The summary text itself is not stored in the trace.</div></div>')
            nav.append(f'<a href="#c{k}" class="comp"><span class="t">{el:.0f}s</span>🗜 compaction #{k}</a>')
        if pt:
            prev_pt = pt
        my_calls = call_by_step.get(s['step_id'], [])
        msg = unq(s.get('message')) if s.get('message') else ''
        # skip pure observation carrier steps (rendered inside their call)
        if s['source'] == 'system' and not msg and not my_calls:
            continue
        if s['source'] == 'agent' and not msg and not my_calls:
            continue
        cls = 'final' if ev == 'final' else ('tool' if my_calls else 'think' if msg else '')
        meta = [f'<b>step {s["step_id"]}</b>', esc(s['source'])]
        if ev:
            meta.append(esc(ev))
        if el:
            meta.append(f't={el:.1f}s')
        if pt:
            meta.append(f'prompt {pt:,} · completion {m.get("completion_tokens", 0):,}')
            label = ', '.join(tc.get('function_name', '?') for _, tc in my_calls) or ev or 'message'
            points.append({'t': el, 'y': pt, 'step': s['step_id'], 'label': label})
        h = [f'<div class="step {cls}" id="s{s["step_id"]}"><div class="box"><div class="meta">{" · ".join(meta)}</div>']
        if msg:
            text = msg if isinstance(msg, str) else json.dumps(msg, ensure_ascii=False, indent=1)
            if s['source'] in ('system', 'user') or ev in ('final', 'text', 'system_instruction', 'task_prompt'):
                if ev == 'system_instruction':
                    h.append(f'<details class="plain"><summary>System instruction ({len(text):,} chars)</summary>{pre(text, clip=False)}</details>')
                else:
                    h.append(f'<div class="prose">{esc(text)}</div>')
            else:
                n_think += 1
                prev = ' '.join(text.split())[:140]
                h.append(f'<details class="thinking"><summary>💭 reasoning · {len(text):,} chars — {esc(prev)}{"…" if len(text) > 140 else ""}</summary>'
                         f'<div class="body">{esc(text)}</div></details>')
        for n, tc in my_calls:
            fn = tc.get('function_name', '?')
            a = unq(tc.get('arguments'))
            o = obs_for.get(n)
            chip, first, obody, nchars = obs_parts(o) if o is not None else ('<span class="chip warn">no result</span>', '', '', 0)
            outside = bool(OUTSIDE.search(json.dumps(a, ensure_ascii=False)))
            flag = '<span class="chip warn">⚠ outside /workspace</span> ' if outside else ''
            h.append(f'<div class="call"><div class="hd"><span>{ICON.get(fn, "🔧")}</span><span class="fn">{esc(fn)}</span>'
                     f'<span class="n">tool call #{n}</span><span class="st">{flag}{chip}</span></div>{render_args(fn, a)}')
            if o is not None:
                h.append(f'<details class="obs"{" open" if nchars < 1200 else ""}><summary>result · {nchars:,} chars — '
                         f'<span class="mono">{esc(first)}</span></summary>{obody}</details>')
            h.append('</div>')
            short = a.get('command') if isinstance(a, dict) and 'command' in a else \
                a.get('filepath') if isinstance(a, dict) and 'filepath' in a else ''
            bad = ' err' if 'chip bad' in chip else ''
            if outside:
                n_outside += 1
            nav.append(f'<a href="#s{s["step_id"]}" class="{bad}" title="{esc(short)}"><span class="t">{el:.0f}s</span>'
                       f'{"⚠" if outside else ""}{ICON.get(fn, "🔧")} {esc(fn)} <span style="color:var(--muted)">{esc(" ".join(str(short).split())[:40])}</span></a>')
        if ev == 'final':
            nav.append(f'<a href="#s{s["step_id"]}"><span class="t">{el:.0f}s</span>🏁 final message</a>')
        h.append('</div></div>')
        body.append(''.join(h))

    resolved = r.get('resolved')
    badge = ('<span class="badge ok">✓ Resolved</span>' if resolved else
             '<span class="badge bad">✗ Not resolved</span>' if resolved is False else '<span class="badge neutral">• result pending</span>')
    fm = t.get('final_metrics') or {}
    stats = [('model', model), ('wall time', f'{r["duration_s"]:.0f}s' if r.get('duration_s') else f'{(steps[-1].get("extra") or {}).get("elapsed_s", 0):.0f}s'),
             ('tool calls', len(calls)), ('LLM calls', len(points)), ('compactions', len(comps)),
             ('calls outside /workspace', n_outside), ('peak prompt', f'{max([p["y"] for p in points] or [0]):,}'), ('total tokens', f'{fm.get("total_tokens", 0):,}')]
    if r:
        stats.append(('test exit code', r.get('exit_code')))
    stat_html = ''.join(f'<div class="stat"><div class="v">{esc(v)}</div><div class="k">{esc(k)}</div></div>' for k, v in stats)
    problem = task.get('problem_statement')
    if not problem:
        tp = next((unq(s.get('message')) for s in steps if (s.get('extra') or {}).get('event_type') == 'task_prompt'), '')
        problem = str(tp)
    extras = []
    if iid in patches and patches[iid] and patches[iid] != 'NO_PATCH':
        extras.append(f'<details class="plain"><summary>Agent patch (submitted)</summary>{diff_block(patches[iid])}</details>')
    if task.get('patch'):
        extras.append(f'<details class="plain"><summary>Gold patch (reference)</summary>{diff_block(task["patch"])}</details>')
    if task.get('test_patch'):
        extras.append(f'<details class="plain"><summary>Test patch (hidden from agent)</summary>{diff_block(task["test_patch"])}</details>')
    rows = ''.join(f'<tr><td>{p["step"]}</td><td>{p["t"]:.0f}s</td><td>{p["y"]:,}</td><td>{esc(p["label"])}</td></tr>' for p in points)
    chart_data = json.dumps({'points': points, 'compactions': comps, 'threshold': args.threshold}).replace('</', '<\\/')

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(iid)} · {esc(run_name)}</title>
<style>{CSS}</style></head><body>
<div class="layout"><aside class="side"><h4>Timeline</h4><div class="nav">
<a href="#top">⬆ overview</a>{''.join(nav)}</div></aside><main id="top">
<div class="card"><div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><h1>{esc(iid)}</h1>{badge}
<span class="sub" style="margin:0">{esc(task.get('repo', ''))} · run <span class="mono">{esc(run_name)}</span></span></div>
<div class="stats">{stat_html}</div></div>
<div class="card"><h4 style="margin:0 0 6px">Problem statement</h4><div class="prose">{esc(problem)}</div>
{''.join(extras)}</div>
<div class="card"><h4 style="margin:0 0 2px">Prompt tokens per LLM call</h4>
<p class="sub" style="font-size:12.5px">Context window 32,768. Dashed line = compaction threshold; C1, C2… = inferred compactions. Hover for detail, click to jump.</p>
<div class="chart-wrap"><div id="chart"></div><div class="tip" id="tip"></div></div>
<details class="plain" style="margin-top:6px"><summary>Table view</summary><table class="data"><thead><tr><th>step</th><th>t</th><th>prompt tokens</th><th>tool</th></tr></thead><tbody>{rows}</tbody></table></details>
<script type="application/json" id="chart-data">{chart_data}</script></div>
<div class="toolbar"><input placeholder="Filter steps (text search)…" oninput="search(this.value)">
<button onclick="setAll('.thinking',true)">Expand reasoning</button><button onclick="setAll('.thinking',false)">Collapse reasoning</button>
<button onclick="setAll('.obs',true)">Expand results</button><button onclick="setAll('.obs',false)">Collapse results</button>
<button onclick="toggleTheme()">◐ Theme</button></div>
{''.join(body)}
</main></div><script>{JS}</script></body></html>"""
    return iid, r, page, model, len(calls), len(comps), n_outside


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
index = []
for run_name, tp, run_dir in traces:
    iid, r, page, model, ncalls, ncomp, nout = render(run_name, tp, run_dir)
    fn = f'{run_name}__{iid}.html'
    (out / fn).write_text(page, encoding='utf-8')
    index.append((run_name, iid, r, model, ncalls, ncomp, nout, fn))
    print('wrote', out / fn)

if len(index) > 1:
    rows = ''.join(
        f'<tr><td><a href="{esc(fn)}">{esc(iid)}</a></td><td class="mono">{esc(rn)}</td><td>{esc(model)}</td>'
        f'<td>{"<span class=\"chip ok\">✓ resolved</span>" if r.get("resolved") else "<span class=\"chip bad\">✗</span>" if r else "pending"}</td>'
        f'<td>{ncalls}</td><td>{ncomp}</td><td>{nout or ""}</td><td>{r.get("duration_s", "")}</td></tr>'
        for rn, iid, r, model, ncalls, ncomp, nout, fn in sorted(index, key=lambda x: (x[1], x[0])))
    (out / 'index.html').write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>Trajectories</title><style>{CSS}</style></head><body>'
        f'<div style="max-width:1100px;margin:20px auto;padding:0 20px"><div class="card"><h1>Trajectories ({len(index)})</h1>'
        f'<div class="toolbar"><input placeholder="Filter…" oninput="for(const tr of document.querySelectorAll(\'tbody tr\'))'
        f'tr.style.display=tr.textContent.toLowerCase().includes(this.value.toLowerCase())?\'\':\'none\'"></div>'
        f'<table class="data idx"><thead><tr><th>task</th><th>run</th><th>model</th><th>result</th><th>tool calls</th>'
        f'<th>compactions</th><th>⚠ outside /workspace</th><th>seconds</th></tr></thead><tbody>{rows}</tbody></table></div></div></body></html>',
        encoding='utf-8')
    print('wrote', out / 'index.html')
