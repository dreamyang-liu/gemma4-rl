"""Export the rollouts of a finished-or-dead run as replay steps, so a fresh run can train on them without re-sampling.

For every step s of the source run: the groups of every wave in order (instance_id + the 8 agent-server-style results,
rebuilt from the TITO proxy log and the harness results.jsonl exactly like agent_server.py does) and the number of
data-buffer draws the step consumed (groups run + solved-task draws skipped, from the train.log summary line).
gemma4_rollout.generate_rollout replays runs/<dst>/replay_steps/step<s>.json when G4RL_REPLAY_DIR is set.

usage: python3 tools/export_replay.py <src run dir> <dst run dir> [--steps 0-4]
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tito_samples  # noqa: E402

G4A = os.environ.get('G4A') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gemma4-agent')

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('dst')
ap.add_argument('--harness-runs', default=os.path.join(G4A, 'runs'))
ap.add_argument('--steps', default='', help='comma-separated step indices to export (default: all)')
args = ap.parse_args()
only = {int(x) for x in args.steps.split(',') if x.strip()}

# per-step skipped-draw counts from the source train.log summary lines
skipped = {}
for line in open(os.path.join(args.src, 'train.log'), errors='replace'):
    m = re.search(r'rollout (\d+): (\d+) waves, (\d+) groups run, .*?(\d+) solved-task draws skipped', line)
    if m:
        skipped[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))

steps = collections.defaultdict(lambda: collections.defaultdict(dict))  # s -> w -> g -> {k: (tid, result)}
for log in sorted(glob.glob(os.path.join(args.src, 'proxy_logs', '*.jsonl'))):
    name = os.path.splitext(os.path.basename(log))[0]
    m = re.search(r'_s(\d+)w(\d+)_g(\d+)_k(\d+)$', name)
    if not m:
        continue
    s, w, g, k = (int(x) for x in m.groups())
    if only and s not in only:
        continue
    rf = os.path.join(args.harness_runs, name, 'results.jsonl')
    rows = [json.loads(l) for l in open(rf)] if os.path.exists(rf) else []
    calls = tito_samples.load_calls(log)
    # The live agent server read the proxy log right after the harness ended (results.jsonl mtime). A generation
    # still running on the engine at that moment (the harness killed its trajectory at the session budget) lands
    # in the log later and was never part of the live sample; drop such records so the replay matches the live run.
    if os.path.exists(rf):
        end = os.path.getmtime(rf) + 2.0
        recs = [json.loads(l) for l in open(log)]
        late = {id(r) for r in recs if r.get('status') == 200 and r.get('ts', 0) + r.get('dt', 0) > end}
        if late:
            keep = [r['tito'] for r in recs if r.get('status') == 200 and id(r) not in late]
            n_late = len(late)
            calls = keep if len(keep) <= len(calls) else calls
            print(f'  {name}: dropped {n_late} record(s) that completed after the harness ended', file=sys.stderr)
    if not rows or not calls:
        res = {'status': 'failed', 'error': 'no result row or no model calls', 'reward': None, 'segments': [], 'metrics': {}}
        tid = rows[-1]['id'] if rows else None
    else:
        row = rows[-1]
        segs = tito_samples.segments_merged(calls)
        res = {'status': 'ok', 'reward': 1.0 if row.get('resolved') else 0.0, 'resolved': bool(row.get('resolved')),
               'metrics': {'tool_calls': row.get('tool_calls'), 'agent_s': row.get('duration_s'), 'error': row.get('error'),
                           'patch_chars': row.get('patch_chars'), 'segments': len(segs), 'replayed_from': name,
                           'compactions': sum(1 for c in calls if c.get('summarizer'))},
               'segments': [{kk: sg[kk] for kk in ('tokens', 'response_length', 'loss_mask', 'rollout_log_probs')} for sg in segs]}
        tid = row['id']
    steps[s][w].setdefault(g, {})[k] = (tid, res)

out_dir = os.path.join(args.dst, 'replay_steps')
os.makedirs(out_dir, exist_ok=True)
for s in sorted(steps):
    n_waves, groups_run, n_skipped = skipped.get(s, (len(steps[s]), sum(len(gs) for gs in steps[s].values()), 0))
    waves = []
    for w in sorted(steps[s]):
        wave = []
        for g in sorted(steps[s][w]):
            ks = steps[s][w][g]
            tids = {t for t, _ in ks.values() if t}
            assert len(tids) == 1, f'step {s} wave {w} group {g}: tasks {tids}'
            wave.append({'instance_id': tids.pop(), 'results': [ks[k][1] for k in sorted(ks)]})
        waves.append(wave)
    # the skipped draws all happened before the first wave's groups were drawn, in one wave they belong to that wave
    draws = [len(waves[0]) + n_skipped] + [len(wv) for wv in waves[1:]]
    rec = {'source': args.src, 'step': s, 'draws': draws, 'waves': waves}
    path = os.path.join(out_dir, f'step{s}.json')
    with open(path, 'w') as f:
        json.dump(rec, f)
    n_traj = sum(len(gr['results']) for wv in waves for gr in wv)
    n_seg = sum(len(r['segments']) for wv in waves for gr in wv for r in gr['results'])
    n_res = sum(1 for wv in waves for gr in wv for r in gr['results'] if r.get('resolved'))
    print(f'step {s}: {len(waves)} waves, {sum(len(wv) for wv in waves)} groups, {n_traj} trajectories ({n_res} resolved), '
          f'{n_seg} segments, draws {draws} -> {path} ({os.path.getsize(path) // 2**20} MB)')
