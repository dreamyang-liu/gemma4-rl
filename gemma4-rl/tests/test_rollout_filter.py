"""Offline test of gemma4_rollout.generate_rollout: DAPO waves, fillers, DP padding and the solved-task skip, with the
agent server and the proxy replaced by synthetic results. Runs in the gemma4-miles image (needs miles.utils.types):

  docker run --rm -v ~/project/gemma4-rl:/root/g4rl:ro -e PYTHONPATH=/root/miles:/root/g4rl \
    --entrypoint python3 gemma4-miles:patched /root/g4rl/tests/test_rollout_filter.py
"""
import os
import sys
import tempfile
import types

STATS = os.path.join(tempfile.mkdtemp(), 'task_stats.json')
os.environ.update({'G4RL_DYNAMIC_SAMPLING': '1', 'G4RL_OVERSAMPLE': '1.6', 'G4RL_MAX_WAVES': '4',
                   'G4RL_SKIP_SOLVED': '1', 'G4RL_SOLVED_RETRY_P': '0.0', 'G4RL_TASK_STATS': STATS})
sys.path.insert(0, '/root/g4rl')
import gemma4_rollout as gr  # noqa: E402
from miles.utils.types import Sample  # noqa: E402

# Synthetic task pool: 'easy' tasks are always solved, 'hard' never, 'mixed' half the time.
TASKS = [f'easy_{i}' for i in range(8)] + [f'mixed_{i}' for i in range(8)] + [f'hard_{i}' for i in range(4)]
N = 4  # n_samples_per_prompt


class Buffer:
    def __init__(self):
        self.pos = 0
        self.draws = []

    def get_samples(self, n):
        out = []
        for _ in range(n):
            tid = TASKS[self.pos % len(TASKS)]
            self.pos += 1
            self.draws.append(tid)
            group = []
            for k in range(N):
                s = Sample(prompt=tid, metadata={'instance_id': tid})
                group.append(s)
            out.append(group)
        return out


def fake_run_groups(groups, tag):
    results = []
    for group in groups:
        tid = group[0].metadata['instance_id']
        res = []
        for k in range(len(group)):
            reward = 1.0 if tid.startswith('easy') else 0.0 if tid.startswith('hard') else float(k % 2)
            seg = {'tokens': [2] * (11 + k), 'response_length': 5, 'loss_mask': [1] * 5, 'rollout_log_probs': [0.0] * 5}
            res.append({'status': 'ok', 'reward': reward, 'segments': [seg] * (1 + k % 2), 'metrics': {}})
        results.append(res)
    return results


gr._run_groups = fake_run_groups
gr.httpx.post = lambda *a, **k: types.SimpleNamespace(raise_for_status=lambda: None)
args = types.SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=N, sglang_router_ip='x', sglang_router_port=1,
                             actor_num_nodes=1, actor_num_gpus_per_node=8, tensor_model_parallel_size=4,
                             pipeline_model_parallel_size=1, context_parallel_size=1, micro_batch_size=1)
buf = Buffer()

out0 = gr.generate_rollout(args, 0, buf)
n0 = sum(len(g) for g in out0)
real0 = [g for g in out0 if not g[0].metadata.get('filler')]
assert n0 % 2 == 0, f'sample count {n0} not DP-aligned'
assert len(real0) == 4, f'rollout 0 kept {len(real0)} groups, want 4'
assert all(g[0].metadata['instance_id'].startswith('mixed') for g in real0), 'kept groups must be the mixed tasks'
assert os.path.exists(STATS), 'task stats not saved'
stats = {k: v for k, v in gr._load_task_stats(STATS).items() if isinstance(v, dict)}
assert all(stats[t]['last_all_pass'] for t in stats if t.startswith('easy')), stats
assert not any(stats[t]['last_all_pass'] for t in stats if not t.startswith('easy')), stats
draws0 = len(buf.draws)

out1 = gr.generate_rollout(args, 1, buf)
real1 = [g for g in out1 if not g[0].metadata.get('filler')]
drawn1 = buf.draws[draws0:]
easy_drawn = [t for t in drawn1 if t.startswith('easy')]
assert len(real1) == 4
assert all(g[0].metadata['instance_id'].startswith('mixed') for g in real1)
# every easy task drawn in rollout 1 that was already flagged must have been skipped, not run
assert all(stats.get(t, {}).get('last_all_pass') for t in easy_drawn if t in stats), 'flagged easy task was run'
stats1 = {k: v for k, v in gr._load_task_stats(STATS).items() if isinstance(v, dict)}
flagged_before = {t for t in stats if stats[t]['last_all_pass']}
rerun = [t for t in flagged_before if stats1[t]['attempts'] > stats[t]['attempts']]
assert not rerun, f'flagged tasks were re-run with retry_p=0: {rerun}'
print(f'ok: rollout 0 -> {len(out0)} groups / {n0} samples; rollout 1 skipped {len([t for t in easy_drawn if t in stats])} '
      f'solved draws, kept {len(real1)} mixed groups; stats for {len(stats1)} tasks at {STATS}')
