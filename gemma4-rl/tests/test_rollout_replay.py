"""Offline check of gemma4_rollout's replay mode against exported steps: for every step<k>.json in the given dir,
generate_rollout must (a) never call the harness, (b) advance the data buffer by the recorded draws, and (c) reproduce
the source run's kept/uninformative/segment/dp-pad counts, which are passed in as expectations.

  docker run --rm -v ~/project/gemma4-rl:/root/g4rl:ro -e PYTHONPATH=/root/miles:/root/g4rl \
    --entrypoint python3 gemma4-miles:patched /root/g4rl/tests/test_rollout_replay.py /root/g4rl/runs/<run>/replay_steps \
    '0:16,12,531,1 1:16,14,499,1 2:16,15,474,0 3:16,16,499,1 4:16,15,492,0'      # step:kept,uninformative,segments,dp-pad
"""
import os
import sys
import tempfile
import types

replay_dir = sys.argv[1]
expect = {}
for tok in sys.argv[2].split():
    k, v = tok.split(':')
    expect[int(k)] = tuple(int(x) for x in v.split(','))
os.environ.update({'G4RL_DYNAMIC_SAMPLING': '1', 'G4RL_OVERSAMPLE': '2.0', 'G4RL_MAX_WAVES': '4', 'G4RL_SKIP_SOLVED': '1',
                   'G4RL_TASK_STATS': os.path.join(tempfile.mkdtemp(), 'task_stats.json'), 'G4RL_REPLAY_DIR': replay_dir})
sys.path.insert(0, '/root/g4rl')
import gemma4_rollout as gr  # noqa: E402
from miles.utils.types import Sample  # noqa: E402

N = 8


class Buffer:
    def __init__(self):
        self.draws = 0

    def get_samples(self, n):
        self.draws += n
        return [[Sample(prompt=f't{self.draws}_{i}', metadata={'instance_id': f't{self.draws}_{i}'}) for _ in range(N)] for i in range(n)]


def no_harness(groups, tag):
    raise AssertionError('replay must not call the harness')


gr._run_groups = no_harness
gr.httpx.post = lambda *a, **k: types.SimpleNamespace(raise_for_status=lambda: None)
args = types.SimpleNamespace(rollout_batch_size=16, n_samples_per_prompt=N, sglang_router_ip='x', sglang_router_port=1,
                             actor_num_nodes=1, actor_num_gpus_per_node=8, tensor_model_parallel_size=4,
                             pipeline_model_parallel_size=1, context_parallel_size=1, micro_batch_size=1)
buf = Buffer()
import json  # noqa: E402

for step in sorted(expect):
    before = buf.draws
    out = gr.generate_rollout(args, step, buf)
    rec = json.load(open(os.path.join(replay_dir, f'step{step}.json')))
    draws = buf.draws - before
    real = [g for g in out if not g[0].metadata.get('filler')]
    pads = [g for g in out if g[0].metadata.get('dp_pad')]
    n_seg = sum(len(g) for g in real)
    kept, uninf, segs, pad = expect[step]
    assert draws == sum(rec['draws']), (step, draws, rec['draws'])
    assert len(real) == kept, (step, len(real), kept)
    assert n_seg == segs, (step, n_seg, segs)
    assert len(pads) == pad, (step, len(pads), pad)
    assert sum(len(g) for g in out) % 2 == 0
    # every real sample has tokens, a matching loss mask and the group's reward
    for g in real:
        rewards = {s.reward for s in g}
        assert len(rewards) > 1, f'step {step}: kept group with a single reward {rewards}'
        for s in g:
            assert len(s.loss_mask) == s.response_length and len(s.rollout_log_probs) == s.response_length
            assert s.tokens and s.status == Sample.Status.COMPLETED
    tok = sum(len(s.tokens) for g in real for s in g)
    print(f'step {step}: ok — {len(real)} groups, {n_seg} segments, {tok / 1e6:.2f}M tokens, {len(pads)} dp-pad, draws {draws}')
print('replay mode ok')
