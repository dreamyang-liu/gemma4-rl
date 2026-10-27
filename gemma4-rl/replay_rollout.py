"""Miles rollout function that replays saved trajectories (for --debug-train-only experiments).

Reads G4RL_REPLAY (JSON: [{g, k, reward, segments: [{tokens, response_length, loss_mask, rollout_log_probs}]}]),
built from TITO proxy logs, and returns the same Samples as gemma4_rollout.generate_rollout would: one Sample per
segment, one rollout_id per trajectory, one group_index per task group. Every call returns the same data, so two runs
that differ in one setting (e.g. the attention backend) train on identical inputs.
"""
import copy
import json
import os

from miles.utils.types import Sample

_DATA = None


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    global _DATA
    assert not evaluation
    if _DATA is None:
        _DATA = json.load(open(os.environ['G4RL_REPLAY']))
    base = data_buffer.get_samples(1)[0][0]
    groups, index = {}, 0
    for t in _DATA:
        for j, seg in enumerate(t['segments']):
            s = copy.copy(base)
            s.metadata = {'replay': True, 'g': t['g'], 'k': t['k'], 'segment': j}
            s.index = index
            index += 1
            s.group_index = rollout_id * 1_000 + t['g']
            s.rollout_id = rollout_id * 1_000_000 + t['g'] * 1_000 + t['k']
            s.tokens = seg['tokens']
            s.response_length = seg['response_length']
            s.loss_mask = seg['loss_mask']
            s.rollout_log_probs = seg['rollout_log_probs']
            s.reward = t['reward']
            s.status = Sample.Status.COMPLETED
            groups.setdefault(t['g'], []).append(s)
    return [groups[g] for g in sorted(groups)]
