"""Miles rollout function (--rollout-function-path gemma4_rollout.generate_rollout): GRPO groups of swegemma shell_v2
trajectories, one group per task.

Each prompt-data row is a task ({"prompt": instance_id, "metadata": {"instance_id": ...}}). For every sample slot of
a group the agent server (agent_server.py) runs one trajectory against the TITO proxy, which serves the current policy
from Miles' SGLang engines, and returns the trajectory's reward and one segment per model call. Each segment becomes
one Miles Sample:
  - tokens = the prompt that vLLM rendered for the call + the tokens the policy sampled;
  - loss_mask covers the sampled tokens only (tokens forced by the thinking budget are 0);
  - rollout_log_probs are the sampler's logprobs of the response tokens;
  - every segment of a trajectory shares its reward and a `rollout_id` that is unique to the trajectory, and keeps the
    group's `group_index`, so GRPO normalizes one reward per trajectory within the task group and broadcasts it
    (miles/ray/rollout/train_data_conversion.py, _normalize_rewards_by_rollout).
A trajectory that failed for infrastructure reasons (agent server status != "ok") is not scored 0. It is replaced by
one filler Sample in a group of its own: its advantage is exactly 0, so it adds no gradient. Miles counts
--global-batch-size in rollouts when Samples carry a rollout_id (utils/dp_schedule.py), and the filler keeps the count
at rollout_batch_size * n_samples_per_prompt, so each rollout step is exactly one optimizer step.

DAPO dynamic sampling: see generate_rollout (G4RL_DYNAMIC_SAMPLING, G4RL_OVERSAMPLE, G4RL_MAX_WAVES).

env: G4RL_AGENT_SERVER (default http://127.0.0.1:8400), G4RL_PROXY (the TITO proxy, default http://127.0.0.1:8300;
     each rollout tells it the router address that Miles picked), G4RL_RUN_TAG (prefix of the sample ids; default "s")
"""
import concurrent.futures as cf
import copy
import itertools
import logging
import math
import json
import os
import random

import httpx

from miles.utils.types import Sample

logger = logging.getLogger(__name__)
AGENT_SERVER = os.environ.get('G4RL_AGENT_SERVER', 'http://127.0.0.1:8400').rstrip('/')
RUN_TAG = os.environ.get('G4RL_RUN_TAG', 's')
PROXY = os.environ.get('G4RL_PROXY', 'http://127.0.0.1:8300').rstrip('/')
_index = itertools.count()


def _run(sample_id, instance_id):
    try:
        r = httpx.post(f'{AGENT_SERVER}/run', json={'sample_id': sample_id, 'instance_id': instance_id},
                       timeout=httpx.Timeout(None))
        if r.status_code != 200:
            return {'status': 'failed', 'error': f'agent server HTTP {r.status_code}: {r.text[:300]}'}
        return r.json()
    except httpx.HTTPError as e:
        return {'status': 'failed', 'error': repr(e)[:300]}


def _dp_align(args):
    """Miles' static micro-batch schedule (bshd, fixed micro_batch_size) needs the sample count of a step to be a
    multiple of dp_size * micro_batch_size; dp_size = actor GPUs / (TP * PP * CP). 1 in the 4+4 layout, 2 in colocate."""
    try:
        gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        dp = gpus // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size)
        return max(1, dp) * max(1, args.micro_batch_size or 1)
    except Exception:
        return 1


def _filler(base, rollout_id, g, k):
    """A stand-in for a dropped trajectory: one token, alone in its group, so its GRPO advantage is 0."""
    s = copy.copy(base)
    s.metadata = {**(base.metadata or {}), 'filler': True}
    s.index = next(_index)
    s.group_index = 900_000_000 + rollout_id * 1_000 + g * 100 + k  # a group of its own
    s.rollout_id = rollout_id * 1_000_000 + g * 1_000 + k
    s.tokens = [2, 2]
    s.response_length = 1
    s.loss_mask = [1]  # a zero mask would make the per-rollout token mean 0/0; the advantage is 0 anyway
    s.rollout_log_probs = [0.0]
    s.reward = 0.0
    s.status = Sample.Status.COMPLETED
    return s


def _load_task_stats(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_task_stats(path, task_stats):
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(task_stats, f, indent=0, sort_keys=True)
    os.replace(tmp, path)


def _update_task_stats(task_stats, group, res, rollout_id):
    """Record one group's outcome for its task; 'last_all_pass' marks a task with no learning signal right now."""
    tid = group[0].metadata['instance_id']
    ok = [r for r in res if r.get('status') == 'ok' and r.get('segments')]
    passes = sum(int(r['reward'] > 0) for r in ok)
    st = task_stats.setdefault(tid, {'attempts': 0, 'trajectories': 0, 'passes': 0, 'last_all_pass': False,
                                     'last_rollout': None})
    st['attempts'] += 1
    st['trajectories'] += len(ok)
    st['passes'] += passes
    st['last_all_pass'] = len(ok) >= 2 and passes == len(ok)
    st['last_rollout'] = rollout_id


def _draw_groups(data_buffer, n, task_stats, skip_solved, retry_p, stats):
    """n task groups from the data buffer, skipping tasks flagged last_all_pass (unless retried). Bounded: the buffer
    cycles epochs on its own, so a skip only consumes the task's slot in the current epoch order."""
    groups, draws, max_draws = [], 0, 4 * n + 8
    while len(groups) < n and draws < max_draws:
        batch = data_buffer.get_samples(n - len(groups))
        if not batch:
            break
        for group in batch:
            draws += 1
            st = task_stats.get(group[0].metadata['instance_id'])
            if skip_solved and st and st.get('last_all_pass') and random.random() >= retry_p:
                stats['skipped_solved'] += 1
                continue
            groups.append(group)
    return groups


USABLE_KEY = '__usable_rate__'  # EMA of usable groups / groups run, stored in the task-stats file


def _wave_size(deficit, oversample, task_stats, adaptive, max_oversample=3.5):
    """Groups to draw for a deficit of usable groups."""
    if deficit <= 0:
        return 0
    if adaptive and isinstance(task_stats.get(USABLE_KEY), (int, float)) and task_stats[USABLE_KEY] > 0:
        n = math.ceil(deficit / task_stats[USABLE_KEY])
        return int(min(max(n, math.ceil(deficit * oversample)), math.ceil(deficit * max_oversample)))
    return math.ceil(deficit * oversample)


def _update_usable_rate(task_stats, groups_run, usable, alpha=0.5):
    if groups_run <= 0:
        return
    rate = usable / groups_run
    prev = task_stats.get(USABLE_KEY)
    task_stats[USABLE_KEY] = rate if not isinstance(prev, (int, float)) else (1 - alpha) * prev + alpha * rate


def _as_task(template, instance_id):
    """A data-buffer-like sample for a recorded task (replay)."""
    s = copy.copy(template)
    s.metadata = {**(template.metadata or {}), 'instance_id': instance_id}
    return s


def _run_groups(groups, tag):
    """Run every trajectory of these groups concurrently; returns results[g][k]."""
    jobs = [(g, k, f'{tag}_g{g}_k{k}', s.metadata['instance_id'])
            for g, group in enumerate(groups) for k, s in enumerate(group)]
    with cf.ThreadPoolExecutor(max(1, len(jobs))) as ex:
        flat = list(ex.map(lambda j: _run(j[2], j[3]), jobs))
    results = [[None] * len(group) for group in groups]
    for (g, k, _, _), res in zip(jobs, flat):
        results[g][k] = res
    return results


def _informative(results):
    """DAPO dynamic sampling: keep a group only if its trajectories disagree (reward std > 0)."""
    rewards = [r['reward'] for r in results if r.get('status') == 'ok' and r.get('segments')]
    return len(rewards) >= 2 and len(set(rewards)) > 1


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """One rollout step: rollout_batch_size groups of n_samples_per_prompt trajectories.

    With G4RL_DYNAMIC_SAMPLING=1 (DAPO), groups whose trajectories all got the same reward carry no advantage and are
    discarded. Waves of ceil(deficit * G4RL_OVERSAMPLE) new groups run until rollout_batch_size informative groups
    exist, at most G4RL_MAX_WAVES waves. Surplus informative groups are discarded, so every step trains on
    trajectories of the current policy only. If the waves run out, the missing groups become fillers (no gradient).

    With G4RL_SKIP_SOLVED=1, tasks whose last group was solved by every trajectory (no learning signal, 55% of the
    groups at a 70% resolve rate) are skipped when drawn again, except with probability G4RL_SOLVED_RETRY_P (a
    cheap regression check). Per-task counts persist in G4RL_TASK_STATS (default /root/out/task_stats.json).

    With G4RL_REPLAY_DIR set and <dir>/step<rollout_id>.json present (tools/export_replay.py), the step is replayed:
    the recorded groups and results stand in for the data buffer and the harness (the buffer is still advanced by
    the recorded number of draws, so later live steps draw the tasks the source run would have drawn); the
    task stats are not updated, they already include the replayed steps.

    With G4RL_ADAPTIVE_OVERSAMPLE=1 the first wave is sized from the usable-group rate observed so far (an EMA kept
    in the task-stats file under the key __usable_rate__): groups = ceil(deficit / rate), clamped to
    [G4RL_OVERSAMPLE, G4RL_OVERSAMPLE_MAX (3.5)] x deficit, so a step usually completes in one wave whether 35% or
    70% of the groups carry a signal. Later waves use the same rule on the remaining deficit.
    """
    assert not evaluation, 'evaluation runs through the Kaggle-replica harness, not this rollout'
    router = f'http://{args.sglang_router_ip}:{args.sglang_router_port}'
    httpx.post(f'{PROXY}/admin/sglang', json={'url': router}, timeout=30).raise_for_status()
    dynamic = os.environ.get('G4RL_DYNAMIC_SAMPLING', '0') == '1'
    oversample = float(os.environ.get('G4RL_OVERSAMPLE', '1.5'))
    max_waves = int(os.environ.get('G4RL_MAX_WAVES', '4'))
    need = args.rollout_batch_size
    skip_solved = os.environ.get('G4RL_SKIP_SOLVED', '0') == '1'
    adaptive = os.environ.get('G4RL_ADAPTIVE_OVERSAMPLE', '0') == '1'
    retry_p = float(os.environ.get('G4RL_SOLVED_RETRY_P', '0.2'))
    stats_path = os.environ.get('G4RL_TASK_STATS', '/root/out/task_stats.json')
    task_stats = _load_task_stats(stats_path)
    kept, template, stats = [], None, {'waves': 0, 'groups_run': 0, 'uninformative': 0, 'trajectories': 0, 'resolved': 0,
                                       'skipped_solved': 0}
    replay = None
    replay_dir = os.environ.get('G4RL_REPLAY_DIR')
    if replay_dir and os.path.exists(os.path.join(replay_dir, f'step{rollout_id}.json')):
        with open(os.path.join(replay_dir, f'step{rollout_id}.json')) as f:
            replay = json.load(f)
        logger.info('rollout %s: REPLAY of %s step %s (%d waves, draws %s)', rollout_id, replay.get('source'),
                    replay.get('step'), len(replay['waves']), replay['draws'])
    while ((replay is not None and stats['waves'] < len(replay['waves']))
           or (replay is None and len(kept) < need and stats['waves'] < (max_waves if dynamic else 1))):
        if replay is not None:
            wave = replay['waves'][stats['waves']]
            drawn = data_buffer.get_samples(replay['draws'][stats['waves']])  # advance the epoch position as the source did
            template = template or drawn[0][0]
            groups = [[_as_task(template, grec['instance_id']) for _ in grec['results']] for grec in wave]
            results = [grec['results'] for grec in wave]
        else:
            n = need - len(kept) if not dynamic else max(need - len(kept),
                                                           _wave_size(need - len(kept), oversample, task_stats, adaptive))
            groups = _draw_groups(data_buffer, n, task_stats, skip_solved, retry_p, stats)
            if not groups:
                logger.warning('rollout %s: the data buffer returned no usable groups; stopping the waves', rollout_id)
                break
            template = template or groups[0][0]
            results = _run_groups(groups, f'{RUN_TAG}{rollout_id}w{stats["waves"]}')
        for group, res in zip(groups, results):
            if replay is None:
                _update_task_stats(task_stats, group, res, rollout_id)
            ok = [r for r in res if r.get('status') == 'ok' and r.get('segments')]
            stats['trajectories'] += len(ok)
            stats['resolved'] += sum(int(r['reward'] > 0) for r in ok)
            if dynamic and not _informative(res):
                stats['uninformative'] += 1
                continue
            kept.append((group, res, stats['waves']))
        stats['waves'] += 1
        stats['groups_run'] += len(groups)
    surplus = max(0, len(kept) - need)
    kept = kept[:need]

    out, fillers, n_failed, n_segments = [], [], 0, 0
    for g, (group, res, wave) in enumerate(kept):
        segs = []
        for k, base in enumerate(group):
            r = res[k]
            if r.get('status') != 'ok' or not r.get('segments'):
                n_failed += 1
                logger.warning('dropping trajectory %s%sw%s g%s k%s: %s', RUN_TAG, rollout_id, wave, g, k, r.get('error'))
                fillers.append(_filler(base, rollout_id, g, k))
                continue
            traj_id = rollout_id * 1_000_000 + g * 1_000 + k
            for j, seg in enumerate(r['segments']):
                s = copy.copy(base)
                s.metadata = {**(base.metadata or {}), 'segment': j, 'n_segments': len(r['segments']),
                              'agent_metrics': r.get('metrics'), 'wave': wave}
                s.index = next(_index)
                s.group_index = rollout_id * 1_000 + g  # the task group: GRPO normalizes within it
                s.rollout_id = traj_id
                s.tokens = seg['tokens']
                s.response_length = seg['response_length']
                s.loss_mask = seg['loss_mask']
                s.rollout_log_probs = seg['rollout_log_probs']
                s.reward = r['reward']
                s.status = Sample.Status.COMPLETED
                segs.append(s)
        n_segments += len(segs)
        if segs:
            out.append(segs)
    # Groups that the waves could not fill: fillers keep the rollout count at rollout_batch_size * n_samples.
    if len(kept) < need:
        logger.warning('rollout %s: only %d of %d informative groups after %d waves; the rest are fillers (no gradient)',
                       rollout_id, len(kept), need, stats['waves'])
    for g in range(len(kept), need):
        for k in range(args.n_samples_per_prompt):
            fillers.append(_filler(template, rollout_id, g, k))
    out += [[f] for f in fillers]
    # Pad to a multiple of dp_size * micro_batch_size (Miles' static schedule asserts on it; the segment count varies).
    # Miles' global_batch_size counts distinct rollout_ids and drops the leftover rollouts of a step, so a pad sample
    # must not add a rollout_id: it borrows the first trajectory's, keeping its own (advantage-0) group.
    align = _dp_align(args)
    n_pad = (-sum(len(g) for g in out)) % align
    for i in range(n_pad):
        f = _filler(template, rollout_id, need + 1 + i, 0)
        f.metadata = {**f.metadata, 'dp_pad': True}
        f.rollout_id = out[0][0].rollout_id
        out.append([f])
    if replay is None and dynamic:
        _update_usable_rate(task_stats, stats['groups_run'], stats['groups_run'] - stats['uninformative'])
    try:
        if replay is None:
            _save_task_stats(stats_path, task_stats)
    except OSError as e:
        logger.warning('rollout %s: could not save task stats to %s: %r', rollout_id, stats_path, e)
    n_solved = sum(1 for st in task_stats.values() if isinstance(st, dict) and st.get('last_all_pass'))
    logger.info('rollout %s: %d waves, %d groups run, %d uninformative, %d surplus, %d kept, %d trajectories '
                '(%d resolved), %d dropped, %d fillers, %d segments, %d dp-pad (align %d), %d solved-task draws skipped '
                '(%d of %d seen tasks flagged solved), usable-rate ema %.2f', rollout_id, stats['waves'], stats['groups_run'],
                stats['uninformative'], surplus, len(kept), stats['trajectories'], stats['resolved'], n_failed,
                len(fillers), n_segments, n_pad, align, stats['skipped_solved'], n_solved,
                sum(1 for st in task_stats.values() if isinstance(st, dict)), task_stats.get(USABLE_KEY) or 0.0)
    return out
