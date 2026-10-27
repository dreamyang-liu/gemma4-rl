"""Run swegemma (same agent, prompt, tools, budgets, compaction as env/kaggle_run.py) on rebench tasks, each in its
own pre-built task image via swegemma's docker sandbox backend.

Runs inside the gemma4-kaggle image with the host docker socket mounted (see rebench_run.sh). Differences from the
Kaggle setting are confined to the sandbox:
  - image: rebench/<instance_id>:latest per task (repo deps pre-installed) instead of the competition sandbox;
  - snapshot: ext/rebench_snapshots/<instance_id>.tgz (built by rebench_snapshots.py; no history after base_commit),
    extracted into /workspace as usual; the image's repo dir is then symlinked to /workspace so its editable install
    resolves to the agent's copy;
  - skipped: competition wheel staging, pip install -e, /sandbox/setup.py and the harness pytest.ini/conftest.py
    (the images carry their own deps and pytest config);
  - env: the images point PYTHONPYCACHEPREFIX / XDG_* / PIP_CACHE_DIR into /workspace; they are moved to /tmp so
    caches never enter the agent's diff. xdist/BLAS worker counts are pinned to the container's 2 CPUs (os.cpu_count()
    sees all host CPUs, and `-n auto` then gets OOM-killed);
  - graphs: swegemma adds the "## Code Intelligence Tools" prompt section only when a graph .json and embedding
    .npz exist for the repo (fastapi/rich have them on Kaggle). Valid empty placeholders are written per repo so the
    prompt matches; graph tools then return empty/error results (the budget-aware prompt forbids them anyway);
  - Python 3.7 images: swegemma's in-container patch helper uses Path.is_relative_to and a list[str] annotation
    (3.9+); both are rewritten to 3.7-compatible forms. Phase 2 passes `-o timeout=0` only if pytest-timeout is
    installed (otherwise it is an unknown ini key, fatal under filterwarnings=error).
--gold verifies the gold patch and the empty patch with swegemma's Phase 2 (verify_task) instead of running an agent.
"""

import argparse
import asyncio
import inspect
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

for name in ('LiteLLM', 'LiteLLM Router', 'LiteLLM Proxy', 'litellm', 'httpx', 'httpcore',
             'openai', 'urllib3', 'opentelemetry', 'opentelemetry.context', 'docker'):
    lg = logging.getLogger(name)
    lg.setLevel(logging.WARNING)
    if 'litellm' in name.lower():
        lg.propagate = False
        lg.handlers.clear()

ap = argparse.ArgumentParser()
ap.add_argument('--tasks-file', required=True)
ap.add_argument('--snapshots', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--tasks', nargs='*')
ap.add_argument('--shard', default='0/1')
ap.add_argument('--gold', action='store_true', help='verify gold and empty patches instead of running the agent')
ap.add_argument('--verification-timeout', type=int, default=900)
ap.add_argument('--external-endpoint', default=None)
ap.add_argument('--external-model', default=None)
args = ap.parse_args()

import litellm
import yaml
from adk_submission import ModelRegistry, discover_adapters
from adk_submission.discovery import discover_declared_models
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.models.lite_llm import LiteLlm
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.harness import agent_runner, container_setup, verification
from swegemma.models import load_tasks

litellm.drop_params = True

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
meta = {json.loads(l)['instance_id']: json.loads(l) for l in open(args.tasks_file)}
tasks = load_tasks(Path(args.tasks_file))
if args.tasks:
    tasks = [t for t in tasks if t.instance_id in set(args.tasks)]
si, sn = map(int, args.shard.split('/'))
tasks = tasks[si::sn]
print(f'Running {len(tasks)} rebench tasks ({"gold check" if args.gold else "agent"})', flush=True)

# ---- sandbox adaptation (see module docstring) ----
CURRENT = {}
SANDBOX_ENV = {'TEST_TMPDIR': '/tmp', 'PYTHONPYCACHEPREFIX': '/tmp/.pycache', 'XDG_CACHE_HOME': '/tmp/.cache',
               'PIP_CACHE_DIR': '/tmp/.cache/pip', 'XDG_CONFIG_HOME': '/tmp/.config',
               'XDG_DATA_HOME': '/tmp/.local/share', 'PYTEST_XDIST_AUTO_NUM_WORKERS': '2', 'OMP_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
_orig_extract = container_setup.extract_snapshot
_orig_exclude = container_setup.setup_git_exclude


def extract_snapshot(docker, container_id, snapshot_path, **kw):
    # swegemma streams into `tar -I pigz ... || tar -xzf -`; without pigz the first tar eats part of stdin, and its
    # success check ("/workspace non-empty") passes on the images' pre-existing /workspace/.cache. Start from an empty
    # /workspace, require .git, and fall back to a staged copy + plain tar.
    docker.exec(container_id, 'find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf {} +')
    _orig_extract(docker, container_id, snapshot_path, **kw)
    if docker.exec(container_id, 'test -d /workspace/.git').exit_code != 0:
        docker.exec(container_id, 'find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf {} +')
        docker.copy_to(container_id, Path(snapshot_path), '/var/tmp/')
        staged = f'/var/tmp/{Path(snapshot_path).name}'
        res = docker.exec(container_id, f'tar --no-same-owner -xzf {staged} -C /workspace && rm -f {staged}')
        if res.exit_code != 0 or docker.exec(container_id, 'test -d /workspace/.git').exit_code != 0:
            raise container_setup.ContainerSetupError(f'snapshot extraction failed: {res.stderr or res.stdout}')
    repo_dir = CURRENT['repo_dir']
    res = docker.exec(container_id, f'rm -rf {repo_dir} && ln -s /workspace {repo_dir}')
    if res.exit_code != 0:
        raise container_setup.ContainerSetupError(f'linking {repo_dir} -> /workspace failed: {res.stderr}')


def setup_git_exclude(docker, container_id):
    _orig_exclude(docker, container_id)
    docker.exec(container_id, 'printf ".cache/\\n.config/\\n.local/\\n" >> /workspace/.git/info/exclude')


def _skip(*a, **k):
    return None


_src = inspect.getsource(container_setup.apply_patch_in_container)
for old, new in (('not resolved_target.is_relative_to(workspace)', 'not str(resolved_target).startswith(str(workspace) + os.sep)'),
                 ('def try_cmd(c: list[str]) -> bool:', 'def try_cmd(c):')):
    assert old in _src, old
    _src = _src.replace(old, new)
exec(compile(_src, container_setup.__file__, 'exec'), container_setup.__dict__)
_orig_sandbox_exec = verification.sandbox_exec


async def sandbox_exec(docker, container_id, command, **kw):
    if ' -m pytest ' in command and '-o timeout=0 ' in command:
        command = command.replace(
            '-o timeout=0 ', '$(python3 -c "import pytest_timeout" 2>/dev/null && echo "-o timeout=0") ')
    return await _orig_sandbox_exec(docker, container_id, command, **kw)


verification.sandbox_exec = sandbox_exec
verification.apply_patch_in_container = container_setup.apply_patch_in_container


for mod in (agent_runner, verification):
    mod.extract_snapshot = extract_snapshot
    mod.setup_git_exclude = setup_git_exclude
    for fn in ('setup_container_wheels', 'install_editable_package', 'install_test_dependencies',
               'setup_workspace_test_config'):
        setattr(mod, fn, _skip)

# ---- agent + model (as kaggle_run.py, external endpoint only) ----
AGENT_DIR = OUT / 'agent'
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR)
shutil.copytree('/kaggle/agent', AGENT_DIR)
declared_models = discover_declared_models(AGENT_DIR)
assert len(declared_models) == 1, f'Expected exactly 1 declared model, got: {sorted(declared_models)}'
declared_model = next(iter(declared_models))
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
models = ModelRegistry()
if not args.gold:
    assert args.external_endpoint and args.external_model, 'agent mode needs --external-endpoint/--external-model'
    models.register(declared_model, LiteLlm(model=f'openai/{args.external_model}', api_base=args.external_endpoint,
                                            api_key=os.environ.get('LOCAL_API_KEY', 'EMPTY'), num_retries=3))

eval_config_file = next((c for n in ('eval_config.yaml', 'eval_config.yml') if (c := AGENT_DIR / n).is_file()), None)
raw = (yaml.safe_load(eval_config_file.read_text(encoding='utf-8')) or {}) if eval_config_file else {}
ev = raw.get('evaluation', raw)
turns_raw = ev.get('max_turns', ev.get('max_llm_calls'))
limits, gen_constraints = build_submission_limits()
graph_dir = OUT / '_graph_placeholders'
graph_dir.mkdir(exist_ok=True)
import numpy as np
for repo in {t.repo for t in tasks}:
    slug = repo.replace('/', '_')
    (graph_dir / f'{slug}.json').write_text(json.dumps(
        {'directed': True, 'multigraph': True, 'graph': {'placeholder': f'no code graph was built for {repo}'},
         'nodes': [], 'edges': []}))
    np.savez_compressed(graph_dir / f'{slug}.npz', placeholder=np.zeros(64, dtype=np.float32))
eval_config = EvalConfig(
    tasks_path=Path(args.tasks_file),
    snapshots_dir=Path(args.snapshots),
    results_dir=OUT / 'results',
    submission_dir=AGENT_DIR,
    models=models,
    sandbox='docker',
    timeout_seconds=int(ev.get('timeout_seconds', 300)),
    verification_timeout_seconds=args.verification_timeout,
    max_time_minutes=float(ev.get('max_time_minutes', 60.0)),
    max_tool_calls=int(ev.get('max_tool_calls', 100)),
    max_turns=int(turns_raw) if turns_raw is not None else None,
    limits=limits,
    generation_constraints=gen_constraints,
    adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
    graph_dir=str(graph_dir),
    embeddings_dir=str(graph_dir),
    verbose=False,
)
evaluator = Evaluator(eval_config)
sandbox = evaluator.sandbox
sandbox.config.environment = dict(SANDBOX_ENV)
sandbox._image_resolved = True


async def gold_check(task):
    snap = Path(args.snapshots) / f'{task.instance_id}.tgz'
    out = {}
    for name, patch in (('gold', task.patch), ('empty', '')):
        t0 = time.perf_counter()
        res = await verification.verify_task(sandbox, eval_config, task, snap, agent_patch=patch,
                                             start_time=t0)
        out[name] = dict(resolved=bool(res.resolved), exit_code=res.test_exit_code, error=res.error,
                         s=round(time.perf_counter() - t0, 1), tail=(res.test_output or '')[-1500:])
    return out


try:
    for idx, task in enumerate(tasks, start=1):
        m = meta[task.instance_id]
        CURRENT['repo_dir'] = m['repo_dir']
        sandbox.config.image = m['image']
        print(f'[{idx}/{len(tasks)}] {task.instance_id}', flush=True)
        t0 = time.time()
        if args.gold:
            r = asyncio.run(gold_check(task))
            row = dict(id=task.instance_id, gold=r['gold']['resolved'], empty=r['empty']['resolved'],
                       usable=r['gold']['resolved'] and not r['empty']['resolved'], detail=r,
                       duration_s=round(time.time() - t0, 1))
        else:
            res = asyncio.run(evaluator.evaluate_task(task=task, task_index=idx, total_tasks=len(tasks)))
            patch = (res.agent_patch or '').strip()
            row = dict(id=task.instance_id, resolved=bool(res.resolved), exit_code=res.test_exit_code,
                       patch_chars=len(patch), tool_calls=res.tool_calls, error=(res.error or '')[:500],
                       duration_s=round(res.duration_seconds, 1), patch=patch)
        print('  ->', json.dumps({k: v for k, v in row.items() if k not in ('detail', 'patch')}), flush=True)
        with open(OUT / 'results.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
finally:
    sandbox.cleanup_all()
