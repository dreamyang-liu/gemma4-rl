"""OpenAI Chat Completions server for RL rollouts: vLLM 0.19.1's own chat serving on CPU, SGLang for the tokens.

swegemma (ADK + LiteLLM) talks to this server exactly as it talks to the Kaggle vLLM server. Everything from the JSON
request to the JSON response is vLLM code (OpenAIServingChat: chat template render, request validation, sampling
defaults from generation_config, max_tokens rules, gemma4 reasoning and tool parsers, tool-call ids, finish_reason,
usage). Only the engine is replaced: `TitoEngineClient.generate` receives vLLM's prompt token ids and SamplingParams,
and samples the completion on SGLang's native /generate (token ids in, token ids and logprobs out).

Engine-side behaviour that vLLM implements below the serving layer is reproduced with vLLM's own classes:
  - SamplingParams.update_from_generation_config (eos and stop token ids), max_tokens from max_model_len;
  - ThinkingTokenBudgetLogitsProcessor (thinking_token_budget): its state machine is run step by step. SGLang samples
    the free stretches between the processor's check points, and the forced `<channel|>` tokens are inserted here;
  - stop checks (scheduler check_stop) and incremental detokenization (FastIncrementalDetokenizer).

Each request/response pair is appended to <log_dir>/<worker>.jsonl in the record format of env/bedrock_proxy.py, plus
`tito`: the prompt ids, the output ids, their sampling logprobs, and which output positions were forced (no gradient).
Route as http://host:port/w/<worker>/v1.

usage (gemma4-kaggle image, CPU only):
  python tito_proxy.py --sglang http://127.0.0.1:30000 --log-dir DIR --port 8000 -- \
      /model --served-model-name gemma-4-31b-it --max-model-len 32768 --tool-call-parser gemma4 \
      --reasoning-parser gemma4 --enable-auto-tool-choice \
      --reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}'
Everything after `--` is parsed by vLLM's own `vllm serve` argument parser, so pass the Kaggle server's arguments.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
import types
import uuid

ap = argparse.ArgumentParser()
ap.add_argument('--sglang', default='', help='SGLang server or router base URL, e.g. http://127.0.0.1:30000; '
                'can also be set at run time with POST /admin/sglang {"url": ...} (Miles picks its router address)')
ap.add_argument('--log-dir', required=True)
ap.add_argument('--host', default='0.0.0.0')
ap.add_argument('--port', type=int, default=8000)
ap.add_argument('--engine-timeout', type=float, default=1800.0, help='seconds for one SGLang /generate call')
ap.add_argument('--prefill-chunk', type=int, default=2048,
                help="the Kaggle server's max_num_batched_tokens (vLLM's <70 GiB default, kaggle_run.py pins it)")
ap.add_argument('--cache-block', type=int, default=16, help="the Kaggle server's KV cache block size")
ap.add_argument('--fake-outputs', default='',
                help='parity test only: JSONL of {"prompt_ids", "output_ids"}; the engine replays these output ids '
                     'for a matching prompt instead of calling SGLang')
own, vllm_argv = (sys.argv[1:sys.argv.index('--')], sys.argv[sys.argv.index('--') + 1:]) if '--' in sys.argv \
    else (sys.argv[1:], [])
args = ap.parse_args(own)

os.environ.setdefault('VLLM_CPU_KVCACHE_SPACE', '0')
# The CUDA build of vLLM only picks its CPU platform when built for CPU. This process never runs a model (its config
# and pre/post-processing are all it uses), so resolve the CPU platform on a host or container without GPUs.
import vllm.platforms as _vllm_platforms  # noqa: E402

_vllm_platforms.builtin_platform_plugins['cpu'] = lambda: 'vllm.platforms.cpu.CpuPlatform'
import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from vllm.config import VllmConfig  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.entrypoints.chat_utils import load_chat_template  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse  # noqa: E402
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat  # noqa: E402
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args  # noqa: E402
from vllm.entrypoints.openai.engine.protocol import ErrorResponse  # noqa: E402
from vllm.entrypoints.openai.models.protocol import BaseModelPath  # noqa: E402
from vllm.entrypoints.openai.models.serving import OpenAIServingModels  # noqa: E402
from vllm.entrypoints.serve.render.serving import OpenAIServingRender  # noqa: E402
from vllm.outputs import CompletionOutput, RequestOutput  # noqa: E402
from vllm.plugins.io_processors import get_io_processor  # noqa: E402
from vllm.renderers import renderer_from_config  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402
from vllm.v1.engine.detokenizer import IncrementalDetokenizer  # noqa: E402
from vllm.v1.sample.logits_processor.builtin import ThinkingTokenBudgetLogitsProcessor  # noqa: E402

# ---- vLLM configuration, parsed exactly as `vllm serve` parses it ----
vparser = make_arg_parser(FlexibleArgumentParser())
vargs = vparser.parse_args(vllm_argv[1:] if vllm_argv[:1] == ['serve'] else vllm_argv)
if getattr(vargs, 'model_tag', None):
    vargs.model = vargs.model_tag
validate_parsed_serve_args(vargs)
engine_args = AsyncEngineArgs.from_cli_args(vargs)
# As EngineArgs.create_engine_config: the serving layer reads the reasoning parser from structured_outputs_config.
if engine_args.reasoning_parser:
    vargs.structured_outputs_config.reasoning_parser = engine_args.reasoning_parser
if engine_args.reasoning_parser_plugin:
    vargs.structured_outputs_config.reasoning_parser_plugin = engine_args.reasoning_parser_plugin
from vllm.entrypoints.openai.api_server import validate_api_server_args  # noqa: E402

validate_api_server_args(vargs)
model_config = engine_args.create_model_config()
model_config.quantization = None  # as `vllm launch render`: preprocessing only, no quantized kernels
vllm_config = VllmConfig(model_config=model_config)
renderer = renderer_from_config(vllm_config)
io_processor = get_io_processor(vllm_config, renderer, model_config.io_processor_plugin)
tokenizer = renderer.tokenizer
generation_config_fields = model_config.try_get_generation_config()
eos_token_id = renderer.get_eos_token_id()

# The thinking-budget processor as the Kaggle server configures it (adk_submission adds --reasoning-config for gemma4).
reasoning_config = getattr(vargs, 'reasoning_config', None)
if isinstance(reasoning_config, str):
    reasoning_config = json.loads(reasoning_config)
if reasoning_config is not None and not isinstance(reasoning_config, dict):
    reasoning_config = {k: getattr(reasoning_config, k) for k in ('reasoning_start_str', 'reasoning_end_str')}
BUDGET = None
if reasoning_config:
    BUDGET = ThinkingTokenBudgetLogitsProcessor(
        types.SimpleNamespace(
            reasoning_config=types.SimpleNamespace(
                reasoning_start_token_ids=tokenizer.encode(reasoning_config['reasoning_start_str'],
                                                           add_special_tokens=False),
                reasoning_end_token_ids=tokenizer.encode(reasoning_config['reasoning_end_str'],
                                                         add_special_tokens=False)),
            scheduler_config=types.SimpleNamespace(max_num_seqs=1)),
        device='cpu', is_pin_memory=False)

FAKE = {}
if args.fake_outputs:
    for line in open(args.fake_outputs):
        r = json.loads(line)
        FAKE[(tuple(r['prompt_ids']), r.get('thinking_token_budget'))] = r['output_ids']

SUPPORTED_DEFAULTS = {'n': 1, 'best_of': None, 'presence_penalty': 0.0, 'frequency_penalty': 0.0,
                      'repetition_penalty': 1.0, 'min_tokens': 0, 'logprobs': None, 'prompt_logprobs': None,
                      'stop': [], 'bad_words': [], 'logit_bias': None, 'allowed_token_ids': None,
                      'structured_outputs': None, 'ignore_eos': False, 'include_stop_str_in_output': False}
RECORDS = {}  # request_id -> tito record, filled by the engine and read by the HTTP handler
SGLANG = {'url': args.sglang.rstrip('/')}
http = None
import contextvars  # noqa: E402

WORKER = contextvars.ContextVar('worker', default='default')
HISTORY = {}  # worker -> its last token sequences (prompt + output), for the prefix-cache estimate


def prefill_steps(worker, prompt_ids):
    """Engine steps that the Kaggle server spends on this prompt before sampling its first token.

    vLLM updates every logits processor once per engine step, also on the steps that only prefill a chunk of the
    prompt, and the thinking-budget processor counts down on each of them. On Kaggle one request runs at a time, so a
    step prefills `--prefill-chunk` new tokens. The new tokens are the prompt minus its prefix-cache hit: the longest
    common prefix with a sequence that this trajectory already ran, in whole blocks, at most len(prompt) - 1.
    """
    best = 0
    for seq in (*HISTORY.get(worker, ()), *GLOBAL_RECENT):
        n = min(len(seq), len(prompt_ids))
        i = 0
        while i < n and seq[i] == prompt_ids[i]:
            i += 1
        best = max(best, i)
    cached = min(best, len(prompt_ids) - 1) // args.cache_block * args.cache_block
    return max(1, -(-(len(prompt_ids) - cached) // args.prefill_chunk)), cached


GLOBAL_RECENT = []  # the last sequences of any trajectory: on Kaggle the previous task's prompts stay in the cache


def remember(worker, seq):
    h = HISTORY.setdefault(worker, [])
    h.append(seq)
    del h[:-4]
    GLOBAL_RECENT.append(seq)
    del GLOBAL_RECENT[:-4]


class EngineError(RuntimeError):
    pass


class TitoEngineClient:
    """The part of vLLM's EngineClient that OpenAIServingChat uses, backed by SGLang."""

    def __init__(self):
        self.vllm_config = vllm_config
        self.model_config = model_config
        self.renderer = renderer
        self.io_processor = io_processor
        self.input_processor = None
        self.errored = False
        self.dead_error = RuntimeError('engine is dead')
        self.is_running = True
        self.is_stopped = False

    async def is_tracing_enabled(self):
        return False

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        prompt_ids = list(prompt['prompt_token_ids'])
        # vLLM's InputProcessor.process_inputs: max_tokens default and generation_config eos / stop token ids.
        params = sampling_params.clone()
        if params.max_tokens is None:
            params.max_tokens = model_config.max_model_len - len(prompt_ids)
        params.update_from_generation_config(generation_config_fields, eos_token_id)
        for k, default in SUPPORTED_DEFAULTS.items():
            v = getattr(params, k, default)
            if v not in (default, None) and not (isinstance(default, list) and not v):
                raise EngineError(f'sampling parameter {k}={v!r} is not supported by the TITO engine')
        worker = WORKER.get()
        steps, cached = prefill_steps(worker, prompt_ids)
        out, logprobs, forced, finish, stop_reason, calls = await self._sample(prompt_ids, params, steps)
        # The KV of the last sampled token is never computed, so it is not part of the cached sequence.
        remember(worker, prompt_ids + out[:-1])
        # vLLM's output processor: incremental detokenization, excluding a stop token from the text.
        detok = IncrementalDetokenizer.from_new_request(tokenizer, types.SimpleNamespace(
            request_id=request_id, sampling_params=params, prompt_token_ids=prompt_ids))
        detok.update(out, stop_terminated=finish == 'stop')
        text = detok.get_next_output_text(finished=True, delta=False)
        RECORDS[request_id] = {'prompt_ids': prompt_ids, 'output_ids': out, 'output_logprobs': logprobs,
                               'forced': forced, 'finish_reason': finish, 'stop_reason': stop_reason,
                               'engine_calls': calls, 'max_tokens': params.max_tokens,
                               'prefill_steps': steps, 'cached_estimate': cached,
                               'sampling': {'temperature': params.temperature, 'top_p': params.top_p,
                                            'top_k': params.top_k, 'min_p': params.min_p,
                                            'thinking_token_budget': params.thinking_token_budget,
                                            'eos_token_id': params.eos_token_id,
                                            'stop_token_ids': sorted(params.stop_token_ids or [])}}
        yield RequestOutput(
            request_id=request_id, prompt=None, prompt_token_ids=prompt_ids, prompt_logprobs=None,
            outputs=[CompletionOutput(index=0, text=text, token_ids=out, cumulative_logprob=None, logprobs=None,
                                      finish_reason=finish, stop_reason=stop_reason)],
            finished=True)

    def _stop(self, out, params, prompt_len):
        """vLLM's scheduler check_stop after the last output token: (finish_reason, stop_reason) or None."""
        last = out[-1]
        if len(out) >= params.min_tokens:
            if last == params.eos_token_id:
                return 'stop', None
            if last in (params.stop_token_ids or ()):
                return 'stop', last
        if prompt_len + len(out) >= model_config.max_model_len or len(out) >= params.max_tokens:
            return 'length', None
        return None

    async def _sample(self, prompt_ids, params, prefill_steps=1):
        out, logprobs, forced = [], [], []
        state = None
        if BUDGET is not None and params.thinking_token_budget is not None:
            state = BUDGET._init_state_entry(prompt_ids, params.thinking_token_budget)
            state['output_tok_ids'] = out
            # The prefill-only engine steps before the step that samples the first token (no output yet).
            for _ in range(prefill_steps - 1):
                BUDGET._update_think_state(state)
        replay = FAKE.get((tuple(prompt_ids), params.thinking_token_budget)) if FAKE else None
        calls = 0
        while True:
            # One engine step: the logits processor updates its state, then forces a token or lets the model sample.
            if state is not None:
                BUDGET._update_think_state(state)
            if state is not None and state['in_end']:
                tok = BUDGET.reasoning_end_token_ids[state['end_count']]
                if replay is not None:
                    assert replay[len(out)] == tok, (f'replayed token {replay[len(out)]} at {len(out)} where the budget '
                                                     f'forces {tok} (prefill steps {prefill_steps}, prompt {len(prompt_ids)})')
                out.append(tok)
                logprobs.append(None)
                forced.append(len(out) - 1)
                done = self._stop(out, params, len(prompt_ids))
                if done:
                    return out, logprobs, forced, done[0], done[1], calls
                continue
            # Free stretch: this step and the next `check_count_down` steps only decrement the processor's counter.
            n_free = state['check_count_down'] + 1 if state is not None else params.max_tokens
            n = min(n_free, params.max_tokens - len(out), model_config.max_model_len - len(prompt_ids) - len(out))
            if replay is not None:
                chunk, chunk_lp = self._replay_chunk(replay, out, n, params, len(prompt_ids))
            else:
                chunk, chunk_lp = await self._sglang(prompt_ids + out, n, params)
            calls += 1
            for i, (tok, lp) in enumerate(zip(chunk, chunk_lp)):
                out.append(tok)
                logprobs.append(lp)
                done = self._stop(out, params, len(prompt_ids))
                if done:
                    return out, logprobs, forced, done[0], done[1], calls
                if state is not None and i < len(chunk) - 1:
                    BUDGET._update_think_state(state)  # a decrement-only step (asserted below)
                    assert not state['in_end'], 'budget processor left its countdown early'
            if len(chunk) < n:
                raise EngineError(f'SGLang returned {len(chunk)} tokens without stopping, {n} expected')

    def _replay_chunk(self, replay, out, n, params, prompt_len):
        chunk = replay[len(out):len(out) + n]
        return chunk, [0.0] * len(chunk)

    async def _sglang(self, input_ids, n, params):
        sp = {'max_new_tokens': n, 'temperature': params.temperature, 'top_p': params.top_p,
              'top_k': params.top_k if params.top_k and params.top_k > 0 else -1, 'min_p': params.min_p,
              'stop_token_ids': sorted({params.eos_token_id, *(params.stop_token_ids or [])} - {None}),
              'skip_special_tokens': False, 'no_stop_trim': True, 'ignore_eos': False}
        if params.seed is not None:
            sp['sampling_seed'] = params.seed
        body = {'input_ids': input_ids, 'sampling_params': sp, 'return_logprob': True}
        if not SGLANG['url']:
            raise EngineError('no SGLang URL set (POST /admin/sglang)')
        r = await http.post(f'{SGLANG["url"]}/generate', json=body, timeout=args.engine_timeout)
        if r.status_code != 200:
            raise EngineError(f'SGLang /generate HTTP {r.status_code}: {r.text[:300]}')
        d = r.json()
        ids = list(d.get('output_ids') or [])
        lps = [x[0] for x in (d.get('meta_info') or {}).get('output_token_logprobs') or []]
        if len(lps) != len(ids):
            raise EngineError(f'SGLang returned {len(ids)} ids and {len(lps)} logprobs')
        return ids, lps


engine = TitoEngineClient()
served = vargs.served_model_name or [vargs.model]
models = OpenAIServingModels(engine_client=engine,
                             base_model_paths=[BaseModelPath(name=n, model_path=vargs.model) for n in served])
chat_template = load_chat_template(vargs.chat_template)
render = OpenAIServingRender(
    model_config=model_config, renderer=renderer, io_processor=io_processor, model_registry=models.registry,
    request_logger=None, chat_template=chat_template,
    chat_template_content_format=vargs.chat_template_content_format,
    trust_request_chat_template=vargs.trust_request_chat_template,
    enable_auto_tools=vargs.enable_auto_tool_choice,
    exclude_tools_when_tool_choice_none=vargs.exclude_tools_when_tool_choice_none,
    tool_parser=vargs.tool_call_parser, reasoning_parser=vargs.structured_outputs_config.reasoning_parser,
    default_chat_template_kwargs=vargs.default_chat_template_kwargs, log_error_stack=vargs.log_error_stack)
chat = OpenAIServingChat(
    engine_client=engine, models=models, response_role=vargs.response_role, openai_serving_render=render,
    request_logger=None, chat_template=chat_template,
    chat_template_content_format=vargs.chat_template_content_format,
    default_chat_template_kwargs=vargs.default_chat_template_kwargs,
    trust_request_chat_template=vargs.trust_request_chat_template,
    return_tokens_as_token_ids=vargs.return_tokens_as_token_ids,
    enable_auto_tools=vargs.enable_auto_tool_choice,
    exclude_tools_when_tool_choice_none=vargs.exclude_tools_when_tool_choice_none,
    tool_parser=vargs.tool_call_parser, reasoning_parser=vargs.structured_outputs_config.reasoning_parser,
    enable_prompt_tokens_details=vargs.enable_prompt_tokens_details,
    enable_force_include_usage=vargs.enable_force_include_usage,
    enable_log_outputs=vargs.enable_log_outputs, enable_log_deltas=vargs.enable_log_deltas)

os.makedirs(args.log_dir, exist_ok=True)
_locks = {}
_glock = threading.Lock()
STATS = {'requests': 0, 'errors': 0, 'forced_tokens': 0}
app = FastAPI()


@app.on_event('startup')
async def _startup():
    global http
    http = httpx.AsyncClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=256))


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.get('/stats')
async def stats():
    return {**STATS, 'sglang': SGLANG['url']}


@app.post('/admin/sglang')
async def set_sglang(raw: Request):
    SGLANG['url'] = str((await raw.json())['url']).rstrip('/')
    return {'sglang': SGLANG['url']}


@app.get('/{prefix:path}v1/models')
async def list_models(prefix: str):
    return JSONResponse((await models.show_available_models()).model_dump())


@app.post('/{prefix:path}v1/chat/completions')
async def chat_completions(prefix: str, raw: Request):
    m = re.fullmatch(r'(?:w/([\w.-]+)/)?', prefix)
    if m is None:
        return JSONResponse({'error': {'message': f'unknown path /{prefix}v1/chat/completions'}}, status_code=404)
    worker = m.group(1) or 'default'
    t0 = time.time()
    body = await raw.json()
    rid = uuid.uuid4().hex
    code, out = 200, None
    try:
        req = ChatCompletionRequest.model_validate({**body, 'request_id': rid})
    except ValidationError as e:
        code, out = 400, {'error': {'message': str(e), 'type': 'BadRequestError', 'code': 400}}
    if out is None:
        if req.stream:
            code, out = 400, {'error': {'message': 'the TITO proxy serves non-streaming requests only', 'code': 400}}
        else:
            try:
                WORKER.set(worker)
                res = await chat.create_chat_completion(req, None)
            except EngineError as e:
                res = None
                code, out = 500, {'error': {'message': str(e), 'type': 'InternalServerError', 'code': 500}}
            if isinstance(res, ErrorResponse):
                code, out = res.error.code, res.model_dump()
            elif isinstance(res, ChatCompletionResponse):
                out = res.model_dump()
    rec = None
    for k in [k for k in RECORDS if k.endswith(rid)]:
        rec = RECORDS.pop(k)
    STATS['requests'] += 1
    if code != 200:
        STATS['errors'] += 1
    if rec:
        STATS['forced_tokens'] += len(rec['forced'])
        rec['summarizer'] = not body.get('tools')  # an ADK compaction call
    with _glock:
        lock = _locks.setdefault(worker, threading.Lock())
    with lock, open(os.path.join(args.log_dir, f'{worker}.jsonl'), 'a') as f:
        f.write(json.dumps({'ts': t0, 'dt': round(time.time() - t0, 2), 'sim_sleep': 0.0, 'status': code,
                            'request': body, 'bedrock_response': out, 'chat_response': out, 'tito': rec}) + '\n')
    return JSONResponse(out, status_code=code)


if __name__ == '__main__':
    print(f'tito proxy on :{args.port} -> {args.sglang} served={served} budget={reasoning_config} '
          f'fake={len(FAKE)}', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
