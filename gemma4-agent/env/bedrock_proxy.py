"""OpenAI Chat Completions -> Bedrock (bedrock-mantle) Responses API proxy for Gemma 4, keeping the reasoning.

Bedrock's Chat Completions route for google.gemma-4-31b drops the thinking text; only the Responses API returns it.
The swegemma/ADK/LiteLLM stack speaks Chat Completions, so this proxy translates both ways:
  - assistant history: reasoning_content|reasoning -> a `reasoning` input item, text -> assistant message,
    tool_calls -> `function_call` items; tool messages -> `function_call_output`.
  - thinking: requests with chat_template_kwargs.enable_thinking (or thinking_token_budget > 0) get
    reasoning.effort=high (Gemma 4 thinking is on/off; Bedrock has no budget control).
  - response: reasoning text -> message.reasoning_content and .reasoning, function calls -> tool_calls,
    usage -> prompt/completion tokens (ADK's token-threshold compaction reads prompt_tokens).
Optional Kaggle-speed emulation (--sim-decode-tps > 0): each response is held until it would have finished on the
competition's 4xL4 W4A16 server, i.e. new_prompt_tokens / sim_prefill_tps + output_tokens / sim_decode_tps, so the
agent's wall-clock budget binds as it does on Kaggle. New prompt tokens = growth over the worker's previous prompt
(vLLM prefix caching), or the whole prompt after a compaction. Responses slower than that are not shortened.
--backend openai forwards requests unchanged to local OpenAI-compatible servers (vLLM) given by --upstream URL[,URL...];
each worker is pinned to one upstream (crc32 of its name) so its prefix cache stays warm. Used to log local sampling.
--backend converse translates to the Bedrock Converse API (bedrock-runtime /model/<id>/converse). On Bedrock this is the
route that returns the thinking of models such as zai.glm-5 and moonshotai.kimi-k2.5 (reasoningContent blocks, with
additionalModelRequestFields.reasoning_effort=high when the request enables thinking) and keeps earlier reasoning blocks
in the prompt. As in Gemma 4's template, thoughts are replayed only for assistant turns after the last user message.
--backend chat instead forwards Chat Completions as-is to bedrock-runtime's OpenAI route (for models that return
reasoning_content there, e.g. us.moonshotai.kimi-k3): Gemma/vLLM-only fields are dropped, history keeps reasoning_content.
vLLM adaptation for the responses backend (both off by default). Bedrock silently drops `reasoning` input items, so
the model never sees its earlier thoughts, while vLLM's Gemma 4 template on Kaggle keeps every thought after the last
user message; Bedrock's prompt_tokens are therefore ~1.6x smaller and ADK compaction (14336 tokens) rarely fires.
  --history-thoughts text  send earlier thoughts as assistant text before the tool call. Special-token strings are
                           encoded as plain text by Bedrock, so a real thought channel can't be reproduced; this keeps
                           the information and roughly the token count.
  --vllm-usage MODEL_DIR   report prompt_tokens as vLLM counts them (that dir's chat_template.jinja + tokenizer.json,
                           enable_thinking, tools), so compaction triggers where it would on Kaggle; the upstream's
                           own count is kept as usage.bedrock_prompt_tokens. Works with every backend, e.g. for
                           other teacher models whose tokenizers count differently.
Every request/response pair is appended to <log_dir>/<worker>.jsonl; route as http://host:port/w/<worker>/v1.

usage: python bedrock_proxy.py --port 8000 --log-dir /logs [--model google.gemma-4-31b]
env:   AWS_REGION, AWS_BEARER_TOKEN_BEDROCK
"""
import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
import zlib
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8000)
ap.add_argument('--log-dir', required=True)
ap.add_argument('--model', default='google.gemma-4-31b')
ap.add_argument('--max-retries', type=int, default=8)
ap.add_argument('--backend', choices=['responses', 'chat', 'openai', 'converse'], default='responses')
ap.add_argument('--upstream', default='', help='openai backend: comma-separated base URLs ending in /v1')
ap.add_argument('--sim-decode-tps', type=float, default=0.0, help='0 = off; Kaggle 4xL4 measured ~31')
ap.add_argument('--sim-prefill-tps', type=float, default=2000.0)
ap.add_argument('--history-thoughts', choices=['reasoning', 'text'], default='reasoning',
                help='responses backend: send earlier thoughts as reasoning items (dropped by Bedrock) or as assistant text')
ap.add_argument('--vllm-usage', default='', help='model dir: report prompt_tokens as vLLM would count them')
ap.add_argument('--summary-retries', type=int, default=0,
                help='chat/converse: resample ADK compaction summaries that continue the agent instead (see bad_summary)')
ap.add_argument('--summary-model', default='',
                help='e.g. google.gemma-4-31b: compaction (no-tools) requests are answered by this Bedrock mantle model')
ap.add_argument('--summary-system', default='',
                help='chat/converse: system instruction added to compaction (no-tools) requests, e.g. for teacher models '
                     'that otherwise continue the transcript instead of summarizing it')
args = ap.parse_args()
if args.backend == 'openai':
    UPSTREAMS = [u.rstrip('/') for u in args.upstream.split(',') if u]
    assert UPSTREAMS, '--backend openai needs --upstream'
    BASE = ','.join(UPSTREAMS)
elif args.backend == 'converse':
    BASE = f"https://bedrock-runtime.{os.environ['AWS_REGION']}.amazonaws.com"
else:
    BASE = (f"https://bedrock-mantle.{os.environ['AWS_REGION']}.api.aws/openai/v1" if args.backend == 'responses' else
            f"https://bedrock-runtime.{os.environ['AWS_REGION']}.amazonaws.com/openai/v1")
KEY = os.environ.get('AWS_BEARER_TOKEN_BEDROCK', 'EMPTY')
os.makedirs(args.log_dir, exist_ok=True)
_locks = {}
_glock = threading.Lock()
STATS = {'requests': 0, 'errors': 0, 'retries': 0, 'sim_sleep_s': 0.0}
_last_prompt = {}  # worker -> previous prompt tokens (for the prefix-cache prefill estimate)


def sim_target(worker, usage):
    p, o = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
    prev = _last_prompt.get(worker, 0)
    _last_prompt[worker] = p
    new = p if p < 0.7 * prev or prev == 0 else max(0, p - prev)
    return new / args.sim_prefill_tps + o / args.sim_decode_tps


def _text(content):
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    return ''.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') in (None, 'text', 'input_text', 'output_text'))


def to_responses(req):
    items = []
    msgs = req.get('messages', [])
    # vLLM's Gemma 4 template only renders thoughts of assistant messages after the last user message
    last_user = max((i for i, m in enumerate(msgs) if m.get('role') == 'user'), default=-1)
    for i, m in enumerate(msgs):
        role = m.get('role')
        if role in ('system', 'developer', 'user'):
            items.append({'role': role, 'content': _text(m.get('content'))})
        elif role == 'assistant':
            r = m.get('reasoning_content') or m.get('reasoning')
            t = _text(m.get('content'))
            if r and args.history_thoughts == 'text':
                if i > last_user:
                    t = r + ('\n\n' + t if t else '')
            elif r:
                items.append({'type': 'reasoning', 'summary': [], 'content': [{'type': 'reasoning_text', 'text': r}]})
            if t:
                items.append({'role': 'assistant', 'content': t})
            for tc in m.get('tool_calls') or []:
                f = tc.get('function', {})
                a = f.get('arguments')
                items.append({'type': 'function_call', 'call_id': tc.get('id'), 'name': f.get('name'),
                              'arguments': a if isinstance(a, str) else json.dumps(a)})
        elif role == 'tool':
            items.append({'type': 'function_call_output', 'call_id': m.get('tool_call_id'), 'output': _text(m.get('content'))})
    body = {'model': args.model, 'input': items}
    tools = [{'type': 'function', 'name': t['function']['name'], 'description': t['function'].get('description', ''),
              'parameters': t['function'].get('parameters') or {'type': 'object', 'properties': {}}}
             for t in req.get('tools') or [] if t.get('type') == 'function']
    if tools:
        body['tools'] = tools
    for k in ('temperature', 'top_p'):
        if req.get(k) is not None:
            body[k] = req[k]
    mx = req.get('max_completion_tokens') or req.get('max_tokens')
    if mx:
        body['max_output_tokens'] = int(mx)
    ctk = req.get('chat_template_kwargs') or {}
    thinking = ctk.get('enable_thinking')
    if thinking is None:
        thinking = (req.get('thinking_token_budget') or 0) > 0
    if thinking:
        body['reasoning'] = {'effort': 'high'}
    return body


def to_chat(resp, req):
    reasoning, text, calls = [], [], []
    for o in resp.get('output', []):
        if o.get('type') == 'reasoning':
            reasoning.append(''.join(c.get('text', '') for c in (o.get('content') or []) + (o.get('summary') or [])))
        elif o.get('type') == 'message':
            text.append(_text(o.get('content')))
        elif o.get('type') == 'function_call':
            calls.append({'id': o.get('call_id'), 'type': 'function', 'function': {'name': o.get('name'), 'arguments': o.get('arguments') or '{}'}})
    msg = {'role': 'assistant', 'content': ''.join(text) or None}
    if reasoning:
        msg['reasoning_content'] = msg['reasoning'] = '\n'.join(reasoning)
    if calls:
        msg['tool_calls'] = calls
    inc = (resp.get('incomplete_details') or {}).get('reason')
    finish = 'length' if inc == 'max_output_tokens' else ('tool_calls' if calls else 'stop')
    u = resp.get('usage') or {}
    usage = {'prompt_tokens': u.get('input_tokens', 0), 'completion_tokens': u.get('output_tokens', 0),
             'total_tokens': u.get('total_tokens', u.get('input_tokens', 0) + u.get('output_tokens', 0)),
             'prompt_tokens_details': {'cached_tokens': (u.get('input_tokens_details') or {}).get('cached_tokens', 0)}}
    return {'id': resp.get('id', 'resp'), 'object': 'chat.completion', 'created': int(time.time()), 'model': req.get('model'),
            'choices': [{'index': 0, 'message': msg, 'finish_reason': finish}], 'usage': usage}


_vllm = None


def vllm_prompt_tokens(req):
    """Prompt tokens of this request as vLLM's OpenAI server would render and count them for Gemma 4."""
    global _vllm
    if _vllm is None:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        tpl = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
            open(os.path.join(args.vllm_usage, 'chat_template.jinja')).read())
        _vllm = (tpl, Tokenizer.from_file(os.path.join(args.vllm_usage, 'tokenizer.json')))
    tpl, tok = _vllm
    msgs = []
    for m in req.get('messages', []):
        m = {k: v for k, v in m.items() if k in ('role', 'content', 'tool_calls', 'tool_call_id', 'name')} | (
            {'reasoning': m.get('reasoning') or m.get('reasoning_content')} if m.get('role') == 'assistant' else {})
        if m.get('tool_calls'):
            m['tool_calls'] = [{**tc, 'function': {**tc['function'], 'arguments': json.loads(tc['function']['arguments'] or '{}')
                                                   if isinstance(tc['function'].get('arguments'), str) else tc['function'].get('arguments')}}
                               for tc in m['tool_calls']]
        msgs.append(m)
    kw = {'tools': req['tools']} if req.get('tools') else {}
    enable = (req.get('chat_template_kwargs') or {}).get('enable_thinking', False)
    text = tpl.render(messages=msgs, bos_token='<bos>', enable_thinking=enable, add_generation_prompt=True, **kw)
    return len(tok.encode(text, add_special_tokens=False).ids)


def _thinking_on(req):
    on = (req.get('chat_template_kwargs') or {}).get('enable_thinking')
    return (req.get('thinking_token_budget') or 0) > 0 if on is None else bool(on)


def to_converse(req):
    msgs = req.get('messages', [])
    last_user = max((i for i, m in enumerate(msgs) if m.get('role') == 'user'), default=-1)
    system, out = [], []

    def add(role, blocks):
        if out and out[-1]['role'] == role:  # Converse needs alternating roles; merge (e.g. several toolResults)
            out[-1]['content'].extend(blocks)
        else:
            out.append({'role': role, 'content': blocks})

    # Converse rejects the whole request if any historical toolUse has an empty or non-[a-zA-Z0-9_-] name or an empty id
    # (a mis-parsed call from the model); give such calls valid placeholders and keep their results paired.
    fixed_ids, n_fixed = {}, 0
    for i, m in enumerate(msgs):
        role = m.get('role')
        if role in ('system', 'developer'):
            system.append({'text': _text(m.get('content'))})
        elif role == 'user':
            add('user', [{'text': _text(m.get('content')) or ' '}])
        elif role == 'assistant':
            blocks = []
            r = m.get('reasoning_content') or m.get('reasoning')
            if r and i > last_user:
                blocks.append({'reasoningContent': {'reasoningText': {'text': r}}})
            t = _text(m.get('content'))
            if t.strip():
                blocks.append({'text': t})
            for tc in m.get('tool_calls') or []:
                a = tc.get('function', {}).get('arguments')
                try:
                    a = json.loads(a or '{}') if isinstance(a, str) else (a or {})
                except ValueError:
                    a = {}
                name, tid = tc['function'].get('name') or '', tc.get('id') or ''
                if not re.fullmatch(r'[a-zA-Z0-9_-]+', name) or not re.fullmatch(r'[a-zA-Z0-9_.:-]+', tid):
                    n_fixed += 1
                    new_id = f'call_fixed_{n_fixed}'
                    fixed_ids[tid] = new_id
                    name, tid = 'invalid_tool_call', new_id
                blocks.append({'toolUse': {'toolUseId': tid, 'name': name, 'input': a if isinstance(a, dict) else {}}})
            add('assistant', blocks or [{'text': ' '}])
        elif role == 'tool':
            tid = m.get('tool_call_id') or ''
            tid = fixed_ids.get(tid, tid) if tid in fixed_ids or re.fullmatch(r'[a-zA-Z0-9_.:-]+', tid) else \
                (list(fixed_ids.values())[-1] if fixed_ids else 'call_fixed_0')
            add('user', [{'toolResult': {'toolUseId': tid, 'content': [{'text': _text(m.get('content')) or ' '}]}}])
    body = {'messages': out}
    if system:
        body['system'] = system
    tools = [{'toolSpec': {'name': t['function']['name'], 'description': t['function'].get('description') or t['function']['name'],
                           'inputSchema': {'json': t['function'].get('parameters') or {'type': 'object', 'properties': {}}}}}
             for t in req.get('tools') or [] if t.get('type') == 'function']
    if tools:
        body['toolConfig'] = {'tools': tools}
    inf = {}
    mx = req.get('max_completion_tokens') or req.get('max_tokens')
    if mx:
        inf['maxTokens'] = int(mx)
    if req.get('temperature') is not None:
        inf['temperature'] = req['temperature']
    if req.get('top_p') is not None:
        inf['topP'] = req['top_p']
    if inf:
        body['inferenceConfig'] = inf
    if _thinking_on(req):
        body['additionalModelRequestFields'] = {'reasoning_effort': 'high'}
    return body


def from_converse(resp, req):
    reasoning, text, calls = [], [], []
    for b in ((resp.get('output') or {}).get('message') or {}).get('content') or []:
        if 'reasoningContent' in b:
            reasoning.append((b['reasoningContent'].get('reasoningText') or {}).get('text', ''))
        elif 'text' in b:
            text.append(b['text'])
        elif 'toolUse' in b:
            tu = b['toolUse']
            calls.append({'id': tu.get('toolUseId'), 'type': 'function',
                          'function': {'name': tu.get('name'), 'arguments': json.dumps(tu.get('input') or {})}})
    msg = {'role': 'assistant', 'content': ''.join(text) or None}
    if reasoning:
        msg['reasoning_content'] = msg['reasoning'] = '\n'.join(reasoning)
    if calls:
        msg['tool_calls'] = calls
    stop = resp.get('stopReason')
    finish = 'tool_calls' if calls else ('length' if stop == 'max_tokens' else 'stop')
    u = resp.get('usage') or {}
    usage = {'prompt_tokens': u.get('inputTokens', 0), 'completion_tokens': u.get('outputTokens', 0),
             'total_tokens': u.get('totalTokens', u.get('inputTokens', 0) + u.get('outputTokens', 0))}
    return {'id': resp.get('ResponseMetadata', {}).get('RequestId', 'converse'), 'object': 'chat.completion',
            'created': int(time.time()), 'model': req.get('model'),
            'choices': [{'index': 0, 'message': msg, 'finish_reason': finish}], 'usage': usage}


_CONTINUATION = re.compile(r"\s*(Let me|Now I|Now let me|I'll|I will|I need to|First,? I)\b")


def bad_summary(req, out):
    """ADK's compaction request (no tools, one user message with the transcript) answered by continuing the agent's
    work instead of summarizing: a short 'Let me ...' line, a raw tool call, or the summarization prompt echoed."""
    if req.get('tools') or not out.get('choices'):
        return False
    ch = out['choices'][0]
    c = (ch.get('message') or {}).get('content') or ''
    return (len(c) < 300 or ch.get('finish_reason') == 'length' or '<tool_call>' in c
            or 'conversation history between a user and an AI agent' in c
            or bool(re.search(r'(?m)^\s*(swe_agent|user|\[model\] said)\s*:', c))  # invented transcript
            or bool(_CONTINUATION.match(c)) or bool(re.search(r"\b(Let me|Now let me)\b", c[:400])))


def gemma_summary(req):
    """Write an ADK compaction summary with Gemma 4 on Bedrock (mantle Responses API) instead of the teacher model:
    on Kaggle the summaries are Gemma's own, and some teachers continue the transcript instead of summarizing it."""
    body = to_responses(req)
    body['model'] = args.summary_model
    code, resp = call_bedrock(body, '/responses', base=f"https://bedrock-mantle.{os.environ['AWS_REGION']}.api.aws/openai/v1")
    return code, resp, (to_chat(resp, req) if code == 200 else resp)


def with_summary_system(req):
    """Compaction requests (no tools) get a system instruction to summarize, not continue (teacher models only)."""
    if req.get('tools') or not args.summary_system:
        return req
    return {**req, 'messages': [{'role': 'system', 'content': args.summary_system}] + list(req.get('messages', []))}


def repair_no_arg_calls(resp, req):
    """Bedrock's Qwen3-Coder parser often returns arguments '{' for tools without parameters (submit_patch,
    get_status); the harness then aborts the attempt. Replace unparsable arguments by '{}' for such tools."""
    no_args = {t['function']['name'] for t in req.get('tools') or [] if t.get('type') == 'function'
               and not ((t['function'].get('parameters') or {}).get('required') or (t['function'].get('parameters') or {}).get('properties'))}
    n = 0
    for c in resp.get('choices') or []:
        for tc in (c.get('message') or {}).get('tool_calls') or []:
            f = tc.get('function', {})
            if f.get('name') in no_args:
                try:
                    json.loads(f.get('arguments') or '{}')
                except ValueError:
                    f['arguments'] = '{}'
                    n += 1
    return n


def malformed_tool_args(resp):
    for c in resp.get('choices') or []:
        for tc in (c.get('message') or {}).get('tool_calls') or []:
            try:
                if not isinstance(json.loads(tc.get('function', {}).get('arguments') or '{}'), dict):
                    return True
            except ValueError:
                return True
    return False


def to_chat_passthrough(req):
    body = {k: v for k, v in req.items() if k not in ('chat_template_kwargs', 'thinking_token_budget')}
    body['model'] = args.model
    body['messages'] = [{k: v for k, v in m.items() if k != 'reasoning'} for m in req.get('messages', [])]
    return body


def call_bedrock(body, route='/responses', base=None):
    data = json.dumps(body).encode()
    for attempt in range(args.max_retries + 1):
        req = urllib.request.Request((base or BASE) + route, data, {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return 200, json.load(r)
        except urllib.error.HTTPError as e:
            payload = e.read().decode(errors='replace')
            if e.code in (429, 500, 502, 503, 504) and attempt < args.max_retries:
                STATS['retries'] += 1
                time.sleep(min(60, 2 ** attempt) * (0.8 + 0.4 * random.random()))
                continue
            return e.code, {'error': {'message': payload[:2000], 'type': 'bedrock_error', 'code': e.code}}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < args.max_retries:
                STATS['retries'] += 1
                time.sleep(min(60, 2 ** attempt))
                continue
            return 502, {'error': {'message': str(e), 'type': 'proxy_error'}}


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip('/').endswith('/models'):
            return self._send(200, {'object': 'list', 'data': [{'id': args.model, 'object': 'model'}]})
        if self.path.startswith('/stats'):
            return self._send(200, STATS)
        return self._send(200, {'status': 'ok'})

    def do_POST(self):
        m = re.match(r'^(?:/w/([\w.-]+))?/v1/chat/completions$', self.path)
        if not m:
            return self._send(404, {'error': {'message': f'unknown path {self.path}'}})
        worker = m.group(1) or 'default'
        req = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        t0 = time.time()
        if args.backend == 'openai':
            up = UPSTREAMS[zlib.crc32(worker.encode()) % len(UPSTREAMS)]
            code, resp = call_bedrock(req, '/chat/completions', base=up)
            out = resp
        elif args.summary_model and not req.get('tools'):
            code, resp, out = gemma_summary(req)
            STATS['summaries_by_summary_model'] = STATS.get('summaries_by_summary_model', 0) + 1
        elif args.backend == 'converse':
            valid = {t['function']['name'] for t in req.get('tools') or [] if t.get('type') == 'function'}
            for _ in range(3 + args.summary_retries):
                code, resp = call_bedrock(to_converse(with_summary_system(req)), f'/model/{args.model}/converse')
                out = from_converse(resp, req) if code == 200 else resp
                if code == 200 and any(tc['function']['name'] not in valid or not tc.get('id')
                                       for tc in out['choices'][0]['message'].get('tool_calls') or []):
                    STATS['invalid_tool_retries'] = STATS.get('invalid_tool_retries', 0) + 1  # mis-parsed tool call
                    continue
                if code == 200 and bad_summary(req, out):
                    STATS['summary_retries'] = STATS.get('summary_retries', 0) + 1
                    continue
                break
        elif args.backend == 'chat':
            body = to_chat_passthrough(with_summary_system(req))
            for _ in range(3 + args.summary_retries):  # resample malformed tool arguments (e.g. '{') or broken summaries
                code, resp = call_bedrock(body, '/chat/completions')
                if code == 200:
                    STATS['repaired_no_arg_calls'] = STATS.get('repaired_no_arg_calls', 0) + repair_no_arg_calls(resp, req)
                if code == 200 and malformed_tool_args(resp):
                    STATS['malformed_retries'] = STATS.get('malformed_retries', 0) + 1
                elif code == 200 and bad_summary(req, resp):
                    STATS['summary_retries'] = STATS.get('summary_retries', 0) + 1
                else:
                    break
            out = resp
            if code == 200:
                for c in out.get('choices', []):
                    if c.get('message', {}).get('reasoning_content'):
                        c['message']['reasoning'] = c['message']['reasoning_content']
        else:
            code, resp = call_bedrock(to_responses(req))
            out = to_chat(resp, req) if code == 200 else resp
        if code == 200 and args.vllm_usage and out.get('usage'):
            u = out['usage']
            try:
                u['bedrock_prompt_tokens'] = u['prompt_tokens']
                u['prompt_tokens'] = vllm_prompt_tokens(req)
                u['total_tokens'] = u['prompt_tokens'] + (u.get('completion_tokens') or 0)
            except Exception as e:  # keep the upstream's count rather than fail the call
                STATS['vllm_usage_errors'] = STATS.get('vllm_usage_errors', 0) + 1
                u['vllm_usage_error'] = repr(e)[:200]
        STATS['requests'] += 1
        if code != 200:
            STATS['errors'] += 1
        sleep = 0.0
        if args.sim_decode_tps > 0 and code == 200:
            sleep = max(0.0, sim_target(worker, out['usage']) - (time.time() - t0))
            STATS['sim_sleep_s'] += sleep
        with _glock:
            lock = _locks.setdefault(worker, threading.Lock())
        with lock, open(os.path.join(args.log_dir, f'{worker}.jsonl'), 'a') as f:
            f.write(json.dumps({'ts': t0, 'dt': round(time.time() - t0, 2), 'sim_sleep': round(sleep, 2), 'status': code, 'request': req,
                                'bedrock_response': resp, 'chat_response': out}) + '\n')
        time.sleep(sleep)
        self._send(code, out)

    def log_message(self, *a):
        pass


print(f'bedrock proxy ({args.backend}) on :{args.port} -> {BASE} model={args.model} sim_decode_tps={args.sim_decode_tps} '
      f'sim_prefill_tps={args.sim_prefill_tps} history_thoughts={args.history_thoughts} vllm_usage={args.vllm_usage or "off"}',
      flush=True)
ThreadingHTTPServer(('0.0.0.0', args.port), H).serve_forever()
