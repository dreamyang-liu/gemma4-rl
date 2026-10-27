"""Minimal repro: Amazon Bedrock (bedrock-mantle OpenAI-compatible endpoint) drops Gemma 4's earlier reasoning in
multi-turn tool use.

Turn 1 asks google.gemma-4-31b (reasoning.effort=high, tool_choice=required) to reason and call a tool. Turn 2 sends
the tool result back in several ways and prints usage.input_tokens. If prior reasoning were part of the prompt, the
variants that carry it (B, C, D, F) would cost about as many tokens as E (the same text sent as assistant content);
instead they cost the same as A (no reasoning at all). Gemma 4's chat template keeps thoughts of model turns after
the last user message (interleaved thinking), so the model should see them.

usage: AWS_REGION=us-west-2 AWS_BEARER_TOKEN_BEDROCK=... python3 bedrock_reasoning_dropped_repro.py
"""
import json
import os
import urllib.error
import urllib.request

BASE = f"https://bedrock-mantle.{os.environ.get('AWS_REGION', 'us-west-2')}.api.aws/openai/v1"
HEADERS = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['AWS_BEARER_TOKEN_BEDROCK']}
MODEL = 'google.gemma-4-31b'


def post(route, body):
    req = urllib.request.Request(BASE + route, json.dumps(body).encode(), HEADERS)
    try:
        return json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        raise SystemExit(f'{route}: HTTP {e.code} {e.read().decode()[:400]}')


TOOLS = [{'type': 'function', 'name': 'run_command', 'description': 'Run a shell command',
          'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}]
USER = {'role': 'user', 'content': 'The file data.csv has a column price. Plan carefully how to compute the median '
                                   'price robustly (headers, quoting, missing values), then run ONE shell command.'}

r1 = post('/responses', {'model': MODEL, 'input': [USER], 'tools': TOOLS, 'tool_choice': 'required', 'store': True,
                         'reasoning': {'effort': 'high'}, 'include': ['reasoning.encrypted_content'],
                         'max_output_tokens': 4000})
reasoning = [o for o in r1['output'] if o['type'] == 'reasoning']
calls = [o for o in r1['output'] if o['type'] == 'function_call']
assert reasoning and calls, 'turn 1 must return reasoning and a function call; rerun'
text = ''.join(c.get('text', '') for o in reasoning for c in o.get('content') or [])
print(f"turn 1: {r1['id']}, reasoning {len(text)} chars, usage {r1['usage']}")
result = {'type': 'function_call_output', 'call_id': calls[0]['call_id'], 'output': '12.5'}


def turn2(label, items, **extra):
    r = post('/responses', {'model': MODEL, 'input': items, 'tools': TOOLS, 'reasoning': {'effort': 'high'},
                            'max_output_tokens': 16, **extra})
    print(f"{label:<62} input_tokens={r['usage']['input_tokens']}")


turn2('A  no reasoning item', [USER] + calls + [result])
turn2('B  reasoning items exactly as returned by turn 1', [USER] + reasoning + calls + [result])
turn2('C  reasoning items without id', [USER] + [{k: v for k, v in o.items() if k != 'id'} for o in reasoning]
      + calls + [result])
turn2('D  previous_response_id (store=true), tool output only', [result], previous_response_id=r1['id'])
turn2('E  same reasoning text as assistant content (control)', [USER, {'role': 'assistant', 'content': text}]
      + calls + [result])

chat_tools = [{'type': 'function', 'function': {k: v for k, v in TOOLS[0].items() if k != 'type'}}]
call = {'id': 'c1', 'type': 'function', 'function': {'name': 'run_command', 'arguments': calls[0]['arguments']}}
for label, extra in (('F  chat/completions, assistant reasoning_content + reasoning', {'reasoning_content': text, 'reasoning': text}),
                     ('G  chat/completions, no reasoning (control)', {})):
    msgs = [{'role': 'user', 'content': USER['content']}, {'role': 'assistant', 'content': None, 'tool_calls': [call], **extra},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': '12.5'}]
    r = post('/chat/completions', {'model': MODEL, 'messages': msgs, 'tools': chat_tools, 'max_tokens': 16})
    print(f"{label:<62} prompt_tokens={r['usage']['prompt_tokens']}")
