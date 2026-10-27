"""One trajectory's TITO proxy log -> training segments with the exact sampled tokens.

A trajectory is the sequence of model calls in one worker log of tito_proxy.py. Call t has the prompt ids that vLLM
rendered for it (P_t) and the ids it sampled (O_t). Consecutive calls share one segment while the next prompt extends
the segment token for token (P_{t+1} starts with the segment's tokens P_s + O_s + ... + P_t' + O_t). Otherwise
(compaction, a summarizer call, a user message after which the chat template drops earlier thoughts, or a re-render
that differs from the sampled tokens) the next call starts a new segment. A segment is therefore exactly what the
model read and wrote, in order:
  - loss_mask 1 on sampled tokens, 0 on prompt / tool / user tokens and on tokens forced by the thinking budget;
  - rollout_log_probs from the sampler on sampled tokens, 0.0 elsewhere.

segments_merged() instead makes one sample per compaction segment (mode "merged", see its docstring).

usage: python -I tito_samples.py LOG.jsonl [per_call|merged]    # prints the segments of one trajectory
"""
import difflib
import json
import sys

TOOL_RESPONSE = 50  # <|tool_response>: the stop token after a tool call, and the start of the tool result


def load_calls(path):
    calls = []
    for line in open(path):
        r = json.loads(line)
        if r.get('status') == 200 and r.get('tito'):
            calls.append(r['tito'])
    return calls


def segments(calls):
    segs = []
    cur = None
    for c in calls:
        p, o = c['prompt_ids'], c['output_ids']
        forced = set(c.get('forced') or ())
        lps = [0.0 if lp is None else lp for lp in c['output_logprobs']]
        gen_mask = [0 if i in forced else 1 for i in range(len(o))]
        if cur is not None and len(p) >= len(cur['tokens']) and p[:len(cur['tokens'])] == cur['tokens']:
            new = p[len(cur['tokens']):]
            cur['tokens'] += new + o
            cur['mask'] += [0] * len(new) + gen_mask
            cur['logprobs'] += [0.0] * len(new) + lps
            cur['calls'] += 1
        else:
            if cur is not None:
                segs.append(cur)
            cur = {'prompt_len': len(p), 'tokens': p + o, 'mask': [0] * len(p) + gen_mask,
                   'logprobs': [0.0] * len(p) + lps, 'calls': 1}
    if cur is not None:
        segs.append(cur)
    for s in segs:
        n = s['prompt_len']
        s['response_length'] = len(s['tokens']) - n
        s['loss_mask'] = s.pop('mask')[n:]
        s['rollout_log_probs'] = s.pop('logprobs')[n:]
    return segs


MIN_MATCH = 0.95  # a call whose re-render matches less of its sampled output ends its segment


def _align(call, prompt):
    """Map the sampled output of `call` onto its re-render in a later prompt that extends call['prompt_ids'].

    Returns ([(sampled index, prompt position)] for matched tokens, number of sampled tokens to train, matched count).
    The re-render runs from len(call['prompt_ids']) through the first <|tool_response>.
    """
    o, forced = call['output_ids'], set(call.get('forced') or ())
    start = len(call['prompt_ids'])
    end = start
    while end < len(prompt) and prompt[end] != TOOL_RESPONSE:
        end += 1
    pairs = []
    sm = difflib.SequenceMatcher(None, o, prompt[start:end + 1], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            pairs += [(i, start + j) for i, j in zip(range(i1, i2), range(j1, j2)) if i not in forced]
    return pairs, len(o) - len(forced), len(pairs)


def segments_merged(calls, min_match=MIN_MATCH):
    """One sample per compaction segment: the calls between compactions share one token sequence.

    Calls t and t+1 are in one segment when neither is a summarizer call, call t stopped at <|tool_response> (it made
    a tool call), P_{t+1} starts with P_t, and the re-render of O_t inside P_{t+1} matches at least `min_match` of
    O_t. Compaction, a summarizer call or a user message (after which the chat template drops earlier thoughts) breaks
    the prefix condition; an output that the history does not keep (e.g. text that the parser or the template drops)
    breaks the match condition, so that call ends its segment and its output is trained as sampled.
    The sample of a segment is P_last + O_last: exactly what the model read on its last call (vLLM's render), then what
    it sampled. Inside P_last, the output of each earlier call t is in its re-rendered form; the template writes a past
    thought as text + '\\n<channel|>' while the model sampled text<channel|>, so the re-render differs from O_t by
    that newline and by occasional re-tokenization. Re-rendered tokens aligned with O_t get loss_mask 1 and their
    sampling logprob; the others (the newline, re-tokenized pieces) get 0 and are not trained. Tokens forced by the
    thinking budget get 0 as everywhere else.
    """
    groups = []
    for c in calls:
        prev = groups[-1][-1] if groups else None
        if (prev is not None and not prev.get('summarizer') and not c.get('summarizer')
                and prev['output_ids'] and prev['output_ids'][-1] == TOOL_RESPONSE
                and len(c['prompt_ids']) >= len(prev['prompt_ids'])
                and c['prompt_ids'][:len(prev['prompt_ids'])] == prev['prompt_ids']):
            _, n_train, n_match = _align(prev, c['prompt_ids'])
            if n_train == 0 or n_match / n_train >= min_match:
                groups[-1].append(c)
                continue
        groups.append([c])
    segs = []
    for g in groups:
        last = g[-1]
        p_last, o_last = last['prompt_ids'], last['output_ids']
        tokens = p_last + o_last
        mask = [0] * len(tokens)
        lps = [0.0] * len(tokens)
        sampled = matched = 0
        for c in g[:-1]:
            olp = [0.0 if lp is None else lp for lp in c['output_logprobs']]
            pairs, n_train, n_match = _align(c, p_last)
            sampled += n_train
            matched += n_match
            for i, j in pairs:
                mask[j], lps[j] = 1, olp[i]
        forced = set(last.get('forced') or ())
        olp = [0.0 if lp is None else lp for lp in last['output_logprobs']]
        for i in range(len(o_last)):
            if i not in forced:  # the last call's output: the sampled tokens exactly
                mask[len(p_last) + i], lps[len(p_last) + i] = 1, olp[i]
                sampled += 1
                matched += 1
        n = len(g[0]['prompt_ids'])
        segs.append({'prompt_len': n, 'tokens': tokens, 'response_length': len(tokens) - n,
                     'loss_mask': mask[n:], 'rollout_log_probs': lps[n:], 'calls': len(g),
                     'sampled': sampled, 'matched': matched})
    return segs

if __name__ == '__main__':
    calls = load_calls(sys.argv[1])
    segs = segments_merged(calls) if sys.argv[2:] == ['merged'] else segments(calls)
    print(f'{len(calls)} calls -> {len(segs)} segments')
    for i, s in enumerate(segs):
        print(f'  seg {i}: calls={s["calls"]} prompt={s["prompt_len"]} response={s["response_length"]} '
              f'trained={sum(s["loss_mask"])} total={len(s["tokens"])}'
              + (f' matched {s["matched"]}/{s["sampled"]} sampled tokens' if 'matched' in s else ''))
