"""Miles rollout for SFT on pre-tokenized rows (--rollout-function-path sft_rollout.generate_rollout).

Each prompt-data row carries metadata {src, row, pad}: a pointer into a tokenized parquet file with input_ids and
unshifted labels (labels[i] == input_ids[i] where token i is a target, else -100). Miles' contract is
loss_mask[j] == 1 iff tokens[P + j] is supervised, with P = len(tokens) - response_length >= 1; the next-token
shift is done inside Miles. Padding rows get a 2-token sequence with an all-zero mask.
"""
import pyarrow.parquet as pq

from miles.utils.types import Sample

_TABLES = {}
_PAD = ([2, 2], [0])


def _load(src, row):
    t = _TABLES.get(src)
    if t is None:
        t = _TABLES[src] = pq.read_table(src, columns=['input_ids', 'labels'])
    return t['input_ids'][row].as_py(), t['labels'][row].as_py()


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    groups = data_buffer.get_samples(args.rollout_batch_size)
    for (s,) in groups:
        m = s.metadata
        if m.get('pad'):
            ids, mask = _PAD[0], [0] * len(_PAD[0])
            first = 1
        else:
            ids, lab = _load(m['src'], m['row'])
            mask = [int(x != -100) for x in lab]
            first = mask.index(1) if any(mask) else len(ids) - 1
            assert first >= 1
        s.tokens = ids
        s.response_length = len(ids) - first
        s.loss_mask = mask[first:]
        s.reward = 0
        s.status = Sample.Status.COMPLETED
    return groups
