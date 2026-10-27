"""Build compact task cards for LLM filtering of rebench_rl_train_portable_v3.jsonl against the 106 dev tasks.
usage: python -I rebench_prep.py SRC.jsonl DEV.jsonl OUT_DIR [--shards 16]"""
import json, os, re, sys

src, dev_path, out = sys.argv[1:4]
n_shards = int(sys.argv[sys.argv.index('--shards') + 1]) if '--shards' in sys.argv else 16


def pstats(p):
    files = re.findall(r'^\+\+\+ (?:b/)?(\S+)', p, re.M)
    ch = sum(1 for l in p.splitlines() if l[:1] in '+-' and not l.startswith(('+++', '---')))
    return files, ch


def clip(s, n):
    s = (s or '').replace('\r', '')
    return s if len(s) <= n else s[:n] + f'\n...[{len(s) - n} more chars]'


dev = [json.loads(l) for l in open(dev_path)]
with open(os.path.join(out, 'dev_reference.md'), 'w') as f:
    f.write('# The 106 local dev tasks (same distribution as the hidden Kaggle tasks)\n\n')
    for r in dev:
        files, ch = pstats(r['patch'])
        tfiles, _ = pstats(r['test_patch'])
        f.write(f"## {r['instance_id']} ({r['repo']}) gold: {len(files)} file(s), {ch} changed lines: {', '.join(files)}; "
                f"tests: {', '.join(tfiles)}\n{clip(r['problem_statement'], 600)}\n\n")

rows = [json.loads(l) for l in open(src)]
cards = []
for r in rows:
    files, ch = pstats(r['patch'])
    tfiles, _ = pstats(r['test_patch'])
    ks, lm = r.get('kaggle_screening', {}), r.get('meta', {}).get('llm_metadata', {})
    cards.append({
        'instance_id': r['instance_id'], 'repo': r['repo'], 'created_at': r.get('created_at'),
        'publisher_difficulty': r.get('screening_difficulty'), 'pr_categories': lm.get('pr_categories'),
        'intent_completeness': lm.get('intent_completeness'), 'llm_detected_issues': lm.get('detected_issues'),
        'test_alignment_issues': lm.get('test_alignment_issues'),
        'gold_files': files, 'gold_changed_lines': ch, 'test_files': tfiles,
        'n_fail_to_pass': len(r.get('FAIL_TO_PASS') or []), 'fail_to_pass_sample': (r.get('FAIL_TO_PASS') or [])[:5],
        'test_run_s': round(ks.get('publisher_test_run_s') or 0),
        'nearest_public_problem_jaccard': ks.get('nearest_public_problem_jaccard'),
        'kaggle_overlap_types': ks.get('kaggle_overlap_types'),
        'has_interface_spec': bool(r.get('interface')),
        'problem_statement': clip(r['problem_statement'], 2200),
        'gold_patch': clip(r['patch'], 1400),
    })
per = -(-len(cards) // n_shards)
for i in range(n_shards):
    with open(os.path.join(out, f'shard_{i:02d}.jsonl'), 'w') as f:
        for c in cards[i * per:(i + 1) * per]:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
print(len(cards), 'cards in', n_shards, 'shards of', per)
