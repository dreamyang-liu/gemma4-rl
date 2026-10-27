# LLM filtering of rebench_rl_train_portable_v3.jsonl (1834 SWE-rebench-v2 Python tasks)

16 Sonnet agents judged every task against the 106 dev tasks (rubric.md, dev_reference.md) from markdown cards
(md_XX/part_*.md: problem statement <= 2000 chars, gold patch <= 1300, test patch <= 1000 chars, publisher flags).
- out_XX.jsonl: per-shard verdicts (similarity / solvable / spec_clear 0-3, verdict, drop_reasons, category, note)
- verdicts_all.jsonl: merged, verdict recomputed from the scores (keep = all >= 2; maybe = one score 1), plus
  duplicate_issue (same repo and issue text, or gold patches overlapping >= 80%; the best-scored copy is kept)
- rebench_keep.jsonl / rebench_keep_maybe.jsonl: source rows + llm_filter, duplicates removed
- v1_out_XX.jsonl: a first, shallow pass (agents saw only ~300-500 chars per task because Read truncates long
  JSON lines); not used.
Built by env/rebench_prep_md.py and env/rebench_merge.py.
