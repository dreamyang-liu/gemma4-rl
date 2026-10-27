---
name: repo-explore
description: One-call repository exploration (occurrences, full definitions, related tests, exception classes), multi-location code viewer, and a pre-submit self check that compiles, imports and runs related existing tests while separating regressions from pre-existing failures.
---

# repo-explore

Call scripts directly with run_skill_script (no need to load this skill first). The first call also installs the scripts as /tmp/repo-explore/*.py for use with run_command. Script output is NOT
limited to 5000 characters, so one call replaces many grep/read_file calls.

## scripts/explore.py — start every task with this
run_skill_script with arguments {"skill_name": "repo-explore", "file_path": "scripts/explore.py", "args": {"terms": "Console.print,soft_wrap,escape"}}
- terms: 2-6 comma-separated identifiers, API names, option names or error-message fragments from the issue.
  Class.method is supported. Prefer specific names over common words.
- Returns: hits in source/tests/docs, full source of matching definitions, related existing test files with
  their test names, the package's own exception/warning classes, and the CHANGELOG head.

## scripts/show.py — read several code locations at once
run_skill_script with arguments {"skill_name": "repo-explore", "file_path": "scripts/show.py", "args": {"targets": "rich/console.py:Console.print, rich/text.py:1200-1290, get_typed_signature"}}
- target forms: path:START-END, path:Symbol, path:Class.method, bare Symbol, or path.

## scripts/check.py — run before submit_patch
run_skill_script with arguments {"skill_name": "repo-explore", "file_path": "scripts/check.py", "args": {"repro": "python3 /tmp/repro.py"}}
- Compiles every changed file, imports changed modules, runs existing tests for the modules you touched
  (add more with "tests": "tests/test_x.py,tests/test_y.py::test_z"), and runs your repro.
- Reports REGRESSIONS caused by your change (you must fix these) separately from pre-existing failures (ignore).
