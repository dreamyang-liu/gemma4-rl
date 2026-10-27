---
name: openhands
description: Shell and file editor for the repository in /workspace (run bash commands; view files with line numbers or directories, create files, replace or insert text, undo the last edit).
license: MIT (str_replace_editor.py is a port of openhands-aci 0.3.1, Copyright (c) All Hands AI)
---
Run the scripts with `run_skill_script`, passing the parameters in `args`.

`scripts/execute_bash.py` runs a shell command:

- `{"command": "cd /workspace && python -m pytest tests/test_x.py -q"}`; optionally `"timeout"` in seconds (at most 290).
- Every command starts in a fresh `/bin/bash -c` in `/workspace`; the directory and variables do not carry over.
- stderr is merged into the output. Output over 30000 characters keeps only its first and last 15000.

`scripts/str_replace_editor.py` views and edits files:

- `{"command": "view", "path": "/workspace/pkg/mod.py"}`; optionally `"view_range": [start, end]` (1-indexed, `end` = -1 for the end of the file). A directory path lists files up to 2 levels deep.
- `{"command": "create", "path": "/workspace/new_file.py", "file_text": "..."}` creates a file; it fails if the file exists.
- `{"command": "str_replace", "path": "...", "old_str": "...", "new_str": "..."}` replaces one exact, unique occurrence of `old_str`.
- `{"command": "insert", "path": "...", "insert_line": N, "new_str": "..."}` inserts text after line N.
- `{"command": "undo_edit", "path": "..."}` reverts the last edit to the file.

Paths must be absolute.
