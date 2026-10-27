---
name: swe-tools
description: Repository tools for this task - run_command, read_file, write_file and str_replace. Every result reports progress (tool call n/30, seconds used) and a budget_warning when you must start editing or submit.
---

# swe-tools

Call these scripts with run_skill_script; there is no need to load this skill first. Arguments of run_skill_script:
skill_name = swe-tools, file_path = scripts/<script>, args = the script's parameters.

- scripts/run_command.py - args: command. Runs bash in the repository root; output is cut at 5000 characters.
  Use $TMPDIR for scratch files.
- scripts/read_file.py - args: filepath, start_line, end_line. At most 150 lines per call.
- scripts/str_replace.py - args: filepath, old_string, new_string. old_string must match one place in the file
  (whitespace differences are tolerated). Returns the diff.
- scripts/write_file.py - args: filepath, content. Creates or overwrites a whole file; only for new files.

Each call counts as one tool call. Every result starts with a progress line; if it starts with a budget_warning,
follow it with your very next call.
