---
name: safe-terminal-ops
description: Safe terminal and CLI operations guide. Enforces non-interactive execution, platform-aware shell syntax (PowerShell vs Bash), git cleanliness, and destructive command guardrails.
---

# Safe Terminal Operations & Command Line Mastery

This skill governs all terminal executions to ensure safety, idempotency, non-interactive execution, and platform compatibility (specifically on Windows PowerShell environments).

---

## 1. Safety Guardrails

### A. Forbidden / High-Risk Operations
* **Destructive Deletion**: Never run unrestricted recursive deletes (`rm -rf`, `Remove-Item -Recurse -Force *`) on repository root or ambiguous directories.
* **Unstaged Data Loss**: Never run `git reset --hard` or `git checkout .` without first checking `git status` to ensure user work is not discarded.
* **Force Pushes**: Never run `git push --force` or `--force-with-lease` without explicit user instruction.
* **Interactive Blockers**: Never run commands that stall waiting for interactive keyboard input without non-interactive flags (e.g., `apt-get`, interactive prompts, editors like `nano` or `vim`).

---

## 2. Platform-Aware Shell Execution (Windows PowerShell)

In this environment, commands run in **PowerShell** on Windows. Observe these syntax rules:

| Operation | Unix Bash | Windows PowerShell |
| :--- | :--- | :--- |
| **List Directory** | `ls -la` | `Get-ChildItem -Force` |
| **Check File Content** | `cat file.txt` | `Get-Content file.txt` or `type file.txt` |
| **Grep / Search** | `grep -rn "pattern" .` | `Select-String -Path ...` or prefer agent search tools (`grep_search`) |
| **Remove Item** | `rm -f file.txt` | `Remove-Item -Force file.txt` |
| **Check Env Var** | `echo $VAR` | `$env:VAR` |
| **Command Chaining** | `cmd1 && cmd2` | `cmd1; if ($?) { cmd2 }` (or `&&` in modern PowerShell 7+) |
| **Path Separators** | `/path/to/file` | Forward slashes work in most tools, but use `\` for native cmd/powershell paths when required |

---

## 3. Non-Interactive Execution Rules

Always pass flags that suppress interactive prompts:
* **Python**: `python -u` (unbuffered stdout), `pip install -q --no-input`
* **Node.js**: `npm install --no-audit --no-fund --yes`, `npx -y`
* **Git**: Pass `-m "message"` for commits; never trigger interactive commit editor.

---

## 4. Git Worktree & Branch Hygiene

Claude Code power users rely on git worktrees and branch hygiene to parallelize work and prevent contamination:
1. **Pre-Command Status**: Check `git status -s` before significant file modifications.
2. **Atomic Commits**: Group related changes logically. Use conventional commit messages: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
3. **Inspect Diffs**: Before committing, inspect `git diff` to ensure no stray debug code, `.env` secrets, or unintended formatting changes are included.
