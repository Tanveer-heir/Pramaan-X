---
name: verification-engineering
description: Enforces Boris Cherny's 'Verification as Force Multiplier' doctrine. Mandates concrete execution checks, automated test runs, sanity probes, and verification scripts before declaring any task complete.
---

# Verification Engineering: The Force Multiplier

In the words of Claude Code's creator Boris Cherny: **"Verification is the force multiplier."**
An AI coding assistant without verification can only evaluate if code *"looks done"*. A Staff-level assistant verifies that code **actually works**.

---

## 1. The Verification Mandate

> [!CRITICAL]
> **Never report that a task is finished without running a verification step.**
> If code was added or modified, it must be executed or tested through a concrete command.

---

## 2. Hierarchy of Verification

Depending on what exists in the repository, apply the highest available tier of verification:

```text
Tier 1: Comprehensive Automated Test Suite (pytest, npm test, cargo test, go test)
   │
Tier 2: Static Typing & Linting (mypy, pyright, tsc, ruff, eslint)
   │
Tier 3: Standalone Verification Script (targeted driver exercising modified functions)
   │
Tier 4: Dry-Run / CLI Smoke Test (running the CLI tool with --help or sample input)
```

---

## 3. Creating Ephemeral Verification Scripts

When the project lacks existing unit tests for a newly implemented feature or bugfix, do NOT ask the user to test it manually. Instead, build a verification script:

1. Create a lightweight test driver in a temporary or scratch location (e.g. `scratch/verify_<feature>.py` or `tests/test_<feature>.py`).
2. Instantiate the modified classes/functions with:
   - Happy path input.
   - Edge case inputs (empty lists, empty strings, null/None, oversized buffers).
   - Known failing inputs (to verify error handling and custom exceptions).
3. Assert expected outputs.
4. Execute via `run_command` and inspect the output.
5. If clean, either persist as a permanent unit test in the test suite or document the successful test execution in the summary.

---

## 4. Pre-Completion Verification Checklist

Before concluding your response:
- [ ] Has the modified file been syntax-checked and linted?
- [ ] Have all unit tests covering the affected module passed?
- [ ] Have regression tests passed (did we break existing tests)?
- [ ] Were any temporary debug logs, test files, or scratch scripts cleaned up (or properly filed)?
- [ ] Is the terminal exit code `0`?
