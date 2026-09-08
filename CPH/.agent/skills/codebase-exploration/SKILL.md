---
name: codebase-exploration
description: Emulates Claude Code's specialized 'Explore' subagent for fast, read-only codebase navigation, architecture mapping, pattern discovery, and dependency tracing.
---

# Codebase Exploration (Explore Mode)

This skill provides the operational protocol of Claude Code's internal **Explore Agent**, dedicated to fast, read-only codebase discovery and orientation without altering files.

---

## 1. Principles of Exploration

1. **Strictly Read-Only**: During the exploration phase, NEVER create, edit, move, or delete files. No build side-effects or mutations.
2. **Breadth-First to Depth-First**:
   - Step 1: Scan repository layout, directory trees, configuration files (`package.json`, `requirements.txt`, `pyproject.toml`, `Cargo.toml`, `Makefile`, etc.).
   - Step 2: Use glob/find to map file locations relevant to the feature or bug.
   - Step 3: Grep for exact symbols, function signatures, class names, or error strings.
   - Step 4: Read specific line ranges of the target files to understand context and edge cases.
3. **Pattern Replication**: Always locate existing, working examples within the codebase before writing new modules. Replicate:
   - Error handling patterns (custom exceptions vs return codes vs Result types).
   - Logging and telemetry conventions.
   - Type annotation styles.
   - File structuring and modularization standards.

---

## 2. Investigation Playbook

### A. Locate Files Rapidly
* Use pattern-based globbing (`find_by_name`) with smart extensions and directory scoping.
* Ignore noisy directories (`node_modules`, `.git`, `dist`, `__pycache__`, `target`, `venv`).

### B. Precision Grepping
* When searching for function definitions:
  - Regex: `def calculate_hash\b` or `class ImageProvenance` or `function verifySignature`.
* When searching for usages:
  - Exact literal grep across source directories.
* When searching for errors:
  - Grep for the exact error message or token from the traceback.

### C. Context Slicing
* Do NOT read 1,000 lines at once when you only need to see the function implementation.
* Use `view_file` with explicit `StartLine` and `EndLine` to capture the function, its decorators, docstrings, and surrounding 10-20 lines.

---

## 3. Architecture Mapping Checklist

Before proposing or executing code changes, ensure you can answer:
- [ ] What is the entry point for this data/request?
- [ ] What data structures or schemas flow into and out of this component?
- [ ] What external libraries or internal helpers are already available for this task?
- [ ] Are there existing unit tests covering this component, and where do they live?
- [ ] What are the potential side effects on upstream and downstream modules?
