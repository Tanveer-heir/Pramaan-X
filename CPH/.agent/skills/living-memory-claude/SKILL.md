---
name: living-memory-claude
description: Best practices for managing persistent context, CLAUDE.md files, and learning from feedback. Keeps instructions lean, actionable, and focused on project-specific gaps.
---

# Living Memory & CLAUDE.md Curation

A core strength of Claude Code workflows is the **living memory loop**: using `CLAUDE.md` to capture project-specific nuances, build rules, and user preferences so the assistant improves persistently over time.

---

## 1. The Rule of the Gap

> [!TIP]
> **Only document "The Gap".**
> Do not fill `CLAUDE.md` with generic software engineering wisdom that modern frontier LLMs already know (e.g. "write clean code", "use meaningful variable names").
> Document **only what the model cannot deduce on its own**:
> - Project-specific architecture patterns.
> - Exact build, test, and lint commands.
> - Non-standard dependencies or package managers.
> - Hard constraints and "Strictly Never Do This" rules.
> - Known quirks or historical bug pitfalls in this codebase.

---

## 2. Size & Conciseness Constraints

* **Ideal Target Length**: 60 to 150 lines.
* **Maximum Limit**: Under 200 lines.
* **Why**: Research and benchmarks prove that massive, multi-hundred-line instruction files cause prompt dilution and rule deprioritization. Keep it concentrated.
* **Modular Extension**: If a project has deep domain guidelines, split them into modular skill files under `skills/<domain>/SKILL.md` rather than bloating the main instruction file.

---

## 3. The Continuous Feedback Loop

Whenever the user provides a correction:
1. **Acknowledge and Apply**: Fix the immediate issue surgically.
2. **Identify the Underlying Rule**: Formulate the concise invariant that would have prevented the mistake.
3. **Persist the Learning**: Update `CLAUDE.md` or the appropriate skill file with the new rule.
4. **Confirm**: Let the user know the rule has been committed to persistent memory.

---

## 4. Standard CLAUDE.md Structure Template

```markdown
# Project Guidelines & Memory

## Build & Test Commands
- Run test suite: `pytest tests/ -v`
- Run single test: `pytest tests/test_provenance.py -k "test_c2pa" -v`
- Type checking: `mypy src/`
- Linting: `ruff check .`

## Architecture & Code Conventions
- Pipeline flow follows: Detection -> Provenance Check -> Origin Tracing -> Dashboard.
- Immutable data structures: Use frozen dataclasses or Pydantic models for evidence objects.
- Error handling: Never raise generic `Exception`; use typed domain exceptions in `src/exceptions.py`.

## Strict Guardrails (Don'ts)
- Do NOT mutate Section 1's manipulation verdicts.
- Do NOT use interactive terminal prompts.
- Do NOT commit `.env` or credentials.
```
