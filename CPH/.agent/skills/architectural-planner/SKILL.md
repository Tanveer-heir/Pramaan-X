---
name: architectural-planner
description: Emulates Claude Code's Plan Mode and phase-gated execution. Enforces structured design, adversarial staff-engineer critique, boundary condition analysis, and verification plans before writing code.
---

# Architectural Planner (Plan Mode & Ultraplan)

This skill guides the planning of non-trivial tasks, complex refactors, and multi-component implementations, adhering to Claude Code's **Plan Mode** discipline.

---

## 1. When to Trigger Plan Mode

Activate this skill whenever a task:
- Involves changes across multiple files or modules.
- Modifies public APIs, schemas, or database models.
- Introduces new third-party dependencies or external integrations.
- Involves significant ambiguity or algorithmic complexity.
- Has failed twice during execution and requires an architectural reset.

---

## 2. The 5-Phase Planning Framework

### Phase 1: Problem Definition & Scope Boundaries
* **Objective**: Define in 1–2 crisp sentences what success looks like.
* **Non-Goals**: Explicitly state what is out of scope to prevent scope creep.
* **Constraints**: Note performance, backward compatibility, dependency, and platform limitations (e.g. Windows PowerShell vs Linux bash).

### Phase 2: Codebase Context & Existing Patterns
* Reference existing files and code lines that govern this domain.
* Identify the exact data types and schemas currently in use.
* List any precedents or similar implementations in the repository to mimic.

### Phase 3: Technical Design & Interface Specifications
* Detail exact function signatures, class interfaces, or data classes to be created or modified.
* Specify input/output schemas (JSON, Pydantic, Protobuf, Dataclasses).
* Address error states, timeouts, edge cases (e.g. empty lists, null metadata, corrupt inputs).

### Phase 4: Step-by-Step Execution Plan
Break down the implementation into atomic, sequential milestones:
1. Milestone 1: Data structures / models / schemas.
2. Milestone 2: Core algorithm / business logic implementation.
3. Milestone 3: Integration into pipeline / entry point.
4. Milestone 4: Test cases & verification.

### Phase 5: Verification & Rollback Protocol
* Define exact commands to test every milestone (e.g. `pytest tests/test_provenance.py -v`).
* Define failure criteria: what signals that the approach is flawed and must be rolled back?

---

## 3. The "Staff Engineer" Adversarial Review

Before executing any plan, subject it to an adversarial self-critique:
- **Simplicity Test**: Is this the simplest solution that works? Can we eliminate 30% of this complexity?
- **Failure Modes**: What happens when an external service times out, an input file is corrupted, or a required field is missing?
- **Regression Risk**: Does this change alter behavior expected by downstream modules?
- **Verification Viability**: Can we definitively prove this works without human manual intervention?

---

## 4. Pivot Rule

> [!CRITICAL]
> If execution hits an unexpected architectural barrier, **DO NOT hammer broken changes or guess fixes**.
> Immediately stop, revert unverified edits, switch back to Plan Mode, update the plan with the new findings, and align before continuing.
