---
name: subagent-orchestration
description: Guidance for spawning, coordinating, and delegating tasks to subagents. Prevents context bloat by isolating research, planning, execution, and review into focused sub-contexts.
---

# Subagent Orchestration & Parallel Workstreams

A key productivity multiplier in Claude Code workflows is delegating heavy or peripheral tasks to subagents and running parallel workstreams rather than overloading a single conversation context.

---

## 1. When to Delegate to a Subagent

Spawn a subagent when:
* **Heavy Research & File Traversal**: A task requires reading dozens of files or searching deep hierarchies that would consume 20k+ tokens of main conversation context.
* **Independent Investigation**: An exploratory probe or benchmark can run asynchronously while the main agent plans next steps.
* **Adversarial Code Review**: Having an independent agent inspect a proposed diff as a Staff Engineer before merging or committing.
* **Isolated Prototyping**: Testing an experimental implementation without dirtying the main working tree.

---

## 2. Best Practices for Subagent Prompts

1. **Self-Contained Objective**: Provide explicit goal, inputs, and expected output format.
2. **Strict Scoping**: Tell the subagent what is out of bounds (e.g. "Do not modify any files; only return a list of affected functions and dependencies").
3. **Verification Requirement**: Explicitly instruct the subagent how to verify its findings before reporting back.
4. **Structured Output**: Request output in markdown tables, bullet points, or JSON for easy ingestion by the parent agent.

---

## 3. Subagent Lifecycle & Coordination

```text
[Parent Agent] ──> Spawns Research Subagent (read-only)
      │                     │
      │                     └── Searches & reads files
      │                     └── Returns concise summary
      │
      ├──> Synthesizes plan & begins execution
      │
      └──> Spawns Review Subagent to audit diff & verify tests
```

* **No Idle Polling**: The system wakes up reactively when subagents report. Do not loop or busy-wait.
* **Context Cleanliness**: Absorb only the distilled findings into the main agent's working memory.
