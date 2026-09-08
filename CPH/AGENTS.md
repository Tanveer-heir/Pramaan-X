# AGENTS.md — Agent Behavior & Operating Rules

> **Strict Protocol**: Operates at par with Anthropic's Claude Code engineering standards.

## Rule Activation
1. **Always refer to the skills library** in `skills/` / `.agent/skills/`:
   - `claude-code-core`: Surgical edits, token stewardship, concise high-signal communication.
   - `codebase-exploration`: Read-only exploration before writing code.
   - `architectural-planner`: Phase-gated planning & Staff Engineer review for complex tasks.
   - `systematic-debugging`: 5-step root-cause analysis (no shotgun debugging).
   - `verification-engineering`: "Verification is the force multiplier" — execute verification before declaring done.
   - `safe-terminal-ops`: Non-interactive CLI commands, PowerShell compliance, destructive command guardrails.
   - `docker-containerization`: Multi-stage builds, compose orchestration, GPU passthrough, container troubleshooting.
   - `backend-management`: API schemas, DB migrations, async worker queues, RBAC, structured logging.
   - `ai-ml-engineering`: Hardware fallback, inference memory hygiene, PDQ hashing, vector search, evaluation metrics.
   - `living-memory-claude`: Keep `CLAUDE.md` and rules updated with newly discovered project constraints.
   - `subagent-orchestration`: Spawn focused subagents for heavy research or adversarial review.

2. **Project Domain**: Section 2 (Provenance Check, Origin Tracing, Investigator Dashboard & Evidence-Grade Reporting) following `section-2-readme (1).md`.
