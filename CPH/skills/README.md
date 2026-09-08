# Claude Code Skills Library

This library equips agents and developers in this repository with the exact workflows, architectural discipline, and operational rigor of Anthropic's **Claude Code** (as architected by Boris Cherny and vetted by the developer community across Twitter/X, Reddit, and GitHub).

---

## Skills Catalog

### 1. Claude Code Core & Agentic Execution
| Skill Name | Purpose | When to Refer |
| :--- | :--- | :--- |
| [`claude-code-core`](./claude-code-core/SKILL.md) | Foundational operational tenets, surgical editing, token discipline, and loop orchestration. | Always active. Governs all actions, tone, and tool discipline. |
| [`codebase-exploration`](./codebase-exploration/SKILL.md) | Emulates the Claude Code `Explore` subagent for fast, read-only navigation, grep search, and pattern discovery. | When investigating a new codebase, locating functions, or understanding architectures. |
| [`architectural-planner`](./architectural-planner/SKILL.md) | Claude Code `Plan Mode` & phase-gated execution with adversarial Staff Engineer self-critique. | Before executing complex, multi-file changes or non-trivial refactors. |
| [`systematic-debugging`](./systematic-debugging/SKILL.md) | Anti-shotgun debugging protocol: Reproduce -> Isolate -> Hypothesize -> Prove -> Surgical Fix. | Whenever diagnosing an error, failed test, or unexpected runtime behavior. |
| [`verification-engineering`](./verification-engineering/SKILL.md) | "Verification is the force multiplier" (tests, linter, builds, smoke tests, verification scripts). | Mandatory before marking any task, feature, or fix as complete. |
| [`safe-terminal-ops`](./safe-terminal-ops/SKILL.md) | Non-interactive execution, Windows PowerShell compatibility, git hygiene, and destructive action guardrails. | When running shell commands, build tools, or git operations. |
| [`living-memory-claude`](./living-memory-claude/SKILL.md) | Curating `CLAUDE.md`, focusing on "The Gap", and persisting learnings from user corrections. | When updating project instructions or receiving user feedback. |
| [`subagent-orchestration`](./subagent-orchestration/SKILL.md) | Parallel workstreams, task decomposition, and context isolation via specialized subagents. | When handling massive exploration, independent research, or pre-commit reviews. |

### 2. Infrastructure, Backend & Machine Learning
| Skill Name | Purpose | When to Refer |
| :--- | :--- | :--- |
| [`docker-containerization`](./docker-containerization/SKILL.md) | Multi-stage Dockerfile design, layer caching, docker-compose orchestration, GPU passthrough, and container troubleshooting. | When building, containerizing, debugging, or deploying microservices and pipelines. |
| [`backend-management`](./backend-management/SKILL.md) | FastAPI/REST API contracts, Alembic database migrations, Redis/Celery async task queues, RBAC, and structured telemetry. | When designing APIs, managing databases, handling background workers, or securing endpoints. |
| [`ai-ml-engineering`](./ai-ml-engineering/SKILL.md) | Model inference pipelines (PyTorch, ONNX Runtime), perceptual hashing (PDQ), FAISS vector search, batching, and quantitative evaluation. | When developing AI/ML detection, perceptual matching, feature extraction, or model evaluation. |

---

## Operating Rule

Any agent operating in this repository **must strictly refer to and uphold the guidelines in these skills** to ensure maximum engineering precision, zero code hallucinations, surgical diffs, and verified outcomes.
