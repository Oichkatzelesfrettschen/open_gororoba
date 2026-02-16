# agents.md

## Workspace Coordination Agents

- Primary coordinator: Codex (task decomposition, execution, and documentation)
- Domain reviewers: user + future contributors

## Roles
- Synchronization Agent: handles origin pull/push and branch tracking.
- Documentation Agent: maintains `TODOWRITE.md`, `ROADMAP_SYNTHESIS.md`, and `GEMINI.md`.
- Audit Agent: executes scripted checks, emits `logs/*.md`.
- Compliance Agent: tracks license and requirement alignment.

## Priority Rules
1. Preserve evidence for every decision in `logs/` and `~/Documents` backups.
2. Non-destructive workflow by default.
3. Every blocking issue must include mitigation and owner before merge.
