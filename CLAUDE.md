# Claude Overlay for open_gororoba

This file is a Claude-specific overlay.
Global policy is canonical in `AGENTS.md`.

## 1. Authority and Scope

- Read `AGENTS.md` first in every session.
- If this file conflicts with `AGENTS.md`, follow `AGENTS.md`.
- Use this file only for Claude runtime behavior and execution style.

## 2. Claude `/init` Checklist

Run this sequence at session start:

1. repo state
   - branch, commit, worktree status
2. policy and objective load
   - open `AGENTS.md`
   - open objective-relevant registry TOML files
3. quality gate baseline
   - plan to run `PYTHONWARNINGS=error make registry`
   - plan to run `PYTHONWARNINGS=error make docs-publish` when docs are touched
4. task plan
   - create explicit step list with dependencies
   - choose what can run in parallel

## 3. Claude Execution Contract

### 3.1 Planning and tracking
- Use granular task tracking for multi-step work.
- Keep implementation tied to registry-first updates.
- Close each task with validation evidence.

### 3.2 Skills usage
- If a listed skill matches the request, use it.
- Read only needed sections of the skill.
- Prefer skill scripts/templates over ad hoc reimplementation.

### 3.3 Agent delegation
- Delegate independent workstreams with explicit ownership.
- Require delegates to report:
  - files touched
  - commands run
  - unresolved risks

### 3.4 MCP orchestration defaults
- Prefer deterministic local tools:
  - filesystem ops via MCP
  - search via ripgrep
  - git state via git tools
- Parallelize independent searches and file reads.

## 4. TOML-first Documentation Rules

- Update authoritative `registry/*.toml` first.
- Treat markdown mirrors as generated outputs.
- Do not hand-edit generated mirrors to fix content.
- Regenerate mirrors after registry changes.

Core commands:

```bash
PYTHONWARNINGS=error make registry
PYTHONWARNINGS=error make docs-publish
make ascii-check
```

## 5. Output and Handoff Standard

For every substantive task, provide:
- what changed
- why it changed
- what was validated
- what remains open

Use precise file references and include command outcomes in summary form.
