<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/entrypoint_docs.toml -->

# Gemini Overlay for open_gororoba

This file is a Gemini-specific overlay.
Global policy is canonical in `AGENTS.md`.

## 1. Authority and Scope

- Start with `AGENTS.md`.
- Use this file for Gemini runtime specifics only.
- If conflict exists, `AGENTS.md` is authoritative.

## 2. Gemini `/init` Checklist

1. capture repo state
   - branch, commit, worktree status
2. load canonical workflow
   - open `AGENTS.md`
   - open relevant `registry/*.toml` sources
3. establish quality-gate plan
   - `PYTHONWARNINGS=error make registry-data`
   - `PYTHONWARNINGS=error make registry`
   - `PYTHONWARNINGS=error make docs-publish` for docs pipeline work
4. build execution plan
   - define tasks, dependencies, and validation points

## 3. Gemini Execution Contract

### 3.1 Planner and tracker discipline
- Maintain explicit stepwise plan for multi-stage work.
- Track status transitions clearly.
- Keep todo and roadmap updates TOML-first.

### 3.2 Skills and tools
- Use matching skills when available.
- Prefer deterministic local MCP tools for file/search/git operations.
- Use MCP bash tools (`run`, `run_background`) instead of the retired `run_shell_command`.
- Run parallel operations only when tasks are independent.

### 3.3 Delegation and integration
- Delegate work by subsystem boundaries.
- Require delegated reports with:
  - touched files
  - checks executed
  - blocking issues
- Integrate centrally and re-validate before completion.

## 4. TOML-first Documentation and Mirror Policy

- Canonical source is `registry/*.toml`.
- Markdown outputs are generated mirrors.
- Edit source TOML, then regenerate.

Core commands:

```bash
PYTHONWARNINGS=error make registry-data
PYTHONWARNINGS=error make registry
PYTHONWARNINGS=error make docs-publish
make ascii-check
```

## 5. Handoff Standard

Every final handoff should include:
- implemented changes
- validation evidence
- unresolved risks
- next recommended steps
