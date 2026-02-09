<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/entrypoint_docs.toml -->

# Agent and Contributor Operating Manual

This is the canonical policy and operating contract for all assistants and
human contributors in this repository.

If any instruction in `CLAUDE.md` or `GEMINI.md` conflicts with this file,
`AGENTS.md` wins.

## 1. File Roles and Authority

| File | Role | Edit policy |
|---|---|---|
| `AGENTS.md` | Global policy and repo `/init` procedure | Canonical, tracked |
| `CLAUDE.md` | Claude-specific overlay | Tracked, must defer to `AGENTS.md` |
| `GEMINI.md` | Gemini-specific overlay | Tracked, must defer to `AGENTS.md` |
| `README.md` | Human-facing repo entrypoint | Tracked |
| `registry/*.toml` | Canonical machine-readable knowledge state | Canonical, tracked |
| `docs/**/*.md` | Generated mirrors and narrative exports | Generated, not canonical |

## 2. Repo Identity and Scope

open_gororoba is a research workbench centered on reproducible algebra,
physics, and data workflows with a Rust-first architecture.

Primary code surfaces:
- Rust workspace under `crates/`
- Python support layer under `src/gemini_physics/`
- Verification scripts under `src/verification/`
- Registry authority under `registry/`

Research claims are hypotheses unless backed by:
1. first-party source citation, and
2. reproducible in-repo verification path.

## 3. Non-Negotiable Hard Rules

### 3.1 ASCII-only
- All repo-authored code and docs must be ASCII-only.
- Exception: immutable transcripts under `convos/`.
- Validate with: `make ascii-check`

### 3.2 Warnings-as-errors
- Python checks and tests must run with `PYTHONWARNINGS=error`.
- Rust lint must pass with `-D warnings`.
- Treat all warnings as failures.

### 3.3 TOML-first documentation
- Authoritative state lives in `registry/*.toml`.
- Markdown in `docs/`, `reports/`, and `data/artifacts/` is generated or mirror
  output and is never the authoritative source.
- Do not treat mirrors as authoritative.

### 3.4 Source-first claims discipline
- Do not treat narrative text as proof.
- Every claim should be falsifiable and tied to verifiable evidence.

### 3.5 Provenance and reproducibility
- No large opaque binary commits.
- Track external data provenance and hashes.
- Tests must not rely on network.

## 4. `/init` Procedure (Mandatory Session Startup)

Every agent session should follow this exact initialization flow.

1. Establish repo context
   - confirm cwd is repo root
   - capture branch, commit, and worktree status

2. Detect pre-existing local changes
   - list modified, staged, untracked files
   - never silently revert existing user changes

3. Load canonical policy and active objectives
   - read `AGENTS.md`
   - read current objective-relevant registry files

4. Validate toolchain readiness
   - check Python and Rust toolchain availability
   - confirm `make` targets are discoverable

5. Build a task plan before wide edits
   - define scoped tasks with dependencies
   - identify owner for each task when delegating to sub-agents

6. Choose execution lane
   - lane A: code and tests
   - lane B: registry and docs pipeline
   - lane C: data/provenance workflows

7. Execute with quality gates
   - run targeted checks as changes land
   - run final validation commands before commit

8. Close loop
   - update registry state first
   - regenerate mirrors
   - summarize outcomes and residual risk

## 5. Skills and Agent Delegation Protocol

### 5.1 Skills
- If a task matches an available skill, use it.
- Read only required sections of the skill file.
- Reuse skill scripts/assets instead of rewriting from scratch.
- If skill is unavailable, report and use best fallback.

### 5.2 Delegation
- Delegate only when it improves speed or quality.
- Assign explicit ownership by file or subsystem.
- Keep one integrator agent responsible for final merge and validation.
- Parallelize independent workstreams (search, audit, static reads).

### 5.3 Ownership handoff format
- Scope: what this agent owns
- Inputs: files and assumptions
- Outputs: expected artifacts
- Validation: commands/tests agent must run

## 6. MCP and CLI Tool Orchestration

Preferred order:
1. MCP filesystem/ripgrep/git tools for deterministic local operations
2. shell commands for build/test execution
3. network tools only when required by objective

Rules:
- Use ripgrep for repo search.
- Parallelize independent reads/searches.
- Avoid destructive git commands unless explicitly requested.
- Do not bypass reproducibility gates to speed up progress.

## 7. Task Tracking and Planning Workflow

Canonical planning state is TOML-first:
- `registry/roadmap.toml`
- `registry/todo.toml`
- `registry/next_actions.toml`

Narrative overlays:
- `registry/roadmap_narrative.toml`
- `registry/todo_narrative.toml`
- `registry/next_actions_narrative.toml`

Generated mirrors:
- `docs/ROADMAP.md`
- `docs/TODO.md`
- `docs/NEXT_ACTIONS.md`

Update policy:
1. update registry TOML first
2. regenerate mirrors
3. validate freshness checks

## 8. TOML-first Documentation Lifecycle

### 8.1 Canonical registries
- claims: `registry/claims.toml`
- insights: `registry/insights.toml`
- experiments: `registry/experiments.toml`
- binaries: `registry/binaries.toml`
- requirements: `registry/requirements.toml`
- narrative corpora:
  - `registry/docs_root_narratives.toml`
  - `registry/research_narratives.toml`
  - `registry/external_sources.toml`
  - `registry/data_artifact_narratives.toml`
  - `registry/reports_narratives.toml`
  - `registry/docs_convos.toml`
  - `registry/book_docs.toml`

### 8.2 Build and publish flow
- Validate registries:
  - `PYTHONWARNINGS=error make registry`
- Publish mirrors:
  - `PYTHONWARNINGS=error make docs-publish`

### 8.3 Mirror discipline
- Generated markdown mirrors are read-only outputs.
- Never edit generated mirrors as source.
- If mirror content is wrong, fix source TOML and regenerate.

## 9. LLM Overlay Files Contract

`CLAUDE.md` and `GEMINI.md` must contain:
- assistant-specific operating notes
- `/init` quick checklist aligned with this file
- tool orchestration notes specific to that runtime

They must not:
- redefine hard rules
- override TOML-first policy
- carry stale numeric counts that drift from registry truth

## 10. Requirements and Navigator Policy

### 10.1 Requirements
- Canonical source:
  - `registry/requirements.toml`
  - `registry/requirements_narrative.toml`
- Generated outputs:
  - `REQUIREMENTS.md`
  - `docs/REQUIREMENTS.md`
  - `docs/requirements/*.md`

### 10.2 Navigator
- Navigator content must be managed TOML-first.
- Any `NAVIGATOR.md` output is a generated projection, not canonical source.

## 11. Validation Gate Checklist Before Commit

Run relevant gates for touched surfaces:

Core:
- `PYTHONWARNINGS=error make check`
- `make ascii-check`

Registry/docs:
- `PYTHONWARNINGS=error make registry`
- `PYTHONWARNINGS=error make docs-publish`

Rust-focused changes:
- `cargo build --workspace -j$(nproc)`
- `cargo test --workspace -j$(nproc)`
- `cargo clippy --workspace -j$(nproc) -- -D warnings`

If full gates are too expensive for iteration, run targeted checks while coding
and full gates before final handoff.

## 12. Quickstart Command Block

```bash
# Registry authority and mirror publish
PYTHONWARNINGS=error make registry
PYTHONWARNINGS=error make docs-publish

# Full quality gate
PYTHONWARNINGS=error make check
make ascii-check

# Rust strict gate
cargo build --workspace -j$(nproc)
cargo test --workspace -j$(nproc)
cargo clippy --workspace -j$(nproc) -- -D warnings
```

## 13. References

- `README.md` - human-facing entrypoint
- `CLAUDE.md` - Claude overlay
- `GEMINI.md` - Gemini overlay
- `Makefile` - build, registry, docs-publish targets
- `src/scripts/analysis/export_registry_markdown_mirrors.py` - mirror exporter
- `src/verification/verify_registry_mirror_freshness.py` - freshness gate
- `registry/` - canonical machine-readable project state
