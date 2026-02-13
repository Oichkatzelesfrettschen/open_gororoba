# Agent and Contributor Operating Manual

This is the canonical policy and operating contract for assistants and human
contributors in this repository.

If any instruction in `CLAUDE.md` or `GEMINI.md` conflicts with this file,
`AGENTS.md` wins.

## File authority

| File | Role | Edit policy |
| --- | --- | --- |
| `AGENTS.md` | Global policy and workflow contract | Canonical, tracked |
| `CLAUDE.md` | Claude-specific overlay | Must defer to `AGENTS.md` |
| `GEMINI.md` | Gemini-specific overlay | Must defer to `AGENTS.md` |
| `README.md` | Human-facing entrypoint | Keep concise and accurate |

## Repository layout contract

- `apps/` contains front-end app surfaces and app-specific UX assets.
- `crates/` contains Rust workspace crates and executable bins.
- `docs/` contains human-facing generated and narrative documentation.
- `registry/` contains TOML authoritative sources.
- `src/verification/` contains deterministic verifier scripts.
- `build/` holds generated artifacts and mirrors.
- `data/` contains datasets, provenance artifacts, and evidence products.
- `scripts/` contains helper automation and migration tools.
- `logs/` is reserved for runtime and pipeline logs.

## Quality policy

- Treat warnings as errors in Rust and Python verification paths.
- Keep authored content ASCII-only unless a file already uses Unicode.
- Do not add network access to test/verifier lanes.
- Prefer deterministic scripts and reproducible command sequences.

## Execution policy

1. Read and validate affected TOML registries before editing mirrors.
2. Edit minimal scope required to resolve the issue.
3. Regenerate mirrors from TOML sources.
4. Run strict checks for touched areas:
   - `PYTHONWARNINGS=error make check`
   - crate-scoped Rust `clippy` and `test` commands
   - targeted registry verifiers
5. Record assumptions and unresolved risks in reports/plans.

## Front-end policy

- Cross-platform app concepts should be staged under `apps/`.
- First implementation lane is `apps/gororoba_studio/`.
- Back-end computation must remain Rust-native and reproducible.
- UI claims must map to executable pipeline outputs.

