# Gemini Experiments: Evidence-First Algebra/Physics Sandbox

`open_gororoba` is a Rust-first research workspace for algebra, physics, data
pipelines, and reproducible claim validation with TOML-first governance.

## Core quickstart

```bash
make install
PYTHONWARNINGS=error make check
make doctor
```

## Rust quality lane

```bash
cargo clippy --workspace -j$(nproc) -- -D warnings
cargo test --workspace -j$(nproc)
```

## Interactive app lane

`gororoba-studio` is the first full-featured front-end control plane for live
thesis pipelines:

```bash
make studio-run
```

Then open `http://127.0.0.1:8088`.

## Governance and docs

- Canonical contributor contract: `AGENTS.md`
- Requirements source of truth: `registry/requirements.toml`
- Operational trackers: `registry/roadmap.toml`, `registry/todo.toml`,
  `registry/next_actions.toml`
- Studio requirements lane: `apps/gororoba_studio/README.md`
- Front-end concept catalog: `reports/frontend_app_design_options_2026_02_12.toml`
