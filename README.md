# open_gororoba

TOML-first, Rust-first research workspace for algebra, lattice, and physics-adjacent experiments.

## Canonical Entry Points

- Policy: `AGENTS.md`
- Workspace: `Cargo.toml`
- Registry source of truth: `registry/*.toml`
- Build and gates: `Makefile`

## Quick Commands

```bash
make registry
make rust-smoke
cargo clippy --workspace -j"$(nproc)" -- -D warnings
cargo test --workspace -j"$(nproc)"
make ascii-check
# before push/sync to origin
make pre-push-gate
```
