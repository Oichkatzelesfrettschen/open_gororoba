# open_gororoba

TOML-first, Rust-first research workspace for algebra, lattice, and physics-adjacent experiments.

## Canonical Entry Points

- Policy: `AGENTS.md`
- Policy overflow and machine docs: `registry/agents_contract.toml`, `registry/crate_source_scout.toml`, `registry/mcp_server_matrix.toml`
- Workspace: `Cargo.toml`
- Registry source of truth: `registry/*.toml`
- Build and gates: `Makefile`

## Quick Commands

```bash
make registry
make rust-smoke
cargo clippy --workspace -j"$(nproc)" -- -D warnings
cargo test --workspace -j"$(nproc)"
# when dependencies change
make dep-audit
make ascii-check
# before push/sync to origin
make pre-push-gate
# optional stricter pre-push (includes dep-audit)
make pre-push-gate-strict
# install hooks
make hooks-install
# install strict pre-push hook
make hooks-install-strict
```
