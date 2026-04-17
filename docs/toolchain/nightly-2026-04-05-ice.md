# Nightly 2026-04-05 ICE Captures

## Context

On 2026-04-04/05, a trial bump from the long-standing pinned
`nightly-2026-03-05` toolchain to `nightly-2026-04-05` produced 18
internal compiler errors (ICEs) while running scoped clippy/test
work. The rustc-ice-*.txt dumps were committed-to-workdir but not
tracked (see `.gitignore:52`). The 18 dumps have been removed from
the repo root as part of Phase 1 of the remediation roadmap.

This ADR preserves the incident context so future maintainers can
reason about similar failures without archeology.

## Symptoms

- Eighteen ICE dumps between 06:19:15 and 06:21:XX on 2026-04-05.
- Pairs of dumps appear (two files per event: 16,693 B and
  7,205 B typical), suggesting the crash produced both a primary
  backtrace and a rustc-driver dump per failure.
- Timing aligns with a batch run of `cargo clippy --workspace`
  and `cargo test` under the new toolchain.

## Why it matters

- `Cargo.toml:2` sets `cargo-features = ["codegen-backend"]`.
  35+ crates in the workspace override `codegen-backend = "llvm"`
  for SIMD/intrinsic/inline-asm reasons (ref `Cargo.toml:441-721`).
- If an SIMD-heavy crate is newly added without an `llvm` override,
  Cranelift is used and can ICE on architectures or instruction
  sequences that the release toolchain handles cleanly.
- Nightly-to-nightly churn in the unstable feature set or in
  Cranelift internals is a recurring ICE source.

## Resolution

Current pin (as of 2026-04-17) remains `nightly-2026-04-05` per
`rust-toolchain.toml:2` -- the ICE pattern was NOT reproducible
after the initial batch and did not block subsequent gate runs.

If ICE recurs:

1. Capture the dump pair (rustc-ice-*.txt) for the triggering
   crate and instruction context.
2. Check whether the crate is missing an
   `[profile.<profile>.package.<crate>] codegen-backend = "llvm"`
   override.
3. Try a two-back nightly (e.g. `nightly-2026-04-01`) via
   `rustup run` without changing the pin.
4. If confirmed bug: file upstream with the dump; pin temporarily
   to the last known-green nightly.

## Rollback

- Replace `rust-toolchain.toml` channel with the last stable
  known-green nightly.
- Revert any nightly-specific feature flag adoption.
- Rerun `make gate-local` to confirm green.

## References

- `rust-toolchain.toml`
- `Cargo.toml:441-721` (codegen-backend overrides)
- `.gitignore:52` (rustc-ice-*.txt)
- Plan phase: P1.S2.T4
