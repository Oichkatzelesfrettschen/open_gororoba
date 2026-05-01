# Dependency Pin Rationale

## Context

The workspace deliberately pins certain transitive-edge crates at older
major versions because newer releases would break the dependency chain
or force breaking-change cascades through several crates. This ADR
records each pin so a future contributor does not "fix" a pin without
understanding the constraint.

Pin discipline: every workspace-level pin lives in
`Cargo.toml [workspace.dependencies]` (root), uses a caret-bounded
version (`^X.Y` or `=X.Y.Z` for hard pins), and has a one-line comment
referencing the relevant section below.

## Decision

### `statrs = "0.18"` -- pinned by nalgebra compatibility

- `statrs 0.18.0` requires `nalgebra ^0.33`.
- `statrs 0.19+` requires `nalgebra ^0.34`.
- The workspace also pins `nalgebra` at `0.33` because:
  - Several internal crates carry typed wrappers around `nalgebra::Vector*`
    aliases that changed between 0.33 and 0.34.
  - Migrating `nalgebra` requires touching at least 12 crates in the
    workspace and re-validating the Cayley-Dickson tower numerics, which
    is a large self-contained piece of Phase 5/8 work.
- Net effect: bumping `statrs` to 0.19 forces a workspace-wide
  `nalgebra` migration, which is out of scope for any single PR and is
  blocked behind that bigger refactor.

Removal trigger: when a `nalgebra 0.33 -> 0.34` migration sprint is
explicitly scheduled, bump `statrs` in the same sprint.

### `nalgebra = "0.33"` -- pinned by statrs and num-dual

- `statrs 0.18` constraint above.
- Earlier audits noted `num-dual` constraints; the constraint trail is
  partially stale ("reason stale/absent" in MEMORY.md). Verify before
  the next migration attempt.

Removal trigger: same as `statrs`.

### `serde_json = "^1.0.139"` -- (informational)

Not strictly pinned, but no recent advisory; documented here so future
audits can match the pattern.

### `pyo3 = "0.28"`, `numpy = "0.28"` -- declared, unused

These are in `workspace.dependencies` but consumed by zero crates and
zero source files (verified 2026-04-30 in T-115). Pending user decision
in `data/output/audit/phase0_open_questions_2026_04_30.toml` Q-2026-04-30-05
on whether to remove or wire a binding.

### `tokio = "^1.40"` -- (informational)

Tokio is the de facto async runtime for the network-touching crates
(`reqwest`, `chromiumoxide`). Bumping should be straightforward; no
known compatibility blocker.

## Pin layering policy

When adding a new pin to `[workspace.dependencies]`:

1. State the upstream constraint that forces the pin in a comment in
   `Cargo.toml`.
2. Add an entry to this ADR under `## Decision` with: name, version,
   reason, removal trigger.
3. If the pin is forced by an upstream advisory or yanked release, also
   add a `[[blocker]]` entry to `registry/upstream_blockers.toml`.
4. If the pin is forced by a security advisory, cross-reference the
   relevant section in `docs/adr/rustsec-dispositions.md`.

## Verification

Lookup commands a contributor can run to verify pin reasons:

```bash
# Show the constraint statrs imposes on nalgebra
cargo tree -p statrs --edges normal --depth 2

# Show all crates depending on a particular pinned version
cargo tree -i nalgebra@0.33

# Confirm pyo3+numpy are consumed by zero crates (T-115)
grep -rE '^\s*pyo3\s*=' crates/*/Cargo.toml | wc -l   # expect: 0
grep -rE 'use pyo3'      crates/ --include='*.rs' | wc -l  # expect: 0
```

## Related

- `docs/adr/rustsec-dispositions.md` -- security-driven advisory ignores
- `registry/upstream_blockers.toml` -- watch list for upstream releases
- Plan: `plans/elucidate-and-build-out-nested-hollerith.md` (T-117).
