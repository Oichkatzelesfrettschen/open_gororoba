# Build cache budget and sweep policy

This file defines size budgets and operational procedures for the on-disk
build cache. See `data/output/audit/2026-04-30/02-cache-sizes.txt` for the
baseline measurements (`.cache=215G`, `gate-target=200G`, `target=21G`).

## Why this file exists

The `.cache/` tree grows with cargo artifacts, sccache (when enabled), and
per-crate experimental targets. Without a budget, it can balloon into the
hundreds of gigabytes (see MEMORY notes on the historical 1.7TB worst-case).
A documented policy lets contributors and CI agents know when to sweep and
when to escalate.

## Budgets

| Cache subtree              | Soft limit | Hard limit | Action when exceeded                |
|----------------------------|-----------:|-----------:|-------------------------------------|
| `.cache/gate-target/`      |       200G |       250G | `make cache-sweep`; re-run gates    |
| `.cache/cbuild/`           |        20G |        40G | `cargo clean -p <crate>` selectively |
| `.cache/cargo-default/`    |       150G |       200G | `cargo sweep --maxsize 100GB`        |
| All `.cache/exp-*-target/` |        50G |       100G | Delete experiments older than 30 days |
| Total `.cache/`            |       250G |       400G | Manual review; possibly retire branches |

The `gate-target` budget exists because a single gate run pulls many crates
into a fully-checked state. Going significantly above 250G usually means a
recent `--workspace` sweep has populated artifacts that should not persist.

## Sweep procedure

1. Run `make cache-check`. This prints the current sizes and exits 0 if all
   subtrees are below their soft limits.
2. Run `make cache-sweep`. Internally this calls `cargo-sweep --maxsize 100GB`
   on `.cache/cargo-default/` and `.cache/gate-target/`.
3. If still over budget, list per-crate occupancy with
   `du -sh .cache/gate-target/release/* | sort -h | tail -20` and remove the
   heaviest crates that are not in the current critical path.
4. NEVER delete `.cache/registry.sqlite3` -- it is the build artifact for the
   compatibility-export TOMLs. Regenerate it via `make registry-build` if it
   becomes stale.

## Experimental target directory convention

When running an experimental flag set (different feature subset, alternate
backend, custom rustc options) that would invalidate the gate cache, route
artifacts into `.cache/exp-<short-name>-target/` instead of `.cache/gate-target/`.
For example:

```
CARGO_TARGET_DIR=.cache/exp-cubecl-vulkan-target/ \
    cargo build -p lbm_vulkan --features cubecl
```

This keeps the gate cache hot for the next normal gate run.

## sccache and remote cache

`sccache` was removed from the build flow as of Sprint 79 (see MEMORY notes).
The decision was driven by sccache's instability with edition-2024 crates and
its tendency to mask compile-time errors. If a CI provider offers a managed
remote cache (GitHub Actions cache, sccache-dist), prefer that over a manual
sccache install.

## Makefile targets

- `make cache-check`: print sizes; exit 1 if any subtree exceeds its hard limit.
- `make cache-check-budget`: like `cache-check` but exits 1 if `.cache/gate-target`
  exceeds 200G specifically. Used by the pre-push hook.
- `make cache-sweep`: shrinks the gate cache via `cargo-sweep --maxsize 100GB`.
- `make cache-sweep-aggressive`: full `cargo clean` of the gate target. Use
  rarely; rebuild from scratch is expensive.

## Alert procedure

If a contributor consistently hits the hard limit on `gate-target/`:

1. Confirm they are using `CARGO_TARGET_DIR=.cache/gate-target` for gate runs
   (default behavior of `make rust-clippy`, `make rust-test`).
2. Check whether they have a stale workspace path triggering a full rebuild
   (e.g., a path-dependency outside the cargo-aware tree).
3. If neither, escalate via MEMORY note and consider raising the limit.

## When to retire experimental dirs

Experimental target dirs survive as long as the experiment is alive. Retire
them when:

- The experiment lands in `gate-target/` (success path).
- The experiment is abandoned for 30+ days.
- Disk pressure forces a sweep regardless of age.

Use `find .cache -maxdepth 1 -name 'exp-*-target' -mtime +30` to list
candidates. Removal is straightforward: `rm -rf .cache/exp-<name>-target/`.

## See also

- `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/MEMORY.md`
  -- "Cargo Cache Architecture (2026-03-23, CRITICAL)" section.
- Stage A audit: `data/output/audit/2026-04-30/02-cache-sizes.txt`.
- Stage B plan task B-Doc2 (this file is the deliverable).
