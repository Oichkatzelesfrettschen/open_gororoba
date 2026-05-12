# Pre-push gate optimization, 2026-05-11..12

Engineering note for future maintainers on the chain of fixes that
brought the local pre-push gate from ~16 minutes per push to ~88 seconds
(an 11x speedup) for typical pure-Rust refactor commits. References
commits live in the git history; this doc explains the diagnostic path
and the architectural decisions so the optimizations stick.

## Symptom

Through the PH-MOD (DEBT-MODULARIZATION Phase 1) emanation work in
early May 2026, every push spent 15-25 minutes in `make gate-local`
before the actual network upload to GitHub. The user pushed back; the
session that followed was the comprehensive RCA + fix campaign.

## Stages of the diagnosis

### Stage 1 — typo (commit `c08f9ef0`)

The Makefile `rust-regression-scoped` target invoked
`cargo run -q -p gororoba_cli_governance --bin workspace-routing`. The
binary actually lives in `gororoba_cli_data`. Cargo returned a
"no bin target" error on every call; `2>/dev/null` swallowed it; the
`||` fallback set `RUST_CLIPPY_SCOPE := RUST_SCOPE`. Both clippy and
nextest ran on the full 33-crate reverse-closure of every hub-crate
change.

Commit `a9edfd86 perf(gate): split clippy/nextest scopes` had claimed
to introduce direct-only scoping for clippy. The scope split was
inert from the moment it landed because of the crate-name typo. The
gate log printed `clippy scope:` and `nextest scope:` lines that
matched, looking plausible; nobody verified the actual underlying
cargo behaviour with a wall-time A/B test.

**Antipattern:** `2>/dev/null` in build infra hid a two-month-old
failure. Prefer cached binaries (skip the cargo invocation entirely),
or `2> >(tee /dev/stderr)` to forward stderr without discarding it.

### Stage 2 — beyond the typo (commits `99412fe7` through `0d9d8767`)

Fixing the typo cut pre-push from ~16 min to ~7-10 min. Five more
issues compounded to keep it slow:

1. **441 integration test binaries linked every push** (~4m30s of
   pure link time). Pre-push was running `cargo nextest run --lib
   --tests`, which compiled every integration test binary. PR CI was
   already doing the full closure on PR open — this was a duplicated
   safety net. Fix: `RUST_NEXTEST_KIND_MODE=lib` default; CI override
   to `all`.

2. **mold + lld installed but neither configured.** Default GNU `ld`
   was used. Fix: `.cargo/config.toml` with
   `[target.x86_64-unknown-linux-gnu] linker = "clang"` +
   `rustflags = ["-C", "link-arg=-fuse-ld=mold"]`. 3-5x linker
   speedup for the link-bound integration test phase.

3. **`repo-utilities` was a `[[bin]]` inside `gororoba_cli_data`.**
   `gororoba_cli_data` declared `algebra_experimental = { workspace = true }`
   as a package-level dep. Every algebra_experimental change cascaded
   through to a `repo-utilities` rebuild in release profile (~53s).
   Fix: extract `repo_utilities` to a standalone workspace member
   with no workspace-crate deps. `gate-target/gate-tools/` cache
   layout already existed; just had to break the dep cascade.

4. **`make check` ran every push** regardless of file types. For pure-Rust
   commits, ansi-check + terminology-gate are unnecessary. Fix: add
   `run_check` flag to `workspace-routing` (true when any non-`.rs`
   file changed); `gate-local` skips `make check` when it is false.

5. **Cargo metadata walks per subcommand.** Each `cargo run`/`cargo build`
   invocation in a fresh process re-reads the full workspace manifest
   graph (~10-30s for a 72-crate workspace). Fix: build the gate-tool
   binaries once at known stable paths under
   `.cache/gate-target/gate-tools/`, then exec them directly. Make
   tracks the source-dep timestamps so rebuilds only happen when the
   bin source file changes.

6. **`cache-check` ran four `du -sm` walks every push** over hundreds
   of GB. Fix: memoize with a 30-minute TTL sentinel file.
   `make cache-check-force` for explicit recomputation.

### Stage 3 — observability (commits `1dd3cda5` through `795fa81a`)

Once steady state landed at 88s, further perf wins required
disproportionate complexity. The session pivoted to observability and
defense-in-depth:

- `sccache` re-enabled in `.cargo/config.toml`. Passes through for
  the local incremental build (~5ms/crate); caches across sessions
  when `CARGO_INCREMENTAL=0` (CI).
- `cargo xtask gate-local` driver: replicates Makefile gate-local
  flow with structured JSONL timing output to
  `data/output/audit/<date>/gate-timing-<unix-ts>.jsonl`. Opt-in via
  `make gate-local-xtask`.
- `cargo xtask gate-timing-summary`: aggregate the JSONL files into
  per-phase stats (count, mean, median, p95, min, max, last).
- `cargo xtask gate-timing-regression-check`: per-phase comparison
  against baseline median with configurable threshold (default 2x).
  Wired into `.github/workflows/ci.yml` as advisory; promote to hard
  gate after baseline accumulates.
- `cargo xtask gate-tools-status`: inspect cached binary mtime vs
  source-dep mtime, surface `STALE` / `MISSING` so "why is the gate
  rebuilding tools every time" becomes a one-line diagnosis.
- `$(GATE_LOCK)`: pid + timestamp file written at gate-local start,
  cleaned via shell trap on EXIT/INT/TERM. `make gate-lock-status`
  reports state. Prevents the wave-2 PH-MOD bug where mid-gate
  source edits broke the test compile.
- `cache-sweep` policy: switched from `--maxsize 100GB` (destructive
  when triggered mid-session) to `--time 7 days` preservation with
  conditional gate-cbuild debug wipe (>14 days only).
  `make cache-sweep-dry-run` shows would-be removals.

## Layered gate model

- **Pre-push (local, `make gate-local` or `cargo xtask gate-local`)**:
  smoke gate. Direct-changed crates only. `--lib` only. Skip make check
  on Rust-only diffs. Target: under 2 minutes warm-cache.
- **PR CI (`gate-ci-rust` in ci.yml)**: full workspace closure +
  `--lib --tests` + integrity gates + governance. Target: 10-15
  minutes.
- **Post-merge CI**: heavy and CUDA suites. Catches anything the
  preceding tiers miss.

If a hub-crate change breaks a downstream consumer, the downstream
consumer's own code didn't change in the same commit — its pre-push
catches the breakage on the consumer-side commit, and PR CI catches
it preemptively on every PR. Pre-push is intentionally a smoke gate.

## Architectural decisions worth preserving

1. **`workspace-routing` lives in `gororoba_cli_data`.** It is one
   `[[bin]]` of many. Always invoke with `-p gororoba_cli_data --bin
   workspace-routing`, or via the cached binary at
   `.cache/gate-target/gate-tools/workspace-routing`.

2. **`repo_utilities` is its own crate.** Do not move it back into
   `gororoba_cli_data`. The cascade-invalidation cost from `algebra_*`
   crates would re-emerge.

3. **mold linker configured in `.cargo/config.toml`.** If a future
   crate or rustc-flag combination becomes incompatible with mold,
   override per-shell with `RUSTFLAGS=` rather than removing the
   config entry.

4. **sccache in `.cargo/config.toml` is benign at incremental.**
   ~5ms/crate overhead for the gate's normal incremental builds, with
   real cross-session wins under `CARGO_INCREMENTAL=0` (CI). Removing
   it would lose the CI win.

5. **Pre-push runs `--lib` only by default.** Do not change this
   without verifying that PR CI runs `--lib --tests`. Integration
   test compile is the largest single contributor to gate wall time.

6. **gate-local must always release `$(GATE_LOCK)`.** The shell trap
   in the gate-local target body handles EXIT/INT/TERM. If you
   refactor gate-local to multiple shells, preserve the trap or move
   the lock management to the xtask driver.

7. **`cargo sweep --time 7` preserves the working set.** The previous
   `--maxsize 100GB` policy was destructive. Do not revert without
   adding `--dry-run` first to measure what would be lost.

## File index

Persistent caches:

- `.cache/gate-target/gate-tools/workspace-routing` -- routing-CLI binary.
- `.cache/gate-target/gate-tools/host-profile.sh` -- pre-computed HOST_*
  shell vars.
- `.cache/gate-target/gate-tools/xtask` -- xtask binary for the opt-in
  driver.
- `.cache/gate-target/gate-tools/cache-check.last` -- memoized
  cache-check output (30 min TTL).
- `.cache/gate-target/gate-tools/gate-local.lock` -- gate-local in-flight
  marker (cleaned by trap on exit).

Per-run artifacts:

- `data/output/audit/<YYYY-MM-DD>/gate-timing-<unix-ts>.jsonl` --
  one JSONL line per phase from `gate-local-xtask` runs.

Reference RCAs:

- `data/output/audit/2026-05-11/pre-push-gate-rca.md` -- v1, scope bug.
- `data/output/audit/2026-05-11/pre-push-gate-rca-v2-comprehensive.md` --
  v2, all five issues.
