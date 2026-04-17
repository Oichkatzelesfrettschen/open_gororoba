# Shell Script Retirement Decision Log

## Context

Per project policy (~/.claude/CLAUDE.md and the debt audit
findings), .sh scripts under `bin/` and `scripts/` violate the
"no shell scripts in this Rust repo" rule. Phase 1 of the
remediation roadmap enumerates and classifies them; ports
happen in follow-up tasks.

Policy target: zero .sh files under `bin/` and `scripts/`.
The only exemptions permitted are third-party binaries (e.g.
micromamba) that must live under `vendor/`.

Task reference: P1.S4.T1 (inventory) and P1.S4.T2 (classify).

## Inventory (16 files as of 2026-04-17)

### Category A -- DELETE (trivial, replaceable by git commands)

| File | Lines | Decision | Rationale |
|---|---|---|---|
| `bin/do_commit.sh` | 2 | DELETE | Hardcoded commit message; replaceable by `git commit -m "..."`. |
| `bin/temp_commit.sh` | 3 | DELETE | Identical intent to do_commit.sh; obsoleted. |
| `bin/build_fix.sh` | 4 | DELETE | One-line `cargo build` redirect; trivial. |
| `bin/check_gemini.sh` | 14 | DELETE | Debug helper from a past investigation; no ongoing use. |

### Category B -- PORT TO MAKEFILE TARGET

| File | Lines | Decision | Target Makefile name |
|---|---|---|---|
| `scripts/cargo_cache_prune.sh` | 24 | PORT | Alias/merge with `make cache-sweep` (already exists). |
| `scripts/cargo_cache_status.sh` | 89 | PORT | Merge capabilities into `make cache-status` or a new `cache-audit`. |
| `scripts/docs-redirect-check.sh` | 104 | PORT | New `make docs-redirect-check`; or absorb into governance-gate. |
| `scripts/run_nanograv_timing_phase1_independent_locked.sh` | 8 | PORT | One-shot wrapper; convert to `make nanograv-phase1`. |

### Category C -- PORT TO RUST BINARY (non-trivial logic)

| File | Lines | Decision | New crate/binary |
|---|---|---|---|
| `scripts/bootstrap_user_local_xdg.sh` | 199 | PORT | `crates/xtask/src/bin/bootstrap_xdg.rs` (host-setup logic). |
| `scripts/detect_native_blas.sh` | 49 | PORT | Add to `xtask host-profile` (already detects CPU topology). |
| `scripts/detect_physical_cores.sh` | 5 | PORT | Already replaced by `xtask host-profile`; DELETE after verification. |
| `scripts/detect_worker_budget.sh` | 11 | PORT | Already replaced by `xtask host-profile`; DELETE after verification. |
| `scripts/profile_tensor_avt.sh` | 211 | PORT | Profiling harness; large port. Convert to a bench-orchestrator binary under gororoba_cli_physics. |
| `scripts/run_lambda_sweep.sh` | 100 | PORT | Research parameter sweep; convert to `gororoba_cli_physics --bin lambda-sweep`. |
| `scripts/run_reynolds_sweep.sh` | 17 | PORT | Research parameter sweep; convert similarly. |

### Category D -- PRESERVE UNDER vendor/

| File | Lines | Decision | Location |
|---|---|---|---|
| `bin/run_quantum_container.sh` | 15 | REVIEW | Docker wrapper; decide whether Docker is still in scope. If yes, relocate to `vendor/` with SHA256-stamped third-party note. Otherwise DELETE. |

### Category E -- BINARY / NON-SCRIPT

| File | Size | Decision | Notes |
|---|---|---|---|
| `bin/micromamba` | external | VENDOR | Third-party binary; relocate to `vendor/` with `docs/toolchain/micromamba.md` capturing sha256. (Task from plan Appendix: P1.S4.T5.) |

## Execution plan (follow-up tasks)

Each category rolls into its own PR to keep diffs reviewable.

1. Category A deletions: single PR, 4 files removed.
2. Category B ports: one PR per script or batched per target area;
   requires the target Makefile rule to be authored first.
3. Category C ports: one PR per Rust binary; may span sprints
   because porting preserves behavior bit-for-bit.
4. Category D review: decision captured in a separate ADR.
5. Category E: vendor relocation + docs.

## Verification

After full retirement:

  find bin scripts -name '*.sh' 2>/dev/null | wc -l   # expect 0
  ls vendor/ | grep -E 'micromamba|third-party'       # expect presence
  make docs-redirect-check                             # expect green

## Related plan tasks

- P1.S4.T1 (inventory)       -- COMPLETE.
- P1.S4.T2 (classify)        -- this document.
- P1.S4.T3 (port Cat B/C)    -- pending; multi-PR.
- P1.S4.T4 (delete Cat A)    -- pending; single PR.
- P1.S4.T5 (vendor Cat E)    -- pending; single PR.
