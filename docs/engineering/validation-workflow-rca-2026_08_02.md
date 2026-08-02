---
description: Root-cause analysis and operating model for repository validation cost and orchestration
last_verified: 2026-08-02
evidence_class: instrumented-local-capture
status: active
---

# Validation workflow RCA and operating model

The repository validation problem is orchestration duplication, not a lack of
checks. The old workflow invokes the same policy functions through nested Make
targets, repeated Cargo processes, incompatible cache paths, and historical
names that hide whether a command builds, verifies, mutates, or reports. The
result consumes compute and human time without producing proportional evidence.

The active design has one declarative entry point per context, one shared build
per dependency tier, direct binary execution for repeated checks, and
compatibility aliases for scripts that still use old names. The aliases do not
form active validation edges.

## Vocabulary

| Historical name | Active name | Meaning |
| --- | --- | --- |
| `release-gate` Cargo profile | `validation` Cargo profile | Optimized diagnostic build profile. The physical target path remains compatible. |
| `gate-local` | `validate-local` | Changed-file local validation before publication. |
| `gate-ci-registry` | `validate-ci-registry` | CI registry policy and evidence validation. |
| `gate-ci-rust` | `validate-ci-rust` | CI Rust, dependency, and schema validation. |
| `gate-audit` | `validate-repository` | Structured repository validation report. |
| `gate-fast` | `validate-static` | Lightweight ASCII, terminology, and dependency checks. |
| `gate-warm` | `validate-static-and-registry` | Lightweight checks plus registry policy validation. |
| `gate-deep` | `validate-comprehensive` | Static, registry, Rust, and dependency validation. |
| `audit-deep` | `audit-comprehensive` | Opt-in broad audit with structured evidence. |
| `audit-deep-structured` | `audit-comprehensive-structured` | Structured broad-audit report and logs. |
| `supply-chain-gate` | `validate-supply-chain` | Dependency and unsafe-source policy validation. |
| `ndlb-gate` | `validate-dataset-experiments` | Dataset, server, and experiment invariant validation. |
| `registry-acceptance-gate*` | `validate-registry` | Registry and evidence validation. |
| `integrity-resolution` | `registry-integrity` | Registry signature and consistency artifact generator/verifier. |
| `governance-verify gate-all` | `governance-verify validate-all` | One-process registry policy validation. |
| `markdown-registry verify-gate-all` | `markdown-registry verify-all` | One-process Markdown owner and inventory validation. |
| `gate-timing-*` | `validation-timing-*` | Validation timing evidence. The reader accepts historical timing files. |

The word `gate` remains only in compatibility aliases, historical evidence
paths, and exact tool names such as `terminology-gate`. New documentation,
targets, reports, and source identifiers use `validate`, `validation`,
`verify`, `registry-integrity`, or a mechanism-specific name.

## Root cause

The old execution model is a tree of wrappers that each believe they own the
whole validation contract:

```text
CI job or pre-push hook
  -> make gate-ci-registry or make gate-ci-rust
    -> nested make target
      -> cargo run for one binary
        -> workspace metadata walk and package selection
          -> binary build or relink
            -> leaf verifier
```

The registry path adds a second tree over the same leaves:

```text
gate-ci-registry
  -> governance-gate-readonly
  -> registry-control-plane-gate-readonly
  -> registry-acceptance-gate-readonly
       -> semantic-atoms
       -> evidence-provenance
       -> integrity-resolution
       -> execution-planning
       -> governance-verify subchecks
       -> markdown-registry subchecks
```

The old path is not a reasonable workflow for five independent reasons.

1. A child target is not a reusable evidence object. It reopens Cargo and
   repeats policy checks instead of exposing a result to its parent.
2. The same `gororoba_cli_data` package carries a broad normal dependency
   closure. A small validation binary therefore inherits a costly package
   build boundary.
3. The old CI cached `target/` while Make validation defaulted to
   `.cache/gate-target`. The cache key and the effective artifact path could
   disagree, so a green cache restore did not guarantee reuse. The active
   validation job uses `.cache/gate-target` and caches the matching build
   directory.
4. `release-gate`, `integrity-resolution`, and `wave` or `batch` aliases make
   historical implementation details look like independent scientific
   evidence. Names do not reveal whether a command is a build, a mutation, or
   a read-only check.
5. Failure reporting can start a fresh checkout and replay a failed composite
   path. A diagnostic rerun then spends more compute without changing the
   failing input or retaining the first result as a causal artifact.
6. The old comprehensive composite requests Clippy explicitly and then calls
   `rust-regression`, whose dependency already owns Clippy. The same lint lane
   therefore runs twice. The canonical `validate-comprehensive` target calls
   `rust-regression` once and records that ownership in the target graph.

The invariant is simple: repeated computation is justified only when it
changes the input, the validator, the execution environment, or the evidence
class. A second process over the same bytes is not independent confirmation.

## Instrumented baseline

The baseline capture used the clean canonical checkout and the isolated
registry worktree. It retained Cargo metadata, the `gororoba_cli_data` normal
dependency tree, Make expansion captures, Cscope output, Cflow output, and a
Rust tag inventory under `/var/tmp/open_gororoba-validation-20260802/`.
Raw captures are temporary. This document records the durable observations.

| Surface | Observation |
| --- | --- |
| Workspace shape | `cargo metadata --no-deps` reports 76 packages, 745 targets, and 496 binaries. |
| Validation package | `cargo tree -p gororoba_cli_data --edges normal` emits 2,362 lines. The package is not a small validation package. |
| Old registry CI expansion | Static Make expansion contains 11 Cargo invocations and 7 nested Make calls. |
| Old Rust CI expansion | Static Make expansion contains 7 Cargo invocations and 5 nested Make calls. |
| Old acceptance expansion | Static Make expansion contains 8 Cargo invocations and 4 nested Make calls. |
| Canonical build | 34.41 seconds in the retained local measurement. |
| Control-plane export | 2 minutes 56 seconds in the retained local measurement. |
| Project counter synchronization | 5 minutes 29 seconds in the retained local measurement. |
| Old integrity profile build | 4 minutes 16 seconds in the retained local measurement. |
| Mirror emission profile build | About 3 minutes 59 seconds in the retained local measurement. |
| Old acceptance run | The first release-profile build reached 6 minutes 24 seconds before the user interrupted the run. The acceptance result is partial and is not a pass. |
| New source compile | The renamed `registry-integrity` source compiled in the `validation` profile in 3 minutes 15 seconds. This is a one-time compile result, not a claim about the full 11-binary bundle. |

The control-plane reindex also exposed a stale generated theorem inventory. The
canonical SQLite snapshot had 132 theorem rows before the workspace proof
manifest was reindexed, while `proofs/_RocqProject` names 162 existing proof
files. Reindexing is the authoritative reconciliation step for that surface.
The four unlinked rows C1635 through C1638 are structural proof surfaces and
remain visible in the P1 queue until claim-level evidence rows exist.

The timing asymmetry explains the human complaint. The repository spends
minutes rebuilding or rediscovering validators whose actual work is often a
small file walk, hash comparison, or SQLite query. The build is evidence only
when the validator binary or its dependency closure changed.

## Call and dependency map

`cflow` and `cscope` are not semantic Rust call-graph tools. The repository has
no hand-written C application path for these checks. The exact C surface is
four generated Rocq extraction units under `proofs/extraction/`. Cscope indexed
those four units. Cflow parsed the two generated C translation units with
`--no-preprocess --depth=3 --main=main`; duplicate extracted symbols appeared
when the units were combined. That result establishes lexical symbol overlap,
not executable independence.

The Rust map uses Universal Ctags plus source and Cargo metadata inspection.
The tag inventory contains 1,296 tags over these load-bearing surfaces:

```text
Makefile
  validate-local
    validation-tools
      core validation bundle (workspace-routing + xtask)
      host-profile from cached xtask
      registry validation tool bundle
      provenance verify-control-plane + project-counter-sync --check
    check, when non-Rust files changed
    rust-regression-scoped, when Rust files changed
    validate-governance, when registry or document policy files changed

Makefile
  validate-comprehensive
    validate-static-and-registry
    rust-regression (including its clippy prerequisite, once)
    cargo audit
    cargo-deny-check

Makefile
  validate-ci
    xtask validate-ci
      make validate-ci-registry
        validate-registry
          markdown-registry verify-all
          governance-verify validate-all
          semantic-atoms --verify
          evidence-provenance --verify
          registry-integrity --verify
          execution-planning --verify
      make validate-ci-rust
        rust-regression
        validate-rust-integrity
          claims-verify
          test-inventory
          registry-check
        cargo-deny-check
        db-schema-drift-check

Makefile
  validate-repository
    the same two CI validation lanes
    one structured report tree under reports/validation/
```

The relevant source owners are:

| Owner | Responsibility |
| --- | --- |
| `Makefile` | Entry-point composition, cache stamp, stable binary paths, and compatibility aliases. |
| `xtask/src/main.rs` | Structured repository report, local timing capture, timing summary, and regression comparison. |
| `crates/gororoba_cli_data/src/bin/markdown_registry.rs` | Markdown owner-map and inventory validation. `verify-all` is the active command. |
| `crates/gororoba_cli_data/src/bin/governance_verify.rs` | Registry policy checks. `validate-all` is the active command. |
| `crates/gororoba_cli_data/src/bin/registry_integrity.rs` | Registry consistency artifacts and signature verification. |
| `crates/gororoba_cli_data/src/bin/semantic_atoms.rs` | Semantic atom consistency. |
| `crates/gororoba_cli_data/src/bin/evidence_provenance.rs` | Evidence provenance consistency. |
| `crates/gororoba_cli_data/src/bin/execution_planning.rs` | Planning and execution contract validation. |
| `crates/gororoba_cli_data/src/bin/claims_verify.rs`, `test_inventory.rs`, `registry_check.rs` | Rust integrity subchecks. |
| `crates/gororoba_cli_provenance` and `project-counter-sync` | Canonical SQLite theorem/binary/claim parity and top-level project counter parity. |

## Implemented operating model

The active Makefile builds the routing proxy and xtask in one slim
`cargo build --profile validation` invocation. It builds the 11 registry and
Rust-integrity binaries plus the canonical provenance verifier and project
counter check in a second shared invocation because those binaries share the
broad `gororoba_cli_data` dependency closure. Both bundles copy to
stable paths under `$(REPO_CARGO_TARGET_DIR)/validation-tools/` and use source
dependency stamps to avoid rebuilding when only registry data changes. The
read-only targets execute those paths directly. A local scope decision pays
only for the slim bundle; registry policy validation pays for the broad bundle
only when its input surface requires it.

The active profile is named `validation`. The old physical `.cache/gate-target`
directory remains to preserve existing cache contents and worktree isolation;
the directory name is not an active workflow concept.

`validate-repository` now runs the two CI validation lanes and does not append a
third workspace `cargo check --workspace --tests`. The Rust regression lane
already compiles the test closure while running clippy and nextest. The extra
check repeated compilation without adding a distinct invariant.

The optional composite vocabulary follows the same ownership rule. Static
checks run under `validate-static`; registry-aware checks run under
`validate-static-and-registry`; the comprehensive target calls
`rust-regression` once instead of requesting its Clippy prerequisite twice.
The old `gate-*` and `audit-deep*` names remain compatibility aliases only.

The pre-push hook invokes `make validate-local`. The Rust pre-push binary also
invokes `make validate-local` directly instead of calling a missing `makew`
wrapper. The CI workflow invokes `./makew validate-ci`, keeps the dedicated
docs lane, and uploads `reports/validation/**` without launching a fresh
failure-only full audit.

## Post-refresh validation result

The first integrated repository run exposes one genuine generated-artifact
defect: `db/schema.sql`, `docs/db/schema.json`, and the Rust database mirror do
not contain the revision tables and `status_note` columns present in the
migration set. `cargo run -p xtask -- db-docs` regenerates all three artifacts;
`db-docs --check` then passes. The retained failed report remains
`reports/validation/2026-08-02/143444/summary.md` and records the causal
failure instead of being overwritten as a pass.

The final repaired run passes both validation lanes in
`reports/validation/2026-08-02/150435/summary.md`:

| Surface | Result |
| --- | --- |
| Registry lane | One cached tool session; 1,448 claims, 183 insights, 232 experiments, 496 binaries, 162 theorems, and 297 proof files remain synchronized. |
| Rust lane | 6,106 workspace tests pass with 48 skipped; the two heavy packages add 1,346 passes with 68 skipped. |
| Dependency policy | Cargo-deny reports zero advisory errors; one yanked `spin 0.10.0` warning remains transitive through Burn/CubeCL and is recorded in the RustSec disposition ADR. |
| Schema artifacts | `db-docs --check` passes after regeneration. |
| Formatting baseline | The separate `fmt-check` surface reports 59 files requiring dprint formatting; the default validation composites do not claim this unresolved baseline as a pass. |

The full run proves the workflow contract. It does not close scientific rows
whose falsifiers require AMDA input, domain-specific sweeps, or formal claim
bindings. Those rows remain in the scientific frontier map and the canonical
planning tables.

The dated report directory is a CI/worktree artifact and remains ignored by
Git. The durable result is this section plus the evidence ledger; CI uploads
the raw lane logs and latest manifest for each validation run.

## Validation contract and falsifiers

The redesign is accepted only when all of these statements remain true:

| Statement | Falsifier |
| --- | --- |
| A local Rust-only change skips non-Rust checks while preserving scoped Rust validation. | `workspace-routing` reports the wrong scope or a Rust-only change runs the non-Rust check. |
| A registry-only change uses the cached validation tools without a Rust workspace regression. | `validate-repository-fast` invokes `rust-regression`, or a cached binary is older than its source dependency. |
| CI registry and Rust lanes share one target root and do not duplicate either validation bundle. | Two Cargo builds occur for the 11-binary registry bundle in one `validate-ci` invocation, or CI cache paths differ from Make paths. |
| `validate-registry` executes each registry invariant once. | The active recipe invokes `governance-verify validate-all` or `markdown-registry verify-all` more than once for the same run. |
| Old names remain usable but do not appear in active CI or pre-push edges. | An active workflow invokes a compatibility alias or a new source comment introduces a historical gate name. |
| Timing artifacts describe validation, not an implicit gate. | A new run writes `gate-timing-*` instead of `validation-timing-*`, except for preserved historical inputs. |

The remaining structural frontier is the broad `gororoba_cli_data` package.
The cache stamp removes repeated process and link work, but it does not make a
large dependency closure small. A future `gororoba_registry_validation` crate
should own the 11 validation binaries or split them into policy and
integrity packages. That work belongs in a measured P1 structural change,
after the current single-session result is captured.

## Reproduction

Run these commands from an isolated worktree. They use the active vocabulary:

```bash
make validate-local
make validate-ci
make validate-repository-fast
make validation-tools-status
cargo run -p xtask -- validation-timing-summary --since-days 30
cargo run -p xtask -- validation-timing-regression-check --baseline-days 14
```

Registry mutation remains distinct from validation:

```bash
cargo run --release -p gororoba_db --bin gororoba-db -- <subcommand>
cargo run --release -p gororoba_cli_provenance --bin provenance -- export-control-plane
make registry-integrity
make validate-registry
```

The first three commands change or regenerate artifacts. `make validate-registry`
is read-only. Keeping mutation and validation separate prevents a verifier from
silently repairing the bytes that it claims to inspect.
