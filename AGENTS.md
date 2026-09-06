---
description: open_gororoba -- pure-Rust scientific computing workspace (Cayley-Dickson algebras, cosmology, LBM, Rocq proofs, SQLite-canonical registry)
last_verified: 2026-05-17
---

# open_gororoba -- agent and developer reference

This file is the canonical operating guide for any agent (Claude,
Codex, Gemini, Mistral, Ollama, DeepSeek, human) working in this
repository. `CLAUDE.md` is a tracked repository-relative symbolic link to `AGENTS.md`, so Claude Code reads this same body and the rules live in one place. The no-symlinks rule in section
"No-Python, no-symlinks, no-shell" governs build and data paths; the agent
entry point is its one declared exception. A `GEMINI.md`, if one is added,
is a regular loader that imports `@AGENTS.md` and carries only Gemini-specific
notes.

Peer references this document inherits style from:

- `~/workspaces/mesa/steinmarder/AGENTS.md` -- C11 + CUDA + SASS-RE
  engine; source of the comment-hygiene policy below.
- `~/workspaces/mesa/mesa-26-gororoba/AGENTS.md` -- Mesa 26.1-devel
  fork; source of the `Assisted-by:` commit-trailer policy.

Where the peer projects deal with hardware reverse engineering, this
project deals with mathematical structures (Cayley-Dickson algebras),
fluid dynamics (Lattice Boltzmann), formal verification (Rocq 9.1.1),
and a SQLite-canonical claim/insight/experiment registry. The
hygiene + voice + AI-disclosure rules transfer unchanged; the
hardware-specific tables are replaced with the scientific stack.

## Read first (in this order)

1. This file (`AGENTS.md`).
2. `~/.claude/CLAUDE.md` -- global user policies (emoji-free text,
   warnings-as-errors, no shortcuts, TodoWrite discipline,
   AskUserQuestion exhaustively).
3. `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/MEMORY.md`
   -- per-project memory index.
4. `~/AGENTS.md` -- home-level cross-project policies.

## High-value entry points

| Path                                                  | Why it matters                                                                                              |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `Cargo.toml` (root)                                   | Workspace members + `[workspace.lints]` (warnings-as-errors source of truth)                                |
| `rust-toolchain.toml`                                 | Stable pin (`1.97.0`); do not bump without coordinating repository validation.                               |
| `.githooks/pre-push`                                  | Five-check validation chain (cache, ansi, terminology, rust-regression, registry policy).                 |
| `Makefile`                                            | Top-level lanes (`make rust-clippy`, `make integrity`, `make cpd-audit`).                                   |
| `registry/canonical/control_plane.sqlite3`            | Canonical write target for the claim/insight/experiment registry.                                           |
| `registry/*.toml`                                     | AUTO-GENERATED read-only compat exports. Do NOT hand-edit.                                                  |
| `crates/gororoba_gpu_bridge/`                         | Canonical type vocabulary (`ComputeBackend`, `HardwareCaps`, `StoragePrecision`).                           |
| `crates/gororoba_gpu_vulkan/`                         | Shared Vulkan helpers (Instance, Adapter, Device, ShaderModule, DispatchScope).                             |
| `crates/gororoba_gpu_cubecl/`                         | Shared cubecl-wgpu probe + test-support macros.                                                             |
| `crates/gororoba_gpu_cuda/`                           | Shared CUDA helpers (Context, DeviceProbe, CompileOptions, ModuleRegistry, LaunchConfig, Telemetry, OptiX). |
| `crates/cd_kernel/`                                   | Cayley-Dickson tower (8D..1024D), TurboQuant, multi-backend kernels.                                        |
| `crates/lbm_3d/`                                      | CPU D3Q19 reference solver (BGK + MRT).                                                                     |
| `crates/lbm_vulkan/` + `crates/lbm_3d_cuda/`          | GPU LBM backends.                                                                                           |
| `proofs/`                                             | Rocq 9.1.1 formal proofs; `make -C proofs vos` for interface check, `make -C proofs vok` for body check.    |
| `docs/engineering/registry_canonical_architecture.md` | Four-layer registry flow: SQLite -> compat TOMLs -> Rust mirrors -> docs.                                   |

## Top-line operating rules

- **Emoji-free, and ASCII by convention**. `ansi-check` is an
  anti-emoji gate, not an ASCII gate: in `--check` mode it fails a file
  only on an emoji or on a control character other than tab, newline or
  carriage return. Smart quotes, en/em dashes, arrows, Greek letters,
  box-drawing and accented characters all pass it. Writing ASCII
  remains the house convention and `--fix` rewrites the typographic
  substitutes through `CHARACTER_POLICY_REPLACEMENTS`, NFKD
  normalization and combining-mark removal, but the gate does not
  enforce it, so a reviewer catches what the hook does not.
  The emoji predicate covers six ranges -- Emoticons, Miscellaneous
  Symbols and Pictographs, Transport and Map, Supplemental Symbols and
  Pictographs, regional-indicator Flags, Variation Selectors -- and
  omits Miscellaneous Symbols (U+2600..U+26FF), Dingbats
  (U+2700..U+27BF) and Symbols and Pictographs Extended-A
  (U+1FA70..U+1FAFF), so a warning sign, check mark or cross mark
  passes. A green `ansi-check` is not proof that a file is emoji-free.
  The check skips binary and build-artifact extensions, everything
  under `data/external/papers`, files over 10 MB, and files that are
  not valid UTF-8. `terminology-gate` is a separate check over the
  eight banned legacy terms in `registry/terminology_standards.toml`
  and has nothing to say about encoding.
- **Warnings-as-errors** via `[workspace.lints]` in root `Cargo.toml`.
  Do NOT bypass with crate-local `#![allow(warnings)]`. Narrow-scope
  `#[allow(clippy::<lint>)]` with a documented rationale is permitted.
- **SQLite-canonical registry**. The 36 TOML files under `registry/`
  are AUTO-GENERATED. The canonical write path is
  `registry/canonical/control_plane.sqlite3`. See the
  "Registry: SQLite-canonical" section below for the exact mutation
  workflow.
- **Pure Rust**. No `.sh` scripts. No `.py` analysis scripts. Use
  PyO3 if a Python library must be wrapped; call it from a typed
  Rust binary.
- **No symlinks** as workarounds. Use a separate `CARGO_TARGET_DIR`
  per worktree.
- **Pre-push hook** at `.githooks/pre-push` (active via
  `core.hooksPath`) runs `make validate-local`, the five-check local
  validation chain. Do not skip with
  `--no-verify` unless explicitly directed and rationale documented
  in the commit message.

## Build environment

- Toolchain: `rust-toolchain.toml` pins stable `1.97.0`,
  edition 2024.
- Default target dir for repository validation:
  `CARGO_TARGET_DIR="$(pwd)/.cache/gate-target"`. The physical path is retained
  for cache compatibility; the logical workflow uses `validate-*` targets.
  Pass it absolute. Cargo resolves a relative `CARGO_TARGET_DIR` against the
  shell's working directory, so a build launched from inside the tree writes a
  nested `.cache/gate-target/debug/.cache/gate-target`, exits 0, and leaves
  every later `stat` on the original path reading a stale artifact. The
  `Makefile` is already safe: `REPO_CARGO_TARGET_DIR` is
  `$(CURDIR)/.cache/gate-target`.
- Per-worktree experimental dirs: `.cache/exp-<name>-target/`.
- Cache budget: gate-target <= 200G; full `.cache` <= 250G; sweep
  with `make cache-sweep` (cargo-sweep --maxsize 100GB) or
  `make cache-sweep-soft` for the in-flight 150G soft cap.
- An isolated worktree on a capacity-constrained filesystem MUST set
  `REPO_CARGO_HOME`, `REPO_CARGO_TARGET_DIR`, and `REPO_CARGO_BUILD_DIR` to
  distinct paths on a filesystem that satisfies the cache budget.
- A linked worktree on the same filesystem as the primary checkout
  SHOULD run the gate with `REPO_SHARE_PRIMARY_CACHE=1`. The roots are
  resolved in `mk/cache_roots.mk`: the cargo-home and the Cargo
  build-dir are shared with the primary checkout, while the target-dir,
  the validation-tools copies, the cache-check sentinel and the
  validate-local lock stay under the worktree's own `.cache/`. The
  build-dir carries the dependency artifacts and is the path the
  sccache hash covers, so sharing it alone turns a cold ~70 min chain
  into ~10 min; keeping the target-dir local means a worktree with
  divergent tool sources never executes another worktree's `xtask` or
  governance binaries. Cargo keys a workspace crate on its content, so
  a byte-identical crate compiled in one worktree is handed to every
  other worktree through the shared build-dir; a tool therefore MUST
  NOT locate the repository through `env!("CARGO_MANIFEST_DIR")`,
  which bakes in the compiling checkout. `repo_root::resolve!()` and
  `repo_root::path!()` read `GOROROBA_REPO_ROOT` (exported by the
  Makefile as `$(CURDIR)`), then walk up from the working directory,
  and fall back to the compile-time path only when both fail. The
  owner is verified as a standard `<checkout>/.git` directory holding a
  `Cargo.toml`; bare, separate-git-dir and submodule layouts are
  rejected and need an explicit absolute `REPO_CACHE_OWNER`. Cleanup
  targets pass every path through `guarded_rm`, which refuses the
  owner's tree from a worktree unless `REPO_ALLOW_SHARED_CLEAN=1`.
  `make print-cache-roots` shows the resolved paths and
  `xtask/tests/cache_roots.rs` pins the layouts.
- The validation-tool cache identity is a sha256 over three inputs: the
  content hash of every `.rs` and `Cargo.toml` under `crates/` and `xtask/`,
  the resolved real path of the running checkout, and the git common-dir
  owner. `mk/cache_roots.mk` names the stamp
  `<validation-tools>/tool-identity.<identity>` and writes
  `tool_identity=`, `source_identity=`, `curdir_real=` and `owner=` into it.
  Two worktrees of one owner carry identical tool sources, so the source hash
  alone rates them equal; the checkout path is the term that separates them,
  and a stamp written for one worktree therefore never satisfies another.
- The identity gates the rebuild decision, not the bytes. Cargo keys a
  workspace crate on content and the shared build-dir hands one worktree's
  compiled crate to every other, so a fresh stamp can still stage a binary
  whose DWARF comp_dir strings name the compiling checkout. The staging step
  (`stage_tool` in the Makefile) strips debug sections from every copy, and
  because workspace sources are compiled by relative path the absolute
  checkout paths that survive are compile-time bake-ins such as
  `env!("CARGO_MANIFEST_DIR")`. `make validation-tools-check-paths` reads each
  staged file and fails on an absolute path under `$(REPO_WORKTREES_ROOT)`
  (default `$(HOME)/worktrees`) that names neither the running checkout nor a
  directory any live entry accounts for.
  `validate-local` runs it after the tools are staged and before a lane
  executes one. `validate-repository` builds the tools through
  `$(MAKE) validation-tools` and does not run the scan.
- `make validation-tools-rebuild` discards every staged binary, stamp and
  identity file through `guarded_rm`, rebuilds through the `validation-tools`
  lane, and reruns the path scan. The validation lock and the cache-check
  sentinel stay in place, so a rebuild beside an in-flight `validate-local`
  leaves that run's lock intact. It also runs `cargo clean` over the tool
  packages and `repo_root`, because discarding the copy alone leaves Cargo
  free to answer the next build from the shared build-dir with the artifact
  another worktree compiled. Use it when a staged tool fails on a path from a
  checkout that no longer exists.
- A removed worktree leaves its path in the shared build-dir's rlibs: on a
  host where two worktrees were removed mid-session, dozens of workspace
  rlibs under `<owner>/.cache/gate-cbuild/<hash>/validation/deps/` carried
  the removed paths in their debug sections, and an unstripped tool copy
  embedded eight references to one of them. Stripping the copy removed all
  eight, which is the measurement behind `stage_tool`; a reference that
  survives the strip is a runtime bake-in and the scan fails on it.
- sccache is host configuration, not a repository guarantee.
  `.cargo/config.toml` names it as the rustc wrapper; its cache size
  and location live in `~/.config/sccache/config`. A cache at its size
  ceiling evicts entries between sessions, so a gate tree of tens of
  GiB needs a ceiling above the tree size (48 GiB on the reference
  host).

### Canonical build

```bash
CARGO_TARGET_DIR="$(pwd)/.cache/gate-target" cargo build --workspace --profile validation
```

Per-crate, with validation semantics:

```bash
CARGO_TARGET_DIR="$(pwd)/.cache/gate-target" cargo clippy -p <crate> --all-targets --profile validation -- -D warnings
CARGO_TARGET_DIR="$(pwd)/.cache/gate-target" cargo nextest run -p <crate> --lib --cargo-profile validation
```

### Pre-push validation chain

| # | Check                    | Purpose                                                                          |
| - | ------------------------ | -------------------------------------------------------------------------------- |
| 1 | cache-check              | Soft cap (150G) + hard cap (200G) on `.cache/`                                   |
| 2 | terminology-gate         | 8 banned legacy terms; prefer `sign_imbalance` for the renamed crate vocabulary. |
| 3 | ansi-check               | Reject emojis and non-whitespace control characters; non-ASCII text passes       |
| 4 | rust-regression-scoped   | Scoped clippy + nextest on changed-crate closure                                 |
| 5 | validate-governance      | Verify registry policy, signatures, cross-references, and checked-in TOMLs      |

Verify hook state: `git config --get core.hooksPath` should print
`.githooks`. The pre-push hook runs repository validation and skips
the validation gate for pushes containing only branch deletions.

## Registry: SQLite-canonical (since 2026-03-23)

- Canonical write target: `registry/canonical/control_plane.sqlite3`.
- `registry/*.toml` are AUTO-GENERATED read-only compat exports.
  Every TOML file in `registry/` starts with the header
  `# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.`
- Source manifest: `registry/source_manifest.toml` declares the 36
  TOMLs that participate in compatibility round-trip verification.
- Architecture walkthrough:
  `docs/engineering/registry_canonical_architecture.md`.
- Audit metric taxonomy:
  `docs/engineering/repo_audit_metric_taxonomy.md`.

### Mutation workflow (MANDATORY)

1. Edit via `gororoba-db` CLI (in `crates/gororoba_db/`) against the
   SQLite.
2. Re-export compatibility lanes:
   `cargo run -p gororoba_cli_provenance --bin provenance -- export-control-plane`.
3. Refresh registry signatures:
   `make registry-integrity`.
4. Commit the SQLite delta + regenerated TOMLs + regenerated
   markdown together in a single atomic commit.

Never hand-edit a file whose first line is the AUTO-GENERATED
header. Doing so will desync the canonical store and produce
content_sha mismatches at the next `validate-governance` run.

`gororoba-db build` is an IMPORT, not a refresh. It calls
`build_fresh`, which deletes the database file and reloads only the
lanes named in `registry/source_manifest.toml`. Claim transition
events, the claim relations they allocate, and the claim revision log
have no lane there, so a rebuild returns those tables empty and no
compatibility TOML can restore them. The append-only triggers do not
catch it, because removing the file issues no DELETE. The command now
refuses when the database holds transition events and requires
`--allow-transition-history-loss` to proceed. Use step 2 above to
refresh exports; reach for `build` only when importing hand-authored
TOML into an empty or expendable database.

Typed empirical contracts use `gororoba-db claim set-evidence --spec
<contract.toml> --actor <actor> --reason <reason>`. Retrieval receipts use
`gororoba-db artifact record-retrieval --spec <receipt.toml>`. Both commands
retain append-only canonical history; destructive imports refuse databases
holding either history. Follow the export and signature steps above. See
`docs/engineering/registry_canonical_architecture.md` for specification fields
and the distinction between historical retrieval expectations and live prestate.

## Research epistemics

Scope: every claim, insight, and experiment that enters the registry.
The "Registry: SQLite-canonical" section above governs how a write
happens; this section governs what earns an ID.

Each rule below cites the repository's own record as its authority. A
rule with no claim ID behind it is a convention; a rule with one is a
finding, and the finding outranks the preference.

### Evidence layers

`docs/engineering/claim-theorem-identity-frontier-2026_08_04.md` fixes
three layers and one invariant: evidence at one layer never promotes a
claim at another.

| Layer                      | Records                                                          | Decisive evidence                                       |
| -------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------- |
| Source proposition         | What a paper, theorem, or cited equation asserts                 | The primary source, cited by name and section           |
| Implementation conformance | Whether code implements that proposition                         | A passing test, a Rocq `Qed.`, a reproducible run       |
| Phenomenological mapping   | Whether parameters or observables support the physical reading   | A preregistered comparison against matched controls     |

Implementation correctness is decisive for its own layer and carries
no weight at the layer above. The `hypothesis_class` distribution
makes the asymmetry measurable: 1221 `verified_claim` against 3
`falsifiable_thesis`. Reading the first count as a scoreboard for the
second is the error the invariant forbids.

Belief, coherence, analogy, and cross-domain recurrence generate
hypotheses. Recurrence becomes evidence only when the recurring
domains share no analysis pipeline, no normalization, and no lift;
recurrence through a shared pipeline measures the pipeline.

### Interpretive depth

Lifts, wavenumber selections, embeddings, normalizations, and
parameter fits stand between an algebraic structure and an observable.
Their count is itself a measurement, and the record shows it dominates
the result.

| Depth | Protocol                                     | Outcome in this repository                                                                                                                              |
| ----- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | Associator applied directly to a signal      | Structure-specific: scrambling the CD sign table at identical 1848-term support collapses ROC-AUC 0.8274 to 0.4750 (C-1632)                             |
| 1     | Algebraically-selected wavenumbers, stacked  | Degenerate null: CD-ZD, G2 Aut(O) and Albert J3(O) each return SNR 0.29 and sl(2) returns 0.23 (C-1372); D=16 through 262144 return identical RMS (C-1366) |
| 2     | 42 assessors -> Herm_3 -> mixing angles      | Fitted agreement at four free parameters through a lift proved non-S_3-equivariant, hence underivable from the algebra (C-1502, C-1489, C-1492)         |

Rules:

- A scientific claim MUST enumerate its intervening maps and declare
  the count.
- A signal MUST NOT be attributed to an algebraic structure until it
  is shown specific to that structure: it survives substitution of a
  different algebra, randomization of the structure at fixed support,
  and removal of each map in turn.
- A result identical across algebras or across dimensions measures the
  pipeline, not the algebra. C-1366 names the mechanism in its own
  case (the `assessor_fraction=0.5` identity).
- A lift that cannot be derived from the structure it expresses MUST
  be recorded as a construction with its free-parameter count, and
  MUST NOT be cited as evidence for that structure. C-1476 is the
  obstruction (rank-2 Jacobian lock on the 42D->3D family), C-1478 the
  construction that breaks it, C-1489 the concession that it is
  project-specific.

`ClaimEvidenceSpec` in `provenance_store::claim_evidence` records maps by
branch and kind, fitted parameter counts, and interpretive depth. A declared
`lift_depth` equals the maximum number of interpretive maps along any branch;
computational maps remain separately enumerated. `depth_status = "not_assessed"`
requires an omitted `lift_depth` and an explicit rationale. An enumerated
computational pipeline therefore does not establish physical interpretive depth.
C-1740, C-1741, and C-1743 carry typed contracts; their physical depth remains unassessed.

### Matched controls

A control MUST preserve support, dimensionality, free-parameter count,
temporal receptive field, and measurement procedure, and destroy only
the proposed causal structure.

- Temporal receptive field is a confound in its own right. I-212
  records the case: an advantage over a one-step baseline partly
  measured temporal asymmetry, and at matched six-sample support the
  ranking reversed, with maximum stepwise rotation at 0.8383 over the
  associator at 0.8274 (C-1633).
- A single control draw is an anecdote. C-1632 states its own limit --
  one ChaCha8 scramble at seed 42 -- and leaves the ensemble open. A
  dominance claim SHOULD report a null distribution, not one
  comparison.

### Falsifiers

`what_would_verify_refute` is a typed outcome table in the C-1740, C-1741, and C-1743
contracts, with separate verification, revision, abandonment, and inconclusive
outcomes. Each contract names decisive experiments and a protocol artifact.
`gororoba-db claim set-evidence` validates the declaration and appends its full
previous and new values before compatibility export. A registered falsifier
defines an adjudication rule; experiment evidence determines the claim outcome.

- A `falsifiable_thesis` or `research_claim` MUST state what would
  refute it, which experiment adjudicates, and which outcome forces
  abandonment. C-1498 is the worked example: a predicted delta_CP near
  93 deg against a measured 195 +/- 25 deg, "Testable by DUNE and
  Hyper-Kamiokande".
- A `verified_claim` at the implementation layer needs no separate
  falsifier, because the test is the falsifier.
- Identify the cheapest DECISIVE falsifier and run it before extending
  a theory. When no cheap test is decisive, record the theory as
  untested; a cheap indecisive proxy manufactures confidence and is
  worse than silence.

- Separate effect evidence, declared target attainment, mechanism specificity
  and practical utility. An investigator-chosen discrimination target needs an
  application-based justification before anyone calls it a usefulness threshold.
  C-1743 and C-1754 retain positive effects below the historical 0.005 ROC-AUC
  target; their practical utility remains unassessed. Preserve historical targets
  and outcomes while correcting active interpretations through canonical revisions.
  Target attainment alone supplies no general scientific pass/fail gate.

### Refuted structures as controls

Refutations are retained permanently and their closure reason is typed
(`Closed/Negative-Result`, `Closed/Obstructed`,
`Closed/Methodology-Insufficient`, `Closed/Analogy`, and the rest of
the vocabulary).

- C-020 is the standing specimen: a 16D zero-divisor adjacency matrix
  that "does NOT represent valid algebra. Found to be
  noise/hallucination when verified against commutator/parity
  matrices." Generated structure that survives visual inspection is
  the characteristic failure mode of LLM-assisted work here, and the
  defense is a mechanical cross-check, not a closer reading.
- A claim reusing a refuted claim's pipeline, lift, or normalization
  inherits that refutation as a prior and MUST state what differs
  before registration.
- Structural analogy raises the evidentiary burden and never lowers
  it. C-036, C-037, C-038, C-040 and C-041 are five refutations of one
  shape: algebraically motivated, numerically near, empirically false.
  An attractive analogy is registered as a conjecture carrying a
  discriminating prediction, and is promoted only when that prediction
  survives.

### Instruments over essays

An interpretive argument left in prose is unfalsifiable by
construction. Convert it into a schema field, a provenance record, an
executable query, or a preregistered comparison.

`registry/scorecard.toml` is the model: `parameter_count` is declared
per observable and the bins are defined by it, with nine bin-1 entries
at `parameter_count = 0` against a bin-2 PMNS fit at four. The
quantity such a scorecard measures is the derivative of agreement with
respect to parameter count, not agreement.

Tracked axes: free-parameter count, interpretive depth, symmetry
obligations discharged or waived, null-model performance, and the
falsification criterion.

Mathematics determines the invariants and the admissible
transformations. Experiment determines whether an invariant reaches an
observable. State each claim at the layer its evidence supports and no
higher; when the decisive experiment has not run, say so plainly and
run it next.

## GPU backend foundation (Wave B)

The workspace consolidates Vulkan / cubecl-wgpu / CUDA call sites
through three helper crates:

| Helper crate                             | What it owns                                                                                                                                                                                                                  | Replaces                                                                           |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `gororoba_gpu_vulkan` (`ash` feature)    | `InstanceBuilder`, `Adapter::pick`, `DeviceBuilder`, `ShaderModule::from_wgsl`, `DescriptorSetLayoutSpec`, `ComputePipelineBuilder`, `DispatchScope`.                                                                         | 10+ hand-rolled Entry::load + create_instance + queue-family scan sites.           |
| `gororoba_gpu_cubecl` (`cubecl` feature) | `Runtime::probe` (panic-safe wgpu adapter probe); `test_support::skip_if_unavailable!` macro.                                                                                                                                 | 4 hand-rolled 11-line `catch_unwind` probes.                                       |
| `gororoba_gpu_cuda` (`cudarc` feature)   | `Context::with_default_device`, `DeviceProbe::query`, `CompileOptions::for_arch`, `ModuleRegistry::load`, `Buffer<T>`, `ManagedBuffer<T>`, `LaunchConfig::launch_1d/2d/3d`, `telemetry::Telemetry`, `optix::PipelineBuilder`. | 48+ ad-hoc `CudaContext::new(0)` + 35+ NVRTC compile-options + 28+ PTX-load sites. |

All three crates default to no-op (empty `default` features).
Enabling the SDK feature pulls in the relevant deps. New code MUST
route through these helpers, not re-invent the boilerplate.

## Scientific stack pinning

- `statrs` 0.18.0 requires `nalgebra` 0.33; do not upgrade
  `nalgebra` without resolving `statrs` first.
- `gauss-quad` 0.2.4, `kodama` 0.3.0, `kiddo` 5.2.4, `petgraph` 0.7,
  `wide` 0.7.
- `cudarc` 0.19.1 (NVRTC runtime compilation), `burn` 0.16+.
- See the per-project MEMORY for pinning rationale and Rocq proof
  patterns (`ring_simplify+lra`, `cbv+ring_simplify`, tower rewrites,
  fuel recursion, Boolean reflection).

## Scientific debt classes (eleven)

Tracked in `plans/repo_debt_roadmap_2026_04_11.toml`:

1. `DEBT-NUMERICAL-ALGORITHM` (e.g., bounded_nelder_mead duplication)
2. `DEBT-STRUCTURAL-ARCHITECTURE` (e.g., data_core fetch/parse mixing)
3. `DEBT-DUPLICATION` (CPD clusters; PMD CPD lane in `make cpd-audit`)
4. `DEBT-TEST-VERIFICATION` (repository validation scope; coverage)
5. `DEBT-BUILD-WORKSPACE` (Makefile sprawl; xtask migration target)
6. `DEBT-DATA-PROVENANCE` (default features and network isolation)
7. `DEBT-GENERATED-ARTIFACT` (registry_mirrors quarantine)
8. `DEBT-DOCUMENTATION-REQUIREMENTS` (audit lane coverage in REQUIREMENTS.md)
9. `DEBT-SCIENTIFIC-EVIDENCE` (claims linked to reproducible artifacts)
10. `DEBT-SUPPLY-CHAIN` (cargo-deny, geiger, machete)
11. `DEBT-FORMAL-VERIFICATION` (admits, axioms, parameters tracked in registry)

## Stage references and baselines

- Stage A audit pack: `data/output/audit/2026-04-30/` (30+ artifacts).
- Debt baseline TOML: `data/output/debt_baseline_2026_04_30.toml`.
- Baseline git tag: `debt-baseline-v0` on commit `970b4da3`.
- Active in-repo roadmap: `plans/repo_debt_roadmap_2026_04_11.toml`.

## Comment + documentation hygiene policy

Scope: all agents (Claude, Codex, Gemini, Mistral, Ollama, DeepSeek,
human). All checked-in artifacts: source code AND markdown docs.

Sourced from `steinmarder/AGENTS.md` "Comment + documentation
hygiene policy"; adapted for the Rust + Rocq + WGSL surface in this
repo.

### Priority order

1. SOURCE CODE comments (HIGHEST -- code outlives docs).
2. MARKDOWN finding-doc bodies (SECONDARY).
3. FILE-HEADER license blocks (THIRD).

### Source code MUST NOT contain

| Banned pattern                           | Example                                        | Move to                                             |
| ---------------------------------------- | ---------------------------------------------- | --------------------------------------------------- |
| Task references                          | `// task #143`                                 | commit message                                      |
| Issue references                         | `// see issue #157`                            | commit message                                      |
| PR numbers                               | `// PR #25`                                    | commit message or `Closes #25:` trailer             |
| Companion-PR breadcrumbs                 | `// companion to PR #...`                      | commit message                                      |
| Phase labels                             | `// Phase 4.4`                                 | commit message                                      |
| Step-of-phase labels                     | `// Step 1 of Phase 3`                         | commit message                                      |
| Wave labels in code                      | `// Wave C2.1 migration`                       | commit message                                      |
| Session dates                            | `// (2026-05-15)`                              | commit message                                      |
| Deictic time                             | `// as of today`, `currently`, `previously`    | rewrite absolute                                    |
| Cross-phase breadcrumbs                  | `// will be exercised when Wave E lands`       | commit message                                      |
| Author tags                              | `// (eirikr)`, `// @claude`                    | delete entirely                                     |
| Deictic refs                             | `// this CD algebra`, `// our LBM solver`      | rewrite with the exact crate / module / type name   |
| Internal-repo paths                      | `// per data/output/audit/2026-04-30/foo.csv`  | rewrite by content, not by path                     |
| NEW personal-name copyright on new files | `// Copyright (c) 2026 <git config user.name>` | use the project-collective form below, or SPDX-only |

The `terminology-gate` hook catches the banned-term subset of this list
automatically, and `ansi-check` catches emojis and stray control
characters. Everything else here is reviewer-enforced, including the
typographic substitutes, which `ansi-check --fix` rewrites but
`ansi-check --check` accepts.

### Source code MUST contain (when domain-specific)

- Absolute identifiers (crate name + module path) for any
  cross-crate citation, so the reader can navigate without context:

| Subject                               | Canonical form                                                     |
| ------------------------------------- | ------------------------------------------------------------------ |
| Cayley-Dickson dimension              | "8D (octonions)", "16D (sedenions)", "32D (pathions)"              |
| LBM lattice                           | "D3Q19 BGK", "D3Q27 MRT"                                           |
| Backend                               | "Vulkan (ash)", "cubecl-wgpu", "CUDA (cudarc)"                     |
| Precision tier                        | "FP32", "FP16", "BF16", "FP8 e4m3", "INT8 SoA"                     |
| Claim / Insight / Experiment / Binary | `C-1234`, `I-0042`, `E-201`, `B-bin_name` (matches registry IDs)   |
| Rocq theorem                          | `Brown1972ChapterIII::brown_3_1_trace`, `Moreno1997::theorem_1_16` |

- Bit-field / SoA layout commentary
  (`SoA: f[channel * n_cells + cell_idx]`).
- Empirical numerical behaviour (`f32 GPU vs f64 CPU drifts ~1e-4
  relative per D3Q19 step at low Mach`).
- Mathematical invariants and non-obvious workaround rationale.
- Reference citation by name + chapter/section, not by line number
  or internal-repo path:

| WRONG (internal)                                    | RIGHT (public)                                                                                                      |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `// per data/sources/moreno_1997_zd_paper.txt:1572` | `// per Moreno (1997), "Zero divisors of the Cayley-Dickson algebras over the reals", J. Algebra 196, Theorem 1.16` |
| `// see audit_2026_04_30.csv row 12`                | `// see DEBT-DUPLICATION class in plans/repo_debt_roadmap_2026_04_11.toml`                                          |

### Markdown finding-doc rules

Markdown bodies MAY carry chronology (every finding-doc has dated
frontmatter -- `last_verified`, `evidence_class`, dated filename,
ordered predecessors). That is intentional.

PR# / task# references MUST triangulate with a durable identifier:

| WRONG (PR# alone)           | RIGHT (durable primary + PR# cross-link)                                                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `the fix landed via PR #38` | `landed in commit f79a37fa (cd_kernel/turboquant/cuda migration to gpu_cuda); PR #38 / branch session/wave-c2-cd-kernel-turboquant-cuda-2026-05-15 for cross-link` |
| `see issue #157`            | `see CLAIMS.md row C-1234 (issue #157 if still open)`                                                                                                              |

### File-header rules

Mesa-conforming reference: `mesa-26-gororoba/docs/submittingpatches.rst`
(mesa-26 branch). Mesa forbids `Co-authored-by:` trailers for AI
tools (reserved for human co-authors) and specifies the
`Assisted-by:` and `Generated-by:` trailers instead. This project
follows mesa's trailer policy AND keeps file headers minimal +
SPDX-focused.

### NEW source files in this workspace

When a license header is appropriate (match adjacent-file style;
many files in this workspace use SPDX-only with no Copyright line,
which is also fine):

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//
// <one-line file description>
```

- MUST NOT fabricate an individual personal-name Copyright line.
  LLM templates default to `Copyright (c) YYYY <git config user.name>`
  -- strip that.
- MUST NOT add `(LLM-assisted)` or any AI tag to the file header.
  AI disclosure goes in the commit trailer.
- SHOULD omit the Copyright line entirely (SPDX-only header) when
  adjacent maintained files do the same.
- License: most crates are `GPL-2.0-or-later`; a few (lbm_3d_cuda,
  quantum_core, stats_core) carry `MIT OR Apache-2.0`. Match the
  surrounding crate's `Cargo.toml` license field.

### EXISTING source files with upstream copyrights

Pre-existing headers on third-party files (cudarc, ash, cubecl
wrappers if any vendored) MUST be preserved verbatim. The license
itself requires preservation: "The above copyright notice and this
permission notice shall be included in all copies or substantial
portions of the Software."

When ADDING new content to an upstream file, the upstream copyright
stays; do NOT add a second project-collective Copyright line on top
of the upstream header.

### Commit messages and PR descriptions

Source of truth: `mesa-26-gororoba/docs/submittingpatches.rst`.

- MUST NOT use `Co-authored-by:` for AI tools -- mesa reserves this
  trailer for HUMAN co-authors.
- WHEN AI participated in the creative process, disclose with the
  mesa-canonical `Assisted-by:` trailer. This project's development
  is multi-LLM + human in the loop; the canonical concise form is:

  ```text
  Assisted-by: Claude (Opus 4.7 1M context), ChatGPT Codex (5.x), Gemini (Flash/Pro 3.x), Mistral, Ollama, DeepSeek
  ```

  List ACTUAL tools used per commit. One comma-separated
  `Assisted-by:` line is fine, or one line per tool -- both match
  mesa's example syntax `Assisted-by: TOOL (OPTIONAL: MODEL)`.
- WHEN AI generated almost the entire code change, use
  `Generated-by:` instead of `Assisted-by:`.
- Trivial / sub-copyrightable / mechanical changes (autocomplete,
  variable rename, format-fix) MAY omit disclosure per mesa policy,
  though SHOULD still note the tool to ease review.

### Past commits with `Co-Authored-By: Claude` trailer

Commits up through 2026-05-17 used the
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
trailer. This violates mesa policy (which forbids
`Co-authored-by:` for AI tools).

Resolution:

- Existing commits are HISTORICAL ARTIFACTS. We do not force-push
  to rewrite git history; the trailers remain in the log.
- NEW commits from the date this policy lands forward MUST use
  `Assisted-by:` instead.
- Crucially: a `git pre-commit` lint (TBD) will block new commits
  with the old trailer once the implementation lands.

### Short form

Invent no fake individual credit. Preserve all pre-existing credit
verbatim (license requires it). New files use SPDX-only headers
matching adjacent crate style. Commit trailers use mesa's
`Assisted-by:` (never `Co-authored-by:` for AI). List the actual
tools used.

### Commenting voice

Adopted from steinmarder/AGENTS.md "Commenting voice"; adapted to
Rust:

- A short paragraph opens with the claim the rest of the block rests on
  ("PUSH-scheme writes are race-free because dst = src + c_i is a
  bijection on cells for each fixed i.").
- Then the primary-source citation, by name -- the function
  (`LbmSolver3D::phase2_streaming`), the paper section, the type
  (`StoragePrecision::Fp32`) -- not an internal-repo file extract.
- Then the consequence in one or two lines, with a fragment of
  code inlined when that is clearer than prose.
- A test name or benchmark reference when the comment exists to
  explain a regression being fixed.
- Env knobs / feature flags described together at the bottom of the
  block when relevant.

The distillate -- subtle shifts on top of the existing voice:

- Default short. One-line trailing comments on the decisive line
  beat a function-header paragraph, unless the function as a whole
  encodes a non-obvious invariant.
- One thought per comment; stack them when steps are distinct,
  rather than fusing into a multi-clause sentence.
- Do not paraphrase the next line of code. The comment is about
  WHY the code is shaped that way -- the numerical constraint, the
  paper section, the measurement -- not what it does.
- When the code is mechanical, no comment. Comments earn their
  place by carrying information that does not survive in the code
  itself.
- Anchor citations by name (paper title + section, Rocq theorem
  identifier, registry C-/I-/E-/B- ID) rather than by line number
  or repo-internal path.
- Multi-paragraph blocks are reserved for the genuine
  numerical-quirk WHYs -- the kind of finding that took a long
  bench measurement or a Rocq proof to land.

Three prose-level leans -- subtle, but they are what makes a comment
feel like everything just makes sense:

- State mechanism in the active voice ("the GPU reads f_in and
  writes the periodic-neighbor cell"), not in the passive
  ("f_in is combined with the offset").
- Let sequence be the explanation. "X. Then Y. Then Z." -- one
  sentence per step, each doing one thing -- beats one sentence
  with three clauses.
- Chain each WHY by what the next step needs, so the comment moves
  forward instead of cataloguing in parallel.
  ("...so the descriptor write can resolve the relocation" tells
  the reader where the explanation is heading.)

Example shape:

> The PUSH-scheme kernel reads 19 own-cell f values. It computes
> rho, u, f_eq, applies BGK collision in registers, then writes
> each post-collision f_i to the periodic-neighbor destination cell.
> For any fixed direction i, src -> src + c_i is a bijection on the
> periodic lattice, so the 19 scattered writes per thread are
> race-free without atomics.

This composes with the comment-hygiene rules above (no task #, no
PR #, no Phase X.Y, no deictic refs). Together: short, active,
sequenced, primary-source-grounded, time-invariant.

### Markdown tracking policy

`.gitignore` denies `*.md` workspace-wide and re-admits named files.
The rationale is in the file itself: markdown is generated or
ephemeral by default, and the allowlist "keeps opportunistic LLM-spam
markdown out of the repo while permitting real report artifacts that
have ownership and removal policy."

- `AGENTS.md`, `CLAUDE.md` and `README.md` are allowlisted
  case-insensitively at every depth.
- The `docs/` tree admits specific paths only (`docs/THEOREMS.md`,
  `docs/REQUIREMENTS.md`, `docs/requirements/`, the generated theorem
  mirror). Markdown under `docs/book/` is mdBook-managed.
- Curated `docs/reports/` markdown MUST be governed by
  `registry/markdown_owner_map.toml` and the markdown inventory gate.
- A new tracked `.md` file MUST be allowlisted before it is written,
  not after `git add` fails.

### Validation and audit lanes

| Target                     | What it checks                                             |
| -------------------------- | ---------------------------------------------------------- |
| `make rust-clippy`         | Workspace clippy with deny warnings                        |
| `make rust-semver-check`   | semver-checks for crate API stability                       |
| `make cargo-deny-check`    | License, advisory, and source policy                        |
| `make dep-audit`           | cargo-audit advisory scan                                   |
| `make cpd-audit CPD_TOP=20`| PMD-driven duplication audit                                |
| `make docs-freshness`      | Generated docs match source registries                      |
| `make integrity`           | Verify lane (mirror + license + overflow)                   |
| `make registry-integrity`  | Regenerates `registry/schema_signatures.toml`               |

Any registry TOML edit drifts `schema_signatures.toml`; run
`make integrity-resolution` rather than patching signatures by hand.

### LLM-readable markdown style

For files an LLM agent might slice-load (AGENTS.md, CLAUDE.md,
finding-docs, memory entries):

- MUST use heading depth <= 3 levels.
- MUST use exactly one H1 per file (the document title).
- MUST include frontmatter on programmatically-loaded files.
- MUST use language tags on code fences (`bash`, `rust`, `toml`,
  `wgsl`).
- SHOULD prefer tables over bullet lists for 3+ comparable items.
- MUST NOT use emoji, ASCII boxes, banner dividers in rules text.
  `ansi-check` blocks the emoji; the boxes and dividers are on the
  author.
- MUST name the exact section or path instead of using relative
  position phrases; the file may be slice-loaded.
- MUST use MUST / MUST NOT / SHOULD imperative voice in rules.

## When ambiguity arises

- Use `AskUserQuestion` early and often.
- Use `TaskCreate`/`TaskUpdate` to plan and track work granularly.
- Prefer the Explore agent for broad codebase searches over running
  multiple greps yourself.
- Never invent file paths; verify with `find`/`grep` first.
- When in doubt about the registry, consult the architecture
  walkthrough at `docs/engineering/registry_canonical_architecture.md`
  before mutating.

## Workspace cross-links

This repo is the primary fluid-dynamics + algebra workspace. Peer
references for cross-cutting workspace docs:

- `~/workspaces/mesa/steinmarder/` -- C11 + CUDA + SASS-RE; source
  of this file's comment-hygiene policy.
- `~/workspaces/mesa/mesa-26-gororoba/` -- Mesa 26.1-devel fork;
  source of this file's `Assisted-by:` commit-trailer policy.
- `~/workspaces/mesa/steinmarder/docs/workspace/` -- cross-cutting
  ccache/distcc/sccache wiring documentation (referenced where
  relevant for cargo + rustc parallel-build hygiene).

## Claude Code notes

These notes came from the retired standalone `CLAUDE.md` loader and hold the Claude Code specifics that a tool-generic guide leaves out. A rule that applies to every agent lives in the sections above.

`AGENTS.md` is authoritative. When a Claude tool, skill, or agent
default conflicts with it, follow `AGENTS.md` and surface the conflict
to the user.

### Also read

| Source                                                                  | Carries                                                    |
| ----------------------------------------------------------------------- | ---------------------------------------------------------- |
| `~/.claude/CLAUDE.md`                                                   | Global user policy, which loads `~/AGENTS.md`              |
| `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/MEMORY.md` | Per-project memory index: claim counts, pitfalls, baselines |
| `data/output/debt_baseline_2026_04_30.toml`                             | Baseline debt numbers                                       |
| `plans/repo_debt_taxonomy_roadmap_2026_06_04.toml`                      | Active 22-class debt taxonomy                               |
| `docs/REQUIREMENTS.md`                                                  | Toolchain prerequisites                                     |

### Claude tool discipline

- **TaskCreate / TaskUpdate**: granular tracking is MANDATORY for
  multi-step work. Mark each subtask `in_progress` before starting and
  `completed` immediately after. One `in_progress` at a time.
- **AskUserQuestion**: ask EARLY when ambiguity would change the work.
  Do NOT begin implementation until the task list reflects the
  clarified scope.
- **Agent + Explore**: prefer the Explore agent for broad codebase
  searches over running 5+ greps directly. Use a specialised agent
  when its description matches the task.
- **Plan mode**: `ExitPlanMode` approves a plan; `AskUserQuestion`
  clarifies requirements within one. Do not substitute one for the
  other.
- **Registry writes**: never hand-edit a file whose first line is the
  AUTO-GENERATED header. Route through `gororoba-db` per the
  "Registry: SQLite-canonical" workflow in `AGENTS.md`.
- **`agents-render`**: MUST NOT be run. The renderer emits a 19-line
  stub and overwrites the hand-maintained `AGENTS.md`.

### Commit trailers for Claude

The full policy is in `AGENTS.md` under "Commit messages and PR
descriptions". The Claude-specific form:

```text
Assisted-by: Claude (Opus 5 1M context)
```

- USE `Assisted-by:`. Mesa reserves `Co-authored-by:` for human
  co-authors.
- DO NOT use the legacy
  `Co-Authored-By: Claude ... <noreply@anthropic.com>` trailer.
  Commits through 2026-05-17 carry it as a historical artifact; do not
  force-push to scrub them.
- Name the model actually used, not the string in this example.
- Commit bodies cover motivation, change, and evidence as prose in one
  to five sentences, cite primary sources by name, and list
  verification commands explicitly.

### Memory hygiene

- `MEMORY.md` is an index: one line per memory file, under ~150 chars.
  Topic files in the same directory hold the body.
- Save only what is non-obvious from the codebase, git history, or
  docs. Code patterns, file paths, and recent commits are derivable.
- Feedback memories record the reason behind the rule, so the edge
  cases stay judgeable. A bare rule rots faster than rule plus reason
  plus when-to-apply.
- Project memories convert relative dates to absolute on save.
- Before recommending from memory, verify the named files, functions,
  and flags still exist. A memory is a snapshot; renames happen.
