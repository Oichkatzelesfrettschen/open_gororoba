---
description: Instrumented architecture map, coefficient derivation, review triage, and falsifiable work queue
last_verified: 2026-08-11
evidence_class: instrumented-static-capture
status: active
---

# Repository architecture and evidence map

This map records the live architecture at commit `3ad0b92214ac01ad319fbc85815095df9d2678af`.
It distinguishes implemented mechanisms, static evidence, host capability, and
unexecuted scientific propositions. The companion artifact is
`data/output/audit/2026-08-10/repository-architecture-static-map.toml`.

The workspace contains 76 packages, 752 targets, 502 binaries, and 261
workspace-declared dependency edges. The canonical SQLite registry contains
1,543 claims, 183 insights, 232 experiments, 502 binaries, 162 theorem rows,
186 claim-transition events, 90 claim relations, and 431 claim revisions. These counts describe
registered material. They do not establish scientific truth or runtime parity.

## Evidence boundary

The capture maps 2,498 handwritten source inputs. It excludes generated Rust
registry mirrors from the source graph because they echo registry data and can
manufacture false self-confirmation in a text search. Universal Ctags records
56,575 symbols. Cscope builds cleanly over the same input list. Cflow records
the C-like declaration surface of eight CUDA translation units.

| Tool | Established result | Explicit limit |
| --- | --- | --- |
| Ctags | 56,575 symbols over handwritten source inputs | Names and lexical locations are not calls or ownership edges. |
| Cscope | Clean lexical definition and text-reference index over 2,498 inputs | Rust callee queries have no semantic call edges. |
| Cflow | `kernels.cu` has 17 declared entries; `kernels_soa.cu` has 22 declared entries | CUDA launch semantics and device execution are outside its C parser. |
| rust-analyzer | It reaches 752 crates, 2,899 modules, 36,174 declarations, and 31,555 bodies | The bounded run times out after 55 seconds and reaches about 3.7 GiB. |
| Cargo metadata | 76 packages, 752 targets, 502 binaries, and 261 workspace edges | It does not represent runtime feature activation or dynamic dispatch. |

The capture records Cscope, Cflow, and Ctags hashes in the companion artifact.
Raw maps remain ephemeral because they are reproducible derived outputs. The
artifact records their input hashes, commands, base commit, and interpretation
limits. A changed source hash invalidates the static map.

The host exposes an NVIDIA AD104 GeForce RTX 4070 Ti with compute capability
8.9, CUDA user-mode driver 13.3, and Vulkan API 1.4.341. This confirms that the
host can run CUDA and Vulkan work. It does not establish a successful kernel
launch, numerical stability, or CPU/GPU agreement.

## Control-plane architecture

The SQLite database is the authority. Compatibility TOMLs, Rust mirrors, and
generated documentation are derived representations. The only valid mutation
route is a typed `gororoba-db` operation followed by a control-plane export and
the derived generators.

```text
gororoba-db claim transition plan
  -> ProvenanceStore::plan_claim_transition
  -> validation-only transition plan

gororoba-db claim transition apply
  -> ProvenanceStore::apply_claim_transition
  -> SQLite claim, event, relation, and revision transaction
  -> provenance export-control-plane
  -> compatibility TOMLs
  -> registry refresh
  -> Rust mirror emission and governance validation
```

`plan_claim_transition` performs the non-mutating admission check. `apply_claim_transition`
commits the status change, append-only event, and relations atomically. A
compatibility TOML is never a source of authority for an existing canonical
database.

This change repairs eight structural failure modes in this path.

| Mechanism | Prior failure | Enforced result |
| --- | --- | --- |
| Export ordering | `registry-export-markdown` refreshed derived consumers before exporting SQLite changes. | It now exports the control plane before refresh and mirror emission. |
| Derived registry coverage | The export target omitted semantic-atoms, evidence-provenance, and final integrity generation. | It now derives semantic and provenance records, signs the complete registry, then emits Rust mirrors. |
| Rebuild protection | `gororoba-db build` guarded transition events but missed revision-only history. | It now refuses a revision-only rebuild without `--allow-transition-history-loss`. |
| Lacuna projection | A generated claim-status ID could collide with a retained manual lacuna and disappear silently. | Generation now fails with the colliding ID and preserves the conflict for repair. |
| Markdown CLI topology | Legacy Make targets invoked removed `markdown-registry` subcommands. | Legacy labels now call the supported inventory, owner-map, corpus, and embedded-content verifiers. |
| Cargo path propagation | Three xtask commands replaced configured Cargo paths with worktree-local defaults before launching nested Cargo processes. | Nested builds retain nonempty `CARGO_HOME` and `CARGO_TARGET_DIR` values, and use local defaults only when no paths are configured. |
| Narrative source provenance | Knowledge-atom verification required retired Markdown paths even though atom extraction reads a retained registry body. | Each source row records the derivation input, origin-path state, and canonical-body SHA-256; verification rechecks the retained body rather than treating an absent historical path as data loss. |
| Generated metadata extraction | HTML provenance comments containing `->` entered the equation-atom corpus as mathematical relations. | The extractor skips complete HTML comments before equation parsing, and a focused test rejects that false relation while retaining real inline mathematics. |

The completed repairs protect evidence integrity. They do not adjudicate any
claim's scientific status.

The initial export regenerates the Rust claim mirror from 1,478 records to the
1,543 records held by the canonical SQLite database. The 65-record difference
is stale representation, not claim deletion. The complete derivation path now
prevents that divergence from reaching the final mirror emission.

## Narrative provenance boundary

The knowledge-atom extractor reads `body_markdown` from
`registry/research_narratives.toml` and
`registry/data_artifact_narratives.toml` when that retained body exists. When a
registered document has no retained body, it reads and hashes the existing
`source_markdown` file. The associated path identifies a pre-migration Markdown
location for a retained body and an active input for a body-less row.

Twenty historical locations are absent from the working tree. Their retained
bodies remain in the registry and still participate in equation and proof-atom
extraction. `structured_corpora.toml` now records one of four derivation inputs:
control-plane claim rows, working-tree Markdown, registry body Markdown, or a
registry document payload. It records whether the historical location is present
or absent and retains the SHA-256 of the exact derivation body. Verification
recomputes a retained-body hash from the registry, recomputes a live-body hash
from disk, and rejects a changed path state.

This preserves the distinction between historical provenance and active source
files. It does not recreate missing Markdown, elevate an archived narrative to
a scientific result, or conceal a missing retained body.

The same capture reveals a second boundary: generated HTML provenance comments
are metadata, not equations. Several retained bodies contain `Source of truth`
comments with an arrow token, and the former heuristic classified those lines as
mapping relations. The extractor now excludes complete HTML comments before
parsing. The generated equation corpus therefore represents source expressions
rather than the generator's own registry references. The rebuild reduces the
corpus from 374 to 317 equation atoms. The 57 removed rows are generated
metadata false positives, not deleted mathematical derivations.

## D3Q19 coefficient derivation

`crates/lbm_3d/src/lattice.rs` defines the CPU reference lattice. It contains
one rest direction, six axis directions, and twelve face diagonals. The weights
are `w_0 = 1/3`, `w_axis = 1/18`, and `w_diag = 1/36`, with `cs_sq = 1/3`.

The normalization follows directly from the multiplicities:

```text
sum_i w_i = 1/3 + 6 * (1/18) + 12 * (1/36) = 1.
```

Opposite velocity pairs cancel the first moment. The weighted second moment
therefore has the isotropic D3Q19 form `sum_i w_i c_i,a c_i,b = cs_sq delta_ab`
with `cs_sq = 1/3`. The code checks weight normalization, the speed of sound,
opposite-velocity symmetry, and equilibrium normalization. It does not yet
retain a direct executable second-moment tensor test. That omission is a
specific P1 verification gap, not evidence that the coefficients are wrong.

The equilibrium implementation evaluates:

```text
f_eq_i = rho * w_i * [1 + (c_i dot u)/cs_sq
                       + (c_i dot u)^2/(2 * cs_sq^2)
                       - (u dot u)/(2 * cs_sq)].
```

The BGK relation in `crates/lbm_3d/src/solver/bgk.rs` gives
`nu = cs_sq * (tau - 1/2)`. Nonnegative kinematic viscosity requires
`tau >= 1/2`. Population positivity and numerical stability require separate
conditions. This derivation connects the coefficient table, equilibrium
normalization, and viscosity constraint. It does not substitute for a
convergence study, a Mach limit, or a GPU comparison.

## Compute backend maps

### CPU LBM

```text
D3Q19Lattice::new
  -> velocities, weights, cs_sq
  -> D3Q19Lattice::equilibrium
  -> BgkCollision or MRT solver state
  -> LbmSolver3D collision and streaming steps
```

The CPU lattice is the numerical reference surface. A parity claim requires an
identical initial distribution, boundary condition, precision declaration,
step count, and compared observables. Matching a final image alone is not a
sufficient parity condition.

### CUDA LBM

```text
Rust solver selection
  -> CUDA module key and source or AOT cubin
  -> gororoba_gpu_cuda::ModuleRegistry
  -> named CUDA function lookup
  -> gororoba_gpu_cuda::LaunchConfig
  -> CUDA kernel declaration in a .cu translation unit
```

Cscope identifies `lbm_step_fused_kernel` in `kernels.cu`. Cflow identifies
its lexical helper surface. Cscope also finds `lbm_step_soa_fused` in both
`kernels_soa.cu` and `kernels_dark_halo.cu`. Equal symbol names are not a
collision by themselves because separate CUDA modules can own independent
namespaces. The required next observation is the loaded module key paired with
the source hash and selected function name for every dispatch path.

### Expanded CUDA and control-plane capture

The expanded capture runs at commit `dd5fcb8b709479393bb87011e58646616df327bb`
with Universal Ctags 6.2.1, Cscope 15.9, and GNU Cflow 1.8. It covers all 34
CUDA translation units under `crates/lbm_3d_cuda/src` and six registry/control-
plane Rust and Make inputs.

| Surface | Established result | Explicit limit |
| --- | --- | --- |
| CUDA Ctags | 206 function definitions, 198 unique names | CUDA declarations are lexical symbols, not compiled module ownership. |
| CUDA Cscope | 42 Rust-requested kernel names; 42 have at least one CUDA definition; two names have two definitions | Cross-language lookup does not prove which source string reaches NVRTC. |
| Per-translation-unit Cflow | 319 function-shaped records and 98 `__global__` records | 1,408 parser diagnostics remain: `kernels_soa.cu` 910, `kernels_sparse_lbm.cu` 468, and `kernels_sparse_map.cu` 30. CUDA launch syntax and device execution remain outside the C parser. |
| Registry Cscope | Six bounded inputs, zero index diagnostics | Rust queries remain lexical and do not establish typed caller edges. |

The normalized input and output hashes are retained in the companion TOML
artifact. The source list hash is
`a6ba175e4f54a06506358e842f9bf3475bce02e27cca097b39c80cc6dbd828fe`.

The expanded control-plane call map is:

```text
gororoba-db claim transition plan
  -> cmd_claim_transition_read_only
  -> ProvenanceStore::plan_claim_transition
  -> validation-only allocation, status, and evidence checks

gororoba-db claim transition apply
  -> ProvenanceStore::apply_claim_transition
  -> SQLite immediate transaction and append-only history
  -> maybe_regen_toml when --regen-toml is set
  -> cargo run -p gororoba_cli_provenance --bin provenance -- export-control-plane
  -> run_export_control_plane
  -> ProvenanceStore::export_control_plane_compat_paths
  -> compatibility TOMLs, theorem markdown, transition exports, and run record

xtask registry-emit-all-mirrors
  -> run_registry_emit_all_mirrors
  -> one validation build of registry-emit and markdown-registry
  -> 23 typed mirror entries
  -> Rust registry mirror outputs
```

### CUDA finding resolution

The selector-to-runner path now has one typed contract. `KernelDispatchSpec`
owns the source label, step symbol, init symbol, storage width, launch mode,
and cells-per-thread value. `select_optimal_kernel` returns that contract in
`KernelSelection`, and `SoaBenchRunner::new_selected` consumes it. The A-A
tiers use a single distribution buffer plus a parity argument; pull tiers keep
the ping-pong launch shape. The selector unit tests compare every returned
field with its dispatch contract.

`kernels.cu` is the sole owner of
`update_tau_from_voudon_frustration_kernel`. The orphan
`kernels_voudon.cu` source is removed from the checkout. The handwritten
`crates/lbm_3d_cuda/cuda_source_ownership.toml` manifest records the only
non-runtime CUDA fixture, and `cargo run -p xtask -- cuda-source-ownership`
fails when a production `.cu` file loses its Rust `include_str!` or
`include_bytes!` edge.

`ModuleRegistry` now records `ModuleProvenance` for every load. NVRTC paths
retain the source SHA-256, compile-options SHA-256, source label, sorted
kernel list, and derived module ID. Opaque PTX/CUBIN paths retain the supplied
artifact label and a deterministic symbol-set identity; callers can provide an
artifact digest through `load_with_identity`. `get_with_provenance` binds a
resolved kernel name to that module record.

The bounded evidence is `cargo check` for `gororoba_gpu_cuda`, `lbm_3d_cuda`,
and `xtask`; `cargo test` for the selector and CUDA provenance units; clippy
with `-D warnings`; the ownership verifier; and a source search proving one
Voudon definition. The selector bridge also preserves the FP8 SM 8.9 admission
rule and labels AoS modules with their owning source constant. These checks do
not claim a CUDA-device launch or numerical parity run.

The post-resolution lexical refresh covers 33 CUDA translation units. Universal
Ctags reports 205 function records and 198 unique function names. Cscope finds
one Voudon definition in `kernels.cu`; the intentional `lbm_step_soa_fused`
module-name duplication remains in `kernels_soa.cu` and `kernels_dark_halo.cu`.
GNU Cflow reports the same 1,408 parser diagnostics as the baseline, so the
diagnostics remain parser-coverage limits rather than evidence of a new source
defect.

The Makefile remains the ordering boundary between the export command, registry
refresh, integrity generation, and mirror emission. The static graph confirms
the order and ownership; it does not execute the mutation or validate the
generated bytes.

The expanded CUDA host-to-device map is:

```text
LbmSolver3DCuda::new or BenchKernelRunner::build
  -> include_str! CUDA source constant
  -> ModuleRegistry::compile_and_load
  -> ModuleRegistry::get named function
  -> LaunchConfig and stream.launch_builder
  -> CUDA function declaration in the selected translation unit

UnifiedInt8Runner::new
  -> kernels_int8_soa.cu source
  -> ModuleRegistry::compile_and_load
  -> ephemeral INT8 step and init functions
  -> Unified Memory launch path
```

Two integration findings emerge from the ownership census:

1. `kernel_selector::select_optimal_kernel` is public and its selection table
   is covered only by module-local tests. No production caller consumes its
   `KernelSelection`, `kernel_name`, or `source_label`. The 42-name census shows
   that the selected names exist, but the policy remains advisory and can drift
   from the constructors in `bench_kernels.rs`. This is a P1 integration gap,
   not a proven runtime failure. The required repair is a typed selector-to-
   runner bridge with a source/name/launch manifest and one test per tier.
2. `kernels_voudon.cu` defines
   `update_tau_from_voudon_frustration_kernel`, while `kernels.cu` defines the
   same symbol and is the only source included by `LbmSolver3DCuda`. No Rust
   inclusion or build edge reaches `kernels_voudon.cu`; `test_bf16.cu` is also
   unreferenced by the Rust source graph. The current graph therefore has no
   duplicate loaded module, but it does have a dead duplicate source owner that
   can drift silently. This is a P1 ownership and evidence gap. The required
   repair is to designate one owner or classify the file as a fixture, then
   enforce the decision in a source-to-module manifest.

`ModuleRegistry` stores a loaded module and pre-resolved names, but it does not
retain a source hash or module identity. That confirms the existing
`cuda-module-symbol-namespace` queue row: a runtime dispatch manifest is still
needed before source ownership, launch shape, and parity can be claimed.

### Vulkan LBM

```text
InstanceBuilder
  -> Adapter::pick for a compute queue
  -> DeviceBuilder
  -> ShaderModule::from_wgsl
  -> descriptor and compute-pipeline construction
  -> DispatchScope submit and synchronization
```

`lbm_vulkan` uses the shared `gororoba_gpu_vulkan` helpers rather than local
instance and queue boilerplate. The observed host Vulkan capability permits a
future dispatch probe. It is not a dispatch result.

### Formal proof and extraction

```text
Rocq source
  -> proofs/Makefile vos interface check
  -> proofs/Makefile vok body check
  -> optional extracted C surface
  -> Cscope and Cflow lexical map
```

The extracted C files support source-navigation only. A Cflow edge in extracted
output is not a theorem proof and not an executable correspondence result. A
research-quality proof record binds theorem identifier, assumptions, toolchain,
`vos` and `vok` outputs, source hash, and the claim or experiment that consumes
the theorem.

## Merged-review census

The live GitHub census covers 133 merged pull requests and 324 review threads.
GitHub marks 82 threads unresolved: 81 current and one outdated. Fifty-two
threads come from `chatgpt-codex-connector`; thirty come from CodeRabbit. The
current repository has no open pull request.

An unresolved GitHub thread is not equivalent to a current defect. The census
contains source fixes, generated-registry observations, review comments against
already superseded code, and scientific objections requiring a new experiment
or proof. The review boundary is therefore explicit.

| Review class | Current disposition | Required evidence before closure |
| --- | --- | --- |
| Structural mechanism defect | Fixed in this change where the live code demonstrates the failure. | Targeted test and the affected registry gate. |
| Generated representation | Do not hand-edit a compatibility TOML. | Canonical SQLite mutation, export, signature refresh, and mirror verification. |
| Scientific-status assertion | Leave the status unchanged in this change. | The stated primary source, declared run, independent check, and a canonical transition spec. |
| Historical or superseded observation | Preserve the thread as historical metadata. | A code comparison at the review commit and current `main`. |

The present repairs close the export-order, revision-only-history, lacuna-ID,
and documentation-policy classes. They intentionally do not use a review
comment as authority to change a scientific claim. The science comments remain
actionable work items only when their evidence path is replayed.

## Prioritized work queue

| Priority | Mechanism | Discriminating question | Required artifact | Completion gate | Falsifier |
| --- | --- | --- | --- | --- | --- |
| P0 | `review-evidence-disposition` | Does each current review observation still match `main` and its cited evidence? | A canonical review-disposition record keyed by PR, thread, path, current commit, and outcome. | Every current thread is classified as fixed, rejected with evidence, or converted to a canonical task. | A thread lacks a current-code comparison. |
| P1 | `d3q19-moment-invariant` | Do the listed D3Q19 coefficients satisfy all zeroth, first, and second moment identities in code? | A CPU test that evaluates the full tensor against `cs_sq delta_ab`. | Exact rational identities pass before floating tolerance projection. | Any off-diagonal term or diagonal coefficient differs. |
| P1 | `cuda-selector-dispatch-bridge` | Does every `KernelSelection` resolve to the source, constructor, and launch shape that execute it? | `KernelDispatchSpec` plus `SoaBenchRunner::new_selected` and selector contract tests. | Resolved in the bounded Rust test and clippy lanes; GPU launch remains unexecuted. | A selector result has no production caller or resolves to a different source/name pair. |
| P1 | `cuda-voudon-source-owner` | Which translation unit owns `update_tau_from_voudon_frustration_kernel`? | Production `kernels.cu` ownership plus `cuda_source_ownership.toml` and the xtask gate. | Resolved: one CUDA definition remains and every non-fixture source has a Rust include edge. | Two source files define the symbol without a declared ownership rule. |
| P1 | `cuda-module-provenance` | Which source and module identity produced a dispatched CUDA symbol? | `ModuleProvenance`, `KernelProvenance`, source SHA-256, option SHA-256, and module-ID unit tests. | Resolved for NVRTC source paths; opaque AOT paths require a caller-supplied artifact digest for content-level proof. | A dispatch record lacks a module ID or claims a source hash it cannot observe. |
| P1 | `cuda-module-symbol-namespace` | Which module owns every dispatched CUDA symbol? | A dispatch manifest containing module key, source or cubin hash, symbol, launch shape, precision, and feature set. | Every production dispatch resolves to one declared module-symbol pair. | One module contains duplicate exported names or a selected symbol lacks an owner. |
| P1 | `d3q19-backend-parity` | Do CPU, CUDA, and Vulkan implementations preserve the same declared observable under identical inputs? | A finite parity matrix with initial-state hash, grid, precision, steps, mass, momentum, NaN count, and field norm. | Each backend passes the declared tolerance or records a bounded divergence cause. | A conservation or field-norm discrepancy exceeds the stated tolerance. |
| P1 | `formal-evidence-binding` | Does each claim using a theorem bind the theorem identity, assumptions, proof output, and source hash? | Typed theorem-evidence records and dated `vos` plus `vok` logs. | Every linked formal claim has one replayable proof evidence record. | A claim links a theorem without assumptions or kernel output. |
| P1 | `claim-status-replay` | Does each review-requested claim status follow from a registered experiment and independent verifier? | Transition plans with artifact IDs, experiment IDs, falsifier result, and source evidence. | A planned transition passes `plan_claim_transition` and then an evidence review. | A status depends on an unregistered, non-reproducible, or non-independent result. |
| P1 | `narrative-source-restoration` | Which retained historical bodies warrant a reviewed working-tree Markdown restoration? | A source-by-source restoration decision with canonical body hash, evidence class, and owner. | Each absent historical location is either restored as an explicitly generated mirror or retained only in the canonical registry. | A restored file differs from its recorded canonical body or appears without an evidence-class decision. |
| P1 | `equation-atom-semantic-classification` | Does each extracted relation denote mathematics rather than an operational or documentation token? | A typed extraction class with source-kind filters and a reviewed false-positive sample. | Every equation atom has a mathematical relation or an explicit non-mathematical class. | A provenance, command, or path marker appears as a mathematical atom. |
| P2 | `validation-package-boundary` | Can registry validation avoid the broad `gororoba_cli_data` dependency closure? | A dependency and timing comparison for a narrow validator package. | The new boundary preserves all gates and reduces measured closure cost. | Any required gate becomes unreachable or output diverges. |
| P2 | `static-capture-automation` | Can the architecture map regenerate from one typed Rust command? | A Rust `xtask` subcommand that writes the map, input list, tool versions, and hashes. | Two clean runs at one commit produce identical normalized manifests. | A changed source hash leaves the manifest unchanged. |

The queue stays finite and evidence-bound. A new scientific action enters only
with a named owner, source or input artifact, falsifier, and completion gate.

## Reproduction commands

Run static capture in an isolated worktree. The commands record static evidence
only and must not be represented as runtime validation.

```bash
CARGO_TARGET_DIR=.cache/exp-repository-architecture-map-target \
  cargo metadata --no-deps --format-version 1

ctags --languages=Rust,C,C++,Cuda,Make,TOML --recurse --fields=+KSn \
  --extras=+q -f /var/tmp/tags-handwritten -L /var/tmp/analyzer-inputs-handwritten.txt

cscope -b -q -k -i /var/tmp/analyzer-inputs-handwritten.txt

cflow --no-preprocess --depth=3 crates/lbm_3d_cuda/src/kernels.cu

CARGO_TARGET_DIR=.cache/exp-repository-architecture-map-target \
  cargo nextest run -p lbm_3d --lib --cargo-profile validation
```

Run the canonical registry workflow after an authorized registry mutation:

```bash
CARGO_TARGET_DIR=.cache/exp-repository-architecture-map-target \
  cargo run -p gororoba_cli_provenance --bin provenance -- export-control-plane
make registry-integrity
make validate-registry
```

The repository architecture now has a bounded map, explicit evidence limits,
and an ordered set of discriminating next experiments. This supports strong
engineering claims without inflating static structure or hardware capability
into scientific proof.
