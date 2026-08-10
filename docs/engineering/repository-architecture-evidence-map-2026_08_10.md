---
description: Instrumented architecture map, coefficient derivation, review triage, and falsifiable work queue
last_verified: 2026-08-10
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
