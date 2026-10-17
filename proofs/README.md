# Rocq proofs and plugin compatibility

The proof toolchain targets Rocq 9.3.0, opam 2.6.1, OCaml 5.5.1,
Rocq Stdlib 9.2.0, and Flocq 4.2.2. The release inventory was checked on
2026-10-16. Rocq core and Stdlib have independent release versions:
Stdlib 9.2.0 is the latest stable library for the Rocq 9.3.0 compiler.
The commands below reproduce installation and verification. The measured
local results appear in the verification table.

## Install through opam

Run from the repository root with opam 2.6.1 available on `PATH`:

```bash
opam update
opam switch create rocq-9.3 ocaml-base-compiler.5.5.1 --jobs="$(nproc)" --yes
opam repository add rocq-released https://rocq-prover.org/opam/released --switch=rocq-9.3 --yes
opam update
opam install --switch=rocq-9.3 rocq-core.9.3.0 rocq-stdlib.9.2.0 coq-flocq.4.2.2 --jobs="$(nproc)" --yes
opam install --switch=rocq-9.3 rocq-elpi.3.5.1 rocq-hierarchy-builder.1.10.3 rocq-mathcomp-ssreflect.2.6.0 coq-ext-lib.0.13.2 --jobs="$(nproc)" --yes
opam repository add gororoba-proof-ports "$PWD/proofs/opam" --switch=rocq-9.3 --yes
opam install --switch=rocq-9.3 rocq-equations.1.3.2+9.3 --with-test --jobs="$(nproc)" --yes
opam list --switch=rocq-9.3 --installed
opam exec --switch=rocq-9.3 -- rocq --version
opam exec --switch=rocq-9.3 -- make -C proofs check-plugins
```

For an existing switch, inspect `opam list --switch=rocq-9.3 --installed`
before running the install commands. The package constraints request the
listed versions explicitly. `opam update` refreshes repository metadata;
`opam install` performs the package changes. Installing plugins through the
same switch rebuilds their native components against its compiler runtime.

The Equations overlay selects upstream commit
`d56dfc2c38feb5054d75ef43540ee89a6beb1a7e` from its `9.3` branch and pins
the source archive checksum. The patch changes development dependency
metadata to Rocq core 9.3.0 and Stdlib 9.2.0. The recipe requests upstream
examples and tests with `--with-test`. A recipe and an upstream port branch
establish source availability; the opam test result establishes execution.
The recipe declares the patch in `extra-files` with its own checksum so opam
copies the verified payload into the source build.

Elpi 3.5.1 requires Dune below 3.24, so the coherent package set uses Dune
3.23.1. The hosted bootstrap pins that version before compiling Rocq to avoid
rebuilding the compiler when plugins are added. `check-plugins` exercises
Equations-generated equations, Elpi commands, HB structures, and MathComp
sequence lemmas. MathComp 2.6 replaces the deprecated `all_ssreflect` import
with `boot` and `order`. The integration test requires those modules without
importing their global arithmetic notations and uses qualified sequence APIs,
which coexist with Stdlib under the fatal default-warning policy.

The final opam solver check reports the installed set as up to date within its
constraints. Dune and dune-configurator 3.24.2 and the Elpi engine 3.9.2 remain
outside that solution: the Rocq Elpi bridge requires Dune below 3.24 and Elpi
below 3.8. A bridge port must adapt and test the engine API and Dune build
metadata, then pass its native-plugin, command, HB and MathComp consumers.
Removing the bounds alone establishes dependency resolution, not compatibility.

## Build every registered proof with all visible CPUs

Run from the repository root:

```bash
opam exec --switch=rocq-9.3 -- make -C proofs --jobs="$(nproc)" clean
opam exec --switch=rocq-9.3 -- make -C proofs --jobs="$(nproc)" --keep-going all
opam exec --switch=rocq-9.3 -- make -C proofs --jobs="$(nproc)" check-admissions
```

`all` follows `_RocqProject` dependencies and compiles the listed theory and
verified sources to `.vo` objects with proof bodies checked. GNU Make passes
its jobserver to the generated proof Makefile. `nproc` supplies every
process-visible CPU; independent files build concurrently.

Rocq 9.3 offers Findlib package installation for projects with exactly one
logical root. The proof corpus uses two logical roots and builds local objects,
so the generator selects its standalone mode explicitly. The generator's
remaining warnings and compiler warnings retain their fatal checks.

Compiled `.vo` objects and native plugins carry compiler and OCaml runtime
identity. A compiler upgrade requires a source-closed rebuild of the proof
corpus and its external libraries in the selected switch. Copying Flocq or
other compiled objects from an older switch can fail even when the logical
library names agree. `clean` removes generated proof products; opam owns the
external library rebuilds. Preserve retained evidence bundles separately.

`make -C proofs vos` checks interfaces while skipping opaque proof bodies.
`make -C proofs vok` checks bodies against available interfaces. Those targets
support iteration; `all` produces the ordinary compiled objects required for
kernel replay and extraction. Admission scans, kernel replay, declaration
assumptions, and extraction correspondence measure separate surfaces. Retain
command output, tool versions, source hashes, and exit statuses before
publishing proof results.

## Capabilities and remaining source ports

The main proof corpus uses Stdlib/Corelib and repository modules. Its external
imports are `Flocq.Core` in `theories/FP24Representable.v` and
`theories/R2VBTransformDP4.v`. Optional extraction requires a larger dependency
graph. Released package bounds below describe the refreshed opam inventory;
source availability alone does not establish a passing Rocq 9.3 build.

| Component and capability | Published compatibility | Concrete Rocq 9.3 work and decisive check |
| --- | --- | --- |
| Elpi 3.5.1: quotations, environment access, tactics and commands | Rocq below 9.4; Elpi 3.7.x | Install released packages; compile and execute representative commands. |
| Hierarchy Builder 1.10.3: factories, builders and packed-class compatibility | Refreshed metadata accepts Rocq 9.0 through 9.3 | Install with Elpi; exercise hierarchy declarations and upstream tests. |
| MathComp ssreflect 2.6.0: reflection, finite structures, sequences and order | Split into same-version boot and order packages; boot accepts Rocq >= 9.0 | Install coherent 2.6.0 packages and compile representative imports and proofs. |
| ExtLib 0.13.2: functional interfaces and reusable proof support | Installable dependency candidate for the selected compiler | Run package installation and the tests of its actual consumers. |
| Equations: dependent matching, recursion, derived equations and elimination | Released 1.3.2+9.2 requires Rocq below 9.3 | Use the checked-in 9.3 source overlay; require its examples and tests to pass. |
| MetaRocq: reification, PCUIC, verified checking and erasure | Published 1.5.1+9.2 requires Rocq below 9.3 | Port all exact-version sibling packages together. Select a source revision compatible with 9.3 APIs; check universe contexts, quotation/unquotation, checker and erasure proofs. A failed plugin load or kernel proof rejects the port. |
| CompCert: verified C semantics and compilation support | 3.18 accepts OCaml 5 but requires Rocq below 9.3 | Port theories and extraction against 9.3 and run compiler verification and tests. Upstream configure accepts 9.0-9.2; removing that guard alone supplies insufficient evidence. |
| WasmCert: WebAssembly semantics and extraction | 2.2.0 requires Rocq below 9.2, MathComp <= 2.5~, wasm exactly 2.0.2 | Adapt the upstream `rocq-master` port to released dependencies, preserving the selected Wasm semantics. Compile semantic proofs and execute interpreter tests against wasm 2.0.2. |
| CertiRocq: verified Gallina-to-C and runtime glue | 0.9.1+9.1 requires Rocq below 9.2; upstream source also pins older CompCert, Equations and WasmCert | Port after Equations, MetaRocq, CompCert and WasmCert. Check compiler theories, native plugins, bootstrap, runtime compilation and repository C correspondence harnesses. |
| RustExtraction: typed erasure and Rust emission | 0.2.1 depends on typed-extraction-common 0.2.1, which requires core and Stdlib below 9.2 | Port typed extraction and its MetaRocq dependencies together. Compile generated Rust and compare it with reference fixtures. Record-to-enum output, borrowed lifetimes and absent Module-Type-to-trait translation in `extraction/RustExtractSpike.v` need separate extractor capability tests. |

Port order follows dependency direction: Equations, MetaRocq, CompCert and
WasmCert, then CertiRocq and typed Rust extraction. MetaRocq `main` follows
Rocq development APIs, including universe and module changes beyond the
stable release; a development branch requires revision selection and testing.
WasmCert's `rocq-master` commit
`882c621bd9316eff815f49a251c77e9547eb35c7` supplies another source candidate,
with development dependencies that require a coherent release overlay.

The existing `extract-check`, `extract-c-check`, `extract-r2vb-check`, and
`test-r2vb-correspondence` targets exercise distinct output surfaces. Run
those targets after their toolchain dependencies pass. Preserve the current
extraction identities and golden artifacts until replacement outputs have
been inspected and correspondence has been measured. A successful core proof
build establishes proof compilation, while each extraction backend needs its
own execution evidence.

The CertiRocq port also needs a repository extraction-runner correction:
`ExtractFP24Admit.v` emits `sint_range_admit`, while `extract-c` and
`extract-c-check` select `fp24_value_admit` filenames. Align the moved and
compiled filenames with the signed interval predicate, then regenerate and
test the C correspondence artifacts under the ported compiler. The function
identities express different admission policies and must remain distinct.

Published upstream sources also retain warning debt. Opam installation uses
each package's build policy; the repository's strict integration tests measure
a separate surface. The observed upstream diagnostics require source changes
before those complete packages can claim warning-free builds:

| Source | Observed diagnostic | Required source migration |
| --- | --- | --- |
| Flocq 4.2.2 | `missing-proof-command`, `notation-for-abbreviation` | Add explicit proof-entry commands and use abbreviation declarations; rerun the numerical correctness proofs under fatal warnings. |
| MathComp 2.6.0 | `at-level-200-changed`, `notation-for-abbreviation` | Adapt notation precedence and abbreviation declarations; rerun parsing, reflection and hierarchy consumers. |
| Retained MetaRocq erasure sources | `ambiguous-paths` between `weaken_env_prop` coercions | Establish one coherent conversion path and rerun the checker, erasure and quotation consumers. |

## Local verification

The 2026-10-16 run used Rocq 9.3.0 and OCaml 5.5.1 in a fresh worktree.
`_RocqProject` contained 310 theory and verified sources; its SHA-256 was
`37f986059454d63bb852ebef96b169ce6fa1646ac85dd12a209156d418b6bcf2`.
The retained ordered source-hash manifest had SHA-256
`cce9841b82b39bd11942da6035a770984c46df8045d35db71c74e90efe5c9fcb`.

| Surface | Command or result |
| --- | --- |
| Complete proof bodies | `make -C proofs --jobs=12 --keep-going all`: 310 `.vo` objects, exit 0, 363.96 seconds elapsed, zero warnings and errors in the build log |
| Complete kernel check | `rocq check -silent -o` over all 310 logical modules and their dependency closure: exit 0, 126.64 seconds elapsed; the context summary retains explicit axioms |
| Admissions | `check-admissions`: five scanner tests passed; all 310 corpus sources contained zero admission tokens; explicit axioms remain separate assumptions |
| OCaml extraction | `extract-check`: generated quaternion extraction matched the retained golden file |
| Compatible plugins | Equations installed through opam with upstream tests enabled; `check-plugins` passed with fatal default warnings |
| Trusted replay | All 24 theorem-execution tests passed, including genuine compiler/kernel execution, configured Flocq replay, historical receipt retention and rejection of wrong objects |
| Rust diagnostics | `cargo clippy --locked --offline -p provenance_store --all-targets --profile validation -- -D warnings` passed |

Retained replay validates matching historical Rocq 9.1.1 / OCaml 5.4.0 and
Rocq 9.2 / OCaml 5.5.0 identities. Fresh execution requires Rocq 9.3.0 /
OCaml 5.5.1. Historical validation retains its witnessed commands and bytes;
the historical allowlist supplies no alternate execution path.

The Make jobserver admitted all 12 process-visible CPUs. The dependency graph
limited ready work during the final Brown chapter chain. Local compilation
and tests establish the listed surfaces; hosted repository validation remains
the authority for integrating the branch.

## Primary sources

- [Rocq releases](https://github.com/rocq-prover/rocq/releases)
- [Rocq Stdlib releases](https://github.com/rocq-prover/stdlib/releases)
- [opam releases](https://github.com/ocaml/opam/releases)
- [OCaml releases](https://github.com/ocaml/ocaml/releases)
- [Rocq opam repository](https://github.com/rocq-prover/opam)
- [Flocq](https://flocq.gitlabpages.inria.fr/)
- [Equations 9.3 source](https://github.com/rocq-prover/equations/tree/d56dfc2c38feb5054d75ef43540ee89a6beb1a7e)
- [Elpi plugin](https://github.com/LPCIC/coq-elpi)
- [Hierarchy Builder](https://github.com/math-comp/hierarchy-builder)
- [MathComp](https://github.com/math-comp/math-comp)
- [ExtLib](https://github.com/coq-community/coq-ext-lib)
- [MetaRocq](https://github.com/MetaRocq/metarocq)
- [CompCert](https://github.com/AbsInt/CompCert)
- [WasmCert development port](https://github.com/WasmCert/WasmCert-Coq/tree/882c621bd9316eff815f49a251c77e9547eb35c7)
- [CertiRocq](https://github.com/CertiRocq/certirocq)
- [Rust extraction](https://github.com/AU-COBRA/coq-rust-extraction)

## Vacuum response composition

The registered vacuum modules separate finite-dimensional mathematics from
physical attribution:

- `FiniteComplexPassivity` uses `CDComplex` amplitudes, supplied diagonal
  spectra, norm-preserving maps, product energies and finite truncation tails.
- `FiniteCompositePassivity` permits arbitrary joint amplitudes, proves the
  additive spectral bound and cyclic ground-state passivity, and checks a
  normalized state that cannot factor into subsystem states.
- `PassiveMechanicalMemory` proves exact finite-matrix quadratic increments,
  storage nonnegativity, coupled auxiliary-memory balance and supply bounds.
  A smooth-trajectory/time-derivative bridge remains a separate obligation.
- `GaussianCoherence` proves PSD sampled variance, coherence range and ideal
  quasistatic echo cancellation under the declared classical Gaussian model,
  and complex dephasing Choi positivity for the explicit I/Z mixture.
- `VacuumCounterexamples` proves the PSD interval of the supplied complex
  depolarizing Choi quadratic form, separately from memory admissibility.
- `VacuumSpectralModels` proves ordered oscillator-line consequences,
  harmonic softening variance and cutoff scaling with explicit parameters.
- `WorldlineInteraction` proves scalar subtraction and two-body selection.
- `VacuumAnalyticLimits` constructs uniformly bounded weighted finite sums
  with arbitrarily large unweighted energy; infinite-series limits remain
  outside that theorem.

The same `all` command compiles the modules through the process-visible CPU
jobserver. The composition run checked all318 registered source bodies,
all318 kernel modules and83 new named declaration contexts. Retained source
and object digests, tool versions and assumption reports live under
`data/output/audit/vacuum-composition/`. The local source/object archive is
retained separately from Git; its identity appears in `replay-results.toml`.
Real-arithmetic proofs and numerical tests measure separate surfaces.
`plans/vacuum_integration_contract.toml` maps all35 original leaves to their
actual evidence and remaining gate. The `vacuum-response-audit` binary unit
tests verify coverage and reject missing, duplicated, cyclic, stale or
unjustifiably closed rows.
