//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->
//!
//! # Requirements: Algebra Engine (Cayley-Dickson, de Marrais, Reggiani)
//!
//! These components primarily live in Rust crates under `crates/gororoba_algebra/`,
//! `crates/cd_kernel/`, and related sibling algebra crates, and are
//! exercised by Rust unit and integration tests.
//!
//! Install:
//! ```ignore
//! make install
//! make rust-smoke
//! make rust-regression
//! ```ignore
//!
//! Notes:
//! - The core algebra replication/validation code depends only on the base install extras (NumPy/SciPy/SymPy/Numba).
//! - Keep `PYTHONWARNINGS=error` enabled in scripts/CI; treat warnings as errors.
//! - The workspace manifest currently requires nightly Cargo because the root `Cargo.toml`
//! uses the unstable `codegen-backend` feature. In practice, the algebra lane should be
//! treated as `cargo +nightly ...` unless a crate-local manifest is used.
//! - Native BLAS backends for the Burn-based `neural_homotopy` path are explicit Cargo feature opt-ins, not defaults.
//! - `cargo check -p neural_homotopy --features openblas-system`
//! - The opt-in boundary is the crate's `[features]` section in `crates/neural_homotopy/Cargo.toml`, enabled with `cargo ... --features <name>`. A detected system library does nothing by itself until a build command opts into the matching feature.
//! - Use `make doctor-blas` (or `sh scripts/detect_native_blas.sh`) to see which native BLAS/LAPACK candidates are present on the host and how they map to the repo's exposed feature surface.
//! - `openblas-system` expects a system OpenBLAS install and is the only repo-exposed native BLAS feature today.
//! - `BLIS` is intentionally not exposed as a repo feature today: `blas-src` supports it, but Burn's ndarray backend does not forward a BLIS selector for this crate.
//! - Burn can support other native BLAS equivalents upstream, but this repo intentionally keeps them off the exposed feature surface until they can coexist with the default `nextest` smoke lane without implicit dependency downloads. A fresh `openblas` source-backed build currently attempts a network fetch through `openblas-src`, so the repo exposes only `openblas-system`. `accelerate` and `blas-netlib` therefore stay upstream-only, while `blis`, `intel-mkl`, and `r` remain `blas-src`-level capabilities that are not forwarded by Burn for this crate.
//! - Artifact generation entrypoints:
//! - `make artifacts-boxkites` (de Marrais exports)
//! - `make artifacts-reggiani` (Reggiani exports)
//! - `make artifacts-motifs` (motif census baseline: 16D/32D + plot)
//! - `make artifacts-motifs-big` (motif census extended: 64D/128D exact + 256D sampled + plot)
//!
//! ## x87 / FP80 lane inside `cd_kernel`
//!
//! The current x87 work lives inside `crates/cd_kernel/` and `crates/algebra_analysis/`.
//! Today this is a real inline-asm backend for reductions and a Jacobi-specific analysis
//! path, but it is not yet a complete standalone "x87 math crate". The immediate safe
//! surface now includes:
//!
//! - `cd_kernel::x87_primitives` for reduction-style kernels such as `x87_sum`,
//! `x87_dot`, `x87_norm_sq`, and `x87_horner`
//! - `cd_kernel::x87_ext80` for exact 80-bit payload storage (`Ext80`) plus x87
//! control-word and status-word wrappers
//! - `cd_kernel::x87_transcendentals` for shared `atan2_ext80`, `sincos_ext80`,
//! `fprem1_ext80`, and reduced trig kernels that surface x87 status bits
//! - `cd_kernel::x87_jacobi_kernels` for reusable Givens/Jacobi micro-kernels
//! (`givens_sincos_f64`, `diagonal_update_f64`, `atan2_sincos_f64`) extracted
//! out of `algebra_analysis`
//! - `gororoba_cli_algebra --bin x87-strategy-bench` for the pinned-core x87/AVX2
//! strategy sweep
//! - `gororoba_cli_algebra --bin jacobi-backend-sweep` for solver-shaped backend
//! sweeps over the reference f64, double-double, and x87 Jacobi lanes
//!
//! Host assumptions for this lane:
//!
//! - `target_arch = "x86_64"`
//! - x87 hardware present (baseline on x86_64)
//! - AVX2/FMA3 only for the hybrid benchmark/dispatch lane, not for the minimal x87 core
//! - One heavy numeric worker per physical core when benchmarking the multicore lane
//!
//! Recommended verification commands:
//!
//! ```ignore
//! cargo +nightly test -p cd_kernel --lib
//! cargo +nightly test -p algebra_analysis --lib x87_jacobi
//! cargo +nightly test -p algebra_analysis --lib reference_jacobi
//! cargo +nightly test -p gororoba_cli_algebra --bin x87_strategy_bench
//! cargo +nightly check -p gororoba_cli_algebra --bin jacobi-backend-sweep
//! cargo +nightly clippy -p cd_kernel -p algebra_analysis -p gororoba_cli_algebra --all-targets -- -D warnings
//! make x87-strategy-bench LEN=65536 REPEATS=7 OUT=reports/benchmarks/x87_strategy_bench_smoke.csv SUMMARY=reports/benchmarks/x87_strategy_bench_smoke.md
//! make jacobi-backend-sweep SIZES=4,8,16 REPEATS=3 OUT=reports/benchmarks/jacobi_backend_sweep_smoke.csv SUMMARY=reports/benchmarks/jacobi_backend_sweep_smoke.md
//! ```ignore
//!
//! Current scope boundary:
//!
//! - Keep numerically sensitive reductions or micro-kernels inside x87 long enough for
//! 80-bit residency to matter.
//! - Prefer status-returning shared kernels at the `Ext80` layer when C2 or x87
//! exception flags affect correctness, such as `FSINCOS` range handling or
//! `FPREM1`-based reduction loops.
//! - Do not assume x87 and AVX2/FMA overlap for free on one Zen 3 core; the hybrid lane
//! should be treated as phase separation or per-core partitioning, not instruction-level
//! interleaving.
//! - If exact ext80 semantics must survive across kernel boundaries, store as 80-bit
//! payloads rather than spilling back to `f64`.
//! - The new transcendental/Jacobi modules explicitly balance the x87 stack and expose
//! status words, but the older x87 reduction surface still needs a broader Rust-inline-asm
//! compliance sweep before the repo can claim that every x87 asm block follows the
//! strictest current reference guidance.
//!
//! ## Broader precision ladder after the x87 split
//!
//! The x87 lane is only one part of the repository's current algebra precision stack:
//!
//! - `cd_kernel::avx2_primitives` is the throughput-oriented binary64 SIMD lane.
//! - `cd_kernel::x87_primitives`, `cd_kernel::x87_ext80`, `cd_kernel::x87_transcendentals`,
//! and `cd_kernel::x87_jacobi_kernels` form the hardware FP80 lane.
//! - `algebra_analysis::double_double` and `algebra_analysis::dd_jacobi` form the
//! software high-precision fallback lane.
//! - `algebra_analysis::precision_policy` is the deterministic selection layer for
//! current Jacobi backends. It makes the default obstruction-spectrum choice
//! explicit instead of leaving it as an architecture-only `cfg` branch. The first
//! offline threshold table currently keeps matrix orders `<= 64` on the
//! `reference_f64` lane and escalates larger orders to x87 or DD. That cutoff
//! is backed by the spectral-proxy plus entrywise obstruction-family boundary
//! artifacts, not by comments alone.
//! - `algebra_analysis::reference_jacobi` is the public f64 oracle lane so binaries,
//! tests, and future policy tooling can compare all three current backends through
//! the same shared solver frame.
//! - The precision layer now also distinguishes solver family from numeric backend.
//! `algebra_analysis::precision_policy` adds `SpectrumObjective`,
//! `MatrixWorkloadClass`, `MatrixStructureHints`, and
//! `SpectrumDispatchInput`/`SpectrumDispatchDecision` so production code can
//! distinguish full-spectrum dense work from few-extremal work and
//! obstruction-structured work instead of only choosing x87 vs DD vs f64.
//! The first production-safe extension is exact isolated-zero-mode deflation for
//! obstruction matrices plus a new partial-spectrum API for largest- and
//! smallest-magnitude eigenvalues.
//! - `algebra_analysis::spectrum_solvers`, `partial_spectrum`, and `block_jacobi`
//! now provide the first solver-family expansion above the old backend-only lane:
//! exact structured deflation, deterministic partial subspace iteration for
//! `LargestAbs { k }` / `SmallestAbs { k }`, and a block-Jacobi prototype with
//! block sizes 2 and 4. The focused follow-up sweeps currently support keeping
//! partial spectrum promoted for few-extremal workloads, keeping block Jacobi
//! benchmark-only on the tested obstruction families, and treating exploratory
//! histogram partitions as structure signals rather than exact quotient
//! reductions. The first histogram-projected structure-aware prototype is now in
//! the benchmark lane as well; it is much faster on the current obstruction
//! families, but its spectrum error is far too large for promotion. A follow-up
//! two-level histogram-lifted prototype materially improves that approximation
//! on `quantized_obstruction_graph`, but it still does not generalize to the
//! real obstruction family. The current structure-aware benchmark now also
//! records a centered cross-cell coupling ratio, and that metric is the clearest
//! explanation so far: it is effectively zero for `quantized_obstruction_graph`,
//! but rises into the 4-7% range for `quantized_shell_permutation` and the real
//! obstruction family.
//! - `algebra_analysis::tests::precision_tier_dispatch` and
//! `gororoba_cli_algebra --bin x87-strategy-bench` plus
//! `gororoba_cli_algebra --bin jacobi-backend-sweep` currently provide evidence and
//! heuristics. The typed Jacobi policy now includes a first offline threshold
//! table seeded from solver-shaped sweep artifacts, but it is still intentionally
//! simple and should be refined only through new reproducible measurements. The
//! sweep harness now includes both spectral proxies and one entrywise
//! zero-diagonal quantized obstruction family plus a second shell/permutation
//! entrywise family, and the sweep target now supports `FAMILIES=...` so focused
//! profiling does not require editing the binary.
//!
//! Immediate roadmap after the x87 extraction:
//!
//! - The shared flat-array Jacobi scaffold now lives in `algebra_analysis::jacobi_shared`,
//! so the next step is to keep additional precision backends plugging into that same
//! algorithm frame instead of re-copying solver orchestration.
//! - Keep benchmark claims and dispatch claims separate even now that the first
//! offline threshold table exists; each threshold change should still be backed
//! by a named sweep artifact or registry input.
//! - Use the native build system targets `make x87-strategy-perf`,
//! `make x87-strategy-hyperfine`, `make x87-strategy-flamegraph`, and
//! `make jacobi-backend-sweep`, `make block-jacobi-backend-sweep`,
//! `make partial-spectrum-bench`, `make structured-spectrum-bench`,
//! `make jacobi-backend-perf`, and
//! `make jacobi-backend-flamegraph` to gather cycle, throughput, call-stack, and
//! solver-shaped backend evidence around the current precision lane.
//! The default `x87-strategy-perf` event set is the conservative user-space tuple
//! `cycles:u,instructions:u,branches:u,branch-misses:u`, because broader
//! `perf stat -d` bundles can include unsupported events on this host.
//! The focused backend-isolated perf pass on `quantized_shell_permutation`
//! did not reproduce a true x87 speed win at orders `68` or `72`, so the current
//! evidence still supports keeping the global obstruction-spectrum cutoff at `64`
//! unless a future family-specific policy is introduced with stronger evidence.
//! The first focused Jacobi flamegraph pass now defaults to release debuginfo for
//! more readable stacks. It successfully exposes a visible `libm` `atan2` hotspot
//! on the `reference_f64` lane, but the current x87 and double-double SVGs remain
//! too monolithic to use as a full internal-hotspot explanation.
//! For deeper x87 attribution, the repo now also supports `make jacobi-backend-samply`.
//! The strongest current path is the `dev` profile plus `PRESYMBOLICATE=1`, which
//! resolves the internal stack through `x87_jacobi`, `jacobi_shared`,
//! `x87_jacobi_kernels`, and `Ext80` helpers. On this host that workflow requires a
//! temporary `sudo -A` relaxation of `/proc/sys/kernel/perf_event_paranoid`, and the
//! current validated command restores the previous value immediately afterward.
//! The same samply lane now also works for the DD and reference backends. It
//! resolves the main `dd_jacobi` solver entrypoints, although the smaller DD
//! helpers are still inlined in the current dev-profile artifact, and it also
//! resolves the reference path through `jacobi_shared`, `reference_jacobi`, and
//! the expected libc/libm `atan2` and `sincos` frames.
//! To reconcile those three raw captures at line level without manual JSON
//! inspection, the repo now also supports `make jacobi-backend-samply-compare`.
//! That Rust-native comparer joins the three presymbolicated captures with their
//! sidecars, prefers repo-scoped inline frames over generic `core` iterator
//! frames, and emits one backend-core/shared/dependency/runtime report. That step
//! fixes the earlier DD misattribution where the first inline frame obscured the
//! fact that the actual solver work was in `dd_jacobi.rs`.
//! For deeper DD attribution, the repo now also supports a profiling-only feature
//! lane: `make jacobi-backend-samply PROFILE=dev FEATURES=profile-dd-hotspots ...`.
//! That mode applies `inline(never)` to the DD solver and core DD arithmetic
//! helpers without changing the normal production policy surface, and it exposes
//! helper-level rows such as `two_sum`, `two_product`, DD `add`/`mul`/`div`/`sqrt`,
//! `dd_rotation`, and `dd_diagonal_update` in the resulting presymbolicated
//! samply artifact. After splitting the solver into explicit phase helpers, that
//! same lane now also shows `dd_find_pivot` and `dd_apply_plane_rotation`. After
//! the direct hi/lo pivot-comparator optimization, the profiled
//! `quantized_shell_permutation` `n=72` run drops from about 98.48 ms to about
//! 46.59 ms with the same reported `max_abs_err`, and the next DD runtime target
//! becomes the plane-rotation/update path rather than pivot scanning. A second
//! DD optimization now specializes that path with `dd_mul_add_pair`, which drops
//! the same profiled lane again to about 45.18 ms, reduces
//! `dd_apply_plane_rotation` from about 36.89 percent to about 32.42 percent,
//! and cuts `two_sum` from about 24.52 percent to about 14.90 percent. That
//! leaves `dd_rotation` plus its `sqrt`/`div` costs as the next DD target.
//! For x87 micro-kernel cost work, the repo now also supports
//! `make x87-givens-microbench` and `make x87-givens-microbench-perf`. The first
//! lane measures the actual composed
//! `atan2`/half-angle/`sincos` Givens path plus the diagonal-update micro-kernel
//! directly, and the first smoke artifact shows the composed transcendental path
//! landing around 102-126 ns/call while the diagonal update lands around
//! 6-8 ns/call. The focused perf-stat follow-up then narrows the obstruction-like
//! hot path to `x87_atan2_sincos`, `givens_sincos_ext80`, and
//! `x87_givens_diagonal_update`, and it says the current half-angle Ext80 path
//! is still within about 1.5 percent of the older full-angle composition while
//! the diagonal update remains about 17 times cheaper. That replaces the older
//! hand-wavy benchmark prose with a direct measurement of the composed Ext80
//! backend we actually ship and closes the immediate x87 decomposition question
//! for the current backend.
//! - The refreshed multicore sweep still supports the same system-level rule:
//! one heavy FP80 worker per physical core is the right starting point, but the
//! x87 lane does not scale monotonically through 6 workers on every workload.
//! Treat `x87_per_chunk` as a semantics-oriented lane for independent FP80 work,
//! not as a guarantee of ideal reduction scaling.
//! - The software DD lane now uses a stable tangent-based Jacobi rotation path and
//! DD diagonal updates, which removes the catastrophic obstruction-family errors
//! that the earlier transcendental-based rotation path showed. The repaired DD
//! lane is still slower than the reference lane on the current entrywise
//! obstruction-family boundary artifacts, so its present niche is accuracy-first
//! portability rather than throughput.
//! - Continue the inline-asm compliance sweep across older x87 modules so the full
//! precision ladder follows one documented backend contract.
//!
//! ## Legacy Python support removed
//!
