//! # Jacobi Samply Comparison
//!
//! Inclusive sample-weighted hotspots joined from the samply profile JSON and the presymbolicated sidecar.
//!
//! ## reference_f64
//!
//! ### Backend-Core Hotspots
//!
//! | lib | function | file:line | category | inclusive_samples | percent |
//! | --- | --- | --- | --- | ---: | ---: |
//! | jacobi-backend-sweep | algebra_analysis::reference_jacobi::symmetric_eigenvalues_f64 | /home/eirikr/Github/open_gororoba/crates/algebra_analysis/src/reference_jacobi.rs:29 | backend_core | 447 | 98.89% |
//!
//! ### Shared/Support Hotspots
//!
//! | lib | function | file:line | category | inclusive_samples | percent |
//! | --- | --- | --- | --- | ---: | ---: |
//! | jacobi-backend-sweep | jacobi_backend_sweep::main | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:171 | repo_support | 451 | 99.78% |
//! | jacobi-backend-sweep | algebra_analysis::jacobi_shared::symmetric_eigenvalues_with_backend::<algebra_analysis::reference_jacobi::f64_rotation, algebra_analysis::reference_jacobi::f64_diagonal_update> | /home/eirikr/Github/open_gororoba/crates/algebra_analysis/src/jacobi_shared.rs:60 | repo_shared | 447 | 98.89% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::measure_backend | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:485 | repo_support | 447 | 98.89% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::run_backend | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:503 | repo_support | 447 | 98.89% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::run_backend_row | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:442 | repo_support | 447 | 98.89% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::build_case | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:209 | repo_support | 3 | 0.66% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::nalgebra_expected_spectrum | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:407 | repo_support | 3 | 0.66% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::write_csv | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:588 | repo_support | 1 | 0.22% |
//!
//! ### External Math/Runtime Hotspots
//!
//! | lib | function | file:line | category | inclusive_samples | percent |
//! | --- | --- | --- | --- | ---: | ---: |
//! | libm.so.6 | _sincos_fma | /usr/src/debug/glibc/glibc/math/../sysdeps/ieee754/dbl-64/s_sincos.c:41 | libm | 6 | 1.33% |
//! | libm.so.6 | _atan2 | /usr/src/debug/glibc/glibc/math/w_atan2_compat.c:38 | libm | 4 | 0.88% |
//! | jacobi-backend-sweep | <nalgebra::base::matrix::Matrix<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn, nalgebra::base::vec_storage::VecStorage<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn>>>::symmetric_eigenvalues | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/linalg/symmetric_eigen.rs:331 | dependency | 3 | 0.66% |
//! | jacobi-backend-sweep | <nalgebra::linalg::symmetric_eigen::SymmetricEigen<f64, nalgebra::base::dimension::Dyn>>::do_decompose | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/linalg/symmetric_eigen.rs:118 | dependency | 3 | 0.66% |
//! | libm.so.6 | _ieee754_atan2_fma | /usr/src/debug/glibc/glibc/math/../sysdeps/ieee754/dbl-64/e_atan2.c:178 | libm | 3 | 0.66% |
//! | jacobi-backend-sweep | <nalgebra::linalg::symmetric_tridiagonal::SymmetricTridiagonal<f64, nalgebra::base::dimension::Dyn>>::new | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/base/blas.rs:499 | dependency | 2 | 0.44% |
//! | jacobi-backend-sweep | <f64 as core::clone::Clone>::clone | /home/eirikr/.rustup/toolchains/nightly-2026-03-05-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/clone.rs:616 | other | 1 | 0.22% |
//! | jacobi-backend-sweep | <nalgebra::base::matrix::Matrix<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::matrix_view::ViewStorage<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Dyn>>>::shape | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/base/matrix.rs:423 | dependency | 1 | 0.22% |
//!
//! ## x87
//!
//! ### Backend-Core Hotspots
//!
//! | lib | function | file:line | category | inclusive_samples | percent |
//! | --- | --- | --- | --- | ---: | ---: |
//! | jacobi-backend-sweep | algebra_analysis::x87_jacobi::symmetric_eigenvalues_x87 | /home/eirikr/Github/open_gororoba/crates/algebra_analysis/src/x87_jacobi.rs:23 | backend_core | 515 | 99.04% |
//! | jacobi-backend-sweep | cd_kernel::x87_jacobi_kernels::givens_sincos_f64 | /home/eirikr/Github/open_gororoba/crates/cd_kernel/src/x87_jacobi_kernels.rs:42 | backend_core | 32 | 6.15% |
//! | jacobi-backend-sweep | cd_kernel::x87_jacobi_kernels::x87_givens_sincos | /home/eirikr/Github/open_gororoba/crates/cd_kernel/src/x87_jacobi_kernels.rs:58 | backend_core | 32 | 6.15% |
//! | jacobi-backend-sweep | cd_kernel::x87_jacobi_kernels::givens_sincos_ext80 | /home/eirikr/Github/open_gororoba/crates/cd_kernel/src/x87_jacobi_kernels.rs:26 | backend_core | 24 | 4.62% |
//! | jacobi-backend-sweep | cd_kernel::x87_transcendentals::sincos_ext80 | /home/eirikr/Github/open_gororoba/crates/cd_kernel/src/x87_transcendentals.rs:91 | backend_core | 8 | 1.54% |
//! | jacobi-backend-sweep | cd_kernel::x87_transcendentals::atan2_ext80 | /home/eirikr/Github/open_gororoba/crates/cd_kernel/src/x87_transcendentals.rs:63 | backend_core | 7 | 1.35% |
//! | jacobi-backend-sweep | RNvNtCslCVGl5UM8FG_9cd_kernel19x87_transcendentals11atan2_ext80__inline_asm_9k7d27ne6g3kobl0kyw62m7hh_n0 |  | backend_core | 6 | 1.15% |
//! | jacobi-backend-sweep | cd_kernel::x87_jacobi_kernels::x87_givens_diagonal_update | /home/eirikr/Github/open_gororoba/crates/cd_kernel/src/x87_jacobi_kernels.rs:70 | backend_core | 6 | 1.15% |
//!
//! ### Shared/Support Hotspots
//!
//! | lib | function | file:line | category | inclusive_samples | percent |
//! | --- | --- | --- | --- | ---: | ---: |
//! | jacobi-backend-sweep | jacobi_backend_sweep::main | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:171 | repo_support | 519 | 99.81% |
//! | jacobi-backend-sweep | algebra_analysis::jacobi_shared::symmetric_eigenvalues_with_backend::<algebra_analysis::x87_jacobi::x87_rotation_f64, cd_kernel::x87_jacobi_kernels::x87_givens_diagonal_update> | /home/eirikr/Github/open_gororoba/crates/algebra_analysis/src/jacobi_shared.rs:60 | repo_shared | 515 | 99.04% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::measure_backend | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:485 | repo_support | 515 | 99.04% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::run_backend | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:513 | repo_support | 515 | 99.04% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::run_backend_row | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:442 | repo_support | 515 | 99.04% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::build_case | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:209 | repo_support | 3 | 0.58% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::nalgebra_expected_spectrum | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:407 | repo_support | 3 | 0.58% |
//!
//! ### External Math/Runtime Hotspots
//!
//! | lib | function | file:line | category | inclusive_samples | percent |
//! | --- | --- | --- | --- | ---: | ---: |
//! | libc.so.6 | _memcpy_avx_unaligned_erms | /usr/src/debug/glibc/glibc/string/../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S:343 | libc | 13 | 2.50% |
//! | jacobi-backend-sweep | <nalgebra::base::matrix::Matrix<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn, nalgebra::base::vec_storage::VecStorage<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn>>>::symmetric_eigenvalues | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/linalg/symmetric_eigen.rs:331 | dependency | 3 | 0.58% |
//! | jacobi-backend-sweep | <nalgebra::linalg::symmetric_eigen::SymmetricEigen<f64, nalgebra::base::dimension::Dyn>>::do_decompose | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/linalg/symmetric_eigen.rs:118 | dependency | 3 | 0.58% |
//! | jacobi-backend-sweep | <nalgebra::base::matrix::Matrix<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn, nalgebra::base::matrix_view::ViewStorageMut<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Dyn>>>::xxgerx::<nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn, nalgebra::base::matrix_view::ViewStorageMut<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Dyn>, nalgebra::base::matrix_view::ViewStorageMut<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Dyn>, <f64 as simba::simd::simd_complex::SimdComplexField>::simd_conjugate> | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/base/matrix_view.rs:1021 | dependency | 2 | 0.38% |
//! | jacobi-backend-sweep | <nalgebra::linalg::symmetric_tridiagonal::SymmetricTridiagonal<f64, nalgebra::base::dimension::Dyn>>::new | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/base/blas.rs:1007 | dependency | 2 | 0.38% |
//! | jacobi-backend-sweep | <alloc::raw_vec::RawVec<f64> as core::ops::drop::Drop>::drop | /home/eirikr/.rustup/toolchains/nightly-2026-03-05-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/alloc/src/raw_vec/mod.rs:423 | other | 1 | 0.19% |
//! | jacobi-backend-sweep | <alloc::raw_vec::RawVecInner>::current_memory | /home/eirikr/.rustup/toolchains/nightly-2026-03-05-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/alloc/src/raw_vec/mod.rs:642 | other | 1 | 0.19% |
//! | jacobi-backend-sweep | <alloc::raw_vec::RawVecInner>::deallocate | /home/eirikr/.rustup/toolchains/nightly-2026-03-05-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/alloc/src/raw_vec/mod.rs:872 | other | 1 | 0.19% |
//!
//! ## double_double
//!
//! ### Backend-Core Hotspots
//!
//! | lib | function | file:line | category | inclusive_samples | percent |
//! | --- | --- | --- | --- | ---: | ---: |
//! | jacobi-backend-sweep | algebra_analysis::dd_jacobi::symmetric_eigenvalues_dd | /home/eirikr/Github/open_gororoba/crates/algebra_analysis/src/dd_jacobi.rs:28 | backend_core | 673 | 99.41% |
//!
//! ### Shared/Support Hotspots
//!
//! | lib | function | file:line | category | inclusive_samples | percent |
//! | --- | --- | --- | --- | ---: | ---: |
//! | jacobi-backend-sweep | jacobi_backend_sweep::main | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:171 | repo_support | 676 | 99.85% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::measure_backend | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:485 | repo_support | 673 | 99.41% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::run_backend | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:506 | repo_support | 673 | 99.41% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::run_backend_row | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:442 | repo_support | 673 | 99.41% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::build_case | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:209 | repo_support | 3 | 0.44% |
//! | jacobi-backend-sweep | jacobi_backend_sweep::nalgebra_expected_spectrum | /home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs:407 | repo_support | 3 | 0.44% |
//!
//! ### External Math/Runtime Hotspots
//!
//! | lib | function | file:line | category | inclusive_samples | percent |
//! | --- | --- | --- | --- | ---: | ---: |
//! | jacobi-backend-sweep | <nalgebra::base::matrix::Matrix<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn, nalgebra::base::vec_storage::VecStorage<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn>>>::symmetric_eigenvalues | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/linalg/symmetric_eigen.rs:331 | dependency | 3 | 0.44% |
//! | jacobi-backend-sweep | <nalgebra::base::matrix::Matrix<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn, nalgebra::base::matrix_view::ViewStorageMut<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Dyn>>>::xxgerx::<nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Dyn, nalgebra::base::matrix_view::ViewStorageMut<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Dyn>, nalgebra::base::matrix_view::ViewStorageMut<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Dyn>, <f64 as simba::simd::simd_complex::SimdComplexField>::simd_conjugate> | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/base/matrix_view.rs:1021 | dependency | 2 | 0.30% |
//! | jacobi-backend-sweep | <nalgebra::linalg::symmetric_eigen::SymmetricEigen<f64, nalgebra::base::dimension::Dyn>>::do_decompose | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/linalg/symmetric_eigen.rs:118 | dependency | 2 | 0.30% |
//! | jacobi-backend-sweep | <nalgebra::linalg::symmetric_tridiagonal::SymmetricTridiagonal<f64, nalgebra::base::dimension::Dyn>>::new | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/base/blas.rs:1007 | dependency | 2 | 0.30% |
//! | jacobi-backend-sweep | <f64 as core::ops::arith::Add>::add | /home/eirikr/.rustup/toolchains/nightly-2026-03-05-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/arith.rs:105 | other | 1 | 0.15% |
//! | jacobi-backend-sweep | <f64 as core::ops::arith::Mul>::mul | /home/eirikr/.rustup/toolchains/nightly-2026-03-05-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/arith.rs:352 | other | 1 | 0.15% |
//! | jacobi-backend-sweep | <nalgebra::base::matrix::Matrix<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::matrix_view::ViewStorageMut<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Dyn>>>::axpy::<nalgebra::base::dimension::Dyn, nalgebra::base::matrix_view::ViewStorage<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Dyn>> | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/base/blas.rs:300 | dependency | 1 | 0.15% |
//! | jacobi-backend-sweep | <nalgebra::base::matrix::Matrix<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::matrix_view::ViewStorageMut<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Const<1>, nalgebra::base::dimension::Dyn>>>::generic_view::<nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>> | /home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/base/matrix_view.rs:617 | dependency | 1 | 0.15% |
//!
//! ## Synthesis
//!
//! - The comparer now prefers repo-scoped inline frames over generic `core` iterator frames, which fixes the earlier DD misattribution.
//! - `reference_f64` should show backend-core lines in `reference_jacobi.rs`, shared/support rows in the sweep/scaffold, dependency rows for `nalgebra`, and external libm trig work.
//! - `x87` should surface the x87 Jacobi path plus shared `cd_kernel` Givens/ext80 helpers, with unresolved inline-asm pseudo-symbols remaining in the external bucket.
//! - `double_double` should now show `dd_jacobi.rs` as backend-core even when smaller DD helpers remain inlined.
//!
