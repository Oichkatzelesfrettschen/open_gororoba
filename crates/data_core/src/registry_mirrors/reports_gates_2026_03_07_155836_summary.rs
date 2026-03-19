//! # Gate Audit (2026-03-07T15:58:36)
//!
//! Output directory: `reports/gates/2026-03-07/155836`
//!
//! | Step | Exit Code | Log |
//! | --- | ---: | --- |
//! | `gate-local` | `2` | `reports/gates/2026-03-07/155836/gate-local.log` |
//!
//! ## gate-local
//!
//! Exit code: `2`
//!
//! ```texttext
//! ... (7396 earlier line(s) omitted)
//!    |     ^^
//! 74 |     ax3.set_title("STRONG-COUPLING UNIFICATION: THE MELTDOWN FIELD", fontsize=28, fontweight='bold', pad=30)
//!    |
//! help: Remove assignment to unused variable `im`
//!
//! E501 Line too long (104 > 100)
//!   --> src/vis_trajectory.py:74:101
//!    |
//! 73 |     im = ax3.imshow(Z, cmap='magma', aspect='auto', origin='lower')
//! 74 |     ax3.set_title("STRONG-COUPLING UNIFICATION: THE MELTDOWN FIELD", fontsize=28, fontweight='bold', pad=30)
//!    |                                                                                                     ^^^^
//! 75 |
//! 76 |     # Annotate Isomorphism points
//!    |
//!
//! Found 651 errors.
//! [*] 221 fixable with the `--fix` option (36 hidden fixes can be enabled with the `--unsafe-fixes` option).
//! make[1]: Leaving directory '/home/eirikr/Github/open_gororoba'
//! make[1]: *** [Makefile:125: lint] Error 1
//! make: *** [Makefile:175: gate-local] Error 2
//! ```texttext
//! | `gate-ci-python` | `2` | `reports/gates/2026-03-07/155836/gate-ci-python.log` |
//!
//! ## gate-ci-python
//!
//! Exit code: `2`
//!
//! ```texttext
//! ... (7391 earlier line(s) omitted)
//!    |     ^^
//! 74 |     ax3.set_title("STRONG-COUPLING UNIFICATION: THE MELTDOWN FIELD", fontsize=28, fontweight='bold', pad=30)
//!    |
//! help: Remove assignment to unused variable `im`
//!
//! E501 Line too long (104 > 100)
//!   --> src/vis_trajectory.py:74:101
//!    |
//! 73 |     im = ax3.imshow(Z, cmap='magma', aspect='auto', origin='lower')
//! 74 |     ax3.set_title("STRONG-COUPLING UNIFICATION: THE MELTDOWN FIELD", fontsize=28, fontweight='bold', pad=30)
//!    |                                                                                                     ^^^^
//! 75 |
//! 76 |     # Annotate Isomorphism points
//!    |
//!
//! Found 651 errors.
//! [*] 221 fixable with the `--fix` option (36 hidden fixes can be enabled with the `--unsafe-fixes` option).
//! make[1]: Leaving directory '/home/eirikr/Github/open_gororoba'
//! make[1]: *** [Makefile:125: lint] Error 1
//! make: *** [Makefile:213: gate-ci-python] Error 2
//! ```texttext
//! | `gate-ci-rust` | `2` | `reports/gates/2026-03-07/155836/gate-ci-rust.log` |
//!
//! ## gate-ci-rust
//!
//! Exit code: `2`
//!
//! ```texttext
//! ... (334 earlier line(s) omitted)
//!       l_0=-1,l_1=-1 all : prefix=2, d=6, N=365, obs=0.2509, null=0.2154+/-0.0027, z=13.28
//!       l_2=-1              : prefix=3, d=5, N=122, obs=0.2763, null=0.1943+/-0.0046, z=17.65
//!       l_2=0               : prefix=3, d=5, N=121, obs=0.2858, null=0.1937+/-0.0044, z=20.82
//!       l_2=+1              : prefix=3, d=5, N=122, obs=0.2763, null=0.1949+/-0.0043, z=19.13
//!
//!     === l_2 Recursion Summary ===
//!     Combined (l_0=-1, l_1=-1): z=13.28
//!       l_2=-1: z=17.65
//!       l_2=0:  z=20.82
//!       l_2=+1: z=19.13
//!
//!     === l_3 level (within l_0=-1, l_1=-1, l_2=-1) ===
//!       l_3=-1: prefix=4, d=4, N=41, obs=0.3006, null=0.1782+/-0.0122, z=10.00
//!       l_3=0 : prefix=4, d=4, N=40, obs=0.3230, null=0.1807+/-0.0138, z=10.30
//!
//!     (test timed out)
//!
//! error: test run failed
//! make[1]: *** [Makefile:308: rust-regression] Error 100
//! make: *** [Makefile:225: gate-ci-rust] Error 2
//! ```texttext
//! | `nextest-list` | `0` | `reports/gates/2026-03-07/155836/nextest-list.log` |
//!
//! ## nextest-list
//!
//! Exit code: `0`
//!
//! ```texttext
//! ... (5128 earlier line(s) omitted)
//! verified_core spectral_dim::tests::range_bounds
//! verified_core::cross_validate cross_validate_90_degree_z
//! verified_core::cross_validate cross_validate_arbitrary_rotation
//! verified_core::cross_validate cross_validate_identity
//! verified_core::cross_validate cross_validate_multiple_axes
//! verified_core::cross_validate_refuted cross_validate_calcagni_spectral_dimension
//! verified_core::cross_validate_refuted cross_validate_democratic_mixing
//! verified_core::cross_validate_refuted cross_validate_gf2_separation
//! verified_core::cross_validate_refuted cross_validate_neg_dim_degeneracy
//! verified_core::cross_validate_refuted cross_validate_parity_clique
//! verified_core::cross_validate_sprint59 cross_validate_binary_entropy_properties
//! verified_core::cross_validate_sprint59 cross_validate_complex_mul
//! verified_core::cross_validate_sprint59 cross_validate_nordtvedt_bd
//! verified_core::cross_validate_sprint59 cross_validate_ppn_gamma_bd
//! verified_core::cross_validate_sprint59 cross_validate_quat_inverse
//! verified_core::cross_validate_sprint59 cross_validate_quat_mul
//! verified_core::cross_validate_sprint59 cross_validate_quat_norm_multiplicative
//! verified_core::cross_validate_sprint59 cross_validate_tcmt_antiresonance
//! verified_core::cross_validate_sprint59 cross_validate_tcmt_unitarity
//!     Finished `test` profile [optimized + debuginfo] target(s) in 0.52s
//! ```texttext
//!
//! Gate audit failed in 3 step(s).
//!
//! Review the per-step logs for full output.
//!
