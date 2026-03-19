//! <!-- AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT. -->
//! <!-- Source of truth: registry/external_sources.toml -->
//! <!-- Canonical write path: registry/canonical/control_plane.sqlite3 -->
//! <!-- Source label: XS-029 -->
//! <!-- Regenerate with: cargo run -p gororoba_cli_data --bin provenance -- export-external-sources -->
//!
//! # x87 / AVX Accumulation Sources
//!
//! ## Scope
//!
//! This dossier hardens the x87 / AVX accumulation claim cluster:
//!
//! - `C-1359`: x87 `FDIVR` / `FUCOMPP` semantics inside the Givens rotation path
//! - `C-1360`: AVX packed-double vs `fma` feature split and single-rounding wording
//! - `C-1361`: measured x87 norm-squared throughput win
//! - `C-1362`: Ogita-Rump-Oishi crossover heuristic and dispatch guidance
//!
//! ## Primary instruction-set references
//!
//! - AMD64 Architecture Programmer's Manual, Volume 5:
//!   instruction semantics for `FDIVR` and `FUCOMPP`
//! - AMD64 Architecture Programmer's Manual, Volume 4:
//!   SIMD / AVX / F16C / FMA-family opcode tables adjacent to the packed-double path
//! - Intel intrinsic reference for `_mm256_fmadd_pd`:
//!   explicit statement that the fused multiply-add forms the product and sum with
//!   an infinite-precision intermediate and rounds once to `float64`
//! - Rust intrinsic docs for `_mm256_extractf128_pd` and `_mm256_hadd_pd`:
//!   confirms the AVX packed-double reduction path used by `hsum256`
//!
//! ## Numerical-analysis reference
//!
//! - Ogita, Rump, Oishi (2005), "Accurate Sum and Dot Product":
//!   provides the compensated-summation error bounds used for the `N = 2048`
//!   crossover arithmetic (`2^-53 / 2^-64`)
//!
//! ## Provenance note
//!
//! The user referenced local files `26568.pdf` / `26569.pdf` during review, but
//! those artifacts are not present inside this repository. The governed source lane
//! for this claim cluster is therefore the official vendor / doc URLs above plus
//! the Ogita-Rump-Oishi paper and the repo-local verifier / benchmark artifacts.
//!
