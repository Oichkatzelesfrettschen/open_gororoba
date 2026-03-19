//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/research_narratives.toml -->
//!
//! # Rust Engineering Standards & Style Guide
//!
//! ## 1. Toolchain & Environment
//!
//! - **Channel:** `nightly` (Required for `cudarc`, `nvrtc`, and some SIMD intrinsics).
//! - **Edition:** 2021 (Move to 2024 when stable).
//! - **Lockfile:** Checked in. `Cargo.lock` must be up to date.
//! - **Strictness:** CI runs with `RUSTFLAGS="-D warnings"`. No warnings allowed in main branch.
//!
//! ## 2. Code Formatting
//!
//! - **Tool:** `rustfmt` (default settings).
//! - **Trigger:** Run `cargo fmt` before every commit.
//! - **Imports:** Grouped by `std`, external crates, and `crate`/`super`/`self`.
//!
//! ## 3. Linting & Clippy
//!
//! - **Tool:** `clippy`.
//! - **Policy:** 
//! - `cargo clippy --all-targets --all-features -- -D warnings`
//! - Explicitly allow specific lints only with documented justification (e.g., `#[allow(clippy::too_many_arguments)]` on legacy physics kernels).
//! - **Custom Lints:** Use `cargo-deny` for license and dependency graph checking (`deny.toml` in root).
//!
//! ## 4. Architecture Patterns
//!
//! ### 4.1. Simulation State
//! - Use **Trait-based backends** for compute agnosticism (e.g., `LbmBackend3D` enum wrapping `Cpu`/`Gpu` impls).
//! - **Configuration:** Plain structs with `serde::Deserialize` for TOML config loading.
//! - **State Management:** `SimulationState` structs own all physics data. Avoid global statics (except strictly managed caches like `NoiseTextureCache`).
//!
//! ### 4.2. Error Handling
//! - **Libraries:** Use `thiserror` to define specific error enums. Public APIs must return `Result<T, crate::Error>`.
//! - **Applications/CLI:** Use `anyhow::Result` for top-level binary entry points.
//! - **Panics:** `unwrap()` and `expect()` are forbidden in library code unless mathematically impossible to fail (and documented with `// SAFETY:` comment). Allowed in tests and prototypes.
//!
//! ### 4.3. GPU/CUDA
//! - **Gate:** All CUDA code must be behind `#[cfg(feature = "gpu")]`.
//! - **Crates:** `lbm_3d_cuda` encapsulates unsafe FFI/NVRTC interactions.
//! - **Safety:** Wrap `unsafe` blocks with comments explaining why invariants are upheld (e.g., pointer validity, buffer sizes).
//!
//! ## 5. Testing Strategy
//!
//! - **Unit Tests:** Co-located in `src/` modules inside `#[cfg(test)] mod tests { ... }`.
//! - **Integration Tests:** In `tests/` directory, testing public API surface.
//! - **Property-based Testing:** Use `proptest` for algebraic invariants and numerical stability ranges.
//! - **Benchmarks:** `criterion` for performance-critical kernels (LBM steps, spectral transforms).
//!
//! ## 6. Documentation
//!
//! - **Public API:** `warn(missing_docs)` enabled. Every `pub` item needs a doc string.
//! - **Intra-doc Links:** Use `[`Item`]` syntax.
//! - **Mathematics:** Use LaTeX math blocks (via `katex` support in mdbook if configured) or ASCII approximations for simple formulas in code comments.
//!
//! ## 7. Version Control
//!
//! - **Commits:** Conventional Commits (e.g., `feat(lbm): add 3d fft`, `fix(gpu): resolve memory leak`).
//! - **Dependencies:** No git dependencies in `Cargo.toml` for production releases; path dependencies allowed for workspace crates.
//!
//! ## 8. Specific Crate Roles
//!
//! - `gororoba_engine`: Orchestration, config, high-level traits.
//! - `lbm_core`: Pure CPU physics kernels, no heavy deps.
//! - `lbm_3d_cuda`: GPU specific implementation, requires NVRTC.
//! - `gororoba_algebra`: Algebra facade crate re-exporting canonical Cayley-Dickson and adjacent math surfaces.
//! - `gororoba_cli`: Binary entry points (`warp-ring-3d`, experiments).
//!
