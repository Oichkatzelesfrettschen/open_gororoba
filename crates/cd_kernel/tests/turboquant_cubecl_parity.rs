//! Bit-identical parity test: `Backend::Cpu` vs `Backend::CubeCL`.
//!
//! # Why this test exists
//!
//! `BackendQuantizer::quantize` exposes three paths to the same
//! mathematical operation: scalar CPU, hand-rolled Vulkan SPIR-V, and
//! cubecl-wgpu through the `#[cube]` kernel. The CPU path is the
//! ground-truth oracle; both GPU paths must produce bit-identical u8
//! output for the same input.
//!
//! The Vulkan path's parity is verified by `turboquant_vulkan_parity.rs`.
//! This file is its cubecl counterpart and the third leg of the
//! parity invariant: any divergence between CPU, Vulkan, and cubecl
//! outputs is a bug in whichever backend disagrees with the other two.
//!
//! # Test isolation
//!
//! - Gated `#[ignore = "gpu"]` -- requires a wgpu-reachable adapter
//!   (Vulkan / Metal / DX12 / WebGPU). CI without GPU support skips
//!   this test by default; running locally requires
//!   `cargo test ... -- --ignored`.
//! - Probes `launcher::is_available()` at runtime and skips cleanly
//!   (without panicking) if no adapter is reachable, so the test can
//!   pass on a misconfigured host without lying about the parity
//!   invariant.
//!
//! # Test inputs
//!
//! - `bits = 3` (8-entry codebook, 7 boundaries)
//! - `head_dim = 128`
//! - `n_inputs = 1024` from a fixed-seed `ChaCha20Rng` -- chosen so
//!   the input has both negative and positive tails relative to the
//!   Lloyd-Max codebook centroids, exercising every bin.
//!
//! # Run command
//!
//! ```sh
//! cargo test -p cd_kernel --features cubecl --release \
//!   turboquant_cubecl_parity -- --ignored --nocapture
//! ```
//!
//! Expected output: "all 1024 indices match" + the test passes.

#![cfg(feature = "cubecl")]

use cd_kernel::lloyd_max::get_codebook;
use cd_kernel::turboquant::backend::{Backend, BackendQuantizer};
use cd_kernel::turboquant::cubecl_backend::launcher::is_available as cubecl_is_available;
use cd_kernel::turboquant::dispatch::SimdLevel;

#[test]
#[ignore = "gpu (cubecl-wgpu requires a Vulkan/Metal/DX12/WebGPU adapter)"]
fn cubecl_parity_3bit_1024() {
    // # Skip-cleanly path
    //
    // Hosts without a wgpu-reachable adapter return false here; we
    // print a notice and exit Ok rather than panicking. This matches
    // the Vulkan parity test's `is_available()` probe pattern.
    if !cubecl_is_available() {
        eprintln!(
            "skip: cubecl_backend::launcher::is_available() is false \
             (no wgpu-reachable adapter)"
        );
        return;
    }

    // # Codebook setup
    //
    // d=128 + bits=3 is a typical TurboQuant configuration: 8-entry
    // codebook indexed by 7 boundary thresholds. The Lloyd-Max
    // codebook is deterministic given (d, bits), so both backends see
    // exactly the same boundaries.
    let bits = 3;
    let codebook = get_codebook(128, bits);

    // # Synthetic input
    //
    // 1024 floats from a fixed seed for reproducibility. StandardNormal
    // produces a Gaussian centered at 0 with stdev 1; the Lloyd-Max
    // boundaries for bits=3 partition this distribution into 8 roughly
    // equiprobable bins, so we expect every bin to be hit.
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let normal = StandardNormal;
    let values: Vec<f32> = (0..1024)
        .map(|_| <StandardNormal as Distribution<f32>>::sample(&normal, &mut rng))
        .collect();

    // # Reference (CPU scalar)
    //
    // The CPU quantizer is the ground-truth oracle. Its boundary
    // search is implemented in `cd_kernel::lloyd_max` and is what
    // every other backend's kernel mirrors structurally.
    let cpu_bq = BackendQuantizer::with_backend(&codebook, bits, Backend::Cpu(SimdLevel::Scalar));
    let mut cpu_indices = vec![0u8; values.len()];
    cpu_bq.quantize(&values, &mut cpu_indices);

    // # cubecl-wgpu path
    //
    // BackendQuantizer::with_backend(...Backend::CubeCL) routes the
    // quantize call through cubecl_backend::launcher::quantize, which
    // dispatches to whatever wgpu adapter the host has. The result
    // must be bit-identical to the CPU reference.
    let cubecl_bq = BackendQuantizer::with_backend(&codebook, bits, Backend::CubeCL);
    let mut cubecl_indices = vec![0u8; values.len()];
    cubecl_bq.quantize(&values, &mut cubecl_indices);

    // # Bit-identical assertion with diagnostic
    //
    // If the cubecl output diverges from the CPU reference, dump the
    // first 10 mismatch positions so the failure mode is obvious. A
    // single-position divergence usually means the boundary search
    // semantics differ (`>` vs `>=`); a many-position divergence
    // usually means the codebook upload path is wrong.
    if cpu_indices != cubecl_indices {
        let mismatches: Vec<(usize, u8, u8)> = cpu_indices
            .iter()
            .zip(cubecl_indices.iter())
            .enumerate()
            .filter(|(_, (c, v))| c != v)
            .map(|(i, (c, v))| (i, *c, *v))
            .take(10)
            .collect();
        let total_mismatches = cpu_indices
            .iter()
            .zip(cubecl_indices.iter())
            .filter(|(c, v)| c != v)
            .count();
        panic!(
            "CPU vs cubecl indices diverge at {} positions; first 10: {:?}",
            total_mismatches, mismatches
        );
    }
    eprintln!("all {} indices match between Backend::Cpu and Backend::CubeCL", values.len());
}
