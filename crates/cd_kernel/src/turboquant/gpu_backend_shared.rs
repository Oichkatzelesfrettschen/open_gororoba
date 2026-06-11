//! Shared GPU-backend utilities for the TurboQuant quantize lane.
//!
//! # Purpose and call sites
//!
//! Both `crate::turboquant::vulkan::quantizer::VulkanQuantizer`
//! (hand-rolled via `ash`) and the cubecl backend (cross-platform via
//! `cubecl-wgpu`) share a four-step contract:
//!
//! ```text
//! 1. Take a &`[f32]` of input values + &`[f32]` of boundaries.
//! 2. Compute a per-input "boundary count" (how many boundaries the
//!    value is strictly greater than).
//! 3. Read back u32 counts on the host.
//! 4. Pack the u32 counts into u8 indices that fit `bits` bit-widths.
//! 5. Return the packed &mut `[u8]` to the caller.
//! ```
//!
//! Steps (1), (2), and (4) have nothing to do with Vulkan or cubecl
//! specifically: validation, the u32->u8 narrowing, and the strict
//! CPU-fallback path are all backend-agnostic. This module owns those
//! pieces so that each backend launcher only owns its dispatch surface
//! (buffer upload + kernel launch + readback). A third backend (e.g.
//! a future cudarc-based path) plugs in by implementing dispatch alone
//! and reusing the helpers here.
//!
//! # Why this module exists
//!
//! Before cubecl landed, the Vulkan launcher had its own pack/validate
//! logic inline. Bringing cubecl in would have duplicated that logic in
//! a second module. Centralizing the math here keeps the Vulkan
//! launcher focused on `vk::*` calls and the cubecl launcher focused on
//! the `ComputeClient`/`Handle` API; both delegate to the same scalar
//! reference for parity testing.
//!
//! # Design contract for backend implementers
//!
//! A new backend conforms to this lane by:
//!
//! 1. Accepting a `QuantizeRequest` (NOT a tuple of disconnected
//!    slices; see `# Why a struct?` below).
//! 2. Returning the per-input u32 boundary count via whatever native
//!    mechanism it has (atomic counter, scalar accumulator, etc.) into
//!    a host-visible `Vec<u32>` of length `req.values.len()`.
//! 3. Calling `pack_u32_counts_to_u8` on that vector to write into
//!    the caller's `&mut [u8]`.
//! 4. Calling `validate_quantize` before any GPU dispatch to reject
//!    malformed inputs without paying GPU init cost.
//!
//! # Why a struct (`QuantizeRequest`) instead of three slices?
//!
//! The three input fields (`values`, `boundaries`, `bits`) form a
//! single logical unit: changing any one without re-validating the
//! others is meaningless. Bundling them into a struct lets us add a
//! fourth or fifth parameter (e.g. a precomputed CDF for adaptive bin
//! selection) without rippling through every backend signature, and
//! lets validation be a method on the bundle.
//!
//! # Concrete numeric example
//!
//! For `bits=2`, `boundaries=[-1.0, 0.0, 1.0]`, `values=[-2.0, 0.0, 0.5, 5.0]`:
//!
//! ```text
//!  v       count of b in boundaries with v > b
//!  -2.0    0    (less than all boundaries)
//!  0.0     1    (greater than -1.0 only)
//!  0.5     2    (greater than -1.0 and 0.0)
//!  5.0     3    (greater than all boundaries)
//! ```
//!
//! Output u8 indices: `[0, 1, 2, 3]`. With `bits=2` the codebook has
//! 2^2 = 4 entries (0..3), so each index fits in 2 bits and the count
//! is always at most 2^bits - 1 = 3. The cast `u32 -> u8` is therefore
//! lossless.
//!
//! # Cross-references
//!
//! - Reference CPU implementation: `quantize_cpu_reference`.
//! - Vulkan dispatch surface: see `vulkan/quantizer.rs::VulkanQuantizer::quantize`.
//! - cubecl dispatch surface: see `cubecl_backend/launcher.rs`.
//! - Architectural overview: `docs/engineering/registry_canonical_architecture.md`
//!   (does not cover GPU quantize; see crate root rustdoc instead).

#![cfg(any(feature = "vulkan", feature = "cubecl"))]

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A boundary-search quantization request as the GPU sees it.
///
/// # Field invariants
///
/// - `boundaries.len() == 2^bits - 1`. The codebook has `2^bits`
///   entries; `2^bits - 1` boundaries partition the real line into
///   `2^bits` bins. Violating this invariant returns an error from
///   `validate_quantize` without dispatching to the GPU.
/// - `bits` is in `1..=8`. Higher widths than u8 are not supported by
///   the index-pack step; `bits=0` would be a degenerate codebook.
/// - `boundaries` MUST be sorted ascending. The shaders assume this and
///   compute `count = sum_{b in boundaries} (v > b ? 1 : 0)`; an
///   unsorted boundary list would silently miscount.
///
/// # Why borrow rather than own?
///
/// Both backend launchers want to upload `values` and `boundaries`
/// to GPU memory exactly once. Owning them in the struct would force
/// callers to clone, and validation does not need ownership.
pub struct QuantizeRequest<'a> {
    /// Input values to quantize. Length determines output length.
    pub values: &'a [f32],
    /// Sorted ascending boundaries; length = `2^bits - 1`.
    pub boundaries: &'a [f32],
    /// Bit-width of each output index. Restricted to `1..=8`.
    pub bits: u32,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate a `QuantizeRequest` against an output buffer.
///
/// Returns the expected u32-index count (== `values.len()`) on success
/// so callers can pass it to `client.empty(n * 4)` without recomputing.
///
/// # Errors
///
/// - `"bits must be in 1..=8"`: `bits` is 0 or > 8.
/// - `"boundaries length must equal 2^bits - 1"`: the codebook
///   invariant is violated.
/// - `"out_indices length must equal values length"`: the output slice
///   is sized differently from the input.
///
/// # Why string errors?
///
/// This helper is meant to be the cheap first line of defense before
/// GPU init. A typed error enum here would propagate into every
/// backend's error type and not add information; the messages are read
/// by humans during debugging, not matched on programmatically.
///
/// # Worked example
///
/// For `bits=3`, the boundary slice must have length `2^3 - 1 = 7`.
/// Passing `boundaries=[0.0, 1.0]` returns the second error variant.
pub fn validate_quantize(
    req: &QuantizeRequest<'_>,
    out_indices: &[u8],
) -> Result<usize, &'static str> {
    if req.bits == 0 || req.bits > 8 {
        return Err("bits must be in 1..=8");
    }
    let expected_boundaries = (1usize << req.bits) - 1;
    if req.boundaries.len() != expected_boundaries {
        return Err("boundaries length must equal 2^bits - 1");
    }
    if out_indices.len() != req.values.len() {
        return Err("out_indices length must equal values length");
    }
    Ok(req.values.len())
}

// ---------------------------------------------------------------------------
// Host-side u32 -> u8 narrowing
// ---------------------------------------------------------------------------

/// Pack u32 boundary-counts into u8 indices.
///
/// The GPU side computes a u32 per input (count of boundaries
/// strictly less than the value). The CPU side narrows it to u8.
///
/// # Why is the cast lossless?
///
/// For `bits <= 8`, the count is bounded by `2^bits - 1 <= 255`,
/// which fits in a u8 without truncation. The Vulkan kernel and the
/// cubecl kernel both maintain this invariant (`bits` is a comptime
/// constant on the cubecl side and a uniform on the Vulkan side).
///
/// # Why a free function instead of an `Iterator::map`?
///
/// Callers always have a pre-allocated `&mut [u8]` they want to fill;
/// allocating a new `Vec<u8>` here would double the host bandwidth
/// during readback. The free function variant zips the pre-allocated
/// buffer with the GPU-readback `&[u32]` directly.
///
/// # Panics in debug builds
///
/// `debug_assert_eq!(counts.len(), out.len())`. In release builds the
/// `iter().zip()` truncates to the shorter slice; callers should have
/// validated lengths via `validate_quantize` beforehand.
///
/// # Concrete example
///
/// ```text
///  counts: `[0, 1, 2, 3, 7]`
///  out:    `[0, 1, 2, 3, 7]`   // identical, narrowed to u8
/// ```
pub fn pack_u32_counts_to_u8(counts: &[u32], out: &mut [u8]) {
    debug_assert_eq!(counts.len(), out.len());
    for (slot, &count) in out.iter_mut().zip(counts.iter()) {
        // Counts are bounded by 2^bits - 1 <= 255 for bits <= 8, so the
        // cast is lossless. This is the inverse of the boundary search
        // each GPU kernel performs and matches the Vulkan shader's
        // `count` accumulator at the descriptor-set boundary.
        *slot = count as u8;
    }
}

// ---------------------------------------------------------------------------
// Strict CPU reference path
// ---------------------------------------------------------------------------

/// Strict CPU reference path for boundary-search quantization.
///
/// Used as:
///
/// 1. The fallback when no GPU backend is available at runtime.
/// 2. The parity oracle in integration tests (Backend::Cpu must equal
///    Backend::Vulkan must equal Backend::CubeCL bit-for-bit).
///
/// # Complexity
///
/// `O(n * 2^bits)` -- one pass per input value, each scanning the
/// boundary list. For `bits <= 8` (the only configuration the rest of
/// the lane supports) this is at most 256 comparisons per input.
///
/// # Why not SIMD here?
///
/// The actual production CPU path (in `cd_kernel::lloyd_max::*`) uses
/// `wide::f32x8` for ~6x throughput. This reference is intentionally
/// scalar so it can serve as the ground-truth oracle: any divergence
/// between SIMD and scalar would be a SIMD bug, and using SIMD here
/// would lose that ability to catch it.
///
/// # Panics in debug builds
///
/// `debug_assert_eq!(out_indices.len(), req.values.len())`. Production
/// callers should validate via `validate_quantize` first.
///
/// # Worked example
///
/// See module-level docs for a concrete `bits=2` walkthrough.
pub fn quantize_cpu_reference(req: &QuantizeRequest<'_>, out_indices: &mut [u8]) {
    debug_assert_eq!(out_indices.len(), req.values.len());
    for (slot, &v) in out_indices.iter_mut().zip(req.values.iter()) {
        let mut count: u32 = 0;
        // The same scalar reduction the cubecl_backend::quantize_kernel
        // performs per thread; see also vulkan/shaders/quantize.comp:30
        // for the SPIR-V counterpart. Holding all three implementations
        // structurally identical is what makes the parity test
        // meaningful.
        for &b in req.boundaries.iter() {
            if v > b {
                count += 1;
            }
        }
        *slot = count as u8;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Unit tests for the shared GPU-backend helpers.
    //!
    //! These do not exercise any GPU code -- they test the pure
    //! functions that backend launchers call before/after dispatch.
    //! The actual cross-backend parity test lives in
    //! `tests/turboquant_vulkan_parity.rs` and (TaskList #103)
    //! `tests/turboquant_cubecl_parity.rs`.

    use super::*;

    /// `bits = 0` is degenerate (codebook has 0 entries) and must be
    /// rejected before any GPU dispatch. `bits > 8` overflows the
    /// pack stage and is also rejected.
    #[test]
    fn validate_rejects_zero_bits() {
        let req = QuantizeRequest {
            values: &[1.0],
            boundaries: &[],
            bits: 0,
        };
        let mut out = [0u8; 1];
        assert!(validate_quantize(&req, &out).is_err());
        let _ = &mut out;
    }

    /// For `bits=3` the codebook has 8 entries and the boundary slice
    /// must have exactly 7 entries. Passing 2 is rejected.
    #[test]
    fn validate_rejects_wrong_boundary_count() {
        let req = QuantizeRequest {
            values: &[1.0],
            boundaries: &[0.0, 1.0],
            bits: 3,
        };
        let out = [0u8; 1];
        assert!(validate_quantize(&req, &out).is_err());
    }

    /// The CPU reference must implement boundary-search semantics:
    /// the index is the count of boundaries strictly less than the
    /// input value. Tests the worked example from the module-level
    /// docs.
    #[test]
    fn cpu_reference_is_boundary_search() {
        let req = QuantizeRequest {
            values: &[-2.0, 0.0, 0.5, 5.0],
            boundaries: &[-1.0, 0.0, 1.0],
            bits: 2,
        };
        let mut out = [0u8; 4];
        quantize_cpu_reference(&req, &mut out);
        assert_eq!(out, [0, 1, 2, 3]);
    }

    /// The pack step is lossless for the full u8 range. We use 0..=255
    /// rather than 0..=2^bits-1 to lock in that the function's
    /// invariant (counts fit in u8) is structural, not a property of
    /// any particular `bits` configuration.
    #[test]
    fn pack_u32_counts_roundtrip() {
        let counts: Vec<u32> = (0u32..=255).collect();
        let mut out = vec![0u8; counts.len()];
        pack_u32_counts_to_u8(&counts, &mut out);
        for (i, &x) in out.iter().enumerate() {
            assert_eq!(x as u32, counts[i]);
        }
    }
}
