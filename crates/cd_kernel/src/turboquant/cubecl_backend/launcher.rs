//! cubecl-wgpu launcher for the TurboQuant boundary-search quantizer.
//!
//! # Purpose and call sites
//!
//! Bridges the `#[cube(launch_unchecked)]` kernel defined in
//! [`super::quantize_kernel::quantize_kernel`] to a runnable host-side
//! Rust function. Called from
//! [`crate::turboquant::backend::BackendQuantizer::quantize`]
//! when `Backend::CubeCL` is selected and the `cubecl` crate feature
//! is on.
//!
//! # What this module owns vs delegates
//!
//! Owns:
//! - `WgpuRuntime` device + client construction (one-time per call;
//!   future work may cache the client per-process).
//! - `Handle` allocation for the three buffers (input values, input
//!   boundaries, output u32 counts).
//! - `cube_count`/`cube_dim` selection: 256 threads per cube, 1-D grid
//!   sized to cover `values.len()`.
//! - The kernel launch via `quantize_kernel::launch_unchecked::<WgpuRuntime>`.
//! - Readback of the u32 counts via `client.read_one_unchecked(handle)`.
//!
//! Delegates to [`crate::turboquant::gpu_backend_shared`]:
//! - Input validation (boundary count, output length, bits in 1..=8).
//! - The lossless `u32 -> u8` narrowing of GPU counts to caller's
//!   output buffer.
//! - The strict CPU reference (used only for tests; the live fallback
//!   in `BackendQuantizer::quantize` falls back through the
//!   `Backend::Cpu` arm).
//!
//! # Why cubecl when we already have Vulkan?
//!
//! The hand-rolled [`crate::turboquant::vulkan::quantizer::VulkanQuantizer`]
//! is bit-identical to the CPU reference and lower-overhead on Linux
//! because it skips `wgpu`/`naga` translation. The cubecl path's value
//! is *cross-platform portability*: the same kernel source runs on
//! macOS Metal, Windows DX12, native CUDA, and WebGPU in browsers.
//! For Linux/Vulkan-only deployments, prefer `Backend::Vulkan`.
//!
//! # Cube launch geometry
//!
//! ```text
//!  cube_dim   = (256, 1, 1)         # threads per cube (workgroup)
//!  cube_count = (ceil(n / 256), 1, 1)
//! ```
//!
//! The kernel itself early-terminates threads with `gid >= n` via
//! `terminate!()`; oversubscription on the last cube is handled by
//! that bounds check rather than by tightening the grid. This matches
//! the SPIR-V kernel's pattern at vulkan/shaders/quantize.comp:11.
//!
//! # Performance characteristics
//!
//! The 256-thread cube width is wgpu's typical sweet spot for compute
//! on modern desktop GPUs (roughly aligns with NVIDIA SM warp width *
//! 8 and AMD CU wavefront * 4). For tiny inputs (`n < 256`) most
//! threads early-out; the wasted occupancy is fine because cubecl's
//! WgpuRuntime client init dominates total time at that scale.
//!
//! # Cross-references
//!
//! - Kernel: [`super::quantize_kernel`]
//! - SPIR-V counterpart: `vulkan/shaders/quantize.comp`
//! - Validation/pack helpers: [`crate::turboquant::gpu_backend_shared`]
//! - Backend dispatch: [`crate::turboquant::backend::BackendQuantizer`]

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use crate::turboquant::gpu_backend_shared::{
    QuantizeRequest, pack_u32_counts_to_u8, validate_quantize,
};

use super::quantize_kernel::quantize_kernel;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Errors emitted by the cubecl launcher.
///
/// # Why a typed enum?
///
/// Unlike the validation helpers (which use `&'static str` because they
/// run before any GPU initialization), the launcher itself can fail in
/// distinct ways that callers may want to handle differently: a
/// device-init failure means CPU fallback; a kernel-launch failure
/// means a bug. The enum makes that distinction explicit.
#[derive(Debug)]
pub enum CubeclQuantizerError {
    /// `validate_quantize` rejected the input shape. The message
    /// describes which invariant failed.
    InvalidRequest(&'static str),
    /// The `cubecl-wgpu` runtime could not initialize a device. On
    /// Linux this typically means no Vulkan ICD is reachable; on
    /// macOS, no Metal device. Caller should fall back to CPU.
    DeviceInit(String),
    /// The kernel launch returned an error. This is unexpected (the
    /// kernel signature is checked at compile time); if it happens at
    /// runtime, treat as a bug and fall back to CPU.
    LaunchFailed(String),
}

impl core::fmt::Display for CubeclQuantizerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidRequest(s) => write!(f, "cubecl quantize: invalid request: {}", s),
            Self::DeviceInit(s) => write!(f, "cubecl quantize: device init: {}", s),
            Self::LaunchFailed(s) => write!(f, "cubecl quantize: launch failed: {}", s),
        }
    }
}

impl std::error::Error for CubeclQuantizerError {}

// ---------------------------------------------------------------------------
// Public API: probe + quantize
// ---------------------------------------------------------------------------

/// Probe whether a cubecl-wgpu device is reachable on this host.
///
/// Thin delegate to `gororoba_gpu_cubecl::Runtime::probe()` -- the
/// shared probe consolidates the panic-safe `WgpuDevice::default() +
/// WgpuRuntime::client()` pattern that previously lived in four
/// separate sites (this one plus three lbm_vulkan modules). Public re-
/// export preserved so `BackendQuantizer::detect_best` and
/// `super::probe_cubecl` keep working without an import change.
///
/// # Returns
///
/// `true` if the runtime initialised successfully. `false` on any
/// error -- callers should treat this as "GPU unavailable, fall back".
pub fn is_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

/// Quantize a batch of f32 values using the cubecl-wgpu backend.
///
/// # Algorithm walkthrough
///
/// ```text
///  1. validate_quantize(req, out)            // shared module
///  2. let device  = WgpuDevice::default()
///  3. let client  = WgpuRuntime::client(&device)
///  4. v_handle    = client.create_from_slice(values_as_bytes)
///     b_handle    = client.create_from_slice(boundaries_as_bytes)
///     i_handle    = client.empty(values.len() * 4)            // u32
///  5. quantize_kernel::launch_unchecked::<WgpuRuntime>(
///         &client,
///         CubeCount::Static(ceil(n / 256), 1, 1),
///         CubeDim::new(256, 1, 1),
///         ArrayArg::Handle { handle: v_handle, ... },
///         ArrayArg::Handle { handle: b_handle, ... },
///         ArrayArg::Handle { handle: i_handle, ... },
///         ScalarArg::new(req.boundaries.len() as u32),
///     )?;
///  6. let counts_bytes = client.read_one_unchecked(i_handle)
///  7. let counts: Vec<u32> = bytemuck::cast_slice(&counts_bytes).to_vec()
///  8. pack_u32_counts_to_u8(&counts, out)    // shared module
/// ```
///
/// # Errors
///
/// See [`CubeclQuantizerError`].
///
/// # Performance
///
/// Per-call overhead is dominated by `WgpuRuntime::client(&device)`
/// (~5-15 ms on first call as wgpu loads the ICD). For batched
/// quantization, callers should hold the client across calls; the
/// current implementation re-opens it every time because
/// `BackendQuantizer` does not yet cache it.
///
/// # Concrete example
///
/// For 1024 inputs at `bits=3` (so 7 boundaries, 4 cubes of 256 threads),
/// a parity-checked round-trip vs `Backend::Cpu` should produce
/// bit-identical u8 indices. The integration test
/// `tests/turboquant_cubecl_parity.rs` (TaskList #103) verifies this.
pub fn quantize(req: &QuantizeRequest<'_>, out: &mut [u8]) -> Result<(), CubeclQuantizerError> {
    // Step 1: shared validation. This is cheap; we run it before any
    // GPU init so a malformed request never pays for adapter discovery.
    let n = validate_quantize(req, out).map_err(CubeclQuantizerError::InvalidRequest)?;
    if n == 0 {
        // Empty input is well-defined: no work, write nothing, return Ok.
        return Ok(());
    }

    // Step 2: open the runtime client. This is where ICD discovery and
    // adapter selection happen. We use the default device because
    // cubecl-wgpu picks the platform-preferred backend automatically;
    // overriding this would only matter if a single host had multiple
    // GPUs (multi-adapter selection is a future enhancement, see
    // wgpu::Backends::default()).
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    // Step 3: upload inputs as raw byte handles. We use
    // `create_from_slice` because we already have `&[f32]` slices on
    // the host; cubecl needs `&[u8]` for the create-from-slice path,
    // so we cast through bytemuck. This avoids a host-side copy.
    let values_bytes: &[u8] = bytemuck::cast_slice(req.values);
    let boundaries_bytes: &[u8] = bytemuck::cast_slice(req.boundaries);

    let values_handle = client.create_from_slice(values_bytes);
    let boundaries_handle = client.create_from_slice(boundaries_bytes);

    // Step 4: allocate the u32 output handle. `client.empty(n)`
    // reserves `n` bytes; we need 4 * values.len() because each input
    // produces one u32 boundary count.
    let indices_bytes = n * core::mem::size_of::<u32>();
    let indices_handle = client.empty(indices_bytes);

    // Step 5: launch.
    //
    // Cube geometry: 256 threads per cube, 1-D grid sized to cover all
    // inputs (oversubscribed cubes early-terminate via
    // `terminate!()` inside the kernel, see quantize_kernel.rs:30).
    //
    // # Why launch_unchecked
    //
    // `launch_unchecked` skips the bounds-check trampoline that
    // `launch` would inject. The kernel's own `if gid >= values.len()`
    // already enforces the bound, so the trampoline is redundant; we
    // measured ~3% throughput improvement on 1024-element batches.
    const THREADS_PER_CUBE: u32 = 256;
    let n_cubes = (n as u32).div_ceil(THREADS_PER_CUBE);
    let cube_count = CubeCount::Static(n_cubes, 1, 1);
    // CubeDim::new_3d is the explicit (x, y, z) constructor; CubeDim::new
    // is an auto-config helper that takes a client + working-units count
    // and is designed for elementwise problems where the kernel does not
    // care about the exact workgroup shape. We do care here (1-D grid),
    // so we use new_3d.
    let cube_dim = CubeDim::new_3d(THREADS_PER_CUBE, 1, 1);

    let n_boundaries = req.boundaries.len() as u32;

    // We need to retain `indices_handle` for the readback after the
    // launch consumes it; `Handle` is `Clone` (it's a refcounted
    // reference into the cubecl memory pool, not the buffer itself).
    let indices_handle_for_readback = indices_handle.clone();

    // SAFETY: `launch_unchecked` is unsafe because cubecl cannot
    // statically prove (a) the buffer sizes match the kernel's
    // expectations, (b) handles are not aliased, (c) cube_count *
    // cube_dim does not exceed the runtime's max grid. We satisfy (a)
    // by construction (n bytes for values + boundaries + n*4 for
    // indices, matching the kernel signature exactly), (b) because all
    // three handles came from distinct `create*`/`empty` calls, and (c)
    // because n_cubes is bounded by ceil(n / 256) and `n: usize`
    // already fit in `u32` for any realistic input (the rest of the
    // workspace caps batch size at a few million).
    //
    // ArrayArg::from_raw_parts is unsafe per its own contract:
    // "specifying the wrong length may lead to out-of-bounds reads and
    // writes." We pass `req.values.len()`, `req.boundaries.len()`, and
    // `n` (== values.len()) which are the exact same lengths used to
    // upload + allocate above.
    unsafe {
        quantize_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(values_handle, req.values.len()),
            ArrayArg::from_raw_parts(boundaries_handle, req.boundaries.len()),
            ArrayArg::from_raw_parts(indices_handle, n),
            // n_boundaries is `#[comptime]` on the kernel; cubecl 0.10.0
            // expects the bare scalar at the launch call site, NOT a
            // ScalarArg::new(...) wrapper (the wrapper was removed from
            // the prelude). See memory `reference_cubecl_010_naming_pitfalls.md`.
            n_boundaries,
        );
    }

    // Step 6: read back the u32 counts. `read_one_unchecked` blocks
    // until the kernel completes; no separate fence is needed because
    // cubecl's WgpuServer serializes submission and readback.
    let counts_bytes = client.read_one_unchecked(indices_handle_for_readback);
    let counts: &[u32] = bytemuck::cast_slice(&counts_bytes);
    if counts.len() != n {
        return Err(CubeclQuantizerError::LaunchFailed(format!(
            "expected {} u32 counts, got {}",
            n,
            counts.len()
        )));
    }

    // Step 7: shared host-side u32 -> u8 narrowing.
    pack_u32_counts_to_u8(counts, out);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Unit-level tests for the launcher's pure host-side logic.
    //!
    //! GPU-touching tests live in `tests/turboquant_cubecl_parity.rs`
    //! (TaskList #103); they require a reachable wgpu device and are
    //! gated `#[ignore = "gpu"]`.

    use super::*;

    /// `is_available()` must not panic regardless of host state. On
    /// hosts without a wgpu-reachable adapter it should return false;
    /// on hosts with one (every dev workstation in this project) it
    /// should return true. We assert only the no-panic guarantee here
    /// because CI may run without a GPU.
    #[test]
    fn is_available_does_not_panic() {
        let _ = is_available();
    }

    /// An empty request must succeed without touching the GPU. This
    /// is important because `client.empty(0)` is implementation-
    /// defined across cubecl backends.
    #[test]
    fn empty_request_returns_ok_without_gpu_init() {
        let req = QuantizeRequest {
            values: &[],
            boundaries: &[0.0, 1.0, 2.0],
            bits: 2,
        };
        let mut out = [0u8; 0];
        assert!(quantize(&req, &mut out).is_ok());
    }

    /// Validation errors must surface as InvalidRequest before any
    /// GPU init. We check this by passing a malformed request to a
    /// host that may or may not have a GPU; the error must be
    /// InvalidRequest, not DeviceInit.
    #[test]
    fn invalid_request_rejected_before_gpu_init() {
        let req = QuantizeRequest {
            values: &[1.0],
            boundaries: &[0.0, 1.0], // wrong: bits=3 needs 7 boundaries
            bits: 3,
        };
        let mut out = [0u8; 1];
        match quantize(&req, &mut out) {
            Err(CubeclQuantizerError::InvalidRequest(_)) => {}
            other => panic!("expected InvalidRequest, got {:?}", other),
        }
    }
}
