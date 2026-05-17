//! Shared cubecl-wgpu runtime helpers for the open_gororoba workspace.
//!
//! WHY: Prior to this crate, four cubecl-using sites
//! (cd_kernel/src/turboquant/cubecl_backend/launcher.rs and three modules in
//! lbm_vulkan: box_counting_cubecl.rs, chingon_cubecl.rs,
//! transform_viscosity_cubecl.rs) each hand-rolled the identical 11-line
//! `is_available()` probe + 2-line `WgpuDevice::default() + WgpuRuntime::client`
//! acquisition sequence. This crate consolidates them.
//!
//! WHAT: Three helper surfaces behind feature `cubecl`:
//!   - `Runtime::probe` -- panic-safe adapter probe (consolidates 4 sites).
//!   - `Runtime::client` -- (device, client) acquisition.
//!   - `test_support::skip_if_unavailable!` macro for parity tests.
//!
//! HOW: Default-features build pulls only `gororoba_gpu_bridge` (re-exports).
//! Enabling `--features cubecl` activates the cubecl + cubecl-wgpu deps.
//!
//! cubecl 0.10 has documented runtime-pitfalls (see project memory
//! `reference_cubecl_010_naming_pitfalls.md`); this crate centralises the
//! probe-without-panic pattern so consumers do not re-discover them.

pub use gororoba_gpu_bridge::{ComputeBackend, HardwareCaps, StoragePrecision};

#[cfg(feature = "cubecl")]
mod runtime;

#[cfg(feature = "cubecl")]
pub use runtime::Runtime;

/// Test-support macros + helpers, available in all builds.
///
/// The `skip_if_unavailable!` macro emits an `eprintln!` + `return` when
/// the cubecl wgpu runtime cannot be initialised. The decision is made
/// inside this crate (via [`__skip_if_unavailable_should_skip`]) so the
/// macro evaluates `feature = "cubecl"` against *this* crate's features
/// at definition time, not the calling crate's features after macro
/// expansion. Without that indirection a downstream test crate that
/// enabled `gororoba_gpu_cubecl/cubecl` but did not also define a
/// same-named local feature would unconditionally take the "feature
/// disabled" branch and silently skip every GPU assertion.
pub mod test_support {
    /// Skip the calling test if the cubecl wgpu runtime is unavailable.
    ///
    /// In a build with `--features cubecl`, this calls
    /// `Runtime::probe()` and `return`s if false.
    ///
    /// In a default build (no cubecl), this always `return`s.
    ///
    /// # Example
    ///
    /// ```ignore
    /// #[test]
    /// fn my_parity_test() {
    ///     gororoba_gpu_cubecl::test_support::skip_if_unavailable!();
    ///     // ... rest of test that needs the runtime ...
    /// }
    /// ```
    #[macro_export]
    macro_rules! skip_if_unavailable {
        () => {
            let (should_skip, reason) = $crate::__skip_if_unavailable_should_skip();
            if should_skip {
                eprintln!("[gpu_cubecl] skip: {}", reason);
                return;
            }
        };
    }
    pub use skip_if_unavailable;
}

/// Internal: decide whether the caller's GPU test should be skipped.
///
/// Returns `(true, reason)` to skip, `(false, _)` to proceed. The
/// feature gate that selects between "runtime probe" and "feature
/// disabled" is evaluated in this crate, so the macro consumer sees
/// the correct decision regardless of which features its own crate
/// enables.
#[doc(hidden)]
#[cfg(feature = "cubecl")]
pub fn __skip_if_unavailable_should_skip() -> (bool, &'static str) {
    if Runtime::probe() {
        (false, "")
    } else {
        (true, "wgpu runtime not available")
    }
}

#[doc(hidden)]
#[cfg(not(feature = "cubecl"))]
pub fn __skip_if_unavailable_should_skip() -> (bool, &'static str) {
    (true, "feature cubecl is disabled in gororoba_gpu_cubecl")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_exports_compile() {
        let _: ComputeBackend = ComputeBackend::CpuScalar;
        let _: StoragePrecision = StoragePrecision::Fp32;
    }
}
