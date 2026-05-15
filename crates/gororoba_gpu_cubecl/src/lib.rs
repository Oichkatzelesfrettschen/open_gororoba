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
/// The `skip_if_unavailable!` macro emits a `println!` + `return` when the
/// cubecl wgpu runtime cannot be initialised. In default builds (no `cubecl`
/// feature) the macro always returns -- callers can write tests gated by
/// `#[cfg(feature = "cubecl")]` if they need the actual probe.
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
            #[cfg(feature = "cubecl")]
            {
                if !$crate::Runtime::probe() {
                    eprintln!("[gpu_cubecl] skip: wgpu runtime not available");
                    return;
                }
            }
            #[cfg(not(feature = "cubecl"))]
            {
                eprintln!("[gpu_cubecl] skip: feature cubecl is disabled");
                return;
            }
        };
    }
    pub use skip_if_unavailable;
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
