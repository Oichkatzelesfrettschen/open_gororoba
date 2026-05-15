//! NVRTC compile-options builder.
//!
//! Consolidates 35+ ad-hoc `cudarc::nvrtc::CompileOptions` sites that
//! independently pick arch / lineinfo / fast_math flags. The canonical
//! variations seen in the workspace:
//!   - Default (no opts) -- gororoba_algebra/src/gpu/*.rs
//!   - lineinfo + fastmath -- lbm_3d_cuda kernels
//!   - hardcoded sm_70 -- grmhd_core/src/gpu.rs
//!   - dynamic arch via `compile_arch()` -- cd_kernel/turboquant
//!
//! This builder exposes them all explicitly so callers see the choice.

use cudarc::nvrtc::{CompileOptions as NvrtcOptions, Ptx, compile_ptx_with_opts};

use crate::error::Result;
use crate::probe::DeviceProbe;

/// Builder for NVRTC compile options.
///
/// # Example
///
/// ```ignore
/// use gororoba_gpu_cuda::CompileOptions;
///
/// let probe = gororoba_gpu_cuda::DeviceProbe::query()?;
/// let opts = CompileOptions::for_arch(probe.major, probe.minor)
///     .lineinfo(true)
///     .fast_math(true)
///     .build();
/// let ptx = CompileOptions::compile_ptx(KERNEL_SRC, &opts)?;
/// # Ok::<(), gororoba_gpu_cuda::CudaError>(())
/// ```
#[derive(Clone, Debug)]
pub struct CompileOptions {
    arch: Option<&'static str>,
    lineinfo: bool,
    fast_math: bool,
}

impl CompileOptions {
    /// Empty options (no arch, no lineinfo, no fast-math). Matches
    /// `Default::default()` in cudarc.
    pub fn empty() -> Self {
        Self {
            arch: None,
            lineinfo: false,
            fast_math: false,
        }
    }

    /// Options targeted at the given compute capability. Picks the
    /// matching `sm_XX` string via `DeviceProbe::compile_arch`-style
    /// table. The returned arch is a static literal -- no allocation.
    pub fn for_arch(major: u32, minor: u32) -> Self {
        let arch: &'static str = match (major, minor) {
            (9, _) => "sm_90",
            (8, 9) => "sm_89",
            (8, m) if (6..=8).contains(&m) => "sm_86",
            (8, _) => "sm_80",
            (7, m) if m >= 5 => "sm_75",
            _ => "sm_52",
        };
        Self {
            arch: Some(arch),
            lineinfo: false,
            fast_math: false,
        }
    }

    /// Options targeted at the device probed via `DeviceProbe::query`.
    pub fn for_device(probe: &DeviceProbe) -> Self {
        Self::for_arch(probe.major, probe.minor)
    }

    /// Set `--lineinfo` (debug-friendly PTX line mapping).
    pub fn lineinfo(mut self, enabled: bool) -> Self {
        self.lineinfo = enabled;
        self
    }

    /// Set `--use_fast_math` (looser-precision intrinsics).
    pub fn fast_math(mut self, enabled: bool) -> Self {
        self.fast_math = enabled;
        self
    }

    /// Materialize the underlying cudarc `CompileOptions`.
    pub fn build(self) -> NvrtcOptions {
        let mut extra_options = Vec::new();
        if self.lineinfo {
            extra_options.push("--generate-line-info".to_string());
        }
        NvrtcOptions {
            arch: self.arch,
            include_paths: Vec::new(),
            use_fast_math: if self.fast_math { Some(true) } else { None },
            options: extra_options,
            ..Default::default()
        }
    }

    /// Compile CUDA C source to PTX with these options. Convenience for
    /// callers that do not need the `NvrtcOptions` instance.
    pub fn compile_ptx(source: &str, opts: &Self) -> Result<Ptx> {
        let nvrtc_opts = opts.clone().build();
        Ok(compile_ptx_with_opts(source, nvrtc_opts)?)
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self::empty()
    }
}

