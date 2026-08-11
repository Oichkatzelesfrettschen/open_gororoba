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
use sha2::{Digest, Sha256};

use crate::{error::Result, probe::DeviceProbe};

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
    include_paths: Vec<String>,
    lineinfo: bool,
    fast_math: bool,
    prec_div: Option<bool>,
    prec_sqrt: Option<bool>,
    ftz: Option<bool>,
    fmad: Option<bool>,
    extra_options: Vec<String>,
}

impl CompileOptions {
    /// Empty options (no arch, no lineinfo, no fast-math). Matches
    /// `Default::default()` in cudarc.
    pub fn empty() -> Self {
        Self {
            arch: None,
            include_paths: Vec::new(),
            lineinfo: false,
            fast_math: false,
            prec_div: None,
            prec_sqrt: None,
            ftz: None,
            fmad: None,
            extra_options: Vec::new(),
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
            (7, _) => "sm_70",
            _ => "sm_52",
        };
        Self {
            arch: Some(arch),
            include_paths: Vec::new(),
            lineinfo: false,
            fast_math: false,
            prec_div: None,
            prec_sqrt: None,
            ftz: None,
            fmad: None,
            extra_options: Vec::new(),
        }
    }

    /// Options targeted at the device probed via `DeviceProbe::query`.
    pub fn for_device(probe: &DeviceProbe) -> Self {
        Self::for_arch(probe.major, probe.minor)
    }

    /// Options targeted at a preselected CUDA architecture string.
    pub fn with_arch(arch: &'static str) -> Self {
        Self {
            arch: Some(arch),
            include_paths: Vec::new(),
            lineinfo: false,
            fast_math: false,
            prec_div: None,
            prec_sqrt: None,
            ftz: None,
            fmad: None,
            extra_options: Vec::new(),
        }
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

    /// Set NVRTC precise-division behavior.
    pub fn prec_div(mut self, enabled: bool) -> Self {
        self.prec_div = Some(enabled);
        self
    }

    /// Set NVRTC precise-square-root behavior.
    pub fn prec_sqrt(mut self, enabled: bool) -> Self {
        self.prec_sqrt = Some(enabled);
        self
    }

    /// Set NVRTC flush-to-zero behavior.
    pub fn ftz(mut self, enabled: bool) -> Self {
        self.ftz = Some(enabled);
        self
    }

    /// Set NVRTC fused-multiply-add behavior.
    pub fn fmad(mut self, enabled: bool) -> Self {
        self.fmad = Some(enabled);
        self
    }

    /// Add a header search path passed through NVRTC's include path list.
    pub fn include_path(mut self, path: impl Into<String>) -> Self {
        self.include_paths.push(path.into());
        self
    }

    /// Add a `-DNAME=value` define to the NVRTC command line.
    pub fn define(mut self, name: &str, value: impl std::fmt::Display) -> Self {
        self.extra_options.push(format!("-D{name}={value}"));
        self
    }

    /// Add an arbitrary NVRTC command-line option.
    pub fn option(mut self, option: impl Into<String>) -> Self {
        self.extra_options.push(option.into());
        self
    }

    /// Materialize the underlying cudarc `CompileOptions`.
    pub fn build(self) -> NvrtcOptions {
        let mut extra_options = Vec::new();
        if self.lineinfo {
            extra_options.push("--generate-line-info".to_string());
        }
        extra_options.extend(self.extra_options);
        NvrtcOptions {
            arch: self.arch,
            include_paths: self.include_paths,
            use_fast_math: if self.fast_math { Some(true) } else { None },
            prec_div: self.prec_div,
            prec_sqrt: self.prec_sqrt,
            ftz: self.ftz,
            fmad: self.fmad,
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

    /// Return a deterministic digest of every option that affects NVRTC output.
    ///
    /// The digest is part of [`crate::ModuleProvenance`]. Include paths and
    /// extra options are length-delimited so adjacent values cannot collide in
    /// the identity input.
    pub fn fingerprint(&self) -> String {
        let mut identity = String::new();
        append_identity_field(&mut identity, self.arch.unwrap_or("<none>"));
        append_identity_field(&mut identity, &self.lineinfo.to_string());
        append_identity_field(&mut identity, &self.fast_math.to_string());
        append_identity_field(
            &mut identity,
            &self
                .prec_div
                .map(|value| value.to_string())
                .unwrap_or_else(|| "<none>".to_string()),
        );
        append_identity_field(
            &mut identity,
            &self
                .prec_sqrt
                .map(|value| value.to_string())
                .unwrap_or_else(|| "<none>".to_string()),
        );
        append_identity_field(
            &mut identity,
            &self
                .ftz
                .map(|value| value.to_string())
                .unwrap_or_else(|| "<none>".to_string()),
        );
        append_identity_field(
            &mut identity,
            &self
                .fmad
                .map(|value| value.to_string())
                .unwrap_or_else(|| "<none>".to_string()),
        );
        for path in &self.include_paths {
            append_identity_field(&mut identity, path);
        }
        for option in &self.extra_options {
            append_identity_field(&mut identity, option);
        }

        let digest = Sha256::digest(identity.as_bytes());
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

fn append_identity_field(identity: &mut String, value: &str) {
    identity.push_str(&value.len().to_string());
    identity.push(':');
    identity.push_str(value);
    identity.push('|');
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::CompileOptions;

    #[test]
    fn fingerprint_changes_when_codegen_options_change() {
        let baseline = CompileOptions::with_arch("sm_89").fingerprint();
        let fast_math = CompileOptions::with_arch("sm_89")
            .fast_math(true)
            .fingerprint();
        let other_arch = CompileOptions::with_arch("sm_90").fingerprint();

        assert_ne!(baseline, fast_math);
        assert_ne!(baseline, other_arch);
        assert_eq!(baseline.len(), 64);
    }
}
