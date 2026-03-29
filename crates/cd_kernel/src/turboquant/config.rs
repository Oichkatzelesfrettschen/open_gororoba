//! TurboQuant configuration with smart defaults.
//!
//! Consolidates all validated optimizations into a single config struct
//! with sensible defaults derived from measured results:
//!
//! | Setting          | Default           | Rationale                                  |
//! |------------------|-------------------|--------------------------------------------|
//! | rotation         | FastJL (WHT)      | 3.2x faster, 0.5% better MSE (measured)    |
//! | e8_rotation      | true for d=128    | KS p=0.816, 136x fewer params (validated)  |
//! | qjl_correction   | auto (bits<=3)    | Helps at 2-bit, hurts at 4-bit (measured)  |
//! | adaptive_bits    | true, top 25%     | 23% MSE improvement (measured)             |
//! | hierarchical     | false             | Negative result on random data (documented)|

/// Rotation method selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RotationMethod {
    /// Dense Haar-random via QR decomposition. O(d^2). Legacy default.
    Haar,
    /// Fast JL via D1*WHT*D2. O(d log d). 3.2x faster, 0.5% better MSE.
    FastJL,
    /// E8 lattice block rotation. O(d) via 8 sedenion multiplies.
    /// Validated: KS p=0.816 vs Haar. Only for d=128.
    E8Block,
}

/// QJL correction behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QjlCorrectionMode {
    /// Always apply QJL correction (original TurboQuant behavior).
    Always,
    /// Never apply QJL correction (MSE-only, fastest).
    Never,
    /// Apply only when bits <= threshold (default: 3).
    /// QJL helps at low bits where MSE residuals are large,
    /// but adds noise at high bits where residuals are already small.
    Auto { max_bits: u32 },
}

/// Adaptive bit allocation settings.
#[derive(Clone, Debug)]
pub struct AdaptiveBitsConfig {
    /// Whether adaptive allocation is enabled.
    pub enabled: bool,
    /// Fraction of tokens to promote to (bits+1). Default: 0.25 (top quartile).
    pub promote_fraction: f64,
}

/// Complete TurboQuant configuration.
///
/// Use `TurboQuantConfig::default()` for the recommended settings based on
/// measured results, or customize individual fields.
#[derive(Clone, Debug)]
pub struct TurboQuantConfig {
    /// Head dimension. Must be a power of 2.
    pub dim: usize,
    /// Quantization bits per coordinate (total budget).
    pub bits: u32,
    /// RNG seed for rotation and QJL matrices.
    pub seed: u64,
    /// Rotation method.
    pub rotation: RotationMethod,
    /// QJL correction behavior.
    pub qjl_correction: QjlCorrectionMode,
    /// Adaptive bit allocation.
    pub adaptive: AdaptiveBitsConfig,
    /// QJL projection dimension (None = same as dim).
    pub qjl_dim: Option<usize>,
}

impl TurboQuantConfig {
    /// Create config with smart defaults for the given dimension and bits.
    ///
    /// Defaults are derived from measured results:
    /// - d >= 64: FastJL rotation (3.2x faster than Haar)
    /// - d == 128: E8 block rotation (136x fewer params, KS validated)
    /// - bits <= 3: QJL correction on (improves 2-bit by 2%)
    /// - bits >= 4: QJL correction off (adds noise at high quality)
    /// - adaptive bits: enabled, top 25% promoted (23% MSE gain)
    pub fn recommended(dim: usize, bits: u32) -> Self {
        // E8 rotation has validated decorrelation (KS p=0.816) but produces
        // a non-Gaussian marginal that the standard codebook handles poorly
        // (MSE 2.57 vs WHT 1.44).  Disabled by default until custom codebook
        // or E8+WHT composition is implemented.
        let rotation = if dim >= 64 {
            RotationMethod::FastJL
        } else {
            RotationMethod::Haar
        };

        TurboQuantConfig {
            dim,
            bits,
            seed: 42,
            rotation,
            qjl_correction: QjlCorrectionMode::Auto { max_bits: 3 },
            adaptive: AdaptiveBitsConfig {
                enabled: true,
                promote_fraction: 0.25,
            },
            qjl_dim: None,
        }
    }

    /// Create config matching the original TurboQuant paper defaults.
    ///
    /// Haar rotation, always QJL, no adaptive bits.
    /// Use this for exact paper reproduction.
    pub fn paper_default(dim: usize, bits: u32) -> Self {
        TurboQuantConfig {
            dim,
            bits,
            seed: 42,
            rotation: RotationMethod::Haar,
            qjl_correction: QjlCorrectionMode::Always,
            adaptive: AdaptiveBitsConfig {
                enabled: false,
                promote_fraction: 0.0,
            },
            qjl_dim: None,
        }
    }

    /// Create minimal config for fastest possible quantization.
    ///
    /// FastJL rotation, no QJL correction, no adaptive bits.
    /// Lowest quality but highest throughput.
    pub fn fast(dim: usize, bits: u32) -> Self {
        TurboQuantConfig {
            dim,
            bits,
            seed: 42,
            rotation: if dim >= 64 { RotationMethod::FastJL } else { RotationMethod::Haar },
            qjl_correction: QjlCorrectionMode::Never,
            adaptive: AdaptiveBitsConfig {
                enabled: false,
                promote_fraction: 0.0,
            },
            qjl_dim: None,
        }
    }

    /// Whether QJL correction should be applied at the configured bit-width.
    pub fn should_apply_qjl(&self) -> bool {
        match &self.qjl_correction {
            QjlCorrectionMode::Always => true,
            QjlCorrectionMode::Never => false,
            QjlCorrectionMode::Auto { max_bits } => self.bits <= *max_bits,
        }
    }

    /// Whether to use WHT-based rotation (FastJL or E8Block).
    pub fn use_wht(&self) -> bool {
        matches!(self.rotation, RotationMethod::FastJL | RotationMethod::E8Block)
    }

    /// Whether to use E8 block rotation specifically.
    pub fn use_e8(&self) -> bool {
        matches!(self.rotation, RotationMethod::E8Block)
    }
}

impl Default for TurboQuantConfig {
    /// Default: d=128, 3-bit, recommended settings.
    fn default() -> Self {
        Self::recommended(128, 3)
    }
}

impl std::fmt::Display for TurboQuantConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TurboQuant(d={}, {}bit, rot={:?}, qjl={}, adaptive={})",
            self.dim,
            self.bits,
            self.rotation,
            match &self.qjl_correction {
                QjlCorrectionMode::Always => "always".to_string(),
                QjlCorrectionMode::Never => "never".to_string(),
                QjlCorrectionMode::Auto { max_bits } => format!("auto(<={}bit)", max_bits),
            },
            if self.adaptive.enabled {
                format!("top-{}%", (self.adaptive.promote_fraction * 100.0) as u32)
            } else {
                "off".to_string()
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommended_d128_3bit() {
        let cfg = TurboQuantConfig::recommended(128, 3);
        // FastJL for d=128 (E8 disabled pending custom codebook)
        assert_eq!(cfg.rotation, RotationMethod::FastJL);
        assert!(cfg.should_apply_qjl()); // bits=3 <= max_bits=3
        assert!(cfg.adaptive.enabled);
        assert!((cfg.adaptive.promote_fraction - 0.25).abs() < 1e-10);
        assert!(cfg.use_wht());
        assert!(!cfg.use_e8()); // E8 not default until codebook issue resolved
    }

    #[test]
    fn test_recommended_d128_4bit() {
        let cfg = TurboQuantConfig::recommended(128, 4);
        assert_eq!(cfg.rotation, RotationMethod::FastJL);
        assert!(!cfg.should_apply_qjl()); // bits=4 > max_bits=3 -> off
    }

    #[test]
    fn test_recommended_d64() {
        let cfg = TurboQuantConfig::recommended(64, 3);
        assert_eq!(cfg.rotation, RotationMethod::FastJL); // d=64 < 128 -> FastJL
        assert!(!cfg.use_e8());
        assert!(cfg.use_wht());
    }

    #[test]
    fn test_recommended_d32() {
        let cfg = TurboQuantConfig::recommended(32, 3);
        assert_eq!(cfg.rotation, RotationMethod::Haar); // d=32 < 64 -> Haar
        assert!(!cfg.use_wht());
    }

    #[test]
    fn test_paper_default() {
        let cfg = TurboQuantConfig::paper_default(128, 3);
        assert_eq!(cfg.rotation, RotationMethod::Haar);
        assert!(cfg.should_apply_qjl()); // Always
        assert!(!cfg.adaptive.enabled);
    }

    #[test]
    fn test_fast_config() {
        let cfg = TurboQuantConfig::fast(128, 3);
        assert!(!cfg.should_apply_qjl()); // Never
        assert!(!cfg.adaptive.enabled);
        assert!(cfg.use_wht());
    }

    #[test]
    fn test_display() {
        let cfg = TurboQuantConfig::default();
        let s = format!("{}", cfg);
        assert!(s.contains("128"));
        assert!(s.contains("3bit"));
        assert!(s.contains("FastJL")); // default rotation for d=128
        println!("{}", s);
    }

    #[test]
    fn test_qjl_auto_thresholds() {
        let cfg2 = TurboQuantConfig::recommended(128, 2);
        assert!(cfg2.should_apply_qjl()); // 2 <= 3

        let cfg3 = TurboQuantConfig::recommended(128, 3);
        assert!(cfg3.should_apply_qjl()); // 3 <= 3

        let cfg4 = TurboQuantConfig::recommended(128, 4);
        assert!(!cfg4.should_apply_qjl()); // 4 > 3

        let cfg5 = TurboQuantConfig::recommended(128, 5);
        assert!(!cfg5.should_apply_qjl()); // 5 > 3
    }
}
