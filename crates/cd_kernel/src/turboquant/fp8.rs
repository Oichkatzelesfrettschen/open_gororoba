//! FP8 E4M3 and E5M2 types for compact quantization metadata storage.
//!
//! FP8 E4M3: 1 sign + 4 exponent + 3 mantissa = 8 bits
//!   Range: +/- 448, precision: ~3 significant digits
//!   Used for: per-group scale factors (replaces f32, saves 24 bits/group)
//!
//! FP8 E5M2: 1 sign + 5 exponent + 2 mantissa = 8 bits
//!   Range: +/- 57344, precision: ~2 significant digits
//!   Used for: zero-points with wider range needs
//!
//! Combined with group_size=16 at d=128:
//!   8 groups * 8 bits = 64 bits overhead (vs 512 bits with f32)
//!   Effective bits at 2-bit: (256 + 64 + 16) / 128 = 2.62 bits/coord
//!
//! Zero-copy via bytemuck (matches steinmarder SoA pattern).

/// FP8 E4M3 format: 1 sign + 4 exponent + 3 mantissa.
///
/// Bias = 7, max exponent = 15, max value = 448.
/// Special: NaN = 0x7F (no infinity representation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Fp8E4M3(pub u8);

/// FP8 E5M2 format: 1 sign + 5 exponent + 2 mantissa.
///
/// Bias = 15, max exponent = 31, max value = 57344.
/// Special: +/-Inf, NaN (like IEEE 754).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Fp8E5M2(pub u8);

// SAFETY: Both are #[repr(transparent)] wrappers around u8
unsafe impl bytemuck::Pod for Fp8E4M3 {}
unsafe impl bytemuck::Zeroable for Fp8E4M3 {}
unsafe impl bytemuck::Pod for Fp8E5M2 {}
unsafe impl bytemuck::Zeroable for Fp8E5M2 {}

impl Fp8E4M3 {
    /// Convert f32 to FP8 E4M3 (truncation, no rounding).
    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Fp8E4M3(0);
        }

        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let f32_exp = ((bits >> 23) & 0xFF) as i32;
        let f32_mant = bits & 0x7FFFFF;

        // E4M3: bias=7, so exponent range [-7, 8] maps to biased [0, 15]
        let exp = f32_exp - 127 + 7; // rebias from f32 (bias=127) to E4M3 (bias=7)

        if exp >= 15 {
            // Overflow -> max finite (no inf in E4M3)
            return Fp8E4M3((sign as u8) << 7 | 0x7E); // max: s_1111_110
        }
        if exp <= 0 {
            // Underflow -> zero (skip denormals for simplicity)
            return Fp8E4M3((sign as u8) << 7);
        }

        // Truncate mantissa from 23 bits to 3 bits
        let mant = (f32_mant >> 20) as u8; // top 3 bits of mantissa
        Fp8E4M3((sign as u8) << 7 | (exp as u8) << 3 | mant)
    }

    /// Convert FP8 E4M3 to f32.
    pub fn to_f32(self) -> f32 {
        let sign = (self.0 >> 7) & 1;
        let exp = ((self.0 >> 3) & 0xF) as i32;
        let mant = (self.0 & 0x7) as u32;

        if exp == 0 && mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        if exp == 0 {
            // Denormal: 2^(1-7) * (mant/8) = 2^-6 * mant/8
            let val = (mant as f32) / 8.0 * (2.0_f32).powi(-6);
            return if sign == 1 { -val } else { val };
        }
        if exp == 15 && mant == 7 {
            return f32::NAN; // E4M3 NaN
        }

        // Normal: 2^(exp-7) * (1 + mant/8)
        let f32_exp = (exp - 7 + 127) as u32;
        let f32_mant = mant << 20; // shift 3-bit mantissa to f32's 23-bit field
        let f32_bits = (sign as u32) << 31 | f32_exp << 23 | f32_mant;
        f32::from_bits(f32_bits)
    }
}

impl Fp8E5M2 {
    /// Convert f32 to FP8 E5M2 (truncation).
    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Fp8E5M2(0);
        }

        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let f32_exp = ((bits >> 23) & 0xFF) as i32;
        let f32_mant = bits & 0x7FFFFF;

        let exp = f32_exp - 127 + 15; // rebias to E5M2 (bias=15)

        if exp >= 31 {
            return Fp8E5M2((sign as u8) << 7 | 0x7C); // +/-Inf
        }
        if exp <= 0 {
            return Fp8E5M2((sign as u8) << 7);
        }

        let mant = (f32_mant >> 21) as u8; // top 2 bits
        Fp8E5M2((sign as u8) << 7 | (exp as u8) << 2 | mant)
    }

    /// Convert FP8 E5M2 to f32.
    pub fn to_f32(self) -> f32 {
        let sign = (self.0 >> 7) & 1;
        let exp = ((self.0 >> 2) & 0x1F) as i32;
        let mant = (self.0 & 0x3) as u32;

        if exp == 0 && mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        if exp == 31 {
            return if mant == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            };
        }
        if exp == 0 {
            let val = (mant as f32) / 4.0 * (2.0_f32).powi(-14);
            return if sign == 1 { -val } else { val };
        }

        let f32_exp = (exp - 15 + 127) as u32;
        let f32_mant = mant << 21;
        let f32_bits = (sign as u32) << 31 | f32_exp << 23 | f32_mant;
        f32::from_bits(f32_bits)
    }
}

/// Compact group metadata using FP8 (8 bits per scale, vs 32 for f32).
///
/// Reduces per-group overhead from 64 bits (f32 scale + f32 zp) to 8 bits
/// (FP8 scale only, symmetric groups assumed after rotation).
///
/// Scales are pre-multiplied by 16.0 before FP8 encoding to shift the
/// typical range `[0.01, 0.5]` into `[0.16, 8.0]` which fits E4M3 precision.
///
/// At d=128, group_size=16: 8 groups * 8 bits = 64 bits metadata.
/// Effective: (256 data + 64 meta + 16 norm) / 128 = 2.62 bits/coord.
#[derive(Clone, Debug)]
pub struct CompactGroupMeta {
    /// FP8 E4M3 scales (pre-multiplied by SCALE_FACTOR), one per group.
    pub scales: Vec<Fp8E4M3>,
    /// FP8 E5M2 zero-points for asymmetric groups (empty if all symmetric).
    pub zero_points: Vec<Fp8E5M2>,
    /// Bitmask: bit i = 1 means group i is asymmetric (has zero_point).
    pub asymmetric_mask: u16,
    pub group_size: usize,
}

/// Scale pre-multiplication factor to shift typical ranges into FP8 precision.
const SCALE_FACTOR: f32 = 16.0;

impl CompactGroupMeta {
    /// Convert from full-precision GroupQuantParams.
    pub fn from_params(params: &super::grouping::GroupQuantParams) -> Self {
        let scales: Vec<Fp8E4M3> = params
            .scales
            .iter()
            .map(|&s| Fp8E4M3::from_f32(s * SCALE_FACTOR))
            .collect();

        let mut asymmetric_mask = 0u16;
        let mut zero_points = Vec::new();
        for (i, (&sym, &zp)) in params
            .symmetric
            .iter()
            .zip(params.zero_points.iter())
            .enumerate()
        {
            if !sym {
                asymmetric_mask |= 1 << i;
                zero_points.push(Fp8E5M2::from_f32(zp));
            }
        }

        CompactGroupMeta {
            scales,
            zero_points,
            asymmetric_mask,
            group_size: params.group_size,
        }
    }

    /// Total metadata bits.
    pub fn total_bits(&self) -> usize {
        let scale_bits = self.scales.len() * 8;
        let zp_bits = self.zero_points.len() * 8;
        let mask_bits = 16; // u16 bitmask
        scale_bits + zp_bits + mask_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_e4m3_roundtrip() {
        // E4M3 range: ~2^-9 to 448. Skip values below minimum representable.
        let test_values = [
            0.0, 1.0, -1.0, 0.5, 2.0, 0.125, 100.0, 0.015625, -42.0, 448.0,
        ];
        for &v in &test_values {
            let fp8 = Fp8E4M3::from_f32(v);
            let back = fp8.to_f32();
            if v == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                let rel_err = (v - back).abs() / v.abs();
                // E4M3 has 3 mantissa bits -> ~12.5% relative error
                assert!(
                    rel_err < 0.2,
                    "FP8 E4M3 roundtrip error too high for {}: got {}, rel_err={:.3}",
                    v,
                    back,
                    rel_err
                );
            }
        }
    }

    #[test]
    fn test_fp8_e5m2_roundtrip() {
        let test_values = [0.0, 1.0, -1.0, 0.5, 2.0, 0.125, 1000.0, -256.0];
        for &v in &test_values {
            let fp8 = Fp8E5M2::from_f32(v);
            let back = fp8.to_f32();
            if v == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                let rel_err = (v - back).abs() / v.abs();
                // E5M2 has 2 mantissa bits -> ~25% relative error
                assert!(
                    rel_err < 0.35,
                    "FP8 E5M2 roundtrip error too high for {}: got {}, rel_err={:.3}",
                    v,
                    back,
                    rel_err
                );
            }
        }
    }

    #[test]
    fn test_fp8_scale_range() {
        // Our group scales are typically 0.01-0.5 (post-rotation unit-norm vectors).
        // With SCALE_FACTOR=16, these become 0.16-8.0 in FP8, which fits E4M3.
        let scales = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5];
        for &s in &scales {
            let fp8 = Fp8E4M3::from_f32(s * SCALE_FACTOR);
            let back = fp8.to_f32() / SCALE_FACTOR;
            let rel_err = (s - back).abs() / s;
            println!(
                "Scale {:.3} -> *16 -> FP8 {:02X} -> /16 -> {:.4} (err {:.1}%)",
                s,
                fp8.0,
                back,
                rel_err * 100.0
            );
            assert!(
                rel_err < 0.15,
                "Scale {:.3} FP8 error too high: {:.3}",
                s,
                rel_err
            );
        }
    }

    #[test]
    fn test_compact_meta_size() {
        // d=128, group_size=16: 8 groups
        // All symmetric: 8 * 8 bits + 16 bits mask = 80 bits
        // 2 asymmetric:  8 * 8 bits + 2 * 8 bits + 16 = 96 bits
        // vs f32: 8 * 64 = 512 bits

        use super::super::grouping::GroupQuantParams;
        let params = GroupQuantParams {
            scales: vec![0.1; 8],
            zero_points: vec![0.0; 8],
            symmetric: vec![true; 8],
            group_size: 16,
            n_groups: 8,
        };
        let compact = CompactGroupMeta::from_params(&params);
        assert_eq!(compact.total_bits(), 8 * 8 + 16); // 80 bits
        println!(
            "All symmetric: {} bits (vs f32: {} bits, {:.1}x savings)",
            compact.total_bits(),
            512,
            512.0 / compact.total_bits() as f64
        );
    }
}
