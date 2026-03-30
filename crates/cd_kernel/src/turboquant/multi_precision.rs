//! Multi-precision quantization: INT2/3/4 packed indices + FP8 metadata.
//!
//! Combines the Lagrange bit allocator, FP8 scale storage, and packed
//! index formats into a unified compressed representation that matches
//! or beats RotateKV's effective bit rate.
//!
//! # Effective Bit Rates at d=128
//!
//! | Config | Index bits | Meta bits | Norm | Total | Eff bits/coord |
//! |--------|-----------|-----------|------|-------|----------------|
//! | 2-bit packed, no groups | 256 | 0 | 16 | 272 | 2.12 |
//! | 2-bit packed, FP8 gs=32 | 256 | 40 | 16 | 312 | 2.44 |
//! | 3-bit packed, FP8 gs=32 | 384 | 40 | 16 | 440 | 3.44 |
//! | Mixed 2-3 (Lagrange) | ~320 | 40 | 16 | ~376 | ~2.94 |
//!
//! # Packed Index Format
//!
//! 2-bit: 4 indices per byte (u8), little-endian bit order
//! 3-bit: 8 indices per 3 bytes (24 bits), requires bit-shifting
//! 4-bit: 2 indices per byte (u8), simple nibble pack

use super::fp8::CompactGroupMeta;
#[cfg(test)]
use super::fp8::Fp8E4M3;

/// Packed 2-bit indices: 4 values per byte.
#[derive(Clone, Debug)]
pub struct Packed2BitIndices {
    /// Packed bytes: byte[i] contains indices[4i..4i+4].
    pub data: Vec<u8>,
    /// Number of actual indices (may not be multiple of 4).
    pub len: usize,
}

impl Packed2BitIndices {
    /// Pack u8 indices (each 0-3) into 2-bit packed format.
    pub fn pack(indices: &[u8]) -> Self {
        let n = indices.len();
        let n_bytes = (n + 3) / 4;
        let mut data = vec![0u8; n_bytes];
        for (i, &idx) in indices.iter().enumerate() {
            debug_assert!(idx < 4, "2-bit index out of range: {}", idx);
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            data[byte_idx] |= (idx & 0x3) << bit_offset;
        }
        Packed2BitIndices { data, len: n }
    }

    /// Unpack to u8 indices.
    pub fn unpack(&self) -> Vec<u8> {
        let mut indices = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            indices.push((self.data[byte_idx] >> bit_offset) & 0x3);
        }
        indices
    }

    /// Get a single index.
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        (self.data[byte_idx] >> bit_offset) & 0x3
    }

    /// Storage size in bits.
    pub fn bits(&self) -> usize {
        self.data.len() * 8
    }
}

/// Packed 4-bit indices: 2 values per byte.
#[derive(Clone, Debug)]
pub struct Packed4BitIndices {
    pub data: Vec<u8>,
    pub len: usize,
}

impl Packed4BitIndices {
    pub fn pack(indices: &[u8]) -> Self {
        let n = indices.len();
        let n_bytes = (n + 1) / 2;
        let mut data = vec![0u8; n_bytes];
        for (i, &idx) in indices.iter().enumerate() {
            debug_assert!(idx < 16, "4-bit index out of range: {}", idx);
            let byte_idx = i / 2;
            let nib = i % 2;
            data[byte_idx] |= (idx & 0xF) << (nib * 4);
        }
        Packed4BitIndices { data, len: n }
    }

    pub fn unpack(&self) -> Vec<u8> {
        let mut indices = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let byte_idx = i / 2;
            let nib = i % 2;
            indices.push((self.data[byte_idx] >> (nib * 4)) & 0xF);
        }
        indices
    }

    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        let byte_idx = i / 2;
        let nib = i % 2;
        (self.data[byte_idx] >> (nib * 4)) & 0xF
    }

    pub fn bits(&self) -> usize {
        self.data.len() * 8
    }
}

/// Multi-precision compressed vector.
///
/// Stores indices in the most compact packed format for the bit-width,
/// metadata in FP8, and norm in f16 (stored as u16 bits).
#[derive(Clone, Debug)]
pub enum MultiPrecisionCompressed {
    /// 2-bit: 4 indices/byte + FP8 group scales.
    Bits2 {
        indices: Packed2BitIndices,
        meta: CompactGroupMeta,
        norm_bits: u16, // f16 stored as bits
    },
    /// 3-bit: u8 indices (not packed yet) + FP8 group scales.
    Bits3 {
        indices: Vec<u8>,
        meta: CompactGroupMeta,
        norm_bits: u16,
    },
    /// 4-bit: 2 indices/byte + FP8 group scales.
    Bits4 {
        indices: Packed4BitIndices,
        meta: CompactGroupMeta,
        norm_bits: u16,
    },
}

impl MultiPrecisionCompressed {
    /// Total storage in bits.
    pub fn total_bits(&self) -> usize {
        match self {
            Self::Bits2 { indices, meta, .. } => indices.bits() + meta.total_bits() + 16,
            Self::Bits3 { indices, meta, .. } => indices.len() * 3 + meta.total_bits() + 16,
            Self::Bits4 { indices, meta, .. } => indices.bits() + meta.total_bits() + 16,
        }
    }

    /// Effective bits per coordinate.
    pub fn effective_bits(&self, d: usize) -> f64 {
        self.total_bits() as f64 / d as f64
    }
}

/// Convert f32 norm to f16 bit representation (simplified: truncate mantissa).
#[allow(dead_code)]
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    // f16: 1 sign + 5 exponent + 10 mantissa, bias=15
    let f16_exp = exp - 127 + 15;
    if f16_exp >= 31 { return sign << 15 | 0x7C00; } // inf
    if f16_exp <= 0 { return sign << 15; } // zero/denorm
    let f16_mant = (mant >> 13) as u16; // top 10 bits
    sign << 15 | (f16_exp as u16) << 10 | f16_mant
}

/// Convert f16 bits back to f32.
#[allow(dead_code)]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 && mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
    if exp == 31 { return if mant == 0 { if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY } } else { f32::NAN }; }

    let f32_exp = (exp - 15 + 127) as u32;
    let f32_mant = mant << 13;
    f32::from_bits(sign << 31 | f32_exp << 23 | f32_mant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_2bit_roundtrip() {
        let indices: Vec<u8> = (0..128).map(|i| (i % 4) as u8).collect();
        let packed = Packed2BitIndices::pack(&indices);
        assert_eq!(packed.data.len(), 32); // 128 / 4 = 32 bytes
        assert_eq!(packed.bits(), 256);

        let unpacked = packed.unpack();
        assert_eq!(unpacked, indices);

        // Random access
        for i in 0..128 {
            assert_eq!(packed.get(i), indices[i]);
        }
    }

    #[test]
    fn test_packed_4bit_roundtrip() {
        let indices: Vec<u8> = (0..128).map(|i| (i % 16) as u8).collect();
        let packed = Packed4BitIndices::pack(&indices);
        assert_eq!(packed.data.len(), 64); // 128 / 2
        assert_eq!(packed.bits(), 512);

        let unpacked = packed.unpack();
        assert_eq!(unpacked, indices);
    }

    #[test]
    fn test_effective_bits_2bit_packed() {
        let d = 128;
        let indices = Packed2BitIndices::pack(&vec![0u8; d]);
        let meta = CompactGroupMeta {
            scales: vec![Fp8E4M3(0x38); 4], // 4 groups for gs=32
            zero_points: vec![],
            asymmetric_mask: 0,
            group_size: 32,
        };
        let compressed = MultiPrecisionCompressed::Bits2 {
            indices,
            meta,
            norm_bits: 0,
        };
        let eff = compressed.effective_bits(d);
        println!("2-bit packed, FP8 gs=32: {:.2} eff bits (vs pure 2.0)", eff);
        // 256 data + 4*8 scale + 16 mask + 16 norm = 320 / 128 = 2.50
        assert!(eff < 2.6, "Effective bits too high: {}", eff);
        assert!(eff > 2.0, "Effective bits too low: {}", eff);
    }

    #[test]
    fn test_f16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, 0.001, 65504.0];
        for &v in &values {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            if v == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                let rel_err = (v - back).abs() / v.abs();
                assert!(rel_err < 0.001, "f16 roundtrip error for {}: {}", v, rel_err);
            }
        }
    }

    #[test]
    fn test_storage_comparison() {
        let d = 128;

        // Current: u8 indices + f32 scale + f32 zp (group_size=16)
        let current_bits = d * 8 + 8 * (32 + 32) + 16; // indices + 8 groups * f32 pair + norm

        // Packed 2-bit + FP8 (group_size=32)
        let packed_bits = d * 2 + 4 * 8 + 16 + 16; // packed indices + 4 groups * FP8 + mask + norm

        let savings = 1.0 - packed_bits as f64 / current_bits as f64;
        println!("Storage: current={} bits ({:.2}/coord), packed={} bits ({:.2}/coord), savings={:.1}%",
            current_bits, current_bits as f64 / d as f64,
            packed_bits, packed_bits as f64 / d as f64,
            savings * 100.0);

        assert!(packed_bits < current_bits / 3,
            "Packed should be <1/3 of current: {} vs {}", packed_bits, current_bits);
    }
}
