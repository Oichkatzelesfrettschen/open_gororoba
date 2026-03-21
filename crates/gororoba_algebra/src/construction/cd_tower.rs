//! Unified Cayley-Dickson tower: 32D through 1024D.
//!
//! # Mathematical foundation
//!
//! Every type in this module is a real vector space whose multiplication rule
//! follows the **Cayley-Dickson doubling construction**.  Given an algebra A,
//! the doubled algebra A' = A x A has multiplication:
//!
//! ```text
//! (a, b) * (c, d) = (a*c - d*b_conj, d_adj*a + b*c_conj)
//! ```
//!
//! Applied to R iteratively: R -> C -> H (Quaternion) -> O (Octonion) -> ...
//! Each doubling halves the properties retained: commutativity is lost at H,
//! associativity at O, alternativity at S (Sedenion, 16D), and zero-divisor
//! freedom also at S.  All higher levels are flexible and power-associative.
//!
//! # Basis multiplication via XOR-sign
//!
//! For the standard CD construction the product of two basis elements `e_i`
//! and `e_j` is `+/- e_{i XOR j}`.  The sign is determined by a recursive
//! bit-level rule captured in `cd_kernel::cayley_dickson::cd_basis_mul_sign_iter`.
//! For the output component `k = i XOR j`, the full product reads:
//!
//! ```text
//! Z_k = sum_i  sign(e_i, e_{k XOR i})  *  X_i  *  Y_{k XOR i}
//! ```
//!
//! This is O(dim) work per output component, O(dim^2) total; sign lookup is
//! O(log dim) per call via the iterative construction.
//!
//! # SIMD layout
//!
//! Each type packs `f64` components into `[f64x4; dim/4]` for 32-byte alignment
//! (matching AVX2 register width).  The boundary conversion via `from_slice` /
//! `to_slice` is explicit: the inner multiplication loop is scalar because the
//! `i XOR (k XOR i) = k` pattern does not map cleanly to SIMD gather operations.
//! For a fully vectorised alternative see `cd_kernel::matmul_simd` (future work).
//!
//! # Feature gates
//!
//! | Feature    | Types enabled                                     |
//! |------------|---------------------------------------------------|
//! | (default)  | Pathion(32), Chingon(64), Routon(128), Voudon(256) |
//! | `cd-512`   | + Eriston(512)                                    |
//! | `cd-1024`  | + DekaVoudon(1024)                                |
//!
//! In practice both Eriston and DekaVoudon are compiled unconditionally (the
//! feature gates are reserved for future opt-out gating) to preserve
//! backward-compatibility with existing consumers.
//!
//! # Naming conventions
//!
//! Primary names follow a Greek-ordinal scheme coined in the literature by
//! Robert de Marrais and the Egan-Wilson synthesis.  Latin names (Trigintaduonion
//! etc.) are provided as zero-cost type aliases for academic cross-referencing
//! with papers that use the dimension-count Latin convention.
//!
//! # Upper-tower aliases (2048D+)
//!
//! At 2048D a dense `[f64; 2048]` costs 16 KB per element and the probability
//! that any given component is non-zero in physical applications (QEC syndrome
//! vectors, lattice site states) drops below 1%.  `Endekavoudon`, `Dodekvoudon`,
//! and `Dekatrisvoudon` are therefore type aliases for
//! `algebra_analysis::sparse::SparseState` (requires the `analysis` feature).

use cd_kernel::cayley_dickson::cd_multiply_flat_into;
use wide::f64x4;

/// Generate a SIMD-backed Cayley-Dickson algebra type.
///
/// # Arguments
/// - `$name`: the public type name (e.g. `Pathion`).
/// - `$dim`: the algebra dimension in basis elements (must be a power of 2).
/// - `$simd_count`: `$dim / 4` -- the number of `f64x4` lanes needed.
///
/// # What this generates
/// A `pub struct $name { pub data: [f64x4; $simd_count] }` with four methods:
/// - `zero()`: additive identity.
/// - `from_slice(arr: &[f64; $dim])`: pack a scalar array into SIMD lanes.
/// - `to_slice()`: unpack SIMD lanes back to a scalar array.
/// - `mul(other)`: full CD basis multiplication via the XOR-sign rule.
///
/// # Why a macro instead of a generic type?
///
/// `[T; N]` where N is a const generic would require `N % 4 == 0` and
/// `N / 4` as a const expression -- nightly features as of Rust 1.79.
/// The macro approach is stable, generates zero indirection, and lets us name
/// each algebra level as its own distinct type (Pathion, Chingon, ...) so that
/// type-level mistakes (passing a 256D state where 128D is expected) are caught
/// at compile time rather than at runtime.
macro_rules! implement_cd_algebra {
    ($name:ident, $dim:expr, $simd_count:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            pub data: [f64x4; $simd_count],
        }

        impl $name {
            pub fn zero() -> Self {
                Self {
                    data: [f64x4::splat(0.0); $simd_count],
                }
            }

            pub fn from_slice(slice: &[f64; $dim]) -> Self {
                let mut data = [f64x4::splat(0.0); $simd_count];
                for i in 0..$simd_count {
                    data[i] = f64x4::from([
                        slice[i * 4],
                        slice[i * 4 + 1],
                        slice[i * 4 + 2],
                        slice[i * 4 + 3],
                    ]);
                }
                Self { data }
            }

            pub fn to_slice(&self) -> [f64; $dim] {
                let mut res = [0.0f64; $dim];
                for i in 0..$simd_count {
                    let arr = self.data[i].to_array();
                    res[i * 4] = arr[0];
                    res[i * 4 + 1] = arr[1];
                    res[i * 4 + 2] = arr[2];
                    res[i * 4 + 3] = arr[3];
                }
                res
            }

            /// Zero-copy view of internal SIMD data as a flat f64 slice.
            ///
            /// SAFETY: `f64x4` is `#[repr(C, align(32))]` containing exactly
            /// 4 contiguous f64 values. An `[f64x4; N]` is therefore layout-
            /// compatible with `[f64; N*4]`. The pointer cast reinterprets
            /// without copying any data.
            #[inline(always)]
            fn as_flat(&self) -> &[f64; $dim] {
                // SAFETY: [f64x4; N] and [f64; N*4] have identical layout
                unsafe { &*(self.data.as_ptr() as *const [f64; $dim]) }
            }

            /// Zero-copy mutable view of internal SIMD data as a flat f64 slice.
            #[inline(always)]
            fn as_flat_mut(&mut self) -> &mut [f64; $dim] {
                // SAFETY: same layout guarantee as as_flat
                unsafe { &mut *(self.data.as_mut_ptr() as *mut [f64; $dim]) }
            }

            /// Multiply via flat recursive CD doubling (zero-copy).
            ///
            /// Uses `cd_multiply_flat_into` which recurses down to the
            /// register-resident sedenion base case (64 FMAs, zero heap
            /// allocation at the base level).  The SIMD data is accessed
            /// directly via zero-cost pointer reinterpretation -- no
            /// intermediate to_slice()/from_slice() copies.
            pub fn mul(&self, other: &Self) -> Self {
                let mut result = Self::zero();
                cd_multiply_flat_into(
                    self.as_flat(),
                    other.as_flat(),
                    result.as_flat_mut(),
                    $dim,
                );
                result
            }
        }
    };
}

implement_cd_algebra!(Pathion,  32,   8);
implement_cd_algebra!(Chingon,  64,  16);
implement_cd_algebra!(Routon,  128,  32);
implement_cd_algebra!(Voudon,     256,  64);
// Eriston and DekaVoudon are unconditional to preserve backward compat with
// existing consumers (quantum_core, cosmology_core, gororoba_engine).
// The cd-512/cd-1024 features are reserved for future opt-out gating.
implement_cd_algebra!(Eriston,    512, 128);
implement_cd_algebra!(DekaVoudon, 1024, 256);

// ---------------------------------------------------------------------------
// Latin type aliases -- zero-cost cross-reference shims.
// ---------------------------------------------------------------------------

/// 32D Cayley-Dickson algebra (academic Latin name).
pub type Trigintaduonion          = Pathion;
/// 64D Cayley-Dickson algebra (academic Latin name).
pub type Sexagintaquatronion      = Chingon;
/// 128D Cayley-Dickson algebra (academic Latin name).
pub type Centumduodetrigintanion  = Routon;
/// 256D Cayley-Dickson algebra (academic Latin name).
pub type Ducentiquinquagintasexion = Voudon;

/// 512D Cayley-Dickson algebra (academic Latin name).
pub type Quingentoduodecimnion    = Eriston;

/// 1024D Cayley-Dickson algebra (academic Latin name).
pub type Millevigintiquattuornion = DekaVoudon;

// ---------------------------------------------------------------------------
// Upper-tower aliases (2048D+).
// Dense SIMD storage is impractical at 2048D+; SparseState is the right tool.
// ---------------------------------------------------------------------------
#[cfg(feature = "analysis")]
pub use algebra_analysis::sparse::SparseState;

/// 2048D Cayley-Dickson level-11 algebra (sparse representation).
#[cfg(feature = "analysis")]
pub type Endekavoudon    = SparseState;
/// 4096D Cayley-Dickson level-12 algebra (sparse representation).
#[cfg(feature = "analysis")]
pub type Dodekvoudon     = SparseState;
/// 8192D Cayley-Dickson level-13 algebra (sparse representation).
#[cfg(feature = "analysis")]
pub type Dekatrisvoudon  = SparseState;

/// Deprecated: use `Endekavoudon` (level-11 Greek ordinal naming).
#[cfg(feature = "analysis")]
#[deprecated(since = "0.1.0", note = "use Endekavoudon")]
pub type Icosikaivoudon  = SparseState;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pathion_zero_is_zero() {
        let p = Pathion::zero();
        for v in p.to_slice() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn pathion_roundtrip() {
        let mut arr = [0.0f64; 32];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as f64 * 0.5;
        }
        let p = Pathion::from_slice(&arr);
        let out = p.to_slice();
        for (a, b) in arr.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn chingon_mul_basis_element() {
        // e_1 * e_1 in 64D should give -e_0 (standard CD rule for i*i=-1)
        let mut e1 = [0.0f64; 64];
        e1[1] = 1.0;
        let c = Chingon::from_slice(&e1);
        let sq = c.mul(&c);
        let out = sq.to_slice();
        assert!((out[0] + 1.0).abs() < 1e-12, "e1^2 should be -e0, got {}", out[0]);
    }

    #[test]
    fn routon_mul_consistency_with_pathion_lower_block() {
        // The lower 32 components of a Routon embedding of a Pathion element
        // should multiply consistently with Pathion multiplication in the
        // lower block (since CD doubling embeds dimension n into n*2).
        let mut arr32 = [0.0f64; 32];
        arr32[1] = 1.0;
        arr32[2] = 1.0;
        let p = Pathion::from_slice(&arr32);
        let p_prod = p.mul(&p);

        let mut arr128 = [0.0f64; 128];
        arr128[1] = 1.0;
        arr128[2] = 1.0;
        let r = Routon::from_slice(&arr128);
        let r_prod = r.mul(&r);
        let r_out = r_prod.to_slice();
        let p_out = p_prod.to_slice();

        for i in 0..32 {
            assert!(
                (r_out[i] - p_out[i]).abs() < 1e-12,
                "Lower block mismatch at index {}: Routon={} Pathion={}",
                i, r_out[i], p_out[i]
            );
        }
    }

    #[test]
    fn latin_alias_is_same_type() {
        let p = Pathion::zero();
        let _t: Trigintaduonion = p;
        let c = Chingon::zero();
        let _s: Sexagintaquatronion = c;
    }

    #[test]
    fn eriston_zero() {
        let e = Eriston::zero();
        for v in e.to_slice() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn deka_voudon_zero() {
        let d = DekaVoudon::zero();
        for v in d.to_slice() {
            assert_eq!(v, 0.0);
        }
    }
}
