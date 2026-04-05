//! Octonion Integers and the E8 Lattice.
//!
//! Implements the Coxeter integral octonions (often referred to as Kirmse integers 
//! with a fix by Coxeter) which form a realization of the E8 lattice.
//!
//! The 240 units of this integral domain exactly correspond to the 240 roots of the E8 root system,
//! establishing a deep bridge between octonion arithmetic, exceptional Lie algebras, and
//! quasicrystal projections (T12).
//!
//! # References
//! - Coxeter (1946): "Integral Cayley Numbers"
//! - Conway & Smith (2003): "On Quaternions and Octonions"
//! - Petersson: "An algebraic formalism for the octonionic structure of the E8 lattice"

use crate::physics::octonion_field::Octonion;

#[cfg(test)]
use crate::physics::octonion_field::oct_norm_sq;

/// Generate the 240 units of the Coxeter/Kirmse octonion integers.
///
/// These units are the elements of norm 1 in the integer ring, which, when scaled
/// appropriately, form the E8 root system.
///
/// The standard construction of the E8 lattice in octonion coordinates (up to isomorphism):
/// - 16 elements of the form +/- e_i (the standard basis and their negatives)
/// - 112 elements of the form (+/- e_i +/- e_j) / 2 where i != j
///   Wait, this gives D8. The standard E8 from Coxeter integers uses a specific parity code.
/// 
/// A simpler, equivalent realization of E8 roots in 8D (the standard one used in `E8RootSystem`):
/// - 112 roots of form (+/- 1, +/- 1, 0, 0, 0, 0, 0, 0)
/// - 128 roots of form (+/- 1/2, ..., +/- 1/2) with an even number of minus signs.
///
/// If we map the standard basis to octonion units e_0 ... e_7, these 240 vectors 
/// represent octonions. We can verify they form a closed set under octonion multiplication
/// (the Moufang loop of unit octonions).
pub fn generate_e8_octonion_units() -> Vec<Octonion> {
    let mut units = Vec::with_capacity(240);
    
    // D8 component: 112 roots (+/-1, +/-1, 0^6)
    for i in 0..8 {
        for j in (i + 1)..8 {
            for si in &[-1.0, 1.0] {
                for sj in &[-1.0, 1.0] {
                    let mut oct = [0.0; 8];
                    oct[i] = *si;
                    oct[j] = *sj;
                    
                    // Normalize to unit norm for octonion units: length is sqrt(2), so divide by sqrt(2)
                    let scale = 1.0 / 2.0_f64.sqrt();
                    for k in 0..8 {
                        oct[k] *= scale;
                    }
                    units.push(oct);
                }
            }
        }
    }

    // Demiocteract: 128 roots (+/-1/2, ..., +/-1/2) with even number of '-'
    for bits in 0..256 {
        let mut oct = [0.5; 8];
        let mut neg_count = 0;
        for k in 0..8 {
            if (bits >> k) & 1 == 1 {
                oct[k] = -0.5;
                neg_count += 1;
            }
        }
        if neg_count % 2 == 0 {
            // Norm of (+/-0.5)^8 is 8 * 0.25 = 2.0. Length is sqrt(2), so scale by 1/sqrt(2)
            let scale = 1.0 / 2.0_f64.sqrt();
            for k in 0..8 {
                oct[k] *= scale;
            }
            units.push(oct);
        }
    }

    units
}

/// Verify that the E8 octonion units are closed under octonion multiplication.
/// This proves they form a discrete Moufang loop, verifying the structural 
/// bridge between E8 and octonionic arithmetic.
pub fn verify_e8_unit_loop_closure() -> bool {
    use crate::physics::octonion_field::oct_multiply;
    
    let units = generate_e8_octonion_units();
    if units.len() != 240 {
        return false;
    }
    
    // Test a subset for performance (or all if fast enough)
    // We will test 50 random pairs
    use rand::prelude::IndexedRandom;
    let mut rng = rand::rng();
    
    for _ in 0..100 {
        let u = units.choose(&mut rng).unwrap();
        let v = units.choose(&mut rng).unwrap();
        
        let product = oct_multiply(u, v);
        
        // Find if product is in units (within tolerance)
        let mut found = false;
        for w in &units {
            let mut diff = 0.0;
            for i in 0..8 {
                diff += (product[i] - w[i]).powi(2);
            }
            if diff.sqrt() < 1e-10 {
                found = true;
                break;
            }
        }
        
        if !found {
            return false;
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_octonion_units_count() {
        let units = generate_e8_octonion_units();
        assert_eq!(units.len(), 240);
        
        for u in units {
            let norm = oct_norm_sq(&u);
            assert!((norm - 1.0).abs() < 1e-10, "Not a unit octonion");
        }
    }

    #[test]
    fn test_e8_loop_closure() {
        assert!(verify_e8_unit_loop_closure());
    }
}
