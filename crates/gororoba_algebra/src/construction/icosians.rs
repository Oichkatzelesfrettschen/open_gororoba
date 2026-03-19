//! Quaternionic Icosians and the E8 / Quasicrystal Bridge.
//!
//! Implements the 120 unit icosians, which form the binary icosahedral group.
//! These are foundational to the quaternionic construction of the E8 lattice
//! and the projection of E8 to 4D/3D quasicrystals (as detailed by Baez).
//!
//! # References
//! - Baez: "From the Icosahedron to E8"
//! - Conway & Smith: "On Quaternions and Octonions"

use crate::physics::quat_rotation::{Quaternion, quat_multiply};

/// The golden ratio phi
const PHI: f64 = 1.618033988749895;
const INV_PHI: f64 = 0.6180339887498949; // 1 / phi

/// Generate the 120 unit icosians.
///
/// The icosians consist of:
/// - 8 permutations of (+/-1, 0, 0, 0)
/// - 16 permutations of (+/-0.5, +/-0.5, +/-0.5, +/-0.5)
/// - 96 even permutations of (+/-phi/2, +/-1/2, +/-1/(2*phi), 0)
pub fn generate_icosians() -> Vec<Quaternion> {
    let mut icosians = Vec::with_capacity(120);

    // 1) 8 elements: permutations of (+/-1, 0, 0, 0)
    for i in 0..4 {
        for &sign in &[-1.0, 1.0] {
            let mut q = [0.0; 4];
            q[i] = sign;
            icosians.push(q);
        }
    }

    // 2) 16 elements: (+/-0.5, +/-0.5, +/-0.5, +/-0.5)
    for bits in 0..16 {
        let mut q = [0.0; 4];
        for (i, component) in q.iter_mut().enumerate() {
            *component = if (bits >> i) & 1 == 1 { 0.5 } else { -0.5 };
        }
        icosians.push(q);
    }

    // 3) 96 elements: even permutations of (+/-phi/2, +/-1/2, +/-1/(2*phi), 0)
    let vals = [PHI / 2.0, 0.5, INV_PHI / 2.0, 0.0];
    
    // Generate all 24 permutations of indices [0, 1, 2, 3]
    // Filter for even permutations
    let mut perms = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            if j == i { continue; }
            for k in 0..4 {
                if k == i || k == j { continue; }
                let l = 6 - i - j - k; // sum of 0,1,2,3 is 6
                
                // Check if even permutation
                let mut inversions = 0;
                let arr = [i, j, k, l];
                for a in 0..4 {
                    for b in (a+1)..4 {
                        if arr[a] > arr[b] { inversions += 1; }
                    }
                }
                
                if inversions % 2 == 0 {
                    perms.push(arr);
                }
            }
        }
    }
    
    // For each even permutation, apply all 16 sign combinations to the 4 slots
    // Wait, the 0 slot doesn't need sign combinations (since +0 = -0),
    // but applying all 16 combinations gives duplicates if we don't filter.
    // Actually, there are 3 non-zero values, so 2^3 = 8 sign combinations per permutation.
    // 12 even permutations * 8 sign combos = 96 elements.
    for perm in perms {
        for sign_bits in 0..8 {
            let mut q = [0.0; 4];
            let mut sign_idx = 0;
            for slot in 0..4 {
                let v = vals[perm[slot]];
                if v > 1e-10 {
                    q[slot] = if (sign_bits >> sign_idx) & 1 == 1 { -v } else { v };
                    sign_idx += 1;
                } else {
                    q[slot] = 0.0;
                }
            }
            icosians.push(q);
        }
    }

    icosians
}

/// Verify that the 120 icosians form a closed group under quaternion multiplication.
pub fn verify_icosian_group_closure() -> bool {
    let icosians = generate_icosians();
    if icosians.len() != 120 {
        return false;
    }
    
    // Test a randomized subset to keep tests fast, or all if feasible.
    // 120 x 120 = 14,400 pairs, we can test all of them quickly.
    for i in 0..120 {
        for j in 0..120 {
            let p = quat_multiply(&icosians[i], &icosians[j]);
            
            // Check if p is in the set of icosians
            let mut found = false;
            for candidate in icosians.iter().take(120) {
                let mut diff = 0.0;
                for idx in 0..4 {
                    diff += (p[idx] - candidate[idx]).powi(2);
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
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icosian_count() {
        let ico = generate_icosians();
        assert_eq!(ico.len(), 120);
        
        for q in ico {
            let norm = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "Icosian is not unit length");
        }
    }

    #[test]
    fn test_icosian_closure() {
        assert!(verify_icosian_group_closure());
    }
}
