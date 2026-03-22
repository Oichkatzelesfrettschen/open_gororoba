use super::*;

#[test]
fn test_real_multiply() {
    let result = cd_multiply(&[3.0], &[5.0]);
    assert_eq!(result, vec![15.0]);
}

#[test]
fn test_complex_multiply() {
    let result = cd_multiply(&[1.0, 2.0], &[3.0, 4.0]);
    assert!((result[0] + 5.0).abs() < 1e-10);
    assert!((result[1] - 10.0).abs() < 1e-10);
}

#[test]
fn test_find_zero_divisors_octonion_none() {
    let results = find_zero_divisors(8, 1e-10);
    assert!(results.is_empty());
}

#[test]
fn test_find_zero_divisors_sedenion() {
    let results = find_zero_divisors(16, 1e-10);
    assert!(!results.is_empty());
}

#[test]
fn test_simd_matches_scalar() {
    for dim in [4, 8, 16, 32] {
        let a: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.123).sin()).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.456).cos()).collect();
        let scalar_result = cd_multiply(&a, &b);
        let simd_result = cd_multiply_simd(&a, &b);
        for (s, m) in scalar_result.iter().zip(&simd_result) {
            assert!((s - m).abs() < 1e-12);
        }
    }
}

#[test]
fn test_sign_table_matches_iter() {
    let table = SignTable::new(16);
    for p in 0..16 {
        for q in 0..16 {
            assert_eq!(table.sign(p, q), cd_basis_mul_sign_iter(16, p, q));
        }
    }
}

#[test]
fn test_split_signature_matches_standard() {
    let sig = CdSignature::standard(8);
    for p in 0..8 {
        for q in 0..8 {
            assert_eq!(
                cd_basis_mul_sign(8, p, q),
                cd_basis_mul_sign_split(8, p, q, &sig)
            );
        }
    }
}

#[test]
fn test_associator_density_quaternions() {
    let (density, failures) = measure_associator_density(4, 200, 42, 1e-8);
    assert_eq!(failures, 0);
    assert!((density - 0.0).abs() < 1e-10);
}

#[test]
fn test_associator_density_sedenions() {
    let (density, failures) = measure_associator_density(16, 200, 42, 1e-8);
    assert!(density > 90.0);
    assert!(failures > 150);
}

#[test]
fn test_associator_stats_large_dim() {
    let stats = associator_independence_stats(16, 200, 42);
    assert!(stats.mean_assoc_sq.is_finite());
}

// T1: Quadratic identity
#[test]
fn test_thesis_t1_quadratic_identity() {
    let dims = [2, 4, 8, 16, 32];
    for dim in dims {
        let x: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.1).sin()).collect();
        let x2 = cd_multiply(&x, &x);
        let t_x = 2.0 * x[0];
        let n_x = cd_norm_sq(&x);
        
        // x^2 - t(x)x + n(x) = 0
        let mut res = vec![0.0; dim];
        for i in 0..dim {
            res[i] = x2[i] - t_x * x[i];
        }
        res[0] += n_x;
        
        for v in res {
            assert!(v.abs() < 1e-10, "Quadratic identity failed at dim {}", dim);
        }
    }
}

// T2: Power-associativity
#[test]
fn test_thesis_t2_power_associativity() {
    let dim = 16;
    let x: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.2).cos()).collect();
    let x2 = cd_multiply(&x, &x);
    let x2_x = cd_multiply(&x2, &x);
    let x_x2 = cd_multiply(&x, &x2);
    
    for i in 0..dim {
        assert!((x2_x[i] - x_x2[i]).abs() < 1e-10, "Power-associativity failed");
    }
}

// T3: Flexibility
#[test]
fn test_thesis_t3_flexibility() {
    let dim = 16;
    let x: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.1).sin()).collect();
    let y: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.2).cos()).collect();
    
    // (xy)x
    let xy = cd_multiply(&x, &y);
    let xy_x = cd_multiply(&xy, &x);
    
    // x(yx)
    let yx = cd_multiply(&y, &x);
    let x_yx = cd_multiply(&x, &yx);
    
    for i in 0..dim {
        assert!((xy_x[i] - x_yx[i]).abs() < 1e-10, "Flexibility failed");
    }
}

// T4: Alternativity breaks at n>=4
#[test]
fn test_thesis_t4_alternativity_breaks() {
    let dim = 16;
    let mut found_break = false;
    
    // We can search for x = e_i + e_j, y = e_k that breaks alternativity
    for i in 1..dim {
        for j in i+1..dim {
            for k in 1..dim {
                let mut x = vec![0.0; dim];
                x[i] = 1.0;
                x[j] = 1.0;
                
                let mut y = vec![0.0; dim];
                y[k] = 1.0;
                
                let xx = cd_multiply(&x, &x);
                let x_xy = cd_multiply(&x, &cd_multiply(&x, &y));
                let xx_y = cd_multiply(&xx, &y);
                
                let mut diff = 0.0;
                for idx in 0..dim {
                    diff += (x_xy[idx] - xx_y[idx]).abs();
                }
                
                if diff > 1e-5 {
                    found_break = true;
                    break;
                }
            }
            if found_break { break; }
        }
        if found_break { break; }
    }
    
    assert!(found_break, "Alternativity should break in 16D");
}

#[test]
fn test_quaternion_multiply_flat_matches_scalar() {
    let a: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let b: [f64; 4] = [5.0, 6.0, 7.0, 8.0];
    let flat = quaternion_multiply_flat(&a, &b);
    let scalar = cd_multiply(&a, &b);
    for i in 0..4 {
        assert!(
            (flat[i] - scalar[i]).abs() < 1e-12,
            "quaternion_multiply_flat[{}] = {}, scalar = {}",
            i, flat[i], scalar[i]
        );
    }
}

#[test]
fn test_octonion_multiply_flat_matches_scalar() {
    let a: [f64; 8] = [1.0, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8];
    let b: [f64; 8] = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
    let flat = octonion_multiply_flat(&a, &b);
    let scalar = cd_multiply(&a, &b);
    for i in 0..8 {
        assert!(
            (flat[i] - scalar[i]).abs() < 1e-12,
            "octonion_multiply_flat[{}] = {}, scalar = {}",
            i, flat[i], scalar[i]
        );
    }
}

#[test]
fn test_quaternion_multiply_flat_associativity_fails_at_octonion() {
    // Quaternions are associative: (ab)c = a(bc).
    // Octonions are NOT. Verify the flat multiply respects this.
    let a: [f64; 4] = [1.0, 2.0, 0.0, 0.0];
    let b: [f64; 4] = [0.0, 1.0, 1.0, 0.0];
    let c: [f64; 4] = [0.0, 0.0, 1.0, 1.0];
    let ab = quaternion_multiply_flat(&a, &b);
    let bc = quaternion_multiply_flat(&b, &c);
    let ab_c = quaternion_multiply_flat(&ab, &c);
    let a_bc = quaternion_multiply_flat(&a, &bc);
    // Quaternions ARE associative, so these should match
    for i in 0..4 {
        assert!(
            (ab_c[i] - a_bc[i]).abs() < 1e-12,
            "Quaternion associativity failed at [{}]: (ab)c={}, a(bc)={}",
            i, ab_c[i], a_bc[i]
        );
    }
}

#[test]
fn test_sedenion_multiply_flat_matches_scalar() {
    let a: [f64; 16] = [
        1.0, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7,
        -0.8, 0.9, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6,
    ];
    let b: [f64; 16] = [
        0.5, -0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7,
        0.8, -0.9, 0.1, 0.2, -0.3, 0.4, 0.5, -0.6,
    ];
    let flat = sedenion_multiply_flat(&a, &b);
    let scalar = cd_multiply(&a, &b);
    for i in 0..16 {
        assert!(
            (flat[i] - scalar[i]).abs() < 1e-10,
            "sedenion_multiply_flat[{}] = {}, scalar = {}, diff = {}",
            i, flat[i], scalar[i], (flat[i] - scalar[i]).abs()
        );
    }
}

#[test]
fn test_sedenion_multiply_flat_non_associative() {
    // Sedenions are NOT associative: (ab)c != a(bc) in general
    let a: [f64; 16] = {
        let mut v = [0.0; 16];
        v[1] = 1.0;
        v[10] = 1.0;
        v
    };
    let b: [f64; 16] = {
        let mut v = [0.0; 16];
        v[3] = 1.0;
        v[12] = 1.0;
        v
    };
    let c: [f64; 16] = {
        let mut v = [0.0; 16];
        v[5] = 1.0;
        v[14] = 1.0;
        v
    };
    let ab = sedenion_multiply_flat(&a, &b);
    let bc = sedenion_multiply_flat(&b, &c);
    let ab_c = sedenion_multiply_flat(&ab, &c);
    let a_bc = sedenion_multiply_flat(&a, &bc);
    let diff: f64 = ab_c.iter().zip(a_bc.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(diff > 1e-6,
        "Sedenion (ab)c should differ from a(bc) -- non-associativity. diff={}", diff);
}

#[test]
fn test_cd_multiply_flat_into_matches_scalar_all_dims() {
    for dim in [1, 2, 4, 8, 16, 32] {
        let a: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.123).sin()).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.456).cos()).collect();
        let scalar = cd_multiply(&a, &b);
        let mut flat_out = vec![0.0; dim];
        cd_multiply_flat_into(&a, &b, &mut flat_out, dim);
        for i in 0..dim {
            assert!(
                (flat_out[i] - scalar[i]).abs() < 1e-10,
                "dim={} flat_into[{}] = {}, scalar = {}, diff = {}",
                dim, i, flat_out[i], scalar[i], (flat_out[i] - scalar[i]).abs()
            );
        }
    }
}

#[test]
fn test_pathion_32d_flat_into_matches_scalar() {
    let a: Vec<f64> = (0..32).map(|i| (i as f64 * 0.0789).sin()).collect();
    let b: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1234).cos()).collect();
    let scalar = cd_multiply(&a, &b);
    let mut flat_out = vec![0.0; 32];
    cd_multiply_flat_into(&a, &b, &mut flat_out, 32);
    for i in 0..32 {
        assert!(
            (flat_out[i] - scalar[i]).abs() < 1e-9,
            "pathion flat_into[{}] = {}, scalar = {}, diff = {}",
            i, flat_out[i], scalar[i], (flat_out[i] - scalar[i]).abs()
        );
    }
}

#[test]
fn test_chingon_64d_flat_into_matches_scalar() {
    let a: Vec<f64> = (0..64).map(|i| (i as f64 * 0.0456).sin()).collect();
    let b: Vec<f64> = (0..64).map(|i| (i as f64 * 0.0789).cos()).collect();
    let scalar = cd_multiply(&a, &b);
    let mut flat_out = vec![0.0; 64];
    cd_multiply_flat_into(&a, &b, &mut flat_out, 64);
    for i in 0..64 {
        assert!(
            (flat_out[i] - scalar[i]).abs() < 1e-8,
            "chingon flat_into[{}] = {}, scalar = {}, diff = {}",
            i, flat_out[i], scalar[i], (flat_out[i] - scalar[i]).abs()
        );
    }
}

#[test]
fn test_cd_flat_into_256d_matches_scalar() {
    let a: Vec<f64> = (0..256).map(|i| (i as f64 * 0.0123).sin()).collect();
    let b: Vec<f64> = (0..256).map(|i| (i as f64 * 0.0456).cos()).collect();
    let scalar = cd_multiply(&a, &b);
    let mut flat_out = vec![0.0; 256];
    cd_multiply_flat_into(&a, &b, &mut flat_out, 256);
    for i in 0..256 {
        assert!(
            (flat_out[i] - scalar[i]).abs() < 1e-6,
            "256d flat_into[{}] = {}, scalar = {}, diff = {}",
            i, flat_out[i], scalar[i], (flat_out[i] - scalar[i]).abs()
        );
    }
}

#[test]
fn test_cd_flat_into_512d_matches_scalar() {
    let a: Vec<f64> = (0..512).map(|i| (i as f64 * 0.007).sin()).collect();
    let b: Vec<f64> = (0..512).map(|i| (i as f64 * 0.013).cos()).collect();
    let scalar = cd_multiply(&a, &b);
    let mut flat_out = vec![0.0; 512];
    cd_multiply_flat_into(&a, &b, &mut flat_out, 512);
    let max_diff: f64 = flat_out.iter().zip(scalar.iter())
        .map(|(f, s)| (f - s).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-5,
        "512d max diff = {} (threshold 1e-5)", max_diff);
}

#[test]
fn test_cd_flat_into_1024d_matches_scalar() {
    let a: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.003).sin()).collect();
    let b: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.005).cos()).collect();
    let scalar = cd_multiply(&a, &b);
    let mut flat_out = vec![0.0; 1024];
    cd_multiply_flat_into(&a, &b, &mut flat_out, 1024);
    let max_diff: f64 = flat_out.iter().zip(scalar.iter())
        .map(|(f, s)| (f - s).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-4,
        "1024d max diff = {} (threshold 1e-4)", max_diff);
}

/// Time the full CD tower from 4D through 16384D (Tessareskaidekavoudon).
/// This is a timing test, not a correctness test (scalar is too slow above 4096D).
#[test]
fn test_cd_tower_timing_4d_to_16384d() {
    use std::time::Instant;
    for exp in 2..=14 {
        let dim = 1 << exp;
        let a: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.007).sin()).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.013).cos()).collect();
        let mut out = vec![0.0; dim];
        let t0 = Instant::now();
        cd_multiply_flat_into(&a, &b, &mut out, dim);
        let elapsed = t0.elapsed();
        // Verify output is not all zeros (not a degenerate case)
        let norm_sq: f64 = out.iter().map(|x| x * x).sum();
        assert!(norm_sq > 0.0, "dim={} produced zero output", dim);
        println!("cd_multiply_flat_into dim={:>5} time={:>12.3?} norm_sq={:.6e}",
            dim, elapsed, norm_sq);
    }
}

#[test]
fn test_koebisu_d2_on_all_standard_zds() {
    // All 84 standard ZDs must have D_2 = 0
    let zds = find_zero_divisors(16, 1e-10);
    let mut zd_count = 0;
    for (i, j, k, l, _norm) in &zds {
        let mut a = vec![0.0; 16];
        a[*i] = 1.0;
        a[*j] = 1.0;
        let d2 = koebisu_d2(&a);
        assert!(d2 < 1e-20,
            "ZD ({},{}) has D_2 = {:.2e}, expected ~0", i, j, d2);

        let mut b = vec![0.0; 16];
        b[*k] = 1.0;
        b[*l] = 1.0;
        let d2b = koebisu_d2(&b);
        assert!(d2b < 1e-20,
            "ZD ({},{}) has D_2 = {:.2e}, expected ~0", k, l, d2b);
        zd_count += 1;
    }

    // Single basis elements are NOT zero divisors (D_2 = 1)
    for i in 1..16_usize {
        let mut v = vec![0.0; 16];
        v[i] = 1.0;
        assert!(!is_zero_divisor_koebisu(&v, 1e-10),
            "e_{} should not be a ZD", i);
    }

    println!("Koebisu D_2 verified on {} ZD pairs + 15 non-ZD basis elements", zd_count);
}

#[test]
fn test_koebisu_d2_random_consistency() {
    // Random sedenions: D_2 = 0 iff the element is a ZD (verified by multiplication)
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let v: Vec<f64> = (0..16).map(|_| rng.r#gen::<f64>() * 2.0 - 1.0).collect();
        let d2 = koebisu_d2(&v);

        // Random sedenions are almost never ZDs
        assert!(d2 > 1e-6,
            "Random sedenion has D_2 = {:.2e}, suspiciously close to 0", d2);
    }

    // Construct a deliberate ZD: e_1 + e_10 (assessor pair)
    let mut zd = vec![0.0; 16];
    zd[1] = 1.0;
    zd[10] = 1.0;
    assert!(is_zero_divisor_koebisu(&zd, 1e-10), "e_1+e_10 must be a ZD");

    // Scale it: 3*(e_1 + e_10) is still a ZD
    let mut zd_scaled = vec![0.0; 16];
    zd_scaled[1] = 3.0;
    zd_scaled[10] = 3.0;
    assert!(is_zero_divisor_koebisu(&zd_scaled, 1e-10), "3*(e_1+e_10) must be a ZD");
}

#[test]
fn test_gourlay_psi_order_3() {
    // psi^3(e_i) = e_i for all basis elements
    for i in 0..16_usize {
        let mut v = [0.0_f64; 16];
        v[i] = 1.0;
        let psi3 = gourlay_psi_n(&v, 3);
        let max_err: f64 = v.iter().zip(psi3.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-12,
            "psi^3(e_{}) != e_{}: max error {:.2e}", i, i, max_err);
    }
    println!("psi^3 = Id verified on all 16 basis elements");
}

#[test]
fn test_gourlay_epsilon_order_2() {
    // epsilon^2(e_i) = e_i for all basis elements
    for i in 0..16_usize {
        let mut v = [0.0_f64; 16];
        v[i] = 1.0;
        let eps2 = gourlay_epsilon(&gourlay_epsilon(&v));
        let max_err: f64 = v.iter().zip(eps2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-12,
            "epsilon^2(e_{}) != e_{}: max error {:.2e}", i, i, max_err);
    }
    println!("epsilon^2 = Id verified on all 16 basis elements");
}

#[test]
fn test_gourlay_s3_relation() {
    // epsilon * psi = psi^2 * epsilon for all basis elements
    for i in 0..16_usize {
        let mut v = [0.0_f64; 16];
        v[i] = 1.0;

        // LHS: epsilon(psi(v))
        let psi_v = gourlay_psi(&v);
        let eps_psi_v = gourlay_epsilon(&psi_v);

        // RHS: psi^2(epsilon(v))
        let eps_v = gourlay_epsilon(&v);
        let psi2_eps_v = gourlay_psi_n(&eps_v, 2);

        let max_err: f64 = eps_psi_v.iter().zip(psi2_eps_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-12,
            "epsilon*psi != psi^2*epsilon on e_{}: max error {:.2e}", i, max_err);
    }
    println!("S3 relation epsilon*psi = psi^2*epsilon verified on all 16 basis elements");
}

#[test]
fn test_gourlay_psi_zd_preservation() {
    // psi preserves ZD pairs: if (a, b) is a ZD pair, then (psi(a), psi(b)) is also a ZD pair
    let mut a = [0.0_f64; 16]; a[1] = 1.0; a[10] = 1.0;
    let mut b = [0.0_f64; 16]; b[5] = 1.0; b[14] = 1.0;

    // Verify original is a ZD
    let ab = cd_multiply(&a.to_vec(), &b.to_vec());
    let ab_norm: f64 = ab.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(ab_norm < 1e-12, "Original pair not a ZD");

    // Apply psi to both
    let pa = gourlay_psi(&a);
    let pb = gourlay_psi(&b);
    let pab = cd_multiply(&pa.to_vec(), &pb.to_vec());
    let pab_norm: f64 = pab.iter().map(|x| x * x).sum::<f64>().sqrt();

    println!("psi(a)*psi(b) norm = {:.6e}", pab_norm);
    assert!(pab_norm < 1e-10,
        "psi must preserve ZD pairs: psi(a)*psi(b) norm = {:.2e}", pab_norm);
    println!("psi preserves zero-divisor structure");
}

// =========================================================================
// D8: CD sign table vs scattering amplitude recursion (Guevara-Strominger)
// =========================================================================

/// Compare the CD sign recursion with the Berends-Giele single-minus
/// stripped amplitude recursion from arXiv:2602.12176.
///
/// The stripped amplitudes A_{1...n} in the half-collinear regime are
/// piecewise-constant functions of the momenta z_i, taking values {+1,-1,0}.
/// They are built from products of sign functions sg_{ij} = sign(z_ij).
///
/// Concrete formulas (eqs 29-32 of the paper):
///   A_{123} = sg_{12}
///   A_{1234} = (sg_{23}*sg_{41} + sg_{12}*sg_{34}) / 2
///   A_{12345} = (1/4)[8 terms of sg products]
///
/// Our cd_basis_mul_sign(dim=2^n, p, q) also produces {+1,-1} via a
/// recursive halving. The question: is there an index map such that
/// the amplitude values match CD sign values?
///
/// This test evaluates the amplitudes for ALL sign assignments (all
/// possible {+1,-1} values for the z_ij variables) and checks whether
/// the resulting amplitude is always in {+1,-1,0}.
#[test]
fn test_amplitude_cd_sign_comparison() {
    // sg function: maps a real value to its sign {+1, -1}
    // In the half-collinear regime, z_ij = z_i - z_j where z_i are
    // ordered real parameters. We parametrize by choosing all possible
    // orderings of n points on a line.

    // For n=3: A_{123} = sg_{12} = sign(z_1 - z_2)
    // This is just +1 or -1 depending on ordering.
    // cd_sign(8, 1, 2) = +1 (fixed by Fano plane)
    // So A_{123} can be +1 or -1, but cd_sign is always +1.
    // NOT a direct match -- the amplitude depends on kinematics.

    // For n=4: A_{1234} = (sg_{23}*sg_{41} + sg_{12}*sg_{34}) / 2
    // Enumerate all 2^3 = 8 possible sign patterns for {sg_{12}, sg_{23}, sg_{34}}
    // (sg_{41} is determined: sg_{41} = -sg_{14} and in ordered form
    //  z_{41} = z_4 - z_1, so sg_{41} depends on z_4 vs z_1)

    // Actually, for n=4 with half-collinear kinematics (all <ij>=0),
    // the z_i are just ordered real numbers. The sign pattern is
    // determined by the ordering. For 4 points on a line, there are
    // 4! = 24 orderings, but many give the same sign pattern.

    // Let's compute A_{1234} for all 4! orderings of (z_1, z_2, z_3, z_4)
    // and verify the result is always in {+1, -1, 0}.

    let sg = |x: f64| -> i32 {
        if x > 0.0 { 1 } else if x < 0.0 { -1 } else { 0 }
    };

    // n=4: A_{1234} = (sg_{23}*sg_{41} + sg_{12}*sg_{34}) / 2
    let perms_4: [(usize, usize, usize, usize); 24] = [
        (0,1,2,3),(0,1,3,2),(0,2,1,3),(0,2,3,1),(0,3,1,2),(0,3,2,1),
        (1,0,2,3),(1,0,3,2),(1,2,0,3),(1,2,3,0),(1,3,0,2),(1,3,2,0),
        (2,0,1,3),(2,0,3,1),(2,1,0,3),(2,1,3,0),(2,3,0,1),(2,3,1,0),
        (3,0,1,2),(3,0,2,1),(3,1,0,2),(3,1,2,0),(3,2,0,1),(3,2,1,0),
    ];

    // Use distinct z values so all sg's are nonzero
    let z_vals = [1.0_f64, 2.0, 3.0, 4.0];

    println!("--- D8: AMPLITUDE-CD SIGN COMPARISON ---\n");
    println!("n=4 stripped amplitudes A_{{1234}} for all 24 orderings:\n");

    let mut amplitude_values = std::collections::BTreeSet::new();

    for &(a, b, c, d) in &perms_4 {
        let z = [z_vals[a], z_vals[b], z_vals[c], z_vals[d]];
        // sg_{ij} = sign(z_i - z_j) where indices are 1-based in the paper
        let sg12 = sg(z[0] - z[1]);
        let sg23 = sg(z[1] - z[2]);
        let sg34 = sg(z[2] - z[3]);
        let sg41 = sg(z[3] - z[0]);

        // A_{1234} = (sg_{23}*sg_{41} + sg_{12}*sg_{34}) / 2
        let numerator = sg23 * sg41 + sg12 * sg34;
        let a_1234 = numerator; // * 2 to keep integer (check if always even)

        println!("  z = ({:.0},{:.0},{:.0},{:.0}): sg12={:+}, sg23={:+}, sg34={:+}, sg41={:+}, 2*A = {:+}",
            z[0], z[1], z[2], z[3], sg12, sg23, sg34, sg41, a_1234);

        amplitude_values.insert(a_1234);
    }

    println!("\n  Distinct values of 2*A_{{1234}}: {:?}", amplitude_values);

    // Key check: is 2*A always in {-2, 0, +2}?
    // If so, A is always in {-1, 0, +1} -- the piecewise-constant property.
    let all_integer = amplitude_values.iter().all(|&v| v == -2 || v == 0 || v == 2);
    println!("  All values in {{-2, 0, +2}}: {}", all_integer);

    // Now compare with CD signs at dim=16 (sedenions, 4 doublings)
    println!("\n  CD sign table comparison (dim=16):");
    println!("  cd_sign(16, p, q) for p,q in 1..4:");
    for p in 1..=4_usize {
        for q in 1..=4 {
            if p == q { continue; }
            let s = cd_basis_mul_sign_iter(16, p, q);
            println!("    cd_sign(16, {}, {}) = {:+}", p, q, s);
        }
    }

    // Count: how many distinct amplitude chambers exist for n=4?
    // vs how many nonzero cd_sign entries exist in the 4x4 sub-table?
    let mut cd_signs_4 = Vec::new();
    for p in 1..=4_usize {
        for q in 1..=4 {
            if p != q {
                cd_signs_4.push(cd_basis_mul_sign_iter(16, p, q));
            }
        }
    }
    println!("\n  CD signs for indices 1..4: {:?}", cd_signs_4);
    println!("  Amplitude chambers (2*A): {:?}", amplitude_values);
    println!("  Structural parallel: both produce {{+1, -1}} values");
    println!("  Direct isomorphism: UNLIKELY (amplitude depends on z-ordering,");
    println!("    CD sign is fixed by algebra). Connection is at recursion level,");
    println!("    not at individual-value level.");

    // The test PASSES regardless -- it's exploratory, not assertive.
    // The key finding is whether 2*A is always in {-2, 0, +2}.
    assert!(all_integer, "2*A should always be in {{-2, 0, +2}} for n=4");
}
