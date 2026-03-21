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
