use gororoba_algebra::{
    Bioctonion, DualOctonion, ParaOctonion,
    construction::{cayley_dickson::CdSignature, octonion::Octonion},
};

fn as_octonion(slice: &[f64]) -> Octonion {
    Octonion::new(slice.try_into().expect("expected an 8D octonion slice"))
}

fn dual_oct_multiply(u: &[f64], v: &[f64]) -> Vec<f64> {
    assert_eq!(u.len(), 16);
    assert_eq!(v.len(), 16);

    let lhs = DualOctonion::new(as_octonion(&u[0..8]), as_octonion(&u[8..16]));
    let rhs = DualOctonion::new(as_octonion(&v[0..8]), as_octonion(&v[8..16]));
    let product = lhs.multiply(&rhs);

    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&product.real.components);
    out.extend_from_slice(&product.dual.components);
    out
}

fn bioct_multiply(u: &[f64], v: &[f64]) -> Vec<f64> {
    assert_eq!(u.len(), 16);
    assert_eq!(v.len(), 16);

    let lhs = Bioctonion::new(as_octonion(&u[0..8]), as_octonion(&u[8..16]));
    let rhs = Bioctonion::new(as_octonion(&v[0..8]), as_octonion(&v[8..16]));
    let product = lhs.multiply(&rhs);

    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&product.real.components);
    out.extend_from_slice(&product.imag.components);
    out
}

fn para_oct_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    let product = ParaOctonion::new(as_octonion(a)).multiply(&ParaOctonion::new(as_octonion(b)));
    product.value.components.to_vec()
}

#[test]
fn test_exotic_sign_balance() {
    println!("--- Exotic Octonion Sign Census ---");

    // A. Dual-Octonions (16 dim)
    let dim16 = 16;
    let mut pos = 0;
    let mut neg = 0;
    let mut zero = 0;

    for i in 0..dim16 {
        for j in 0..dim16 {
            let u = {
                let mut v = vec![0.0; dim16];
                v[i] = 1.0;
                v
            };
            let v = {
                let mut v = vec![0.0; dim16];
                v[j] = 1.0;
                v
            };

            let prod = dual_oct_multiply(&u, &v);

            // Check sign of the single non-zero component (if any)
            let mut found = false;
            for &val in &prod {
                if val.abs() > 0.1 {
                    if val > 0.0 {
                        pos += 1;
                    } else {
                        neg += 1;
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                zero += 1;
            }
        }
    }

    println!("Dual-Octonion (16x16):");
    println!("  Pos: {}, Neg: {}, Zero: {}", pos, neg, zero);
    let total_nz = pos + neg;
    if total_nz > 0 {
        println!(
            "  Neg Fraction (of non-zero): {:.4}",
            neg as f64 / total_nz as f64
        );
    }

    // B. Bi-Octonions (16 dim)
    pos = 0;
    neg = 0;
    zero = 0;
    for i in 0..dim16 {
        for j in 0..dim16 {
            let u = {
                let mut v = vec![0.0; dim16];
                v[i] = 1.0;
                v
            };
            let v = {
                let mut v = vec![0.0; dim16];
                v[j] = 1.0;
                v
            };

            let prod = bioct_multiply(&u, &v);

            let mut found = false;
            for &val in &prod {
                if val.abs() > 0.1 {
                    if val > 0.0 {
                        pos += 1;
                    } else {
                        neg += 1;
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                zero += 1;
            }
        }
    }

    println!("Bi-Octonion (16x16):");
    println!("  Pos: {}, Neg: {}, Zero: {}", pos, neg, zero);
    let total_nz = pos + neg;
    if total_nz > 0 {
        println!(
            "  Neg Fraction (of non-zero): {:.4}",
            neg as f64 / total_nz as f64
        );
    }

    // C. Para-Octonions (8 dim)
    let dim8 = 8;
    pos = 0;
    neg = 0;
    zero = 0;
    for i in 0..dim8 {
        for j in 0..dim8 {
            let u = {
                let mut v = vec![0.0; dim8];
                v[i] = 1.0;
                v
            };
            let v = {
                let mut v = vec![0.0; dim8];
                v[j] = 1.0;
                v
            };

            let prod = para_oct_multiply(&u, &v);

            let mut found = false;
            for &val in &prod {
                if val.abs() > 0.1 {
                    if val > 0.0 {
                        pos += 1;
                    } else {
                        neg += 1;
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                zero += 1;
            }
        }
    }
    println!("Para-Octonion (8x8):");
    println!("  Pos: {}, Neg: {}, Zero: {}", pos, neg, zero);
    let total_nz = pos + neg;
    println!("  Neg Fraction: {:.4}", neg as f64 / total_nz as f64);
}

#[test]
fn test_exotic_imbalance() {
    // Check Associativity for these algebras
    println!("--- Exotic Associativity ---");

    // Dual-Octonions
    let dim16 = 16;
    let mut frustrated = 0;
    let mut total = 0;
    // Sample a subset to save time? 16^3 is 4096. Fast enough.

    // Use indices 1..dim to skip identity e0?
    // Dual-Oct identity is (1, 0). Indices 0.
    // Indices 8 is (0, 1) = epsilon.
    // Let's check a few interesting triples.

    for i in 1..8 {
        // Real imaginary
        for j in 9..16 {
            // Dual imaginary
            let k = 8; // epsilon

            // Check (e_i * e_j) * e_k vs e_i * (e_j * e_k)
            // e_i is Oct, e_j is Oct*eps, e_k is eps.

            // Construct vectors
            let u = {
                let mut v = vec![0.0; dim16];
                v[i] = 1.0;
                v
            };
            let v = {
                let mut v = vec![0.0; dim16];
                v[j] = 1.0;
                v
            };
            let w = {
                let mut v = vec![0.0; dim16];
                v[k] = 1.0;
                v
            };

            let res1 = dual_oct_multiply(&dual_oct_multiply(&u, &v), &w);
            let res2 = dual_oct_multiply(&u, &dual_oct_multiply(&v, &w));

            let mut diff = 0.0;
            for idx in 0..dim16 {
                diff += (res1[idx] - res2[idx]).abs();
            }

            if diff > 0.1 {
                frustrated += 1;
            }
            total += 1;
        }
    }
    println!("Dual-Oct Imbalance (Sample): {} / {}", frustrated, total);
}

#[test]
fn test_generalized_hybrid_ladder() {
    println!("--- Imbalance Hyper-Ladder (Associativity) ---");
    use gororoba_algebra::construction::cayley_dickson::cd_multiply_split;
    use rand::RngExt; // We need random sampling for dim 32

    // Helper to calculate imbalance (non-associative triples / total triples)
    let calc_associativity_imbalance = |dim: usize, gammas: &[i32], samples: usize| -> f64 {
        let sig = CdSignature::from_gammas(gammas);
        let mut frustrated = 0;
        let mut total = 0;
        let mut rng = rand::rng();

        let iter_limit = if dim <= 16 { dim * dim * dim } else { samples };

        for _ in 0..iter_limit {
            // Random distinct non-zero indices?
            // Or exhaustive for small dim?
            let (_i, _j, _k) = (0, 0, 0);
            if dim <= 16 {
                // Exhaustive loop logic mapped to linear index?
                // Just use the loop if dim is small.
                // But mixing logic is messy. Let's just use random for all > 8?
                // Actually, dim 16 is 4096. Fast.
                // Let's use exhaustive for <= 16.
                continue;
            }

            // Random sampling for dim 32+
            let i = rng.random_range(1..dim);
            let j = rng.random_range(1..dim);
            let k = rng.random_range(1..dim);
            if i == j || j == k || i == k {
                continue;
            }

            let v_i = {
                let mut v = vec![0.0; dim];
                v[i] = 1.0;
                v
            };
            let v_j = {
                let mut v = vec![0.0; dim];
                v[j] = 1.0;
                v
            };
            let v_k = {
                let mut v = vec![0.0; dim];
                v[k] = 1.0;
                v
            };

            // (ab)c
            let res1 = cd_multiply_split(&cd_multiply_split(&v_i, &v_j, &sig), &v_k, &sig);
            // a(bc)
            let res2 = cd_multiply_split(&v_i, &cd_multiply_split(&v_j, &v_k, &sig), &sig);

            let mut diff = 0.0;
            for idx in 0..dim {
                diff += (res1[idx] - res2[idx]).abs();
            }

            if diff > 0.1 {
                frustrated += 1;
            }
            total += 1;
        }

        // Exhaustive fallback for dim <= 16
        if dim <= 16 {
            frustrated = 0;
            total = 0;
            for i in 1..dim {
                for j in 1..dim {
                    if i == j {
                        continue;
                    }
                    for k in 1..dim {
                        if k == i || k == j {
                            continue;
                        }
                        let v_i = {
                            let mut v = vec![0.0; dim];
                            v[i] = 1.0;
                            v
                        };
                        let v_j = {
                            let mut v = vec![0.0; dim];
                            v[j] = 1.0;
                            v
                        };
                        let v_k = {
                            let mut v = vec![0.0; dim];
                            v[k] = 1.0;
                            v
                        };
                        let res1 =
                            cd_multiply_split(&cd_multiply_split(&v_i, &v_j, &sig), &v_k, &sig);
                        let res2 =
                            cd_multiply_split(&v_i, &cd_multiply_split(&v_j, &v_k, &sig), &sig);
                        let mut diff = 0.0;
                        for idx in 0..dim {
                            diff += (res1[idx] - res2[idx]).abs();
                        }
                        if diff > 0.1 {
                            frustrated += 1;
                        }
                        total += 1;
                    }
                }
            }
        }

        frustrated as f64 / total as f64
    };

    let dims: Vec<u32> = vec![16, 32];

    for &dim in &dims {
        println!("Dimension {}:", dim);
        let depth = dim.trailing_zeros() as usize;
        let samples = 10000;

        let g_std = vec![-1; depth];
        println!(
            "  Standard: {:.5}",
            calc_associativity_imbalance(dim as usize, &g_std, samples)
        );

        let g_split = vec![1; depth];
        println!(
            "  Split   : {:.5}",
            calc_associativity_imbalance(dim as usize, &g_split, samples)
        );

        let mut g_hybrid = vec![1; depth];
        g_hybrid[depth - 1] = -1;
        println!(
            "  Hybrid  : {:.5}",
            calc_associativity_imbalance(dim as usize, &g_hybrid, samples)
        );
    }
}
