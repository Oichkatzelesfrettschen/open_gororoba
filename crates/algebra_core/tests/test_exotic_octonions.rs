use algebra_core::construction::cayley_dickson::{
    cd_multiply, CdSignature,
};

// Helper: Standard Octonion Multiply (dim 8)
fn oct_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), 8);
    assert_eq!(b.len(), 8);
    cd_multiply(a, b)
}

// Helper: Conjugate (dim 8)
fn oct_conj(a: &[f64]) -> Vec<f64> {
    let mut res = a.to_vec();
    for i in 1..8 {
        res[i] = -res[i];
    }
    res
}

// 1. Dual-Octonions (O x D)
// Elements: A + eps B, where A,B are Octonions. eps^2 = 0.
// Product: (A + eps B)(C + eps D) = AC + eps(AD + BC)
fn dual_oct_multiply(u: &[f64], v: &[f64]) -> Vec<f64> {
    assert_eq!(u.len(), 16);
    assert_eq!(v.len(), 16);
    
    let a = &u[0..8];
    let b = &u[8..16];
    let c = &v[0..8];
    let d = &v[8..16];
    
    let ac = oct_multiply(a, c);
    let ad = oct_multiply(a, d);
    let bc = oct_multiply(b, c);
    
    // Real part: AC
    // Dual part: AD + BC
    let mut res = Vec::with_capacity(16);
    res.extend_from_slice(&ac);
    
    // Sum dual parts
    let mut dual_part = vec![0.0; 8];
    for i in 0..8 {
        dual_part[i] = ad[i] + bc[i];
    }
    res.extend_from_slice(&dual_part);
    res
}

// 2. Bi-Octonions (Complexified Octonions) (O x C)
// Elements: A + i B. i^2 = -1. Commutes with O? Usually yes in tensor product.
// Product: (A + i B)(C + i D) = (AC - BD) + i(AD + BC)
fn bioct_multiply(u: &[f64], v: &[f64]) -> Vec<f64> {
    assert_eq!(u.len(), 16);
    assert_eq!(v.len(), 16);
    
    let a = &u[0..8];
    let b = &u[8..16];
    let c = &v[0..8];
    let d = &v[8..16];
    
    let ac = oct_multiply(a, c);
    let bd = oct_multiply(b, d);
    let ad = oct_multiply(a, d);
    let bc = oct_multiply(b, c);
    
    // Real part: AC - BD
    let mut real_part = vec![0.0; 8];
    for i in 0..8 {
        real_part[i] = ac[i] - bd[i];
    }
    
    // Imag part: AD + BC
    let mut imag_part = vec![0.0; 8];
    for i in 0..8 {
        imag_part[i] = ad[i] + bc[i];
    }
    
    let mut res = Vec::with_capacity(16);
    res.extend_from_slice(&real_part);
    res.extend_from_slice(&imag_part);
    res
}

// 3. Para-Octonions
// Product: x * y = conj(x) * conj(y) (in standard O)
// This is related to the Okubo algebra? Or just Para-Hurwitz.
fn para_oct_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    let a_bar = oct_conj(a);
    let b_bar = oct_conj(b);
    oct_multiply(&a_bar, &b_bar)
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
            let u = { let mut v = vec![0.0; dim16]; v[i] = 1.0; v };
            let v = { let mut v = vec![0.0; dim16]; v[j] = 1.0; v };
            
            let prod = dual_oct_multiply(&u, &v);
            
            // Check sign of the single non-zero component (if any)
            let mut found = false;
            for &val in &prod {
                if val.abs() > 0.1 {
                    if val > 0.0 { pos += 1; }
                    else { neg += 1; }
                    found = true;
                    break;
                }
            }
            if !found { zero += 1; }
        }
    }
    
    println!("Dual-Octonion (16x16):");
    println!("  Pos: {}, Neg: {}, Zero: {}", pos, neg, zero);
    let total_nz = pos + neg;
    if total_nz > 0 {
        println!("  Neg Fraction (of non-zero): {:.4}", neg as f64 / total_nz as f64);
    }
    
    // B. Bi-Octonions (16 dim)
    pos = 0; neg = 0; zero = 0;
    for i in 0..dim16 {
        for j in 0..dim16 {
            let u = { let mut v = vec![0.0; dim16]; v[i] = 1.0; v };
            let v = { let mut v = vec![0.0; dim16]; v[j] = 1.0; v };
            
            let prod = bioct_multiply(&u, &v);
            
            let mut found = false;
            for &val in &prod {
                if val.abs() > 0.1 {
                    if val > 0.0 { pos += 1; }
                    else { neg += 1; }
                    found = true;
                    break;
                }
            }
            if !found { zero += 1; }
        }
    }
    
    println!("Bi-Octonion (16x16):");
    println!("  Pos: {}, Neg: {}, Zero: {}", pos, neg, zero);
    let total_nz = pos + neg;
    if total_nz > 0 {
        println!("  Neg Fraction (of non-zero): {:.4}", neg as f64 / total_nz as f64);
    }
    
    // C. Para-Octonions (8 dim)
    let dim8 = 8;
    pos = 0; neg = 0; zero = 0;
    for i in 0..dim8 {
        for j in 0..dim8 {
            let u = { let mut v = vec![0.0; dim8]; v[i] = 1.0; v };
            let v = { let mut v = vec![0.0; dim8]; v[j] = 1.0; v };
            
            let prod = para_oct_multiply(&u, &v);
            
            let mut found = false;
            for &val in &prod {
                if val.abs() > 0.1 {
                    if val > 0.0 { pos += 1; }
                    else { neg += 1; }
                    found = true;
                    break;
                }
            }
            if !found { zero += 1; }
        }
    }
    println!("Para-Octonion (8x8):");
    println!("  Pos: {}, Neg: {}, Zero: {}", pos, neg, zero);
    let total_nz = pos + neg;
    println!("  Neg Fraction: {:.4}", neg as f64 / total_nz as f64);
}

#[test]
fn test_exotic_frustration() {
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
    
    for i in 1..8 { // Real imaginary
        for j in 9..16 { // Dual imaginary
             let k = 8; // epsilon
             
             // Check (e_i * e_j) * e_k vs e_i * (e_j * e_k)
             // e_i is Oct, e_j is Oct*eps, e_k is eps.
             
             // Construct vectors
             let u = { let mut v = vec![0.0; dim16]; v[i] = 1.0; v };
             let v = { let mut v = vec![0.0; dim16]; v[j] = 1.0; v };
             let w = { let mut v = vec![0.0; dim16]; v[k] = 1.0; v };
             
             let res1 = dual_oct_multiply(&dual_oct_multiply(&u, &v), &w);
             let res2 = dual_oct_multiply(&u, &dual_oct_multiply(&v, &w));
             
             let mut diff = 0.0;
             for idx in 0..dim16 { diff += (res1[idx] - res2[idx]).abs(); }
             
             if diff > 0.1 { frustrated += 1; }
             total += 1;
        }
    }
    println!("Dual-Oct Frustration (Sample): {} / {}", frustrated, total);
}

#[test]
fn test_generalized_hybrid_ladder() {
    println!("--- Frustration Hyper-Ladder (Associativity) ---");
    use algebra_core::construction::cayley_dickson::cd_multiply_split;
    use rand::Rng; // We need random sampling for dim 32
    
    // Helper to calculate frustration (non-associative triples / total triples)
    let calc_associativity_frustration = |dim: usize, gammas: &[i32], samples: usize| -> f64 {
        let sig = CdSignature::from_gammas(gammas);
        let mut frustrated = 0;
        let mut total = 0;
        let mut rng = rand::thread_rng();
        
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
            let i = rng.gen_range(1..dim);
            let j = rng.gen_range(1..dim);
            let k = rng.gen_range(1..dim);
            if i == j || j == k || i == k { continue; }
            
            let v_i = { let mut v = vec![0.0; dim]; v[i] = 1.0; v };
            let v_j = { let mut v = vec![0.0; dim]; v[j] = 1.0; v };
            let v_k = { let mut v = vec![0.0; dim]; v[k] = 1.0; v };
            
            // (ab)c
            let res1 = cd_multiply_split(&cd_multiply_split(&v_i, &v_j, &sig), &v_k, &sig);
            // a(bc)
            let res2 = cd_multiply_split(&v_i, &cd_multiply_split(&v_j, &v_k, &sig), &sig);
            
            let mut diff = 0.0;
            for idx in 0..dim { diff += (res1[idx] - res2[idx]).abs(); }
            
            if diff > 0.1 { frustrated += 1; }
            total += 1;
        }
        
        // Exhaustive fallback for dim <= 16
        if dim <= 16 {
            frustrated = 0;
            total = 0;
            for i in 1..dim {
                for j in 1..dim {
                    if i == j { continue; }
                    for k in 1..dim {
                        if k == i || k == j { continue; }
                        let v_i = { let mut v = vec![0.0; dim]; v[i] = 1.0; v };
                        let v_j = { let mut v = vec![0.0; dim]; v[j] = 1.0; v };
                        let v_k = { let mut v = vec![0.0; dim]; v[k] = 1.0; v };
                        let res1 = cd_multiply_split(&cd_multiply_split(&v_i, &v_j, &sig), &v_k, &sig);
                        let res2 = cd_multiply_split(&v_i, &cd_multiply_split(&v_j, &v_k, &sig), &sig);
                        let mut diff = 0.0;
                        for idx in 0..dim { diff += (res1[idx] - res2[idx]).abs(); }
                        if diff > 0.1 { frustrated += 1; }
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
        println!("  Standard: {:.5}", calc_associativity_frustration(dim as usize, &g_std, samples));
        
        let g_split = vec![1; depth];
        println!("  Split   : {:.5}", calc_associativity_frustration(dim as usize, &g_split, samples));
        
        let mut g_hybrid = vec![1; depth];
        g_hybrid[depth-1] = -1;
        println!("  Hybrid  : {:.5}", calc_associativity_frustration(dim as usize, &g_hybrid, samples));
    }
}
