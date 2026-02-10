use algebra_core::construction::cayley_dickson::{
    cd_multiply_split, CdSignature,
};
use std::collections::HashSet;

/// Represents a simple zero-divisor candidate: e_i + s * e_j
/// where s is +1 or -1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SimpleBlade {
    i: usize,
    j: usize,
    sign: i8, // +1 or -1
}

impl SimpleBlade {
    fn new(i: usize, j: usize, sign: i8) -> Self {
        assert!(i < j);
        Self { i, j, sign }
    }

    /// Convert to dense vector for split-octonions (dim 8)
    fn to_vec(&self) -> Vec<f64> {
        let mut v = vec![0.0; 8];
        v[self.i] = 1.0;
        v[self.j] = self.sign as f64;
        // Normalize? For ZD check, normalization doesn't matter.
        // But for consistency, let's keep it unnormalized or 1/sqrt(2).
        // Let's use integers for logic, floats for multiply.
        v
    }
}

#[test]
fn test_split_octonion_basis_squares() {
    // 1. Verify basis element squares in Split-Octonions
    // Standard Octonions: e_i^2 = -1 for i=1..7
    // Split Octonions: Signature is (+1, +1, +1) usually?
    // Let's check CdSignature::split(8) which sets all gamma=+1.
    
    let dim = 8;
    let sig = CdSignature::split(dim);
    
    println!("Split-Octonion Basis Squares:");
    let mut squares = Vec::new();
    for i in 0..dim {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        let sq = cd_multiply_split(&v, &v, &sig);
        
        // The result should be +/- 1 (real part).
        // We expect some to be +1 (timelike/split) and some -1 (spacelike).
        let val = sq[0];
        squares.push(val as i32);
        println!("e_{}^2 = {}", i, val);
    }
    
    // e_0^2 is always 1.
    assert_eq!(squares[0], 1);
    
    // Count signature (how many +1, how many -1)
    let pos_count = squares.iter().filter(|&&x| x == 1).count();
    let neg_count = squares.iter().filter(|&&x| x == -1).count();
    
    println!("Signature: {} positive, {} negative", pos_count, neg_count);

    // Deterministic split-octonion basis-square pattern from CdSignature::split(8).
    assert_eq!(squares, vec![1, 1, 1, -1, 1, -1, -1, 1]);
    assert_eq!(pos_count, 5);
    assert_eq!(neg_count, 3);

    // Imaginary basis (e1..e7) has signature (4,3).
    let imag_pos = squares[1..].iter().filter(|&&x| x == 1).count();
    let imag_neg = squares[1..].iter().filter(|&&x| x == -1).count();
    assert_eq!(imag_pos, 4);
    assert_eq!(imag_neg, 3);
}

#[test]
fn test_split_octonion_zero_divisor_census() {
    let dim = 8;
    let sig = CdSignature::split(dim);
    
    // 2. Search for Simple Zero Divisors: (e_i +/- e_j) * (e_k +/- e_l) = 0
    // In Sedenions, these are the "Assessors".
    
    let mut blades = Vec::new();
    // Generate all simple blades
    for i in 0..dim {
        for j in (i+1)..dim {
            blades.push(SimpleBlade::new(i, j, 1));
            blades.push(SimpleBlade::new(i, j, -1));
        }
    }
    
    println!("Scanning {} candidate blades...", blades.len());
    assert_eq!(blades.len(), 56);
    
    let mut zd_pairs = Vec::new();
    
    for (idx1, b1) in blades.iter().enumerate() {
        for (idx2, b2) in blades.iter().enumerate() {
            if idx1 >= idx2 { continue; } // Distinct pairs
            
            // Optimization: Disjoint indices?
            // In Sedenions, ZDs are typically disjoint or share specific relations.
            // Let's just brute force multiply.
            
            let v1 = b1.to_vec();
            let v2 = b2.to_vec();
            
            let prod = cd_multiply_split(&v1, &v2, &sig);
            let norm_sq: f64 = prod.iter().map(|x| x*x).sum();
            
            if norm_sq < 1e-10 {
                // FOUND A ZERO DIVISOR PAIR
                
                // Check if the blades themselves are null vectors (isotropic)
                // In split signature, e_i^2 can be 1 or -1.
                // v = e_i + s*e_j. 
                // v^2 = e_i^2 + s(e_i e_j + e_j e_i) + e_j^2
                //     = e_i^2 + e_j^2 (since anticommuting terms cancel in CD, or commute?)
                // Actually, let's just calculate the product v*v to see if it is null.
                
                let sq1 = cd_multiply_split(&v1, &v1, &sig);
                let sq2 = cd_multiply_split(&v2, &v2, &sig);
                
                let is_null1 = sq1.iter().map(|x| x*x).sum::<f64>() < 1e-10;
                let is_null2 = sq2.iter().map(|x| x*x).sum::<f64>() < 1e-10;
                
                zd_pairs.push((b1.clone(), b2.clone(), is_null1, is_null2));
            }
        }
    }
    
    println!("Found {} Zero-Divisor Pairs among simple blades.", zd_pairs.len());
    
    let proper_zds: Vec<_> = zd_pairs.iter().filter(|(_,_,n1,n2)| !*n1 && !*n2).collect();
    let null_zds: Vec<_> = zd_pairs.iter().filter(|(_,_,n1,n2)| *n1 || *n2).collect();
    
    println!("Proper ZDs (neither factor is null): {}", proper_zds.len());
    println!("Null-Null ZDs (involving null vectors): {}", null_zds.len());

    // Deterministic split-octonion census for this simple-blade family.
    assert_eq!(zd_pairs.len(), 52);
    assert_eq!(proper_zds.len(), 4);
    assert_eq!(null_zds.len(), 48);

    // 3. Analyze Structure
    // Do they form a graph?
    if !zd_pairs.is_empty() {
        // Build adjacency graph
        // Nodes = blades participating in at least one ZD pair
        // Edges = ZD relationship
        
        let mut nodes = HashSet::new();
        for (a, b, _, _) in &zd_pairs {
            nodes.insert(a);
            nodes.insert(b);
        }
        
        println!("Unique blades involved: {}", nodes.len());
        assert_eq!(nodes.len(), 32);
        
        // In Sedenions, we had 42 primitive assessors.
        // What is the count here?
        
        // Also check if these are just "Null Vector * Orthogonal Null Vector"?
        // In Split algebras, zero divisors are often associated with the light cone.
        
        // Let's print a few examples
        for (i, (a, b, n1, n2)) in zd_pairs.iter().take(20).enumerate() {
            println!("ZD #{}: (e_{} {:+} e_{}) * (e_{} {:+} e_{}) = 0  [Null? {}, {}]", 
                i, a.i, a.sign, a.j, b.i, b.sign, b.j, n1, n2);
        }
    }
}
