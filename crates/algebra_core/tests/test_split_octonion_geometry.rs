use algebra_core::construction::cayley_dickson::{cd_multiply_split, CdSignature};
use std::collections::{HashMap, HashSet};

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

const ZERO_TOL: f64 = 1e-10;

fn generate_simple_blades(dim: usize) -> Vec<SimpleBlade> {
    let mut blades = Vec::new();
    for i in 0..dim {
        for j in (i + 1)..dim {
            blades.push(SimpleBlade::new(i, j, 1));
            blades.push(SimpleBlade::new(i, j, -1));
        }
    }
    blades
}

fn is_null_vector(v: &[f64], sig: &CdSignature) -> bool {
    cd_multiply_split(v, v, sig)
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        < ZERO_TOL
}

fn enumerate_simple_blade_zd_pairs(
    blades: &[SimpleBlade],
    sig: &CdSignature,
) -> Vec<(SimpleBlade, SimpleBlade, bool, bool)> {
    let mut pairs = Vec::new();
    for (idx1, b1) in blades.iter().enumerate() {
        for (idx2, b2) in blades.iter().enumerate() {
            if idx1 >= idx2 {
                continue;
            }
            let v1 = b1.to_vec();
            let v2 = b2.to_vec();
            let prod = cd_multiply_split(&v1, &v2, sig);
            let norm_sq: f64 = prod.iter().map(|x| x * x).sum();
            if norm_sq < ZERO_TOL {
                pairs.push((*b1, *b2, is_null_vector(&v1, sig), is_null_vector(&v2, sig)));
            }
        }
    }
    pairs
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

    // 2. Search for SIMPLE-BLADE zero divisors: (e_i +/- e_j) * (e_k +/- e_l) = 0.
    // Scope note: this is a restricted family census over two-term blades only.
    // It is NOT the full wedge 2-blade census used by C-547 (128-pair claim).

    let blades = generate_simple_blades(dim);

    println!("Scanning {} candidate blades...", blades.len());
    assert_eq!(blades.len(), 56);
    let zd_pairs = enumerate_simple_blade_zd_pairs(&blades, &sig);

    println!(
        "Found {} Zero-Divisor Pairs among simple blades.",
        zd_pairs.len()
    );

    let proper_zds: Vec<_> = zd_pairs
        .iter()
        .filter(|(_, _, n1, n2)| !*n1 && !*n2)
        .collect();
    let null_zds: Vec<_> = zd_pairs
        .iter()
        .filter(|(_, _, n1, n2)| *n1 || *n2)
        .collect();

    println!("Proper ZDs (neither factor is null): {}", proper_zds.len());
    println!("Null-Null ZDs (involving null vectors): {}", null_zds.len());

    // Deterministic counts for the restricted simple-blade family.
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

        // Let's print a few examples
        for (i, (a, b, n1, n2)) in zd_pairs.iter().take(20).enumerate() {
            println!(
                "ZD #{}: (e_{} {:+} e_{}) * (e_{} {:+} e_{}) = 0  [Null? {}, {}]",
                i, a.i, a.sign, a.j, b.i, b.sign, b.j, n1, n2
            );
        }
    }
}

#[test]
fn test_split_octonion_sign_census() {
    // 4. Exact count of structure constant signs in 8x8 table
    // Hypothesis: Negative fraction = 3/8 (24/64)

    let dim = 8;
    let sig = CdSignature::split(dim);

    let mut pos_count = 0;
    let mut neg_count = 0;
    let mut zero_count = 0;

    println!("Sign Census (8x8):");

    for i in 0..dim {
        for j in 0..dim {
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

            let prod = cd_multiply_split(&v_i, &v_j, &sig);

            // Find the non-zero component
            let mut found = false;
            for (_k, &val) in prod.iter().enumerate() {
                if val.abs() > 0.1 {
                    if val > 0.0 {
                        pos_count += 1;
                    } else {
                        neg_count += 1;
                    }
                    found = true;
                    // println!("e_{} * e_{} = {:+} e_{}", i, j, val, _k);
                    break;
                }
            }
            if !found {
                zero_count += 1;
            }
        }
    }

    let total = pos_count + neg_count + zero_count;
    println!("Total entries: {}", total);
    println!("Positive: {}", pos_count);
    println!("Negative: {}", neg_count);
    println!("Zero: {}", zero_count);

    let neg_fraction = neg_count as f64 / total as f64;
    println!(
        "Negative Fraction: {} / {} = {}",
        neg_count, total, neg_fraction
    );

    assert_eq!(total, 64);
    assert_eq!(pos_count, 40);
    assert_eq!(neg_count, 24);
    assert_eq!(zero_count, 0);
    assert!((neg_fraction - 0.375).abs() < 1e-12);

    // The "Split-Octonion Attractor" Insight claims this should be 0.375 (24/64).
    println!("CONFIRMED: Exactly 24/64 negative entries (3/8).");
}

#[test]
fn test_null_cloud_topology() {
    // 5. Detailed topology of the SIMPLE-BLADE zero-divisor cloud
    // Goal: Classify into Proper-Proper, Mixed (Proper-Null), and Pure (Null-Null).
    // Analyze the Pure set for the "missing 6".

    let dim = 8;
    let sig = CdSignature::split(dim);
    let blades = generate_simple_blades(dim);
    let zd_pairs = enumerate_simple_blade_zd_pairs(&blades, &sig);

    let mut pp_count = 0;
    let mut mixed_count = 0;
    let mut nn_count = 0;
    let mut nn_pairs = Vec::new();

    for (b1, b2, is_null1, is_null2) in zd_pairs {
        if !is_null1 && !is_null2 {
            pp_count += 1;
        } else if is_null1 && is_null2 {
            nn_count += 1;
            nn_pairs.push((b1, b2));
        } else {
            mixed_count += 1;
        }
    }

    println!("Topology Census:");
    println!("Proper-Proper: {}", pp_count);
    println!("Mixed (Proper-Null): {}", mixed_count);
    println!("Pure Null-Null: {}", nn_count);
    assert_eq!(pp_count, 4);
    assert_eq!(mixed_count, 24);
    assert_eq!(nn_count, 24);

    // Analyze the Pure Null-Null graph
    let mut nodes = HashSet::new();
    for (a, b) in &nn_pairs {
        nodes.insert(a);
        nodes.insert(b);
    }
    println!("Pure Null Graph: {} nodes, {} edges", nodes.len(), nn_count);
    assert_eq!(nodes.len(), 24);
    assert_eq!(nn_pairs.len(), 24);

    // Check degrees
    let mut degrees = HashMap::new();
    for (a, b) in &nn_pairs {
        *degrees.entry(a).or_insert(0) += 1;
        *degrees.entry(b).or_insert(0) += 1;
    }
    assert_eq!(degrees.len(), 24);
    assert!(degrees.values().all(|&d| d == 2));

    let mut degree_counts = HashMap::new();
    for d in degrees.values() {
        *degree_counts.entry(d).or_insert(0) += 1;
    }

    println!("Degree distribution in Null Cloud:");
    for (d, c) in degree_counts {
        println!("Degree {}: {} nodes", d, c);
    }

    // Analyze Connected Components (Cycles)
    let mut visited = HashSet::new();
    let mut components = Vec::new();
    let mut adjacency = HashMap::new();

    // Build adjacency list
    for (a, b) in &nn_pairs {
        adjacency.entry(a).or_insert_with(Vec::new).push(b);
        adjacency.entry(b).or_insert_with(Vec::new).push(a);
    }

    let all_nodes: Vec<_> = adjacency.keys().cloned().collect();

    for start_node in all_nodes {
        if visited.contains(start_node) {
            continue;
        }

        let mut component = Vec::new();
        let mut stack = vec![start_node];
        visited.insert(start_node);

        while let Some(node) = stack.pop() {
            component.push(node);
            if let Some(neighbors) = adjacency.get(node) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor);
                        stack.push(neighbor);
                    }
                }
            }
        }
        components.push(component.len());
    }

    components.sort();
    println!("Connected Components (Cycle Lengths): {:?}", components);
    assert_eq!(components, vec![6, 6, 6, 6]);

    // The "6-element difference" hypothesis:
    // Sedenions have 42 primitive ZDs.
    // If we have 48 Null-Nulls, the diff is exactly 6.
    // Let's see if 48 is the Pure count or the Total-Proper count.

    let total_improper = mixed_count + nn_count;
    println!("Total Improper (Mixed + Pure): {}", total_improper);
    assert_eq!(total_improper, 48);
}

#[test]
fn test_split_octonion_triad_frustration() {
    // 6. Triad Frustration Census
    // Count non-associative triples: (e_i * e_j) * e_k != e_i * (e_j * e_k)
    // Sedenions (dim 16) have 42 frustrated triads.
    // What about Split-Octonions (dim 8)?
    
    let dim = 8;
    let sig = CdSignature::split(dim);
    
    let mut frustrated_count = 0;
    let mut total_triples = 0;
    
    println!("Triad Frustration Census (dim 8):");
    
    for i in 1..dim {
        for j in 1..dim {
            if i == j { continue; }
            for k in 1..dim {
                if k == i || k == j { continue; }
                
                total_triples += 1;
                
                let v_i = { let mut v = vec![0.0; dim]; v[i] = 1.0; v };
                let v_j = { let mut v = vec![0.0; dim]; v[j] = 1.0; v };
                let v_k = { let mut v = vec![0.0; dim]; v[k] = 1.0; v };
                
                // (e_i * e_j) * e_k
                let res1 = cd_multiply_split(&cd_multiply_split(&v_i, &v_j, &sig), &v_k, &sig);
                // e_i * (e_j * e_k)
                let res2 = cd_multiply_split(&v_i, &cd_multiply_split(&v_j, &v_k, &sig), &sig);
                
                let mut diff = 0.0;
                for idx in 0..dim {
                    diff += (res1[idx] - res2[idx]).abs();
                }
                
                if diff > 0.1 {
                    frustrated_count += 1;
                    // println!("Frustrated: (e_{} * e_{}) * e_{} != e_{} * (e_{} * e_{})", i, j, k, i, j, k);
                }
            }
        }
    }
    
    println!("Total triples (i,j,k distinct, non-zero): {}", total_triples);
    println!("Frustrated triples: {}", frustrated_count);
    
    // Frustration Ratio for this specific algebra
    let ratio = frustrated_count as f64 / total_triples as f64;
    println!("Frustration Ratio (Local): {}", ratio);
    
    // Comparison to Sedenion 42
    // Sedenion triples are usually counted as 'sets' {i,j,k}.
    // My loop counts permutations (i,j,k). 
    // For a frustrated triad, typically all 6 permutations are frustrated?
    // 42 * 6 = 252.
    
    if frustrated_count == 0 {
        println!("CONFIRMED: Split-Octonions are ASSOCIATIVE (0 frustrated triads).");
    } else {
        println!("SURPRISE: Split-Octonions are NOT associative. Found {} frustrated triples.", frustrated_count);
    }
}

#[test]
fn test_standard_octonion_triad_frustration() {
    // 7. Baseline: Standard Octonions (dim 8, all gamma = -1)
    
    let dim = 8;
    // We need a standard signature (all -1)
    // The CdSignature struct likely has a way to create this.
    // Looking at the code, CdSignature::split(8) might be different from Standard.
    // Let's manually create one if needed, or check if there's a ::standard(dim).
    // For now, let's assume we can pass a custom gamma list.
    
    // Actually, let's just use the default CD which is standard.
    use algebra_core::construction::cayley_dickson::cd_multiply;
    
    let mut frustrated_count = 0;
    let mut total_triples = 0;
    
    println!("Standard Octonion Associativity Census:");
    
    for i in 1..dim {
        for j in 1..dim {
            if i == j { continue; }
            for k in 1..dim {
                if k == i || k == j { continue; }
                total_triples += 1;
                
                let v_i = { let mut v = vec![0.0; dim]; v[i] = 1.0; v };
                let v_j = { let mut v = vec![0.0; dim]; v[j] = 1.0; v };
                let v_k = { let mut v = vec![0.0; dim]; v[k] = 1.0; v };
                
                let res1 = cd_multiply(&cd_multiply(&v_i, &v_j), &v_k);
                let res2 = cd_multiply(&v_i, &cd_multiply(&v_j, &v_k));
                
                let mut diff = 0.0;
                for idx in 0..dim {
                    diff += (res1[idx] - res2[idx]).abs();
                }
                
                if diff > 0.1 {
                    frustrated_count += 1;
                }
            }
        }
    }
    
    println!("Total triples: {}", total_triples);
    println!("Frustrated triples: {}", frustrated_count);
    let ratio = frustrated_count as f64 / total_triples as f64;
    println!("Frustration Ratio (Standard Octonions): {}", ratio);
}

#[test]
fn test_sedenion_triad_census() {
    // 8. Sedenion Triad Census (dim 16)
    
    let dim = 16;
    use algebra_core::construction::cayley_dickson::cd_multiply;
    
    let mut frustrated_count = 0;
    let mut total_triples = 0;
    
    println!("Sedenion (dim 16) Associativity Census:");
    
    // We only need to check one permutation per triad {i,j,k} where i*j = +/- k
    // because associativity is a property of the triad set in CD basis.
    
    for i in 1..dim {
        for j in (i+1)..dim {
            // Find k such that e_i * e_j = +/- e_k
            let v_i = { let mut v = vec![0.0; dim]; v[i] = 1.0; v };
            let v_j = { let mut v = vec![0.0; dim]; v[j] = 1.0; v };
            let prod = cd_multiply(&v_i, &v_j);
            
            let mut k = 0;
            for (idx, &val) in prod.iter().enumerate() {
                if val.abs() > 0.1 {
                    k = idx;
                    break;
                }
            }
            
            // We have a triad {i, j, k}. Check its associativity.
            // Triad is associative if (e_i * e_j) * e_k = e_i * (e_j * e_k)
            // But wait, for any triad i,j,k where i*j=k, the product is always associative?
            // No! In Sedenions, (e_i * e_j) * e_k can be - e_i * (e_j * e_k).
            
            let v_k = { let mut v = vec![0.0; dim]; v[k] = 1.0; v };
            let res1 = cd_multiply(&cd_multiply(&v_i, &v_j), &v_k);
            let res2 = cd_multiply(&v_i, &cd_multiply(&v_j, &v_k));
            
            let mut diff = 0.0;
            for idx in 0..dim {
                diff += (res1[idx] - res2[idx]).abs();
            }
            
            total_triples += 1;
            if diff > 0.1 {
                frustrated_count += 1;
            }
        }
    }
    
    println!("Unique triads (sets {{i,j,k}}): {}", total_triples);
    println!("Associative triads: {}", total_triples - frustrated_count);
    println!("Frustrated triads (Non-Associative): {}", frustrated_count);
}

#[test]
fn test_sedenion_zero_divisor_census() {
    // 9. Standard Sedenion Zero Divisor Census (dim 16)
    // Looking for pairs of simple blades (e_i +/- e_j) * (e_k +/- e_l) = 0
    
    let dim = 16;
    use algebra_core::construction::cayley_dickson::cd_multiply;
    
    let mut blades = Vec::new();
    for i in 0..dim {
        for j in (i+1)..dim {
            blades.push(SimpleBlade::new(i, j, 1));
            blades.push(SimpleBlade::new(i, j, -1));
        }
    }
    
    let mut zd_pairs = Vec::new();
    for (idx1, b1) in blades.iter().enumerate() {
        for (idx2, b2) in blades.iter().enumerate() {
            if idx1 >= idx2 { continue; }
            
            let v1 = b1.to_vec_16(); // Need a to_vec_16
            let v2 = b2.to_vec_16();
            
            let prod = cd_multiply(&v1, &v2);
            let norm_sq: f64 = prod.iter().map(|x| x*x).sum();
            
            if norm_sq < 1e-10 {
                zd_pairs.push((b1.clone(), b2.clone()));
            }
        }
    }
    
    println!("Sedenion Simple Zero-Divisor Pairs: {}", zd_pairs.len());
}

#[test]
fn test_unit_census() {
    // 10. Fundamental Unit Census
    // A 'unit' is a set of 4 indices {i,j,k,l} that participates in ZDs.
    
    // Sedenions
    let dim16 = 16;
    let mut units16 = HashSet::new();
    use algebra_core::construction::cayley_dickson::cd_multiply;
    
    for i in 0..dim16 {
        for j in (i+1)..dim16 {
            for k in 0..dim16 {
                for l in (k+1)..dim16 {
                    if i == k || i == l || j == k || j == l { continue; }
                    
                    let v1 = SimpleBlade::new(i, j, 1).to_vec_16();
                    let v2 = SimpleBlade::new(k, l, 1).to_vec_16();
                    let prod = cd_multiply(&v1, &v2);
                    if prod.iter().map(|x| x*x).sum::<f64>() < 1e-10 {
                        let mut set = vec![i, j, k, l];
                        set.sort();
                        units16.insert(set);
                    }
                }
            }
        }
    }
    println!("Sedenion Fundamental Units (sets of 4): {}", units16.len());
    
    // Split-Octonions
    let dim8 = 8;
    let sig8 = CdSignature::split(dim8);
    let mut units8 = HashSet::new();
    use algebra_core::construction::cayley_dickson::cd_multiply_split;
    
    for i in 0..dim8 {
        for j in (i+1)..dim8 {
            for k in 0..dim8 {
                for l in (k+1)..dim8 {
                    if i == k || i == l || j == k || j == l { continue; }
                    
                    let v1 = SimpleBlade::new(i, j, 1).to_vec();
                    let v2 = SimpleBlade::new(k, l, 1).to_vec();
                    let prod = cd_multiply_split(&v1, &v2, &sig8);
                    if prod.iter().map(|x| x*x).sum::<f64>() < 1e-10 {
                        let mut set = vec![i, j, k, l];
                        set.sort();
                        units8.insert(set);
                    }
                }
            }
        }
    }
    println!("Split-Octonion Fundamental Units (sets of 4): {}", units8.len());
}

impl SimpleBlade {
    fn to_vec_16(&self) -> Vec<f64> {
        let mut v = vec![0.0; 16];
        v[self.i] = 1.0;
        v[self.j] = self.sign as f64;
        v
    }
}
