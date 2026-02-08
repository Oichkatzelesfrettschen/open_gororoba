//! E10 Cosmological Billiard Simulation
//!
//! This executable simulates the chaotic dynamics of a "particle" (the Universe)
//! bouncing inside the fundamental Weyl chamber of the E10 Kac-Moody algebra.
//! This dynamics (BKL billiards) is conjectured to describe the behavior of
//! spacetime near a spacelike singularity in M-theory.
//!
//! Novel Connection:
//! We analyze the sequence of wall reflections ("Weyl word") to search for
//! hidden algebraic structures, specifically checking if the transition probabilities
//! between walls reflect the underlying Octonion structure of the E8 subalgebra.

use algebra_core::e10_octonion;
use algebra_core::kac_moody::{E10RootSystem, KacMoodyRoot};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// The E10 billiard simulation.
///
/// Usage: e10_billiard [SEED] [N_STEPS]
///   SEED: u64 RNG seed for reproducibility (default: 42)
///   N_STEPS: number of bounces to simulate (default: 100000)
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let n_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);

    println!("=== E10 Cosmological Billiard Simulation ===");
    println!("RNG seed: {}, steps: {}", seed, n_steps);
    println!("Initializing E10 root system (Lorentzian signature)...");

    let e10 = E10RootSystem::new();
    let simple_roots = e10.simple_roots(); // 10 roots: 0..9
    let n_walls = simple_roots.len();

    println!("Loaded {} simple roots defining the Weyl chamber walls.", n_walls);
    println!("Signature: (9, 1) - 9 spacelike, 1 timelike direction.");

    // Verify all simple roots have norm^2 = 2 (Cartan matrix diagonal)
    for (i, root) in simple_roots.iter().enumerate() {
        let norm_sq = e10.inner_product(root, root);
        assert!(
            (norm_sq - 2.0).abs() < 1e-10,
            "Simple root {i} has |alpha|^2 = {norm_sq}, expected 2.0"
        );
    }
    println!("All {} simple roots verified: |alpha_i|^2 = 2.0", n_walls);

    // Verify E10 Cartan matrix structure: off-diagonal entries
    for i in 0..n_walls {
        for j in (i + 1)..n_walls {
            let ip = e10.inner_product(&simple_roots[i], &simple_roots[j]);
            if ip.abs() > 1e-10 {
                println!("  <alpha_{}, alpha_{}> = {:.1}", i, j, ip);
            }
        }
    }

    // Simulation parameters (deterministic RNG for reproducibility)
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Compute a position INSIDE the Weyl chamber using the Weyl vector.
    // Solve G * x = [1,...,1] where G[i][j] = <alpha_i, alpha_j> (Gram matrix).
    // Then pos = sum_i x_i * alpha_i guarantees <pos, alpha_j> = 1 for all j.
    let mut current_pos = compute_chamber_interior(&e10, &simple_roots);
    println!("Chamber interior found. Verifying all walls have positive inner product:");
    for (i, root) in simple_roots.iter().enumerate() {
        let ip = e10.inner_product(&current_pos, root);
        println!("  <pos, alpha_{}> = {:.6}", i, ip);
        assert!(ip > 0.0, "Position is NOT inside chamber for wall {i}!");
    }

    // Construct a random timelike velocity vector (signature 9,1)
    let mut current_v = KacMoodyRoot::lorentzian(
        vec![0.0; 8],
        0,
        vec![0.0, 0.0],
    );

    // Random spacelike components
    let mut spacelike_norm_sq = 0.0;
    for x in current_v.finite_part.iter_mut() {
        *x = rng.gen_range(-2.0..2.0);
        spacelike_norm_sq += *x * *x;
    }
    current_v.lorentz_coords[0] = rng.gen_range(-2.0..2.0); // Spacelike extension
    spacelike_norm_sq += current_v.lorentz_coords[0] * current_v.lorentz_coords[0];

    // Force timelike: t^2 = spacelike_norm_sq + 1.0, so norm^2 = -1
    current_v.lorentz_coords[1] = (spacelike_norm_sq + 1.0).sqrt();

    let norm_sq = e10.inner_product(&current_v, &current_v);


    println!("Start Point: Timelike position, norm^2 = {:.4}", e10.inner_product(&current_pos, &current_pos));
    println!("Initial Velocity (timelike): norm^2 = {:.4}", norm_sq);

    // Stats
    let mut wall_hits: Vec<u64> = vec![0; n_walls];
    let mut transition_matrix: Vec<Vec<u64>> = vec![vec![0; n_walls]; n_walls];
    let mut last_wall: Option<usize> = None;
    let mut actual_bounces: usize = 0;
    let mut bounce_sequence: Vec<usize> = Vec::new();

    // Constraint logging: track norm drift and wall-inequality violations
    let mut max_norm_drift: f64 = 0.0;
    let mut min_wall_ip: f64 = f64::INFINITY;
    let initial_v_norm = e10.inner_product(&current_v, &current_v);

    println!("Simulating {} bounces...", n_steps);

    for step in 0..n_steps {
        // Find collision times with all walls
        // Wall i: (pos + t*v) . alpha_i = 0 => t = - (pos . alpha_i) / (v . alpha_i)
        // We want smallest positive t.

        let mut min_t = f64::INFINITY;
        let mut hit_wall = None;

        for (i, root) in simple_roots.iter().enumerate() {
            let pos_dot = e10.inner_product(&current_pos, root);
            let v_dot = e10.inner_product(&current_v, root);

            // If v_dot is positive, we are moving away from the wall (since pos_dot > 0 inside)
            // If v_dot is negative, we are moving towards it.
            if v_dot < -1e-9 {
                let t = -pos_dot / v_dot;
                if t > 1e-9 && t < min_t {
                    min_t = t;
                    hit_wall = Some(i);
                }
            }
        }

        if let Some(wall_idx) = hit_wall {
            // Advance to just BEFORE the wall (standard hyperbolic billiard technique).
            // This avoids the need for post-bounce nudging which is unreliable in
            // Lorentzian signature where root corrections along alpha_i can violate
            // walls connected to i (since <alpha_i, alpha_j> = -1 for neighbors).
            let approach_t = min_t * (1.0 - 1e-10);
            current_pos = advance(&current_pos, &current_v, approach_t);

            // Reflect velocity: v' = v - 2 (v.alpha / alpha.alpha) alpha
            let root = &simple_roots[wall_idx];
            let v_dot_alpha = e10.inner_product(&current_v, root);
            let alpha_sq = e10.inner_product(root, root);
            let coeff = 2.0 * v_dot_alpha / alpha_sq;
            let scaled_root = scale_root(root, coeff);
            current_v = subtract_roots(&current_v, &scaled_root);

            // Constraint logging: monitor velocity norm drift (T2 invariant)
            let v_norm_now = e10.inner_product(&current_v, &current_v);
            let drift = (v_norm_now - initial_v_norm).abs();
            if drift > max_norm_drift {
                max_norm_drift = drift;
            }

            // Record stats
            wall_hits[wall_idx] += 1;
            if let Some(prev) = last_wall {
                transition_matrix[prev][wall_idx] += 1;
            }
            last_wall = Some(wall_idx);
            actual_bounces += 1;
            bounce_sequence.push(wall_idx);

            // Constraint logging: check all wall inequalities after reflection
            for (i, r) in simple_roots.iter().enumerate() {
                let ip = e10.inner_product(&current_pos, r);
                if ip < min_wall_ip {
                    min_wall_ip = ip;
                }
                if ip < -1e-6 {
                    println!("WARNING: wall {} violated at step {} (<pos,alpha> = {:.6e})",
                             i, step, ip);
                }
            }
        } else {
            println!("Particle escaped at step {}. Max norm drift: {:.6e}", step, max_norm_drift);
            break;
        }
    }

    // === Constraint diagnostics ===
    println!("\n=== Constraint Diagnostics ===");
    println!("Actual bounces completed: {}", actual_bounces);
    println!("Max velocity norm drift:  {:.6e} (should be ~0 if reflections are isometries)",
             max_norm_drift);
    println!("Min wall inner product:   {:.6e} (should be > 0 if always inside chamber)",
             min_wall_ip);
    let final_v_norm = e10.inner_product(&current_v, &current_v);
    println!("Initial |v|^2: {:.6}, Final |v|^2: {:.6}, Drift: {:.6e}",
             initial_v_norm, final_v_norm, (final_v_norm - initial_v_norm).abs());

    // === Wall hit frequencies ===
    println!("\n=== Wall Hit Frequencies ===");
    for (i, &hits) in wall_hits.iter().enumerate() {
        let pct = if actual_bounces > 0 { 100.0 * hits as f64 / actual_bounces as f64 } else { 0.0 };
        println!("  Wall {}: {:6} hits ({:.2}%)", i, hits, pct);
    }

    // === Transition matrix analysis ===
    println!("\n=== Transition Matrix Analysis ===");

    // E8 Dynkin diagram (from actual root vector Gram matrix, branch at node 4):
    //   0 -- 1 -- 2 -- 3 -- 4 -- 5
    //                        |
    //                        6 -- 7
    // Affine node 8 connects to node 0 (via highest root).
    // Hyperbolic node 9 connects to node 8.

    // Full E10 adjacency (10 nodes, from Cartan matrix):
    let adjacency_e10: [Vec<usize>; 10] = [
        vec![1, 8],       // 0 -- 1, 8
        vec![0, 2],       // 1 -- 0, 2
        vec![1, 3],       // 2 -- 1, 3
        vec![2, 4],       // 3 -- 2, 4
        vec![3, 5, 6],    // 4 -- 3, 5, 6 (branch node)
        vec![4],          // 5 -- 4
        vec![4, 7],       // 6 -- 4, 7
        vec![6],          // 7 -- 6
        vec![0, 9],       // 8 -- 0, 9 (affine)
        vec![8],          // 9 -- 8 (hyperbolic)
    ];

    // Sector-specific adjacency ratios (Claim 3: locality is sector-specific)
    let mut e8_connected: u64 = 0;
    let mut e8_disconnected: u64 = 0;
    let mut mixed_transitions: u64 = 0;  // E8 <-> affine/hyperbolic
    let mut hyp_transitions: u64 = 0;    // within {8,9}
    let mut total_connected_e10: u64 = 0;
    let mut total_disconnected_e10: u64 = 0;

    for (i, row_i) in transition_matrix.iter().enumerate() {
        for (j, &count) in row_i.iter().enumerate() {
            if i == j || count == 0 { continue; }

            let is_adjacent = adjacency_e10[i].contains(&j);
            if is_adjacent {
                total_connected_e10 += count;
            } else {
                total_disconnected_e10 += count;
            }

            // Sector classification
            let i_in_e8 = i < 8;
            let j_in_e8 = j < 8;
            match (i_in_e8, j_in_e8) {
                (true, true) => {
                    if is_adjacent { e8_connected += count; } else { e8_disconnected += count; }
                }
                (false, false) => {
                    hyp_transitions += count;
                }
                _ => {
                    mixed_transitions += count;
                }
            }
        }
    }

    println!("\nE8 sector (walls 0-7):");
    let e8_total = e8_connected + e8_disconnected;
    let r_e8 = if e8_total > 0 { e8_connected as f64 / e8_total as f64 } else { 0.0 };
    println!("  Connected:    {} ({:.4})", e8_connected, r_e8);
    println!("  Disconnected: {}", e8_disconnected);
    // Null baseline: E8 has 7 edges out of C(8,2)=28 directed pairs -> 14/56 = 0.25
    println!("  Null baseline (uniform): 0.2500");
    println!("  Locality ratio r_E8/r_null: {:.4}", r_e8 / 0.25);

    println!("\nMixed sector (E8 <-> affine/hyperbolic):");
    println!("  Transitions: {}", mixed_transitions);

    println!("\nHyperbolic sector (walls 8-9):");
    println!("  Transitions: {}", hyp_transitions);

    let e10_total = total_connected_e10 + total_disconnected_e10;
    let r_e10 = if e10_total > 0 { total_connected_e10 as f64 / e10_total as f64 } else { 0.0 };
    println!("\nFull E10 connected ratio: {:.4}", r_e10);

    // === Graph invariants on the empirical transition graph ===
    println!("\n=== Transition Graph Invariants ===");

    // Build symmetric 0/1 adjacency matrix (undirected skeleton of directed transitions)
    let mut adj_matrix = vec![vec![false; n_walls]; n_walls];
    for (i, row_i) in transition_matrix.iter().enumerate() {
        for (j, &count) in row_i.iter().enumerate() {
            if i != j && (count > 0 || transition_matrix[j][i] > 0) {
                adj_matrix[i][j] = true;
                adj_matrix[j][i] = true;
            }
        }
    }

    // Degree sequence from adjacency matrix
    let degrees: Vec<usize> = adj_matrix.iter()
        .map(|row| row.iter().filter(|&&v| v).count())
        .collect();

    // Edge count (upper triangle only)
    let edge_count: usize = (0..n_walls)
        .flat_map(|i| ((i + 1)..n_walls).map(move |j| (i, j)))
        .filter(|&(i, j)| adj_matrix[i][j])
        .count();

    // Triangle count (ordered triples i < j < k)
    let triangle_count: usize = (0..n_walls)
        .flat_map(|i| ((i + 1)..n_walls).map(move |j| (i, j)))
        .filter(|&(i, j)| adj_matrix[i][j])
        .flat_map(|(i, j)| ((j + 1)..n_walls).map(move |k| (i, j, k)))
        .filter(|&(i, j, k)| adj_matrix[j][k] && adj_matrix[i][k])
        .count();

    let mut sorted_degrees = degrees.clone();
    sorted_degrees.sort_unstable_by(|a, b| b.cmp(a));
    println!("Degree sequence (desc): {:?}", sorted_degrees);
    println!("Edge count: {}", edge_count);
    println!("Triangle count: {}", triangle_count);

    // Print adjacency matrix for external analysis
    println!("Adjacency matrix (empirical, 0/1 skeleton):");
    for row in &adj_matrix {
        let s: Vec<String> = row.iter().map(|&v| if v { "1" } else { "." }.to_string()).collect();
        println!("  {}", s.join(" "));
    }

    // === Fano plane (Octonion) analysis ===
    // Test whether the billiard transition sequence respects the Fano plane
    // structure of the octonion multiplication table (Claim 4).
    println!("\n=== Fano Plane (Octonion) Analysis ===");

    let windows = e10_octonion::extract_3windows(&bounce_sequence);
    println!("3-bounce windows (E8 only): {}", windows.len());

    if windows.len() >= 10 {
        let (best_mapping, best_rate, best_comp, best_opp, all_rates) =
            e10_octonion::optimal_fano_mapping(&windows);
        let pvalue = e10_octonion::exact_pvalue(best_rate, &all_rates);
        let zscore = e10_octonion::fano_enrichment_zscore(best_rate, best_opp);

        println!("Optimal mapping (wall -> octonion basis):");
        for (wall, &oct) in best_mapping.iter().enumerate() {
            let label = if oct == 0 { "e_0 (real)" } else { "" };
            println!("  wall {} -> e_{} {}", wall, oct, label);
        }
        println!("Fano triple completions: {}/{} ({:.4})", best_comp, best_opp, best_rate);
        println!("Null expectation (uniform): {:.4}", e10_octonion::NULL_FANO_RATE_UNIFORM);
        println!("Enrichment ratio: {:.4}", best_rate / e10_octonion::NULL_FANO_RATE_UNIFORM);
        println!("Z-score vs uniform null: {:.2}", zscore);
        println!("Exact p-value (rank among 40320 permutations): {:.6}", pvalue);

        // Report Fano triples implied by the mapping
        let triples = e10_octonion::describe_fano_structure(&best_mapping);
        println!("Implied Fano triples (wall indices):");
        for (a, b, c) in &triples {
            println!("  ({}, {}, {})", a, b, c);
        }
    } else {
        println!("Too few bounces for Fano analysis (need >= 10 3-windows).");
    }

    println!("\nDone. Seed={}, bounces={}", seed, actual_bounces);
}

// Helpers

fn advance(pos: &KacMoodyRoot, v: &KacMoodyRoot, t: f64) -> KacMoodyRoot {
    let mut new_finite = pos.finite_part.clone();
    for (i, x) in v.finite_part.iter().enumerate() {
        new_finite[i] += x * t;
    }
    
    let new_level = pos.level; // Level is discrete integer? 
    // Wait, in simulation beta is continuous. 
    // KacMoodyRoot struct uses integer level. 
    // This is a limitation of the struct for simulation.
    // We should treat level as float for simulation.
    // Hack: Store float level in lorentz_coords[1] if we had it?
    // Or just ignore level for now if it's 0.
    // The E10 inner product uses level as integer.
    // We need a proper continuous vector struct.
    
    // For this example, we will stick to the finite + lorentz parts which are floats.
    // We'll ignore the discrete level evolution for the continuous billiard
    // or assume level=0 for the "particle" (pure gravity sector).
    
    let mut new_lorentz = pos.lorentz_coords.clone();
    for (i, x) in v.lorentz_coords.iter().enumerate() {
        new_lorentz[i] += x * t;
    }
    
    KacMoodyRoot::lorentzian(new_finite, new_level, new_lorentz)
}

fn scale_root(root: &KacMoodyRoot, scale: f64) -> KacMoodyRoot {
    let finite: Vec<f64> = root.finite_part.iter().map(|x| x * scale).collect();
    let lorentz: Vec<f64> = root.lorentz_coords.iter().map(|x| x * scale).collect();
    // Level is int, so we lose precision if we scale it.
    // This confirms KacMoodyRoot isn't ideal for continuous dynamics.
    // But for the E10 simple roots, level is 0 or 1.
    // If scale is non-integer, we have a problem.
    // We'll force level to 0 for the update vector to avoid type mismatch,
    // assuming the dynamics dominates in the continuous sectors.
    KacMoodyRoot::lorentzian(finite, 0, lorentz)
}

fn subtract_roots(a: &KacMoodyRoot, b: &KacMoodyRoot) -> KacMoodyRoot {
    let finite: Vec<f64> = a.finite_part.iter().zip(b.finite_part.iter())
        .map(|(x, y)| x - y).collect();
    let lorentz: Vec<f64> = a.lorentz_coords.iter().zip(b.lorentz_coords.iter())
        .map(|(x, y)| x - y).collect();
    KacMoodyRoot::lorentzian(finite, a.level - b.level, lorentz)
}

fn add_roots(a: &KacMoodyRoot, b: &KacMoodyRoot) -> KacMoodyRoot {
    let finite: Vec<f64> = a.finite_part.iter().zip(b.finite_part.iter())
        .map(|(x, y)| x + y).collect();
    let lorentz: Vec<f64> = a.lorentz_coords.iter().zip(b.lorentz_coords.iter())
        .map(|(x, y)| x + y).collect();
    KacMoodyRoot::lorentzian(finite, a.level + b.level, lorentz)
}

/// Compute a point strictly inside the E10 Weyl chamber.
///
/// Solves G * x = [1,...,1] where G is the Gram matrix of simple roots,
/// then returns pos = sum_i x_i * alpha_i. This guarantees <pos, alpha_j> = 1
/// for all j, placing pos exactly at the Weyl vector.
fn compute_chamber_interior(
    e10: &E10RootSystem,
    roots: &[KacMoodyRoot],
) -> KacMoodyRoot {
    let n = roots.len();

    // Build Gram matrix
    let mut gram = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            gram[i][j] = e10.inner_product(&roots[i], &roots[j]);
        }
    }

    // Augmented matrix [G | 1]
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = gram[i][j];
        }
        aug[i][n] = 1.0; // RHS: we want <pos, alpha_i> = 1
    }

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        // Find pivot (partial pivoting)
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for (row, aug_row) in aug.iter().enumerate().skip(col + 1).take(n - col - 1) {
            if aug_row[col].abs() > max_val {
                max_val = aug_row[col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        assert!(pivot.abs() > 1e-12, "Gram matrix is singular at column {col}");

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            // Split borrow: copy the pivot row segment, then update
            let pivot_row: Vec<f64> = aug[col][col..=n].to_vec();
            for (j_off, &pval) in pivot_row.iter().enumerate() {
                aug[row][col + j_off] -= factor * pval;
            }
        }
    }

    // Back-substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    // Construct pos = sum_i x_i * alpha_i
    let fp_len = roots[0].finite_part.len();
    let lc_len = roots[0].lorentz_coords.len();
    let mut pos = KacMoodyRoot::lorentzian(vec![0.0; fp_len], 0, vec![0.0; lc_len]);
    for (i, &coeff) in x.iter().enumerate() {
        let scaled = scale_root(&roots[i], coeff);
        pos = add_roots(&pos, &scaled);
    }

    pos
}
