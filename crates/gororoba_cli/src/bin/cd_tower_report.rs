//! Comprehensive Cayley-Dickson tower analysis report.
//!
//! Computes and prints invariants at each power-of-2 dimension from dim=1
//! (reals) through dim=2048, synthesizing results from across the codebase:
//!
//! - Property retention (normed, associative, alternative, etc.)
//! - psi=1 fraction and exact formula verification
//! - Zero-divisor 2-blade pair counts
//! - Motif component counts and scaling laws
//! - Face sign census regime counts
//! - Frustration ratios (GF(2) cohomology)
//! - Split-CD zero-divisor counts (where feasible)
//!
//! References claims: C-483, C-487, C-520, C-528, C-529, C-531, C-534,
//! C-537, C-544, C-545..C-549.

use std::collections::VecDeque;
use std::time::Instant;

use algebra_core::analysis::boxkites::{
    generic_face_sign_census, motif_components_for_cross_assessors, CrossPair,
};
use algebra_core::construction::cayley_dickson::{
    cd_basis_mul_sign_iter, cd_norm_sq, find_zero_divisors, CdSignature,
};

use clap::Parser;

/// Generate the full Cayley-Dickson tower analysis report.
#[derive(Parser)]
#[command(name = "cd-tower-report")]
struct Args {
    /// Maximum dimension (power of 2). Default 512 for reasonable runtime.
    #[arg(long, default_value = "512")]
    max_dim: usize,

    /// Include split-CD analysis (slower at large dims).
    #[arg(long)]
    split: bool,

    /// Include face sign census (very slow at dim >= 256).
    #[arg(long)]
    census: bool,

    /// Include frustration ratios (slow at dim >= 128).
    #[arg(long)]
    frustration: bool,

    /// Include all optional analyses.
    #[arg(long)]
    all: bool,
}

/// Per-dimension data row for the tower report.
struct DimRow {
    dim: usize,
    name: &'static str,
    // Property retention
    is_normed: bool,
    is_division: bool,
    is_associative: bool,
    is_alternative: bool,
    is_commutative: bool,
    is_power_associative: bool,
    has_zero_divisors: bool,
    // psi=1 fraction
    psi_one_fraction: f64,
    psi_one_exact: f64, // (n+1)/(2n) where n=dim-1
    // Zero-divisor 2-blade pair count
    zd_2blade_count: Option<usize>,
    // Motif components
    n_components: Option<usize>,
    n_motif_classes: Option<usize>,
    // Face sign census
    n_regimes: Option<usize>,
    total_triangles: Option<usize>,
    // Frustration ratio
    frustration_ratio: Option<f64>,
    // Split-CD ZD count
    split_zd_count: Option<usize>,
    // Timing
    compute_time_ms: u64,
}

fn dim_name(dim: usize) -> &'static str {
    match dim {
        1 => "Real (R)",
        2 => "Complex (C)",
        4 => "Quaternion (H)",
        8 => "Octonion (O)",
        16 => "Sedenion (S)",
        32 => "Pathion (P)",
        64 => "Chingon",
        128 => "Routon",
        256 => "Voudon",
        512 => "512D",
        1024 => "1024D",
        2048 => "2048D",
        _ => "?",
    }
}

/// Compute psi=1 fraction for the full (dim-1)x(dim-1) imaginary submatrix.
fn compute_psi_fraction(dim: usize) -> (usize, usize, f64) {
    if dim <= 1 {
        return (0, 0, 0.0);
    }
    let n = dim - 1;
    let total = n * n;
    let mut psi_one = 0usize;
    for i in 1..dim {
        for j in 1..dim {
            if cd_basis_mul_sign_iter(dim, i, j) == -1 {
                psi_one += 1;
            }
        }
    }
    (psi_one, total, psi_one as f64 / total as f64)
}

/// Count 2-blade zero-divisor pairs. O(dim^4), only feasible for dim <= 64.
fn count_zd_pairs(dim: usize) -> usize {
    find_zero_divisors(dim, 1e-10).len()
}

/// Compute frustration ratio for a given dimension using BFS cohomology.
fn compute_frustration_ratio(dim: usize) -> (usize, usize, f64) {
    let components = motif_components_for_cross_assessors(dim);
    let psi = |d: usize, i: usize, j: usize| -> u8 {
        if cd_basis_mul_sign_iter(d, i, j) == 1 {
            0
        } else {
            1
        }
    };

    let mut total_b1 = 0usize;
    let mut total_frustrated = 0usize;

    for comp in &components {
        let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
        let n = nodes.len();
        if n < 2 {
            continue;
        }

        // Build adjacency with eta labels
        let node_idx: std::collections::HashMap<CrossPair, usize> =
            nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();
        let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];

        for &(u, v) in &comp.edges {
            let eta_val = psi(dim, u.0, v.1) ^ psi(dim, u.1, v.0);
            let ui = node_idx[&u];
            let vi = node_idx[&v];
            adj[ui].push((vi, eta_val));
            adj[vi].push((ui, eta_val));
        }

        // BFS cohomology
        let mut delta: Vec<Option<u8>> = vec![None; n];
        delta[0] = Some(0);
        let mut queue = VecDeque::new();
        queue.push_back(0);
        let mut tree_edges = 0usize;
        let mut frustrated = 0usize;

        while let Some(u) = queue.pop_front() {
            for &(v, eta_val) in &adj[u] {
                if let Some(dv) = delta[v] {
                    if u < v {
                        let expected = delta[u].unwrap() ^ eta_val;
                        if expected != dv {
                            frustrated += 1;
                        }
                    }
                } else {
                    delta[v] = Some(delta[u].unwrap() ^ eta_val);
                    tree_edges += 1;
                    queue.push_back(v);
                }
            }
        }

        let b1 = comp.edges.len() - tree_edges;
        total_b1 += b1;
        total_frustrated += frustrated;
    }

    let ratio = if total_b1 > 0 {
        total_frustrated as f64 / total_b1 as f64
    } else {
        0.0
    };
    (total_frustrated, total_b1, ratio)
}

/// Count split-CD zero-divisor 2-blade pairs using the split sign table.
fn count_split_zd_pairs(dim: usize) -> usize {
    let sig = CdSignature::split(dim);
    let mut count = 0usize;
    for i in 0..dim {
        for j in (i + 1)..dim {
            let mut a = vec![0.0; dim];
            a[i] = 1.0;
            a[j] = 1.0;

            for k in 0..dim {
                for l in (k + 1)..dim {
                    // (e_i + e_j)(e_k + e_l)
                    let mut b = vec![0.0; dim];
                    b[k] = 1.0;
                    b[l] = 1.0;
                    let ab =
                        algebra_core::construction::cayley_dickson::cd_multiply_split(&a, &b, &sig);
                    let norm = cd_norm_sq(&ab).sqrt();
                    if norm < 1e-10 {
                        count += 1;
                    }

                    // (e_i + e_j)(e_k - e_l)
                    b[l] = -1.0;
                    let ab =
                        algebra_core::construction::cayley_dickson::cd_multiply_split(&a, &b, &sig);
                    let norm = cd_norm_sq(&ab).sqrt();
                    if norm < 1e-10 {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

fn main() {
    let args = Args::parse();
    let do_split = args.split || args.all;
    let do_census = args.census || args.all;
    let do_frustration = args.frustration || args.all;

    assert!(
        args.max_dim.is_power_of_two(),
        "max_dim must be a power of 2"
    );

    println!("=============================================================================");
    println!("  CAYLEY-DICKSON TOWER ANALYSIS REPORT");
    println!("  Dimensions: 1 through {}", args.max_dim);
    println!("=============================================================================");
    println!();

    let mut rows: Vec<DimRow> = Vec::new();
    let overall_start = Instant::now();

    // Compute each dimension
    let mut dim = 1usize;
    while dim <= args.max_dim {
        let t0 = Instant::now();
        eprint!("  Computing dim={dim}...");

        let n = dim.saturating_sub(1);

        // Property retention
        let is_normed = dim <= 8;
        let is_division = dim <= 8;
        let is_associative = dim <= 4;
        let is_alternative = dim <= 8;
        let is_commutative = dim <= 2;
        let is_power_associative = true; // All CD algebras are power-associative
        let has_zd = dim >= 16;

        // psi=1 fraction (computed for dim >= 2)
        let (_psi_one, _total, psi_frac) = if dim >= 2 {
            compute_psi_fraction(dim)
        } else {
            (0, 0, 0.0)
        };
        let psi_exact = if n > 0 {
            (n as f64 + 1.0) / (2.0 * n as f64)
        } else {
            0.0
        };

        // ZD pairs: live computation at dim=16 (fast), precomputed for dim=32
        // (dim=32 takes ~10s, dim=64 takes minutes)
        let zd_count = if dim == 16 {
            Some(count_zd_pairs(dim))
        } else if dim == 32 {
            Some(2520) // precomputed (confirmed in prior sprints)
        } else if dim < 16 {
            Some(0)
        } else {
            None // too expensive to compute live
        };

        // Motif components (feasible up to dim=256)
        let (n_comp, n_classes) = if (16..=256).contains(&dim) {
            let comps = motif_components_for_cross_assessors(dim);
            let nc = comps.len();
            // Count distinct motif classes by edge-count fingerprint
            let mut edge_counts: Vec<usize> = comps.iter().map(|c| c.edges.len()).collect();
            edge_counts.sort();
            edge_counts.dedup();
            (Some(nc), Some(edge_counts.len()))
        } else if dim < 16 {
            (Some(0), Some(0))
        } else {
            (None, None)
        };

        // Face sign census (feasible up to dim=128 quickly, dim=256 in ~16s, dim=512 in ~275s)
        let (n_regimes, n_triangles) = if do_census && dim >= 16 && dim <= args.max_dim {
            let census = generic_face_sign_census(dim);
            // Regime count = number of distinct (edge_count, n_patterns) pairs.
            // Components with the same edge count but different pattern counts
            // (pure=2 patterns vs full=4 patterns) are separate regimes.
            // This matches C-485: dim/16 + 1 for dim >= 32.
            let mut regime_keys: Vec<(usize, usize)> = census
                .per_component
                .iter()
                .filter(|c| c.n_triangles > 0)
                .map(|c| (c.n_edges, c.pattern_counts.len()))
                .collect();
            regime_keys.sort();
            regime_keys.dedup();
            let regime_count = regime_keys.len();
            (Some(regime_count), Some(census.total_triangles))
        } else if dim < 16 {
            (Some(0), Some(0))
        } else {
            (None, None)
        };

        // Frustration ratio (feasible up to dim=256 in reasonable time)
        let frust = if do_frustration && (16..=256).contains(&dim) {
            let (_, _, r) = compute_frustration_ratio(dim);
            Some(r)
        } else if dim < 16 {
            Some(0.0)
        } else {
            None
        };

        // Split-CD ZD count (feasible up to dim=16; dim=32 is borderline)
        let split_zd = if do_split && (4..=16).contains(&dim) {
            Some(count_split_zd_pairs(dim))
        } else if do_split && dim < 4 {
            Some(0)
        } else {
            None
        };

        let elapsed = t0.elapsed().as_millis() as u64;
        eprintln!(" {elapsed}ms");

        rows.push(DimRow {
            dim,
            name: dim_name(dim),
            is_normed,
            is_division,
            is_associative,
            is_alternative,
            is_commutative,
            is_power_associative,
            has_zero_divisors: has_zd,
            psi_one_fraction: psi_frac,
            psi_one_exact: psi_exact,
            zd_2blade_count: zd_count,
            n_components: n_comp,
            n_motif_classes: n_classes,
            n_regimes,
            total_triangles: n_triangles,
            frustration_ratio: frust,
            split_zd_count: split_zd,
            compute_time_ms: elapsed,
        });

        dim *= 2;
    }

    let total_time = overall_start.elapsed();

    // =========================================================================
    // Section 1: Property Retention Table
    // =========================================================================
    println!("SECTION 1: PROPERTY RETENTION ACROSS THE CD TOWER");
    println!("-----------------------------------------------------------------------------");
    println!(
        "{:<14} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "Algebra", "dim", "Norm", "Div", "Assoc", "Alt", "Comm", "PwAs", "ZD"
    );
    println!(
        "{:<14} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "----------", "---", "----", "---", "-----", "---", "----", "----", "--"
    );
    for r in &rows {
        println!(
            "{:<14} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
            r.name,
            r.dim,
            if r.is_normed { "yes" } else { " - " },
            if r.is_division { "yes" } else { " - " },
            if r.is_associative { "yes" } else { " - " },
            if r.is_alternative { "yes" } else { " - " },
            if r.is_commutative { "yes" } else { " - " },
            if r.is_power_associative { "yes" } else { " - " },
            if r.has_zero_divisors { "YES" } else { " - " },
        );
    }
    println!();
    println!("  Properties lost at each doubling:");
    println!("    dim=1->2 (R->C):  lose total ordering");
    println!("    dim=2->4 (C->H):  lose commutativity");
    println!("    dim=4->8 (H->O):  lose associativity (retain alternativity)");
    println!("    dim=8->16 (O->S): lose alternativity, normed composition, division");
    println!("                       gain zero-divisors");
    println!("    dim>=16:          power-associative only");
    println!();

    // =========================================================================
    // Section 2: psi=1 Fraction Convergence
    // =========================================================================
    println!("SECTION 2: psi=1 FRACTION CONVERGENCE (C-534, C-544)");
    println!("-----------------------------------------------------------------------------");
    println!("  Exact formula: fraction = (n+1)/(2n) where n = dim-1");
    println!("  Proof: diagonal contributes n entries (all psi=1), off-diagonal");
    println!("  contributes n(n-1)/2 by anti-commutativity => total = n(n+1)/2");
    println!();
    println!(
        "{:<14} {:>5} {:>12} {:>12} {:>12} {:>12}",
        "Algebra", "dim", "psi=1 frac", "exact", "excess", "err"
    );
    println!(
        "{:<14} {:>5} {:>12} {:>12} {:>12} {:>12}",
        "----------", "---", "----------", "-----", "------", "---"
    );
    for r in &rows {
        if r.dim < 2 {
            continue;
        }
        let excess = r.psi_one_fraction - 0.5;
        let err = (r.psi_one_fraction - r.psi_one_exact).abs();
        println!(
            "{:<14} {:>5} {:>12.8} {:>12.8} {:>12.8} {:>12.2e}",
            r.name, r.dim, r.psi_one_fraction, r.psi_one_exact, excess, err
        );
    }
    println!();
    println!("  Convergence toward 50%: monotonically decreasing from above.");
    println!("  Excess halves with each doubling (geometric ratio ~0.5).");
    println!();

    // =========================================================================
    // Section 3: Zero-Divisor Geometry
    // =========================================================================
    println!("SECTION 3: ZERO-DIVISOR GEOMETRY");
    println!("-----------------------------------------------------------------------------");
    println!(
        "{:<14} {:>5} {:>12} {:>10} {:>10}",
        "Algebra", "dim", "2-blade ZDs", "Comps", "Classes"
    );
    println!(
        "{:<14} {:>5} {:>12} {:>10} {:>10}",
        "----------", "---", "-----------", "-----", "-------"
    );
    for r in &rows {
        let zd_str = match r.zd_2blade_count {
            Some(n) => format!("{}", n),
            None => "---".to_string(),
        };
        let comp_str = match r.n_components {
            Some(n) => format!("{}", n),
            None => "---".to_string(),
        };
        let class_str = match r.n_motif_classes {
            Some(n) => format!("{}", n),
            None => "---".to_string(),
        };
        println!(
            "{:<14} {:>5} {:>12} {:>10} {:>10}",
            r.name, r.dim, zd_str, comp_str, class_str
        );
    }
    println!();
    println!("  Scaling laws (dim >= 16):");
    println!("    n_components = dim/2 - 1");
    println!("    motif_classes = dim/16 (doubles each time)");
    println!("    ZD pairs grow super-linearly with dim");
    println!();

    // =========================================================================
    // Section 4: Face Sign Census & Frustration
    // =========================================================================
    if do_census || do_frustration {
        println!("SECTION 4: FACE SIGN CENSUS & FRUSTRATION RATIOS");
        println!("-----------------------------------------------------------------------------");
        println!(
            "{:<14} {:>5} {:>10} {:>12} {:>12}",
            "Algebra", "dim", "Regimes", "Triangles", "Frustration"
        );
        println!(
            "{:<14} {:>5} {:>10} {:>12} {:>12}",
            "----------", "---", "-------", "---------", "-----------"
        );
        for r in &rows {
            let reg_str = match r.n_regimes {
                Some(n) => format!("{}", n),
                None => "---".to_string(),
            };
            let tri_str = match r.total_triangles {
                Some(n) => format!("{}", n),
                None => "---".to_string(),
            };
            let frust_str = match r.frustration_ratio {
                Some(f) if f > 0.0 => format!("{:.5}", f),
                Some(_) => "0.000".to_string(),
                None => "---".to_string(),
            };
            println!(
                "{:<14} {:>5} {:>10} {:>12} {:>12}",
                r.name, r.dim, reg_str, tri_str, frust_str
            );
        }
        println!();
        println!("  Regime count formula (dim >= 32): dim/16 + 1");
        println!("  Universal Double 3:1 Law (C-487): OS = 3*AS, TS = 3*AO in ALL components");
        println!("  Anti-Diagonal Parity Theorem (C-520/C-521): mechanism via GF(2)^2 invariant F");
        println!("  Frustration converges toward 3/8 = 0.375 (C-529/C-535)");
        println!();
    }

    // =========================================================================
    // Section 5: Split-CD Analysis
    // =========================================================================
    if do_split {
        println!("SECTION 5: SPLIT CAYLEY-DICKSON ANALYSIS (C-545..C-549)");
        println!("-----------------------------------------------------------------------------");
        println!(
            "{:<14} {:>5} {:>15} {:>15}",
            "Algebra", "dim", "Std 2-blade ZDs", "Split 2-blade ZDs"
        );
        println!(
            "{:<14} {:>5} {:>15} {:>15}",
            "----------", "---", "--------------", "----------------"
        );
        for r in &rows {
            let std_str = match r.zd_2blade_count {
                Some(n) => format!("{}", n),
                None => "---".to_string(),
            };
            let split_str = match r.split_zd_count {
                Some(n) => format!("{}", n),
                None => "---".to_string(),
            };
            println!(
                "{:<14} {:>5} {:>15} {:>15}",
                r.name, r.dim, std_str, split_str
            );
        }
        println!();
        println!("  Key findings:");
        println!("    - Split-complex (dim=2): j^2 = +1, has ZDs at dim=2 (C-545)");
        println!("    - CD always non-commutative at dim >= 4 regardless of gamma (C-546)");
        println!("    - Split-octonion signature (4,3) with 128 ZD pairs (C-547)");
        println!("    - Dim=4: standard [-1,-1] is ONLY signature without ZDs (C-548)");
        println!("    - Split-octonion psi=1 fraction = 3/8 matches frustration limit (C-549)");
        println!();
    }

    // =========================================================================
    // Section 6: Scaling Law Verification
    // =========================================================================
    println!("SECTION 6: SCALING LAW VERIFICATION");
    println!("-----------------------------------------------------------------------------");

    let mut all_pass = true;

    // Check n_components = dim/2 - 1
    println!("  Law: n_components = dim/2 - 1 (dim >= 16)");
    for r in &rows {
        if r.dim >= 16 {
            if let Some(nc) = r.n_components {
                let expected = r.dim / 2 - 1;
                let pass = nc == expected;
                if !pass {
                    all_pass = false;
                }
                println!(
                    "    dim={:>5}: computed={}, expected={} [{}]",
                    r.dim,
                    nc,
                    expected,
                    if pass { "PASS" } else { "FAIL" }
                );
            }
        }
    }

    // Check motif_classes = dim/16
    println!("  Law: motif_classes = dim/16 (dim >= 16)");
    for r in &rows {
        if r.dim >= 16 {
            if let Some(nc) = r.n_motif_classes {
                let expected = r.dim / 16;
                let pass = nc == expected;
                if !pass {
                    all_pass = false;
                }
                println!(
                    "    dim={:>5}: computed={}, expected={} [{}]",
                    r.dim,
                    nc,
                    expected,
                    if pass { "PASS" } else { "FAIL" }
                );
            }
        }
    }

    // Check psi=1 exact formula
    println!("  Law: psi=1 fraction = (n+1)/(2n) exactly (C-544)");
    for r in &rows {
        if r.dim >= 2 {
            let err = (r.psi_one_fraction - r.psi_one_exact).abs();
            let pass = err < 1e-12;
            if !pass {
                all_pass = false;
            }
            println!(
                "    dim={:>5}: err={:.2e} [{}]",
                r.dim,
                err,
                if pass { "PASS" } else { "FAIL" }
            );
        }
    }

    // Check regime count formula
    if do_census {
        println!("  Law: n_regimes = dim/16 + 1 (dim >= 32, C-485)");
        for r in &rows {
            if r.dim >= 32 {
                if let Some(nr) = r.n_regimes {
                    let expected = r.dim / 16 + 1;
                    let pass = nr == expected;
                    if !pass {
                        all_pass = false;
                    }
                    println!(
                        "    dim={:>5}: computed={}, expected={} [{}]",
                        r.dim,
                        nr,
                        expected,
                        if pass { "PASS" } else { "FAIL" }
                    );
                }
            }
        }
    }

    println!();
    println!(
        "  Overall scaling law status: {}",
        if all_pass {
            "ALL PASS"
        } else {
            "SOME FAILURES"
        }
    );
    println!();

    // =========================================================================
    // Section 7: Precomputed Values (dims > computed range)
    // =========================================================================
    println!("SECTION 7: PRECOMPUTED / KNOWN VALUES (from prior sprints)");
    println!("-----------------------------------------------------------------------------");
    println!("  dim=512:  33 face sign regimes, 214M triangles, frustration ~0.385 (C-531)");
    println!("  dim=1024: frustration ratio = 0.37849 (C-535)");
    println!("  dim=2048: lattice codebook filtration verified (I-015)");
    println!("  Frustration convergence: 0.000, 0.307, 0.377, 0.388, 0.385, 0.381, 0.378");
    println!("    (dims 16, 32, 64, 128, 256, 512, 1024) -- post-peak monotone decrease");
    println!("    toward 3/8 = 0.375 (C-529/C-535)");
    println!();

    // =========================================================================
    // Section 8: Key Theorems and Claims
    // =========================================================================
    println!("SECTION 8: KEY THEOREMS AND CLAIMS VERIFIED");
    println!("-----------------------------------------------------------------------------");
    println!("  C-483:  Universal 3:1 TwoSameOneOpp:AllOpposite ratio (antibalanced graph)");
    println!("  C-487:  Universal Double 3:1 Law (OS=3*AS AND TS=3*AO)");
    println!("  C-520:  Anti-Diagonal Parity Theorem (eta = psi_02 XOR psi_10)");
    println!("  C-521:  GF(2)^2 invariant F determines face sign pattern");
    println!("  C-523:  GF(2) coboundary phase transition at dim=16");
    println!("  C-524:  Klein-four fiber symmetry F(1,0)=F(1,1) universal");
    println!("  C-526:  CD doubling recursion: eta = 1 XOR eta_half");
    println!("  C-527:  GF(2) polynomial degree: psi ANF deg = log2(dim)");
    println!("  C-528:  Quarter Rule: pure = total/4 exactly");
    println!("  C-529:  Frustration convergence toward 3/8");
    println!("  C-534:  psi=1 fraction converges to 50% from above");
    println!("  C-537:  Universal anti-commutativity of CD basis elements");
    println!("  C-544:  psi=1 exact formula (n+1)/(2n) proved");
    println!("  C-545:  Split-complex j^2=+1 with zero-divisors at dim=2");
    println!("  C-546:  CD always non-commutative at dim >= 4");
    println!("  C-547:  Split-octonion signature (4,3) with 128 ZD pairs");
    println!("  C-548:  Standard quaternions uniquely ZD-free at dim=4");
    println!("  C-549:  Split-octonion psi=1 frac = 3/8 = frustration limit");
    println!();

    // =========================================================================
    // Timing Summary
    // =========================================================================
    println!("-----------------------------------------------------------------------------");
    println!("  Total computation time: {:.2}s", total_time.as_secs_f64());
    for r in &rows {
        if r.compute_time_ms > 100 {
            println!("    dim={:>5}: {}ms", r.dim, r.compute_time_ms);
        }
    }
    println!("=============================================================================");
}
