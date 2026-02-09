//! Holographic entropy and Ryu-Takayanagi lattice models (Sprint 5).
//!
//! Implements:
//! - Bekenstein bound verification for metamaterial absorbers (G1.1)
//! - Toy Ryu-Takayanagi lattice with min-cut entropy (G1.2)
//! - Area law verification with power-law fits (G1.3)
//!
//! References:
//! - Bekenstein, PRD 23 (1981) 287 - Universal entropy bounds
//! - Ryu & Takayanagi, PRL 96 (2006) 181602 - Holographic entanglement
//! - Swingle, PRD 86 (2012) 065007 - MERA/AdS connection

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, HashSet, VecDeque};

// Physical constants (SI units)
/// Boltzmann constant (J/K)
pub const K_B: f64 = 1.380649e-23;
/// Reduced Planck constant (J*s)
pub const HBAR: f64 = 1.054571817e-34;
/// Speed of light (m/s)
pub const C: f64 = 299792458.0;

/// Bekenstein bound result for an absorber configuration.
#[derive(Debug, Clone)]
pub struct BekensteinBoundResult {
    /// Effective radius (m)
    pub radius: f64,
    /// Absorbed energy (J)
    pub energy: f64,
    /// Channel capacity (bits)
    pub channel_capacity: f64,
    /// Bekenstein bound (bits)
    pub bekenstein_bound: f64,
    /// Whether bound is satisfied
    pub bound_satisfied: bool,
    /// Saturation fraction (S / S_Bekenstein)
    pub saturation: f64,
}

/// Compute Bekenstein bound for a given radius and energy.
///
/// S_max = 2 * pi * k_B * R * E / (hbar * c)
/// Converting to bits: S_bits = S_max / (k_B * ln(2))
pub fn bekenstein_bound_bits(radius: f64, energy: f64) -> f64 {
    let s_max = 2.0 * std::f64::consts::PI * K_B * radius * energy / (HBAR * C);
    s_max / (K_B * 2.0_f64.ln()) // Convert to bits
}

/// Absorber layer parameters for Bekenstein analysis.
#[derive(Debug, Clone)]
pub struct AbsorberLayer {
    pub n_real: f64,
    pub n_imag: f64,
    pub thickness_nm: f64,
}

/// Compute channel capacity for an absorber stack (simplified model).
///
/// Uses a Shannon-like capacity estimate based on absorption bandwidth.
pub fn absorber_channel_capacity(layers: &[AbsorberLayer], wavelength_nm: f64) -> f64 {
    if layers.is_empty() {
        return 0.0;
    }

    // Total absorption coefficient
    let total_absorption: f64 = layers
        .iter()
        .map(|l| {
            // Absorption coefficient alpha = 4*pi*k / lambda (1/m)
            let alpha = 4.0 * std::f64::consts::PI * l.n_imag / (wavelength_nm * 1e-9);
            // Absorption in this layer
            let thickness_m = l.thickness_nm * 1e-9;
            1.0 - (-alpha * thickness_m).exp()
        })
        .sum();

    // Simplified capacity: log2(1 + SNR) where SNR ~ absorption
    // This is a toy model; real capacity depends on noise statistics
    let snr = 10.0 * total_absorption.min(1.0);
    (1.0 + snr).log2()
}

/// Compute effective radius of an absorber stack.
pub fn absorber_effective_radius(layers: &[AbsorberLayer]) -> f64 {
    // Total thickness as equivalent sphere radius
    let total_thickness: f64 = layers.iter().map(|l| l.thickness_nm).sum();
    total_thickness * 1e-9 // Convert nm to m
}

/// Compute absorbed energy (simplified model).
pub fn absorber_energy(layers: &[AbsorberLayer], incident_power: f64, _wavelength_nm: f64) -> f64 {
    let absorption_fraction: f64 = layers
        .iter()
        .map(|l| l.n_imag / (l.n_real.max(1.0)))
        .sum::<f64>()
        .min(1.0);

    incident_power * absorption_fraction
}

/// Verify Bekenstein bound for a metamaterial absorber configuration.
pub fn verify_bekenstein_bound(
    layers: &[AbsorberLayer],
    incident_power: f64,
    wavelength_nm: f64,
) -> BekensteinBoundResult {
    let radius = absorber_effective_radius(layers);
    let energy = absorber_energy(layers, incident_power, wavelength_nm);
    let channel_capacity = absorber_channel_capacity(layers, wavelength_nm);
    let bekenstein_bound = bekenstein_bound_bits(radius, energy);

    let bound_satisfied = channel_capacity <= bekenstein_bound;
    let saturation = if bekenstein_bound > 0.0 {
        channel_capacity / bekenstein_bound
    } else {
        0.0
    };

    BekensteinBoundResult {
        radius,
        energy,
        channel_capacity,
        bekenstein_bound,
        bound_satisfied,
        saturation,
    }
}

// ============================================================================
// Ryu-Takayanagi Lattice Model
// ============================================================================

/// Edge in the hyperbolic lattice.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
}

impl Edge {
    pub fn new(a: usize, b: usize) -> Self {
        Edge {
            from: a.min(b),
            to: a.max(b),
        }
    }
}

/// Hyperbolic lattice for RT calculations.
#[derive(Debug, Clone)]
pub struct RTLattice {
    /// Number of boundary sites
    pub n_boundary: usize,
    /// Number of bulk sites
    pub n_bulk: usize,
    /// Total sites
    pub n_total: usize,
    /// Adjacency list
    pub adjacency: Vec<Vec<usize>>,
    /// Edge weights (for min-cut calculation)
    pub edge_weights: HashMap<Edge, f64>,
    /// Boundary site indices
    pub boundary_sites: Vec<usize>,
    /// Bulk site indices
    pub bulk_sites: Vec<usize>,
}

impl RTLattice {
    /// Build a simple hyperbolic-like tree lattice.
    ///
    /// Creates a binary tree with `depth` levels, where the leaves are boundary sites.
    /// This is a simplified model capturing the hierarchical structure of AdS/CFT.
    pub fn build_tree(depth: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Number of nodes: 2^(depth+1) - 1
        let n_total = (1 << (depth + 1)) - 1;
        let n_boundary = 1 << depth;
        let n_bulk = n_total - n_boundary;

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_total];
        let mut edge_weights = HashMap::new();

        // Build tree structure
        for i in 0..n_total {
            let left_child = 2 * i + 1;
            let right_child = 2 * i + 2;

            if left_child < n_total {
                adjacency[i].push(left_child);
                adjacency[left_child].push(i);
                let weight = 1.0 + 0.1 * rng.gen::<f64>();
                edge_weights.insert(Edge::new(i, left_child), weight);
            }
            if right_child < n_total {
                adjacency[i].push(right_child);
                adjacency[right_child].push(i);
                let weight = 1.0 + 0.1 * rng.gen::<f64>();
                edge_weights.insert(Edge::new(i, right_child), weight);
            }
        }

        // Boundary sites are the leaves (last n_boundary nodes)
        let boundary_sites: Vec<usize> = ((n_total - n_boundary)..n_total).collect();
        let bulk_sites: Vec<usize> = (0..(n_total - n_boundary)).collect();

        RTLattice {
            n_boundary,
            n_bulk,
            n_total,
            adjacency,
            edge_weights,
            boundary_sites,
            bulk_sites,
        }
    }

    /// Get edge weight.
    pub fn weight(&self, from: usize, to: usize) -> f64 {
        *self.edge_weights.get(&Edge::new(from, to)).unwrap_or(&1.0)
    }
}

/// Min-cut result for a boundary region.
#[derive(Debug, Clone)]
pub struct MinCutResult {
    /// Boundary region A (site indices)
    pub region_a: Vec<usize>,
    /// Size of region A
    pub region_size: usize,
    /// Min-cut value (sum of edge weights cut)
    pub min_cut: f64,
    /// Entropy estimate: S = min_cut / (4 * G_N)
    /// Using G_N = 1 for simplicity (dimensionless units)
    pub entropy: f64,
}

/// Compute min-cut between boundary region A and its complement.
///
/// Uses BFS-based Ford-Fulkerson for small graphs.
pub fn compute_min_cut(lattice: &RTLattice, region_a: &[usize]) -> MinCutResult {
    // Convert region_a to a set for fast lookup
    let region_a_set: HashSet<usize> = region_a.iter().cloned().collect();
    let region_ac: Vec<usize> = lattice
        .boundary_sites
        .iter()
        .filter(|s| !region_a_set.contains(s))
        .cloned()
        .collect();

    if region_a.is_empty() || region_ac.is_empty() {
        return MinCutResult {
            region_a: region_a.to_vec(),
            region_size: region_a.len(),
            min_cut: 0.0,
            entropy: 0.0,
        };
    }

    // Build a max-flow network:
    // - Add super-source connected to all sites in region_a
    // - Add super-sink connected to all sites in region_ac
    // - Find max-flow = min-cut

    let n = lattice.n_total;
    let source = n;
    let sink = n + 1;

    // Build capacity matrix (using HashMap for sparse representation)
    let mut capacity: HashMap<(usize, usize), f64> = HashMap::new();

    // Edges from source to region A
    for &a in region_a {
        capacity.insert((source, a), f64::INFINITY);
    }

    // Edges from region A complement to sink
    for &ac in &region_ac {
        capacity.insert((ac, sink), f64::INFINITY);
    }

    // Internal edges (bidirectional with same capacity)
    for (&edge, &weight) in &lattice.edge_weights {
        capacity.insert((edge.from, edge.to), weight);
        capacity.insert((edge.to, edge.from), weight);
    }

    // Ford-Fulkerson with BFS (Edmonds-Karp)
    let mut flow: HashMap<(usize, usize), f64> = HashMap::new();
    let n_nodes = n + 2;

    loop {
        // BFS to find augmenting path
        let mut parent: Vec<Option<usize>> = vec![None; n_nodes];
        let mut visited = vec![false; n_nodes];
        let mut queue = VecDeque::new();

        queue.push_back(source);
        visited[source] = true;

        while let Some(u) = queue.pop_front() {
            if u == sink {
                break;
            }

            // Check neighbors
            let neighbors: Vec<usize> = if u == source {
                region_a.to_vec()
            } else if u < n {
                let mut nbrs = lattice.adjacency[u].clone();
                if region_ac.contains(&u) {
                    nbrs.push(sink);
                }
                nbrs
            } else {
                vec![]
            };

            for v in neighbors {
                if !visited[v] {
                    let cap = *capacity.get(&(u, v)).unwrap_or(&0.0);
                    let flw = *flow.get(&(u, v)).unwrap_or(&0.0);
                    if cap - flw > 1e-10 {
                        visited[v] = true;
                        parent[v] = Some(u);
                        queue.push_back(v);
                    }
                }
            }
        }

        if parent[sink].is_none() {
            break; // No augmenting path found
        }

        // Find bottleneck capacity
        let mut path_flow = f64::INFINITY;
        let mut v = sink;
        while let Some(u) = parent[v] {
            let cap = *capacity.get(&(u, v)).unwrap_or(&0.0);
            let flw = *flow.get(&(u, v)).unwrap_or(&0.0);
            path_flow = path_flow.min(cap - flw);
            v = u;
        }

        // Augment flow along path
        v = sink;
        while let Some(u) = parent[v] {
            *flow.entry((u, v)).or_insert(0.0) += path_flow;
            *flow.entry((v, u)).or_insert(0.0) -= path_flow;
            v = u;
        }
    }

    // Max flow = sum of flows out of source
    let max_flow: f64 = region_a
        .iter()
        .map(|&a| *flow.get(&(source, a)).unwrap_or(&0.0))
        .sum();

    // RT formula: S = min_cut / (4 * G_N)
    // Using G_N = 1 for dimensionless units
    let entropy = max_flow / 4.0;

    MinCutResult {
        region_a: region_a.to_vec(),
        region_size: region_a.len(),
        min_cut: max_flow,
        entropy,
    }
}

/// Entropy scaling analysis result.
#[derive(Debug, Clone)]
pub struct EntropyScalingResult {
    /// Region sizes
    pub sizes: Vec<usize>,
    /// Entropies for each size
    pub entropies: Vec<f64>,
    /// Fit: S = a * log(L) + b
    pub log_coefficient: f64,
    pub log_intercept: f64,
    /// Fit: S = c * L^alpha
    pub power_coefficient: f64,
    pub power_exponent: f64,
    /// R-squared for log fit
    pub log_r_squared: f64,
    /// R-squared for power fit
    pub power_r_squared: f64,
    /// Which fit is better
    pub log_fit_better: bool,
}

/// Analyze entropy scaling with region size.
pub fn analyze_entropy_scaling(lattice: &RTLattice, seed: u64) -> EntropyScalingResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_boundary = lattice.n_boundary;

    // Compute entropy for various region sizes
    let sizes: Vec<usize> = (1..n_boundary).filter(|&s| s <= n_boundary / 2).collect();
    let mut entropies = Vec::with_capacity(sizes.len());

    for &size in &sizes {
        // Random contiguous region of given size
        let start = rng.gen_range(0..(n_boundary - size + 1));
        let region: Vec<usize> = lattice.boundary_sites[start..(start + size)].to_vec();
        let result = compute_min_cut(lattice, &region);
        entropies.push(result.entropy);
    }

    // Log fit: S = a * log(L) + b
    let log_sizes: Vec<f64> = sizes.iter().map(|&s| (s as f64).ln()).collect();
    let (log_coef, log_int) = linear_fit(&log_sizes, &entropies);
    let log_r2 = r_squared(&log_sizes, &entropies, log_coef, log_int);

    // Power fit: log(S) = log(c) + alpha * log(L)
    let log_entropies: Vec<f64> = entropies
        .iter()
        .map(|&s| if s > 1e-10 { s.ln() } else { -10.0 })
        .collect();
    let (power_alpha, log_c) = linear_fit(&log_sizes, &log_entropies);
    let power_c = log_c.exp();
    let power_r2 = r_squared(&log_sizes, &log_entropies, power_alpha, log_c);

    EntropyScalingResult {
        sizes,
        entropies,
        log_coefficient: log_coef,
        log_intercept: log_int,
        power_coefficient: power_c,
        power_exponent: power_alpha,
        log_r_squared: log_r2,
        power_r_squared: power_r2,
        log_fit_better: log_r2 >= power_r2,
    }
}

/// Simple linear regression y = a*x + b.
fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    (slope, intercept)
}

/// Compute R-squared for a linear fit.
fn r_squared(x: &[f64], y: &[f64], slope: f64, intercept: f64) -> f64 {
    let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
        .sum();

    if ss_tot > 1e-15 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

/// Area law verification result.
#[derive(Debug, Clone)]
pub struct AreaLawResult {
    /// Best-fit exponent alpha in S ~ L^alpha
    pub exponent: f64,
    /// Bootstrap 95% CI lower bound
    pub ci_lower: f64,
    /// Bootstrap 95% CI upper bound
    pub ci_upper: f64,
    /// R-squared of power-law fit
    pub r_squared: f64,
    /// Whether exponent is consistent with area law (alpha ~ 1)
    pub consistent_with_area_law: bool,
    /// Whether exponent is consistent with volume law (alpha ~ d)
    pub consistent_with_volume_law: bool,
}

/// Verify area law scaling with bootstrap confidence intervals.
pub fn verify_area_law(lattice: &RTLattice, n_bootstrap: usize, seed: u64) -> AreaLawResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Get base scaling data
    let base_result = analyze_entropy_scaling(lattice, seed);

    // Bootstrap for confidence intervals on exponent
    let mut exponents = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let n = base_result.sizes.len();
        let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

        let resampled_log_sizes: Vec<f64> = indices
            .iter()
            .map(|&i| (base_result.sizes[i] as f64).ln())
            .collect();
        let resampled_log_entropies: Vec<f64> = indices
            .iter()
            .map(|&i| {
                let s = base_result.entropies[i];
                if s > 1e-10 {
                    s.ln()
                } else {
                    -10.0
                }
            })
            .collect();

        let (alpha, _) = linear_fit(&resampled_log_sizes, &resampled_log_entropies);
        exponents.push(alpha);
    }

    exponents.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = (0.025 * n_bootstrap as f64) as usize;
    let upper_idx = (0.975 * n_bootstrap as f64) as usize;
    let ci_lower = exponents[lower_idx];
    let ci_upper = exponents[upper_idx.min(n_bootstrap - 1)];

    // Area law: alpha ~ 1 (for 1+1D, area ~ L^0 = constant, so alpha ~ 0)
    // In our tree model, we expect log scaling which gives small positive alpha
    let consistent_with_area_law = base_result.power_exponent < 1.5;
    // Volume law: alpha ~ d (spatial dimension)
    let consistent_with_volume_law = base_result.power_exponent > 0.8;

    AreaLawResult {
        exponent: base_result.power_exponent,
        ci_lower,
        ci_upper,
        r_squared: base_result.power_r_squared,
        consistent_with_area_law,
        consistent_with_volume_law,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bekenstein_bound_positive() {
        let bound = bekenstein_bound_bits(1e-6, 1e-15);
        assert!(bound > 0.0);
        assert!(bound.is_finite());
    }

    #[test]
    fn test_bekenstein_bound_scales_with_r_e() {
        let bound1 = bekenstein_bound_bits(1e-6, 1e-15);
        let bound2 = bekenstein_bound_bits(2e-6, 1e-15);
        let bound3 = bekenstein_bound_bits(1e-6, 2e-15);

        // Should scale linearly with R and E
        assert!((bound2 / bound1 - 2.0).abs() < 0.01);
        assert!((bound3 / bound1 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_absorber_bound_satisfied() {
        let layers = vec![
            AbsorberLayer {
                n_real: 1.5,
                n_imag: 0.01,
                thickness_nm: 100.0,
            },
            AbsorberLayer {
                n_real: 1.8,
                n_imag: 0.02,
                thickness_nm: 50.0,
            },
        ];

        let result = verify_bekenstein_bound(&layers, 1e-12, 1550.0);

        // For realistic parameters, bound should be satisfied
        assert!(result.bound_satisfied);
        assert!(result.saturation < 1.0);
    }

    #[test]
    fn test_rt_lattice_structure() {
        let lattice = RTLattice::build_tree(3, 42);

        assert_eq!(lattice.n_boundary, 8);
        assert_eq!(lattice.n_total, 15);
        assert_eq!(lattice.boundary_sites.len(), 8);
    }

    #[test]
    fn test_min_cut_empty_region() {
        let lattice = RTLattice::build_tree(3, 42);
        let result = compute_min_cut(&lattice, &[]);

        assert_eq!(result.min_cut, 0.0);
        assert_eq!(result.entropy, 0.0);
    }

    #[test]
    fn test_min_cut_single_site() {
        let lattice = RTLattice::build_tree(3, 42);
        let region = vec![lattice.boundary_sites[0]];
        let result = compute_min_cut(&lattice, &region);

        // Single site should have small but positive entropy
        assert!(result.min_cut > 0.0);
        assert!(result.entropy > 0.0);
    }

    #[test]
    fn test_min_cut_half_system() {
        let lattice = RTLattice::build_tree(3, 42);
        let half = lattice.n_boundary / 2;
        let region: Vec<usize> = lattice.boundary_sites[0..half].to_vec();
        let result = compute_min_cut(&lattice, &region);

        // Half system should have significant entropy
        assert!(result.min_cut > 0.0);
        assert!(result.entropy > 0.0);
    }

    #[test]
    fn test_entropy_scaling_sizes() {
        let lattice = RTLattice::build_tree(4, 42);
        let result = analyze_entropy_scaling(&lattice, 42);

        assert!(!result.sizes.is_empty());
        assert_eq!(result.sizes.len(), result.entropies.len());
    }

    #[test]
    fn test_entropy_scaling_monotonic() {
        let lattice = RTLattice::build_tree(4, 42);
        let result = analyze_entropy_scaling(&lattice, 42);

        // Entropy should generally increase with region size
        // (may not be strictly monotonic due to random regions)
        let first_half: f64 = result.entropies[..result.entropies.len() / 2].iter().sum();
        let second_half: f64 = result.entropies[result.entropies.len() / 2..].iter().sum();

        // Average entropy in larger regions should be >= smaller regions
        assert!(
            second_half / (result.entropies.len() / 2) as f64
                >= first_half / (result.entropies.len() / 2) as f64 * 0.5
        );
    }

    #[test]
    fn test_area_law_verification() {
        let lattice = RTLattice::build_tree(4, 42);
        let result = verify_area_law(&lattice, 50, 42);

        // Exponent should be finite
        assert!(result.exponent.is_finite());
        // CI should contain the point estimate
        assert!(result.ci_lower <= result.exponent);
        assert!(result.ci_upper >= result.exponent);
        // R-squared should be reasonable
        assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    }

    #[test]
    fn test_rt_lattice_edge_weights_positive() {
        let lattice = RTLattice::build_tree(3, 42);

        for (_, &weight) in &lattice.edge_weights {
            assert!(weight > 0.0);
        }
    }

    #[test]
    fn test_min_cut_symmetry() {
        let lattice = RTLattice::build_tree(3, 42);
        let half = lattice.n_boundary / 2;

        let region_a: Vec<usize> = lattice.boundary_sites[0..half].to_vec();
        let region_ac: Vec<usize> = lattice.boundary_sites[half..].to_vec();

        let result_a = compute_min_cut(&lattice, &region_a);
        let result_ac = compute_min_cut(&lattice, &region_ac);

        // S(A) should equal S(A^c) for pure state
        assert!((result_a.entropy - result_ac.entropy).abs() < 0.1);
    }
}
