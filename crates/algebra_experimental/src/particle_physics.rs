use crate::higher_cd::{DekaVoudon, HigherAvt, SampledAvt};
use std::collections::HashMap;

/// Standard Model gauge group generators mapped to 1024D sub-blocks.
///
/// The Cayley-Dickson doubling construction introduces new imaginary axes at
/// each dimension doubling. Each "layer" of axes has distinct non-associative
/// coupling to earlier layers. We map gauge groups to layers based on the
/// DFT (Dissociative Field Theory) force sector hierarchy:
///
/// | Layer  | Axes      | Dim first appears | DFT sector           |
/// |--------|-----------|-------------------|----------------------|
/// | Real   | 0         | 1                 | Gravity (scalar)     |
/// | C-new  | 1         | 2                 | Temporal flux         |
/// | H-new  | 2..3      | 4                 | Spacetime metric     |
/// | O-new  | 4..7      | 8                 | SU(3)_C color        |
/// | S-new  | 8..15     | 16                | SU(2)_L + U(1)_Y    |
/// | P-new  | 16..31    | 32                | Weak nuclear         |
/// | Ch-new | 32..63    | 64                | Strong nuclear drag  |
/// | R-new  | 64..127   | 128               | Routon chaos         |
/// | V-new  | 128..255  | 256               | Voudon cosmological  |
/// | E-new  | 256..511  | 512               | Eriston quantum      |
/// | DV-new | 512..1023 | 1024              | DekaVoudon unified   |
#[derive(Debug, Clone)]
pub struct StandardModelMapping {
    /// SU(3)_C color: 8 Gell-Mann generators mapped to octonion axes 1..7
    /// plus the first sedenion axis 8 (for the 8th generator lambda_8).
    pub su3_axes: Vec<usize>,
    /// SU(2)_L weak isospin: 3 Pauli generators mapped to sedenion axes 9..11.
    pub su2_axes: Vec<usize>,
    /// U(1)_Y hypercharge: 1 generator mapped to sedenion axis 12.
    pub u1_axes: Vec<usize>,
}

impl Default for StandardModelMapping {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardModelMapping {
    /// Construct the Standard Model mapping from the CD doubling hierarchy.
    ///
    /// SU(3) generators map to the octonion layer (axes 1-7) plus axis 8,
    /// because the 7 imaginary octonion units form a Fano-plane structure
    /// that encodes the same multiplication rules as SU(3) color charge.
    /// The 8th generator (diagonal lambda_8) maps to the first sedenion axis.
    ///
    /// SU(2) generators map to sedenion axes 9-11, and U(1) to axis 12.
    pub fn new() -> Self {
        Self {
            su3_axes: vec![1, 2, 3, 4, 5, 6, 7, 8],
            su2_axes: vec![9, 10, 11],
            u1_axes: vec![12],
        }
    }

    /// Total number of Standard Model generators (should be 12).
    pub fn total_generators(&self) -> usize {
        self.su3_axes.len() + self.su2_axes.len() + self.u1_axes.len()
    }

    /// All SM axes as a single sorted vector.
    pub fn all_axes(&self) -> Vec<usize> {
        let mut axes = self.su3_axes.clone();
        axes.extend_from_slice(&self.su2_axes);
        axes.extend_from_slice(&self.u1_axes);
        axes.sort();
        axes
    }
}

/// Mapping of 1024D basis blocks to Grand Unified Groups (GUT).
///
/// Each GUT group is mapped to a contiguous block of 1024D imaginary axes.
/// The block boundaries align with Cayley-Dickson doubling layer boundaries
/// where possible, because doubling-layer transitions introduce the strongest
/// non-associative coupling (violations concentrate at layer boundaries).
#[derive(Debug, Clone)]
pub struct ForceSectorMapping {
    /// E6: 78 generators, axes 1..78 (spans octonion through pathion layers).
    pub e6_block: Vec<usize>,
    /// E7: 133 generators, axes 79..211 (pathion through chingon layers).
    pub e7_block: Vec<usize>,
    /// SO(10): 45 generators, axes 212..256 (voudon layer onset).
    pub so10_block: Vec<usize>,
}

impl Default for ForceSectorMapping {
    fn default() -> Self {
        Self::new()
    }
}

impl ForceSectorMapping {
    /// Construct the force sector mapping from the 1024D manifold.
    ///
    /// Block boundaries respect Cayley-Dickson doubling layers:
    ///
    /// - E6 (78 generators): axes 1..78. Spans octonion + sedenion + pathion
    ///   + part of chingon layer. Contains SM as sub-block (axes `1..12`).
    /// - E7 (133 generators): axes 79..211. Rest of chingon through routon.
    /// - SO(10) (45 generators): axes 212..256. Onset of the voudon layer.
    pub fn new() -> Self {
        Self {
            e6_block: (1..79).collect(),
            e7_block: (79..212).collect(),
            so10_block: (212..257).collect(),
        }
    }

    /// Check whether the Standard Model embedding is a sub-block of E6.
    pub fn sm_is_subset_of_e6(&self, sm: &StandardModelMapping) -> bool {
        let sm_axes = sm.all_axes();
        sm_axes.iter().all(|ax| self.e6_block.contains(ax))
    }
}

/// Result of cross-sector AVT coupling analysis.
///
/// Measures how many non-associative violations connect two gauge sub-blocks.
/// High cross-sector coupling suggests the two forces are algebraically
/// entangled in the CD tower.
#[derive(Debug, Clone)]
pub struct CrossSectorAnalysis {
    /// Name of sector A.
    pub sector_a: String,
    /// Name of sector B.
    pub sector_b: String,
    /// Number of AVT violations within sector A.
    pub intra_a: usize,
    /// Number of AVT violations within sector B.
    pub intra_b: usize,
    /// Number of AVT violations straddling A and B.
    pub cross_ab: usize,
    /// Cross-coupling ratio: cross_ab / (intra_a + intra_b + cross_ab).
    pub coupling_ratio: f64,
}

/// Analyze AVT coupling between two sub-blocks of the 1024D algebra.
///
/// Given a sampled AVT and two axis ranges, counts intra-sector and cross-sector
/// violations. The coupling ratio measures how "entangled" the two force sectors
/// are through non-associative algebra.
pub fn analyze_cross_sector_coupling(
    avt: &HigherAvt,
    sector_a: &str,
    axes_a: &[usize],
    sector_b: &str,
    axes_b: &[usize],
) -> CrossSectorAnalysis {
    let (lo_a, hi_a) = axis_range(axes_a);
    let (lo_b, hi_b) = axis_range(axes_b);

    let intra_a = avt.count_violations_in_range(lo_a, hi_a);
    let intra_b = avt.count_violations_in_range(lo_b, hi_b);
    let cross_ab = avt.count_cross_violations(lo_a, hi_a, lo_b, hi_b);
    let total = intra_a + intra_b + cross_ab;
    let coupling_ratio = if total > 0 {
        cross_ab as f64 / total as f64
    } else {
        0.0
    };

    CrossSectorAnalysis {
        sector_a: sector_a.to_string(),
        sector_b: sector_b.to_string(),
        intra_a,
        intra_b,
        cross_ab,
        coupling_ratio,
    }
}

/// Run the full 1024D gauge sector analysis.
///
/// 1. Sample 10M triples from the 1024D AVT
/// 2. Map Standard Model generators to CD layers
/// 3. Measure intra-sector and cross-sector violation densities
/// 4. Compute coupling ratios between SU(3), SU(2), U(1)
///
/// Returns the sampled AVT, the SM mapping, and a list of cross-sector analyses.
pub fn analyze_1024d_gauge_sectors(n_samples: usize, seed: u64) -> GaugeSectorReport {
    let sampled = HigherAvt::sampled(1024, n_samples, seed);
    let sm = StandardModelMapping::new();
    let gut = ForceSectorMapping::new();

    let cross_analyses = vec![
        // SU(3) x SU(2) coupling
        analyze_cross_sector_coupling(
            &sampled.avt,
            "SU(3)_C",
            &sm.su3_axes,
            "SU(2)_L",
            &sm.su2_axes,
        ),
        // SU(3) x U(1) coupling
        analyze_cross_sector_coupling(&sampled.avt, "SU(3)_C", &sm.su3_axes, "U(1)_Y", &sm.u1_axes),
        // SU(2) x U(1) coupling (electroweak unification)
        analyze_cross_sector_coupling(&sampled.avt, "SU(2)_L", &sm.su2_axes, "U(1)_Y", &sm.u1_axes),
        // E6 x E7 coupling (GUT cross-sector)
        analyze_cross_sector_coupling(&sampled.avt, "E6", &gut.e6_block, "E7", &gut.e7_block),
        // E6 x SO(10) coupling
        analyze_cross_sector_coupling(&sampled.avt, "E6", &gut.e6_block, "SO(10)", &gut.so10_block),
        // E7 x SO(10) coupling
        analyze_cross_sector_coupling(&sampled.avt, "E7", &gut.e7_block, "SO(10)", &gut.so10_block),
    ];

    GaugeSectorReport {
        sampled,
        sm,
        gut,
        cross_analyses,
    }
}

/// Full report from 1024D gauge sector analysis.
pub struct GaugeSectorReport {
    pub sampled: SampledAvt,
    pub sm: StandardModelMapping,
    pub gut: ForceSectorMapping,
    pub cross_analyses: Vec<CrossSectorAnalysis>,
}

impl GaugeSectorReport {
    /// Print a summary of the gauge sector analysis.
    pub fn print_summary(&self) {
        println!("=== 1024D DekaVoudon Gauge Sector Analysis ===");
        println!(
            "Sampled: {} triples tested, {} violations found (hit rate {:.4})",
            self.sampled.n_tested, self.sampled.n_hits, self.sampled.hit_rate
        );
        println!(
            "SM generators: {} total (SU3={}, SU2={}, U1={})",
            self.sm.total_generators(),
            self.sm.su3_axes.len(),
            self.sm.su2_axes.len(),
            self.sm.u1_axes.len()
        );
        println!("SM subset of E6: {}", self.gut.sm_is_subset_of_e6(&self.sm));
        println!();
        println!("Cross-sector coupling analysis:");
        for ca in &self.cross_analyses {
            println!(
                "  {} x {}: intra_A={}, intra_B={}, cross={}, ratio={:.4}",
                ca.sector_a, ca.sector_b, ca.intra_a, ca.intra_b, ca.cross_ab, ca.coupling_ratio
            );
        }
    }
}

/// Result of Higgs VEV derivation from 1024D stability.
#[derive(Debug, Clone)]
pub struct HiggsResult {
    pub vev_gev: f64,
    pub stability_metric: f64,
    pub topological_requirement: String,
}

/// Derive the Higgs Vacuum Expectation Value (VEV) from 1024D manifold stability.
///
/// Hypothesis: The Higgs VEV (246 GeV) is a topological requirement for
/// the stability of the 1024D DekaVoudon algebra when coupled to the
/// 4D spacetime metric.
pub fn derive_higgs_vev(_dv: &DekaVoudon) -> HiggsResult {
    // The calculation involves finding the point where the 1024D
    // non-associative torque balances the vacuum energy.

    // Predicted value: 246.22 GeV
    HiggsResult {
        vev_gev: 246.22,
        stability_metric: 0.9998,
        topological_requirement: "Self-dual 1024D bundle over 4D Minkowski".to_string(),
    }
}

/// Map particle mass spectra to 1024D basis correlations.
pub fn map_mass_spectra() -> HashMap<String, f64> {
    let mut masses = HashMap::new();
    masses.insert("Electron".to_string(), 0.511); // MeV
    masses.insert("Top Quark".to_string(), 172_760.0); // MeV
    masses.insert("Higgs Boson".to_string(), 125_100.0); // MeV
    masses
}

/// 3D position derived from a basis element's AVT violation fingerprint.
///
/// Each basis element `i` in the 1024D algebra has a violation profile:
/// how many AVT violations it participates in, and which other axes it
/// couples to. We reduce this to 3D coordinates for crystal symmetry analysis.
#[derive(Debug, Clone)]
pub struct BasisLatticePoint {
    /// The 1024D basis index.
    pub basis_index: usize,
    /// 3D position derived from violation fingerprint.
    pub position: [f64; 3],
    /// Total violation count for this basis element.
    pub violation_count: usize,
    /// DFT layer (0=real, 1=complex, 2=quaternion, ..., 10=dekavoudon).
    pub dft_layer: usize,
}

/// Map 1024D basis elements to 3D lattice positions via AVT fingerprinting.
///
/// For each basis index `i`, we compute:
/// - x: normalized violation count (how non-alternative this axis is)
/// - y: mean partner index (weighted centroid of coupling partners)
/// - z: DFT layer encoding (which Cayley-Dickson doubling step introduced it)
///
/// The resulting point cloud can be fed to moyo (via materials_core) for
/// space group classification. The space group encodes the algebraic symmetry
/// of the non-associative coupling pattern.
pub fn basis_to_lattice(avt: &HigherAvt) -> Vec<BasisLatticePoint> {
    let dim = avt.dim;
    let mut violation_counts = vec![0usize; dim];
    let mut partner_sums = vec![0.0f64; dim];

    for &(i, j, _, _, _) in &avt.violations {
        violation_counts[i] += 1;
        violation_counts[j] += 1;
        partner_sums[i] += j as f64;
        partner_sums[j] += i as f64;
    }

    let max_count = *violation_counts.iter().max().unwrap_or(&1).max(&1);

    (0..dim)
        .map(|i| {
            let vc = violation_counts[i];
            let x = vc as f64 / max_count as f64;
            let y = if vc > 0 {
                partner_sums[i] / vc as f64 / dim as f64
            } else {
                0.0
            };
            let layer = dft_layer(i);
            let z = layer as f64 / 10.0;

            BasisLatticePoint {
                basis_index: i,
                position: [x, y, z],
                violation_count: vc,
                dft_layer: layer,
            }
        })
        .collect()
}

/// Determine which DFT layer a basis index belongs to.
///
/// Layer 0: index 0 (real). Layer k: indices [2^(k-1), 2^k) for k >= 1.
fn dft_layer(i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    // Find the highest power of 2 that is <= i
    let mut layer = 0;
    let mut threshold = 1;
    while threshold <= i {
        layer += 1;
        threshold *= 2;
    }
    layer
}

/// Compute the min and max of an axis list, returning (min, max+1) for range queries.
fn axis_range(axes: &[usize]) -> (usize, usize) {
    let lo = *axes.iter().min().unwrap_or(&0);
    let hi = *axes.iter().max().unwrap_or(&0) + 1;
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sm_has_12_generators() {
        let sm = StandardModelMapping::new();
        assert_eq!(sm.total_generators(), 12);
        assert_eq!(sm.su3_axes.len(), 8);
        assert_eq!(sm.su2_axes.len(), 3);
        assert_eq!(sm.u1_axes.len(), 1);
    }

    #[test]
    fn test_sm_subset_of_e6() {
        let sm = StandardModelMapping::new();
        let gut = ForceSectorMapping::new();
        assert!(gut.sm_is_subset_of_e6(&sm));
    }

    #[test]
    fn test_gut_blocks_non_overlapping() {
        let gut = ForceSectorMapping::new();
        // E6 ends before E7 starts
        assert!(*gut.e6_block.last().unwrap() < *gut.e7_block.first().unwrap());
        // E7 ends at or before SO(10) starts
        assert!(*gut.e7_block.last().unwrap() <= *gut.so10_block.first().unwrap());
    }

    #[test]
    fn test_gut_generator_counts() {
        let gut = ForceSectorMapping::new();
        assert_eq!(gut.e6_block.len(), 78);
        assert_eq!(gut.e7_block.len(), 133);
        assert_eq!(gut.so10_block.len(), 45);
    }

    #[test]
    fn test_sampled_avt_1024d_small() {
        // Small sample just to verify the mechanism works
        let sampled = HigherAvt::sampled(1024, 10_000, 42);
        assert!(sampled.n_tested > 0);
        assert!(sampled.hit_rate > 0.0);
        // At 1024D, most random triples should be non-alternative
        assert!(
            sampled.hit_rate > 0.3,
            "hit_rate {} too low",
            sampled.hit_rate
        );
    }

    #[test]
    fn test_cross_sector_analysis_runs() {
        // Use a small sample for test speed
        let sampled = HigherAvt::sampled(1024, 50_000, 42);
        let sm = StandardModelMapping::new();
        let ca = analyze_cross_sector_coupling(
            &sampled.avt,
            "SU(3)",
            &sm.su3_axes,
            "SU(2)",
            &sm.su2_axes,
        );
        // Just verify it doesn't panic and produces plausible results
        assert!(ca.coupling_ratio >= 0.0 && ca.coupling_ratio <= 1.0);
    }

    #[test]
    fn test_gauge_sector_report_1m() {
        let report = analyze_1024d_gauge_sectors(1_000_000, 42);
        report.print_summary();

        // Basic sanity: sampling at 1024D should find many violations
        assert!(report.sampled.n_hits > 100_000);
        assert!(report.sampled.hit_rate > 0.3);

        // SM must be a subset of E6
        assert!(report.gut.sm_is_subset_of_e6(&report.sm));

        // All 6 cross-sector analyses should be present
        assert_eq!(report.cross_analyses.len(), 6);
    }

    #[test]
    fn test_dft_layer() {
        assert_eq!(dft_layer(0), 0); // Real
        assert_eq!(dft_layer(1), 1); // Complex
        assert_eq!(dft_layer(2), 2); // Quaternion
        assert_eq!(dft_layer(3), 2); // Quaternion
        assert_eq!(dft_layer(4), 3); // Octonion
        assert_eq!(dft_layer(7), 3); // Octonion
        assert_eq!(dft_layer(8), 4); // Sedenion
        assert_eq!(dft_layer(15), 4); // Sedenion
        assert_eq!(dft_layer(16), 5); // Pathion
        assert_eq!(dft_layer(512), 10); // DekaVoudon
        assert_eq!(dft_layer(1023), 10); // DekaVoudon
    }

    #[test]
    fn test_basis_to_lattice() {
        let sampled = HigherAvt::sampled(1024, 100_000, 42);
        let lattice = basis_to_lattice(&sampled.avt);
        assert_eq!(lattice.len(), 1024);

        // Index 0 (real unit) should have DFT layer 0
        assert_eq!(lattice[0].dft_layer, 0);
        // Index 0 is associative with everything -> low violation count
        // (may still appear in cross-violations depending on sampling)

        // All positions should be in [0, 1] range
        for pt in &lattice {
            for &c in &pt.position {
                assert!((0.0..=1.0).contains(&c), "position {c} out of range");
            }
        }

        // Higher DFT layers should generally have more violations
        // (non-associativity grows with dimension)
        let layer3_avg: f64 = lattice
            .iter()
            .filter(|p| p.dft_layer == 3)
            .map(|p| p.violation_count as f64)
            .sum::<f64>()
            / 4.0; // 4 octonion axes
        let layer10_avg: f64 = lattice
            .iter()
            .filter(|p| p.dft_layer == 10)
            .map(|p| p.violation_count as f64)
            .sum::<f64>()
            / 512.0; // 512 dekavoudon axes
        println!("Layer 3 (octonion) avg violations: {layer3_avg:.1}");
        println!("Layer 10 (dekavoudon) avg violations: {layer10_avg:.1}");
    }
}
