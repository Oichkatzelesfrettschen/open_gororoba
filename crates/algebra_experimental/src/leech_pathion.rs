//! Leech lattice embedding into Cayley-Dickson tower algebras.
//!
//! The Leech lattice (Lambda_24) is the unique 24D even unimodular lattice with
//! no vectors of norm 2. Since 24 is not a power of 2, there is no 24D
//! Cayley-Dickson algebra. The nearest CD algebra is the 32D Pathion (2^5).
//!
//! This module provides:
//! - `LeechPathionEmbedding`: 24D -> 32D subspace embedding with 8D dark sector
//! - `LeechCdEmbedding`: Generalized 24D -> any CD dim embedding (32, 64, 128, ...)
//! - `CrossSectorResult`: Multiplication structure analysis across Leech/dark sectors
//! - `LeechTowerSweep`: Sweep all CD dimensions and collect dark sector statistics
//!
//! # Dark sector instability theorem
//!
//! For the Pathion (dim=32), axes 24-31 form an 8D dark complement. Dark x Dark
//! products always land in the Leech sector because XOR of two indices with bit 4
//! set clears bit 4. This generalizes: at any CD dim=2^n, the dark sector
//! D = {e_24, ..., e_{2^n - 1}} has products that partially or fully escape to
//! lower-index sectors.
//!
//! # References
//! - BIB-0316 (Conway 1968): Leech lattice characterization
//! - BIB-0310 (Brown & Loeb 1997): Alternativity failure and zero divisors above dim 8
//! - BIB-0127 (Conway & Sloane 1999): Sphere Packings, Lattices and Groups

use cd_kernel::cayley_dickson::SignTable;

/// Leech lattice dimension.
pub const LEECH_DIM: usize = 24;
/// Pathion CD algebra dimension (2^5).
pub const PATHION_DIM: usize = 32;
/// Dark sector dimension for Pathion (32 - 24 = 8, isomorphic to octonion dimension).
pub const DARK_SECTOR_DIM: usize = PATHION_DIM - LEECH_DIM;

/// Subspace embedding of the 24D Leech lattice into the 32D Pathion CD algebra.
///
/// The Leech lattice occupies axes 0..23; the 8D orthogonal complement (axes 24..31)
/// forms the "dark sector". Under Pathion multiplication, dark x dark products
/// escape to the Leech sector (XOR clears bit 4), creating algebraic entanglement.
///
/// # Dark sector instability proof
///
/// **Theorem**: The dark sector D = {e_24, ..., e_31} is NOT a subalgebra of the
/// Pathion. For all i, j in D with i != j, the product index i XOR j has bit 4
/// cleared (since both operands have bit 4 set), so i XOR j < 16, which is in the
/// Leech sector. Concretely: 24 XOR 25 = 1, 24 XOR 31 = 7, 30 XOR 31 = 1, etc.
/// Every dark-dark product escapes to Leech. The cross-sector analysis confirms:
/// dark_dark_to_dark = 0, dark_dark_to_leech = 56 (all 8*7 non-identity pairs).
///
/// **Consequence**: The dark sector acts as a "charge space" whose products generate
/// Leech-sector observables. The entanglement fraction (fraction of all cross-sector
/// products landing in the opposite sector) is 0.7091, confirming algebraic
/// entanglement between Leech and dark degrees of freedom.
pub struct LeechPathionEmbedding {
    /// Leech sector axes (0..23).
    pub leech_axes: [usize; LEECH_DIM],
    /// Dark sector axes (24..31).
    pub dark_axes: [usize; DARK_SECTOR_DIM],
}

impl LeechPathionEmbedding {
    /// Create the canonical embedding: Leech occupies axes 0..23, dark sector 24..31.
    pub fn canonical() -> Self {
        let mut leech_axes = [0usize; LEECH_DIM];
        let mut dark_axes = [0usize; DARK_SECTOR_DIM];
        for (i, ax) in leech_axes.iter_mut().enumerate() {
            *ax = i;
        }
        for (i, ax) in dark_axes.iter_mut().enumerate() {
            *ax = LEECH_DIM + i;
        }
        Self {
            leech_axes,
            dark_axes,
        }
    }

    /// Embed a 24D Leech lattice vector into 32D Pathion space by zero-padding.
    ///
    /// The resulting vector has the Leech components in axes 0..23 and zeros in
    /// axes 24..31 (dark sector). This preserves all inner products and norms.
    pub fn embed(&self, leech_vec: &[f64; LEECH_DIM]) -> [f64; PATHION_DIM] {
        let mut result = [0.0f64; PATHION_DIM];
        for (i, &ax) in self.leech_axes.iter().enumerate() {
            result[ax] = leech_vec[i];
        }
        result
    }

    /// Project a 32D Pathion vector down to its 24D Leech component.
    pub fn project_leech(&self, pathion_vec: &[f64; PATHION_DIM]) -> [f64; LEECH_DIM] {
        let mut result = [0.0f64; LEECH_DIM];
        for (i, &ax) in self.leech_axes.iter().enumerate() {
            result[i] = pathion_vec[ax];
        }
        result
    }

    /// Extract the 8D dark sector component from a 32D Pathion vector.
    pub fn project_dark(&self, pathion_vec: &[f64; PATHION_DIM]) -> [f64; DARK_SECTOR_DIM] {
        let mut result = [0.0f64; DARK_SECTOR_DIM];
        for (i, &ax) in self.dark_axes.iter().enumerate() {
            result[i] = pathion_vec[ax];
        }
        result
    }

    /// Verify that embedding preserves the L2 norm (isometry).
    pub fn norm_preserved(&self, leech_vec: &[f64; LEECH_DIM]) -> bool {
        let embedded = self.embed(leech_vec);
        let leech_norm_sq: f64 = leech_vec.iter().map(|x| x * x).sum();
        let pathion_norm_sq: f64 = embedded.iter().map(|x| x * x).sum();
        (leech_norm_sq - pathion_norm_sq).abs() < 1e-12
    }

    /// Classify a CD basis index as Leech sector or dark sector.
    pub fn is_dark_axis(&self, axis: usize) -> bool {
        (LEECH_DIM..PATHION_DIM).contains(&axis)
    }

    /// Analyze the cross-sector multiplication structure of the Pathion.
    ///
    /// For all pairs (i, j) where i is in Leech (0..23) and j is in dark (24..31),
    /// compute the product index i XOR j and sign. Returns the fraction of cross-
    /// sector products that land in the dark sector vs Leech sector.
    pub fn cross_sector_analysis(&self) -> CrossSectorResult {
        let table = SignTable::new(PATHION_DIM);
        let mut leech_to_leech = 0usize;
        let mut leech_to_dark = 0usize;
        let mut dark_to_leech = 0usize;
        let mut dark_to_dark = 0usize;

        // Leech x Dark products
        for &i in &self.leech_axes {
            for &j in &self.dark_axes {
                let product_idx = i ^ j;
                let _sign = table.sign(i, j);
                if product_idx < LEECH_DIM {
                    leech_to_leech += 1;
                } else {
                    leech_to_dark += 1;
                }
            }
        }

        // Dark x Dark products
        for (di, &i) in self.dark_axes.iter().enumerate() {
            for &j in &self.dark_axes[di + 1..] {
                let product_idx = i ^ j;
                let _sign = table.sign(i, j);
                if product_idx < LEECH_DIM {
                    dark_to_leech += 1;
                } else {
                    dark_to_dark += 1;
                }
            }
        }

        CrossSectorResult {
            leech_dark_to_leech: leech_to_leech,
            leech_dark_to_dark: leech_to_dark,
            dark_dark_to_leech: dark_to_leech,
            dark_dark_to_dark: dark_to_dark,
        }
    }
}

/// Result of cross-sector multiplication analysis.
#[derive(Debug, Clone)]
pub struct CrossSectorResult {
    /// Count of Leech x Dark products landing in Leech sector.
    pub leech_dark_to_leech: usize,
    /// Count of Leech x Dark products landing in dark sector.
    pub leech_dark_to_dark: usize,
    /// Count of Dark x Dark products landing in Leech sector.
    pub dark_dark_to_leech: usize,
    /// Count of Dark x Dark products landing in dark sector.
    pub dark_dark_to_dark: usize,
}

impl CrossSectorResult {
    /// Fraction of all cross-sector products that entangle sectors.
    pub fn entanglement_fraction(&self) -> f64 {
        let total = self.leech_dark_to_leech
            + self.leech_dark_to_dark
            + self.dark_dark_to_leech
            + self.dark_dark_to_dark;
        if total == 0 {
            return 0.0;
        }
        // Entanglement = products that cross sector boundaries
        let crossing = self.leech_dark_to_leech + self.dark_dark_to_leech;
        crossing as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// Generalized Leech embedding into arbitrary CD dimensions
// ---------------------------------------------------------------------------

/// Generalized embedding of the 24D Leech lattice into any CD algebra dim >= 32.
///
/// For CD dimension N (must be a power of 2, N >= 32):
/// - Leech sector: axes 0..23 (always 24D)
/// - Dark sector: axes 24..N-1 (N-24 dimensional)
///
/// The dark sector grows with CD dimension:
/// - Pathion (32):  8D dark sector (octonion-sized)
/// - Chingon (64):  40D dark sector
/// - Routon (128):  104D dark sector
/// - Voudon (256):  232D dark sector
/// - Eriston (512): 488D dark sector
///
/// Cross-sector multiplication structure changes with dimension because the
/// XOR product topology depends on which bits are shared between operands.
pub struct LeechCdEmbedding {
    /// CD algebra dimension.
    pub cd_dim: usize,
    /// Dark sector size (cd_dim - 24).
    pub dark_dim: usize,
}

impl LeechCdEmbedding {
    /// Create a Leech embedding for the given CD dimension.
    ///
    /// Panics if `cd_dim` is not a power of 2 or is less than 32.
    pub fn new(cd_dim: usize) -> Self {
        assert!(cd_dim.is_power_of_two(), "CD dim must be a power of 2");
        assert!(cd_dim >= 32, "CD dim must be >= 32 to contain 24D Leech");
        Self {
            cd_dim,
            dark_dim: cd_dim - LEECH_DIM,
        }
    }

    /// Classify a basis index as Leech (true) or dark (false).
    pub fn is_leech(&self, axis: usize) -> bool {
        axis < LEECH_DIM
    }

    /// Classify a basis index as dark sector.
    pub fn is_dark(&self, axis: usize) -> bool {
        axis >= LEECH_DIM && axis < self.cd_dim
    }

    /// Analyze the full cross-sector multiplication structure.
    ///
    /// Counts how products between Leech x Dark and Dark x Dark sectors
    /// distribute across the Leech and dark sectors.
    pub fn cross_sector_analysis(&self) -> CrossSectorResult {
        let table = SignTable::new(self.cd_dim);
        let mut leech_to_leech = 0usize;
        let mut leech_to_dark = 0usize;
        let mut dark_to_leech = 0usize;
        let mut dark_to_dark = 0usize;

        // Leech x Dark products: i in 0..24, j in 24..cd_dim
        for i in 0..LEECH_DIM {
            for j in LEECH_DIM..self.cd_dim {
                let product_idx = i ^ j;
                let _sign = table.sign(i, j);
                if product_idx < LEECH_DIM {
                    leech_to_leech += 1;
                } else {
                    leech_to_dark += 1;
                }
            }
        }

        // Dark x Dark products: i, j both in 24..cd_dim, i < j
        for i in LEECH_DIM..self.cd_dim {
            for j in (i + 1)..self.cd_dim {
                let product_idx = i ^ j;
                let _sign = table.sign(i, j);
                if product_idx < LEECH_DIM {
                    dark_to_leech += 1;
                } else {
                    dark_to_dark += 1;
                }
            }
        }

        CrossSectorResult {
            leech_dark_to_leech: leech_to_leech,
            leech_dark_to_dark: leech_to_dark,
            dark_dark_to_leech: dark_to_leech,
            dark_dark_to_dark: dark_to_dark,
        }
    }

    /// Analyze only the XOR product topology (no SignTable needed).
    ///
    /// This is much faster than `cross_sector_analysis` because it only checks
    /// where products land (Leech vs dark) based on XOR index arithmetic,
    /// without computing actual signs. Suitable for large dimensions.
    pub fn xor_sector_analysis(&self) -> CrossSectorResult {
        let mut leech_to_leech = 0usize;
        let mut leech_to_dark = 0usize;
        let mut dark_to_leech = 0usize;
        let mut dark_to_dark = 0usize;

        // Leech x Dark
        for i in 0..LEECH_DIM {
            for j in LEECH_DIM..self.cd_dim {
                let product_idx = i ^ j;
                if product_idx < LEECH_DIM {
                    leech_to_leech += 1;
                } else {
                    leech_to_dark += 1;
                }
            }
        }

        // Dark x Dark (i < j)
        for i in LEECH_DIM..self.cd_dim {
            for j in (i + 1)..self.cd_dim {
                let product_idx = i ^ j;
                if product_idx < LEECH_DIM {
                    dark_to_leech += 1;
                } else {
                    dark_to_dark += 1;
                }
            }
        }

        CrossSectorResult {
            leech_dark_to_leech: leech_to_leech,
            leech_dark_to_dark: leech_to_dark,
            dark_dark_to_leech: dark_to_leech,
            dark_dark_to_dark: dark_to_dark,
        }
    }
}

/// Summary of dark sector statistics for one CD dimension.
#[derive(Debug, Clone)]
pub struct DarkSectorSummary {
    /// CD algebra dimension.
    pub cd_dim: usize,
    /// CD algebra name.
    pub cd_name: &'static str,
    /// Dark sector dimension (cd_dim - 24).
    pub dark_dim: usize,
    /// Cross-sector multiplication results.
    pub cross_sector: CrossSectorResult,
    /// Entanglement fraction.
    pub entanglement_fraction: f64,
    /// Fraction of dark x dark products that stay in dark sector.
    pub dark_dark_retention: f64,
}

/// Sweep all CD dimensions from 32 to `max_dim` and collect dark sector statistics.
///
/// Uses `xor_sector_analysis` (fast, no SignTable) by default. For dimensions
/// where sign information is needed, use the individual `LeechCdEmbedding`.
pub fn sweep_leech_tower(max_dim: usize) -> Vec<DarkSectorSummary> {
    use super::cd_tower_violations::CD_TOWER;

    CD_TOWER
        .iter()
        .filter(|&&(dim, _)| dim >= 32 && dim <= max_dim)
        .map(|&(dim, name)| {
            let emb = LeechCdEmbedding::new(dim);
            let cs = emb.xor_sector_analysis();
            let entanglement = cs.entanglement_fraction();
            let dd_total = cs.dark_dark_to_leech + cs.dark_dark_to_dark;
            let dark_retention = if dd_total > 0 {
                cs.dark_dark_to_dark as f64 / dd_total as f64
            } else {
                0.0
            };
            DarkSectorSummary {
                cd_dim: dim,
                cd_name: name,
                dark_dim: dim - LEECH_DIM,
                cross_sector: cs,
                entanglement_fraction: entanglement,
                dark_dark_retention: dark_retention,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Pathion-specific tests (32D) ---

    #[test]
    fn test_leech_pathion_embedding_isometry() {
        let emb = LeechPathionEmbedding::canonical();
        // Unit vector along first axis
        let mut v = [0.0f64; LEECH_DIM];
        v[0] = 1.0;
        assert!(emb.norm_preserved(&v));

        // Arbitrary non-trivial vector
        let v2 = [
            1.0, 2.0, 3.0, 0.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7.0,
        ];
        assert!(emb.norm_preserved(&v2));
    }

    #[test]
    fn test_leech_pathion_roundtrip() {
        let emb = LeechPathionEmbedding::canonical();
        let v = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ];
        let embedded = emb.embed(&v);
        let projected = emb.project_leech(&embedded);
        for i in 0..LEECH_DIM {
            assert!(
                (projected[i] - v[i]).abs() < 1e-12,
                "roundtrip mismatch at axis {i}: {} vs {}",
                projected[i],
                v[i]
            );
        }
        // Dark sector should be zero
        let dark = emb.project_dark(&embedded);
        for (i, &d) in dark.iter().enumerate().take(DARK_SECTOR_DIM) {
            assert!(
                d.abs() < 1e-12,
                "dark sector should be zero, got {} at axis {}",
                d,
                i
            );
        }
    }

    #[test]
    fn test_dark_sector_xor_structure() {
        // Verify: i XOR j for i, j both in 24..31 always lands in 0..7 (Leech sector).
        let emb = LeechPathionEmbedding::canonical();
        for &i in &emb.dark_axes {
            for &j in &emb.dark_axes {
                if i == j {
                    continue;
                }
                let product_idx = i ^ j;
                assert!(
                    product_idx < LEECH_DIM,
                    "dark x dark product should land in Leech: {} XOR {} = {}",
                    i,
                    j,
                    product_idx
                );
            }
        }
    }

    #[test]
    fn test_cross_sector_analysis() {
        let emb = LeechPathionEmbedding::canonical();
        let result = emb.cross_sector_analysis();

        // All dark x dark products land in Leech (XOR clears bit 4)
        assert_eq!(
            result.dark_dark_to_dark, 0,
            "dark x dark cannot stay in dark: i XOR j clears bit 4"
        );
        assert!(
            result.dark_dark_to_leech > 0,
            "dark x dark must produce Leech-sector products"
        );

        // Total Leech x Dark pairs = 24 * 8 = 192
        let total_ld = result.leech_dark_to_leech + result.leech_dark_to_dark;
        assert_eq!(
            total_ld,
            LEECH_DIM * DARK_SECTOR_DIM,
            "should cover all Leech x Dark pairs"
        );
    }

    #[test]
    fn test_is_dark_axis() {
        let emb = LeechPathionEmbedding::canonical();
        for i in 0..24 {
            assert!(!emb.is_dark_axis(i), "axis {i} should be Leech sector");
        }
        for i in 24..32 {
            assert!(emb.is_dark_axis(i), "axis {i} should be dark sector");
        }
    }

    // --- Generalized Leech embedding tests ---

    #[test]
    fn test_leech_cd_embedding_pathion_matches() {
        // LeechCdEmbedding at dim=32 should produce identical cross-sector
        // results as LeechPathionEmbedding.
        let pathion = LeechPathionEmbedding::canonical();
        let pathion_cs = pathion.cross_sector_analysis();

        let general = LeechCdEmbedding::new(32);
        let general_cs = general.cross_sector_analysis();

        assert_eq!(
            pathion_cs.leech_dark_to_leech,
            general_cs.leech_dark_to_leech
        );
        assert_eq!(pathion_cs.leech_dark_to_dark, general_cs.leech_dark_to_dark);
        assert_eq!(pathion_cs.dark_dark_to_leech, general_cs.dark_dark_to_leech);
        assert_eq!(pathion_cs.dark_dark_to_dark, general_cs.dark_dark_to_dark);
    }

    #[test]
    fn test_xor_matches_full_analysis_at_32() {
        // xor_sector_analysis should produce identical sector counts as
        // cross_sector_analysis (which also computes signs).
        let emb = LeechCdEmbedding::new(32);
        let full = emb.cross_sector_analysis();
        let xor = emb.xor_sector_analysis();

        assert_eq!(full.leech_dark_to_leech, xor.leech_dark_to_leech);
        assert_eq!(full.leech_dark_to_dark, xor.leech_dark_to_dark);
        assert_eq!(full.dark_dark_to_leech, xor.dark_dark_to_leech);
        assert_eq!(full.dark_dark_to_dark, xor.dark_dark_to_dark);
    }

    #[test]
    fn test_dark_dim_grows_with_cd_dim() {
        for &dim in &[32, 64, 128, 256, 512] {
            let emb = LeechCdEmbedding::new(dim);
            assert_eq!(emb.dark_dim, dim - 24);
        }
    }

    #[test]
    fn test_chingon_dark_sector_structure() {
        // At dim=64 (Chingon), dark sector is 40D (axes 24..63).
        // Dark x Dark: i XOR j for i, j in 24..63.
        // Some products land in Leech (0..23), some stay in dark (24..63).
        let emb = LeechCdEmbedding::new(64);
        let cs = emb.xor_sector_analysis();

        // Dark x Dark should have both Leech and dark landing zones at dim >= 64
        // because the dark sector is large enough to contain XOR products.
        // e.g., 24 XOR 25 = 1 (Leech), but 32 XOR 33 = 1 (also Leech),
        //       while 24 XOR 32 = 56 (dark).
        assert!(
            cs.dark_dark_to_leech > 0,
            "some dark products should land in Leech"
        );
        // At dim=64, dark sector has 40 axes, so some dark x dark products
        // can stay in dark: e.g., 24 XOR 48 = 40 (still dark at dim=64).
        assert!(
            cs.dark_dark_to_dark > 0,
            "at dim=64, some dark x dark products stay in dark"
        );

        // Total dark pairs = C(40, 2) = 780
        let dd_total = cs.dark_dark_to_leech + cs.dark_dark_to_dark;
        assert_eq!(dd_total, 40 * 39 / 2, "should cover all C(40,2) dark pairs");
    }

    #[test]
    fn test_routon_dark_sector_structure() {
        // At dim=128 (Routon), dark sector is 104D (axes 24..127).
        let emb = LeechCdEmbedding::new(128);
        let cs = emb.xor_sector_analysis();

        let dd_total = cs.dark_dark_to_leech + cs.dark_dark_to_dark;
        assert_eq!(dd_total, 104 * 103 / 2);

        // Entanglement fraction should be well-defined
        let ef = cs.entanglement_fraction();
        assert!(
            ef > 0.0 && ef < 1.0,
            "entanglement fraction {ef} should be in (0,1)"
        );
    }

    #[test]
    fn test_sweep_leech_tower() {
        let summaries = sweep_leech_tower(512);
        // Should include: 32, 64, 128, 256, 512 = 5 entries
        assert_eq!(summaries.len(), 5);

        // Verify dark dims
        assert_eq!(summaries[0].dark_dim, 8); // Pathion
        assert_eq!(summaries[1].dark_dim, 40); // Chingon
        assert_eq!(summaries[2].dark_dim, 104); // Routon
        assert_eq!(summaries[3].dark_dim, 232); // Voudon
        assert_eq!(summaries[4].dark_dim, 488); // Eriston

        // All should have non-zero entanglement
        for s in &summaries {
            assert!(
                s.entanglement_fraction > 0.0,
                "dim={} should have non-zero entanglement",
                s.cd_dim
            );
        }

        // Pathion: dark_dark_retention should be 0 (all escape to Leech)
        assert!(
            (summaries[0].dark_dark_retention).abs() < 1e-12,
            "Pathion dark x dark should have 0% retention"
        );

        // Higher dims: dark_dark_retention should be > 0
        for s in &summaries[1..] {
            assert!(
                s.dark_dark_retention > 0.0,
                "dim={} should have positive dark retention",
                s.cd_dim
            );
        }
    }

    #[test]
    fn test_entanglement_fraction_trend() {
        // As CD dimension grows, the dark sector dominates and the fraction
        // of products landing in the 24D Leech sector should decrease.
        let summaries = sweep_leech_tower(512);
        for i in 1..summaries.len() {
            // Entanglement fraction should generally decrease with dimension
            // (more dark axes means fewer XOR products land in 0..23)
            // This is a structural property of XOR topology.
            assert!(
                summaries[i].entanglement_fraction <= summaries[i - 1].entanglement_fraction + 0.01,
                "entanglement should not increase significantly: dim={} ({:.4}) vs dim={} ({:.4})",
                summaries[i].cd_dim,
                summaries[i].entanglement_fraction,
                summaries[i - 1].cd_dim,
                summaries[i - 1].entanglement_fraction
            );
        }
    }

    #[test]
    fn test_dark_dark_retention_increases() {
        // As CD dimension grows, the dark sector retains more of its own products.
        let summaries = sweep_leech_tower(512);
        for i in 1..summaries.len() {
            assert!(
                summaries[i].dark_dark_retention >= summaries[i - 1].dark_dark_retention,
                "dark retention should increase: dim={} ({:.4}) vs dim={} ({:.4})",
                summaries[i].cd_dim,
                summaries[i].dark_dark_retention,
                summaries[i - 1].cd_dim,
                summaries[i - 1].dark_dark_retention
            );
        }
    }
}
