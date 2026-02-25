//! Charged-particle multiplicity tables per centrality for heavy-ion collisions.
//!
//! Hardcoded dNch/deta values from published ALICE, CMS, and PHENIX measurements.
//! Converted to dNch/dy using the Jacobian deta/dy ~ 1.1 at midrapidity.
//!
//! References:
//! - ALICE Pb-Pb 5.02 TeV: PLB 772 (2017) 567 (arXiv:1612.08966)
//! - ALICE Pb-Pb 2.76 TeV: PRL 106 (2011) 032301 (arXiv:1012.1657)
//! - PHENIX Au-Au 200 GeV: PRC 71 (2005) 034908 (arXiv:nucl-ex/0409015)
//! - ALICE Xe-Xe 5.44 TeV: PLB 790 (2019) 35 (arXiv:1805.04432)

/// Multiplicity measurement for a single centrality bin.
#[derive(Debug, Clone)]
pub struct MultiplicityBin {
    /// Centrality bin lower edge (fraction, e.g. 0.0 for 0%).
    pub cent_lo: f64,
    /// Centrality bin upper edge (fraction).
    pub cent_hi: f64,
    /// dNch/deta at midrapidity.
    pub dnch_deta: f64,
    /// Statistical + systematic error on dNch/deta.
    pub dnch_deta_err: f64,
}

impl MultiplicityBin {
    /// Convert dNch/deta to dNch/dy using Jacobian ~ 1.1 at midrapidity.
    #[must_use]
    pub fn dnch_dy(&self) -> f64 {
        self.dnch_deta / 1.1
    }

    /// Error on dNch/dy.
    #[must_use]
    pub fn dnch_dy_err(&self) -> f64 {
        self.dnch_deta_err / 1.1
    }
}

/// Collision system identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionSystem {
    PbPb5020,
    PbPb2760,
    AuAu200,
    XeXe5440,
}

impl CollisionSystem {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::PbPb5020 => "Pb-Pb 5.02 TeV",
            Self::PbPb2760 => "Pb-Pb 2.76 TeV",
            Self::AuAu200 => "Au-Au 200 GeV",
            Self::XeXe5440 => "Xe-Xe 5.44 TeV",
        }
    }
}

/// ALICE Pb-Pb 5.02 TeV dNch/deta at midrapidity.
/// Source: ALICE Collaboration, PLB 772 (2017) 567, Table 1.
#[must_use]
pub fn alice_pbpb_5020_multiplicity() -> Vec<MultiplicityBin> {
    vec![
        MultiplicityBin {
            cent_lo: 0.00,
            cent_hi: 0.05,
            dnch_deta: 1943.0,
            dnch_deta_err: 54.0,
        },
        MultiplicityBin {
            cent_lo: 0.05,
            cent_hi: 0.10,
            dnch_deta: 1586.0,
            dnch_deta_err: 46.0,
        },
        MultiplicityBin {
            cent_lo: 0.10,
            cent_hi: 0.20,
            dnch_deta: 1180.0,
            dnch_deta_err: 31.0,
        },
        MultiplicityBin {
            cent_lo: 0.20,
            cent_hi: 0.30,
            dnch_deta: 786.0,
            dnch_deta_err: 20.0,
        },
        MultiplicityBin {
            cent_lo: 0.30,
            cent_hi: 0.40,
            dnch_deta: 512.0,
            dnch_deta_err: 15.0,
        },
        MultiplicityBin {
            cent_lo: 0.40,
            cent_hi: 0.50,
            dnch_deta: 318.0,
            dnch_deta_err: 12.0,
        },
        MultiplicityBin {
            cent_lo: 0.50,
            cent_hi: 0.60,
            dnch_deta: 183.0,
            dnch_deta_err: 8.0,
        },
        MultiplicityBin {
            cent_lo: 0.60,
            cent_hi: 0.70,
            dnch_deta: 96.3,
            dnch_deta_err: 5.8,
        },
        MultiplicityBin {
            cent_lo: 0.70,
            cent_hi: 0.80,
            dnch_deta: 44.9,
            dnch_deta_err: 3.4,
        },
        MultiplicityBin {
            cent_lo: 0.80,
            cent_hi: 0.90,
            dnch_deta: 17.5,
            dnch_deta_err: 1.8,
        },
    ]
}

/// ALICE Pb-Pb 2.76 TeV dNch/deta at midrapidity.
/// Source: ALICE Collaboration, PRL 106 (2011) 032301.
#[must_use]
pub fn alice_pbpb_2760_multiplicity() -> Vec<MultiplicityBin> {
    vec![
        MultiplicityBin {
            cent_lo: 0.00,
            cent_hi: 0.05,
            dnch_deta: 1601.0,
            dnch_deta_err: 60.0,
        },
        MultiplicityBin {
            cent_lo: 0.05,
            cent_hi: 0.10,
            dnch_deta: 1294.0,
            dnch_deta_err: 49.0,
        },
        MultiplicityBin {
            cent_lo: 0.10,
            cent_hi: 0.20,
            dnch_deta: 966.0,
            dnch_deta_err: 37.0,
        },
        MultiplicityBin {
            cent_lo: 0.20,
            cent_hi: 0.30,
            dnch_deta: 649.0,
            dnch_deta_err: 23.0,
        },
        MultiplicityBin {
            cent_lo: 0.30,
            cent_hi: 0.40,
            dnch_deta: 426.0,
            dnch_deta_err: 15.0,
        },
        MultiplicityBin {
            cent_lo: 0.40,
            cent_hi: 0.50,
            dnch_deta: 261.0,
            dnch_deta_err: 9.0,
        },
        MultiplicityBin {
            cent_lo: 0.50,
            cent_hi: 0.60,
            dnch_deta: 149.0,
            dnch_deta_err: 6.0,
        },
        MultiplicityBin {
            cent_lo: 0.60,
            cent_hi: 0.70,
            dnch_deta: 76.0,
            dnch_deta_err: 4.0,
        },
        MultiplicityBin {
            cent_lo: 0.70,
            cent_hi: 0.80,
            dnch_deta: 35.0,
            dnch_deta_err: 2.0,
        },
    ]
}

/// PHENIX Au-Au 200 GeV dNch/deta at midrapidity.
/// Source: PHENIX Collaboration, PRC 71 (2005) 034908.
#[must_use]
pub fn phenix_auau_200_multiplicity() -> Vec<MultiplicityBin> {
    vec![
        MultiplicityBin {
            cent_lo: 0.00,
            cent_hi: 0.05,
            dnch_deta: 687.0,
            dnch_deta_err: 37.0,
        },
        MultiplicityBin {
            cent_lo: 0.05,
            cent_hi: 0.10,
            dnch_deta: 560.0,
            dnch_deta_err: 27.0,
        },
        MultiplicityBin {
            cent_lo: 0.10,
            cent_hi: 0.20,
            dnch_deta: 416.0,
            dnch_deta_err: 21.0,
        },
        MultiplicityBin {
            cent_lo: 0.20,
            cent_hi: 0.30,
            dnch_deta: 280.0,
            dnch_deta_err: 14.0,
        },
        MultiplicityBin {
            cent_lo: 0.30,
            cent_hi: 0.40,
            dnch_deta: 185.0,
            dnch_deta_err: 10.0,
        },
        MultiplicityBin {
            cent_lo: 0.40,
            cent_hi: 0.50,
            dnch_deta: 118.0,
            dnch_deta_err: 7.0,
        },
        MultiplicityBin {
            cent_lo: 0.50,
            cent_hi: 0.60,
            dnch_deta: 71.6,
            dnch_deta_err: 5.0,
        },
        MultiplicityBin {
            cent_lo: 0.60,
            cent_hi: 0.70,
            dnch_deta: 40.2,
            dnch_deta_err: 3.5,
        },
    ]
}

/// ALICE Xe-Xe 5.44 TeV dNch/deta at midrapidity.
/// Source: ALICE Collaboration, PLB 790 (2019) 35.
#[must_use]
pub fn alice_xexe_5440_multiplicity() -> Vec<MultiplicityBin> {
    vec![
        MultiplicityBin {
            cent_lo: 0.00,
            cent_hi: 0.05,
            dnch_deta: 1167.0,
            dnch_deta_err: 29.0,
        },
        MultiplicityBin {
            cent_lo: 0.05,
            cent_hi: 0.10,
            dnch_deta: 939.0,
            dnch_deta_err: 23.0,
        },
        MultiplicityBin {
            cent_lo: 0.10,
            cent_hi: 0.20,
            dnch_deta: 690.0,
            dnch_deta_err: 16.0,
        },
        MultiplicityBin {
            cent_lo: 0.20,
            cent_hi: 0.30,
            dnch_deta: 453.0,
            dnch_deta_err: 11.0,
        },
        MultiplicityBin {
            cent_lo: 0.30,
            cent_hi: 0.40,
            dnch_deta: 287.0,
            dnch_deta_err: 8.0,
        },
        MultiplicityBin {
            cent_lo: 0.40,
            cent_hi: 0.50,
            dnch_deta: 172.0,
            dnch_deta_err: 6.0,
        },
        MultiplicityBin {
            cent_lo: 0.50,
            cent_hi: 0.60,
            dnch_deta: 96.7,
            dnch_deta_err: 4.3,
        },
        MultiplicityBin {
            cent_lo: 0.60,
            cent_hi: 0.70,
            dnch_deta: 49.7,
            dnch_deta_err: 3.0,
        },
    ]
}

/// Get multiplicity table for a given collision system.
#[must_use]
pub fn multiplicity_table(system: CollisionSystem) -> Vec<MultiplicityBin> {
    match system {
        CollisionSystem::PbPb5020 => alice_pbpb_5020_multiplicity(),
        CollisionSystem::PbPb2760 => alice_pbpb_2760_multiplicity(),
        CollisionSystem::AuAu200 => phenix_auau_200_multiplicity(),
        CollisionSystem::XeXe5440 => alice_xexe_5440_multiplicity(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbpb_5020_centrality_ordering() {
        let table = alice_pbpb_5020_multiplicity();
        // dNch/deta should decrease with increasing centrality percentile
        for w in table.windows(2) {
            assert!(
                w[0].dnch_deta > w[1].dnch_deta,
                "dNch/deta should decrease: {} > {} for {}-{}% vs {}-{}%",
                w[0].dnch_deta,
                w[1].dnch_deta,
                w[0].cent_lo * 100.0,
                w[0].cent_hi * 100.0,
                w[1].cent_lo * 100.0,
                w[1].cent_hi * 100.0
            );
        }
    }

    #[test]
    fn test_pbpb_5020_central_value() {
        let table = alice_pbpb_5020_multiplicity();
        // 0-5% central: dNch/deta = 1943 +/- 54
        assert_eq!(table[0].dnch_deta, 1943.0);
    }

    #[test]
    fn test_dnch_dy_conversion() {
        let bin = MultiplicityBin {
            cent_lo: 0.0,
            cent_hi: 0.05,
            dnch_deta: 1100.0,
            dnch_deta_err: 50.0,
        };
        let dy = bin.dnch_dy();
        assert!(
            (dy - 1000.0).abs() < 1.0,
            "dNch/dy = {} (expected 1000)",
            dy
        );
    }

    #[test]
    fn test_all_systems_nonempty() {
        for sys in [
            CollisionSystem::PbPb5020,
            CollisionSystem::PbPb2760,
            CollisionSystem::AuAu200,
            CollisionSystem::XeXe5440,
        ] {
            let table = multiplicity_table(sys);
            assert!(!table.is_empty(), "{:?} table should not be empty", sys);
        }
    }
}
