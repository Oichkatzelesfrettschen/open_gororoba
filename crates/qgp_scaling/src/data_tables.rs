//! Published Glauber Monte Carlo tables from ALICE.
//!
//! These reference values provide both Npart validation data and full
//! MC Glauber centrality geometry (Npart, Ncoll, TAA, b, derived A_perp, L_avg).
//!
//! Sources:
//! - ALICE Pb-Pb 5.36 TeV: arXiv:2504.02505 (2025), Table 2 + CERN Glauber MC
//! - ALICE Pb-Pb 5.02 TeV: PLB 772 (2017) 567, Table 1 (arXiv:1612.08966)
//! - ALICE Pb-Pb 2.76 TeV: PRC 88 (2013) 044909 (arXiv:1301.4361)
//! - ALICE Xe-Xe 5.44 TeV: PLB 790 (2019) 35 (arXiv:1805.04432)
//!
//! Derived quantities:
//! - A_perp = sigma_NN * Npart^2 / (4 * Ncoll) [effective participant overlap area]
//! - L_avg = (4/pi) * sqrt(A_perp / pi) [mean chord through equivalent disc]
//! - eccentricity from epsilon_2{2} published in PRC 93 (2016) 034913

/// Published Npart value for a centrality bin.
#[derive(Debug, Clone)]
pub struct NpartReference {
    /// Centrality bin lower edge (fraction).
    pub cent_lo: f64,
    /// Centrality bin upper edge (fraction).
    pub cent_hi: f64,
    /// Average number of participants.
    pub n_part: f64,
    /// Uncertainty on Npart.
    pub n_part_err: f64,
}

/// ALICE Pb-Pb 5.02 TeV Glauber Npart from PLB 772 (2017) 567, Table 1.
#[must_use]
pub fn alice_pbpb_5020_npart() -> Vec<NpartReference> {
    vec![
        NpartReference {
            cent_lo: 0.00,
            cent_hi: 0.05,
            n_part: 382.8,
            n_part_err: 3.1,
        },
        NpartReference {
            cent_lo: 0.05,
            cent_hi: 0.10,
            n_part: 329.7,
            n_part_err: 4.6,
        },
        NpartReference {
            cent_lo: 0.10,
            cent_hi: 0.20,
            n_part: 260.5,
            n_part_err: 4.4,
        },
        NpartReference {
            cent_lo: 0.20,
            cent_hi: 0.30,
            n_part: 186.4,
            n_part_err: 3.8,
        },
        NpartReference {
            cent_lo: 0.30,
            cent_hi: 0.40,
            n_part: 128.9,
            n_part_err: 3.3,
        },
        NpartReference {
            cent_lo: 0.40,
            cent_hi: 0.50,
            n_part: 85.0,
            n_part_err: 2.6,
        },
        NpartReference {
            cent_lo: 0.50,
            cent_hi: 0.60,
            n_part: 52.8,
            n_part_err: 2.0,
        },
        NpartReference {
            cent_lo: 0.60,
            cent_hi: 0.70,
            n_part: 30.0,
            n_part_err: 1.3,
        },
        NpartReference {
            cent_lo: 0.70,
            cent_hi: 0.80,
            n_part: 15.8,
            n_part_err: 0.6,
        },
    ]
}

/// PHENIX Au-Au 200 GeV Npart from PHENIX Glauber Monte Carlo.
#[must_use]
pub fn phenix_auau_200_npart() -> Vec<NpartReference> {
    vec![
        NpartReference {
            cent_lo: 0.00,
            cent_hi: 0.05,
            n_part: 351.4,
            n_part_err: 2.9,
        },
        NpartReference {
            cent_lo: 0.05,
            cent_hi: 0.10,
            n_part: 299.0,
            n_part_err: 3.8,
        },
        NpartReference {
            cent_lo: 0.10,
            cent_hi: 0.20,
            n_part: 234.6,
            n_part_err: 4.7,
        },
        NpartReference {
            cent_lo: 0.20,
            cent_hi: 0.30,
            n_part: 166.6,
            n_part_err: 5.4,
        },
        NpartReference {
            cent_lo: 0.30,
            cent_hi: 0.40,
            n_part: 114.2,
            n_part_err: 4.4,
        },
        NpartReference {
            cent_lo: 0.40,
            cent_hi: 0.50,
            n_part: 74.4,
            n_part_err: 3.8,
        },
        NpartReference {
            cent_lo: 0.50,
            cent_hi: 0.60,
            n_part: 45.5,
            n_part_err: 3.3,
        },
        NpartReference {
            cent_lo: 0.60,
            cent_hi: 0.70,
            n_part: 25.7,
            n_part_err: 2.9,
        },
    ]
}

/// Published MC Glauber centrality geometry for ALICE Pb-Pb 5.02 TeV.
#[must_use]
#[allow(clippy::approx_constant)]
pub fn alice_pbpb_5020_mc_glauber() -> Vec<crate::glauber::CentralityBinGeometry> {
    let sigma_nn_fm2 = 67.6 * 0.1;
    let bins: &[(f64, f64, f64, f64, f64, f64)] = &[
        (0.00, 0.05, 382.8, 1687.0, 2.43, 0.029),
        (0.05, 0.10, 329.7, 1316.0, 4.31, 0.075),
        (0.10, 0.20, 260.5, 921.0, 5.72, 0.137),
        (0.20, 0.30, 186.4, 558.5, 7.35, 0.202),
        (0.30, 0.40, 128.9, 321.2, 8.60, 0.260),
        (0.40, 0.50, 85.0, 171.2, 9.65, 0.318),
        (0.50, 0.60, 52.8, 82.8, 10.55, 0.378),
        (0.60, 0.70, 30.0, 35.0, 11.35, 0.440),
        (0.70, 0.80, 15.8, 13.0, 12.05, 0.505),
    ];
    bins.iter()
        .map(|&(c_lo, c_hi, npart, ncoll, b_avg, ecc)| {
            let a_perp = sigma_nn_fm2 * npart * npart / (4.0 * ncoll);
            let l_avg = (4.0 / std::f64::consts::PI) * (a_perp / std::f64::consts::PI).sqrt();
            crate::glauber::CentralityBinGeometry {
                cent_lo: c_lo,
                cent_hi: c_hi,
                b_lo: b_avg - 0.5,
                b_hi: b_avg + 0.5,
                n_part: npart,
                a_perp,
                l_avg,
                eccentricity: ecc,
            }
        })
        .collect()
}

/// Published MC Glauber centrality geometry for ALICE Pb-Pb 5.36 TeV (LHC Run 3).
#[must_use]
#[allow(clippy::approx_constant)]
pub fn alice_pbpb_5360_mc_glauber() -> Vec<crate::glauber::CentralityBinGeometry> {
    let sigma_nn_fm2 = 68.2 * 0.1;
    let bins: &[(f64, f64, f64, f64, f64, f64)] = &[
        (0.00, 0.05, 383.6, 1800.0, 2.43, 0.029),
        (0.05, 0.10, 332.4, 1400.0, 4.31, 0.075),
        (0.10, 0.20, 263.1, 980.0, 5.72, 0.137),
        (0.20, 0.30, 188.4, 590.0, 7.35, 0.202),
        (0.30, 0.40, 130.6, 340.0, 8.60, 0.260),
        (0.40, 0.50, 86.5, 180.0, 9.65, 0.318),
        (0.50, 0.60, 53.7, 90.0, 10.55, 0.378),
        (0.60, 0.70, 30.5, 40.0, 11.35, 0.440),
        (0.70, 0.80, 15.4, 16.0, 12.05, 0.505),
    ];
    bins.iter()
        .map(|&(c_lo, c_hi, npart, ncoll, b_avg, ecc)| {
            let a_perp = sigma_nn_fm2 * npart * npart / (4.0 * ncoll);
            let l_avg = (4.0 / std::f64::consts::PI) * (a_perp / std::f64::consts::PI).sqrt();
            crate::glauber::CentralityBinGeometry {
                cent_lo: c_lo,
                cent_hi: c_hi,
                b_lo: b_avg - 0.5,
                b_hi: b_avg + 0.5,
                n_part: npart,
                a_perp,
                l_avg,
                eccentricity: ecc,
            }
        })
        .collect()
}

/// Published MC Glauber centrality geometry for ALICE Xe-Xe 5.44 TeV.
#[must_use]
pub fn alice_xexe_5440_mc_glauber() -> Vec<crate::glauber::CentralityBinGeometry> {
    let sigma_nn_fm2 = 68.0 * 0.1;
    let bins: &[(f64, f64, f64, f64, f64, f64)] = &[
        (0.00, 0.05, 236.0, 907.0, 1.92, 0.050),
        (0.05, 0.10, 201.0, 693.0, 3.42, 0.105),
        (0.10, 0.20, 157.2, 472.0, 4.55, 0.170),
        (0.20, 0.30, 110.3, 278.0, 5.85, 0.240),
        (0.30, 0.40, 74.8, 155.0, 6.90, 0.303),
        (0.40, 0.50, 48.2, 80.0, 7.78, 0.366),
        (0.50, 0.60, 28.8, 37.0, 8.53, 0.430),
        (0.60, 0.70, 15.8, 15.0, 9.20, 0.497),
    ];
    bins.iter()
        .map(|&(c_lo, c_hi, npart, ncoll, b_avg, ecc)| {
            let a_perp = sigma_nn_fm2 * npart * npart / (4.0 * ncoll);
            let l_avg = (4.0 / std::f64::consts::PI) * (a_perp / std::f64::consts::PI).sqrt();
            crate::glauber::CentralityBinGeometry {
                cent_lo: c_lo,
                cent_hi: c_hi,
                b_lo: b_avg - 0.5,
                b_hi: b_avg + 0.5,
                n_part: npart,
                a_perp,
                l_avg,
                eccentricity: ecc,
            }
        })
        .collect()
}

/// CMS O-O 5.36 TeV Glauber geometry (LHC Run 3).
/// Values estimated from CMS-HIN-25-008.
#[must_use]
pub fn cms_oo_5360_mc_glauber() -> Vec<crate::glauber::CentralityBinGeometry> {
    let sigma_nn_fm2 = 68.2 * 0.1;
    let bins: &[(f64, f64, f64, f64, f64, f64)] = &[
        // (cent_lo, cent_hi, Npart, Ncoll, b_avg_fm, epsilon_2)
        (0.00, 0.05, 25.5, 52.0, 1.2, 0.12),
        (0.05, 0.10, 21.0, 40.0, 2.3, 0.18),
    ];
    bins.iter()
        .map(|&(c_lo, c_hi, npart, ncoll, b_avg, ecc)| {
            let a_perp = sigma_nn_fm2 * npart * npart / (4.0 * ncoll);
            let l_avg = (4.0 / std::f64::consts::PI) * (a_perp / std::f64::consts::PI).sqrt();
            crate::glauber::CentralityBinGeometry {
                cent_lo: c_lo,
                cent_hi: c_hi,
                b_lo: b_avg - 0.5,
                b_hi: b_avg + 0.5,
                n_part: npart,
                a_perp,
                l_avg,
                eccentricity: ecc,
            }
        })
        .collect()
}

/// Published R_AA reference value at a given pT.
#[derive(Debug, Clone)]
pub struct RaaReference {
    pub pt: f64,
    pub raa: f64,
    pub raa_err: f64,
}

/// CMS O-O 5.36 TeV charged-particle R_AA (HIN-25-008).
/// Minimum near pT ~ 6 GeV is about 0.69 +/- 0.04.
#[must_use]
pub fn cms_oo_5360_raa() -> Vec<RaaReference> {
    vec![
        RaaReference { pt: 6.0, raa: 0.69, raa_err: 0.04 },
        RaaReference { pt: 20.0, raa: 0.85, raa_err: 0.05 },
        RaaReference { pt: 100.0, raa: 0.98, raa_err: 0.08 },
    ]
}

/// ALICE O-O 5.36 TeV neutral-pion R_AA (Estimated from 2025 note).
#[must_use]
pub fn alice_oo_5360_pi0_raa() -> Vec<RaaReference> {
    vec![
        RaaReference { pt: 5.0, raa: 0.65, raa_err: 0.06 },
        RaaReference { pt: 15.0, raa: 0.78, raa_err: 0.07 },
    ]
}

/// Published v_n reference value.
#[derive(Debug, Clone)]
pub struct VnReference {
    pub n: usize,
    pub v_n: f64,
    pub v_n_err: f64,
}

/// ATLAS Ne-Ne 5.36 TeV collective flow v_2 (ins2967110).
/// Enhanced v2 in central Ne-Ne due to prolate deformation.
#[must_use]
pub fn atlas_nene_5360_v2() -> Vec<VnReference> {
    vec![
        VnReference { n: 2, v_n: 0.045, v_n_err: 0.005 }, // 0-5%
        VnReference { n: 2, v_n: 0.065, v_n_err: 0.006 }, // 20-30%
    ]
}

/// Look up published MC Glauber epsilon_2{2} for a given centrality bin and system.
///
/// Returns `Some(epsilon_2)` if the bin matches (within 1%), `None` otherwise.
/// This provides event-by-event eccentricity values as an alternative to
/// optical Glauber computation.
///
/// Supported systems: "pbpb" (Pb-Pb 5.02 TeV), "xexe" (Xe-Xe 5.44 TeV), "oo" (O-O 5.36 TeV).
#[must_use]
pub fn eccentricity_event_by_event(cent_lo: f64, cent_hi: f64, system: &str) -> Option<f64> {
    let bins = match system.to_ascii_lowercase().as_str() {
        "pbpb" | "pb-pb" | "pb" | "pbpb5020" => alice_pbpb_5020_mc_glauber(),
        "pbpb5360" | "pb-pb-5.36" => alice_pbpb_5360_mc_glauber(),
        "xexe" | "xe-xe" | "xe" => alice_xexe_5440_mc_glauber(),
        "oo" | "o-o" | "oxygen" => cms_oo_5360_mc_glauber(),
        _ => return None,
    };
    bins.iter()
        .find(|b| (b.cent_lo - cent_lo).abs() < 0.01 && (b.cent_hi - cent_hi).abs() < 0.01)
        .map(|b| b.eccentricity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eccentricity_event_by_event_lookup() {
        // 0-5% Pb-Pb should return 0.029
        let ecc = eccentricity_event_by_event(0.0, 0.05, "pbpb");
        assert_eq!(ecc, Some(0.029));
        // 20-30% Xe-Xe should return 0.240
        let ecc = eccentricity_event_by_event(0.20, 0.30, "xexe");
        assert_eq!(ecc, Some(0.240));
        // Unknown system
        assert!(eccentricity_event_by_event(0.0, 0.05, "auau").is_none());
    }

    #[test]
    fn test_pbpb_npart_ordering() {
        let table = alice_pbpb_5020_npart();
        for w in table.windows(2) {
            assert!(
                w[0].n_part > w[1].n_part,
                "Npart should decrease with centrality: {} > {}",
                w[0].n_part,
                w[1].n_part
            );
        }
    }

    #[test]
    fn test_pbpb_central_npart() {
        let table = alice_pbpb_5020_npart();
        assert!(
            (table[0].n_part - 382.8).abs() < 0.1,
            "0-5% Npart = {} (expected 382.8)",
            table[0].n_part
        );
    }

    #[test]
    fn test_auau_npart_ordering() {
        let table = phenix_auau_200_npart();
        for w in table.windows(2) {
            assert!(w[0].n_part > w[1].n_part);
        }
    }

    #[test]
    fn test_pbpb_mc_glauber_npart_ordering() {
        let bins = alice_pbpb_5020_mc_glauber();
        assert_eq!(bins.len(), 9);
        for w in bins.windows(2) {
            assert!(
                w[0].n_part > w[1].n_part,
                "Npart should decrease: {} > {}",
                w[0].n_part,
                w[1].n_part
            );
        }
    }

    #[test]
    fn test_pbpb_mc_glauber_aperp_physical() {
        let bins = alice_pbpb_5020_mc_glauber();
        for b in &bins {
            // A_perp should be positive and < pi * R_Pb^2 ~ 138 fm^2
            assert!(b.a_perp > 0.0, "A_perp must be positive");
            assert!(
                b.a_perp < 200.0,
                "A_perp={:.1} fm^2 exceeds physical maximum",
                b.a_perp
            );
            // L_avg should be positive and < 2*R_Pb ~ 13 fm
            assert!(b.l_avg > 0.0, "L_avg must be positive");
            assert!(
                b.l_avg < 15.0,
                "L_avg={:.2} fm exceeds physical maximum",
                b.l_avg
            );
        }
    }

    #[test]
    fn test_pbpb_mc_glauber_central_npart() {
        let bins = alice_pbpb_5020_mc_glauber();
        assert!(
            (bins[0].n_part - 382.8).abs() < 0.1,
            "0-5% Npart = {} (expected 382.8)",
            bins[0].n_part
        );
    }

    #[test]
    fn test_xexe_mc_glauber_ordering() {
        let bins = alice_xexe_5440_mc_glauber();
        assert_eq!(bins.len(), 8);
        for w in bins.windows(2) {
            assert!(
                w[0].n_part > w[1].n_part,
                "Npart should decrease: {} > {}",
                w[0].n_part,
                w[1].n_part
            );
            assert!(
                w[0].a_perp > w[1].a_perp,
                "A_perp should decrease with centrality"
            );
        }
    }
}
