//! Published Glauber model Npart tables for validation.
//!
//! These reference values are used to validate the optical Glauber model
//! implementation against published ALICE results.
//!
//! Source: ALICE Collaboration, PLB 772 (2017) 567, Table 1 (Pb-Pb 5.02 TeV).

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
///
/// These are the official ALICE Glauber Monte Carlo results using
/// sigma_NN = 67.6 mb and a Woods-Saxon nuclear density profile.
/// Our optical Glauber with hard-sphere density should match within ~2-5%.
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
/// Source: PHENIX Collaboration, PRC 71 (2005) 034908.
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
