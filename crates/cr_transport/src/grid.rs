//! Logarithmic rigidity grid for the momentum axis of the PTE.
//!
//! Rigidity R (GV) is the canonical momentum variable for cosmic rays:
//!   R = p*c / (Z*e)
//! It is the quantity that controls particle bending radius in a magnetic field.
//! A logarithmic spacing d_ln_r = const ensures equal fractional resolution
//! across 4 decades (0.1 to 1000 GV), which is critical for the power-law
//! spectra that characterize cosmic ray distributions.

/// Logarithmic rigidity grid covering [r_min_gv, r_max_gv].
#[derive(Clone, Debug)]
pub struct RigidityGrid {
    pub n_p: usize,
    pub r_min_gv: f64,
    pub r_max_gv: f64,
    /// ln(R) at bin centers.
    pub ln_r: Vec<f64>,
    /// Uniform log-spacing.
    pub d_ln_r: f64,
}

impl RigidityGrid {
    /// Create a new grid with `n_p` logarithmically spaced bins.
    pub fn new(n_p: usize, r_min_gv: f64, r_max_gv: f64) -> Self {
        assert!(n_p >= 2, "need at least 2 rigidity bins");
        assert!(r_min_gv > 0.0 && r_max_gv > r_min_gv);
        let ln_min = r_min_gv.ln();
        let ln_max = r_max_gv.ln();
        let d_ln_r = (ln_max - ln_min) / (n_p - 1) as f64;
        let ln_r = (0..n_p).map(|i| ln_min + i as f64 * d_ln_r).collect();
        Self { n_p, r_min_gv, r_max_gv, ln_r, d_ln_r }
    }

    /// Rigidity in GV at bin index `i`.
    #[inline]
    pub fn rigidity(&self, i: usize) -> f64 {
        self.ln_r[i].exp()
    }

    /// Non-relativistic kinetic energy in GeV for a particle of rest mass m_rest_gev.
    /// Uses E_kin = sqrt(p^2*c^2 + m^2*c^4) - m*c^2 with p*c = Z*e*R (Z=1 for proton).
    pub fn kinetic_energy_gev(&self, i: usize, m_rest_gev: f64) -> f64 {
        let r_gv = self.rigidity(i);
        // For a singly charged particle (Z=1): p*c = R [in GeV since 1 GV * e = 1 GeV/c]
        let pc = r_gv; // GeV
        (pc * pc + m_rest_gev * m_rest_gev).sqrt() - m_rest_gev
    }

    /// Nearest bin index for rigidity `r_gv`. Clamps to [0, n_p-1].
    pub fn momentum_index(&self, r_gv: f64) -> usize {
        if r_gv <= self.r_min_gv {
            return 0;
        }
        if r_gv >= self.r_max_gv {
            return self.n_p - 1;
        }
        let ln_r = r_gv.ln();
        let i = ((ln_r - self.ln_r[0]) / self.d_ln_r).round() as usize;
        i.min(self.n_p - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_rigidity() {
        let grid = RigidityGrid::new(40, 0.1, 1000.0);
        for i in 0..grid.n_p {
            let r = grid.rigidity(i);
            let j = grid.momentum_index(r);
            assert_eq!(i, j, "round-trip failed at bin {i}");
        }
    }

    #[test]
    fn test_proton_kinetic_energy_consistency() {
        // Proton rest mass 0.938272 GeV
        let m_p = 0.938272_f64;
        let grid = RigidityGrid::new(20, 1.0, 100.0);
        // At R=1 GV, E_kin ~ 0.433 GeV (relativistic)
        let e_kin = grid.kinetic_energy_gev(0, m_p);
        assert!(e_kin > 0.0);
        // E_kin < rest mass at R=1 GV (below relativistic threshold)
        assert!(e_kin < m_p, "E_kin at 1 GV should be sub-rest-mass for proton");
        // Verify monotonically increasing with rigidity
        for i in 1..grid.n_p {
            let e_prev = grid.kinetic_energy_gev(i - 1, m_p);
            let e_curr = grid.kinetic_energy_gev(i, m_p);
            assert!(e_curr > e_prev, "energy not monotone at bin {i}");
        }
    }

    #[test]
    fn test_clamp_behavior() {
        let grid = RigidityGrid::new(10, 0.1, 1000.0);
        assert_eq!(grid.momentum_index(0.0), 0);
        assert_eq!(grid.momentum_index(1e6), grid.n_p - 1);
    }
}
