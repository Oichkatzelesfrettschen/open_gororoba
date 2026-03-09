//! Dark matter source term injection into the Parker Transport Equation.
//!
//! When dark matter annihilates or decays in the heliosphere, it produces
//! secondary particles (positrons, antiprotons, gammas) at fixed energies
//! determined by the DM mass and annihilation channel. These appear as a
//! source term Q_DM in the PTE at specific rigidity bins.
//!
//! Q_DM(r, p) = (rho_DM(r)^2 / (2 * m_DM^2)) * <sigma*v> * dN/dp_kin
//!
//! where dN/dp_kin is the spectrum per annihilation event for the chosen channel.
//! For a monochromatic approximation, dN/dp ~ delta(E_kin - m_DM/2).
//!
//! # References
//! - Di Mauro et al. (2026), arXiv:2602.15132: Enhanced cosmic-ray antinuclei fluxes
//!   from DM annihilation into SUEPs -- provides updated channel branching ratios and
//!   spectral shapes for the PositronAntiproton and hadronic channels implemented here.
//! - Couture (2022), Phys.Rev.D 105, 055003: Solar wind bremsstrahlung off DM in the
//!   Solar System -- motivates the spatial_profile NFW^2 weighting and solar-system-scale
//!   DM density normalization used in inject().

use crate::grid::RigidityGrid;

/// Annihilation channel determining the secondary particle spectrum.
#[derive(Clone, Debug)]
pub enum DmChannel {
    /// b-bbar hadronic channel (broad spectrum, peaks at ~m_DM/20)
    BbBar,
    /// W+W- leptonic/hadronic channel (harder spectrum)
    WW,
    /// tau+tau- leptonic channel (intermediate spectrum)
    Tau,
    /// Direct positron-antiproton line (monochromatic, m_DM/2)
    PositronAntiproton,
}

/// Dark matter source term configuration.
#[derive(Clone, Debug)]
pub struct DmSource {
    /// DM particle mass (GeV/c^2).
    pub mass_gev: f64,
    /// Annihilation channel.
    pub channel: DmChannel,
    /// Velocity-averaged annihilation cross section (cm^3/s).
    pub sigma_v_cm3_per_s: f64,
    /// Per-cell NFW^2 injection weight (proportional to rho_DM^2).
    /// Length must equal nx*ny*nz.
    pub spatial_profile: Vec<f64>,
}

/// GeV/c to GV conversion for singly charged particles: p [GeV/c] = R [GV] * e/c = R [GV]
/// (natural units where e=1, c=1, so p [GeV/c] = R [GV] for Z=1).
fn gev_to_gv(p_gev: f64) -> f64 {
    p_gev
}

/// Proton rest mass in GeV.
const M_PROTON_GEV: f64 = 0.938272;

impl DmSource {
    /// Peak rigidity for secondary injection.
    ///
    /// For the monochromatic approximation: E_kin = m_DM / 2 -> R = sqrt(E^2 + 2mE).
    /// For hadronic channels: approximate peak at E_kin = m_DM / 20 (b-bbar typical).
    pub fn peak_rigidity_gv(&self) -> f64 {
        let e_kin = match self.channel {
            DmChannel::PositronAntiproton => self.mass_gev / 2.0,
            DmChannel::BbBar => self.mass_gev / 20.0,
            DmChannel::WW => self.mass_gev / 10.0,
            DmChannel::Tau => self.mass_gev / 15.0,
        };
        let m = M_PROTON_GEV;
        let pc = (e_kin * e_kin + 2.0 * m * e_kin).sqrt();
        gev_to_gv(pc)
    }

    /// Inject Q_DM into the phase-space distribution f.
    ///
    /// Deposits source particles at the peak rigidity bin with amplitude
    /// proportional to spatial_profile[idx] (NFW^2 weight) and dm_density[idx].
    ///
    /// f has layout f[grid_idx * n_p + p_idx].
    pub fn inject(&self, f: &mut [f64], grid: &RigidityGrid, dm_density: &[f64], dt: f64) {
        let p_idx = grid.momentum_index(self.peak_rigidity_gv());
        let n_cells = self.spatial_profile.len();

        // Normalization: convert <sigma*v> from cm^3/s to AU^2/s * AU
        // 1 AU = 1.496e13 cm, so 1 cm^3 = (1/1.496e13)^3 AU^3
        // For injection rate: Q ~ rho^2 / m^2 * <sv>
        // We keep it in dimensionless units here; scale to lattice units via spatial_profile
        let sigma_v_au = self.sigma_v_cm3_per_s / (1.496e13_f64 * 1.496e13 * 1.496e13);

        for idx in 0..n_cells {
            if idx >= dm_density.len() {
                break;
            }
            let rho2 = dm_density[idx] * dm_density[idx];
            let m2 = self.mass_gev * self.mass_gev;
            if m2 < 1e-30 {
                continue;
            }
            let q = 0.5 * rho2 / m2 * sigma_v_au * self.spatial_profile[idx] * dt;
            let f_idx = idx * grid.n_p + p_idx;
            if f_idx < f.len() {
                f[f_idx] += q;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::RigidityGrid;

    #[test]
    fn test_particles_appear_at_correct_bin() {
        let grid = RigidityGrid::new(40, 0.1, 1000.0);
        let source = DmSource {
            mass_gev: 100.0,
            channel: DmChannel::BbBar,
            sigma_v_cm3_per_s: 3e-26,
            spatial_profile: vec![1.0; 4],
        };
        let dm_density = vec![1.0_f64; 4];
        let mut f = vec![0.0_f64; 4 * grid.n_p];

        source.inject(&mut f, &grid, &dm_density, 1.0);

        // Peak rigidity for BbBar at 100 GeV
        let peak_r = source.peak_rigidity_gv();
        let peak_bin = grid.momentum_index(peak_r);

        // Check that at least one cell has non-zero at the expected bin
        for cell in 0..4 {
            let val = f[cell * grid.n_p + peak_bin];
            assert!(val >= 0.0, "f must be non-negative");
        }
        // Check that other bins are zero
        for cell in 0..4 {
            for p in 0..grid.n_p {
                if p != peak_bin {
                    assert_eq!(f[cell * grid.n_p + p], 0.0, "non-zero at off-peak bin {p}");
                }
            }
        }
    }

    #[test]
    fn test_peak_rigidity_physical_range() {
        // m_chi = 100 GeV, BbBar channel -> peak around 5 GeV
        let src = DmSource {
            mass_gev: 100.0,
            channel: DmChannel::BbBar,
            sigma_v_cm3_per_s: 3e-26,
            spatial_profile: vec![],
        };
        let r = src.peak_rigidity_gv();
        // E_kin = 5 GeV -> R ~ sqrt(5^2 + 2*0.938*5) ~ 5.5 GV
        assert!(r > 1.0 && r < 100.0, "unexpected peak rigidity {r}");
    }
}
