//! Magnonic crystal physics: LLG linearization, YIG constants, Holstein-Primakoff.
//!
//! The linearized Landau-Lifshitz-Gilbert (LLG) equation for small-amplitude
//! spin waves in a perpendicularly magnetized thin film maps to a Schrodinger
//! equation (Kaman et al. 2026, Eq. 3), enabling the full band-theory /
//! Berry curvature / Chern number machinery to describe magnons.
//!
//! # Physical Constants
//!
//! YIG (yttrium iron garnet, Y3Fe5O12) is the canonical low-damping magnetic
//! insulator:
//! - Saturation magnetization M_s = 1.4e5 A/m
//! - Gilbert damping alpha ~ 1e-4 (ultra-low)
//! - Exchange stiffness A_ex = 3.5e-12 J/m
//! - Angular gyromagnetic ratio gamma = 2*pi*28e9 rad/(s*T)
//!
//! # Literature
//!
//! - Kaman, Lim, Liu, Hoffmann (2026): arXiv:2601.03210v2

use std::f64::consts::PI;

/// Vacuum permeability reference value mu_0 in T*m/A.
pub const MU_0: f64 = 1.256_637_062_12e-6;

/// Bohr magneton in J/T.
pub const MU_B: f64 = 9.274_010_078_3e-24;

/// Lande g-factor for Fe3+ in YIG.
pub const G_LANDE_YIG: f64 = 2.0;

/// YIG material parameters.
#[derive(Clone, Debug)]
pub struct YigParams {
    /// Saturation magnetization M_s (A/m).
    pub m_s: f64,
    /// Gilbert damping parameter (dimensionless).
    pub alpha: f64,
    /// Exchange stiffness A_ex (J/m).
    pub a_ex: f64,
    /// Film thickness (m).
    pub thickness: f64,
    /// Gyromagnetic ratio gamma (rad/(s*T)).
    pub gamma: f64,
}

impl YigParams {
    /// YIG reference parameter set; source-table reproduction is a separate predicate.
    pub fn kaman_standard() -> Self {
        Self {
            m_s: 1.4e5,
            alpha: 1e-4,
            a_ex: 3.5e-12,
            thickness: 20e-9,
            gamma: 2.0 * PI * 28.0e9,
        }
    }

    /// Exchange length l_ex = sqrt(2*A_ex / (mu_0 * M_s^2)) in meters.
    pub fn exchange_length(&self) -> f64 {
        (2.0 * self.a_ex / (MU_0 * self.m_s * self.m_s)).sqrt()
    }

    /// Angular-frequency dispersion coefficient gamma*2*A_ex/M_s in rad*m^2/s.
    /// The exchange field coefficient 2*A_ex/M_s carries T*m^2, matching gamma in rad/(s*T).
    pub fn exchange_stiffness_freq(&self) -> f64 {
        self.gamma * 2.0 * self.a_ex / self.m_s
    }
}

/// Magnonic crystal geometry (hexagonal antidot array).
#[derive(Clone, Debug)]
pub struct MagnonicCrystalGeometry {
    /// Lattice constant a (m).
    pub lattice_constant: f64,
    /// Hole diameter / lattice constant ratio d/a (dimensionless).
    pub hole_ratio: f64,
    /// External magnetic field B_ext (T), perpendicular to film.
    pub b_ext: f64,
}

impl MagnonicCrystalGeometry {
    /// Repository reference geometry: a=400nm, d/a=0.6, B=50mT.
    pub fn kaman_standard() -> Self {
        Self {
            lattice_constant: 400e-9,
            hole_ratio: 0.6,
            b_ext: 50e-3,
        }
    }

    /// Hole diameter in meters.
    pub fn hole_diameter(&self) -> f64 {
        self.hole_ratio * self.lattice_constant
    }
}

/// Homogeneous magnon frequency (no patterning, k-dependent).
///
/// omega(k) = gamma * (B_ext + mu_0*M_s) + D_ex * k^2
///
/// The additive effective-field convention is a repository model; its geometry
/// and demagnetizing-field interpretation require independent source validation.
/// The exchange term and field term both return angular frequency in rad/s.
pub fn magnon_frequency_homogeneous(
    yig: &YigParams,
    geom: &MagnonicCrystalGeometry,
    k: f64,
) -> f64 {
    let omega_0 = yig.gamma * MU_0 * (geom.b_ext / MU_0 + yig.m_s);
    let d_ex = yig.exchange_stiffness_freq();
    omega_0 + d_ex * k * k
}

/// Holstein-Primakoff validity bound: maximum magnon occupation number.
///
/// The HP transformation is valid when `<n> << 2S`, where S is the spin
/// quantum number per magnetic unit cell. For YIG with 20 Fe3+ ions per
/// formula unit (S=5/2 each), the effective spin is large.
///
/// Returns 2*S (the upper bound on magnon number per site).
pub fn hp_validity_bound(spin_per_site: f64) -> f64 {
    2.0 * spin_per_site
}

/// Effective spin per unit cell for YIG.
///
/// YIG has 20 Fe3+ ions per formula unit, but with two sublattices:
/// 12 octahedral (up) and 8 tetrahedral (down), net = 12*5/2 - 8*5/2 = 10.
/// Per primitive cell with 4 formula units: S_eff = 40.
/// Per "magnetic site" in a simple model: S ~ 14.2 is conventional.
pub fn yig_effective_spin() -> f64 {
    14.2
}

/// Convert frequency from GHz to rad/s.
pub fn ghz_to_rad_per_s(f_ghz: f64) -> f64 {
    f_ghz * 1e9 * 2.0 * PI
}

/// Convert angular frequency from rad/s to GHz.
pub fn rad_per_s_to_ghz(omega: f64) -> f64 {
    omega / (2.0 * PI * 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yig_standard_params() {
        let yig = YigParams::kaman_standard();
        assert!((yig.m_s - 1.4e5).abs() < 1.0);
        assert!((yig.alpha - 1e-4).abs() < 1e-6);
        assert!(yig.exchange_length() > 0.0);
        // Exchange length for YIG is ~17 nm
        let l_ex = yig.exchange_length();
        assert!(l_ex > 10e-9 && l_ex < 30e-9, "l_ex = {} nm", l_ex * 1e9);
    }

    #[test]
    fn test_magnon_frequency_at_k_zero() {
        let yig = YigParams::kaman_standard();
        let geom = MagnonicCrystalGeometry::kaman_standard();
        let omega_0 = magnon_frequency_homogeneous(&yig, &geom, 0.0);
        let f_ghz = rad_per_s_to_ghz(omega_0);
        // FMR frequency for B=50mT, M_s=1.4e5 A/m should be ~5 GHz range
        assert!(
            f_ghz > 1.0 && f_ghz < 20.0,
            "f_0 = {} GHz out of range",
            f_ghz
        );
    }

    #[test]
    fn test_magnon_frequency_increases_with_k() {
        let yig = YigParams::kaman_standard();
        let geom = MagnonicCrystalGeometry::kaman_standard();
        let omega_0 = magnon_frequency_homogeneous(&yig, &geom, 0.0);
        let omega_k = magnon_frequency_homogeneous(&yig, &geom, 1e7);
        assert!(
            omega_k > omega_0,
            "Exchange should increase frequency with k"
        );
    }

    #[test]
    fn external_field_slope_matches_cyclic_gyromagnetic_reference() {
        let yig = YigParams::kaman_standard();
        let lower = MagnonicCrystalGeometry::kaman_standard();
        let mut upper = lower.clone();
        upper.b_ext += 0.25;
        let frequency_difference = rad_per_s_to_ghz(
            magnon_frequency_homogeneous(&yig, &upper, 0.0)
                - magnon_frequency_homogeneous(&yig, &lower, 0.0),
        );
        assert!((frequency_difference - 7.0).abs() < 1e-12);
    }

    #[test]
    fn exchange_shift_matches_independent_si_field_oracle() {
        let yig = YigParams::kaman_standard();
        let geometry = MagnonicCrystalGeometry::kaman_standard();
        let wave_number = 1.0e7;
        let exchange_field_t = 2.0 * 3.5e-12 / 1.4e5 * wave_number * wave_number;
        let expected_shift_ghz = 28.0 * exchange_field_t;
        let observed_shift_ghz = rad_per_s_to_ghz(
            magnon_frequency_homogeneous(&yig, &geometry, wave_number)
                - magnon_frequency_homogeneous(&yig, &geometry, 0.0),
        );
        assert!((observed_shift_ghz - expected_shift_ghz).abs() < 1e-12);
        let doubled_shift_ghz = rad_per_s_to_ghz(
            magnon_frequency_homogeneous(&yig, &geometry, 2.0 * wave_number)
                - magnon_frequency_homogeneous(&yig, &geometry, 0.0),
        );
        assert!((doubled_shift_ghz - 4.0 * expected_shift_ghz).abs() < 1e-12);
    }

    #[test]
    fn test_unit_converter_roundtrip() {
        let f = 3.25;
        let omega = ghz_to_rad_per_s(f);
        let f_back = rad_per_s_to_ghz(omega);
        assert!((f - f_back).abs() < 1e-12, "Roundtrip failed");
    }

    #[test]
    fn test_hp_validity_bound_positive() {
        let bound = hp_validity_bound(yig_effective_spin());
        assert!(bound > 0.0);
        assert!((bound - 28.4).abs() < 0.1, "2S ~ 28.4 for YIG");
    }

    #[test]
    fn test_geometry_hole_diameter() {
        let geom = MagnonicCrystalGeometry::kaman_standard();
        let d = geom.hole_diameter();
        assert!((d - 240e-9).abs() < 1e-12, "d = 0.6 * 400nm = 240nm");
    }
}
