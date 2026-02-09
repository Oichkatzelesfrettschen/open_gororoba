//! Relativistic polytropic equation of state.
//!
//! Provides the full relativistic EOS needed for TOV solvers, neutron star
//! structure, and compact object modeling. Unlike the simpler `PolytropicEos`
//! in `gravastar.rs` (which uses Newtonian sound speed dp/drho), this module
//! uses the relativistic sound speed c_s^2 = gamma P / (epsilon + P) where
//! epsilon is the total energy density including rest mass.
//!
//! Derived from Rocq formalization: rocq/theories/Compact/EOS.v
//!
//! # Key equations (geometric units: c = G = 1)
//!
//!   P = K rho^gamma
//!   epsilon = rho + P / (gamma - 1)
//!   c_s^2 = gamma P / (epsilon + P)
//!   h = (epsilon + P) / rho   (specific enthalpy)
//!
//! # References
//!
//! - Shapiro & Teukolsky (1983): Black Holes, White Dwarfs, and Neutron Stars
//! - Oppenheimer & Volkoff (1939): Phys. Rev. 55, 374
//! - Tolman (1939): Phys. Rev. 55, 364

// ============================================================================
// Common adiabatic indices
// ============================================================================

/// Non-relativistic degenerate electrons: gamma = 5/3.
///
/// Used for white dwarf cores and low-density neutron star crust.
/// Polytropic index n = 3/2.
pub const GAMMA_NONREL_DEGENERATE: f64 = 5.0 / 3.0;

/// Ultra-relativistic degenerate electrons: gamma = 4/3.
///
/// Used for massive white dwarfs near the Chandrasekhar limit
/// and radiation-dominated matter. Polytropic index n = 3.
pub const GAMMA_ULTRAREL_DEGENERATE: f64 = 4.0 / 3.0;

/// Stiff equation of state: gamma = 2.
///
/// Maximum stiffness consistent with causality (c_s -> 1 as rho -> inf).
/// Polytropic index n = 1.
pub const GAMMA_STIFF: f64 = 2.0;

/// Radiation-dominated: gamma = 4/3 (same as ultra-relativistic).
pub const GAMMA_RADIATION: f64 = 4.0 / 3.0;

// ============================================================================
// Polytrope struct
// ============================================================================

/// Relativistic polytropic equation of state parameters.
///
/// P = K rho^gamma, where all quantities are in geometric units (c = G = 1).
/// The energy density includes rest mass: epsilon = rho + P/(gamma-1).
#[derive(Clone, Copy, Debug)]
pub struct Polytrope {
    /// Polytropic constant (units depend on gamma)
    pub k: f64,
    /// Adiabatic index (must be > 1 for valid thermodynamics)
    pub gamma: f64,
}

impl Polytrope {
    /// Create a new polytropic EOS with validation.
    ///
    /// # Panics
    /// Panics if k <= 0 or gamma <= 1.
    pub fn new(k: f64, gamma: f64) -> Self {
        assert!(k > 0.0, "K must be positive, got {k}");
        assert!(gamma > 1.0, "gamma must be > 1, got {gamma}");
        Self { k, gamma }
    }

    /// Non-relativistic degenerate electron gas (gamma = 5/3).
    pub fn nonrel_degenerate(k: f64) -> Self {
        Self::new(k, GAMMA_NONREL_DEGENERATE)
    }

    /// Ultra-relativistic degenerate electron gas (gamma = 4/3).
    pub fn ultrarel_degenerate(k: f64) -> Self {
        Self::new(k, GAMMA_ULTRAREL_DEGENERATE)
    }

    /// Stiff EOS (gamma = 2), the maximum causality-respecting stiffness.
    pub fn stiff(k: f64) -> Self {
        Self::new(k, GAMMA_STIFF)
    }

    /// Whether this EOS has valid parameters: K > 0 and gamma > 1.
    pub fn is_valid(&self) -> bool {
        self.k > 0.0 && self.gamma > 1.0
    }

    /// Pressure from rest-mass density: P = K rho^gamma.
    pub fn pressure(&self, rho: f64) -> f64 {
        self.k * rho.powf(self.gamma)
    }

    /// Total energy density: epsilon = rho + P/(gamma - 1).
    ///
    /// In geometric units (c = 1): includes rest mass energy (rho)
    /// plus internal (thermal) energy P/(gamma - 1).
    pub fn energy_density(&self, rho: f64) -> f64 {
        let p = self.pressure(rho);
        rho + p / (self.gamma - 1.0)
    }

    /// Relativistic sound speed squared: c_s^2 = gamma P / (epsilon + P).
    ///
    /// This differs from the Newtonian dP/drho = K gamma rho^{gamma-1}.
    /// The relativistic formula ensures c_s^2 <= 1 for gamma <= 2.
    pub fn sound_speed_sq(&self, rho: f64) -> f64 {
        let p = self.pressure(rho);
        let eps = self.energy_density(rho);
        self.gamma * p / (eps + p)
    }

    /// Relativistic sound speed: c_s = sqrt(gamma P / (epsilon + P)).
    pub fn sound_speed(&self, rho: f64) -> f64 {
        self.sound_speed_sq(rho).sqrt()
    }

    /// Specific enthalpy: h = (epsilon + P) / rho.
    ///
    /// Used in relativistic hydrodynamics and TOV integration.
    /// For a polytrope: h = 1 + gamma/(gamma-1) K rho^{gamma-1}.
    pub fn enthalpy(&self, rho: f64) -> f64 {
        let p = self.pressure(rho);
        let eps = self.energy_density(rho);
        (eps + p) / rho
    }

    /// Log-enthalpy: H = ln(h).
    ///
    /// Convenient variable for TOV integration (monotonically decreasing
    /// from center to surface).
    pub fn log_enthalpy(&self, rho: f64) -> f64 {
        self.enthalpy(rho).ln()
    }

    /// Rest-mass density from pressure (inverse relation).
    ///
    /// rho = (P / K)^{1/gamma}
    pub fn density_from_pressure(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        (p / self.k).powf(1.0 / self.gamma)
    }

    /// Rest-mass density from specific enthalpy (inverse relation).
    ///
    /// From h = 1 + gamma/(gamma-1) K rho^{gamma-1}:
    ///   rho = ((h-1)(gamma-1) / (gamma K))^{1/(gamma-1)}
    pub fn density_from_enthalpy(&self, h: f64) -> f64 {
        if h <= 1.0 {
            return 0.0;
        }
        let factor = (h - 1.0) * (self.gamma - 1.0) / (self.gamma * self.k);
        factor.powf(1.0 / (self.gamma - 1.0))
    }

    /// Check if EOS is causal at given density (c_s <= 1).
    pub fn is_causal(&self, rho: f64) -> bool {
        self.sound_speed_sq(rho) <= 1.0
    }

    /// Check if EOS is causal at all densities.
    ///
    /// For polytropes, this is guaranteed when gamma <= 2
    /// (the stiff EOS is the limiting causal case).
    pub fn is_globally_causal(&self) -> bool {
        self.gamma <= GAMMA_STIFF
    }
}

// ============================================================================
// Polytropic index conversions
// ============================================================================

/// Polytropic index from adiabatic index: n = 1/(gamma - 1).
///
/// Common indices:
///   n = 3/2 (gamma = 5/3): non-relativistic degenerate
///   n = 3   (gamma = 4/3): relativistic degenerate / Chandrasekhar limit
///   n = 1   (gamma = 2):   stiff EOS
pub fn polytropic_index(gamma: f64) -> f64 {
    1.0 / (gamma - 1.0)
}

/// Adiabatic index from polytropic index: gamma = 1 + 1/n.
pub fn gamma_from_index(n: f64) -> f64 {
    1.0 + 1.0 / n
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Construction and validation --

    #[test]
    fn test_valid_polytrope() {
        let eos = Polytrope::new(1.0, 2.0);
        assert!(eos.is_valid());
    }

    #[test]
    #[should_panic(expected = "gamma must be > 1")]
    fn test_invalid_gamma() {
        let _ = Polytrope::new(1.0, 1.0); // gamma = 1 not allowed
    }

    #[test]
    #[should_panic(expected = "K must be positive")]
    fn test_invalid_k() {
        let _ = Polytrope::new(-1.0, 2.0);
    }

    // -- Named constructors --

    #[test]
    fn test_nonrel_degenerate() {
        let eos = Polytrope::nonrel_degenerate(1.0);
        assert!((eos.gamma - 5.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_ultrarel_degenerate() {
        let eos = Polytrope::ultrarel_degenerate(1.0);
        assert!((eos.gamma - 4.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_stiff() {
        let eos = Polytrope::stiff(1.0);
        assert!((eos.gamma - 2.0).abs() < 1e-14);
    }

    // -- Pressure --

    #[test]
    fn test_pressure() {
        let eos = Polytrope::new(2.0, 2.0);
        // P = 2 * 3^2 = 18
        assert!((eos.pressure(3.0) - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_pressure_stiff_is_linear() {
        // For gamma=2, K=1: P = rho^2
        let eos = Polytrope::stiff(1.0);
        assert!((eos.pressure(5.0) - 25.0).abs() < 1e-10);
    }

    // -- Energy density --

    #[test]
    fn test_energy_exceeds_rest_mass() {
        let eos = Polytrope::new(1.0, 2.0);
        let rho = 1.0;
        let eps = eos.energy_density(rho);
        // eps = rho + P/(gamma-1) = 1 + 1/1 = 2
        assert!(eps > rho, "eps = {eps}, rho = {rho}");
        assert!((eps - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_energy_density_formula() {
        let eos = Polytrope::new(0.5, 5.0 / 3.0);
        let rho = 2.0;
        let p = eos.pressure(rho); // 0.5 * 2^{5/3}
        let eps = eos.energy_density(rho); // 2 + p / (5/3 - 1) = 2 + p / (2/3) = 2 + 1.5*p
        let expected = rho + p / (5.0 / 3.0 - 1.0);
        assert!((eps - expected).abs() < 1e-10);
    }

    // -- Sound speed --

    #[test]
    fn test_relativistic_sound_speed_subluminal() {
        // For gamma <= 2, c_s^2 should always be < 1
        let eos = Polytrope::stiff(1.0);
        for &rho in &[0.01, 0.1, 1.0, 10.0, 100.0, 1e6] {
            let cs2 = eos.sound_speed_sq(rho);
            assert!(cs2 < 1.0, "c_s^2 = {cs2} at rho = {rho} (should be < 1)");
        }
    }

    #[test]
    fn test_sound_speed_approaches_limit() {
        // For stiff EOS (gamma=2), c_s^2 -> 1 as rho -> infinity
        let eos = Polytrope::stiff(1.0);
        let cs2_low = eos.sound_speed_sq(0.01);
        let cs2_high = eos.sound_speed_sq(1e10);
        assert!(
            cs2_high > cs2_low,
            "Sound speed should increase with density"
        );
        assert!(cs2_high > 0.99, "c_s^2 at high density should approach 1");
    }

    #[test]
    fn test_sound_speed_positive() {
        let eos = Polytrope::nonrel_degenerate(1.0);
        let cs = eos.sound_speed(1.0);
        assert!(cs > 0.0 && cs < 1.0);
    }

    // -- Enthalpy --

    #[test]
    fn test_enthalpy_exceeds_unity() {
        // h = (eps + P)/rho > 1 for any positive pressure
        let eos = Polytrope::new(1.0, 2.0);
        let h = eos.enthalpy(1.0);
        assert!(h > 1.0, "h = {h}");
    }

    #[test]
    fn test_enthalpy_formula() {
        let eos = Polytrope::new(1.0, 2.0);
        let rho = 2.0;
        // P = 2^2 = 4, eps = 2 + 4/1 = 6, h = (6+4)/2 = 5
        let h = eos.enthalpy(rho);
        assert!((h - 5.0).abs() < 1e-14, "h = {h}");
    }

    #[test]
    fn test_log_enthalpy_positive() {
        let eos = Polytrope::new(1.0, 2.0);
        let lh = eos.log_enthalpy(1.0);
        assert!(lh > 0.0, "H = {lh} (should be positive since h > 1)");
    }

    // -- Inverse relations --

    #[test]
    fn test_density_from_pressure_roundtrip() {
        let eos = Polytrope::new(0.5, 5.0 / 3.0);
        let rho = 3.0;
        let p = eos.pressure(rho);
        let rho_recovered = eos.density_from_pressure(p);
        assert!(
            (rho_recovered - rho).abs() < 1e-10,
            "rho = {rho}, recovered = {rho_recovered}"
        );
    }

    #[test]
    fn test_density_from_enthalpy_roundtrip() {
        let eos = Polytrope::new(1.0, 2.0);
        let rho = 2.0;
        let h = eos.enthalpy(rho);
        let rho_recovered = eos.density_from_enthalpy(h);
        assert!(
            (rho_recovered - rho).abs() < 1e-10,
            "rho = {rho}, recovered = {rho_recovered}"
        );
    }

    #[test]
    fn test_density_from_zero_pressure() {
        let eos = Polytrope::new(1.0, 2.0);
        assert!((eos.density_from_pressure(0.0)).abs() < 1e-14);
    }

    #[test]
    fn test_density_from_enthalpy_at_surface() {
        // At surface: h = 1 (zero pressure), rho -> 0
        let eos = Polytrope::new(1.0, 2.0);
        let rho = eos.density_from_enthalpy(1.0);
        assert!(rho.abs() < 1e-14, "rho at h=1: {rho}");
    }

    // -- Causality --

    #[test]
    fn test_is_causal_stiff() {
        let eos = Polytrope::stiff(1.0);
        assert!(eos.is_causal(1.0));
        assert!(eos.is_causal(1e6));
        assert!(eos.is_globally_causal());
    }

    #[test]
    fn test_is_globally_causal_nonrel() {
        let eos = Polytrope::nonrel_degenerate(1.0);
        assert!(eos.is_globally_causal()); // gamma = 5/3 < 2
    }

    // -- Polytropic index conversions --

    #[test]
    fn test_polytropic_index() {
        assert!((polytropic_index(5.0 / 3.0) - 1.5).abs() < 1e-14); // n = 3/2
        assert!((polytropic_index(4.0 / 3.0) - 3.0).abs() < 1e-14); // n = 3
        assert!((polytropic_index(2.0) - 1.0).abs() < 1e-14); // n = 1
    }

    #[test]
    fn test_gamma_from_index() {
        assert!((gamma_from_index(1.5) - 5.0 / 3.0).abs() < 1e-14);
        assert!((gamma_from_index(3.0) - 4.0 / 3.0).abs() < 1e-14);
        assert!((gamma_from_index(1.0) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_index_roundtrip() {
        for &gamma in &[1.1, 4.0 / 3.0, 5.0 / 3.0, 2.0, 3.0] {
            let n = polytropic_index(gamma);
            let gamma_back = gamma_from_index(n);
            assert!(
                (gamma_back - gamma).abs() < 1e-12,
                "gamma = {gamma}, roundtrip = {gamma_back}"
            );
        }
    }

    // -- Newtonian vs relativistic sound speed comparison --

    #[test]
    fn test_relativistic_differs_from_newtonian() {
        // Newtonian: c_s^2 = K gamma rho^{gamma-1}
        // Relativistic: c_s^2 = gamma P / (eps + P)
        // They agree in the non-relativistic limit (P << rho) but diverge
        // when P is comparable to rho.
        let eos = Polytrope::stiff(1.0); // gamma = 2, K = 1
        let rho: f64 = 10.0;

        let newtonian = eos.k * eos.gamma * rho.powf(eos.gamma - 1.0);
        let relativistic = eos.sound_speed_sq(rho);

        // Newtonian: 1 * 2 * 10 = 20 (superluminal!)
        // Relativistic: must be < 1
        assert!(newtonian > 1.0, "Newtonian c_s^2 = {newtonian}");
        assert!(relativistic < 1.0, "Relativistic c_s^2 = {relativistic}");
    }
}
