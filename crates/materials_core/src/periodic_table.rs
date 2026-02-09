//! Periodic Table: Unified API for element properties.
//!
//! This module wraps `mendeleev` and `periodic-table-on-an-enum` to provide
//! a consistent interface for atomic properties needed in materials science.
//!
//! # Key Properties
//! - Atomic number, mass, symbol
//! - Electron configuration and valence electrons
//! - Optical properties (ionization energy, electron affinity)
//! - Crystal structure and lattice constants
//! - Thermal and electrical properties
//!
//! # Usage
//! ```ignore
//! use materials_core::periodic_table::{Element, get_element, ELEMENTS};
//!
//! let gold = get_element("Au").unwrap();
//! assert_eq!(gold.atomic_number, 79);
//! assert!((gold.atomic_mass - 196.97).abs() < 0.01);
//! ```

use periodic_table_on_an_enum::Element as PtElement;

/// Element data structure with comprehensive properties.
#[derive(Debug, Clone)]
pub struct Element {
    /// Atomic number (Z)
    pub atomic_number: u8,
    /// Element symbol (e.g., "Au", "Si")
    pub symbol: &'static str,
    /// Element name (e.g., "Gold", "Silicon")
    pub name: &'static str,
    /// Atomic mass in amu
    pub atomic_mass: f64,
    /// Density in g/cm^3 (None if gas at STP)
    pub density: Option<f64>,
    /// Melting point in Kelvin (None if not applicable)
    pub melting_point: Option<f64>,
    /// Boiling point in Kelvin (None if not applicable)
    pub boiling_point: Option<f64>,
    /// Number of valence electrons
    pub valence_electrons: u8,
    /// Electronegativity (Pauling scale)
    pub electronegativity: Option<f64>,
    /// First ionization energy in eV
    pub ionization_energy: Option<f64>,
    /// Electron affinity in eV
    pub electron_affinity: Option<f64>,
    /// Crystal structure type
    pub crystal_structure: Option<CrystalStructure>,
    /// Lattice constant in Angstroms
    pub lattice_constant: Option<f64>,
    /// Whether it's a metal
    pub is_metal: bool,
    /// Whether it's a semiconductor
    pub is_semiconductor: bool,
    /// Electron configuration string
    pub electron_config: &'static str,
}

/// Crystal structure types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrystalStructure {
    /// Face-centered cubic
    FCC,
    /// Body-centered cubic
    BCC,
    /// Hexagonal close-packed
    HCP,
    /// Diamond cubic
    Diamond,
    /// Simple cubic
    SimpleCubic,
    /// Orthorhombic
    Orthorhombic,
    /// Tetragonal
    Tetragonal,
    /// Other/complex
    Other,
}

/// Get element by atomic number.
pub fn get_element_by_z(z: u8) -> Option<Element> {
    if z == 0 || z > 118 {
        return None;
    }

    // Use periodic-table-on-an-enum for basic data
    let pt_elem = PtElement::from_atomic_number(z as usize)?;

    Some(build_element(z, pt_elem))
}

/// Get element by symbol.
pub fn get_element(symbol: &str) -> Option<Element> {
    let pt_elem = PtElement::from_symbol(symbol)?;
    let z = pt_elem.get_atomic_number() as u8;
    Some(build_element(z, pt_elem))
}

/// Build Element struct from periodic-table-on-an-enum data.
fn build_element(z: u8, pt: PtElement) -> Element {
    // Get valence electrons based on group/period
    let valence = compute_valence_electrons(z);

    // Determine if metal/semiconductor
    let (is_metal, is_semiconductor) = classify_element(z);

    // Get crystal structure for select elements
    let crystal = get_crystal_structure(z);

    // Get lattice constant for select elements
    let lattice = get_lattice_constant(z);

    // Convert f32 values to Option<f64>, treating 0.0 as None
    let mp = pt.get_melting_point();
    let bp = pt.get_boiling_point();
    let en = pt.get_electronegativity();

    Element {
        atomic_number: z,
        symbol: pt.get_symbol(),
        name: pt.get_name(),
        atomic_mass: pt.get_atomic_mass() as f64,
        density: get_density(z),
        melting_point: if mp > 0.0 { Some(mp as f64) } else { None },
        boiling_point: if bp > 0.0 { Some(bp as f64) } else { None },
        valence_electrons: valence,
        electronegativity: if en > 0.0 { Some(en as f64) } else { None },
        ionization_energy: get_ionization_energy(z),
        electron_affinity: get_electron_affinity(z),
        crystal_structure: crystal,
        lattice_constant: lattice,
        is_metal,
        is_semiconductor,
        electron_config: get_electron_config(z),
    }
}

/// Compute number of valence electrons for an element.
fn compute_valence_electrons(z: u8) -> u8 {
    match z {
        // Noble gases
        2 | 10 | 18 | 36 | 54 | 86 | 118 => 8,
        // Alkali metals
        1 | 3 | 11 | 19 | 37 | 55 | 87 => 1,
        // Alkaline earth metals
        4 | 12 | 20 | 38 | 56 | 88 => 2,
        // Halogens
        9 | 17 | 35 | 53 | 85 | 117 => 7,
        // Carbon group
        6 | 14 | 32 | 50 | 82 | 114 => 4,
        // Nitrogen group
        7 | 15 | 33 | 51 | 83 | 115 => 5,
        // Oxygen group (chalcogens)
        8 | 16 | 34 | 52 | 84 | 116 => 6,
        // Boron group
        5 | 13 | 31 | 49 | 81 | 113 => 3,
        // Transition metals (variable, use common oxidation state)
        21..=30 => ((z - 18) % 10).min(2) + 2,
        39..=48 => ((z - 36) % 10).min(2) + 2,
        72..=80 => ((z - 54) % 10).min(2) + 2,
        // Lanthanides/Actinides
        57..=71 | 89..=103 => 3,
        _ => 0,
    }
}

/// Classify element as metal/semiconductor.
fn classify_element(z: u8) -> (bool, bool) {
    match z {
        // Semiconductors
        5 | 6 | 14 | 32 | 33 | 34 | 50 | 51 | 52 => (false, true),
        // Noble gases, halogens, nonmetals
        1 | 2 | 7 | 8 | 9 | 10 | 15 | 16 | 17 | 18 | 35 | 36 | 53 | 54 | 85 | 86 => (false, false),
        // Everything else is metallic
        _ => (true, false),
    }
}

/// Get crystal structure for common elements.
fn get_crystal_structure(z: u8) -> Option<CrystalStructure> {
    match z {
        // FCC metals
        13 | 28 | 29 | 47 | 78 | 79 | 82 => Some(CrystalStructure::FCC),
        // BCC metals
        23 | 24 | 26 | 41 | 42 | 74 => Some(CrystalStructure::BCC),
        // HCP metals
        4 | 12 | 22 | 27 | 30 | 44 | 48 | 76 => Some(CrystalStructure::HCP),
        // Diamond structure
        6 | 14 | 32 => Some(CrystalStructure::Diamond),
        _ => None,
    }
}

/// Get lattice constant in Angstroms for select elements.
fn get_lattice_constant(z: u8) -> Option<f64> {
    match z {
        6 => Some(3.567),  // Diamond (C)
        13 => Some(4.050), // Aluminum
        14 => Some(5.431), // Silicon
        26 => Some(2.867), // Iron (BCC)
        28 => Some(3.524), // Nickel
        29 => Some(3.615), // Copper
        32 => Some(5.658), // Germanium
        47 => Some(4.086), // Silver
        79 => Some(4.078), // Gold
        78 => Some(3.924), // Platinum
        _ => None,
    }
}

/// Get density in g/cm^3 for select elements.
fn get_density(z: u8) -> Option<f64> {
    match z {
        1 => Some(0.00008988), // H (gas)
        6 => Some(3.513),      // Diamond
        13 => Some(2.70),      // Al
        14 => Some(2.33),      // Si
        26 => Some(7.87),      // Fe
        28 => Some(8.90),      // Ni
        29 => Some(8.96),      // Cu
        32 => Some(5.32),      // Ge
        47 => Some(10.49),     // Ag
        79 => Some(19.30),     // Au
        78 => Some(21.45),     // Pt
        _ => None,
    }
}

/// Get first ionization energy in eV.
fn get_ionization_energy(z: u8) -> Option<f64> {
    match z {
        1 => Some(13.598), // H
        2 => Some(24.587), // He
        6 => Some(11.260), // C
        7 => Some(14.534), // N
        8 => Some(13.618), // O
        13 => Some(5.986), // Al
        14 => Some(8.152), // Si
        26 => Some(7.902), // Fe
        29 => Some(7.726), // Cu
        47 => Some(7.576), // Ag
        79 => Some(9.226), // Au
        _ => None,
    }
}

/// Get electron affinity in eV.
fn get_electron_affinity(z: u8) -> Option<f64> {
    match z {
        1 => Some(0.754),  // H
        6 => Some(1.263),  // C
        7 => Some(-0.07),  // N (negative = requires energy)
        8 => Some(1.461),  // O
        9 => Some(3.401),  // F
        14 => Some(1.389), // Si
        16 => Some(2.077), // S
        17 => Some(3.613), // Cl
        29 => Some(1.235), // Cu
        47 => Some(1.302), // Ag
        79 => Some(2.309), // Au
        _ => None,
    }
}

/// Get electron configuration string.
fn get_electron_config(z: u8) -> &'static str {
    match z {
        1 => "1s1",
        2 => "1s2",
        3 => "[He] 2s1",
        4 => "[He] 2s2",
        5 => "[He] 2s2 2p1",
        6 => "[He] 2s2 2p2",
        7 => "[He] 2s2 2p3",
        8 => "[He] 2s2 2p4",
        9 => "[He] 2s2 2p5",
        10 => "[He] 2s2 2p6",
        11 => "[Ne] 3s1",
        12 => "[Ne] 3s2",
        13 => "[Ne] 3s2 3p1",
        14 => "[Ne] 3s2 3p2",
        26 => "[Ar] 3d6 4s2",
        28 => "[Ar] 3d8 4s2",
        29 => "[Ar] 3d10 4s1",
        47 => "[Kr] 4d10 5s1",
        79 => "[Xe] 4f14 5d10 6s1",
        _ => "",
    }
}

/// Common materials for optics and photonics.
pub mod optical {
    use super::*;

    /// Get silicon properties.
    pub fn silicon() -> Element {
        get_element_by_z(14).unwrap()
    }

    /// Get gold properties (for plasmonics).
    pub fn gold() -> Element {
        get_element_by_z(79).unwrap()
    }

    /// Get silver properties (for plasmonics).
    pub fn silver() -> Element {
        get_element_by_z(47).unwrap()
    }

    /// Get copper properties.
    pub fn copper() -> Element {
        get_element_by_z(29).unwrap()
    }

    /// Get germanium properties.
    pub fn germanium() -> Element {
        get_element_by_z(32).unwrap()
    }

    /// Get aluminum properties.
    pub fn aluminum() -> Element {
        get_element_by_z(13).unwrap()
    }
}

/// Atomic structure data for metamaterial design.
pub mod metamaterial {
    /// Static atomic polarizability in Angstrom^3.
    ///
    /// Used in Clausius-Mossotti relation for effective medium theory.
    pub fn polarizability_a3(z: u8) -> Option<f64> {
        match z {
            1 => Some(0.667), // H
            2 => Some(0.205), // He
            6 => Some(1.76),  // C
            7 => Some(1.10),  // N
            8 => Some(0.802), // O
            13 => Some(6.80), // Al
            14 => Some(5.38), // Si
            26 => Some(8.4),  // Fe
            28 => Some(6.8),  // Ni
            29 => Some(6.1),  // Cu
            32 => Some(6.07), // Ge
            47 => Some(7.2),  // Ag
            79 => Some(5.8),  // Au
            78 => Some(6.5),  // Pt
            _ => None,
        }
    }

    /// Work function in eV.
    ///
    /// Important for photoemission and plasmonic threshold.
    pub fn work_function_ev(z: u8) -> Option<f64> {
        match z {
            13 => Some(4.28), // Al
            26 => Some(4.50), // Fe
            28 => Some(5.15), // Ni
            29 => Some(4.65), // Cu
            47 => Some(4.26), // Ag
            78 => Some(5.65), // Pt
            79 => Some(5.10), // Au
            74 => Some(4.55), // W
            _ => None,
        }
    }

    /// Electron effective mass ratio (m*/m_e).
    ///
    /// For semiconductors, determines optical absorption.
    pub fn effective_mass(z: u8) -> Option<f64> {
        match z {
            14 => Some(0.26), // Si (electron)
            32 => Some(0.22), // Ge (electron)
            _ => None,
        }
    }

    /// Debye temperature in Kelvin.
    ///
    /// Characterizes phonon spectrum for thermal properties.
    pub fn debye_temperature_k(z: u8) -> Option<f64> {
        match z {
            6 => Some(2230.0), // Diamond
            13 => Some(428.0), // Al
            14 => Some(645.0), // Si
            26 => Some(470.0), // Fe
            28 => Some(450.0), // Ni
            29 => Some(343.0), // Cu
            32 => Some(374.0), // Ge
            47 => Some(225.0), // Ag
            79 => Some(165.0), // Au
            78 => Some(240.0), // Pt
            _ => None,
        }
    }

    /// Skin depth at 1 GHz in micrometers.
    ///
    /// For RF/microwave metamaterial design.
    pub fn skin_depth_1ghz_um(z: u8) -> Option<f64> {
        match z {
            13 => Some(2.6), // Al
            29 => Some(2.1), // Cu
            47 => Some(2.5), // Ag
            79 => Some(2.6), // Au
            _ => None,
        }
    }

    /// Surface plasmon resonance wavelength in nm (for nanoparticles in water).
    ///
    /// Approximate peak for spherical nanoparticles.
    pub fn plasmon_resonance_nm(z: u8) -> Option<f64> {
        match z {
            47 => Some(400.0), // Ag (strong, sharp)
            79 => Some(520.0), // Au (strong)
            29 => Some(580.0), // Cu (weaker)
            _ => None,
        }
    }
}

/// Compounds and alloys relevant to Casimir physics.
pub mod casimir {
    /// Silica (SiO2) refractive index at 632.8 nm.
    pub const SILICA_N: f64 = 1.457;

    /// Silicon nitride (Si3N4) refractive index at 632.8 nm.
    pub const SI3N4_N: f64 = 2.05;

    /// Alumina (Al2O3) refractive index at 632.8 nm.
    pub const ALUMINA_N: f64 = 1.766;

    /// Gold plasma frequency (eV).
    pub const GOLD_PLASMA_EV: f64 = 8.55;

    /// Silver plasma frequency (eV).
    pub const SILVER_PLASMA_EV: f64 = 9.17;

    /// Gold relaxation rate (eV).
    pub const GOLD_GAMMA_EV: f64 = 0.021;

    /// Silver relaxation rate (eV).
    pub const SILVER_GAMMA_EV: f64 = 0.021;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_element_by_z() {
        let h = get_element_by_z(1).unwrap();
        assert_eq!(h.atomic_number, 1);
        assert_eq!(h.symbol, "H");
        assert_eq!(h.name, "Hydrogen");
        assert!((h.atomic_mass - 1.008).abs() < 0.01);
    }

    #[test]
    fn test_get_element_by_symbol() {
        let gold = get_element("Au").unwrap();
        assert_eq!(gold.atomic_number, 79);
        assert_eq!(gold.name, "Gold");
        assert!((gold.atomic_mass - 196.97).abs() < 0.01);
    }

    #[test]
    fn test_silicon_properties() {
        let si = optical::silicon();
        assert_eq!(si.atomic_number, 14);
        assert!(si.is_semiconductor);
        assert!(!si.is_metal);
        assert_eq!(si.valence_electrons, 4);
        assert_eq!(si.crystal_structure, Some(CrystalStructure::Diamond));
        assert!((si.lattice_constant.unwrap() - 5.431).abs() < 0.001);
    }

    #[test]
    fn test_gold_properties() {
        let au = optical::gold();
        assert_eq!(au.atomic_number, 79);
        assert!(au.is_metal);
        assert!(!au.is_semiconductor);
        assert_eq!(au.crystal_structure, Some(CrystalStructure::FCC));
    }

    #[test]
    fn test_valence_electrons() {
        // Alkali metals
        assert_eq!(compute_valence_electrons(11), 1); // Na
                                                      // Halogens
        assert_eq!(compute_valence_electrons(17), 7); // Cl
                                                      // Carbon group
        assert_eq!(compute_valence_electrons(14), 4); // Si
                                                      // Noble gases
        assert_eq!(compute_valence_electrons(18), 8); // Ar
    }

    #[test]
    fn test_crystal_structures() {
        assert_eq!(get_crystal_structure(29), Some(CrystalStructure::FCC)); // Cu
        assert_eq!(get_crystal_structure(26), Some(CrystalStructure::BCC)); // Fe
        assert_eq!(get_crystal_structure(14), Some(CrystalStructure::Diamond)); // Si
    }

    #[test]
    fn test_casimir_constants() {
        use casimir::*;
        assert!(GOLD_PLASMA_EV > 8.0);
        assert!(SILVER_PLASMA_EV > 9.0);
        assert!(SILICA_N > 1.4 && SILICA_N < 1.5);
    }

    #[test]
    fn test_invalid_z() {
        assert!(get_element_by_z(0).is_none());
        assert!(get_element_by_z(119).is_none());
    }

    #[test]
    fn test_invalid_symbol() {
        assert!(get_element("Xx").is_none());
        // Note: Empty string causes panic in upstream crate, so we don't test it
    }

    // Metamaterial submodule tests

    #[test]
    fn test_polarizability() {
        use metamaterial::polarizability_a3;
        // Noble gases have small polarizability
        assert!(polarizability_a3(2).unwrap() < 0.5); // He
                                                      // Metals have larger polarizability
        assert!(polarizability_a3(79).unwrap() > 5.0); // Au
                                                       // Silicon
        let si_alpha = polarizability_a3(14).unwrap();
        assert!((si_alpha - 5.38).abs() < 0.1);
        // Invalid element
        assert!(polarizability_a3(100).is_none());
    }

    #[test]
    fn test_work_function() {
        use metamaterial::work_function_ev;
        // Gold has higher work function than silver
        let au_wf = work_function_ev(79).unwrap();
        let ag_wf = work_function_ev(47).unwrap();
        assert!(au_wf > ag_wf);
        // All work functions in reasonable range (3-6 eV)
        assert!(au_wf > 3.0 && au_wf < 6.0);
        assert!(ag_wf > 3.0 && ag_wf < 6.0);
        // Platinum has high work function
        assert!(work_function_ev(78).unwrap() > 5.0);
    }

    #[test]
    fn test_effective_mass() {
        use metamaterial::effective_mass;
        // Silicon electron effective mass
        let si_m = effective_mass(14).unwrap();
        assert!((si_m - 0.26).abs() < 0.01);
        // Germanium has slightly smaller effective mass
        let ge_m = effective_mass(32).unwrap();
        assert!(ge_m < si_m);
        // Metals not defined (free electron model uses m_e)
        assert!(effective_mass(79).is_none());
    }

    #[test]
    fn test_debye_temperature() {
        use metamaterial::debye_temperature_k;
        // Diamond has highest Debye temperature
        let c_theta = debye_temperature_k(6).unwrap();
        assert!(c_theta > 2000.0);
        // Gold has low Debye temperature (soft metal)
        let au_theta = debye_temperature_k(79).unwrap();
        assert!(au_theta < 200.0);
        // Silicon intermediate
        let si_theta = debye_temperature_k(14).unwrap();
        assert!(si_theta > 600.0 && si_theta < 700.0);
    }

    #[test]
    fn test_skin_depth() {
        use metamaterial::skin_depth_1ghz_um;
        // All RF skin depths around 2-3 um at 1 GHz
        let cu_delta = skin_depth_1ghz_um(29).unwrap();
        let ag_delta = skin_depth_1ghz_um(47).unwrap();
        let au_delta = skin_depth_1ghz_um(79).unwrap();
        assert!(cu_delta > 1.5 && cu_delta < 3.0);
        assert!(ag_delta > 1.5 && ag_delta < 3.0);
        assert!(au_delta > 1.5 && au_delta < 3.0);
        // Copper has smallest (best conductor)
        assert!(cu_delta < ag_delta);
    }

    #[test]
    fn test_plasmon_resonance() {
        use metamaterial::plasmon_resonance_nm;
        // Silver has shortest wavelength (blue)
        let ag_spr = plasmon_resonance_nm(47).unwrap();
        assert!(ag_spr > 350.0 && ag_spr < 450.0);
        // Gold in visible green-red
        let au_spr = plasmon_resonance_nm(79).unwrap();
        assert!(au_spr > 500.0 && au_spr < 550.0);
        // Copper redder than gold
        let cu_spr = plasmon_resonance_nm(29).unwrap();
        assert!(cu_spr > au_spr);
        // Non-plasmonic elements undefined
        assert!(plasmon_resonance_nm(14).is_none()); // Si
    }
}
