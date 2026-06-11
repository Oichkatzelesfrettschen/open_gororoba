//! Experimental Protocols for Octonionic and E8 Characterization.
//!
//! Provides structured procedures for mapping piezoelectric responses to
//! octonionic tensors, collective mode searching in Fano arrays, and
//! precision assembly verification for E8 lattices.
//!
//! Migrated from Appendix C of the Advanced Theoretical Developments.

use crate::{
    lie::e8::root_system::E8Lattice,
    physics::octonion_field::{Octonion, oct_multiply},
};

/// Results of an octonionic tensor mapping experiment (Protocol C.1.1).
#[derive(Debug, Clone)]
pub struct OctonionicMappingResult {
    /// Fitted octonionic coefficients `[d0, d1, ..., d7]`
    pub coefficients: Octonion,
    /// Mean squared error of the fit
    pub mse: f64,
}

/// Simulation of an octonionic decomposition measurement sequence.
pub fn octonionic_measurement_sequence(stress_tensor: &[f64; 8]) -> Octonion {
    // In a real experiment, this would interface with a lock-in amplifier.
    // Here we simulate the octonionic response: R = D * S
    let d_true = [3.8e-12, 0.8e-12, 0.8e-12, 1.2e-12, 0.9e-12, 0.0, 0.0, 0.0];
    oct_multiply(&d_true, stress_tensor)
}

/// Collective mode characterization for a 7-element Fano Array (Protocol C.2.1).
pub struct FanoArrayProtocol {
    pub n_elements: usize,
    pub coupling_matrix: Vec<Vec<f64>>,
}

impl FanoArrayProtocol {
    pub fn new() -> Self {
        Self {
            n_elements: 7,
            coupling_matrix: vec![vec![0.0; 7]; 7],
        }
    }

    /// Search for collective resonant modes in the array.
    pub fn find_collective_modes(&self, frequencies: &[f64]) -> Vec<(f64, f64)> {
        let mut resonances = Vec::new();
        for &f in frequencies {
            // Simulate collective response magnitude
            // Peak near 1000 Hz for this toy model
            let response = 1.0 / (1.0 + (f - 1000.0).powi(2) * 1e-4);
            if response > 0.8 {
                resonances.push((f, response));
            }
        }
        resonances
    }
}

impl Default for FanoArrayProtocol {
    fn default() -> Self {
        Self::new()
    }
}

/// Precision assembly and mounting for an E8 Array (Protocol C.3.1).
pub struct E8AssemblySystem {
    pub lattice: E8Lattice,
    pub position_tolerance_m: f64,
}

impl E8AssemblySystem {
    pub fn new() -> Self {
        Self {
            lattice: E8Lattice::new(),
            position_tolerance_m: 10e-6, // 10 micrometers
        }
    }

    /// Verify rod placement against E8 root positions.
    pub fn verify_placement(&self, rod_id: usize, measured_pos: [f64; 8]) -> bool {
        if let Some(root) = self.lattice.roots.get(rod_id) {
            let mut dist_sq = 0.0;
            for (index, coordinate) in measured_pos.iter().enumerate() {
                dist_sq += (root.coords[index] - coordinate).powi(2);
            }
            dist_sq.sqrt() <= self.position_tolerance_m
        } else {
            false
        }
    }
}

impl Default for E8AssemblySystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Safety system for high-voltage rod drive (Appendix D.1).
pub struct ExperimentalSafetySystem {
    pub max_voltage_v: f64,
    pub max_current_a: f64,
    pub emergency_shutoff: bool,
}

impl ExperimentalSafetySystem {
    pub fn new() -> Self {
        Self {
            max_voltage_v: 500.0,
            max_current_a: 0.01,
            emergency_shutoff: false,
        }
    }

    /// Monitor electrical parameters and trigger shutoff if limits exceeded.
    pub fn monitor(&mut self, voltages: &[f64], currents: &[f64]) -> bool {
        for &v in voltages {
            if v.abs() > self.max_voltage_v {
                self.emergency_shutoff = true;
                return true;
            }
        }
        for &i in currents {
            if i.abs() > self.max_current_a {
                self.emergency_shutoff = true;
                return true;
            }
        }
        false
    }
}

impl Default for ExperimentalSafetySystem {
    fn default() -> Self {
        Self::new()
    }
}
