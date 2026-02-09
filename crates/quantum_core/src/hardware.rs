//! Quantum Hardware Abstraction Layer.
//!
//! Provides traits and types for modeling different quantum computing platforms:
//! - Neutral atom arrays (Rydberg blockade)
//! - Superconducting qubits (IBM, Google)
//! - Trapped ions (IonQ, Quantinuum)
//! - Photonic systems
//!
//! # Architecture
//!
//! The `HardwareProfile` trait defines the interface for hardware-specific
//! properties like qubit count, connectivity, gate fidelities, and timing.
//! Concrete implementations (NeutralAtomProfile, SuperconductingProfile) provide
//! platform-specific parameters.
//!
//! # Literature
//!
//! - Preskill (2018): Quantum Computing in the NISQ era and beyond
//! - Henriet et al. (2020): Quantum computing with neutral atoms
//! - Arute et al. (2019): Quantum supremacy using superconducting processor

use std::collections::HashSet;

/// Qubit connectivity topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QubitTopology {
    /// Linear chain: qubit i connects to i-1 and i+1.
    Linear,
    /// 2D grid: qubit (r,c) connects to (r+/-1, c) and (r, c+/-1).
    Grid { rows: usize, cols: usize },
    /// All-to-all connectivity (e.g., trapped ions, some neutral atoms).
    AllToAll,
    /// Custom connectivity defined by edge list.
    Custom { edges: HashSet<(usize, usize)> },
}

impl QubitTopology {
    /// Get all edges in the connectivity graph.
    pub fn edges(&self, n_qubits: usize) -> Vec<(usize, usize)> {
        match self {
            QubitTopology::Linear => (0..n_qubits.saturating_sub(1))
                .map(|i| (i, i + 1))
                .collect(),
            QubitTopology::Grid { rows, cols } => {
                let mut edges = Vec::new();
                for r in 0..*rows {
                    for c in 0..*cols {
                        let idx = r * cols + c;
                        // Right neighbor
                        if c + 1 < *cols {
                            edges.push((idx, idx + 1));
                        }
                        // Down neighbor
                        if r + 1 < *rows {
                            edges.push((idx, idx + cols));
                        }
                    }
                }
                edges
            }
            QubitTopology::AllToAll => {
                let mut edges = Vec::new();
                for i in 0..n_qubits {
                    for j in (i + 1)..n_qubits {
                        edges.push((i, j));
                    }
                }
                edges
            }
            QubitTopology::Custom { edges } => edges.iter().cloned().collect(),
        }
    }

    /// Check if two qubits are directly connected.
    pub fn are_connected(&self, n_qubits: usize, q1: usize, q2: usize) -> bool {
        let (a, b) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
        match self {
            QubitTopology::Linear => b == a + 1,
            QubitTopology::Grid { rows: _, cols } => {
                let (r1, c1) = (a / cols, a % cols);
                let (r2, c2) = (b / cols, b % cols);
                (r1 == r2 && c2 == c1 + 1) || (c1 == c2 && r2 == r1 + 1)
            }
            QubitTopology::AllToAll => a < n_qubits && b < n_qubits,
            QubitTopology::Custom { edges } => edges.contains(&(a, b)) || edges.contains(&(b, a)),
        }
    }
}

/// Native gate set available on hardware.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NativeGate {
    /// Single-qubit X rotation (often decomposed from arbitrary rotations).
    Rx,
    /// Single-qubit Y rotation.
    Ry,
    /// Single-qubit Z rotation (often virtual/free).
    Rz,
    /// Hadamard gate.
    H,
    /// T gate (pi/8 rotation).
    T,
    /// S gate (pi/4 rotation).
    S,
    /// Controlled-Z gate (common in superconducting).
    CZ,
    /// Controlled-X (CNOT) gate.
    CX,
    /// iSWAP gate (common in superconducting).
    ISWAP,
    /// sqrt(iSWAP) gate.
    SqrtISWAP,
    /// Molmer-Sorensen gate (trapped ions).
    MS,
    /// CZ-like gate from Rydberg blockade (neutral atoms).
    RydbergCZ,
    /// Arbitrary single-qubit unitary.
    U3,
}

/// Gate timing information.
#[derive(Debug, Clone)]
pub struct GateTiming {
    /// Gate execution time in microseconds.
    pub duration_us: f64,
    /// Setup/preparation time in microseconds.
    pub setup_us: f64,
}

impl GateTiming {
    pub fn new(duration_us: f64) -> Self {
        Self {
            duration_us,
            setup_us: 0.0,
        }
    }

    pub fn with_setup(duration_us: f64, setup_us: f64) -> Self {
        Self {
            duration_us,
            setup_us,
        }
    }

    pub fn total_us(&self) -> f64 {
        self.duration_us + self.setup_us
    }
}

/// Error rates for different operations.
#[derive(Debug, Clone)]
pub struct ErrorRates {
    /// Single-qubit gate error rate.
    pub single_qubit: f64,
    /// Two-qubit gate error rate.
    pub two_qubit: f64,
    /// Measurement/readout error rate.
    pub readout: f64,
    /// State preparation error rate.
    pub preparation: f64,
}

impl Default for ErrorRates {
    fn default() -> Self {
        Self {
            single_qubit: 1e-4,
            two_qubit: 1e-2,
            readout: 1e-2,
            preparation: 1e-3,
        }
    }
}

/// Coherence times.
#[derive(Debug, Clone)]
pub struct CoherenceTimes {
    /// T1 relaxation time in microseconds.
    pub t1_us: f64,
    /// T2 dephasing time in microseconds.
    pub t2_us: f64,
}

impl CoherenceTimes {
    pub fn new(t1_us: f64, t2_us: f64) -> Self {
        Self { t1_us, t2_us }
    }

    /// Compute decoherence probability for given gate time.
    pub fn decoherence_prob(&self, gate_time_us: f64) -> f64 {
        let t1_decay = 1.0 - (-gate_time_us / self.t1_us).exp();
        let t2_decay = 1.0 - (-gate_time_us / self.t2_us).exp();
        // Combined error (simplified model)
        1.0 - (1.0 - t1_decay) * (1.0 - t2_decay)
    }
}

/// Trait for quantum hardware profiles.
///
/// Implementations provide platform-specific parameters for:
/// - Qubit count and connectivity
/// - Native gate set and timing
/// - Error rates and coherence times
/// - Platform-specific constraints
pub trait HardwareProfile: Send + Sync {
    /// Hardware platform name.
    fn name(&self) -> &str;

    /// Number of qubits available.
    fn qubit_count(&self) -> usize;

    /// Qubit connectivity topology.
    fn topology(&self) -> &QubitTopology;

    /// Set of native gates available.
    fn native_gates(&self) -> &[NativeGate];

    /// Check if a gate is natively supported.
    fn supports_gate(&self, gate: &NativeGate) -> bool {
        self.native_gates().contains(gate)
    }

    /// Gate timing for a specific gate type.
    fn gate_timing(&self, gate: &NativeGate) -> GateTiming;

    /// Error rates for operations.
    fn error_rates(&self) -> &ErrorRates;

    /// Coherence times (T1, T2).
    fn coherence_times(&self) -> &CoherenceTimes;

    /// Maximum circuit depth before decoherence dominates.
    fn max_practical_depth(&self) -> usize {
        // Estimate based on T2 and typical gate time
        let avg_gate_time = self.gate_timing(&NativeGate::CZ).total_us();
        let t2 = self.coherence_times().t2_us;
        ((t2 / avg_gate_time) * 0.5) as usize // Conservative: use 50% of T2
    }

    /// Estimate circuit fidelity given depth and two-qubit gate count.
    fn estimate_fidelity(&self, depth: usize, two_qubit_gates: usize) -> f64 {
        let er = self.error_rates();
        let single_qubit_ops = depth * self.qubit_count();
        let single_fidelity = (1.0 - er.single_qubit).powi(single_qubit_ops as i32);
        let two_fidelity = (1.0 - er.two_qubit).powi(two_qubit_gates as i32);
        single_fidelity * two_fidelity
    }
}

/// Ideal hardware with no errors (for testing/simulation).
#[derive(Debug, Clone)]
pub struct IdealHardware {
    n_qubits: usize,
    topology: QubitTopology,
    error_rates: ErrorRates,
    coherence_times: CoherenceTimes,
}

impl IdealHardware {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            topology: QubitTopology::AllToAll,
            error_rates: ErrorRates {
                single_qubit: 0.0,
                two_qubit: 0.0,
                readout: 0.0,
                preparation: 0.0,
            },
            coherence_times: CoherenceTimes::new(f64::MAX, f64::MAX),
        }
    }

    pub fn with_topology(n_qubits: usize, topology: QubitTopology) -> Self {
        Self {
            n_qubits,
            topology,
            error_rates: ErrorRates {
                single_qubit: 0.0,
                two_qubit: 0.0,
                readout: 0.0,
                preparation: 0.0,
            },
            coherence_times: CoherenceTimes::new(f64::MAX, f64::MAX),
        }
    }
}

impl HardwareProfile for IdealHardware {
    fn name(&self) -> &str {
        "Ideal"
    }

    fn qubit_count(&self) -> usize {
        self.n_qubits
    }

    fn topology(&self) -> &QubitTopology {
        &self.topology
    }

    fn native_gates(&self) -> &[NativeGate] {
        &[
            NativeGate::Rx,
            NativeGate::Ry,
            NativeGate::Rz,
            NativeGate::H,
            NativeGate::T,
            NativeGate::S,
            NativeGate::CZ,
            NativeGate::CX,
        ]
    }

    fn gate_timing(&self, _gate: &NativeGate) -> GateTiming {
        GateTiming::new(0.0) // Ideal: instantaneous
    }

    fn error_rates(&self) -> &ErrorRates {
        &self.error_rates
    }

    fn coherence_times(&self) -> &CoherenceTimes {
        &self.coherence_times
    }

    fn max_practical_depth(&self) -> usize {
        usize::MAX
    }

    fn estimate_fidelity(&self, _depth: usize, _two_qubit_gates: usize) -> f64 {
        1.0
    }
}

/// Neutral atom hardware profile (Rydberg blockade systems).
///
/// Models platforms like QuEra Aquila, Pasqal, and Atom Computing.
/// Uses optical tweezers to trap atoms and Rydberg excitation for entanglement.
///
/// Key characteristics:
/// - Long coherence times (hyperfine qubits)
/// - High connectivity (all-to-all within blockade radius)
/// - Native RydbergCZ gate via blockade mechanism
///
/// # Literature
/// - Henriet et al. (2020): Quantum computing with neutral atoms
/// - Ebadi et al. (2021): Quantum phases of matter on a 256-atom processor
#[derive(Debug, Clone)]
pub struct NeutralAtomProfile {
    n_qubits: usize,
    topology: QubitTopology,
    /// Blockade radius in micrometers.
    blockade_radius_um: f64,
    /// Atom spacing in micrometers.
    atom_spacing_um: f64,
    error_rates: ErrorRates,
    coherence_times: CoherenceTimes,
    native_gates: Vec<NativeGate>,
}

impl NeutralAtomProfile {
    /// Create a default neutral atom profile with typical parameters.
    ///
    /// Uses QuEra Aquila-like parameters:
    /// - T1 ~ 10 ms, T2 ~ 1 ms (hyperfine encoding)
    /// - Single-qubit error ~ 3e-4
    /// - Two-qubit error ~ 5e-3
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            topology: QubitTopology::AllToAll, // Within blockade radius
            blockade_radius_um: 10.0,
            atom_spacing_um: 4.0,
            error_rates: ErrorRates {
                single_qubit: 3e-4,
                two_qubit: 5e-3,
                readout: 3e-3,
                preparation: 1e-3,
            },
            coherence_times: CoherenceTimes::new(10_000.0, 1_000.0), // 10ms T1, 1ms T2
            native_gates: vec![
                NativeGate::Rx,
                NativeGate::Ry,
                NativeGate::Rz,
                NativeGate::RydbergCZ,
            ],
        }
    }

    /// Create a neutral atom profile with 2D grid layout.
    ///
    /// Atoms arranged in rows x cols grid with given spacing.
    pub fn with_grid(rows: usize, cols: usize, spacing_um: f64) -> Self {
        let n_qubits = rows * cols;
        Self {
            n_qubits,
            topology: QubitTopology::Grid { rows, cols },
            blockade_radius_um: 10.0,
            atom_spacing_um: spacing_um,
            error_rates: ErrorRates {
                single_qubit: 3e-4,
                two_qubit: 5e-3,
                readout: 3e-3,
                preparation: 1e-3,
            },
            coherence_times: CoherenceTimes::new(10_000.0, 1_000.0),
            native_gates: vec![
                NativeGate::Rx,
                NativeGate::Ry,
                NativeGate::Rz,
                NativeGate::RydbergCZ,
            ],
        }
    }

    /// Builder: set blockade radius.
    pub fn with_blockade_radius(mut self, radius_um: f64) -> Self {
        self.blockade_radius_um = radius_um;
        self
    }

    /// Builder: set error rates.
    pub fn with_error_rates(mut self, rates: ErrorRates) -> Self {
        self.error_rates = rates;
        self
    }

    /// Builder: set coherence times.
    pub fn with_coherence_times(mut self, times: CoherenceTimes) -> Self {
        self.coherence_times = times;
        self
    }

    /// Get the blockade radius.
    pub fn blockade_radius(&self) -> f64 {
        self.blockade_radius_um
    }

    /// Check if two atoms can interact via Rydberg blockade.
    ///
    /// True if their distance is less than the blockade radius.
    pub fn can_interact(&self, q1: usize, q2: usize) -> bool {
        match &self.topology {
            QubitTopology::AllToAll => true,
            QubitTopology::Grid { rows: _, cols } => {
                let (r1, c1) = (q1 / cols, q1 % cols);
                let (r2, c2) = (q2 / cols, q2 % cols);
                let dr = (r1 as f64 - r2 as f64) * self.atom_spacing_um;
                let dc = (c1 as f64 - c2 as f64) * self.atom_spacing_um;
                let dist = (dr * dr + dc * dc).sqrt();
                dist < self.blockade_radius_um
            }
            _ => self.topology.are_connected(self.n_qubits, q1, q2),
        }
    }
}

impl HardwareProfile for NeutralAtomProfile {
    fn name(&self) -> &str {
        "NeutralAtom"
    }

    fn qubit_count(&self) -> usize {
        self.n_qubits
    }

    fn topology(&self) -> &QubitTopology {
        &self.topology
    }

    fn native_gates(&self) -> &[NativeGate] {
        &self.native_gates
    }

    fn gate_timing(&self, gate: &NativeGate) -> GateTiming {
        match gate {
            // Single-qubit gates via Raman transitions
            NativeGate::Rx | NativeGate::Ry => GateTiming::with_setup(0.5, 0.1),
            NativeGate::Rz => GateTiming::new(0.01), // Virtual, nearly free
            // Rydberg CZ gate via blockade
            NativeGate::RydbergCZ => GateTiming::with_setup(0.3, 0.2),
            // Decomposed gates
            NativeGate::H => GateTiming::with_setup(0.5, 0.1),
            NativeGate::CZ | NativeGate::CX => GateTiming::with_setup(0.5, 0.3),
            // Not native, synthesized
            _ => GateTiming::with_setup(1.0, 0.5),
        }
    }

    fn error_rates(&self) -> &ErrorRates {
        &self.error_rates
    }

    fn coherence_times(&self) -> &CoherenceTimes {
        &self.coherence_times
    }
}

/// Superconducting qubit hardware profile.
///
/// Models platforms like IBM Quantum (transmon qubits) and Google Sycamore.
/// Uses microwave pulses for gate operations on fixed-frequency or tunable transmons.
///
/// Key characteristics:
/// - Fixed 2D grid topology with nearest-neighbor connectivity
/// - Fast gates (nanoseconds) but shorter coherence times (microseconds)
/// - Native CZ or iSWAP two-qubit gates
///
/// # Literature
/// - Arute et al. (2019): Quantum supremacy using superconducting processor
/// - Jurcevic et al. (2021): Demonstration of quantum volume 64
#[derive(Debug, Clone)]
pub struct SuperconductingProfile {
    n_qubits: usize,
    topology: QubitTopology,
    /// Hardware vendor/architecture.
    vendor: SuperconductingVendor,
    error_rates: ErrorRates,
    coherence_times: CoherenceTimes,
    native_gates: Vec<NativeGate>,
}

/// Superconducting hardware vendor/architecture variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperconductingVendor {
    /// IBM Quantum systems (heavy-hex topology, CX native).
    Ibm,
    /// Google Sycamore (2D grid, sqrt-iSWAP native).
    Google,
    /// Rigetti (octagonal topology, CZ native).
    Rigetti,
    /// Generic superconducting (user-defined).
    Generic,
}

impl SuperconductingProfile {
    /// Create an IBM-like superconducting profile.
    ///
    /// Uses heavy-hex topology with CX/CZ native gates.
    /// Typical IBM Falcon/Hummingbird parameters.
    pub fn ibm(n_qubits: usize) -> Self {
        // IBM uses heavy-hex topology, approximate as grid for now
        let side = (n_qubits as f64).sqrt().ceil() as usize;
        Self {
            n_qubits,
            topology: QubitTopology::Grid {
                rows: side,
                cols: side,
            },
            vendor: SuperconductingVendor::Ibm,
            error_rates: ErrorRates {
                single_qubit: 2e-4,
                two_qubit: 1e-2,
                readout: 1e-2,
                preparation: 5e-4,
            },
            coherence_times: CoherenceTimes::new(200.0, 100.0), // 200us T1, 100us T2
            native_gates: vec![NativeGate::Rx, NativeGate::Rz, NativeGate::CX],
        }
    }

    /// Create a Google Sycamore-like superconducting profile.
    ///
    /// Uses 2D grid with sqrt-iSWAP native gate.
    pub fn google(rows: usize, cols: usize) -> Self {
        Self {
            n_qubits: rows * cols,
            topology: QubitTopology::Grid { rows, cols },
            vendor: SuperconductingVendor::Google,
            error_rates: ErrorRates {
                single_qubit: 1.5e-3,
                two_qubit: 6e-3,
                readout: 3.8e-2, // Sycamore readout
                preparation: 5e-4,
            },
            coherence_times: CoherenceTimes::new(15.0, 10.0), // Shorter for flux-tunable
            native_gates: vec![
                NativeGate::Rx,
                NativeGate::Ry,
                NativeGate::Rz,
                NativeGate::SqrtISWAP,
            ],
        }
    }

    /// Create a generic superconducting profile with custom parameters.
    pub fn generic(n_qubits: usize, topology: QubitTopology) -> Self {
        Self {
            n_qubits,
            topology,
            vendor: SuperconductingVendor::Generic,
            error_rates: ErrorRates::default(),
            coherence_times: CoherenceTimes::new(100.0, 50.0),
            native_gates: vec![
                NativeGate::Rx,
                NativeGate::Ry,
                NativeGate::Rz,
                NativeGate::CZ,
            ],
        }
    }

    /// Builder: set error rates.
    pub fn with_error_rates(mut self, rates: ErrorRates) -> Self {
        self.error_rates = rates;
        self
    }

    /// Builder: set coherence times.
    pub fn with_coherence_times(mut self, times: CoherenceTimes) -> Self {
        self.coherence_times = times;
        self
    }

    /// Get the hardware vendor.
    pub fn vendor(&self) -> SuperconductingVendor {
        self.vendor
    }
}

impl HardwareProfile for SuperconductingProfile {
    fn name(&self) -> &str {
        match self.vendor {
            SuperconductingVendor::Ibm => "IBM-Superconducting",
            SuperconductingVendor::Google => "Google-Sycamore",
            SuperconductingVendor::Rigetti => "Rigetti",
            SuperconductingVendor::Generic => "Superconducting",
        }
    }

    fn qubit_count(&self) -> usize {
        self.n_qubits
    }

    fn topology(&self) -> &QubitTopology {
        &self.topology
    }

    fn native_gates(&self) -> &[NativeGate] {
        &self.native_gates
    }

    fn gate_timing(&self, gate: &NativeGate) -> GateTiming {
        match gate {
            // Single-qubit gates (microwave pulses)
            NativeGate::Rx | NativeGate::Ry => GateTiming::new(0.020), // 20 ns
            NativeGate::Rz => GateTiming::new(0.0),                    // Virtual Z, free
            NativeGate::H => GateTiming::new(0.020),
            // Two-qubit gates
            NativeGate::CX => GateTiming::with_setup(0.300, 0.050), // ~300ns for cross-resonance
            NativeGate::CZ => GateTiming::with_setup(0.040, 0.010), // ~40ns for tunable coupler
            NativeGate::ISWAP => GateTiming::with_setup(0.030, 0.010),
            NativeGate::SqrtISWAP => GateTiming::with_setup(0.020, 0.010),
            // Synthesized gates
            _ => GateTiming::with_setup(0.100, 0.020),
        }
    }

    fn error_rates(&self) -> &ErrorRates {
        &self.error_rates
    }

    fn coherence_times(&self) -> &CoherenceTimes {
        &self.coherence_times
    }
}

/// Trapped ion hardware profile.
///
/// Models platforms like IonQ and Quantinuum (Honeywell).
/// Uses laser-driven gates on trapped atomic ions.
///
/// Key characteristics:
/// - All-to-all connectivity via ion shuttling
/// - Very high gate fidelity
/// - Native Molmer-Sorensen gate
/// - Long coherence times but slower gates
///
/// # Literature
/// - Bruzewicz et al. (2019): Trapped-ion quantum computing: Progress and challenges
#[derive(Debug, Clone)]
pub struct TrappedIonProfile {
    n_qubits: usize,
    topology: QubitTopology,
    error_rates: ErrorRates,
    coherence_times: CoherenceTimes,
    native_gates: Vec<NativeGate>,
}

impl TrappedIonProfile {
    /// Create a default trapped ion profile with IonQ-like parameters.
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            topology: QubitTopology::AllToAll, // Ion shuttling enables all-to-all
            error_rates: ErrorRates {
                single_qubit: 3e-5, // Very high fidelity
                two_qubit: 5e-3,
                readout: 1e-3,
                preparation: 1e-4,
            },
            coherence_times: CoherenceTimes::new(1_000_000.0, 500_000.0), // ~1s T1, 0.5s T2
            native_gates: vec![
                NativeGate::Rx,
                NativeGate::Ry,
                NativeGate::Rz,
                NativeGate::MS, // Molmer-Sorensen
            ],
        }
    }

    /// Builder: set error rates.
    pub fn with_error_rates(mut self, rates: ErrorRates) -> Self {
        self.error_rates = rates;
        self
    }

    /// Builder: set coherence times.
    pub fn with_coherence_times(mut self, times: CoherenceTimes) -> Self {
        self.coherence_times = times;
        self
    }
}

impl HardwareProfile for TrappedIonProfile {
    fn name(&self) -> &str {
        "TrappedIon"
    }

    fn qubit_count(&self) -> usize {
        self.n_qubits
    }

    fn topology(&self) -> &QubitTopology {
        &self.topology
    }

    fn native_gates(&self) -> &[NativeGate] {
        &self.native_gates
    }

    fn gate_timing(&self, gate: &NativeGate) -> GateTiming {
        match gate {
            // Single-qubit gates (laser pulses)
            NativeGate::Rx | NativeGate::Ry => GateTiming::new(10.0), // 10 us
            NativeGate::Rz => GateTiming::new(0.1),                   // AC Stark shift, fast
            // Molmer-Sorensen gate
            NativeGate::MS => GateTiming::with_setup(200.0, 50.0), // ~200 us
            // Synthesized from MS
            NativeGate::CX | NativeGate::CZ => GateTiming::with_setup(250.0, 50.0),
            _ => GateTiming::with_setup(100.0, 20.0),
        }
    }

    fn error_rates(&self) -> &ErrorRates {
        &self.error_rates
    }

    fn coherence_times(&self) -> &CoherenceTimes {
        &self.coherence_times
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_topology_edges() {
        let topo = QubitTopology::Linear;
        let edges = topo.edges(5);
        assert_eq!(edges, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn test_grid_topology_edges() {
        let topo = QubitTopology::Grid { rows: 2, cols: 3 };
        let edges = topo.edges(6);
        // 0-1-2
        // | | |
        // 3-4-5
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(1, 2)));
        assert!(edges.contains(&(0, 3)));
        assert!(edges.contains(&(1, 4)));
        assert!(edges.contains(&(2, 5)));
        assert!(edges.contains(&(3, 4)));
        assert!(edges.contains(&(4, 5)));
        assert_eq!(edges.len(), 7);
    }

    #[test]
    fn test_all_to_all_topology() {
        let topo = QubitTopology::AllToAll;
        let edges = topo.edges(4);
        assert_eq!(edges.len(), 6); // C(4,2) = 6
        assert!(topo.are_connected(4, 0, 3));
        assert!(topo.are_connected(4, 1, 2));
    }

    #[test]
    fn test_ideal_hardware() {
        let hw = IdealHardware::new(10);
        assert_eq!(hw.qubit_count(), 10);
        assert_eq!(hw.estimate_fidelity(100, 50), 1.0);
        assert_eq!(hw.max_practical_depth(), usize::MAX);
    }

    #[test]
    fn test_coherence_decoherence_prob() {
        let ct = CoherenceTimes::new(100.0, 50.0);
        let prob = ct.decoherence_prob(10.0);
        assert!(prob > 0.0 && prob < 1.0);
        // At t=0, prob should be ~0
        assert!(ct.decoherence_prob(0.001) < 0.001);
    }

    #[test]
    fn test_neutral_atom_profile() {
        let hw = NeutralAtomProfile::new(100);
        assert_eq!(hw.qubit_count(), 100);
        assert_eq!(hw.name(), "NeutralAtom");
        assert!(hw.supports_gate(&NativeGate::RydbergCZ));
        assert!(!hw.supports_gate(&NativeGate::ISWAP));

        // Coherence times should be long (milliseconds)
        let ct = hw.coherence_times();
        assert!(ct.t1_us > 1000.0); // > 1ms
        assert!(ct.t2_us > 100.0); // > 0.1ms
    }

    #[test]
    fn test_neutral_atom_grid() {
        let hw = NeutralAtomProfile::with_grid(5, 5, 4.0);
        assert_eq!(hw.qubit_count(), 25);

        // Grid topology
        if let QubitTopology::Grid { rows, cols } = hw.topology() {
            assert_eq!(*rows, 5);
            assert_eq!(*cols, 5);
        } else {
            panic!("Expected Grid topology");
        }

        // Atoms within blockade radius can interact
        assert!(hw.can_interact(0, 1)); // Adjacent
        assert!(hw.can_interact(0, 5)); // One row down
    }

    #[test]
    fn test_neutral_atom_blockade_interaction() {
        let hw = NeutralAtomProfile::with_grid(10, 10, 4.0).with_blockade_radius(8.0); // 8um radius, 4um spacing

        // Nearest neighbors: 4um apart -> can interact
        assert!(hw.can_interact(0, 1));

        // Diagonal neighbors: sqrt(32) ~ 5.66um apart -> can interact
        assert!(hw.can_interact(0, 11));

        // Two steps away: 8um apart -> just at boundary
        // Note: distance = 2 * spacing = 8.0, blockade = 8.0
        // This is at the boundary, so can_interact returns false (< not <=)
        assert!(!hw.can_interact(0, 2));
    }

    #[test]
    fn test_superconducting_ibm() {
        let hw = SuperconductingProfile::ibm(27);
        assert!(hw.qubit_count() >= 27);
        assert_eq!(hw.name(), "IBM-Superconducting");
        assert_eq!(hw.vendor(), SuperconductingVendor::Ibm);
        assert!(hw.supports_gate(&NativeGate::CX));
        assert!(!hw.supports_gate(&NativeGate::MS));

        // Fast gates (nanoseconds converted to microseconds)
        let timing = hw.gate_timing(&NativeGate::Rx);
        assert!(timing.duration_us < 1.0); // < 1us

        // Grid topology
        matches!(hw.topology(), QubitTopology::Grid { .. });
    }

    #[test]
    fn test_superconducting_google() {
        let hw = SuperconductingProfile::google(6, 6);
        assert_eq!(hw.qubit_count(), 36);
        assert_eq!(hw.name(), "Google-Sycamore");
        assert!(hw.supports_gate(&NativeGate::SqrtISWAP));

        // sqrt-iSWAP timing
        let timing = hw.gate_timing(&NativeGate::SqrtISWAP);
        assert!(timing.total_us() < 0.1); // Very fast
    }

    #[test]
    fn test_trapped_ion_profile() {
        let hw = TrappedIonProfile::new(11);
        assert_eq!(hw.qubit_count(), 11);
        assert_eq!(hw.name(), "TrappedIon");
        assert!(hw.supports_gate(&NativeGate::MS));

        // All-to-all connectivity
        assert_eq!(*hw.topology(), QubitTopology::AllToAll);

        // Very long coherence times (seconds!)
        let ct = hw.coherence_times();
        assert!(ct.t1_us > 100_000.0); // > 0.1s
        assert!(ct.t2_us > 100_000.0);

        // But slower gates
        let timing = hw.gate_timing(&NativeGate::MS);
        assert!(timing.duration_us > 100.0); // > 100us
    }

    #[test]
    fn test_fidelity_comparison() {
        // Compare fidelity across platforms for same circuit
        let depth = 20;
        let two_qubit_gates = 50;

        let ideal = IdealHardware::new(10);
        let atoms = NeutralAtomProfile::new(10);
        let sc = SuperconductingProfile::ibm(10);
        let ions = TrappedIonProfile::new(10);

        let f_ideal = ideal.estimate_fidelity(depth, two_qubit_gates);
        let f_atoms = atoms.estimate_fidelity(depth, two_qubit_gates);
        let f_sc = sc.estimate_fidelity(depth, two_qubit_gates);
        let f_ions = ions.estimate_fidelity(depth, two_qubit_gates);

        // Ideal should be perfect
        assert_eq!(f_ideal, 1.0);

        // All real hardware should have fidelity < 1
        assert!(f_atoms < 1.0);
        assert!(f_sc < 1.0);
        assert!(f_ions < 1.0);

        // Trapped ions typically have best single-qubit fidelity
        assert!(f_ions > f_sc);
    }

    #[test]
    fn test_max_practical_depth() {
        let atoms = NeutralAtomProfile::new(10);
        let sc = SuperconductingProfile::ibm(10);
        let ions = TrappedIonProfile::new(10);

        // Neutral atoms: long T2 / moderate gate time -> medium depth
        let d_atoms = atoms.max_practical_depth();
        assert!(d_atoms > 100);

        // Superconducting: short T2 / fast gates -> similar depth
        let d_sc = sc.max_practical_depth();
        assert!(d_sc > 100);

        // Trapped ions: very long T2 / slow gates -> can be high
        let d_ions = ions.max_practical_depth();
        assert!(d_ions > d_atoms);
    }

    #[test]
    fn test_hardware_profile_trait_object() {
        // Verify trait objects work (Send + Sync bounds)
        let profiles: Vec<Box<dyn HardwareProfile>> = vec![
            Box::new(IdealHardware::new(10)),
            Box::new(NeutralAtomProfile::new(10)),
            Box::new(SuperconductingProfile::ibm(10)),
            Box::new(TrappedIonProfile::new(10)),
        ];

        for hw in &profiles {
            assert!(hw.qubit_count() > 0);
            assert!(!hw.name().is_empty());
        }
    }
}
