//! Metropolitan Hybrid Quantum Network Emulator.
//! Integrates Diamond NV centers, QFC (Frequency Conversion), and ORCA memories.
//! Simulates Barrett-Kok entanglement generation over a telecom fiber link.

use crate::diamond_nv::DiamondNV;
use crate::quantum_frequency_conversion::QuantumFrequencyConverter;
use crate::orca_memory::OrcaMemory;

#[derive(Debug)]
pub struct HybridQuantumNode {
    pub nv_center: DiamondNV,
    pub qfc: QuantumFrequencyConverter,
    pub orca: OrcaMemory,
}

impl Default for HybridQuantumNode {
    fn default() -> Self {
        Self {
            nv_center: DiamondNV::new(0.99999, 50.0, 300.0), // Enriched 12C NV
            qfc: QuantumFrequencyConverter::nv_to_telecom(),
            orca: OrcaMemory::telecom_rubidium(),
        }
    }
}

pub struct MetropolitanNetwork {
    pub node_a: HybridQuantumNode,
    pub node_b: HybridQuantumNode,
    /// Fiber length in km
    pub fiber_length_km: f64,
    /// Telecom fiber loss in dB/km (typically 0.2 for 1550nm band)
    pub fiber_loss_db_per_km: f64,
    /// Detector efficiency (e.g. 0.60 for SNSPD)
    pub detector_efficiency: f64,
    /// Two-photon interference visibility (HOM)
    pub hom_visibility: f64,
}

impl Default for MetropolitanNetwork {
    fn default() -> Self {
        Self {
            node_a: HybridQuantumNode::default(),
            node_b: HybridQuantumNode::default(),
            fiber_length_km: 25.0, // Delft to Hague distance
            fiber_loss_db_per_km: 0.2, // standard telecom fiber
            detector_efficiency: 0.60, // SNSPD efficiency
            hom_visibility: 0.79, // From Stolk et al. 2022
        }
    }
}

impl MetropolitanNetwork {
    /// Calculate the overall transmission efficiency from the ZPL emission to the central beamsplitter
    pub fn transmission_efficiency(&self) -> f64 {
        // NV emission into ZPL is ~3%
        let zpl_fraction = 0.03;
        
        // QFC efficiency
        let qfc_eff = self.node_a.qfc.device_efficiency; // assume symmetric
        
        // Fiber loss (half distance to central station)
        let loss_db = (self.fiber_length_km / 2.0) * self.fiber_loss_db_per_km;
        let fiber_transmission = 10.0f64.powf(-loss_db / 10.0);
        
        // Output efficiency per attempt
        zpl_fraction * qfc_eff * fiber_transmission * self.detector_efficiency
    }

    /// Estimate heralded entanglement rate using a single-click (Barrett-Kok) protocol.
    /// Rate scales with sqrt(transmission) or linearly with transmission depending on protocol details.
    /// In modern single-click protocols, it scales roughly linearly per node, so P_success ~ 2 * p * (1-p) ~ 2p.
    pub fn estimated_entanglement_rate_hz(&self, repetition_rate_hz: f64) -> f64 {
        let p_click = self.transmission_efficiency();
        
        // Probability of exactly one click from two nodes
        let p_success = 2.0 * p_click * (1.0 - p_click);
        
        repetition_rate_hz * p_success
    }

    /// Estimate the resulting fidelity of the entangled state.
    pub fn estimated_fidelity(&self) -> f64 {
        // Fidelity is bounded by HOM visibility and decoherence during the attempt.
        // Simplified model combining visibility and dark counts/noise.
        let base_fidelity = (1.0 + self.hom_visibility) / 2.0;
        
        // Assume NV spin decoherence during the short communication time (~100 us for 25km) is negligible
        // T2 > 1ms.
        let comm_time = (self.fiber_length_km * 1e3) / 2.0e8; // c in fiber ~ 2e8 m/s
        let t2 = self.node_a.nv_center.t2_coherence_time();
        
        let decoherence_factor = (-comm_time / t2).exp();
        
        base_fidelity * decoherence_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transmission_efficiency() {
        let net = MetropolitanNetwork::default();
        let eff = net.transmission_efficiency();
        // 0.03 (ZPL) * 0.17 (QFC) * 0.56 (25km/2 fiber) * 0.60 (det) ~ 0.0017
        assert!(eff > 0.0001 && eff < 0.01, "eff is {}", eff);
    }

    #[test]
    fn test_entanglement_rate() {
        let net = MetropolitanNetwork::default();
        // At 100 kHz attempt rate, rate should be a few Hz or less (like 0.022 Hz in reality with lower repetition/collection)
        // Note: Real experiments have even lower collection efficiency into the first fiber before QFC.
        let rate = net.estimated_entanglement_rate_hz(100_000.0);
        assert!(rate > 0.0);
    }

    #[test]
    fn test_fidelity() {
        let net = MetropolitanNetwork::default();
        let f = net.estimated_fidelity();
        // Base fidelity ~ 0.895 based on 0.79 HOM
        assert!(f > 0.5 && f < 1.0);
    }
}
