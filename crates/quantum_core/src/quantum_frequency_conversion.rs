//! Quantum Frequency Conversion using Difference Frequency Generation (DFG).
//! Translates visible photons (e.g. 637 nm NV zero-phonon line) to telecom L-band (e.g. 1588 nm).

#[derive(Debug, Clone, Copy)]
pub struct QuantumFrequencyConverter {
    pub input_wavelength_nm: f64,
    pub pump_wavelength_nm: f64,
    pub output_wavelength_nm: f64,
    /// Overall device efficiency (e.g., 0.17 for 17%)
    pub device_efficiency: f64,
    /// Noise photons generated per second
    pub noise_rate_hz: f64,
    /// Filter bandwidth in nm
    pub filter_bandwidth_nm: f64,
    /// Filter rejection in dB
    pub filter_rejection_db: f64,
}

impl Default for QuantumFrequencyConverter {
    fn default() -> Self {
        Self::nv_to_telecom()
    }
}

impl QuantumFrequencyConverter {
    /// Standard setup from Dréau et al. (2018) for NV centers.
    pub fn nv_to_telecom() -> Self {
        // 1/637 - 1/1064 = 1/1588.3
        Self {
            input_wavelength_nm: 637.0,
            pump_wavelength_nm: 1064.0,
            output_wavelength_nm: 1588.0,
            device_efficiency: 0.17,    // 17%
            noise_rate_hz: 10.0,        // Low noise after filtering
            filter_bandwidth_nm: 0.004, // 4 pm bandwidth (500 MHz)
            filter_rejection_db: 55.0,  // 55 dB rejection
        }
    }

    /// Two-stage cascaded setup from SiV to telecom C-band (Schäfer et al., 2025)
    pub fn siv_cascaded() -> Self {
        Self {
            input_wavelength_nm: 737.0,
            pump_wavelength_nm: 2812.6, // Long wavelength pump avoids noise
            output_wavelength_nm: 1550.0, // Telecom C-band
            device_efficiency: 0.356,   // 35.6% efficiency
            noise_rate_hz: 0.1,         // < 0.1 Hz noise
            filter_bandwidth_nm: 0.01,
            filter_rejection_db: 80.0,
        }
    }

    /// Verifies energy conservation: 1/lambda_out = 1/lambda_in - 1/lambda_pump
    pub fn verify_energy_conservation(&self) -> bool {
        let expected_inv = (1.0 / self.input_wavelength_nm) - (1.0 / self.pump_wavelength_nm);
        let actual_inv = 1.0 / self.output_wavelength_nm;

        // Allow for some tolerance due to rounding
        (expected_inv - actual_inv).abs() < 1e-4
    }

    /// Calculates the probability of a signal photon successfully converting and exiting,
    /// along with the expected number of noise photons in a given window.
    pub fn convert_pulse(&self, input_photons: f64, window_s: f64) -> (f64, f64) {
        let converted_signal = input_photons * self.device_efficiency;
        let noise_photons = self.noise_rate_hz * window_s;
        (converted_signal, noise_photons)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_conservation() {
        let qfc = QuantumFrequencyConverter::nv_to_telecom();
        assert!(qfc.verify_energy_conservation());
    }

    #[test]
    fn test_conversion_efficiency() {
        let qfc = QuantumFrequencyConverter::nv_to_telecom();
        let (signal, noise) = qfc.convert_pulse(1.0, 1e-9); // 1 photon, 1 ns window
        assert!((signal - 0.17).abs() < 1e-6);
        assert!(noise < 1e-6);
    }
}
