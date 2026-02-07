//! Observational spectral band definitions and photometric utilities.
//!
//! Defines standard astronomical observational bands (EHT radio, ALMA sub-mm,
//! Johnson V optical, Chandra X-ray) with center frequencies, bandwidths,
//! and filter response functions.
//!
//! Also provides unit conversions between frequency, wavelength, and photon
//! energy, plus the Vega magnitude system for flux-to-magnitude conversion.
//!
//! These utilities complement the disk spectrum functions in [`super::novikov_thorne`],
//! enabling synthetic multi-wavelength observations of accretion disk models.
//!
//! References:
//!   - Event Horizon Telescope Collaboration (2019): ApJL 875, L1
//!   - Johnson & Morgan (1953): ApJ 117, 313 (UBVRI system)
//!   - Chandra X-ray Center: 0.5--10 keV bandpass

// ============================================================================
// Physical constants for spectral calculations (SI)
// ============================================================================

/// Speed of light [m/s] (SI, for wavelength conversions).
const C_SI: f64 = 2.997_924_58e8;

/// Planck constant [J s] (SI).
const H_SI: f64 = 6.626_070_15e-34;

/// eV to Joules conversion.
const EV_TO_JOULES: f64 = 1.602_176_634e-19;

/// keV to Joules conversion.
const KEV_TO_JOULES: f64 = 1e3 * EV_TO_JOULES;

// ============================================================================
// Spectral band definitions
// ============================================================================

/// An observational spectral band with center frequency and bandwidth.
#[derive(Debug, Clone, Copy)]
pub struct SpectralBand {
    /// Band name.
    pub name: &'static str,
    /// Center frequency [Hz].
    pub center_hz: f64,
    /// Bandwidth (FWHM) [Hz].
    pub bandwidth_hz: f64,
}

/// Event Horizon Telescope 230 GHz band (1.3 mm).
pub const BAND_EHT: SpectralBand = SpectralBand {
    name: "Radio (EHT 230 GHz)",
    center_hz: 2.3e11,
    bandwidth_hz: 4e9,
};

/// ALMA 100 GHz sub-millimeter band (3 mm).
pub const BAND_ALMA: SpectralBand = SpectralBand {
    name: "Sub-mm (ALMA 100 GHz)",
    center_hz: 1e11,
    bandwidth_hz: 8e9,
};

/// Johnson V-band optical (540 nm, ~5.55e14 Hz).
pub const BAND_OPTICAL_V: SpectralBand = SpectralBand {
    name: "Optical (V-band 540 nm)",
    center_hz: C_SI / 540.0e-9,
    // Bandwidth ~80 nm FWHM in frequency space:
    // delta_nu = c * delta_lambda / lambda^2 ~ 2.7e13 Hz
    bandwidth_hz: C_SI * 80.0e-9 / (540.0e-9 * 540.0e-9),
};

/// Chandra X-ray band (0.5--10 keV, center ~3 keV ~ 7.25e17 Hz).
pub const BAND_XRAY: SpectralBand = SpectralBand {
    name: "X-ray (Chandra 0.5-10 keV)",
    center_hz: 3.0 * KEV_TO_JOULES / H_SI,
    bandwidth_hz: (10.0 - 0.5) * KEV_TO_JOULES / H_SI,
};

// ============================================================================
// Unit conversions
// ============================================================================

/// Convert frequency [Hz] to wavelength [m].
///
/// lambda = c / nu
pub fn frequency_to_wavelength(frequency: f64) -> f64 {
    if frequency < 1e6 {
        return f64::INFINITY;
    }
    C_SI / frequency
}

/// Convert wavelength [m] to frequency [Hz].
///
/// nu = c / lambda
pub fn wavelength_to_frequency(wavelength: f64) -> f64 {
    if wavelength <= 0.0 {
        return f64::INFINITY;
    }
    C_SI / wavelength
}

/// Convert photon energy [keV] to frequency [Hz].
///
/// nu = E / h
pub fn energy_kev_to_frequency(energy_kev: f64) -> f64 {
    energy_kev * KEV_TO_JOULES / H_SI
}

/// Convert frequency [Hz] to photon energy [keV].
///
/// E = h * nu
pub fn frequency_to_energy_kev(frequency: f64) -> f64 {
    H_SI * frequency / KEV_TO_JOULES
}

// ============================================================================
// Filter response functions
// ============================================================================

/// Gaussian filter response centered at `center_hz` with given FWHM bandwidth.
///
/// R(nu) = exp(-0.5 * ((nu - center) / sigma)^2)
///
/// where sigma = FWHM / (2 * sqrt(2 * ln 2)).
pub fn gaussian_filter(frequency: f64, center_hz: f64, bandwidth_hz: f64) -> f64 {
    if bandwidth_hz < 1e6 {
        return 0.0;
    }
    let sigma = bandwidth_hz / (2.0 * (2.0_f64.ln() * 2.0).sqrt());
    let x = (frequency - center_hz) / sigma;
    (-0.5 * x * x).exp()
}

/// Rectangular (top-hat) filter response.
///
/// Returns 1.0 if frequency is within [center - bw/2, center + bw/2], else 0.0.
pub fn rectangular_filter(frequency: f64, center_hz: f64, bandwidth_hz: f64) -> f64 {
    let half_bw = bandwidth_hz / 2.0;
    if frequency >= center_hz - half_bw && frequency <= center_hz + half_bw {
        1.0
    } else {
        0.0
    }
}

/// Evaluate the Gaussian filter response for a predefined spectral band.
pub fn band_response(band: &SpectralBand, frequency: f64) -> f64 {
    gaussian_filter(frequency, band.center_hz, band.bandwidth_hz)
}

// ============================================================================
// Magnitude system
// ============================================================================

/// Convert flux to apparent magnitude (Vega system).
///
/// m = -2.5 * log10(F / F_ref)
///
/// Returns 99.0 for non-positive flux (conventional faint limit).
pub fn flux_to_magnitude(flux: f64, reference_flux: f64) -> f64 {
    if flux <= 0.0 || reference_flux <= 0.0 {
        return 99.0;
    }
    -2.5 * (flux / reference_flux).log10()
}

/// Convert apparent magnitude to flux (Vega system).
///
/// F = F_ref * 10^(-m / 2.5)
pub fn magnitude_to_flux(magnitude: f64, reference_flux: f64) -> f64 {
    reference_flux * 10.0_f64.powf(-magnitude / 2.5)
}

/// V-band zero-point flux (Vega) [erg s^-1 cm^-2 Hz^-1].
///
/// F_0(V) = 3631 Jy = 3.631e-20 erg s^-1 cm^-2 Hz^-1 (AB system).
/// For Vega system: ~3.64e-20 (close enough for our purposes).
pub const VEGA_V_FLUX: f64 = 3.64e-20;

// ============================================================================
// Multi-band sampling
// ============================================================================

/// Integrated flux through a spectral band given discrete frequency/flux samples.
///
/// Uses trapezoidal integration with the band's Gaussian filter response.
///
/// Arguments:
///   frequencies: frequency samples [Hz], must be sorted ascending
///   fluxes: corresponding flux densities [erg s^-1 cm^-2 Hz^-1]
///   band: the spectral band to integrate through
///
/// Returns band-integrated flux [erg s^-1 cm^-2].
pub fn integrate_through_band(
    frequencies: &[f64],
    fluxes: &[f64],
    band: &SpectralBand,
) -> f64 {
    if frequencies.len() < 2 || fluxes.len() < 2 {
        return 0.0;
    }
    let n = frequencies.len().min(fluxes.len());
    let mut sum = 0.0;
    for i in 0..n - 1 {
        let freq_mid = 0.5 * (frequencies[i] + frequencies[i + 1]);
        let flux_mid = 0.5 * (fluxes[i] + fluxes[i + 1]);
        let dfreq = frequencies[i + 1] - frequencies[i];
        let response = band_response(band, freq_mid);
        sum += response * flux_mid * dfreq;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Unit conversions --

    #[test]
    fn test_frequency_wavelength_roundtrip() {
        let nu = 5e14; // Visible light
        let lambda = frequency_to_wavelength(nu);
        let nu_back = wavelength_to_frequency(lambda);
        assert!(
            (nu_back - nu).abs() / nu < 1e-12,
            "roundtrip: {nu} -> {lambda} -> {nu_back}"
        );
    }

    #[test]
    fn test_visible_light_wavelength() {
        let nu = C_SI / 500e-9; // 500 nm -> frequency
        let lambda = frequency_to_wavelength(nu);
        assert!(
            (lambda - 500e-9).abs() < 1e-12,
            "500nm light: lambda = {lambda}"
        );
    }

    #[test]
    fn test_energy_frequency_roundtrip() {
        let e_kev = 6.0; // Iron K-alpha
        let nu = energy_kev_to_frequency(e_kev);
        let e_back = frequency_to_energy_kev(nu);
        assert!(
            (e_back - e_kev).abs() < 1e-10,
            "roundtrip: {e_kev} keV -> {nu} Hz -> {e_back} keV"
        );
    }

    #[test]
    fn test_1kev_frequency() {
        // 1 keV ~ 2.418e17 Hz
        let nu = energy_kev_to_frequency(1.0);
        assert!(
            (nu - 2.418e17).abs() / 2.418e17 < 0.01,
            "1 keV = {nu} Hz"
        );
    }

    #[test]
    fn test_frequency_to_wavelength_low_input() {
        let lambda = frequency_to_wavelength(0.0);
        assert!(lambda.is_infinite());
    }

    // -- Band definitions --

    #[test]
    fn test_eht_band_center() {
        assert!((BAND_EHT.center_hz - 2.3e11).abs() < 1e8);
    }

    #[test]
    fn test_optical_v_band_center() {
        // 540 nm -> ~5.55e14 Hz
        let expected = C_SI / 540.0e-9;
        assert!(
            (BAND_OPTICAL_V.center_hz - expected).abs() / expected < 1e-6,
            "V-band center = {} Hz",
            BAND_OPTICAL_V.center_hz
        );
    }

    #[test]
    fn test_xray_band_center() {
        // 3 keV ~ 7.25e17 Hz
        let expected = energy_kev_to_frequency(3.0);
        assert!(
            (BAND_XRAY.center_hz - expected).abs() / expected < 1e-6,
            "X-ray center = {} Hz",
            BAND_XRAY.center_hz
        );
    }

    // -- Filter functions --

    #[test]
    fn test_gaussian_filter_peak() {
        let r = gaussian_filter(2.3e11, 2.3e11, 4e9);
        assert!((r - 1.0).abs() < 1e-12, "peak response = {r}");
    }

    #[test]
    fn test_gaussian_filter_at_half_max() {
        // FWHM means response = 0.5 at center +/- bandwidth/2
        let r = gaussian_filter(2.3e11 + 2e9, 2.3e11, 4e9);
        assert!((r - 0.5).abs() < 0.01, "half-max response = {r}");
    }

    #[test]
    fn test_gaussian_filter_far_away() {
        let r = gaussian_filter(1e14, 2.3e11, 4e9);
        assert!(r < 1e-100, "far from band: {r}");
    }

    #[test]
    fn test_rectangular_filter_inside() {
        let r = rectangular_filter(2.3e11, 2.3e11, 4e9);
        assert!(r == 1.0);
    }

    #[test]
    fn test_rectangular_filter_outside() {
        let r = rectangular_filter(2.3e11 + 3e9, 2.3e11, 4e9);
        assert!(r == 0.0);
    }

    #[test]
    fn test_rectangular_filter_edge() {
        let center = 2.3e11;
        let bw = 4e9;
        let r = rectangular_filter(center + bw / 2.0, center, bw);
        assert!(r == 1.0, "edge should be included");
    }

    // -- Magnitude system --

    #[test]
    fn test_flux_to_magnitude_zero_point() {
        // Zero-magnitude star has flux = reference_flux
        let m = flux_to_magnitude(VEGA_V_FLUX, VEGA_V_FLUX);
        assert!((m - 0.0).abs() < 1e-12, "m(F_0) = {m}");
    }

    #[test]
    fn test_magnitude_roundtrip() {
        let m = 5.0;
        let f = magnitude_to_flux(m, VEGA_V_FLUX);
        let m_back = flux_to_magnitude(f, VEGA_V_FLUX);
        assert!(
            (m_back - m).abs() < 1e-10,
            "roundtrip: m={m} -> F={f} -> m={m_back}"
        );
    }

    #[test]
    fn test_five_magnitudes_is_100x() {
        // 5 magnitudes difference = factor of 100 in flux
        let f1 = magnitude_to_flux(0.0, VEGA_V_FLUX);
        let f2 = magnitude_to_flux(5.0, VEGA_V_FLUX);
        assert!(
            (f1 / f2 - 100.0).abs() < 1e-6,
            "5 mag ratio = {}",
            f1 / f2
        );
    }

    #[test]
    fn test_negative_flux_returns_99() {
        let m = flux_to_magnitude(-1.0, VEGA_V_FLUX);
        assert!(m == 99.0);
    }

    // -- Multi-band integration --

    #[test]
    fn test_integrate_flat_spectrum() {
        // Flat spectrum at 1.0 through a band should give ~ bandwidth
        let n = 1000;
        let freqs: Vec<f64> = (0..n).map(|i| 2.0e11 + i as f64 * 2e8).collect();
        let fluxes: Vec<f64> = vec![1.0; n];
        let integrated = integrate_through_band(&freqs, &fluxes, &BAND_EHT);
        // For Gaussian filter: integral ~ flux * sqrt(2pi) * sigma
        // sigma = FWHM / 2.355 = 4e9 / 2.355 ~ 1.7e9
        // integral ~ 1.0 * sqrt(2pi) * 1.7e9 ~ 4.25e9
        assert!(
            integrated > 1e9 && integrated < 1e10,
            "flat spectrum through EHT: {integrated}"
        );
    }

    #[test]
    fn test_integrate_empty_spectrum() {
        let integrated = integrate_through_band(&[], &[], &BAND_EHT);
        assert!(integrated == 0.0);
    }

    #[test]
    fn test_integrate_off_band_returns_zero() {
        // Spectrum at optical frequencies, integrate through radio band
        let freqs: Vec<f64> = (0..100).map(|i| 4e14 + i as f64 * 1e12).collect();
        let fluxes: Vec<f64> = vec![1.0; 100];
        let integrated = integrate_through_band(&freqs, &fluxes, &BAND_EHT);
        assert!(integrated < 1e-100, "off-band: {integrated}");
    }
}
