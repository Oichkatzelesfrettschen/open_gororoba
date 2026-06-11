//! WMM 2025 (World Magnetic Model) provider.
//!
//! WMM is the standard geomagnetic model for navigation, produced by
//! NOAA/NCEI and the British Geological Survey. Updated every 5 years.
//! Degree/order up to 12, valid 2025-2030.
//!
//! Source: NOAA/NCEI, <https://www.ncei.noaa.gov/products/world-magnetic-model>
//! Reference: Chulliat et al. (2024), NOAA Technical Report

/// WMM 2025 coefficient file URLs.
///
/// The ZIP contains WMM.COF (the coefficient file) and supporting documents.
const WMM_URLS: &[&str] = &[
    "https://www.ncei.noaa.gov/sites/default/files/2024-12/WMM2025COF.zip",
    "https://www.ngdc.noaa.gov/geomag/WMM/data/WMM2025/WMM2025COF.zip",
];

simple_provider! {
    /// WMM 2025 geomagnetic model provider.
    pub struct Wmm2025Provider;
    name = "WMM 2025 Coefficients";
    output = "wmm2025.zip";
    urls = WMM_URLS;
}
