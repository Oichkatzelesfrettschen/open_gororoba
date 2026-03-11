//! IGRF-13 (International Geomagnetic Reference Field) provider.
//!
//! IGRF-13 provides the main geomagnetic field as spherical harmonic
//! coefficients from 1900 to 2025, with predictive secular variation
//! to 2025. Degree/order up to 13.
//!
//! Source: NOAA/NCEI, https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
//! Reference: Alken et al. (2021), Earth, Planets and Space 73, 49

/// IGRF-13 coefficient file URLs.
const IGRF_URLS: &[&str] = &[
    "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf13coeffs.txt",
    "https://www.ngdc.noaa.gov/IAGA/vmod/igrf13coeffs.txt",
];

simple_provider! {
    /// IGRF-13 geomagnetic field coefficient provider.
    pub struct Igrf13Provider;
    name = "IGRF-13 Coefficients";
    output = "igrf13coeffs.txt";
    urls = IGRF_URLS;
}
