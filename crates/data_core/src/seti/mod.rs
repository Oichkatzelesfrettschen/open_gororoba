//! SETI signal processing: Doppler drift search and cadence discrimination.
//!
//! Pure Rust implementation of the core turboSETI algorithms:
//! - `doppler`: De-Doppler shift-and-add search for narrowband signals
//! - `cadence`: ABACAD cadence filter for RFI discrimination

pub mod cadence;
pub mod doppler;
