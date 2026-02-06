//! Pantheon+ .dat whitespace-delimited format parser.
//!
//! The Pantheon+ data release uses a custom whitespace-delimited format
//! with a header line that may or may not start with '#'.
//!
//! This module re-exports the parsing logic from catalogs::pantheon
//! since the format is tightly coupled to the dataset.

pub use crate::catalogs::pantheon::parse_pantheon_dat;
