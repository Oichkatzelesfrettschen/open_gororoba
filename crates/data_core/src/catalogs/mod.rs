//! Catalog parsers for cosmological and astrophysical datasets.
//!
//! Each module provides:
//! - Download URLs and metadata
//! - A DatasetProvider implementation for fetching
//! - Typed record structs with CSV parsing

pub mod chime;
pub mod gwtc;
pub mod atnf;
pub mod mcgill;
pub mod fermi_gbm;
pub mod planck;
pub mod desi_bao;
pub mod pantheon;
pub mod nanograv;
pub mod sdss;
pub mod gaia;
pub mod tsi;
