//! Catalog parsers for cosmological and astrophysical datasets.
//!
//! Each module provides:
//! - Download URLs and metadata
//! - A DatasetProvider implementation for fetching
//! - Typed record structs with CSV parsing

pub mod atnf;
pub mod chime;
pub mod desi_bao;
pub mod eht;
pub mod fermi_gbm;
pub mod gaia;
pub mod gwtc;
pub mod hipparcos;
pub mod jarvis;
pub mod landsat;
pub mod mcgill;
pub mod nanograv;
pub mod pantheon;
pub mod planck;
pub mod sdss;
pub mod sorce;
pub mod tsi;
pub mod union3;
