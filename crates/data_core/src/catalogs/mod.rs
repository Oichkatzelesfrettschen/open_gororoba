//! Catalog parsers for cosmological and astrophysical datasets.
//!
//! Each module provides:
//! - Download URLs and metadata
//! - A DatasetProvider implementation for fetching
//! - Typed record structs with CSV parsing

pub mod aflow;
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

pub use eht::{list_tgz_members, tgz_member_count, validate_eht_archive};
pub use hipparcos::{
    parse_hip_number, validate_hipparcos_format, HIPPARCOS_EXPECTED_ROWS, HIPPARCOS_LINE_WIDTH,
    HIPPARCOS_PIPE_COUNT,
};
pub use landsat::{count_stac_assets, extract_cloud_cover, validate_stac_schema};
pub use tsi::{compare_tsis_sorce, TsiOverlapResult};
