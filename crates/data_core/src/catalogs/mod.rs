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

pub use eht::{list_tgz_members, validate_eht_archive, tgz_member_count};
pub use hipparcos::{validate_hipparcos_format, parse_hip_number, HIPPARCOS_LINE_WIDTH, HIPPARCOS_PIPE_COUNT, HIPPARCOS_EXPECTED_ROWS};
pub use landsat::{validate_stac_schema, extract_cloud_cover, count_stac_assets};
pub use tsi::{compare_tsis_sorce, TsiOverlapResult};
