//! Catalog parsers for cosmological and astrophysical datasets.
//!
//! Each module provides:
//! - Download URLs and metadata
//! - A DatasetProvider implementation for fetching
//! - Typed record structs with CSV parsing

pub mod ace_mag;
pub mod aflow;
pub mod atnf;
pub mod bepicolombo;
pub mod bl_filterbank;
pub mod cassini;
pub mod chime;
pub mod des_y6;
pub mod desi_bao;
pub mod eht;
pub mod fermi_gbm;
pub mod gaia;
pub mod gaia_mw_rotation;
pub mod gwtc;
pub mod helios;
pub mod hi_cube;
pub mod hic_raa;
pub mod hipparcos;
pub mod hst;
pub mod ibex;
pub mod imap;
pub mod imp8;
pub mod jarvis;
pub mod juno;
pub mod jwst;
pub mod landsat;
#[cfg(feature = "fits")]
pub mod lotss;
pub mod manga;
pub mod mcgill;
pub mod nanograv;
pub mod new_horizons;
pub mod omni;
pub mod pantheon;
pub mod pdg;
pub mod pioneer;
pub mod planck;
pub mod psp;
pub mod psp_fields;
pub mod psp_spc;
pub mod psp_spi;
pub mod psp_sqtn;
pub mod sdss;
pub mod soho_celias;
pub mod solar_orbiter;
pub mod solar_orbiter_mag;
pub mod solar_orbiter_rpw;
pub mod solar_orbiter_rpw_density;
pub mod solar_orbiter_rpw_hfr;
pub mod solar_orbiter_rpw_tnr;
pub mod solar_orbiter_swa;
pub mod solar_wind;
pub mod sorce;
pub mod sparc;
pub mod spdf_fleet;
pub mod spdf_merged;
pub mod stereo_plastic;
pub mod things;
pub mod tsi;
pub mod ulysses;
pub mod union3;
pub mod voyager;
pub mod voyager_crs;
pub mod voyager_crs_flux;
pub mod voyager_pws;
pub mod wind_swe;
pub mod wow;

pub use eht::{list_tgz_members, tgz_member_count, validate_eht_archive};
pub use hipparcos::{
    HIPPARCOS_EXPECTED_ROWS, HIPPARCOS_LINE_WIDTH, HIPPARCOS_PIPE_COUNT, parse_hip_number,
    validate_hipparcos_format,
};
pub use landsat::{count_stac_assets, extract_cloud_cover, validate_stac_schema};
pub use tsi::{TsiOverlapResult, compare_tsis_sorce};
