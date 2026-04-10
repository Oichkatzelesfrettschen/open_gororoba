//! Catalog parsers for cosmological and astrophysical datasets.
//!
//! Each module provides:
//! - Download URLs and metadata
//! - A DatasetProvider implementation for fetching
//! - Typed record structs with CSV parsing

pub mod ace_mag;
#[cfg(feature = "fetch")]
pub mod ace_mag_fetch;
pub mod aflow;
#[cfg(feature = "fetch")]
pub mod aflow_fetch;
pub mod atnf;
pub mod bepicolombo;
#[cfg(feature = "fetch")]
pub mod bepicolombo_fetch;
pub mod bl_filterbank;
pub mod cassini;
pub mod chime;
pub mod cluster;
#[cfg(feature = "fetch")]
pub mod cluster_fetch;
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
#[cfg(feature = "fetch")]
pub mod hst_fetch;
pub mod ibex;
pub mod imap;
pub mod imp8;
pub mod intermagnet;
pub mod jarvis;
#[cfg(feature = "fetch")]
pub mod jarvis_fetch;
pub mod juno;
#[cfg(feature = "fetch")]
pub mod juno_fetch;
pub mod jwst;
#[cfg(feature = "fetch")]
pub mod jwst_fetch;
pub mod landsat;
#[cfg(feature = "fits")]
pub mod lotss;
pub mod manga;
pub(crate) mod mast_public_metadata;
pub mod maven_mag;
#[cfg(feature = "fetch")]
pub mod maven_mag_fetch;
pub mod mcgill;
pub mod messenger;
#[cfg(feature = "fetch")]
pub mod messenger_fetch;
pub mod mms;
pub mod nanograv;
pub mod new_horizons;
#[cfg(feature = "fetch")]
pub mod new_horizons_fetch;
pub mod omni;
pub mod pantheon;
pub mod pdg;
pub mod pioneer;
pub mod planck;
pub mod psp;
#[cfg(feature = "fetch")]
pub mod psp_fetch;
pub mod psp_fields;
pub mod psp_spc;
pub mod psp_spi;
pub mod psp_sqtn;
pub mod rosetta;
pub mod sdo_hmi;
pub mod sdss;
pub mod soho_celias;
pub mod solar_orbiter;
#[cfg(feature = "fetch")]
pub mod solar_orbiter_fetch;
pub mod solar_orbiter_mag;
pub mod solar_orbiter_rpw;
pub mod solar_orbiter_rpw_density;
pub mod solar_orbiter_rpw_hfr;
pub mod solar_orbiter_rpw_tnr;
pub mod solar_orbiter_swa;
#[cfg(feature = "fetch")]
pub mod solar_orbiter_swa_fetch;
pub mod solar_wind;
#[cfg(feature = "fetch")]
pub mod solar_wind_fetch;
pub mod sorce;
pub mod sparc;
pub mod spdf_fleet;
pub mod spdf_merged;
pub mod stereo_plastic;
#[cfg(feature = "fetch")]
pub mod stereo_plastic_fetch;
pub mod swarm_mag;
pub mod themis;
#[cfg(feature = "fetch")]
pub mod themis_fetch;
pub mod things;
#[cfg(feature = "fetch")]
pub mod things_fetch;
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
