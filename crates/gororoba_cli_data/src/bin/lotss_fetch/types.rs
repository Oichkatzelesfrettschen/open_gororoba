//! Clap Cli + subcommand enum + ValueEnum types for the
//! `lotss-fetch` binary. Includes `Cli`, `Cmd` subcommand enum,
//! `ReleaseArg` + `InputFormatArg` ValueEnums and their label()
//! impl helpers.
//!
//! Fields are pub(crate). Uses #[path] indirection because this
//! binary has an explicit Cargo.toml path.

use clap::{Parser, Subcommand, ValueEnum};
use data_core::catalogs::lotss::LoTSSRelease;
use std::path::PathBuf;

// ---- CLI surface -------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "lotss-fetch",
    about = "LoTSS catalog download, DR3 tile caching, and MaNGA crossmatch"
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) cmd: Cmd,
}

#[derive(Subcommand)]
pub(crate) enum Cmd {
    /// Download the full DR1 or DR2 FITS catalog.
    Download {
        #[arg(long)]
        release: ReleaseArg,
        /// Output path override (default: data/external/radio_surveys/lotss_<release>.fits).
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Query the LoTSS DR3 VO Cone Search endpoint for a single tile.
    ConeSearch {
        #[arg(long)]
        release: ReleaseArg,
        /// Centre RA (degrees J2000).
        #[arg(long)]
        ra_center: f64,
        /// Centre Dec (degrees J2000).
        #[arg(long)]
        dec_center: f64,
        /// Search radius (degrees).
        #[arg(long, default_value = "1.0")]
        radius: f64,
        /// Output path override (default: data/external/radio_surveys/lotss_dr3_tile_<ra>_<dec>.xml).
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Verify an existing downloaded catalog file or DR3 tile cache.
    Verify {
        #[arg(long)]
        release: ReleaseArg,
        /// Path to the FITS catalog, VOTable tile, or DR3 tile directory.
        #[arg(long)]
        input: Option<PathBuf>,
    },
    /// Print a brief summary of a FITS catalog.
    Summary {
        /// Path to a LoTSS FITS file.
        #[arg(long)]
        input: PathBuf,
    },
    /// Tile the selected MaNGA sky footprint and fetch DR3 cone-search VOTables.
    MangaFootprint {
        /// Directory where DR3 tile XML files are stored.
        #[arg(long, default_value = "data/external/radio_surveys/dr3_tiles")]
        tile_dir: PathBuf,
        /// Path to the tile sweep summary TOML.
        #[arg(long, default_value = "reports/lotss_dr3_manga_footprint_summary.toml")]
        summary_out: PathBuf,
        /// Selected MaNGA galaxy CSV, typically dapall_selection.csv.
        #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
        manga_selection: PathBuf,
        /// MaNGA DRPall FITS with target coordinates.
        #[arg(long, default_value = "data/external/manga/drpall-v3_1_1.fits")]
        manga_drpall: PathBuf,
        /// Fall back to the coarse historical survey bounding-box sweep.
        #[arg(long, default_value_t = false)]
        full_bounding_box: bool,
        /// Tile half-width in degrees.
        #[arg(long, default_value = "1.0")]
        tile_radius: f64,
        /// Delay between fresh DR3 tile requests to avoid hammering the VO service.
        #[arg(long, default_value = "250")]
        request_delay_ms: u64,
    },
    /// Join the selected MaNGA sample to DRPall coordinates and report sky coverage.
    MangaPreflight {
        /// Selected MaNGA galaxy CSV, typically dapall_selection.csv.
        #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
        manga_selection: PathBuf,
        /// MaNGA DRPall FITS with target coordinates.
        #[arg(long, default_value = "data/external/manga/drpall-v3_1_1.fits")]
        manga_drpall: PathBuf,
        /// Output report path (default: reports/lotss_manga_preflight_YYYY-MM-DD.toml).
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Crossmatch the selected MaNGA sample against LoTSS DR2 or DR3.
    CrossmatchManga {
        #[arg(long)]
        release: ReleaseArg,
        /// Input format: FITS bulk catalog or DR3 tile directory.
        #[arg(long)]
        input_format: Option<InputFormatArg>,
        /// Path to the FITS catalog or DR3 tile directory.
        #[arg(long)]
        input: Option<PathBuf>,
        /// Selected MaNGA galaxy CSV, typically dapall_selection.csv.
        #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
        manga_selection: PathBuf,
        /// MaNGA DRPall FITS with target coordinates.
        #[arg(long, default_value = "data/external/manga/drpall-v3_1_1.fits")]
        manga_drpall: PathBuf,
        /// Match radius in arcseconds.
        #[arg(long, default_value = "3.0")]
        radius_arcsec: f64,
        /// Output crossmatch CSV (default: data/external/manga/manga_lotss_xmatch_<release>.csv).
        #[arg(long)]
        output: Option<PathBuf>,
        /// Output report TOML (default: reports/lotss_manga_crossmatch_<release>_YYYY-MM-DD.toml).
        #[arg(long)]
        report: Option<PathBuf>,
        /// DR3 footprint summary TOML. Required for strict DR3 tile analysis unless --allow-partial.
        #[arg(long)]
        summary: Option<PathBuf>,
        /// Permit DR3 tile analysis when the footprint summary reports failures or is missing.
        #[arg(long, default_value_t = false)]
        allow_partial: bool,
        /// Limit the number of worker threads. Defaults to one worker per detected physical core.
        #[arg(long)]
        workers: Option<usize>,
        /// Override the streaming chunk size for FITS scans. Defaults to an L3-aware auto value.
        #[arg(long)]
        chunk_rows: Option<usize>,
    },
}

/// Release selector for the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ReleaseArg {
    Dr1,
    Dr2,
    Dr3,
}

impl ReleaseArg {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Dr1 => "DR1",
            Self::Dr2 => "DR2",
            Self::Dr3 => "DR3",
        }
    }

    pub(crate) fn as_catalog_release(self) -> LoTSSRelease {
        match self {
            Self::Dr1 => LoTSSRelease::DR1,
            Self::Dr2 => LoTSSRelease::DR2,
            Self::Dr3 => LoTSSRelease::DR3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum InputFormatArg {
    Fits,
    Dr3Tiles,
}

impl InputFormatArg {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Fits => "fits",
            Self::Dr3Tiles => "dr3-tiles",
        }
    }
}
