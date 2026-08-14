//! Dispatcher over the Voyager ingest and analysis lanes.
//!
//! Each lane owns a sibling module exposing a `Cli` argument struct and a `run`
//! entry point, so the lanes share one link unit instead of one executable
//! apiece. The modules sit under `src/bin/voyager/` rather than in the library,
//! which keeps them out of every binary that imports `gororoba_cli_physics` and
//! links them only here.

use clap::{Parser, Subcommand};

mod arrow_probe;
mod cdf_ingest;
mod crs_compare;
mod encounter_track;
mod hapi_ingest;
mod ism_multichannel;
mod mag_pls_cd;
mod pws_ingest;

#[derive(Parser)]
#[command(name = "voyager", about = "Voyager ingest and analysis lanes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Memory-map a Voyager Arrow artifact, assert schema, and locate telemetry bounds for a target timestamp
    ArrowProbe(arrow_probe::Cli),
    /// Ingest Voyager 48-second MAG CDF files into HeliosphereFeatureRow CSV
    CdfIngest(cdf_ingest::Cli),
    /// Voyager CRS comparison in shape or calibrated absolute-flux mode
    CrsCompare(crs_compare::Cli),
    /// Fuse promoted encounter Arrow telemetry with merged Voyager spatial coordinates and map them into lattice indices
    EncounterTrack(encounter_track::Cli),
    /// Ingest Voyager HAPI CSV data into HeliosphereFeatureRow format
    HapiIngest(hapi_ingest::Cli),
    /// Multi-instrument ISM CD analysis: MAG + PWS combined 20-channel embedding
    IsmMultichannel(ism_multichannel::Cli),
    /// Voyager MAG+PLS 7-channel CD: joint magnetic and plasma embedding
    MagPlsCd(mag_pls_cd::Cli),
    /// Ingest Voyager PWS spectrum analyzer CDF files into a multi-channel CSV
    PwsIngest(pws_ingest::Cli),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::ArrowProbe(cli) => arrow_probe::run(cli),
        Command::CdfIngest(cli) => cdf_ingest::run(cli),
        // The CRS comparison lane reports through stdout and exits: it has no
        // fallible tail to propagate.
        Command::CrsCompare(cli) => {
            crs_compare::run(cli);
            Ok(())
        }
        Command::EncounterTrack(cli) => encounter_track::run(cli),
        Command::HapiIngest(cli) => hapi_ingest::run(cli),
        Command::IsmMultichannel(cli) => ism_multichannel::run(cli),
        Command::MagPlsCd(cli) => mag_pls_cd::run(cli),
        Command::PwsIngest(cli) => pws_ingest::run(cli),
    }
}
