//! Dispatcher over the solar-system ephemeris lanes.
//!
//! Each lane owns a sibling module exposing a `Cli` argument struct and a `run`
//! entry point, so the lanes share one link unit instead of one executable
//! apiece. The modules sit under `src/bin/ephemeris/` rather than in the
//! library, which keeps them out of every binary that imports
//! `gororoba_cli_physics` and links them only here.

use clap::{Parser, Subcommand};

mod positions;

#[derive(Parser)]
#[command(
    name = "ephemeris",
    about = "Solar-system body positions from a JPL DE-series kernel"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Heliocentric ecliptic positions of the planet barycentres at an epoch
    Positions(positions::Cli),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Positions(cli) => positions::run(cli),
    }
}
