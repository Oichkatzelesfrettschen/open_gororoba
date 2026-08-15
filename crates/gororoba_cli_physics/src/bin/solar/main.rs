//! Dispatcher over the solar-wind and solar-activity lanes.
//!
//! Each lane owns a sibling module exposing a `Cli` argument struct and a `run`
//! entry point, so the lanes share one link unit instead of one executable
//! apiece. The modules sit under `src/bin/solar/` rather than in the library,
//! which keeps them out of every binary that imports `gororoba_cli_physics` and
//! links them only here.

use clap::{Parser, Subcommand};

mod aia_cd;
mod cycle_crosscorr;
mod flare_cd;
mod storm_propagation;
mod wind_dm_mhd;
mod wind_ic;
mod wind_mhd_sim;

#[derive(Parser)]
#[command(name = "solar", about = "Solar wind and solar activity lanes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

// `wind_ic::Cli` carries ~190 clap fields and dwarfs the other six payloads.
// One `Command` exists per process: `Cli::parse()` builds it and the match in
// `main` destructures it immediately, so the widest variant costs one stack
// frame rather than per-element storage. Boxing is the lint's suggested fix and
// does not apply, because clap's `Subcommand` derive requires each payload to
// implement `Args` and `Box<T>` does not.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Command {
    /// SDO AIA multi-channel CD: 6-band EUV plus HMI SHARP through the X9.3 flare
    AiaCd(aia_cd::Cli),
    /// Solar cycle cross-correlation: 1 AU fleet CD against Voyager CD
    CycleCrosscorr(cycle_crosscorr::Cli),
    /// Solar flare CD associator: magnetic topology transitions in SHARP keywords
    FlareCd(flare_cd::Cli),
    /// Solar storm propagation tracker: CME from SDO through ACE to the Voyager ISM
    StormPropagation(storm_propagation::Cli),
    /// D3Q19 LBM plus MHD with dark-matter gravitational coupling
    WindDmMhd(wind_dm_mhd::Cli),
    /// Real-data solar wind initial-condition generator
    WindIc(wind_ic::Cli),
    /// D3Q19 LBM plus MHD simulation of the magnetized solar wind
    WindMhdSim(wind_mhd_sim::Cli),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::AiaCd(cli) => aia_cd::run(cli),
        Command::CycleCrosscorr(cli) => cycle_crosscorr::run(cli),
        Command::FlareCd(cli) => flare_cd::run(cli),
        Command::StormPropagation(cli) => storm_propagation::run(cli),
        Command::WindDmMhd(cli) => wind_dm_mhd::run(cli),
        Command::WindIc(cli) => wind_ic::run(cli),
        Command::WindMhdSim(cli) => wind_mhd_sim::run(cli),
    }
}
