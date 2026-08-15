//! Dispatcher over the zero-divisor graph spectral lanes.
//!
//! Both lanes build a zero-divisor graph Laplacian and read a spectral
//! observable off it -- nearest-neighbour spacing statistics in one case, the
//! random-walk spectral dimension in the other -- so they resolve the same
//! dependency closure and share one link unit here. The modules sit under
//! `src/bin/zd_spectral/` rather than in the library, which keeps them out of
//! every binary that imports `gororoba_cli_data`.
//!
//! The name carries the observable rather than the crate because `zd` alone is
//! already claimed: `gororoba_cli`, `gororoba_cli_algebra` and
//! `gororoba_cli_data` each hold a `zd-` group, and two packages cannot both
//! publish a target named `zd` into one target directory.

use clap::{Parser, Subcommand};

mod dimension;
mod quantum_chaos;

#[derive(Parser)]
#[command(
    name = "zd-spectral",
    about = "Spectral observables over zero-divisor graph Laplacians"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// NNSD analysis of zero-divisor graph Laplacian spectra
    QuantumChaos(quantum_chaos::Cli),
    /// Compute spectral dimension of zero-divisor graphs across CD dimensions
    Dimension(dimension::Cli),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::QuantumChaos(cli) => quantum_chaos::run(cli),
        Command::Dimension(cli) => dimension::run(cli),
    }
}
