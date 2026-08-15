//! Dispatcher over the zero-divisor structure lanes.
//!
//! Both lanes enumerate zero-divisor pairs in the Cayley-Dickson tower and
//! differ only in what they report over that enumeration, so they resolve the
//! same dependency closure and share one link unit here. The modules sit under
//! `src/bin/zd_algebra/` rather than in the library, which keeps them out of
//! every binary that imports `gororoba_cli_algebra`.
//!
//! The name carries the crate rather than the domain because
//! `gororoba_cli`, `gororoba_cli_algebra` and `gororoba_cli_data` each hold a
//! `zd-` group, and two packages cannot both publish a target named `zd` into
//! one target directory.

use clap::{Parser, Subcommand};

mod crystal_bands;
mod search;

#[derive(Parser)]
#[command(
    name = "zd-algebra",
    about = "Zero-divisor structure lanes over the Cayley-Dickson tower"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Search for zero-divisor pairs in Cayley-Dickson algebras
    Search(search::Args),
    /// ZD adjacency crystal band analysis across CD dimensions
    CrystalBands(crystal_bands::Args),
}

fn main() {
    match Cli::parse().command {
        Command::Search(args) => search::run(args),
        Command::CrystalBands(args) => crystal_bands::run(args),
    }
}
