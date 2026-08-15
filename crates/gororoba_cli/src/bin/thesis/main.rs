//! Dispatcher over the thesis evidence lanes.
//!
//! Each lane owns a sibling module exposing an argument struct and a `run`
//! entry point, so the lanes share one link unit instead of one executable
//! apiece. The modules sit under `src/bin/thesis/` rather than in the library,
//! which keeps them out of every binary that imports `gororoba_cli`.
//!
//! The `synthesis` lane drives `lbm_3d_cuda` and carries
//! `#[cfg(feature = "gpu")]` on both its module and its variant; a gated module
//! behind an ungated variant breaks the default build, so the two attributes
//! move together. The remaining seven lanes build in the default configuration.

use clap::{Parser, Subcommand};

mod cross_tx1;
mod cross_tx2;
mod cross_tx3;
mod lab;
mod program_sweep;
mod support_42;
#[cfg(feature = "gpu")]
mod synthesis;
mod synthesis_engine;

#[derive(Parser)]
#[command(name = "thesis", about = "Thesis evidence generation lanes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate an evidence-first thesis support bundle for the 42-lane synthesis
    #[command(name = "42-support")]
    Support42(support_42::Args),
    /// TX-1: imbalance-modulated collision dynamics (T1 x T4)
    CrossTx1(cross_tx1::Args),
    /// TX-2: viscosity-to-filtration loop (T2 x T4)
    CrossTx2(cross_tx2::Args),
    /// TX-3: topological persistence of the viscosity landscape
    CrossTx3(cross_tx3::Args),
    /// Experiment-lab concept explorer for the imbalance-topology-viscosity hypothesis
    Lab(lab::Args),
    /// Generate deterministic thesis evidence artifacts for STPT-006..009
    ProgramSweep(program_sweep::Args),
    /// Thesis synthesis and experiment engine
    #[cfg(feature = "gpu")]
    Synthesis(synthesis::Cli),
    /// Orchestrate all four grand synthesis thesis pipelines
    SynthesisEngine(synthesis_engine::Args),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    // Six lanes predate the workspace `anyhow` convention and return a boxed
    // error, which is neither `Send` nor `Sync` and so cannot cross
    // `anyhow::Error::from`. Rendering it preserves the message.
    let boxed = |err: Box<dyn std::error::Error>| anyhow::anyhow!("{err}");
    match Cli::parse().command {
        Command::Support42(args) => support_42::run(args),
        Command::CrossTx1(args) => cross_tx1::run(args).map_err(boxed),
        Command::CrossTx2(args) => cross_tx2::run(args).map_err(boxed),
        Command::CrossTx3(args) => cross_tx3::run(args).map_err(boxed),
        Command::Lab(args) => lab::run(args).map_err(boxed),
        Command::ProgramSweep(args) => program_sweep::run(args).map_err(boxed),
        #[cfg(feature = "gpu")]
        Command::Synthesis(cli) => synthesis::run(cli).map_err(boxed),
        Command::SynthesisEngine(args) => synthesis_engine::run(args).map_err(boxed),
    }
}
