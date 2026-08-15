//! Dispatcher over the NANOGrav 15-year timing-release lanes.
//!
//! Each lane owns a sibling module exposing an `Args` payload and a `run` entry
//! point, so the ten lanes share one link unit instead of one executable
//! apiece. The modules sit under `src/bin/nanograv/` rather than in the
//! library, which keeps them out of every binary that imports
//! `gororoba_cli_data` and links them only here.
//!
//! The lanes derive `clap::Args` through the fully-qualified path while keeping
//! the struct name `Args`. Importing the trait as `use clap::Args` beside a
//! local `struct Args` is E0255, and the qualified derive avoids renaming the
//! binding at every field access inside each lane.

use clap::{Parser, Subcommand};

mod avt_filter;
mod entropy_audit;
mod gauge_resonance;
mod propagation_audit;
mod synthetic_gen;
mod timing_inventory;
mod timing_phase1_independent;
mod timing_phase1_refit;
mod timing_refit_preflight;
mod vacuum_symmetry;

#[derive(Parser)]
#[command(name = "nanograv", about = "NANOGrav 15-year timing-release lanes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Cross-validated AVT mean-shift audit over NANOGrav timing residuals
    AvtFilter(avt_filter::Args),
    /// Correlates NANOGrav pulsar residuals with 1024D DekaVoudon entropy bounds
    EntropyAudit(entropy_audit::Args),
    /// Correlates NANOGrav pulsar variance with 1024D Standard Model gauge sector cross-coupling
    GaugeResonance(gauge_resonance::Args),
    /// Per-pulsar propagation diagnostics and first-pass Hellings-Downs pair audit
    PropagationAudit(propagation_audit::Args),
    /// Generates a synthetic NANOGrav dataset with pure Hellings-Downs correlation and standard noise
    SyntheticGen(synthetic_gen::Args),
    /// Inventory the full timing release into per-pulsar coverage summaries
    TimingInventory(timing_inventory::Args),
    /// Independent TOA-driven Phase 1 timing-engine pilot over all Phase 1 frontend/backend groups
    TimingPhase1Independent(timing_phase1_independent::Args),
    /// Linearized WLS/GLS pilot refit over the six Phase 1 timing-model scout pulsars
    TimingPhase1Refit(timing_phase1_refit::Args),
    /// Typed .par preflight inventory for the timing-refit lane
    TimingRefitPreflight(timing_refit_preflight::Args),
    /// Detects point group symmetry in the 1024D DekaVoudon sign imbalance field
    VacuumSymmetry(vacuum_symmetry::Args),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::AvtFilter(args) => avt_filter::run(args),
        Command::EntropyAudit(args) => entropy_audit::run(args),
        Command::GaugeResonance(args) => gauge_resonance::run(args),
        Command::PropagationAudit(args) => propagation_audit::run(args),
        Command::SyntheticGen(args) => synthetic_gen::run(args),
        Command::TimingInventory(args) => timing_inventory::run(args),
        Command::TimingPhase1Independent(args) => timing_phase1_independent::run(args),
        Command::TimingPhase1Refit(args) => timing_phase1_refit::run(args),
        Command::TimingRefitPreflight(args) => timing_refit_preflight::run(args),
        Command::VacuumSymmetry(args) => vacuum_symmetry::run(args),
    }
}
