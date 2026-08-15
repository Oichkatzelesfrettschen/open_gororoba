//! Dispatcher over the harmonic-halo analysis lanes.
//!
//! Each lane owns a sibling module exposing a `Cli` argument struct and a `run`
//! entry point, so the lanes share one link unit instead of one executable
//! apiece. The modules sit under `src/bin/harmonic_halo/` rather than in the
//! library, which keeps them out of every binary that imports
//! `gororoba_cli_physics` and links them only here.

use clap::{Parser, Subcommand};
mod galaxy_phase;
mod injection;
mod mw;
mod rotation_curve;
mod signal_analysis;
mod stacking;
mod stacking_hi;
mod stacking_manga;
mod visualize;

#[derive(Parser)]
#[command(name = "harmonic-halo", about = "Harmonic-halo analysis lanes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Galaxy-ensemble Rayleigh phase coherence test at CD-ZD wavenumbers
    GalaxyPhase(galaxy_phase::Cli),
    /// ZD signal injection-recovery sweep for pipeline validation
    Injection(injection::Cli),
    /// Analyze Milky Way rotation curve for harmonic halo modulation
    Mw(mw::Cli),
    /// Compute NFW rotation curve with harmonic halo modulation
    RotationCurve(rotation_curve::Cli),
    /// Non-static STFT + derivative + Rayleigh phase coherence analysis
    SignalAnalysis(signal_analysis::Cli),
    /// Stack rotation curve residuals to detect N-mode harmonic halo signature
    Stacking(stacking::Cli),
    /// Stack HI rotation curve residuals for harmonic halo detection with PCA eigenmode analysis
    StackingHi(stacking_hi::Cli),
    /// Stack MaNGA rotation curve residuals for harmonic halo detection
    StackingManga(stacking_manga::Cli),
    /// Generate PGFPlots panels for E-183/E-192 null result manuscript
    Visualize(visualize::Cli),
}

fn main() -> anyhow::Result<()> {
    // Lanes log through `log`, and the logger is process-global, so the
    // dispatcher installs it once for whichever lane runs.
    env_logger::init();
    match Cli::parse().command {
        Command::GalaxyPhase(cli) => galaxy_phase::run(cli),
        Command::Injection(cli) => injection::run(cli),
        Command::Mw(cli) => mw::run(cli),
        Command::RotationCurve(cli) => rotation_curve::run(cli),
        Command::SignalAnalysis(cli) => signal_analysis::run(cli),
        Command::Stacking(cli) => stacking::run(cli),
        Command::StackingHi(cli) => stacking_hi::run(cli),
        Command::StackingManga(cli) => stacking_manga::run(cli),
        Command::Visualize(cli) => visualize::run(cli),
    }
}
