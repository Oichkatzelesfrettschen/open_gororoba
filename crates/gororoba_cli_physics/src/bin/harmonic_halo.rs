//! Dispatcher over the harmonic-halo analysis lanes.
//!
//! Each lane owns a module under `gororoba_cli_physics::harmonic_halo` exposing
//! a `Cli` argument struct and a `run` entry point, so the lanes share one link
//! unit instead of one executable apiece.

use clap::{Parser, Subcommand};
use gororoba_cli_physics::harmonic_halo;

#[derive(Parser)]
#[command(name = "harmonic-halo", about = "Harmonic-halo analysis lanes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Galaxy-ensemble Rayleigh phase coherence test at CD-ZD wavenumbers
    GalaxyPhase(harmonic_halo::galaxy_phase::Cli),
    /// ZD signal injection-recovery sweep for pipeline validation
    Injection(harmonic_halo::injection::Cli),
    /// Analyze Milky Way rotation curve for harmonic halo modulation
    Mw(harmonic_halo::mw::Cli),
    /// Compute NFW rotation curve with harmonic halo modulation
    RotationCurve(harmonic_halo::rotation_curve::Cli),
    /// Non-static STFT + derivative + Rayleigh phase coherence analysis
    SignalAnalysis(harmonic_halo::signal_analysis::Cli),
    /// Stack rotation curve residuals to detect N-mode harmonic halo signature
    Stacking(harmonic_halo::stacking::Cli),
    /// Stack HI rotation curve residuals for harmonic halo detection with PCA eigenmode analysis
    StackingHi(harmonic_halo::stacking_hi::Cli),
    /// Stack MaNGA rotation curve residuals for harmonic halo detection
    StackingManga(harmonic_halo::stacking_manga::Cli),
    /// Generate PGFPlots panels for E-183/E-192 null result manuscript
    Visualize(harmonic_halo::visualize::Cli),
}

fn main() -> anyhow::Result<()> {
    // Lanes log through `log`, and the logger is process-global, so the
    // dispatcher installs it once for whichever lane runs.
    env_logger::init();
    match Cli::parse().command {
        Command::GalaxyPhase(cli) => harmonic_halo::galaxy_phase::run(cli),
        Command::Injection(cli) => harmonic_halo::injection::run(cli),
        Command::Mw(cli) => harmonic_halo::mw::run(cli),
        Command::RotationCurve(cli) => harmonic_halo::rotation_curve::run(cli),
        Command::SignalAnalysis(cli) => harmonic_halo::signal_analysis::run(cli),
        Command::Stacking(cli) => harmonic_halo::stacking::run(cli),
        Command::StackingHi(cli) => harmonic_halo::stacking_hi::run(cli),
        Command::StackingManga(cli) => harmonic_halo::stacking_manga::run(cli),
        Command::Visualize(cli) => harmonic_halo::visualize::run(cli),
    }
}
