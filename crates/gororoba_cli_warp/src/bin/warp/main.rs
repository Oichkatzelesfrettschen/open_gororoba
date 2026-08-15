//! Dispatcher over the warp-drive lattice lanes.
//!
//! Each lane owns a sibling module exposing a `run` entry point, so the
//! fourteen lanes share one link unit instead of one executable apiece. The
//! modules sit under `src/bin/warp/` rather than in the library, which keeps
//! them out of every binary that imports `gororoba_cli_warp`.
//!
//! Unlike the other collapsed clusters these lanes never adopted clap: nine of
//! the fourteen read positional arguments straight off `std::env::args()` and
//! index them by ordinal. Converting them to flags would rewrite every
//! reproduction command recorded against them, so each lane instead takes its
//! argument vector as a slice and the dispatcher forwards whatever follows the
//! lane name verbatim. `warp-gpu-smoke 5.0 5 out` becomes
//! `warp gpu-smoke 5.0 5 out`, ordinal for ordinal.
//!
//! `gororoba_cli_warp` builds only with its `gpu` feature, which its own
//! library forces: `warp_precision_suite_ops` and `warp_runner` import
//! `lbm_3d_cuda` with no `cfg`, so `--no-default-features` fails at the library
//! before any binary is reached. The manifest therefore declares `gpu` on the
//! dispatcher, which restates what the crate already requires.

use clap::{Parser, Subcommand};
use std::error::Error;

mod acceptance_gate;
mod basemark_suite;
mod bench_precision;
mod enstrophy_test;
mod fastpath_analyze;
mod fastpath_suite;
mod gpu_experiment;
mod gpu_smoke;
mod lbm_step_test;
mod precision_matrix;
mod precision_suite;
mod production_suite;
mod ring_3d;
mod ring_integration;

/// Positional arguments forwarded to a lane without interpretation.
///
/// `allow_hyphen_values` keeps clap from claiming a lane's own `--profile` or
/// `-h`, and `trailing_var_arg` stops it splitting the tail into further
/// subcommands.
#[derive(clap::Args)]
struct Forwarded {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

#[derive(Parser)]
#[command(name = "warp", about = "Warp-drive lattice simulation and gate lanes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// HDF5 acceptance gate over warp run artifacts
    AcceptanceGate(Forwarded),
    /// Basemark sweep across grid sizes, precisions and timing modes
    BasemarkSuite(Forwarded),
    /// Precision benchmark, in the ordinal form the precision suite accepts
    BenchPrecision(Forwarded),
    /// Single-shot enstrophy readback from the CUDA D3Q19 backend
    EnstrophyTest,
    /// Report over a completed fastpath production directory
    FastpathAnalyze(Forwarded),
    /// BF16 fastpath sweep followed by a production run
    FastpathSuite(Forwarded),
    /// Warp turbulence experiments A, B and C
    GpuExperiment(Forwarded),
    /// Short CUDA smoke run with trace output
    GpuSmoke(Forwarded),
    /// Timed 128^3 FP32 D3Q19 step loop
    LbmStepTest,
    /// Precision matrix, in the ordinal form the precision suite accepts
    PrecisionMatrix(Forwarded),
    /// Precision suite: `bench` and `matrix` modes
    PrecisionSuite(Forwarded),
    /// Production runs at 128^3 and 256^3 with optional FP32 reference
    ProductionSuite(Forwarded),
    /// 128^3 BF16 warp ring with the E7 spectral filter
    // clap derives `ring3d` from the variant, which does not match the
    // `warp-ring-3d` target this lane replaces.
    #[command(name = "ring-3d")]
    Ring3d,
    /// Engine-backed 64^2 warp ring integration over the Kolmogorov triad
    RingIntegration,
}

/// Rebuild the argument vector a lane would have seen as its own executable.
///
/// Every ordinal lane indexes from `args.get(1)`, because slot zero held the
/// program name. Forwarding the user's arguments alone would shift each lane's
/// defaults one position left, and the lanes fall back to defaults rather than
/// erroring, so the shift would be silent. Reinstating a program-name slot
/// keeps `get(1)` meaning the first user argument.
fn lane_argv(lane: &str, forwarded: &[String]) -> Vec<String> {
    let mut argv = Vec::with_capacity(forwarded.len() + 1);
    argv.push(format!("warp {lane}"));
    argv.extend_from_slice(forwarded);
    argv
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    match Cli::parse().command {
        Command::AcceptanceGate(f) => acceptance_gate::run(&lane_argv("acceptance-gate", &f.args)),
        Command::BasemarkSuite(f) => basemark_suite::run(&lane_argv("basemark-suite", &f.args)),
        Command::BenchPrecision(f) => bench_precision::run(&lane_argv("bench-precision", &f.args)),
        Command::EnstrophyTest => enstrophy_test::run(),
        Command::FastpathAnalyze(f) => {
            fastpath_analyze::run(&lane_argv("fastpath-analyze", &f.args))
        }
        Command::FastpathSuite(f) => fastpath_suite::run(&lane_argv("fastpath-suite", &f.args)),
        Command::GpuExperiment(f) => gpu_experiment::run(&lane_argv("gpu-experiment", &f.args)),
        Command::GpuSmoke(f) => gpu_smoke::run(&lane_argv("gpu-smoke", &f.args)),
        Command::LbmStepTest => lbm_step_test::run(),
        Command::PrecisionMatrix(f) => {
            precision_matrix::run(&lane_argv("precision-matrix", &f.args))
        }
        Command::PrecisionSuite(f) => precision_suite::run(&lane_argv("precision-suite", &f.args)),
        Command::ProductionSuite(f) => {
            production_suite::run(&lane_argv("production-suite", &f.args))
        }
        Command::Ring3d => ring_3d::run(),
        Command::RingIntegration => ring_integration::run(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Cli, lane_argv};
    use clap::Parser;

    #[test]
    fn splice_puts_the_first_user_argument_at_slot_one() {
        let argv = lane_argv("gpu-smoke", &["5.0".to_string(), "5".to_string()]);
        assert_eq!(argv.len(), 3);
        assert_eq!(argv[0], "warp gpu-smoke");
        assert_eq!(argv[1], "5.0");
        assert_eq!(argv[2], "5");
    }

    #[test]
    fn splice_of_no_arguments_still_fills_slot_zero() {
        let argv = lane_argv("precision-suite", &[]);
        assert_eq!(argv, vec!["warp precision-suite".to_string()]);
        assert!(argv.get(1).is_none());
    }

    /// A lane flag reaches the lane rather than the dispatcher.
    #[test]
    fn hyphenated_lane_arguments_pass_through() {
        let cli = Cli::parse_from(["warp", "acceptance-gate", "--profile", "canonical-300s"]);
        let super::Command::AcceptanceGate(f) = cli.command else {
            panic!("expected the acceptance-gate lane");
        };
        let argv = lane_argv("acceptance-gate", &f.args);
        assert_eq!(argv[1], "--profile");
        assert_eq!(argv[2], "canonical-300s");
    }

    /// `acceptance-gate` skips slot zero itself, so the splice must survive
    /// that second skip with the user's first argument intact.
    #[test]
    fn splice_survives_a_lane_that_skips_slot_zero() {
        let argv = lane_argv("acceptance-gate", &["--allow-empty".to_string()]);
        let seen: Vec<String> = argv.iter().skip(1).cloned().collect();
        assert_eq!(seen, vec!["--allow-empty".to_string()]);
    }

    /// A lane taking no arguments rejects them rather than ignoring them.
    #[test]
    fn no_argument_lanes_reject_extra_arguments() {
        assert!(Cli::try_parse_from(["warp", "ring-3d"]).is_ok());
        assert!(Cli::try_parse_from(["warp", "ring-3d", "128"]).is_err());
    }

    /// Every lane answers to the tail of the target name it replaces. clap
    /// derives `ring3d` from `Ring3d`, which would silently drop the hyphen
    /// `warp-ring-3d` carried, so the mapping is asserted rather than assumed.
    #[test]
    fn lane_names_match_the_targets_they_replace() {
        for lane in [
            "acceptance-gate",
            "basemark-suite",
            "bench-precision",
            "enstrophy-test",
            "fastpath-analyze",
            "fastpath-suite",
            "gpu-experiment",
            "gpu-smoke",
            "lbm-step-test",
            "precision-matrix",
            "precision-suite",
            "production-suite",
            "ring-3d",
            "ring-integration",
        ] {
            assert!(
                Cli::try_parse_from(["warp", lane]).is_ok(),
                "lane `{lane}` is unreachable"
            );
        }
    }
}
