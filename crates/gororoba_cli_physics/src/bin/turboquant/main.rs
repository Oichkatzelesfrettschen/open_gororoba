//! Dispatcher over the TurboQuant KV-cache quantization lanes.
//!
//! Each lane owns a sibling module exposing a `Cli` argument struct and a `run`
//! entry point, so the lanes share one link unit instead of one executable
//! apiece. The modules sit under `src/bin/turboquant/` rather than in the
//! library, which keeps them out of every binary that imports
//! `gororoba_cli_physics` and links them only here.

use clap::{Parser, Subcommand};

mod bench;
mod cd_fidelity;
mod comparison;
mod onnx_eval;
mod production;
mod profile;
mod real_kv;
mod sedenion_rotation;
mod simd_bench;
mod sweep;
mod validate;

#[derive(Parser)]
#[command(name = "turboquant", about = "TurboQuant KV-cache quantization lanes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// TurboQuant pipeline benchmark: rotation + quantization + QJL
    Bench(bench::Cli),
    /// Measure phase-geometry distortion from KV-cache quantization with the 32D (pathion) associator
    CdFidelity(cd_fidelity::Cli),
    /// Head-to-head comparison: TurboQuant vs KIVI vs NSNQuant
    Comparison(comparison::Cli),
    /// TurboQuant real-model evaluation via ONNX Runtime
    OnnxEval(onnx_eval::Cli),
    /// Production LLM benchmark with TurboQuant KV cache
    Production(production::Cli),
    /// Tight-loop quantization pass for perf and flamegraph capture
    Profile(profile::Cli),
    /// Evaluate TurboQuant on real LLM KV cache tensors
    RealKv(real_kv::Cli),
    /// Compare 16D (sedenion) left-multiplication rotation against Haar random rotation
    SedenionRotation(sedenion_rotation::Cli),
    /// Compare scalar against SIMD Lloyd-Max codebook lookup
    SimdBench(simd_bench::Cli),
    /// Definitive all-methods comparison sweep
    Sweep(sweep::Cli),
    /// TurboQuant quality validation: cosine, top-k, compression ratio
    Validate(validate::Cli),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Bench(cli) => bench::run(cli),
        Command::CdFidelity(cli) => cd_fidelity::run(cli),
        Command::Comparison(cli) => comparison::run(cli),
        Command::OnnxEval(cli) => onnx_eval::run(cli),
        Command::Production(cli) => production::run(cli),
        // The profiling lane is infallible: it allocates, loops and prints.
        Command::Profile(cli) => {
            profile::run(cli);
            Ok(())
        }
        Command::RealKv(cli) => real_kv::run(cli),
        Command::SedenionRotation(cli) => sedenion_rotation::run(cli),
        Command::SimdBench(cli) => simd_bench::run(cli),
        Command::Sweep(cli) => sweep::run(cli),
        Command::Validate(cli) => validate::run(cli),
    }
}
