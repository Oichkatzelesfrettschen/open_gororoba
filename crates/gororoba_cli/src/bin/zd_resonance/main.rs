//! Dispatcher over the zero-divisor resonance lanes.
//!
//! Each lane owns a sibling module exposing a `Cli` argument struct and a `run`
//! entry point, so the lanes share one link unit instead of one executable
//! apiece. The modules sit under `src/bin/zd_resonance/` rather than in the
//! library, which keeps them out of every binary that imports `gororoba_cli`.
//!
//! Three lanes drive `lbm_3d_cuda` and carry `#[cfg(feature = "gpu")]` on both
//! the module and its variant; the `sweep` lane reaches Vulkan through
//! `lbm_vulkan` and builds in the default configuration. A gated module behind
//! an ungated variant breaks the default build, so the two attributes move
//! together.
//!
//! The name carries the observable rather than the crate because `zd` alone is
//! already claimed: `gororoba_cli`, `gororoba_cli_algebra` and
//! `gororoba_cli_data` each hold a `zd-` group, and two packages cannot both
//! publish a target named `zd` into one target directory.

use clap::{Parser, Subcommand};

#[cfg(feature = "gpu")]
mod bf16;
#[cfg(feature = "gpu")]
mod cuda;
#[cfg(feature = "gpu")]
mod four_d;
mod sweep;

#[derive(Parser)]
#[command(
    name = "zd-resonance",
    about = "Sedenion zero-divisor spectral resonance lanes"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 4D ZD resonance via the D3Q19 batch kernel
    #[cfg(feature = "gpu")]
    #[command(name = "4d")]
    FourD(four_d::Cli),
    /// CUDA BF16 zero-divisor resonance experiments
    #[cfg(feature = "gpu")]
    Bf16(bf16::Args),
    /// Multi-subcommand CUDA ZD resonance detection at 128^3 and 256^3
    #[cfg(feature = "gpu")]
    Cuda(cuda::Cli),
    /// ZD resonance falsification sweep over tau, control and lambda
    Sweep(sweep::Cli),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    match Cli::parse().command {
        #[cfg(feature = "gpu")]
        Command::FourD(cli) => four_d::run(cli),
        #[cfg(feature = "gpu")]
        Command::Bf16(args) => bf16::run(args),
        #[cfg(feature = "gpu")]
        Command::Cuda(cli) => cuda::run(cli),
        // The sweep lane predates the workspace `anyhow` convention and returns
        // a boxed error, which is neither `Send` nor `Sync` and so cannot cross
        // `anyhow::Error::from`. Rendering it preserves the message.
        Command::Sweep(cli) => sweep::run(cli).map_err(|err| anyhow::anyhow!("{err}")),
    }
}
