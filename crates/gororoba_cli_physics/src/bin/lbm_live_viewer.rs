//! Interactive LBM viewer built on backend-neutral viewer contracts.
//!
//! The current production path uses a CUDA LBM backend adapter that exposes the
//! real solver density field through `gororoba_view_core::ViewerFrameSource`.
//! The frontend loop, input handling, and frame transport are split into local
//! modules so later Vulkan, CPU, and OptiX adapters can plug into the same
//! viewer without rewriting the windowing code.
//!
//! # Current backend
//!
//! ```text
//! CUDA LBM solver
//!     |
//!     v  ViewerFrameSource::copy_frame(FrameMode::Volume3d)
//! frontend transport: volume -> colored slice framebuffer
//!     |
//!     v
//! minifb window
//! ```
//!
//! This intentionally visualizes the actual CUDA solver state instead of a
//! separate renderer-owned simulation.

use anyhow::Result;
use clap::{Parser, ValueEnum};

#[path = "lbm_live_viewer/backend.rs"]
mod backend;
#[path = "lbm_live_viewer/camera_input.rs"]
mod camera_input;
#[path = "lbm_live_viewer/frontend.rs"]
mod frontend;
#[path = "lbm_live_viewer/transport.rs"]
mod transport;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ViewerBackend {
    Cpu,
    Cuda,
}

#[derive(Parser, Debug)]
#[command(name = "lbm-live-viewer", version)]
struct Config {
    /// Execution backend for the viewer source.
    #[arg(long, value_enum, default_value_t = ViewerBackend::Cuda)]
    backend: ViewerBackend,

    /// Grid side length.
    #[arg(long, default_value_t = 64)]
    grid: usize,

    /// LBM steps per displayed frame.
    #[arg(long, default_value_t = 10)]
    steps_per_frame: usize,

    /// Relaxation time tau.
    #[arg(long, default_value_t = 0.7)]
    tau: f64,

    /// Window width in pixels.
    #[arg(long, default_value_t = 1280)]
    width: u32,

    /// Window height in pixels.
    #[arg(long, default_value_t = 960)]
    height: u32,

    /// Maximum frames to run before exit. Zero means unlimited.
    #[arg(long, default_value_t = 0)]
    max_frames: u64,

    /// Use MRT collision.
    #[arg(long, default_value_t = true)]
    use_mrt: bool,
}

fn run_viewer(cfg: &Config) -> Result<()> {
    let mut source = backend::build_viewer_source(
        match cfg.backend {
            ViewerBackend::Cpu => backend::ViewerBackendKind::Cpu,
            ViewerBackend::Cuda => backend::ViewerBackendKind::Cuda,
        },
        cfg.grid,
        cfg.tau,
        cfg.use_mrt,
    )?;
    frontend::run_frontend(
        source.as_mut(),
        frontend::FrontendConfig {
            width: cfg.width,
            height: cfg.height,
            max_frames: cfg.max_frames,
            initial_steps_per_frame: cfg.steps_per_frame,
        },
    )
}

fn main() -> Result<()> {
    let cfg = Config::parse();
    run_viewer(&cfg)
}
