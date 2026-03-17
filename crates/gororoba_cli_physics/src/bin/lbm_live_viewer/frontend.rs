//! Frontend loop for the backend-neutral live viewer.
//!
//! This module owns the `minifb` event loop, status formatting, and frame
//! transport. It intentionally depends only on `ViewerFrameSource` plus a local
//! reset extension trait so backend adapters stay swappable.

use crate::backend::ResettableViewerSource;
use crate::camera_input::{ViewerInteractionState, apply_window_input};
use crate::transport::render_packet_to_argb;
use anyhow::Result;
use gororoba_gpu_bridge::FrameMode;
use minifb::{Key, Window, WindowOptions};
use std::time::Instant;

/// Runtime configuration for the generic viewer frontend.
#[derive(Debug, Clone, Copy)]
pub struct FrontendConfig {
    pub width: u32,
    pub height: u32,
    pub max_frames: u64,
    pub initial_steps_per_frame: usize,
}

/// Run the shared frontend loop against a backend adapter.
pub fn run_frontend<S>(source: &mut S, cfg: FrontendConfig) -> Result<()>
where
    S: ResettableViewerSource + ?Sized,
{
    let initial_meta = source.frame_metadata();
    let mut window = Window::new(
        &format!(
            "{} | {}^3 | gororoba",
            initial_meta.title, initial_meta.grid.nx
        ),
        cfg.width as usize,
        cfg.height as usize,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("Window creation failed: {e}"))?;
    window.set_target_fps(60);

    let mut state = ViewerInteractionState::new(initial_meta.grid, cfg.initial_steps_per_frame);
    let mut framebuffer = vec![0u32; cfg.width as usize * cfg.height as usize];
    let frame_mode = if source.supported_frame_modes().contains(&FrameMode::Volume3d) {
        FrameMode::Volume3d
    } else {
        source.frame_metadata().preferred_frame_mode
    };

    let mut frame_count = 0u64;
    let start = Instant::now();
    let mut last_report = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let actions = apply_window_input(&window, &mut state, source.frame_metadata().grid);
        if actions.request_reset {
            source.reset_simulation()?;
            state.clamp_slice_index(source.frame_metadata().grid);
        }
        if !state.paused {
            source.step_simulation(state.steps_per_frame)?;
        }
        let packet = source.copy_frame(frame_mode)?;
        render_packet_to_argb(
            &packet,
            &state,
            &mut framebuffer,
            cfg.width as usize,
            cfg.height as usize,
        )?;
        window
            .update_with_buffer(&framebuffer, cfg.width as usize, cfg.height as usize)
            .map_err(|e| anyhow::anyhow!("Window update failed: {e}"))?;

        frame_count += 1;
        let meta = source.frame_metadata();
        if last_report.elapsed().as_secs_f64() >= 1.0 {
            let elapsed = start.elapsed().as_secs_f64();
            let fps = frame_count as f64 / elapsed;
            let mlups = meta.mlups_hint.unwrap_or(0.0);
            let status = if state.paused { "PAUSED" } else { "LIVE" };
            window.set_title(&format!(
                "{} | {} | {:.0} FPS | {:.0} MLUPS | {} steps/f | slice {}:{}",
                meta.title,
                status,
                fps,
                mlups,
                state.steps_per_frame,
                state.slice_axis.as_str(),
                state.slice_index
            ));
            last_report = Instant::now();
        }

        if cfg.max_frames > 0 && frame_count >= cfg.max_frames {
            break;
        }
    }

    Ok(())
}
