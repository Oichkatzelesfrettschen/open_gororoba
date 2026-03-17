//! Frame transport helpers for `lbm-live-viewer`.
//!
//! These helpers translate backend-neutral frame packets into the concrete ARGB
//! framebuffer expected by `minifb`. They intentionally know nothing about CUDA
//! or Vulkan runtime details.

use crate::camera_input::{SliceAxis, ViewerInteractionState};
use anyhow::Result;
use gororoba_view_core::{GridShape3d, SliceFrameRgba8, ViewerFramePacket, VolumeFrameF32};

/// Render a backend-neutral frame packet into a `minifb` ARGB framebuffer.
pub fn render_packet_to_argb(
    packet: &ViewerFramePacket,
    state: &ViewerInteractionState,
    framebuffer: &mut [u32],
    fb_width: usize,
    fb_height: usize,
) -> Result<()> {
    match packet {
        ViewerFramePacket::VolumeF32(volume) => {
            render_volume_slice_to_argb(volume, state, framebuffer, fb_width, fb_height);
            Ok(())
        }
        ViewerFramePacket::SliceRgba8(slice) => {
            blit_rgba_slice(slice, framebuffer, fb_width, fb_height);
            Ok(())
        }
        ViewerFramePacket::Particles(_) => {
            framebuffer.fill(0xFF00_0000);
            anyhow::bail!("particle packets are not yet supported by lbm-live-viewer")
        }
    }
}

fn render_volume_slice_to_argb(
    volume: &VolumeFrameF32,
    state: &ViewerInteractionState,
    framebuffer: &mut [u32],
    fb_width: usize,
    fb_height: usize,
) {
    let grid = volume.grid;
    let slice_index = state.slice_index.min(state.slice_axis.max_index(grid));
    let (mut field_min, mut field_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &value in &volume.values {
        if value.is_finite() {
            field_min = field_min.min(value);
            field_max = field_max.max(value);
        }
    }
    if !field_min.is_finite() || !field_max.is_finite() {
        field_min = 0.0;
        field_max = 1.0;
    }
    let lut = viridis_lut();
    let (sw, sh) = slice_dimensions(grid, state.slice_axis);
    let scale_x = sw as f32 / fb_width as f32;
    let scale_y = sh as f32 / fb_height as f32;
    let denom = (field_max - field_min).max(1.0e-10);
    for row in 0..fb_height {
        let sy = ((row as f32 * scale_y) as usize).min(sh.saturating_sub(1));
        let row_base = row * fb_width;
        for col in 0..fb_width {
            let sx = ((col as f32 * scale_x) as usize).min(sw.saturating_sub(1));
            let idx = volume_index(grid, state.slice_axis, slice_index, sx, sy);
            let value = volume.values.get(idx).copied().unwrap_or(0.0);
            let t = ((value - field_min) / denom).clamp(0.0, 1.0);
            framebuffer[row_base + col] = lut[(t * 255.0) as usize];
        }
    }
}

fn blit_rgba_slice(
    slice: &SliceFrameRgba8,
    framebuffer: &mut [u32],
    fb_width: usize,
    fb_height: usize,
) {
    let scale_x = slice.width as f32 / fb_width as f32;
    let scale_y = slice.height as f32 / fb_height as f32;
    for row in 0..fb_height {
        let sy = ((row as f32 * scale_y) as usize).min(slice.height as usize - 1);
        let row_base = row * fb_width;
        for col in 0..fb_width {
            let sx = ((col as f32 * scale_x) as usize).min(slice.width as usize - 1);
            let idx = (sy * slice.width as usize + sx) * 4;
            let r = slice.pixels[idx] as u32;
            let g = slice.pixels[idx + 1] as u32;
            let b = slice.pixels[idx + 2] as u32;
            framebuffer[row_base + col] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
        }
    }
}

fn slice_dimensions(grid: GridShape3d, axis: SliceAxis) -> (usize, usize) {
    match axis {
        SliceAxis::X => (grid.ny as usize, grid.nz as usize),
        SliceAxis::Y => (grid.nx as usize, grid.nz as usize),
        SliceAxis::Z => (grid.nx as usize, grid.ny as usize),
    }
}

fn volume_index(
    grid: GridShape3d,
    axis: SliceAxis,
    slice_index: usize,
    sx: usize,
    sy: usize,
) -> usize {
    let nx = grid.nx as usize;
    let ny = grid.ny as usize;
    match axis {
        SliceAxis::X => sy * ny * nx + sx * nx + slice_index.min(nx - 1),
        SliceAxis::Y => sy * ny * nx + slice_index.min(ny - 1) * nx + sx,
        SliceAxis::Z => slice_index.min(grid.nz as usize - 1) * ny * nx + sy * nx + sx,
    }
}

fn viridis_lut() -> [u32; 256] {
    let mut lut = [0u32; 256];
    for (i, item) in lut.iter_mut().enumerate() {
        let t = i as f32 / 255.0;
        let r = ((-1.27 * t + 2.47) * t * t * 255.0).clamp(0.0, 255.0) as u32;
        let g = ((0.83 * t - 1.72) * t * t * 255.0 + 127.0 * t).clamp(0.0, 255.0) as u32;
        let b = ((4.28 * (1.0 - t) - 1.0) * (1.0 - t) * 255.0).clamp(0.0, 255.0) as u32;
        *item = 0xFF00_0000 | (r << 16) | (g << 8) | b;
    }
    lut
}
