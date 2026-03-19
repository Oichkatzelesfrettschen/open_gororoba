//! Lightweight raster helpers for backend-neutral viewer packets.
//!
//! This crate intentionally stays free of CUDA, Vulkan, and solver-specific
//! dependencies. It provides small CPU-side helpers for:
//!
//! - slicing dense scalar volumes into ARGB framebuffers
//! - blitting RGBA packets to ARGB framebuffers
//! - simple particle rasterization for interactive inspection

use gororoba_view_core::{GridShape3d, ParticleFrame, SliceFrameRgba8};

/// Principal slice axis through a 3D volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceAxis {
    X,
    Y,
    Z,
}

impl SliceAxis {
    /// Maximum valid slice index along this axis.
    #[must_use]
    pub fn max_index(self, grid: GridShape3d) -> usize {
        match self {
            Self::X => grid.nx as usize - 1,
            Self::Y => grid.ny as usize - 1,
            Self::Z => grid.nz as usize - 1,
        }
    }
}

/// Color map used for scalar volume rasterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMap {
    Viridis,
    Inferno,
    Turbo, // Neon/Cyberpunk
}

/// Rasterize a complex fractal potential into an ARGB framebuffer using domain coloring.
///
/// Implements the logic from src/vis_hyper_fractal.py.
pub fn render_hyper_fractal_to_argb(
    framebuffer: &mut [u32],
    width: usize,
    height: usize,
    z_min: (f64, f64),
    z_max: (f64, f64),
) {
    let (xmin, ymin) = z_min;
    let (xmax, ymax) = z_max;
    let dx = (xmax - xmin) / width as f64;
    let dy = (ymax - ymin) / height as f64;

    let lut = lookup_table(ColorMap::Turbo);

    for row in 0..height {
        let y = ymin + row as f64 * dy;
        let row_base = row * width;
        for col in 0..width {
            let x = xmin + col as f64 * dx;
            
            // V(z) = sum_{n=1}^7 exp(i*n*pi/4) / (z^n + 0.1)
            let mut v_re = 0.0;
            let mut v_im = 0.0;
            
            for n in 1..=7 {
                let phase = (n as f64) * std::f64::consts::PI / 4.0;
                let p_re = phase.cos();
                let p_im = phase.sin();
                
                // z^n (complex power)
                let mut zn_re = 1.0;
                let mut zn_im = 0.0;
                for _ in 0..n {
                    let tmp_re = zn_re * x - zn_im * y;
                    let tmp_im = zn_re * y + zn_im * x;
                    zn_re = tmp_re;
                    zn_im = tmp_im;
                }
                
                // Denominator: z^n + 0.1
                let d_re = zn_re + 0.1;
                let d_im = zn_im;
                let d_mag_sq = d_re * d_re + d_im * d_im + 1e-12;
                
                // term = phase / denominator
                v_re += (p_re * d_re + p_im * d_im) / d_mag_sq;
                v_im += (p_im * d_re - p_re * d_im) / d_mag_sq;
            }
            
            let mag = (v_re * v_re + v_im * v_im).sqrt();
            // Log scaling for visibility
            let t = ((mag + 1e-9).ln() + 5.0) / 10.0; // Map [-5, 5] -> [0, 1]
            let t_clamped = t.clamp(0.0, 1.0);
            framebuffer[row_base + col] = lut[(t_clamped * 255.0) as usize];
        }
    }
}

/// Configuration for rendering one scalar volume slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceRasterSpec {
    pub axis: SliceAxis,
    pub slice_index: usize,
    pub color_map: ColorMap,
}

/// Rasterize a scalar volume slice into a `minifb`-style ARGB framebuffer.
pub fn render_scalar_volume_slice_to_argb(
    framebuffer: &mut [u32],
    framebuffer_dim: (usize, usize),
    values: &[f32],
    grid: GridShape3d,
    spec: SliceRasterSpec,
) {
    let (fb_width, fb_height) = framebuffer_dim;
    let (mut field_min, mut field_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &value in values {
        if value.is_finite() {
            field_min = field_min.min(value);
            field_max = field_max.max(value);
        }
    }
    if !field_min.is_finite() || !field_max.is_finite() {
        field_min = 0.0;
        field_max = 1.0;
    }
    let lut = lookup_table(spec.color_map);
    let slice_index = spec.slice_index.min(spec.axis.max_index(grid));
    let (sw, sh) = slice_dimensions(grid, spec.axis);
    let scale_x = sw as f32 / fb_width as f32;
    let scale_y = sh as f32 / fb_height as f32;
    let denom = (field_max - field_min).max(1.0e-10);
    for row in 0..fb_height {
        let sy = ((row as f32 * scale_y) as usize).min(sh.saturating_sub(1));
        let row_base = row * fb_width;
        for col in 0..fb_width {
            let sx = ((col as f32 * scale_x) as usize).min(sw.saturating_sub(1));
            let idx = volume_index(grid, spec.axis, slice_index, sx, sy);
            let value = values.get(idx).copied().unwrap_or(0.0);
            let t = ((value - field_min) / denom).clamp(0.0, 1.0);
            framebuffer[row_base + col] = lut[(t * 255.0) as usize];
        }
    }
}

/// Blit an RGBA packet to an ARGB framebuffer with nearest-neighbor scaling.
pub fn blit_rgba_to_argb(
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

/// Rasterize a particle frame with an automatic XY fit into an ARGB
/// framebuffer.
pub fn render_particles_autofit_xy_to_argb(
    particles: &ParticleFrame,
    framebuffer: &mut [u32],
    fb_width: usize,
    fb_height: usize,
) {
    framebuffer.fill(0xFF00_0000);
    if particles.positions.is_empty() {
        return;
    }
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for pos in &particles.positions {
        min_x = min_x.min(pos[0]);
        max_x = max_x.max(pos[0]);
        min_y = min_y.min(pos[1]);
        max_y = max_y.max(pos[1]);
    }
    let span_x = (max_x - min_x).max(1.0e-6);
    let span_y = (max_y - min_y).max(1.0e-6);
    for (idx, pos) in particles.positions.iter().enumerate() {
        let x = (((pos[0] - min_x) / span_x) * (fb_width.saturating_sub(1)) as f32) as usize;
        let y = (((pos[1] - min_y) / span_y) * (fb_height.saturating_sub(1)) as f32) as usize;
        let vel_mag = particles
            .velocities
            .get(idx)
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .unwrap_or(0.0);
        let t = (vel_mag / (1.0 + vel_mag)).clamp(0.0, 1.0);
        let color = lookup_table(ColorMap::Inferno)[(t * 255.0) as usize];
        framebuffer[y * fb_width + x] = color;
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

fn lookup_table(color_map: ColorMap) -> [u32; 256] {
    let mut lut = [0u32; 256];
    for (i, item) in lut.iter_mut().enumerate() {
        let t = i as f32 / 255.0;
        let (r, g, b) = match color_map {
            ColorMap::Viridis => (
                ((-1.27 * t + 2.47) * t * t * 255.0).clamp(0.0, 255.0) as u32,
                ((0.83 * t - 1.72) * t * t * 255.0 + 127.0 * t).clamp(0.0, 255.0)
                    as u32,
                ((4.28 * (1.0 - t) - 1.0) * (1.0 - t) * 255.0).clamp(0.0, 255.0)
                    as u32,
            ),
            ColorMap::Inferno => (
                ((2.74 * t - 1.78) * t * 255.0 + 10.0).clamp(0.0, 255.0) as u32,
                ((-3.0 * (t - 0.65).powi(2) + 0.78) * 255.0).clamp(0.0, 255.0)
                    as u32,
                ((1.97 * (1.0 - t) - 0.19) * (1.0 - t) * 255.0).clamp(0.0, 255.0)
                    as u32,
            ),
            ColorMap::Turbo => {
                // Neon/Cyberpunk palette approximation
                // Dark -> Purple -> Blue -> Cyan -> Green -> White
                if t < 0.2 {
                    let f = t / 0.2;
                    ( (f * 100.0) as u32, 0, (f * 200.0) as u32 )
                } else if t < 0.4 {
                    let f = (t - 0.2) / 0.2;
                    ( (100.0 - f * 100.0) as u32, (f * 128.0) as u32, 200 + (f * 55.0) as u32 )
                } else if t < 0.6 {
                    let f = (t - 0.4) / 0.2;
                    ( 0, 128 + (f * 127.0) as u32, 255 )
                } else if t < 0.8 {
                    let f = (t - 0.6) / 0.2;
                    ( (f * 128.0) as u32, 255, 255 - (f * 200.0) as u32 )
                } else {
                    let f = (t - 0.8) / 0.2;
                    ( 128 + (f * 127.0) as u32, 255, 55 + (f * 200.0) as u32 )
                }
            }
        };
        *item = 0xFF00_0000 | (r << 16) | (g << 8) | b;
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particle_raster_handles_empty_input() {
        let particles = ParticleFrame {
            positions: Vec::new(),
            velocities: Vec::new(),
        };
        let mut framebuffer = vec![0xFFFF_FFFF; 16];
        render_particles_autofit_xy_to_argb(&particles, &mut framebuffer, 4, 4);
        assert!(framebuffer.iter().all(|&px| px == 0xFF00_0000));
    }
}
