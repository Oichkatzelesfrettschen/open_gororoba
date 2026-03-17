//! Frame transport helpers for `lbm-live-viewer`.
//!
//! These helpers translate backend-neutral frame packets into the concrete ARGB
//! framebuffer expected by `minifb`. They intentionally know nothing about CUDA
//! or Vulkan runtime details.

use crate::camera_input::ViewerInteractionState;
use anyhow::Result;
use gororoba_view_core::ViewerFramePacket;
use gororoba_view_raster::{
    ColorMap, SliceRasterSpec, blit_rgba_to_argb,
    render_particles_autofit_xy_to_argb, render_scalar_volume_slice_to_argb,
};

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
            render_scalar_volume_slice_to_argb(
                framebuffer,
                (fb_width, fb_height),
                &volume.values,
                volume.grid,
                SliceRasterSpec {
                    axis: state.slice_axis,
                    slice_index: state.slice_index,
                    color_map: ColorMap::Viridis,
                },
            );
            Ok(())
        }
        ViewerFramePacket::SliceRgba8(slice) => {
            blit_rgba_to_argb(slice, framebuffer, fb_width, fb_height);
            Ok(())
        }
        ViewerFramePacket::Particles(_) => {
            if let ViewerFramePacket::Particles(particles) = packet {
                render_particles_autofit_xy_to_argb(
                    particles,
                    framebuffer,
                    fb_width,
                    fb_height,
                );
                Ok(())
            } else {
                unreachable!()
            }
        }
    }
}
