//! Vulkan swapchain presenter for interactive display.
//!
//! Bridges the GororobaEngine's compute-rendered storage image to a
//! winit window surface via swapchain presentation.
//!
//! Architecture:
//! 1. GororobaEngine renders to `render_image` (RGBA8_UNORM storage image)
//! 2. SwapchainPresenter blits `render_image` to the acquired swapchain image
//! 3. Swapchain presents to the window surface
//!
//! The swapchain uses FIFO present mode (vsync) for smooth display.
//! The blit handles format conversion and scaling automatically.

use ash::{Device, vk};
use std::sync::Arc;

/// Swapchain configuration.
#[derive(Debug, Clone)]
pub struct SwapchainConfig {
    /// Window width in pixels.
    pub width: u32,
    /// Window height in pixels.
    pub height: u32,
    /// Preferred present mode (FIFO = vsync, MAILBOX = triple buffer).
    pub vsync: bool,
}

impl Default for SwapchainConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 960,
            vsync: true,
        }
    }
}

/// Manages a Vulkan swapchain for presenting rendered frames to a window.
pub struct SwapchainPresenter {
    device: Arc<Device>,
    config: SwapchainConfig,
    /// Command pool for blit commands.
    cmd_pool: vk::CommandPool,
    /// Pre-allocated command buffer for blitting.
    cmd_buf: vk::CommandBuffer,
    /// Fence for frame synchronization.
    fence: vk::Fence,
    /// Queue for submission.
    queue: vk::Queue,
}

impl SwapchainPresenter {
    /// Create a new swapchain presenter.
    ///
    /// NOTE: Full swapchain creation requires VK_KHR_surface + VK_KHR_swapchain
    /// device extensions, which are NOT enabled in the current VulkanContext.
    /// This struct provides the command buffer management for blitting;
    /// the actual swapchain creation requires the windowed VulkanContext variant.
    pub fn new(
        device: Arc<Device>,
        queue: vk::Queue,
        queue_family: u32,
        config: SwapchainConfig,
    ) -> Result<Self, vk::Result> {
        let pool_info = vk::CommandPoolCreateInfo {
            queue_family_index: queue_family,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None) }?;

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let cmd_bufs = unsafe { device.allocate_command_buffers(&alloc_info) }?;

        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo {
                    flags: vk::FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
                None,
            )
        }?;

        Ok(Self {
            device,
            config,
            cmd_pool,
            cmd_buf: cmd_bufs[0],
            fence,
            queue,
        })
    }

    /// Record a blit command from the source image to the destination image.
    ///
    /// Transitions source to TRANSFER_SRC, destination to TRANSFER_DST,
    /// blits with linear filtering, then transitions destination to PRESENT_SRC.
    pub fn record_blit(
        &self,
        src_image: vk::Image,
        dst_image: vk::Image,
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> Result<vk::CommandBuffer, vk::Result> {
        let cmd = self.cmd_buf;

        unsafe {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;

            // Transition dst to TRANSFER_DST
            let barrier_dst = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image: dst_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_dst],
            );

            // Transition src to TRANSFER_SRC
            let barrier_src = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image: src_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_src],
            );

            // Blit with linear filtering (handles scaling)
            let blit_region = vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src_width as i32,
                        y: src_height as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: dst_width as i32,
                        y: dst_height as i32,
                        z: 1,
                    },
                ],
            };
            self.device.cmd_blit_image(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit_region],
                vk::Filter::LINEAR,
            );

            // Transition dst to PRESENT_SRC for swapchain presentation
            let barrier_present = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                image: dst_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_present],
            );

            // Transition src back to GENERAL for next compute dispatch
            let barrier_src_back = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                new_layout: vk::ImageLayout::GENERAL,
                image: src_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_src_back],
            );

            self.device.end_command_buffer(cmd)?;
        }

        Ok(cmd)
    }

    /// Submit the blit command buffer and wait for completion.
    pub fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<(), vk::Result> {
        unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.fence])?;

            let submit = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            };
            self.device
                .queue_submit(self.queue, &[submit], self.fence)?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        Ok(())
    }

    /// Get the presenter configuration.
    pub fn config(&self) -> &SwapchainConfig {
        &self.config
    }
}

impl Drop for SwapchainPresenter {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swapchain_config_default() {
        let cfg = SwapchainConfig::default();
        assert_eq!(cfg.width, 1280);
        assert_eq!(cfg.height, 960);
        assert!(cfg.vsync);
    }
}
