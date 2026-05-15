//! Dispatch + fence-gated wait helper.
//!
//! Consolidates the begin_command_buffer + cmd_bind_pipeline +
//! cmd_bind_descriptor_sets + cmd_dispatch + end_command_buffer +
//! queue_submit + wait_for_fences pattern from 5+ sites in
//! lbm_vulkan (besag_clifford_vulkan.rs:1005-1059 / 1141-1186 / 1236-1300
//! / 1326-1351 / 1421-1446) and cd_kernel turboquant (quantizer.rs:379-450).

use std::sync::Arc;

use ash::vk;

use crate::{
    device::Device,
    error::{Result, VulkanError},
    pipeline::ComputePipeline,
};

/// Scoped dispatch: allocates command pool + buffer + fence on construction,
/// records and submits on `dispatch`, waits for completion before returning.
///
/// The fence + command buffer + pool are reusable across dispatches: call
/// `reset` between submissions.
pub struct DispatchScope {
    device: Arc<Device>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl DispatchScope {
    /// Create a new scope bound to the device's primary queue.
    pub fn new(device: &Device) -> Result<Self> {
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family_index())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: device outlives the returned pool via Arc<Device>.
        let command_pool = unsafe { device.raw().create_command_pool(&pool_ci, None) }?;

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: pool was just created; alloc returns a single buffer.
        let command_buffers = unsafe { device.raw().allocate_command_buffers(&alloc_ci) }?;
        let command_buffer = command_buffers
            .first()
            .copied()
            .ok_or(VulkanError::UnsupportedFeature("alloc returned empty"))?;

        let fence_ci = vk::FenceCreateInfo::default();
        // SAFETY: fence has no dependencies; new + signaled-state defaults.
        let fence = unsafe { device.raw().create_fence(&fence_ci, None) }?;

        Ok(Self {
            device: Arc::new(device.clone()),
            queue: device.queue(),
            command_pool,
            command_buffer,
            fence,
        })
    }

    /// Record + submit a 1D / 2D / 3D compute dispatch and wait for it
    /// to complete. Caller binds buffers via `descriptor_set` before
    /// calling.
    ///
    /// `group_count_x` * `group_count_y` * `group_count_z` is the total
    /// workgroup invocations (Vulkan's `vkCmdDispatch` arguments).
    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        descriptor_set: vk::DescriptorSet,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
        timeout_ns: u64,
    ) -> Result<()> {
        let cb = self.command_buffer;
        let dev = self.device.raw();
        let begin_ci = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: cb is owned by this scope; begin -> bind -> dispatch ->
        // end is the standard Vulkan compute sequence with no pointer
        // lifetime concerns beyond the local descriptor_set value.
        unsafe {
            dev.begin_command_buffer(cb, &begin_ci)?;
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline.raw());
            dev.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout(),
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            dev.cmd_dispatch(cb, group_count_x, group_count_y, group_count_z);
            dev.end_command_buffer(cb)?;
        }

        let cbs = [cb];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cbs);
        // SAFETY: cbs slice borrowed for the duration of submit; fence is
        // ours.
        unsafe {
            dev.queue_submit(self.queue, std::slice::from_ref(&submit_info), self.fence)?;
        }
        // SAFETY: fence is signaled by the submit; wait until it fires.
        // Map VK_TIMEOUT (vk::Result::TIMEOUT) to our typed Timeout variant
        // so callers can distinguish a missed deadline from a hard Vulkan
        // error.
        let wait_result =
            unsafe { dev.wait_for_fences(std::slice::from_ref(&self.fence), true, timeout_ns) };
        match wait_result {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => {
                return Err(VulkanError::Timeout {
                    timeout_ns,
                    context: "dispatch",
                });
            }
            Err(other) => return Err(VulkanError::Vk(other)),
        }
        // SAFETY: fence was just observed signaled (or we returned above);
        // safe to reset and reuse for the next dispatch.
        unsafe {
            dev.reset_fences(std::slice::from_ref(&self.fence))?;
            dev.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
        }
        Ok(())
    }
}

impl Drop for DispatchScope {
    fn drop(&mut self) {
        // SAFETY: All handles were created by Self::new above; Arc<Device>
        // keeps the device alive past this Drop.
        unsafe {
            self.device.raw().destroy_fence(self.fence, None);
            self.device
                .raw()
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
