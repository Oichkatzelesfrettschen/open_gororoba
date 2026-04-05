use crate::{
    VulkanContext,
    compute::{BufferSet, VulkanEngineError, compile_wgsl},
};
use ash::{Device, vk};
use gpu_allocator::{MemoryLocation, vulkan::*};
use std::{
    ffi::CString,
    sync::{Arc, Mutex},
};

type Result<T> = std::result::Result<T, VulkanEngineError>;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AlignmentParams {
    n_vectors: u32,
    n_orientations: u32,
}

/// Vulkan-accelerated Box-Kite alignment engine.
pub struct VulkanBoxKiteAlignmentEngine {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    uniform_buffer: Option<BufferSet>,
}

impl VulkanBoxKiteAlignmentEngine {
    pub fn try_new(ctx: &VulkanContext) -> Result<Self> {
        let device = ctx.device.clone();

        let shader_src = include_str!("shaders/box_kite_alignment.wgsl");
        let code = compile_wgsl(shader_src)?;
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: code.len() * 4,
                    p_code: code.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;

        let dsl = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    binding_count: 6,
                    p_bindings: [
                        // 0: vectors (read)
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        // 1: orientations (read)
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        // 2: bk_indices (read)
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        // 3: out_max_alignment (write)
                        vk::DescriptorSetLayoutBinding {
                            binding: 3,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        // 4: out_best_orient (write)
                        vk::DescriptorSetLayoutBinding {
                            binding: 4,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        // 5: params (uniform)
                        vk::DescriptorSetLayoutBinding {
                            binding: 5,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                    ]
                    .as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    set_layout_count: 1,
                    p_set_layouts: &dsl,
                    ..Default::default()
                },
                None,
            )
        }?;

        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo {
                    stage: vk::PipelineShaderStageCreateInfo {
                        stage: vk::ShaderStageFlags::COMPUTE,
                        module,
                        p_name: CString::new("main").unwrap().as_ptr(),
                        ..Default::default()
                    },
                    layout,
                    ..Default::default()
                }],
                None,
            )
        }
        .map_err(|e| VulkanEngineError::Vulkan(e.1))?[0];

        unsafe {
            device.destroy_shader_module(module, None);
        }

        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: 2,
                    p_pool_sizes: [
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 5,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                        },
                    ]
                    .as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;

        let set = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &dsl,
                ..Default::default()
            })
        }?[0];

        let mut allocator = ctx
            .allocator
            .lock()
            .map_err(|_| VulkanEngineError::LockError)?;
        let uniform = Self::create_buf(
            &device,
            &mut allocator,
            256,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "align_u",
            MemoryLocation::CpuToGpu,
        )?;

        Ok(Self {
            device,
            allocator: ctx.allocator.clone(),
            pipeline,
            layout,
            descriptor_layout: dsl,
            descriptor_pool: pool,
            descriptor_set: set,
            uniform_buffer: Some(uniform),
        })
    }

    fn create_buf(
        device: &Device,
        allocator: &mut Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        name: &str,
        loc: MemoryLocation,
    ) -> Result<BufferSet> {
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo {
                    size,
                    usage,
                    ..Default::default()
                },
                None,
            )
        }?;
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = allocator.allocate(&AllocationCreateDesc {
            name,
            requirements: reqs,
            location: loc,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;
        Ok(BufferSet { buffer, allocation })
    }

    pub fn run_alignment_scan(
        &mut self,
        ctx: &VulkanContext,
        vectors: &[f64],
        orientations: &[u32],
        bk_indices: &[u32],
    ) -> Result<(Vec<f64>, Vec<u32>)> {
        let n_vectors = (vectors.len() / 16) as u32;
        let n_orientations = (orientations.len() / 16) as u32;

        let mut allocator = self
            .allocator
            .lock()
            .map_err(|_| VulkanEngineError::LockError)?;

        // 1. Create and upload buffers
        let v_buf = Self::create_buf(
            &self.device,
            &mut allocator,
            (vectors.len() * 8) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "v_buf",
            MemoryLocation::CpuToGpu,
        )?;
        let o_buf = Self::create_buf(
            &self.device,
            &mut allocator,
            (orientations.len() * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "o_buf",
            MemoryLocation::CpuToGpu,
        )?;
        let bk_buf = Self::create_buf(
            &self.device,
            &mut allocator,
            (bk_indices.len() * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "bk_buf",
            MemoryLocation::CpuToGpu,
        )?;
        let out_m_buf = Self::create_buf(
            &self.device,
            &mut allocator,
            (n_vectors * 8) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "out_m",
            MemoryLocation::GpuToCpu,
        )?;
        let out_b_buf = Self::create_buf(
            &self.device,
            &mut allocator,
            (n_vectors * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "out_b",
            MemoryLocation::GpuToCpu,
        )?;

        // Upload data
        unsafe {
            let p = v_buf.allocation.mapped_ptr().unwrap().as_ptr() as *mut f64;
            std::ptr::copy_nonoverlapping(vectors.as_ptr(), p, vectors.len());
            let p = o_buf.allocation.mapped_ptr().unwrap().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(orientations.as_ptr(), p, orientations.len());
            let p = bk_buf.allocation.mapped_ptr().unwrap().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(bk_indices.as_ptr(), p, bk_indices.len());

            let u_buf = self.uniform_buffer.as_ref().unwrap();
            let p = u_buf.allocation.mapped_ptr().unwrap().as_ptr() as *mut AlignmentParams;
            std::ptr::write(
                p,
                AlignmentParams {
                    n_vectors,
                    n_orientations,
                },
            );
        }

        // 2. Update descriptor set
        let infos = [
            vk::DescriptorBufferInfo {
                buffer: v_buf.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: o_buf.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: bk_buf.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: out_m_buf.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: out_b_buf.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: self.uniform_buffer.as_ref().unwrap().buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| vk::WriteDescriptorSet {
                dst_set: self.descriptor_set,
                dst_binding: i as u32,
                descriptor_count: 1,
                descriptor_type: if i == 5 {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                },
                p_buffer_info: info,
                ..Default::default()
            })
            .collect();
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }

        // 3. Execute compute
        unsafe {
            let pool = self.device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags: vk::CommandPoolCreateFlags::TRANSIENT,
                    ..Default::default()
                },
                None,
            )?;
            let cmd = self
                .device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                    command_pool: pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                    ..Default::default()
                })?[0];

            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            self.device.cmd_dispatch(cmd, n_vectors.div_ceil(256), 1, 1);
            self.device.end_command_buffer(cmd)?;

            self.device.queue_submit(
                ctx.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd,
                    ..Default::default()
                }],
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(ctx.queue)?;
            self.device.destroy_command_pool(pool, None);
        }

        // 4. Read back
        let out_m = unsafe {
            let p = out_m_buf.allocation.mapped_ptr().unwrap().as_ptr() as *const f64;
            std::slice::from_raw_parts(p, n_vectors as usize).to_vec()
        };
        let out_b = unsafe {
            let p = out_b_buf.allocation.mapped_ptr().unwrap().as_ptr() as *const u32;
            std::slice::from_raw_parts(p, n_vectors as usize).to_vec()
        };

        // Cleanup temp buffers
        allocator.free(v_buf.allocation)?;
        allocator.free(o_buf.allocation)?;
        allocator.free(bk_buf.allocation)?;
        allocator.free(out_m_buf.allocation)?;
        allocator.free(out_b_buf.allocation)?;
        unsafe {
            self.device.destroy_buffer(v_buf.buffer, None);
            self.device.destroy_buffer(o_buf.buffer, None);
            self.device.destroy_buffer(bk_buf.buffer, None);
            self.device.destroy_buffer(out_m_buf.buffer, None);
            self.device.destroy_buffer(out_b_buf.buffer, None);
        }

        Ok((out_m, out_b))
    }
}

impl Drop for VulkanBoxKiteAlignmentEngine {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            if let Some(ub) = self.uniform_buffer.take() {
                self.device.destroy_buffer(ub.buffer, None);
                if let Ok(mut allocator) = self.allocator.lock() {
                    let _ = allocator.free(ub.allocation);
                }
            }
        }
    }
}
