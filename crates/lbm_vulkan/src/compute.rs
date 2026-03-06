use crate::{Precision, VulkanContext};
use ash::{Device, vk};
use gpu_allocator::{MemoryLocation, vulkan::*};
use std::{
    ffi::CString,
    sync::{Arc, Mutex},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VulkanEngineError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
    #[error("GPU allocator error: {0}")]
    Allocator(#[from] gpu_allocator::AllocationError),
    #[error("Failed to lock allocator mutex")]
    LockError,
    #[error("Memory mapping failed: {0}")]
    MappingError(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Shader compilation error: {0}")]
    ShaderError(String),
    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),
}

type Result<T> = std::result::Result<T, VulkanEngineError>;

/// Unified Engine for Sedenion-LBM Simulations
pub struct GororobaEngine {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    lbm_pipeline: ComputePipeline,
    zd_pipeline: ComputePipeline,
    render_pipeline: ComputePipeline,
    pathion_sink_pipeline: ComputePipeline,
    f_buffers: [BufferSet; 2],
    rho_buffer: BufferSet,
    u_buffer: BufferSet,
    tau_buffer: BufferSet,
    force_buffer: BufferSet,
    entropy_buffer: BufferSet,
    accum_buffer: BufferSet,
    render_image: ImageSet,
    grid_dim: (u32, u32, u32),
    screen_dim: (u32, u32),
    step_counter: u64,
    precision: Precision,
}

struct ComputePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    uniform_buffer: BufferSet,
}

pub(crate) struct BufferSet {
    pub(crate) buffer: vk::Buffer,
    pub(crate) allocation: Allocation,
}

struct ImageSet {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Allocation,
    readback: BufferSet,
}

pub(crate) fn compile_wgsl(source: &str) -> Result<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?;
    let mut words = Vec::new();
    let mut writer = naga::back::spv::Writer::new(&naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    })
    .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?;
    writer
        .write(&module, &info, None, &None, &mut words)
        .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?;
    Ok(words)
}

impl GororobaEngine {
    pub fn new(
        ctx: &VulkanContext,
        grid_dim: (u32, u32, u32),
        screen_dim: (u32, u32),
        precision: Precision,
    ) -> Result<Self> {
        let device = ctx.device.clone();
        let n_cells = (grid_dim.0 * grid_dim.1 * grid_dim.2) as u64;
        let f_elem_bytes = precision.bytes_per_element() as u64;
        let mut allocator = ctx
            .allocator
            .lock()
            .map_err(|_| VulkanEngineError::LockError)?;

        let f_a = Self::create_buf_internal(
            &device,
            &mut allocator,
            n_cells * 19 * f_elem_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "f_a",
            MemoryLocation::GpuOnly,
        )?;
        let f_b = Self::create_buf_internal(
            &device,
            &mut allocator,
            n_cells * 19 * f_elem_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "f_b",
            MemoryLocation::GpuOnly,
        )?;
        let rho = Self::create_buf_internal(
            &device,
            &mut allocator,
            n_cells * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "rho",
            MemoryLocation::GpuToCpu,
        )?;
        let u = Self::create_buf_internal(
            &device,
            &mut allocator,
            n_cells * 3 * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "u",
            MemoryLocation::GpuToCpu,
        )?;
        let tau = Self::create_buf_internal(
            &device,
            &mut allocator,
            n_cells * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "tau",
            MemoryLocation::GpuOnly,
        )?;
        let force = Self::create_buf_internal(
            &device,
            &mut allocator,
            n_cells * 3 * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "force",
            MemoryLocation::CpuToGpu,
        )?;
        let entropy = Self::create_buf_internal(
            &device,
            &mut allocator,
            n_cells * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "entropy",
            MemoryLocation::GpuOnly,
        )?;
        let accum_buffer = Self::create_buf_internal(
            &device,
            &mut allocator,
            n_cells * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "accum",
            MemoryLocation::GpuToCpu,
        )?;

        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: screen_dim.0,
                height: screen_dim.1,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            ..Default::default()
        };
        let render_image = unsafe { device.create_image(&image_info, None) }?;
        let img_alloc = allocator.allocate(&AllocationCreateDesc {
            name: "r_img",
            requirements: unsafe { device.get_image_memory_requirements(render_image) },
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_image_memory(render_image, img_alloc.memory(), img_alloc.offset()) }?;
        let render_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo {
                    image: render_image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
        }?;
        let readback = Self::create_buf_internal(
            &device,
            &mut allocator,
            (screen_dim.0 * screen_dim.1 * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            "read",
            MemoryLocation::GpuToCpu,
        )?;
        drop(allocator);

        let lbm_pipeline = Self::create_lbm_pipeline(
            &device, ctx, &f_a, &f_b, &rho, &u, &tau, &force, &entropy, precision,
        )?;
        let zd_pipeline = Self::create_zd_pipeline(&device, ctx, &tau)?;
        let pathion_sink_pipeline = Self::create_pathion_sink_pipeline(&device, ctx, &tau, &force, &accum_buffer)?;
        let render_pipeline =
            Self::create_render_pipeline(&device, ctx, &entropy, &tau, render_view, screen_dim)?;

        Ok(Self {
            device,
            allocator: ctx.allocator.clone(),
            lbm_pipeline,
            zd_pipeline,
            pathion_sink_pipeline,
            render_pipeline,
            f_buffers: [f_a, f_b],
            rho_buffer: rho,
            u_buffer: u,
            tau_buffer: tau,
            force_buffer: force,
            entropy_buffer: entropy,
            accum_buffer,
            render_image: ImageSet {
                image: render_image,
                view: render_view,
                allocation: img_alloc,
                readback,
            },
            grid_dim,
            screen_dim,
            step_counter: 0,
            precision,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_lbm_pipeline(
        device: &Arc<Device>,
        ctx: &VulkanContext,
        f_a: &BufferSet,
        f_b: &BufferSet,
        rho: &BufferSet,
        u: &BufferSet,
        tau: &BufferSet,
        force: &BufferSet,
        entropy: &BufferSet,
        precision: Precision,
    ) -> Result<ComputePipeline> {
        let lbm_src = match precision {
            Precision::FP16 => include_str!("../shaders/lbm_f16.wgsl"),
            _ => include_str!("../shaders/lbm.wgsl"),
        };
        let code = compile_wgsl(lbm_src)
            .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?;
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
                    binding_count: 8,
                    p_bindings: [
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 3,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 4,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 5,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 6,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 7,
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
                        p_name: CString::new("main")
                            .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?
                            .as_ptr(),
                        ..Default::default()
                    },
                    layout,
                    ..Default::default()
                }],
                None,
            )
        }
        .map_err(|e| VulkanEngineError::Vulkan(e.1))?[0];
        let mut allocator = ctx
            .allocator
            .lock()
            .map_err(|_| VulkanEngineError::LockError)?;
        let uniform = Self::create_buf_internal(
            device,
            &mut allocator,
            256,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "l_u",
            MemoryLocation::CpuToGpu,
        )?;
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 2,
                    pool_size_count: 2,
                    p_pool_sizes: [
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 14,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 2,
                        },
                    ]
                    .as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;
        let sets = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 2,
                p_set_layouts: [dsl, dsl].as_ptr(),
                ..Default::default()
            })
        }?;
        for (i, &set) in sets.iter().enumerate() {
            let (fin, fout) = if i == 0 {
                (f_a.buffer, f_b.buffer)
            } else {
                (f_b.buffer, f_a.buffer)
            };
            let infos = [
                vk::DescriptorBufferInfo {
                    buffer: fin,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: fout,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: rho.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: u.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: tau.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: force.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: entropy.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: uniform.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
            ];
            let writes = infos
                .iter()
                .enumerate()
                .map(|(b, info)| vk::WriteDescriptorSet {
                    dst_set: set,
                    dst_binding: b as u32,
                    descriptor_count: 1,
                    descriptor_type: if b == 7 {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    },
                    p_buffer_info: info,
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        unsafe {
            device.destroy_shader_module(module, None);
        }
        Ok(ComputePipeline {
            pipeline,
            layout,
            descriptor_layout: dsl,
            descriptor_pool: pool,
            descriptor_sets: sets,
            uniform_buffer: uniform,
        })
    }

    fn create_zd_pipeline(
        device: &Arc<Device>,
        ctx: &VulkanContext,
        tau: &BufferSet,
    ) -> Result<ComputePipeline> {
        let code = compile_wgsl(include_str!("../shaders/zd_gen.wgsl"))
            .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?;
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
                    binding_count: 2,
                    p_bindings: [
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
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
                        p_name: CString::new("main")
                            .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?
                            .as_ptr(),
                        ..Default::default()
                    },
                    layout,
                    ..Default::default()
                }],
                None,
            )
        }
        .map_err(|e| VulkanEngineError::Vulkan(e.1))?[0];
        let mut allocator = ctx
            .allocator
            .lock()
            .map_err(|_| VulkanEngineError::LockError)?;
        let uniform = Self::create_buf_internal(
            device,
            &mut allocator,
            256,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "z_u",
            MemoryLocation::CpuToGpu,
        )?;
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: 2,
                    p_pool_sizes: [
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
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
        let sets = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &dsl,
                ..Default::default()
            })
        }?;
        unsafe {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet {
                        dst_set: sets[0],
                        dst_binding: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: tau.buffer,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: sets[0],
                        dst_binding: 1,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: uniform.buffer,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                ],
                &[],
            )
        };
        unsafe {
            device.destroy_shader_module(module, None);
        }
        Ok(ComputePipeline {
            pipeline,
            layout,
            descriptor_layout: dsl,
            descriptor_pool: pool,
            descriptor_sets: sets,
            uniform_buffer: uniform,
        })
    }

    fn create_pathion_sink_pipeline(
        device: &Arc<Device>,
        ctx: &VulkanContext,
        tau: &BufferSet,
        force: &BufferSet,
        accum: &BufferSet,
    ) -> Result<ComputePipeline> {
        let code = compile_wgsl(include_str!("../shaders/pathion_sink.wgsl"))
            .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?;
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
                    binding_count: 4,
                    p_bindings: [
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 3,
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
                        p_name: CString::new("main")
                            .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?
                            .as_ptr(),
                        ..Default::default()
                    },
                    layout,
                    ..Default::default()
                }],
                None,
            )
        }
        .map_err(|e| VulkanEngineError::Vulkan(e.1))?[0];
        let mut allocator = ctx
            .allocator
            .lock()
            .map_err(|_| VulkanEngineError::LockError)?;
        let uniform = Self::create_buf_internal(
            device,
            &mut allocator,
            256,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "ps_u",
            MemoryLocation::CpuToGpu,
        )?;
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: 2,
                    p_pool_sizes: [
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 3,
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
        let sets = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &dsl,
                ..Default::default()
            })
        }?;
        unsafe {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet {
                        dst_set: sets[0],
                        dst_binding: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: tau.buffer,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: sets[0],
                        dst_binding: 1,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: force.buffer,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: sets[0],
                        dst_binding: 2,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: accum.buffer,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: sets[0],
                        dst_binding: 3,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: uniform.buffer,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                ],
                &[],
            )
        };
        unsafe {
            device.destroy_shader_module(module, None);
        }
        Ok(ComputePipeline {
            pipeline,
            layout,
            descriptor_layout: dsl,
            descriptor_pool: pool,
            descriptor_sets: sets,
            uniform_buffer: uniform,
        })
    }

    fn create_render_pipeline(
        device: &Arc<Device>,
        ctx: &VulkanContext,
        entropy: &BufferSet,
        tau: &BufferSet,
        view: vk::ImageView,
        _dim: (u32, u32),
    ) -> Result<ComputePipeline> {
        let code = compile_wgsl(include_str!("../shaders/render.wgsl"))
            .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?;
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
                    binding_count: 4,
                    p_bindings: [
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 3,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
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
                        p_name: CString::new("main")
                            .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?
                            .as_ptr(),
                        ..Default::default()
                    },
                    layout,
                    ..Default::default()
                }],
                None,
            )
        }
        .map_err(|e| VulkanEngineError::Vulkan(e.1))?[0];
        let mut allocator = ctx
            .allocator
            .lock()
            .map_err(|_| VulkanEngineError::LockError)?;
        let uniform = Self::create_buf_internal(
            device,
            &mut allocator,
            256,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "r_u",
            MemoryLocation::CpuToGpu,
        )?;
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: 3,
                    p_pool_sizes: [
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 2,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
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
        let sets = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &dsl,
                ..Default::default()
            })
        }?;
        unsafe {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet {
                        dst_set: sets[0],
                        dst_binding: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: entropy.buffer,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: sets[0],
                        dst_binding: 1,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            image_view: view,
                            image_layout: vk::ImageLayout::GENERAL,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: sets[0],
                        dst_binding: 2,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: uniform.buffer,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: sets[0],
                        dst_binding: 3,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: tau.buffer,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                ],
                &[],
            )
        };
        unsafe {
            device.destroy_shader_module(module, None);
        }
        Ok(ComputePipeline {
            pipeline,
            layout,
            descriptor_layout: dsl,
            descriptor_pool: pool,
            descriptor_sets: sets,
            uniform_buffer: uniform,
        })
    }

    fn create_buf_internal(
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

    /// Buffer handle for the density (rho) field. Used by BC engine for
    /// descriptor set binding.
    pub fn rho_buffer_handle(&self) -> vk::Buffer {
        self.rho_buffer.buffer
    }

    /// Buffer handle for the velocity (u) field.
    pub fn u_buffer_handle(&self) -> vk::Buffer {
        self.u_buffer.buffer
    }

    /// Buffer handle for the relaxation time (tau) field.
    pub fn tau_buffer_handle(&self) -> vk::Buffer {
        self.tau_buffer.buffer
    }

    /// Shared reference to the logical device.
    pub fn device_ref(&self) -> &Arc<Device> {
        &self.device
    }

    /// Shared reference to the GPU memory allocator.
    pub fn allocator_ref(&self) -> &Arc<Mutex<Allocator>> {
        &self.allocator
    }

    /// Return the precision mode used for distribution buffers.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Upload raw bytes for the distribution function buffer.
    ///
    /// `f_bytes` must be exactly `n_cells * 19 * precision.bytes_per_element()`
    /// bytes, laid out in SoA order matching the active shader variant.
    pub fn upload_initial_state_bytes(
        &mut self,
        ctx: &VulkanContext,
        f_bytes: &[u8],
        force: &[f32],
    ) -> Result<()> {
        let f_size = f_bytes.len() as u64;
        let mut allocator = self
            .allocator
            .lock()
            .map_err(|_| VulkanEngineError::LockError)?;
        let staging = Self::create_buf_internal(
            &self.device,
            &mut allocator,
            f_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            "stg",
            MemoryLocation::CpuToGpu,
        )?;
        unsafe {
            let mapped_ptr = staging.allocation.mapped_ptr().ok_or_else(|| {
                VulkanEngineError::MappingError("Failed to map staging buffer".to_string())
            })?;
            std::ptr::copy_nonoverlapping(
                f_bytes.as_ptr(),
                mapped_ptr.as_ptr() as *mut u8,
                f_bytes.len(),
            );
        }
        unsafe {
            let mapped_ptr = self.force_buffer.allocation.mapped_ptr().ok_or_else(|| {
                VulkanEngineError::MappingError("Failed to map force buffer".to_string())
            })?;
            let f_ptr = mapped_ptr.as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(force.as_ptr(), f_ptr, force.len());
        }
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
            self.device.cmd_copy_buffer(
                cmd,
                staging.buffer,
                self.f_buffers[0].buffer,
                &[vk::BufferCopy {
                    size: f_size,
                    ..Default::default()
                }],
            );
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
            self.device.destroy_buffer(staging.buffer, None);
        }
        allocator.free(staging.allocation)?;
        Ok(())
    }

    /// Upload f32 distribution function data. Convenience wrapper around
    /// [`upload_initial_state_bytes`](Self::upload_initial_state_bytes) for FP32 mode.
    pub fn upload_initial_state(
        &mut self,
        ctx: &VulkanContext,
        f_init: &[f32],
        force: &[f32],
    ) -> Result<()> {
        let f_size = (f_init.len() * 4) as u64;
        let mut allocator = self
            .allocator
            .lock()
            .map_err(|_| VulkanEngineError::LockError)?;
        let staging = Self::create_buf_internal(
            &self.device,
            &mut allocator,
            f_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            "stg",
            MemoryLocation::CpuToGpu,
        )?;
        unsafe {
            let mapped_ptr = staging.allocation.mapped_ptr().ok_or_else(|| {
                VulkanEngineError::MappingError("Failed to map staging buffer".to_string())
            })?;
            std::ptr::copy_nonoverlapping(
                f_init.as_ptr(),
                mapped_ptr.as_ptr() as *mut f32,
                f_init.len(),
            );
        }
        unsafe {
            let mapped_ptr = self.force_buffer.allocation.mapped_ptr().ok_or_else(|| {
                VulkanEngineError::MappingError("Failed to map force buffer".to_string())
            })?;
            let f_ptr = mapped_ptr.as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(force.as_ptr(), f_ptr, force.len());
        }
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
            self.device.cmd_copy_buffer(
                cmd,
                staging.buffer,
                self.f_buffers[0].buffer,
                &[vk::BufferCopy {
                    size: f_size,
                    ..Default::default()
                }],
            );
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
            self.device.destroy_buffer(staging.buffer, None);
        }
        allocator.free(staging.allocation)?;
        Ok(())
    }

    /// Read back the full rho density field from GPU memory.
    ///
    /// The rho buffer is allocated as `GpuToCpu`, so `mapped_ptr()` is valid
    /// after the compute shader writes to it. Caller MUST ensure the GPU has
    /// finished writing (e.g. via `queue_wait_idle`) before calling this.
    pub fn read_rho_field(&self) -> Result<Vec<f32>> {
        let ptr = self
            .rho_buffer
            .allocation
            .mapped_ptr()
            .ok_or_else(|| {
                VulkanEngineError::MappingError("Failed to map rho buffer for reading".to_string())
            })?
            .as_ptr() as *const f32;
        let n = (self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2) as usize;
        let data = unsafe { std::slice::from_raw_parts(ptr, n) };
        Ok(data.to_vec())
    }

    /// Return the grid dimensions.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        self.grid_dim
    }

    pub fn get_diagnostics(&self) -> Result<(f32, f32)> {
        let ptr = self
            .rho_buffer
            .allocation
            .mapped_ptr()
            .ok_or_else(|| {
                VulkanEngineError::MappingError(
                    "Failed to map rho buffer for diagnostics".to_string(),
                )
            })?
            .as_ptr() as *const f32;
        let n = (self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2) as usize;
        let data = unsafe { std::slice::from_raw_parts(ptr, n) };
        let total_mass: f32 = data.iter().sum();
        let max_rho = data.iter().cloned().fold(0.0, f32::max);
        Ok((total_mass, max_rho))
    }

    pub fn step(&mut self, cmd: vk::CommandBuffer, frame: u32) -> Result<()> {
        unsafe {
            let zd_pc = ZdGenConstants {
                nx: self.grid_dim.0,
                ny: self.grid_dim.1,
                nz: self.grid_dim.2,
                tau_base: 0.55,
                tau_amp: 0.2,
                lambda: 5.0,
                time: frame as f32,
            };
            let mapped_ptr = self
                .zd_pipeline
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .ok_or_else(|| {
                    VulkanEngineError::MappingError("Failed to map ZD uniform buffer".to_string())
                })?;
            std::ptr::write(mapped_ptr.as_ptr() as *mut ZdGenConstants, zd_pc);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.zd_pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.zd_pipeline.layout,
                0,
                &self.zd_pipeline.descriptor_sets,
                &[],
            );
            self.device.cmd_dispatch(
                cmd,
                self.grid_dim.0.div_ceil(8),
                self.grid_dim.1.div_ceil(8),
                self.grid_dim.2.div_ceil(8),
            );

            let set_idx = (self.step_counter % 2) as usize;
            self.step_counter += 1;
            let lbm_pc = LbmConstants {
                nx: self.grid_dim.0,
                ny: self.grid_dim.1,
                nz: self.grid_dim.2,
                gx: 0.0,
                gy: -0.0001,
                gz: 0.0,
            };
            let mapped_ptr = self
                .lbm_pipeline
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .ok_or_else(|| {
                    VulkanEngineError::MappingError("Failed to map LBM uniform buffer".to_string())
                })?;
            std::ptr::write(mapped_ptr.as_ptr() as *mut LbmConstants, lbm_pc);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.lbm_pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.lbm_pipeline.layout,
                0,
                &[self.lbm_pipeline.descriptor_sets[set_idx]],
                &[],
            );
            self.device.cmd_dispatch(
                cmd,
                self.grid_dim.0.div_ceil(8),
                self.grid_dim.1.div_ceil(8),
                self.grid_dim.2.div_ceil(8),
            );

            let render_pc = RenderConstants {
                nx: self.grid_dim.0,
                ny: self.grid_dim.1,
                nz: self.grid_dim.2,
                width: self.screen_dim.0,
                height: self.screen_dim.1,
                time: frame as f32,
            };
            let mapped_ptr = self
                .render_pipeline
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .ok_or_else(|| {
                    VulkanEngineError::MappingError(
                        "Failed to map render uniform buffer".to_string(),
                    )
                })?;
            std::ptr::write(mapped_ptr.as_ptr() as *mut RenderConstants, render_pc);
            let barrier = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                image: self.render_image.image,
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
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.render_pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.render_pipeline.layout,
                0,
                &self.render_pipeline.descriptor_sets,
                &[],
            );
            self.device.cmd_dispatch(
                cmd,
                self.screen_dim.0.div_ceil(16),
                self.screen_dim.1.div_ceil(16),
                1,
            );
            let barrier2 = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image: self.render_image.image,
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
                &[barrier2],
            );
            self.device.cmd_copy_image_to_buffer(
                cmd,
                self.render_image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.render_image.readback.buffer,
                &[vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        layer_count: 1,
                        ..Default::default()
                    },
                    image_extent: vk::Extent3D {
                        width: self.screen_dim.0,
                        height: self.screen_dim.1,
                        depth: 1,
                    },
                    ..Default::default()
                }],
            );
        }
        Ok(())
    }

    /// Advance the simulation by one step, with an optional Pathion heat sink applied
    /// at the event horizon boundary.
    pub fn step_with_sink(
        &mut self,
        cmd: vk::CommandBuffer,
        frame: u32,
        mass: f32,
        spin: f32,
        coupling: f32,
        damping: f32,
    ) -> Result<()> {
        unsafe {
            // 1. Generate ZD field (Viscosity modulation)
            let zd_pc = ZdGenConstants {
                nx: self.grid_dim.0,
                ny: self.grid_dim.1,
                nz: self.grid_dim.2,
                tau_base: 0.55,
                tau_amp: 0.2,
                lambda: 5.0,
                time: frame as f32,
            };
            let mapped_ptr = self
                .zd_pipeline
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .ok_or_else(|| {
                    VulkanEngineError::MappingError("Failed to map ZD uniform buffer".to_string())
                })?;
            std::ptr::write(mapped_ptr.as_ptr() as *mut ZdGenConstants, zd_pc);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.zd_pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.zd_pipeline.layout,
                0,
                &self.zd_pipeline.descriptor_sets,
                &[],
            );
            self.device.cmd_dispatch(
                cmd,
                self.grid_dim.0.div_ceil(8),
                self.grid_dim.1.div_ceil(8),
                self.grid_dim.2.div_ceil(8),
            );

            // Barrier: ensure tau is written before Pathion sink reads it
            let barrier = vk::BufferMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                buffer: self.tau_buffer.buffer,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );

            // 2. Pathion Sink (Energy damping at horizon)
            let ps_pc = PathionSinkConstants {
                nx: self.grid_dim.0,
                ny: self.grid_dim.1,
                nz: self.grid_dim.2,
                mass,
                spin,
                coupling,
                damping,
                dt: 1.0,
            };
            let mapped_ptr = self
                .pathion_sink_pipeline
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .ok_or_else(|| {
                    VulkanEngineError::MappingError(
                        "Failed to map pathion sink uniform buffer".to_string(),
                    )
                })?;
            std::ptr::write(mapped_ptr.as_ptr() as *mut PathionSinkConstants, ps_pc);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pathion_sink_pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pathion_sink_pipeline.layout,
                0,
                &self.pathion_sink_pipeline.descriptor_sets,
                &[],
            );
            self.device.cmd_dispatch(
                cmd,
                self.grid_dim.0.div_ceil(8),
                self.grid_dim.1.div_ceil(8),
                self.grid_dim.2.div_ceil(8),
            );

            // Barrier: ensure force and accum are written before LBM
            let force_barrier = vk::BufferMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                buffer: self.force_buffer.buffer,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[force_barrier],
                &[],
            );

            // 3. LBM Step
            let set_idx = (self.step_counter % 2) as usize;
            self.step_counter += 1;
            let lbm_pc = LbmConstants {
                nx: self.grid_dim.0,
                ny: self.grid_dim.1,
                nz: self.grid_dim.2,
                gx: 0.0,
                gy: 0.0,
                gz: 0.0, // Force comes from initial data
            };
            let mapped_ptr = self
                .lbm_pipeline
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .ok_or_else(|| {
                    VulkanEngineError::MappingError("Failed to map LBM uniform buffer".to_string())
                })?;
            std::ptr::write(mapped_ptr.as_ptr() as *mut LbmConstants, lbm_pc);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.lbm_pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.lbm_pipeline.layout,
                0,
                &[self.lbm_pipeline.descriptor_sets[set_idx]],
                &[],
            );
            self.device.cmd_dispatch(
                cmd,
                self.grid_dim.0.div_ceil(8),
                self.grid_dim.1.div_ceil(8),
                self.grid_dim.2.div_ceil(8),
            );
        }
        Ok(())
    }

    /// Read back the total accumulated energy from the Pathion Sink.
    pub fn read_accumulated_energy(&self) -> Result<f64> {
        let ptr = self
            .accum_buffer
            .allocation
            .mapped_ptr()
            .ok_or_else(|| {
                VulkanEngineError::MappingError("Failed to map accum buffer for reading".to_string())
            })?
            .as_ptr() as *const f32;
        let n = (self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2) as usize;
        let data = unsafe { std::slice::from_raw_parts(ptr, n) };
        let sum: f64 = data.iter().map(|&v| v as f64).sum();
        Ok(sum)
    }

    /// Advance the simulation by one step with caller-specified ZD parameters.
    ///
    /// Unlike `step()`, which uses hardcoded tau_base=0.55, tau_amp=0.2, lambda=5.0,
    /// this method accepts arbitrary ZD parameters for sweep experiments.
    pub fn step_with_params(
        &mut self,
        cmd: vk::CommandBuffer,
        frame: u32,
        tau_base: f32,
        tau_amp: f32,
        lambda: f32,
    ) -> Result<()> {
        unsafe {
            let zd_pc = ZdGenConstants {
                nx: self.grid_dim.0,
                ny: self.grid_dim.1,
                nz: self.grid_dim.2,
                tau_base,
                tau_amp,
                lambda,
                time: frame as f32,
            };
            let mapped_ptr = self
                .zd_pipeline
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .ok_or_else(|| {
                    VulkanEngineError::MappingError("Failed to map ZD uniform buffer".to_string())
                })?;
            std::ptr::write(mapped_ptr.as_ptr() as *mut ZdGenConstants, zd_pc);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.zd_pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.zd_pipeline.layout,
                0,
                &self.zd_pipeline.descriptor_sets,
                &[],
            );
            self.device.cmd_dispatch(
                cmd,
                self.grid_dim.0.div_ceil(8),
                self.grid_dim.1.div_ceil(8),
                self.grid_dim.2.div_ceil(8),
            );

            let set_idx = (self.step_counter % 2) as usize;
            self.step_counter += 1;
            let lbm_pc = LbmConstants {
                nx: self.grid_dim.0,
                ny: self.grid_dim.1,
                nz: self.grid_dim.2,
                gx: 0.0,
                gy: -0.0001,
                gz: 0.0,
            };
            let mapped_ptr = self
                .lbm_pipeline
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .ok_or_else(|| {
                    VulkanEngineError::MappingError("Failed to map LBM uniform buffer".to_string())
                })?;
            std::ptr::write(mapped_ptr.as_ptr() as *mut LbmConstants, lbm_pc);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.lbm_pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.lbm_pipeline.layout,
                0,
                &[self.lbm_pipeline.descriptor_sets[set_idx]],
                &[],
            );
            self.device.cmd_dispatch(
                cmd,
                self.grid_dim.0.div_ceil(8),
                self.grid_dim.1.div_ceil(8),
                self.grid_dim.2.div_ceil(8),
            );

            // No render pass in parameterized step -- caller handles readback
        }
        Ok(())
    }

    pub fn save_frame(&self, path: &str) -> Result<()> {
        let ptr = self
            .render_image
            .readback
            .allocation
            .mapped_ptr()
            .ok_or_else(|| {
                VulkanEngineError::MappingError(
                    "Failed to map readback buffer for save_frame".to_string(),
                )
            })?
            .as_ptr() as *const u8;
        let (w, h) = self.screen_dim;
        let byte_count = (w * h * 4) as usize;
        let mut pixels = vec![0u8; byte_count];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, pixels.as_mut_ptr(), pixels.len());
        }
        image::save_buffer(path, &pixels, w, h, image::ColorType::Rgba8)?;
        Ok(())
    }

    /// Read back the rendered RGBA pixels from GPU memory as a byte vector.
    ///
    /// Returns `screen_dim.0 * screen_dim.1 * 4` bytes in R8G8B8A8_UNORM format.
    /// Caller MUST ensure the GPU has finished rendering (via `queue_wait_idle`)
    /// before calling this.
    pub fn read_render_pixels(&self) -> Result<Vec<u8>> {
        let ptr = self
            .render_image
            .readback
            .allocation
            .mapped_ptr()
            .ok_or_else(|| {
                VulkanEngineError::MappingError(
                    "Failed to map readback buffer for read_render_pixels".to_string(),
                )
            })?
            .as_ptr() as *const u8;
        let (w, h) = self.screen_dim;
        let byte_count = (w * h * 4) as usize;
        let mut pixels = vec![0u8; byte_count];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, pixels.as_mut_ptr(), byte_count);
        }
        Ok(pixels)
    }

    /// Read back the velocity field from GPU memory.
    ///
    /// Returns a flat vector of `3 * nx * ny * nz` f32 values
    /// in SoA layout: `[ux_0..ux_N, uy_0..uy_N, uz_0..uz_N]`.
    /// Caller MUST ensure the GPU has finished computing (via
    /// `queue_wait_idle`) before calling this.
    pub fn read_velocity_field(&self) -> Result<Vec<f32>> {
        let ptr = self
            .u_buffer
            .allocation
            .mapped_ptr()
            .ok_or_else(|| {
                VulkanEngineError::MappingError("Failed to map u buffer for reading".to_string())
            })?
            .as_ptr() as *const f32;
        let n = (self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2 * 3) as usize;
        let data = unsafe { std::slice::from_raw_parts(ptr, n) };
        Ok(data.to_vec())
    }

    /// Return the screen (render target) dimensions.
    pub fn screen_dim(&self) -> (u32, u32) {
        self.screen_dim
    }
}

impl Drop for GororobaEngine {
    // Vulkan allocations must be extracted for gpu_allocator::free(). Wrapping
    // every allocation in Option<Allocation> would change all access sites for
    // no runtime benefit. The zeroed placeholder is immediately dropped.
    #[allow(clippy::mem_replace_with_uninit)]
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let mut allocator = match self.allocator.lock() {
                Ok(a) => a,
                Err(_) => {
                    log::error!("Failed to lock allocator in GororobaEngine::drop");
                    return;
                }
            };

            // Shared destroy logic
            self.device
                .destroy_pipeline(self.lbm_pipeline.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.lbm_pipeline.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.lbm_pipeline.descriptor_layout, None);
            self.device
                .destroy_descriptor_pool(self.lbm_pipeline.descriptor_pool, None);
            self.device
                .destroy_buffer(self.lbm_pipeline.uniform_buffer.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.lbm_pipeline.uniform_buffer.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free lbm uniform buffer: {e}");
            }

            self.device
                .destroy_pipeline(self.zd_pipeline.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.zd_pipeline.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.zd_pipeline.descriptor_layout, None);
            self.device
                .destroy_descriptor_pool(self.zd_pipeline.descriptor_pool, None);
            self.device
                .destroy_buffer(self.zd_pipeline.uniform_buffer.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.zd_pipeline.uniform_buffer.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free zd uniform buffer: {e}");
            }

            self.device
                .destroy_pipeline(self.pathion_sink_pipeline.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pathion_sink_pipeline.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.pathion_sink_pipeline.descriptor_layout, None);
            self.device
                .destroy_descriptor_pool(self.pathion_sink_pipeline.descriptor_pool, None);
            self.device
                .destroy_buffer(self.pathion_sink_pipeline.uniform_buffer.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.pathion_sink_pipeline.uniform_buffer.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free pathion_sink uniform buffer: {e}");
            }

            self.device
                .destroy_pipeline(self.render_pipeline.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.render_pipeline.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.render_pipeline.descriptor_layout, None);
            self.device
                .destroy_descriptor_pool(self.render_pipeline.descriptor_pool, None);
            self.device
                .destroy_buffer(self.render_pipeline.uniform_buffer.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.render_pipeline.uniform_buffer.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free render uniform buffer: {e}");
            }

            self.device.destroy_image_view(self.render_image.view, None);
            self.device.destroy_image(self.render_image.image, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.render_image.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free render image: {e}");
            }
            self.device
                .destroy_buffer(self.render_image.readback.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.render_image.readback.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free readback buffer: {e}");
            }

            self.device.destroy_buffer(self.f_buffers[0].buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.f_buffers[0].allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free f_buffers[0]: {e}");
            }
            self.device.destroy_buffer(self.f_buffers[1].buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.f_buffers[1].allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free f_buffers[1]: {e}");
            }
            self.device.destroy_buffer(self.rho_buffer.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.rho_buffer.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free rho buffer: {e}");
            }
            self.device.destroy_buffer(self.u_buffer.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.u_buffer.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free u buffer: {e}");
            }
            self.device.destroy_buffer(self.tau_buffer.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.tau_buffer.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free tau buffer: {e}");
            }
            self.device.destroy_buffer(self.force_buffer.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.force_buffer.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free force buffer: {e}");
            }
            self.device.destroy_buffer(self.entropy_buffer.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.entropy_buffer.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free entropy buffer: {e}");
            }
            self.device.destroy_buffer(self.accum_buffer.buffer, None);
            if let Err(e) = allocator.free(std::mem::replace(
                &mut self.accum_buffer.allocation,
                std::mem::zeroed(),
            )) {
                log::error!("Failed to free accum buffer: {e}");
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct LbmConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    gx: f32,
    gy: f32,
    gz: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct RenderConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    width: u32,
    height: u32,
    time: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ZdGenConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    tau_base: f32,
    tau_amp: f32,
    lambda: f32,
    time: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PathionSinkConstants {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub mass: f32,
    pub spin: f32,
    pub coupling: f32,
    pub damping: f32,
    pub dt: f32,
}
