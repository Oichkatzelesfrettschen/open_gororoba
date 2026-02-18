use ash::{vk, Device};
use std::ffi::CString;
use std::sync::Arc;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;
use crate::VulkanContext;
use std::mem::size_of;
use std::sync::Mutex;

#[allow(dead_code)]
pub struct LbmComputePipeline {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    
    // Buffers
    f_in_buffer: Option<(vk::Buffer, Allocation)>,
    f_out_buffer: Option<(vk::Buffer, Allocation)>,
    rho_buffer: Option<(vk::Buffer, Allocation)>,
    u_buffer: Option<(vk::Buffer, Allocation)>,
    pub tau_buffer: Option<(vk::Buffer, Allocation)>,
    pub force_buffer: Option<(vk::Buffer, Allocation)>,
    pub entropy_buffer: Option<(vk::Buffer, Allocation)>,
    
    pub grid_dim: (u32, u32, u32),
}

impl Drop for LbmComputePipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let mut allocator = self.allocator.lock().unwrap();
            let mut free = |opt: &mut Option<(vk::Buffer, Allocation)>| {
                if let Some((b, a)) = opt.take() {
                    self.device.destroy_buffer(b, None);
                    let _ = allocator.free(a);
                }
            };
            free(&mut self.f_in_buffer);
            free(&mut self.f_out_buffer);
            free(&mut self.rho_buffer);
            free(&mut self.u_buffer);
            free(&mut self.tau_buffer);
            free(&mut self.force_buffer);
            free(&mut self.entropy_buffer);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

impl LbmComputePipeline {
    pub fn new(ctx: &VulkanContext, grid_dim: (u32, u32, u32)) -> Result<Self, Box<dyn std::error::Error>> {
        let device = ctx.device.clone();
        let compiler = shaderc::Compiler::new().unwrap();
        let source = include_str!("../shaders/lbm.comp.glsl");
        let binary = compiler.compile_into_spirv(source, shaderc::ShaderKind::Compute, "lbm.comp", "main", None)?;
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: binary.as_binary_u8().len(),
            p_code: binary.as_binary().as_ptr(),
            ..Default::default()
        };
        let shader_module = unsafe { device.create_shader_module(&shader_module_create_info, None) }?;

        let mut bindings = Vec::new();
        for i in 0..7 {
            bindings.push(vk::DescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
                _marker: std::marker::PhantomData,
            });
        }
        let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }?;

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<LbmPushConstants>() as u32,
        };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            ..Default::default()
        };
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

        let entry_point_name = CString::new("main")?;
        let shader_stage_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry_point_name.as_ptr(),
            ..Default::default()
        };
        let pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: shader_stage_info,
            layout: pipeline_layout,
            ..Default::default()
        };
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) }.map_err(|e| e.1)?[0];
        unsafe { device.destroy_shader_module(shader_module, None) };

        let n_cells = (grid_dim.0 * grid_dim.1 * grid_dim.2) as u64;
        let mut allocator = ctx.allocator.lock().unwrap();
        let mut create_buffer = |size, usage, name| -> Result<(vk::Buffer, Allocation), Box<dyn std::error::Error>> {
             let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                size,
                usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let buffer = unsafe { device.create_buffer(&buffer_info, None) }?;
            let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
            let allocation = allocator.allocate(&AllocationCreateDesc {
                name, requirements: reqs, location: MemoryLocation::CpuToGpu, linear: true, allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;
            Ok((buffer, allocation))
        };

        let usage_rw = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let f_in = create_buffer(n_cells * 19 * 4, usage_rw, "f_in")?;
        let f_out = create_buffer(n_cells * 19 * 4, usage_rw, "f_out")?;
        let rho = create_buffer(n_cells * 4, usage_rw, "rho")?;
        let u = create_buffer(n_cells * 3 * 4, usage_rw, "u")?;
        let tau = create_buffer(n_cells * 4, usage_rw, "tau")?;
        let force = create_buffer(n_cells * 3 * 4, usage_rw, "force")?;
        let entropy = create_buffer(n_cells * 4, usage_rw, "entropy")?;

        let pool_size = vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 7 };
        let pool_info = vk::DescriptorPoolCreateInfo { s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO, max_sets: 1, pool_size_count: 1, p_pool_sizes: &pool_size, ..Default::default() };
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;
        let alloc_info = vk::DescriptorSetAllocateInfo { s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO, descriptor_pool, descriptor_set_count: 1, p_set_layouts: &descriptor_set_layout, ..Default::default() };
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }?;

        let buffer_infos = [
            vk::DescriptorBufferInfo { buffer: f_in.0, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: f_out.0, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: rho.0, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: u.0, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: tau.0, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: force.0, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: entropy.0, offset: 0, range: vk::WHOLE_SIZE },
        ];
        let mut writes = Vec::new();
        for (i, buf_info) in buffer_infos.iter().enumerate() {
            writes.push(vk::WriteDescriptorSet { s_type: vk::StructureType::WRITE_DESCRIPTOR_SET, dst_set: descriptor_sets[0], dst_binding: i as u32, descriptor_count: 1, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, p_buffer_info: buf_info, ..Default::default() });
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        Ok(Self { device, allocator: ctx.allocator.clone(), pipeline, pipeline_layout, descriptor_set_layout, descriptor_pool, descriptor_sets, f_in_buffer: Some(f_in), f_out_buffer: Some(f_out), rho_buffer: Some(rho), u_buffer: Some(u), tau_buffer: Some(tau), force_buffer: Some(force), entropy_buffer: Some(entropy), grid_dim })
    }

    pub fn write_inputs(&mut self, tau: &[f32], force: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        if let Some((_, alloc)) = &self.tau_buffer && !tau.is_empty() {
            let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *mut f32;
            unsafe { std::ptr::copy_nonoverlapping(tau.as_ptr(), ptr, tau.len()) };
        }
        if let Some((_, alloc)) = &self.force_buffer && !force.is_empty() {
            let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *mut f32;
            unsafe { std::ptr::copy_nonoverlapping(force.as_ptr(), ptr, force.len()) };
        }
        Ok(())
    }

    pub fn read_entropy(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if let Some((_, alloc)) = &self.entropy_buffer {
            let n = (self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2) as usize;
            let mut out = vec![0.0; n];
            let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *const f32;
            unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n) };
            Ok(out)
        } else { Ok(vec![]) }
    }

    pub fn record_command_buffer(&self, cmd: vk::CommandBuffer) {
        let device = &self.device;
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &self.descriptor_sets, &[]);
            let push_constants = LbmPushConstants { nx: self.grid_dim.0, ny: self.grid_dim.1, nz: self.grid_dim.2, global_tau_scale: 1.0 };
            let constants_bytes = std::slice::from_raw_parts(&push_constants as *const LbmPushConstants as *const u8, size_of::<LbmPushConstants>());
            device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, constants_bytes);
            device.cmd_dispatch(cmd, self.grid_dim.0.div_ceil(8), self.grid_dim.1.div_ceil(8), self.grid_dim.2.div_ceil(8));
            let memory_barrier = vk::MemoryBarrier { s_type: vk::StructureType::MEMORY_BARRIER, src_access_mask: vk::AccessFlags::SHADER_WRITE, dst_access_mask: vk::AccessFlags::SHADER_READ, ..Default::default() };
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[memory_barrier], &[], &[]);
        }
    }
}

pub struct ZdGenPipeline {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    grid_dim: (u32, u32, u32),
}

impl Drop for ZdGenPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ZdGenPushConstants { nx: u32, ny: u32, nz: u32, tau_base: f32, tau_amp: f32, lambda: f32 }

impl ZdGenPipeline {
    pub fn new(ctx: &VulkanContext, grid_dim: (u32, u32, u32), tau_buffer: vk::Buffer) -> Result<Self, Box<dyn std::error::Error>> {
        let device = ctx.device.clone();
        let compiler = shaderc::Compiler::new().unwrap();
        let source = include_str!("../shaders/zd_gen.comp.glsl");
        let binary = compiler.compile_into_spirv(source, shaderc::ShaderKind::Compute, "zd_gen.comp", "main", None)?;
        let shader_module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo { s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO, code_size: binary.as_binary_u8().len(), p_code: binary.as_binary().as_ptr(), ..Default::default() }, None) }?;
        let bindings = [vk::DescriptorSetLayoutBinding { binding: 0, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData }];
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo { s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO, binding_count: 1, p_bindings: bindings.as_ptr(), ..Default::default() }, None) }?;
        let pipeline_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo { s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO, set_layout_count: 1, p_set_layouts: &descriptor_set_layout, push_constant_range_count: 1, p_push_constant_ranges: &vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: size_of::<ZdGenPushConstants>() as u32 }, ..Default::default() }, None) }?;
        let entry = CString::new("main")?;
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo { s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO, stage: vk::PipelineShaderStageCreateInfo { s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO, stage: vk::ShaderStageFlags::COMPUTE, module: shader_module, p_name: entry.as_ptr(), ..Default::default() }, layout: pipeline_layout, ..Default::default() }], None) }.map_err(|e| e.1)?[0];
        unsafe { device.destroy_shader_module(shader_module, None) };
        let descriptor_pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo { s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO, max_sets: 1, pool_size_count: 1, p_pool_sizes: &vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }, ..Default::default() }, None) }?;
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo { s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO, descriptor_pool, descriptor_set_count: 1, p_set_layouts: &descriptor_set_layout, ..Default::default() }) }?;
        unsafe { device.update_descriptor_sets(&[vk::WriteDescriptorSet { s_type: vk::StructureType::WRITE_DESCRIPTOR_SET, dst_set: descriptor_sets[0], dst_binding: 0, descriptor_count: 1, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, p_buffer_info: &vk::DescriptorBufferInfo { buffer: tau_buffer, offset: 0, range: vk::WHOLE_SIZE }, ..Default::default() }], &[]) };
        Ok(Self { device, pipeline, pipeline_layout, descriptor_set_layout, descriptor_pool, descriptor_sets, grid_dim })
    }

    pub fn record_command_buffer(&self, cmd: vk::CommandBuffer) {
        let device = &self.device;
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &self.descriptor_sets, &[]);
            let pc = ZdGenPushConstants { nx: self.grid_dim.0, ny: self.grid_dim.1, nz: self.grid_dim.2, tau_base: 0.6, tau_amp: 0.2, lambda: 5.0 };
            device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(&pc as *const ZdGenPushConstants as *const u8, size_of::<ZdGenPushConstants>()));
            device.cmd_dispatch(cmd, self.grid_dim.0.div_ceil(8), self.grid_dim.1.div_ceil(8), self.grid_dim.2.div_ceil(8));
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[vk::MemoryBarrier { s_type: vk::StructureType::MEMORY_BARRIER, src_access_mask: vk::AccessFlags::SHADER_WRITE, dst_access_mask: vk::AccessFlags::SHADER_READ, ..Default::default() }], &[], &[]);
        }
    }
}

// ==========================================
// Raymarching Renderer Pipeline
// ==========================================

pub struct VulkanRenderer {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    
    // Output image and buffer for readback
    pub render_image: vk::Image,
    pub render_allocation: Option<Allocation>,
    pub render_view: vk::ImageView,
    pub readback_buffer: vk::Buffer,
    pub readback_allocation: Option<Allocation>,
    
    width: u32,
    height: u32,
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let mut allocator = self.allocator.lock().unwrap();
            
            if let Some(alloc) = self.readback_allocation.take() {
                self.device.destroy_buffer(self.readback_buffer, None);
                allocator.free(alloc).unwrap();
            }
            
            self.device.destroy_image_view(self.render_view, None);
            
            if let Some(alloc) = self.render_allocation.take() {
                self.device.destroy_image(self.render_image, None);
                allocator.free(alloc).unwrap();
            }

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RenderPushConstants { nx: u32, ny: u32, nz: u32, width: u32, height: u32, time: f32 }

impl VulkanRenderer {
    pub fn new(ctx: &VulkanContext, width: u32, height: u32, field_buffer: vk::Buffer) -> Result<Self, Box<dyn std::error::Error>> {
        let device = ctx.device.clone();
        let compiler = shaderc::Compiler::new().unwrap();
        let source = include_str!("../shaders/render.comp.glsl");
        let binary = compiler.compile_into_spirv(source, shaderc::ShaderKind::Compute, "render.comp", "main", None)?;
        let shader_module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo { s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO, code_size: binary.as_binary_u8().len(), p_code: binary.as_binary().as_ptr(), ..Default::default() }, None) }?;

        let bindings = [
            vk::DescriptorSetLayoutBinding { binding: 0, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 1, descriptor_type: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
        ];
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo { s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO, binding_count: 2, p_bindings: bindings.as_ptr(), ..Default::default() }, None) }?;
        let pipeline_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo { s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO, set_layout_count: 1, p_set_layouts: &descriptor_set_layout, push_constant_range_count: 1, p_push_constant_ranges: &vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: size_of::<RenderPushConstants>() as u32 }, ..Default::default() }, None) }?;
        let entry = CString::new("main")?;
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo { s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO, stage: vk::PipelineShaderStageCreateInfo { s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO, stage: vk::ShaderStageFlags::COMPUTE, module: shader_module, p_name: entry.as_ptr(), ..Default::default() }, layout: pipeline_layout, ..Default::default() }], None) }.map_err(|e| e.1)?[0];
        unsafe { device.destroy_shader_module(shader_module, None) };

        // Create Image
        let mut allocator = ctx.allocator.lock().unwrap();
        let image_info = vk::ImageCreateInfo { s_type: vk::StructureType::IMAGE_CREATE_INFO, image_type: vk::ImageType::TYPE_2D, format: vk::Format::R8G8B8A8_UNORM, extent: vk::Extent3D { width, height, depth: 1 }, mip_levels: 1, array_layers: 1, samples: vk::SampleCountFlags::TYPE_1, tiling: vk::ImageTiling::OPTIMAL, usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC, sharing_mode: vk::SharingMode::EXCLUSIVE, initial_layout: vk::ImageLayout::UNDEFINED, ..Default::default() };
        let render_image = unsafe { device.create_image(&image_info, None) }?;
        let image_reqs = unsafe { device.get_image_memory_requirements(render_image) };
        let render_allocation = allocator.allocate(&AllocationCreateDesc { name: "render_image", requirements: image_reqs, location: MemoryLocation::GpuOnly, linear: false, allocation_scheme: AllocationScheme::GpuAllocatorManaged })?;
        unsafe { device.bind_image_memory(render_image, render_allocation.memory(), render_allocation.offset()) }?;
        let render_view = unsafe { device.create_image_view(&vk::ImageViewCreateInfo { s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO, image: render_image, view_type: vk::ImageViewType::TYPE_2D, format: vk::Format::R8G8B8A8_UNORM, subresource_range: vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }, ..Default::default() }, None) }?;

        // Create Readback Buffer
        let buffer_info = vk::BufferCreateInfo { s_type: vk::StructureType::BUFFER_CREATE_INFO, size: (width * height * 4) as u64, usage: vk::BufferUsageFlags::TRANSFER_DST, sharing_mode: vk::SharingMode::EXCLUSIVE, ..Default::default() };
        let readback_buffer = unsafe { device.create_buffer(&buffer_info, None) }?;
        let buffer_reqs = unsafe { device.get_buffer_memory_requirements(readback_buffer) };
        let readback_allocation = allocator.allocate(&AllocationCreateDesc { name: "readback_buffer", requirements: buffer_reqs, location: MemoryLocation::GpuToCpu, linear: true, allocation_scheme: AllocationScheme::GpuAllocatorManaged })?;
        unsafe { device.bind_buffer_memory(readback_buffer, readback_allocation.memory(), readback_allocation.offset()) }?;

        let descriptor_pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo { s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO, max_sets: 1, pool_size_count: 2, p_pool_sizes: [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }, vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1 }].as_ptr(), ..Default::default() }, None) }?;
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo { s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO, descriptor_pool, descriptor_set_count: 1, p_set_layouts: &descriptor_set_layout, ..Default::default() }) }?;
        unsafe { device.update_descriptor_sets(&[vk::WriteDescriptorSet { s_type: vk::StructureType::WRITE_DESCRIPTOR_SET, dst_set: descriptor_sets[0], dst_binding: 0, descriptor_count: 1, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, p_buffer_info: &vk::DescriptorBufferInfo { buffer: field_buffer, offset: 0, range: vk::WHOLE_SIZE }, ..Default::default() }, vk::WriteDescriptorSet { s_type: vk::StructureType::WRITE_DESCRIPTOR_SET, dst_set: descriptor_sets[0], dst_binding: 1, descriptor_count: 1, descriptor_type: vk::DescriptorType::STORAGE_IMAGE, p_image_info: &vk::DescriptorImageInfo { image_view: render_view, image_layout: vk::ImageLayout::GENERAL, ..Default::default() }, ..Default::default() }], &[]) };

        Ok(Self { device, allocator: ctx.allocator.clone(), pipeline, pipeline_layout, descriptor_set_layout, descriptor_pool, descriptor_sets, render_image, render_allocation: Some(render_allocation), render_view, readback_buffer, readback_allocation: Some(readback_allocation), width, height })
    }

    pub fn record_command_buffer(&self, cmd: vk::CommandBuffer, nx: u32, ny: u32, nz: u32, time: f32) {
        unsafe {
            // Transition image to general layout
            let barrier = vk::ImageMemoryBarrier { s_type: vk::StructureType::IMAGE_MEMORY_BARRIER, old_layout: vk::ImageLayout::UNDEFINED, new_layout: vk::ImageLayout::GENERAL, src_access_mask: vk::AccessFlags::empty(), dst_access_mask: vk::AccessFlags::SHADER_WRITE, image: self.render_image, subresource_range: vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }, ..Default::default() };
            self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);

            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &self.descriptor_sets, &[]);
            let pc = RenderPushConstants { nx, ny, nz, width: self.width, height: self.height, time };
            self.device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(&pc as *const RenderPushConstants as *const u8, size_of::<RenderPushConstants>()));
            self.device.cmd_dispatch(cmd, self.width.div_ceil(16), self.height.div_ceil(16), 1);

            // Copy to buffer
            let barrier = vk::ImageMemoryBarrier { s_type: vk::StructureType::IMAGE_MEMORY_BARRIER, old_layout: vk::ImageLayout::GENERAL, new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL, src_access_mask: vk::AccessFlags::SHADER_WRITE, dst_access_mask: vk::AccessFlags::TRANSFER_READ, image: self.render_image, subresource_range: vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }, ..Default::default() };
            self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
            let copy_region = vk::BufferImageCopy { buffer_offset: 0, image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 }, image_extent: vk::Extent3D { width: self.width, height: self.height, depth: 1 }, ..Default::default() };
            self.device.cmd_copy_image_to_buffer(cmd, self.render_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.readback_buffer, &[copy_region]);
        }
    }

    pub fn save_frame(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let size = (self.width * self.height * 4) as usize;
        let mut pixels = vec![0u8; size];
        let ptr = self.readback_allocation.as_ref().unwrap().mapped_ptr().unwrap().as_ptr() as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(ptr, pixels.as_mut_ptr(), size) };
        image::save_buffer(path, &pixels, self.width, self.height, image::ColorType::Rgba8)?;
        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct LbmPushConstants { nx: u32, ny: u32, nz: u32, global_tau_scale: f32 }
