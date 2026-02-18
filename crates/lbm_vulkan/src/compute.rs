use ash::{vk, Device};
use std::ffi::CString;
use std::sync::Arc;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation; 
use crate::VulkanContext;
use std::mem::size_of;
use std::sync::Mutex;

// Helper to compile WGSL to SPIR-V using naga
fn compile_wgsl(source: &str, _name: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let module = naga::front::wgsl::parse_str(source)?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)?;
    
    let mut words = Vec::new();
    let mut writer = naga::back::spv::Writer::new(&naga::back::spv::Options {
        lang_version: (1, 3),
        flags: naga::back::spv::WriterFlags::empty(),
        ..Default::default()
    })?;
    writer.write(&module, &info, None, &None, &mut words)?;
    Ok(words)
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct LbmConstants { nx: u32, ny: u32, nz: u32, global_tau_scale: f32 }

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ZdGenConstants { nx: u32, ny: u32, nz: u32, tau_base: f32, tau_amp: f32, lambda: f32 }

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RenderConstants { nx: u32, ny: u32, nz: u32, width: u32, height: u32, time: f32 }

pub struct LbmComputePipeline {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    
    // Ping-Pong Sets: [0] = A->B, [1] = B->A
    descriptor_sets: Vec<vk::DescriptorSet>,
    step_counter: u64,
    
    // Buffers
    f_a_buffer: Option<(vk::Buffer, Allocation)>,
    f_b_buffer: Option<(vk::Buffer, Allocation)>,
    rho_buffer: Option<(vk::Buffer, Allocation)>,
    u_buffer: Option<(vk::Buffer, Allocation)>,
    pub tau_buffer: Option<(vk::Buffer, Allocation)>,
    pub force_buffer: Option<(vk::Buffer, Allocation)>,
    pub entropy_buffer: Option<(vk::Buffer, Allocation)>,
    uniform_buffer: Option<(vk::Buffer, Allocation)>,

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
            free(&mut self.f_a_buffer);
            free(&mut self.f_b_buffer);
            free(&mut self.rho_buffer);
            free(&mut self.u_buffer);
            free(&mut self.tau_buffer);
            free(&mut self.force_buffer);
            free(&mut self.entropy_buffer);
            free(&mut self.uniform_buffer);

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
        
        // 1. Compile Shaders
        let lbm_code = compile_wgsl(include_str!("../shaders/lbm.wgsl"), "lbm")?;
        
        let lbm_module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo { code_size: lbm_code.len()*4, p_code: lbm_code.as_ptr(), ..Default::default() }, None) }?;

        // 2. Layouts
        // LBM Layout: 7 Storage, 1 Uniform
        let bindings = [
            vk::DescriptorSetLayoutBinding { binding: 0, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 1, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 2, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 3, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 4, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 5, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 6, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 7, descriptor_type: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
        ];
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo { binding_count: 8, p_bindings: bindings.as_ptr(), ..Default::default() }, None) }?;

        // Pipelines
        let pipeline_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo { set_layout_count: 1, p_set_layouts: &descriptor_set_layout, ..Default::default() }, None) }?;

        let entry = CString::new("main")?;
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo { stage: vk::PipelineShaderStageCreateInfo { stage: vk::ShaderStageFlags::COMPUTE, module: lbm_module, p_name: entry.as_ptr(), ..Default::default() }, layout: pipeline_layout, ..Default::default() }], None) }.map_err(|e| e.1)?[0];

        unsafe { device.destroy_shader_module(lbm_module, None) };

        // Buffers
        let n_cells = (grid_dim.0 * grid_dim.1 * grid_dim.2) as u64;
        let mut allocator = ctx.allocator.lock().unwrap();
        let mut create = |size, usage, name, loc| -> Result<(vk::Buffer, Allocation), Box<dyn std::error::Error>> {
             let buffer = unsafe { device.create_buffer(&vk::BufferCreateInfo { size, usage, sharing_mode: vk::SharingMode::EXCLUSIVE, ..Default::default() }, None) }?;
             let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
             let alloc = allocator.allocate(&AllocationCreateDesc { name, requirements: reqs, location: loc, linear: true, allocation_scheme: AllocationScheme::GpuAllocatorManaged })?;
             unsafe { device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset()) }?;
             Ok((buffer, alloc))
        };

        let usage_rw = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let f_a = create(n_cells * 19 * 4, usage_rw, "f_a", MemoryLocation::GpuOnly)?;
        let f_b = create(n_cells * 19 * 4, usage_rw, "f_b", MemoryLocation::GpuOnly)?;
        let rho = create(n_cells * 4, usage_rw, "rho", MemoryLocation::GpuOnly)?;
        let u = create(n_cells * 3 * 4, usage_rw, "u", MemoryLocation::GpuOnly)?;
        let tau = create(n_cells * 4, usage_rw, "tau", MemoryLocation::CpuToGpu)?;
        let force = create(n_cells * 3 * 4, usage_rw, "force", MemoryLocation::CpuToGpu)?;
        let entropy = create(n_cells * 4, usage_rw, "entropy", MemoryLocation::GpuToCpu)?;
        let uniform = create(size_of::<LbmConstants>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, "lbm_const", MemoryLocation::CpuToGpu)?;

        // Descriptors: 2 sets for Ping-Pong
        let pool_sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 20 }, vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 2 }];
        let pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo { max_sets: 2, pool_size_count: 2, p_pool_sizes: pool_sizes.as_ptr(), ..Default::default() }, None) }?;
        
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo { descriptor_pool: pool, descriptor_set_count: 2, p_set_layouts: [descriptor_set_layout, descriptor_set_layout].as_ptr(), ..Default::default() }) }?;

        // Update Sets
        let update = |set, f_in: vk::Buffer, f_out: vk::Buffer| {
            let infos = [
                vk::DescriptorBufferInfo { buffer: f_in, offset: 0, range: vk::WHOLE_SIZE },
                vk::DescriptorBufferInfo { buffer: f_out, offset: 0, range: vk::WHOLE_SIZE },
                vk::DescriptorBufferInfo { buffer: rho.0, offset: 0, range: vk::WHOLE_SIZE },
                vk::DescriptorBufferInfo { buffer: u.0, offset: 0, range: vk::WHOLE_SIZE },
                vk::DescriptorBufferInfo { buffer: tau.0, offset: 0, range: vk::WHOLE_SIZE },
                vk::DescriptorBufferInfo { buffer: force.0, offset: 0, range: vk::WHOLE_SIZE },
                vk::DescriptorBufferInfo { buffer: entropy.0, offset: 0, range: vk::WHOLE_SIZE },
            ];
            let u_info = vk::DescriptorBufferInfo { buffer: uniform.0, offset: 0, range: vk::WHOLE_SIZE };
            let mut w = Vec::new();
            for (i, info) in infos.iter().enumerate() {
                w.push(vk::WriteDescriptorSet { dst_set: set, dst_binding: i as u32, descriptor_count: 1, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, p_buffer_info: info, ..Default::default() });
            }
            w.push(vk::WriteDescriptorSet { dst_set: set, dst_binding: 7, descriptor_count: 1, descriptor_type: vk::DescriptorType::UNIFORM_BUFFER, p_buffer_info: &u_info, ..Default::default() });
            unsafe { device.update_descriptor_sets(&w, &[]) };
        };

        update(sets[0], f_a.0, f_b.0); // Set 0: A -> B
        update(sets[1], f_b.0, f_a.0); // Set 1: B -> A

        Ok(Self {
            device, allocator: ctx.allocator.clone(), pipeline, pipeline_layout, descriptor_set_layout, descriptor_pool: pool,
            descriptor_sets: vec![sets[0], sets[1]], // [0]: A->B, [1]: B->A
            step_counter: 0,
            f_a_buffer: Some(f_a), f_b_buffer: Some(f_b), rho_buffer: Some(rho), u_buffer: Some(u),
            tau_buffer: Some(tau), force_buffer: Some(force), entropy_buffer: Some(entropy), uniform_buffer: Some(uniform),
            grid_dim
        })
    }

    pub fn write_inputs(&mut self, tau: &[f32], force: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        if let Some((_, alloc)) = &self.tau_buffer
            && !tau.is_empty() {
            let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *mut f32;
            unsafe { std::ptr::copy_nonoverlapping(tau.as_ptr(), ptr, tau.len()) };
        }
        if let Some((_, alloc)) = &self.force_buffer
            && !force.is_empty() {
            let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *mut f32;
            unsafe { std::ptr::copy_nonoverlapping(force.as_ptr(), ptr, force.len()) };
        }
        Ok(())
    }

    pub fn write_state(&mut self, ctx: &VulkanContext, f_init: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        if let Some((f_buf, _)) = &self.f_a_buffer
            && !f_init.is_empty() {
            let size = (f_init.len() * 4) as u64;
                let mut allocator = self.allocator.lock().unwrap();
                
                // 1. Staging Buffer
                let staging_info = vk::BufferCreateInfo {
                    s_type: vk::StructureType::BUFFER_CREATE_INFO,
                    size,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                };
                let staging_buf = unsafe { self.device.create_buffer(&staging_info, None) }?;
                let reqs = unsafe { self.device.get_buffer_memory_requirements(staging_buf) };
                let staging_alloc = allocator.allocate(&AllocationCreateDesc {
                    name: "staging",
                    requirements: reqs,
                    location: MemoryLocation::CpuToGpu,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })?;
                unsafe { self.device.bind_buffer_memory(staging_buf, staging_alloc.memory(), staging_alloc.offset()) }?;

                // 2. Write to Staging
                let ptr = staging_alloc.mapped_ptr().unwrap().as_ptr() as *mut f32;
                unsafe { std::ptr::copy_nonoverlapping(f_init.as_ptr(), ptr, f_init.len()) };

                // 3. Copy Command
                let pool_info = vk::CommandPoolCreateInfo {
                    s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
                    queue_family_index: ctx.queue_family_index,
                    flags: vk::CommandPoolCreateFlags::TRANSIENT,
                    ..Default::default()
                };
                let pool = unsafe { self.device.create_command_pool(&pool_info, None) }?;
                let alloc_info = vk::CommandBufferAllocateInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                    command_pool: pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                    ..Default::default()
                };
                let cmd = unsafe { self.device.allocate_command_buffers(&alloc_info) }?[0];

                let begin_info = vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                };
                unsafe {
                    self.device.begin_command_buffer(cmd, &begin_info)?;
                    let region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size };
                    self.device.cmd_copy_buffer(cmd, staging_buf, *f_buf, &[region]);
                    self.device.end_command_buffer(cmd)?;
                    
                    let submit_info = vk::SubmitInfo {
                        s_type: vk::StructureType::SUBMIT_INFO,
                        command_buffer_count: 1,
                        p_command_buffers: &cmd,
                        ..Default::default()
                    };
                    self.device.queue_submit(ctx.queue, &[submit_info], vk::Fence::null())?;
                    self.device.queue_wait_idle(ctx.queue)?;

                    self.device.destroy_command_pool(pool, None);
                    self.device.destroy_buffer(staging_buf, None);
                    allocator.free(staging_alloc)?;
                }
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

    pub fn record_command_buffer(&mut self, cmd: vk::CommandBuffer) {
        unsafe {
            // Ping-Pong Logic
            let set_idx = (self.step_counter % 2) as usize;
            self.step_counter += 1;

            let pc = LbmConstants { nx: self.grid_dim.0, ny: self.grid_dim.1, nz: self.grid_dim.2, global_tau_scale: 1.0 };
            if let Some((_, alloc)) = &self.uniform_buffer {
                let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *mut LbmConstants;
                std::ptr::write(ptr, pc);
            }

            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.descriptor_sets[set_idx]], &[]);
            self.device.cmd_dispatch(cmd, self.grid_dim.0.div_ceil(8), self.grid_dim.1.div_ceil(8), self.grid_dim.2.div_ceil(8));
            
            let barrier = vk::MemoryBarrier { s_type: vk::StructureType::MEMORY_BARRIER, src_access_mask: vk::AccessFlags::SHADER_WRITE, dst_access_mask: vk::AccessFlags::SHADER_READ, ..Default::default() };
            self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        }
    }
}

// ... [ZdGenPipeline and VulkanRenderer remain mostly same but need ensuring they use Uniforms correctly] ...
// I will include them to ensure file completeness and correct Uniform Buffer usage for ZdGen and Render as previously implemented.

pub struct ZdGenPipeline {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    uniform_buffer: Option<(vk::Buffer, Allocation)>,
    grid_dim: (u32, u32, u32),
}

impl Drop for ZdGenPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let mut allocator = self.allocator.lock().unwrap();
            if let Some((b, a)) = self.uniform_buffer.take() {
                self.device.destroy_buffer(b, None);
                let _ = allocator.free(a);
            }
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

impl ZdGenPipeline {
    pub fn new(ctx: &VulkanContext, grid_dim: (u32, u32, u32), tau_buffer: vk::Buffer) -> Result<Self, Box<dyn std::error::Error>> {
        let device = ctx.device.clone();
        let code = compile_wgsl(include_str!("../shaders/zd_gen.wgsl"), "zd_gen")?;
        let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo { code_size: code.len()*4, p_code: code.as_ptr(), ..Default::default() }, None) }?;

        let bindings = [
            vk::DescriptorSetLayoutBinding { binding: 0, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 1, descriptor_type: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData }
        ];
        let dsl = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo { binding_count: 2, p_bindings: bindings.as_ptr(), ..Default::default() }, None) }?;
        let layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo { set_layout_count: 1, p_set_layouts: &dsl, ..Default::default() }, None) }?;
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo { stage: vk::PipelineShaderStageCreateInfo { stage: vk::ShaderStageFlags::COMPUTE, module, p_name: CString::new("main")?.as_ptr(), ..Default::default() }, layout, ..Default::default() }], None) }.map_err(|e| e.1)?[0];
        unsafe { device.destroy_shader_module(module, None) };

        let mut allocator = ctx.allocator.lock().unwrap();
        let buffer = unsafe { device.create_buffer(&vk::BufferCreateInfo { size: size_of::<ZdGenConstants>() as u64, usage: vk::BufferUsageFlags::UNIFORM_BUFFER, sharing_mode: vk::SharingMode::EXCLUSIVE, ..Default::default() }, None) }?;
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let alloc = allocator.allocate(&AllocationCreateDesc { name: "zd_uniform", requirements: reqs, location: MemoryLocation::CpuToGpu, linear: true, allocation_scheme: AllocationScheme::GpuAllocatorManaged })?;
        unsafe { device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset()) }?;

        let pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo { max_sets: 1, pool_size_count: 2, p_pool_sizes: [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }, vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 }].as_ptr(), ..Default::default() }, None) }?;
        let set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo { descriptor_pool: pool, descriptor_set_count: 1, p_set_layouts: &dsl, ..Default::default() }) }?[0];

        let buf_info = vk::DescriptorBufferInfo { buffer: tau_buffer, offset: 0, range: vk::WHOLE_SIZE };
        let uni_info = vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE };
        unsafe { device.update_descriptor_sets(&[
            vk::WriteDescriptorSet { dst_set: set, dst_binding: 0, descriptor_count: 1, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, p_buffer_info: &buf_info, ..Default::default() },
            vk::WriteDescriptorSet { dst_set: set, dst_binding: 1, descriptor_count: 1, descriptor_type: vk::DescriptorType::UNIFORM_BUFFER, p_buffer_info: &uni_info, ..Default::default() }
        ], &[]) };

        Ok(Self { device, allocator: ctx.allocator.clone(), pipeline, pipeline_layout: layout, descriptor_set_layout: dsl, descriptor_pool: pool, descriptor_sets: vec![set], uniform_buffer: Some((buffer, alloc)), grid_dim })
    }

    pub fn record_command_buffer(&self, cmd: vk::CommandBuffer) {
        unsafe {
            if let Some((_, alloc)) = &self.uniform_buffer {
                let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *mut ZdGenConstants;
                std::ptr::write(ptr, ZdGenConstants { nx: self.grid_dim.0, ny: self.grid_dim.1, nz: self.grid_dim.2, tau_base: 0.6, tau_amp: 0.2, lambda: 5.0 });
            }
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &self.descriptor_sets, &[]);
            self.device.cmd_dispatch(cmd, self.grid_dim.0.div_ceil(8), self.grid_dim.1.div_ceil(8), self.grid_dim.2.div_ceil(8));
            self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[vk::MemoryBarrier { s_type: vk::StructureType::MEMORY_BARRIER, src_access_mask: vk::AccessFlags::SHADER_WRITE, dst_access_mask: vk::AccessFlags::SHADER_READ, ..Default::default() }], &[], &[]);
        }
    }
}

pub struct VulkanRenderer {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub render_image: vk::Image,
    pub render_allocation: Option<Allocation>,
    pub render_view: vk::ImageView,
    pub readback_buffer: vk::Buffer,
    pub readback_allocation: Option<Allocation>,
    uniform_buffer: Option<(vk::Buffer, Allocation)>,
    width: u32, height: u32,
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let mut allocator = self.allocator.lock().unwrap();
            if let Some((b, a)) = self.uniform_buffer.take() { self.device.destroy_buffer(b, None); allocator.free(a).unwrap(); }
            if let Some(a) = self.readback_allocation.take() { self.device.destroy_buffer(self.readback_buffer, None); allocator.free(a).unwrap(); }
            self.device.destroy_image_view(self.render_view, None);
            if let Some(a) = self.render_allocation.take() { self.device.destroy_image(self.render_image, None); allocator.free(a).unwrap(); }
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

impl VulkanRenderer {
    pub fn new(ctx: &VulkanContext, width: u32, height: u32, field_buffer: vk::Buffer) -> Result<Self, Box<dyn std::error::Error>> {
        let device = ctx.device.clone();
        let code = compile_wgsl(include_str!("../shaders/render.wgsl"), "render")?;
        let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo { code_size: code.len()*4, p_code: code.as_ptr(), ..Default::default() }, None) }?;

        let bindings = [
            vk::DescriptorSetLayoutBinding { binding: 0, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 1, descriptor_type: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
            vk::DescriptorSetLayoutBinding { binding: 2, descriptor_type: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE, p_immutable_samplers: std::ptr::null(), _marker: std::marker::PhantomData },
        ];
        let dsl = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo { binding_count: 3, p_bindings: bindings.as_ptr(), ..Default::default() }, None) }?;
        let layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo { set_layout_count: 1, p_set_layouts: &dsl, ..Default::default() }, None) }?;
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo { stage: vk::PipelineShaderStageCreateInfo { stage: vk::ShaderStageFlags::COMPUTE, module, p_name: CString::new("main")?.as_ptr(), ..Default::default() }, layout, ..Default::default() }], None) }.map_err(|e| e.1)?[0];
        unsafe { device.destroy_shader_module(module, None) };

        let mut allocator = ctx.allocator.lock().unwrap();
        let image = unsafe { device.create_image(&vk::ImageCreateInfo { image_type: vk::ImageType::TYPE_2D, format: vk::Format::R8G8B8A8_UNORM, extent: vk::Extent3D { width, height, depth: 1 }, mip_levels: 1, array_layers: 1, samples: vk::SampleCountFlags::TYPE_1, tiling: vk::ImageTiling::OPTIMAL, usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC, sharing_mode: vk::SharingMode::EXCLUSIVE, initial_layout: vk::ImageLayout::UNDEFINED, ..Default::default() }, None) }?;
        let image_alloc = allocator.allocate(&AllocationCreateDesc { name: "render_image", requirements: unsafe { device.get_image_memory_requirements(image) }, location: MemoryLocation::GpuOnly, linear: false, allocation_scheme: AllocationScheme::GpuAllocatorManaged })?;
        unsafe { device.bind_image_memory(image, image_alloc.memory(), image_alloc.offset()) }?;
        let view = unsafe { device.create_image_view(&vk::ImageViewCreateInfo { image, view_type: vk::ImageViewType::TYPE_2D, format: vk::Format::R8G8B8A8_UNORM, subresource_range: vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }, ..Default::default() }, None) }?;

        let buffer = unsafe { device.create_buffer(&vk::BufferCreateInfo { size: (width*height*4) as u64, usage: vk::BufferUsageFlags::TRANSFER_DST, sharing_mode: vk::SharingMode::EXCLUSIVE, ..Default::default() }, None) }?;
        let buffer_alloc = allocator.allocate(&AllocationCreateDesc { name: "readback", requirements: unsafe { device.get_buffer_memory_requirements(buffer) }, location: MemoryLocation::GpuToCpu, linear: true, allocation_scheme: AllocationScheme::GpuAllocatorManaged })?;
        unsafe { device.bind_buffer_memory(buffer, buffer_alloc.memory(), buffer_alloc.offset()) }?;

        let u_buf = unsafe { device.create_buffer(&vk::BufferCreateInfo { size: size_of::<RenderConstants>() as u64, usage: vk::BufferUsageFlags::UNIFORM_BUFFER, sharing_mode: vk::SharingMode::EXCLUSIVE, ..Default::default() }, None) }?;
        let u_alloc = allocator.allocate(&AllocationCreateDesc { name: "render_uniform", requirements: unsafe { device.get_buffer_memory_requirements(u_buf) }, location: MemoryLocation::CpuToGpu, linear: true, allocation_scheme: AllocationScheme::GpuAllocatorManaged })?;
        unsafe { device.bind_buffer_memory(u_buf, u_alloc.memory(), u_alloc.offset()) }?;

        let pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo { max_sets: 1, pool_size_count: 3, p_pool_sizes: [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }, vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1 }, vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 }].as_ptr(), ..Default::default() }, None) }?;
        let set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo { descriptor_pool: pool, descriptor_set_count: 1, p_set_layouts: &dsl, ..Default::default() }) }?[0];

        let buf_info = vk::DescriptorBufferInfo { buffer: field_buffer, offset: 0, range: vk::WHOLE_SIZE };
        let img_info = vk::DescriptorImageInfo { image_view: view, image_layout: vk::ImageLayout::GENERAL, sampler: vk::Sampler::null() };
        let uni_info = vk::DescriptorBufferInfo { buffer: u_buf, offset: 0, range: vk::WHOLE_SIZE };
        unsafe { device.update_descriptor_sets(&[
            vk::WriteDescriptorSet { dst_set: set, dst_binding: 0, descriptor_count: 1, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, p_buffer_info: &buf_info, ..Default::default() },
            vk::WriteDescriptorSet { dst_set: set, dst_binding: 1, descriptor_count: 1, descriptor_type: vk::DescriptorType::STORAGE_IMAGE, p_image_info: &img_info, ..Default::default() },
            vk::WriteDescriptorSet { dst_set: set, dst_binding: 2, descriptor_count: 1, descriptor_type: vk::DescriptorType::UNIFORM_BUFFER, p_buffer_info: &uni_info, ..Default::default() }
        ], &[]) };

        Ok(Self { device, allocator: ctx.allocator.clone(), pipeline, pipeline_layout: layout, descriptor_set_layout: dsl, descriptor_pool: pool, descriptor_sets: vec![set], render_image: image, render_allocation: Some(image_alloc), render_view: view, readback_buffer: buffer, readback_allocation: Some(buffer_alloc), uniform_buffer: Some((u_buf, u_alloc)), width, height })
    }

    pub fn record_command_buffer(&self, cmd: vk::CommandBuffer, nx: u32, ny: u32, nz: u32, time: f32) {
        unsafe {
            if let Some((_, alloc)) = &self.uniform_buffer {
                let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *mut RenderConstants;
                std::ptr::write(ptr, RenderConstants { nx, ny, nz, width: self.width, height: self.height, time });
            }
            
            let barrier = vk::ImageMemoryBarrier { s_type: vk::StructureType::IMAGE_MEMORY_BARRIER, old_layout: vk::ImageLayout::UNDEFINED, new_layout: vk::ImageLayout::GENERAL, src_access_mask: vk::AccessFlags::empty(), dst_access_mask: vk::AccessFlags::SHADER_WRITE, image: self.render_image, subresource_range: vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }, ..Default::default() };
            self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);

            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &self.descriptor_sets, &[]);
            self.device.cmd_dispatch(cmd, self.width.div_ceil(16), self.height.div_ceil(16), 1);

            let barrier2 = vk::ImageMemoryBarrier { s_type: vk::StructureType::IMAGE_MEMORY_BARRIER, old_layout: vk::ImageLayout::GENERAL, new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL, src_access_mask: vk::AccessFlags::SHADER_WRITE, dst_access_mask: vk::AccessFlags::TRANSFER_READ, image: self.render_image, subresource_range: vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }, ..Default::default() };
            self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[barrier2]);

            let copy = vk::BufferImageCopy { buffer_offset: 0, image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 }, image_extent: vk::Extent3D { width: self.width, height: self.height, depth: 1 }, ..Default::default() };
            self.device.cmd_copy_image_to_buffer(cmd, self.render_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.readback_buffer, &[copy]);
        }
    }

    pub fn save_frame(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let size = (self.width * self.height * 4) as usize;
        let mut pixels = vec![0u8; size];
        if let Some(alloc) = &self.readback_allocation {
            let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *const u8;
            unsafe { std::ptr::copy_nonoverlapping(ptr, pixels.as_mut_ptr(), size) };
            image::save_buffer(path, &pixels, self.width, self.height, image::ColorType::Rgba8)?;
        }
        Ok(())
    }
}
