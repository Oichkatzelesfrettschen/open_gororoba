use ash::{vk, Device};
use std::ffi::CString;
use std::sync::Arc;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation; 
use crate::VulkanContext;
use std::mem::size_of;

#[allow(dead_code)]
pub struct LbmComputePipeline {
    device: Arc<Device>,
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
    
    grid_dim: (u32, u32, u32),
}

impl LbmComputePipeline {
    pub fn new(ctx: &VulkanContext, grid_dim: (u32, u32, u32)) -> Result<Self, Box<dyn std::error::Error>> {
        let device = ctx.device.clone();

        // 1. Compile Shader
        let compiler = shaderc::Compiler::new().unwrap();
        let source = include_str!("../shaders/lbm.comp.glsl");
        let binary = compiler.compile_into_spirv(
            source, 
            shaderc::ShaderKind::Compute, 
            "lbm.comp", 
            "main", 
            None
        )?;
        
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: binary.as_binary_u8().len(),
            p_code: binary.as_binary().as_ptr(),
            ..Default::default()
        };
        let shader_module = unsafe { device.create_shader_module(&shader_module_create_info, None) }?;

        // 2. Descriptor Layout
        let bindings = [
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
        ];

        let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }?;

        // 3. Pipeline Layout
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

        // 4. Compute Pipeline
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
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) }
            .map_err(|e| e.1)?[0];

        // Cleanup shader module
        unsafe { device.destroy_shader_module(shader_module, None) };

        // 5. Allocate Buffers
        let n_cells = (grid_dim.0 * grid_dim.1 * grid_dim.2) as u64;
        let f_size = n_cells * 19 * 4; // float * 19
        let rho_size = n_cells * 4;    // float
        let u_size = n_cells * 3 * 4;  // float * 3

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
                name,
                requirements: reqs,
                location: MemoryLocation::GpuOnly,
                linear: true, 
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;
            Ok((buffer, allocation))
        };

        let f_in = create_buffer(f_size, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, "f_in")?;
        let f_out = create_buffer(f_size, vk::BufferUsageFlags::STORAGE_BUFFER, "f_out")?;
        let rho = create_buffer(rho_size, vk::BufferUsageFlags::STORAGE_BUFFER, "rho")?;
        let u = create_buffer(u_size, vk::BufferUsageFlags::STORAGE_BUFFER, "u")?;

        // 6. Descriptor Sets
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4,
        };
        let pool_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
            ..Default::default()
        };
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;
        
        let alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &descriptor_set_layout,
            ..Default::default()
        };
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }?;

        // Update Descriptors
        let buffer_infos = [
            vk::DescriptorBufferInfo { buffer: f_in.0, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: f_out.0, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: rho.0, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: u.0, offset: 0, range: vk::WHOLE_SIZE },
        ];
        
        let writes = [
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_sets[0],
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_infos[0],
                ..Default::default()
            },
             vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_sets[0],
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_infos[1],
                ..Default::default()
            },
             vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_sets[0],
                dst_binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_infos[2],
                ..Default::default()
            },
             vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_sets[0],
                dst_binding: 3,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_infos[3],
                ..Default::default()
            },
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        Ok(Self {
            device,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            f_in_buffer: Some(f_in),
            f_out_buffer: Some(f_out),
            rho_buffer: Some(rho),
            u_buffer: Some(u),
            grid_dim,
        })
    }

    pub fn record_command_buffer(&self, cmd: vk::CommandBuffer) {
        let device = &self.device;
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &self.descriptor_sets,
                &[],
            );
            
            let push_constants = LbmPushConstants {
                nx: self.grid_dim.0,
                ny: self.grid_dim.1,
                nz: self.grid_dim.2,
                tau: 0.6, // Typical relaxation time
            };
            
            let constants_bytes = std::slice::from_raw_parts(
                &push_constants as *const LbmPushConstants as *const u8,
                size_of::<LbmPushConstants>(),
            );

            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                constants_bytes,
            );

            // Dispatch
            let group_size_x = 8;
            let group_size_y = 8;
            let group_size_z = 8;
            let dispatch_x = self.grid_dim.0.div_ceil(group_size_x);
            let dispatch_y = self.grid_dim.1.div_ceil(group_size_y);
            let dispatch_z = self.grid_dim.2.div_ceil(group_size_z);

            device.cmd_dispatch(cmd, dispatch_x, dispatch_y, dispatch_z);
            
            // Memory Barrier for next step (swap buffers logic needed outside or here)
            // For now, just a barrier to ensure write completion
            let memory_barrier = vk::MemoryBarrier {
                s_type: vk::StructureType::MEMORY_BARRIER,
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };
            
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct LbmPushConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    tau: f32,
}
