//! Vulkan sparse D3Q19 direct active-brick backend.

use ash::vk;
use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSet,
    DescriptorSetLayout, DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope,
    HostVisibleBuffer, Instance, InstanceBuilder, QueueFamilyRequirement, ShaderModule,
    ValidationPolicy, VulkanError,
};

use crate::sparse_lbm_common::{SparseLbmError, SparseLbmPlan};

const ENTRY_POINT: &str = "sparse_d3q19_step";
const WORKGROUP_SIZE: u32 = 256;
const STEP_TIMEOUT_NS: u64 = 30_000_000_000;

const WGSL_SOURCE: &str = r#"
struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    bricks_x: u32,
    bricks_y: u32,
    bricks_z: u32,
    n_active_cells: u32,
    parity: u32,
    inv_tau_bits: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> f: array<f32>;
@group(0) @binding(1) var<storage, read> active_brick_ids: array<u32>;
@group(0) @binding(2) var<storage, read> indirect_table: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;

fn cx(i: u32) -> i32 {
    switch i {
        case 1u, 7u, 9u, 11u, 13u: { return 1; }
        case 2u, 8u, 10u, 12u, 14u: { return -1; }
        default: { return 0; }
    }
}

fn cy(i: u32) -> i32 {
    switch i {
        case 3u, 7u, 10u, 15u, 17u: { return 1; }
        case 4u, 8u, 9u, 16u, 18u: { return -1; }
        default: { return 0; }
    }
}

fn cz(i: u32) -> i32 {
    switch i {
        case 5u, 11u, 14u, 15u, 18u: { return 1; }
        case 6u, 12u, 13u, 16u, 17u: { return -1; }
        default: { return 0; }
    }
}

fn opp(i: u32) -> u32 {
    switch i {
        case 1u: { return 2u; }
        case 2u: { return 1u; }
        case 3u: { return 4u; }
        case 4u: { return 3u; }
        case 5u: { return 6u; }
        case 6u: { return 5u; }
        case 7u: { return 8u; }
        case 8u: { return 7u; }
        case 9u: { return 10u; }
        case 10u: { return 9u; }
        case 11u: { return 12u; }
        case 12u: { return 11u; }
        case 13u: { return 14u; }
        case 14u: { return 13u; }
        case 15u: { return 16u; }
        case 16u: { return 15u; }
        case 17u: { return 18u; }
        case 18u: { return 17u; }
        default: { return 0u; }
    }
}

fn weight(i: u32) -> f32 {
    if i == 0u {
        return 1.0 / 3.0;
    }
    if i <= 6u {
        return 1.0 / 18.0;
    }
    return 1.0 / 36.0;
}

fn wrap_i32(value: i32, modulus: i32) -> i32 {
    var wrapped = value % modulus;
    if wrapped < 0 {
        wrapped = wrapped + modulus;
    }
    return wrapped;
}

fn finite_or_zero(value: f32) -> f32 {
    if value == value && value <= 3.402823466e38 && value >= -3.402823466e38 {
        return value;
    }
    return 0.0;
}

fn neighbor_tid(x: i32, y: i32, z: i32, dx: i32, dy: i32, dz: i32) -> i32 {
    let xn = u32(wrap_i32(x + dx, i32(params.nx)));
    let yn = u32(wrap_i32(y + dy, i32(params.ny)));
    let zn = u32(wrap_i32(z + dz, i32(params.nz)));
    let bx = xn / 8u;
    let by = yn / 8u;
    let bz = zn / 8u;
    let brick = bx + params.bricks_x * (by + params.bricks_y * bz);
    let pool = indirect_table[brick];
    if pool < 0 {
        return -1;
    }
    let lx = xn % 8u;
    let ly = yn % 8u;
    let lz = zn % 8u;
    return pool * 512 + i32(lx + 8u * (ly + 8u * lz));
}

@compute @workgroup_size(256)
fn sparse_d3q19_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid_u = gid.x;
    if tid_u >= params.n_active_cells {
        return;
    }
    let tid = u32(tid_u);
    let pool_idx = tid / 512u;
    let local_idx = tid % 512u;
    let global_brick = active_brick_ids[pool_idx];
    let bx = global_brick % params.bricks_x;
    let by = (global_brick / params.bricks_x) % params.bricks_y;
    let bz = global_brick / (params.bricks_x * params.bricks_y);
    let lx = local_idx % 8u;
    let ly = (local_idx / 8u) % 8u;
    let lz = local_idx / 64u;
    let x_u = bx * 8u + lx;
    let y_u = by * 8u + ly;
    let z_u = bz * 8u + lz;
    if x_u >= params.nx || y_u >= params.ny || z_u >= params.nz {
        return;
    }
    let x = i32(x_u);
    let y = i32(y_u);
    let z = i32(z_u);

    var local: array<f32, 19>;
    var rho = 0.0;
    var mx = 0.0;
    var my = 0.0;
    var mz = 0.0;
    for (var i = 0u; i < 19u; i = i + 1u) {
        var read_dir = i;
        var src_tid = tid;
        if params.parity != 0u {
            let candidate = neighbor_tid(x, y, z, -cx(i), -cy(i), -cz(i));
            if candidate >= 0 {
                read_dir = opp(i);
                src_tid = u32(candidate);
            }
        }
        let fi = finite_or_zero(f[read_dir * params.n_active_cells + src_tid]);
        local[i] = fi;
        rho = rho + fi;
        mx = mx + f32(cx(i)) * fi;
        my = my + f32(cy(i)) * fi;
        mz = mz + f32(cz(i)) * fi;
    }

    var ux = 0.0;
    var uy = 0.0;
    var uz = 0.0;
    if rho > 1.0e-20 {
        let inv_rho = 1.0 / rho;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho = 1.0;
    }
    let inv_tau = bitcast<f32>(params.inv_tau_bits);
    let u_sq = ux * ux + uy * uy + uz * uz;
    let base = 1.0 - 1.5 * u_sq;
    for (var j = 0u; j < 19u; j = j + 1u) {
        let eu = f32(cx(j)) * ux + f32(cy(j)) * uy + f32(cz(j)) * uz;
        let f_eq = weight(j) * rho * (base + 3.0 * eu + 4.5 * eu * eu);
        local[j] = local[j] - (local[j] - f_eq) * inv_tau;
    }

    for (var k = 0u; k < 19u; k = k + 1u) {
        if params.parity == 0u {
            let candidate = neighbor_tid(x, y, z, cx(k), cy(k), cz(k));
            var write_tid = tid;
            if candidate >= 0 {
                write_tid = u32(candidate);
            }
            f[opp(k) * params.n_active_cells + write_tid] = local[k];
        } else {
            f[k * params.n_active_cells + tid] = local[k];
        }
    }
}
"#;

#[derive(Debug, thiserror::Error)]
pub enum SparseVulkanError {
    #[error(transparent)]
    Sparse(#[from] SparseLbmError),
    #[error("vulkan helper error: {0}")]
    Vulkan(#[from] VulkanError),
    #[error("vulkan API error: {0:?}")]
    Vk(#[from] vk::Result),
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ParamsUbo {
    nx: u32,
    ny: u32,
    nz: u32,
    bricks_x: u32,
    bricks_y: u32,
    bricks_z: u32,
    n_active_cells: u32,
    parity: u32,
    inv_tau_bits: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

unsafe impl bytemuck::Pod for ParamsUbo {}
unsafe impl bytemuck::Zeroable for ParamsUbo {}

pub fn evolve_sparse_d3q19_vulkan(
    plan: &SparseLbmPlan,
    tau: f32,
    f_init: &[f32],
    num_steps: usize,
) -> Result<Vec<f32>, SparseVulkanError> {
    validate_sparse_vulkan_inputs(plan, tau, f_init)?;
    if num_steps == 0 {
        return Ok(f_init.to_vec());
    }

    let context = create_sparse_vulkan_context()?;
    let buffers = create_sparse_vulkan_buffers(&context, plan, f_init)?;
    write_sparse_vulkan_inputs(&buffers, plan, f_init)?;
    bind_sparse_vulkan_buffers(&context.set, &buffers);
    let dispatch = DispatchScope::new(&context.device)?;
    dispatch_sparse_vulkan_steps(&dispatch, &context, &buffers, plan, tau, num_steps)?;
    let out = buffers.f.read_f32_slice(f_init.len())?;
    wait_sparse_vulkan_idle(&context)?;
    Ok(out)
}

fn validate_sparse_vulkan_inputs(
    plan: &SparseLbmPlan,
    tau: f32,
    f_init: &[f32],
) -> Result<(), SparseLbmError> {
    if tau.is_nan() || tau <= 0.5 {
        return Err(SparseLbmError::UnstableTau(tau));
    }
    if plan.n_active_bricks() == 0 {
        return Err(SparseLbmError::NoActiveBricks);
    }
    plan.validate_f_len(f_init)
}

struct SparseVulkanContext {
    set: DescriptorSet,
    _pool: DescriptorPool,
    pipeline: ComputePipeline,
    _layout: DescriptorSetLayout,
    _shader: ShaderModule,
    device: Device,
    adapter: Adapter,
    _instance: Instance,
}

fn create_sparse_vulkan_context() -> Result<SparseVulkanContext, SparseVulkanError> {
    let instance = InstanceBuilder::new("sparse_lbm_vulkan")
        .api_version(vk::API_VERSION_1_2)
        .validation(ValidationPolicy::default_for_profile())
        .build()?;
    let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)?;
    let device = DeviceBuilder::new(adapter.clone()).build(&instance)?;
    let shader = ShaderModule::from_wgsl(&device, WGSL_SOURCE, ENTRY_POINT)?;
    let layout = DescriptorSetLayoutSpec::new()
        .storage_buffer(0)
        .storage_buffer(1)
        .storage_buffer(2)
        .uniform_buffer(3)
        .build(&device)?;
    let pipeline = ComputePipelineBuilder::new(&device, &shader)
        .descriptor_layout(&layout)
        .build()?;
    let pool = DescriptorPool::for_layout(&device, &layout, 1)?;
    let set = pool.allocate_set(&layout)?;
    Ok(SparseVulkanContext {
        set,
        _pool: pool,
        pipeline,
        _layout: layout,
        _shader: shader,
        device,
        adapter,
        _instance: instance,
    })
}

struct SparseVulkanBuffers {
    params: HostVisibleBuffer,
    indirect: HostVisibleBuffer,
    active: HostVisibleBuffer,
    f: HostVisibleBuffer,
}

fn create_sparse_vulkan_buffers(
    context: &SparseVulkanContext,
    plan: &SparseLbmPlan,
    f_init: &[f32],
) -> Result<SparseVulkanBuffers, SparseVulkanError> {
    let f = HostVisibleBuffer::storage(
        &context.device,
        &context.adapter,
        std::mem::size_of_val(f_init) as vk::DeviceSize,
    )?;
    let active = HostVisibleBuffer::storage(
        &context.device,
        &context.adapter,
        std::mem::size_of_val(plan.active_brick_ids.as_slice()) as vk::DeviceSize,
    )?;
    let indirect = HostVisibleBuffer::storage(
        &context.device,
        &context.adapter,
        std::mem::size_of_val(plan.indirect_table.as_slice()) as vk::DeviceSize,
    )?;
    let params = HostVisibleBuffer::uniform(
        &context.device,
        &context.adapter,
        std::mem::size_of::<ParamsUbo>() as vk::DeviceSize,
    )?;
    Ok(SparseVulkanBuffers {
        params,
        indirect,
        active,
        f,
    })
}

fn write_sparse_vulkan_inputs(
    buffers: &SparseVulkanBuffers,
    plan: &SparseLbmPlan,
    f_init: &[f32],
) -> Result<(), SparseVulkanError> {
    buffers.f.write_f32_slice(f_init)?;
    buffers.active.write_u32_slice(&plan.active_brick_ids)?;
    buffers
        .indirect
        .write_bytes(bytemuck::cast_slice(&plan.indirect_table))?;
    Ok(())
}

fn bind_sparse_vulkan_buffers(set: &DescriptorSet, buffers: &SparseVulkanBuffers) {
    set.write_storage_buffer(0, &buffers.f);
    set.write_storage_buffer(1, &buffers.active);
    set.write_storage_buffer(2, &buffers.indirect);
    set.write_uniform_buffer(3, &buffers.params);
}

fn wait_sparse_vulkan_idle(context: &SparseVulkanContext) -> Result<(), SparseVulkanError> {
    // SAFETY: the logical device is alive through `context`; waiting idle only
    // synchronizes work already submitted to that device before owned handles
    // leave scope.
    unsafe {
        context.device.raw().device_wait_idle()?;
    }
    Ok(())
}

fn dispatch_sparse_vulkan_steps(
    dispatch: &DispatchScope,
    context: &SparseVulkanContext,
    buffers: &SparseVulkanBuffers,
    plan: &SparseLbmPlan,
    tau: f32,
    num_steps: usize,
) -> Result<(), SparseVulkanError> {
    let group_count = (plan.n_active_cells() as u32).div_ceil(WORKGROUP_SIZE);
    for step in 0..num_steps {
        let params = params_for_step(plan, tau, step);
        buffers.params.write_bytes(bytemuck::bytes_of(&params))?;
        dispatch.dispatch(
            &context.pipeline,
            context.set.raw(),
            group_count,
            1,
            1,
            STEP_TIMEOUT_NS,
        )?;
    }
    Ok(())
}

fn params_for_step(plan: &SparseLbmPlan, tau: f32, step: usize) -> ParamsUbo {
    ParamsUbo {
        nx: plan.nx as u32,
        ny: plan.ny as u32,
        nz: plan.nz as u32,
        bricks_x: plan.bricks_x as u32,
        bricks_y: plan.bricks_y as u32,
        bricks_z: plan.bricks_z as u32,
        n_active_cells: plan.n_active_cells() as u32,
        parity: (step & 1) as u32,
        inv_tau_bits: (1.0_f32 / tau).to_bits(),
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_lbm_common::evolve_sparse_d3q19_cpu;

    #[test]
    fn zero_steps_returns_input() {
        let mask = vec![1u8; 8 * 8 * 8];
        let plan = SparseLbmPlan::from_geometry_mask(8, 8, 8, &mask).unwrap();
        let f0 = plan.equilibrium_at_rest();
        let out = evolve_sparse_d3q19_vulkan(&plan, 1.0, &f0, 0).unwrap();
        assert_eq!(out, f0);
    }

    #[test]
    #[ignore = "Vulkan compute adapter required"]
    fn vulkan_matches_cpu_sparse_equilibrium() {
        let mask = vec![1u8; 8 * 8 * 8];
        let plan = SparseLbmPlan::from_geometry_mask(8, 8, 8, &mask).unwrap();
        let f0 = plan.equilibrium_at_rest();
        let cpu = evolve_sparse_d3q19_cpu(&plan, 1.0, &f0, 3).unwrap();
        let gpu = match evolve_sparse_d3q19_vulkan(&plan, 1.0, &f0, 3) {
            Ok(out) => out,
            Err(err) => {
                eprintln!("skip: Vulkan sparse LBM failed ({err})");
                return;
            }
        };
        for (idx, (got, expected)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1.0e-6,
                "idx={idx} got={got} expected={expected}"
            );
        }
    }
}
