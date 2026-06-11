//! Vulkan compute path for the CUDA-style GRMHD conservative advance.
//!
//! The shader mirrors `kernels_grmhd.cu`: metric precompute, primitive to
//! conservative conversion, flux construction, centered flux divergence, and
//! forward Euler update over 8-channel SoA buffers. WGSL storage uses FP32, so
//! parity checks compare against the CPU mirror after the same FP32 casts.

use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSet,
    DescriptorSetLayout, DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope,
    HostVisibleBuffer, Instance, InstanceBuilder, QueueFamilyRequirement, ShaderModule,
    ValidationPolicy,
};

pub const NPRIM: usize = 8;
pub const NCONS: usize = 8;
pub const GRMHD_VULKAN_ENTRY_POINT: &str = "grmhd_step";
const WORKGROUP_SIZE: u32 = 256;
const DISPATCH_TIMEOUT_NS: u64 = 30_000_000_000;

const OP_PRECOMPUTE_METRIC: u32 = 0;
const OP_PRIM2CON: u32 = 1;
const OP_COMPUTE_FLUX: u32 = 2;
const OP_FLUX_DIVERGENCE: u32 = 3;
const OP_EULER_UPDATE: u32 = 4;

pub const GRMHD_VULKAN_WGSL: &str = r#"
const PI: f32 = 3.1415926535897932384626433832795;
const NPRIM: u32 = 8u;
const NCONS: u32 = 8u;
const RHO: u32 = 0u;
const UU: u32 = 1u;
const V1: u32 = 2u;
const V2: u32 = 3u;
const V3: u32 = 4u;
const B1: u32 = 5u;
const B2: u32 = 6u;
const B3: u32 = 7u;

struct F32Buffer {
    values: array<f32>,
};

struct Params {
    n1: u32,
    n2: u32,
    n3: u32,
    n_total: u32,
    r_min: f32,
    r_max: f32,
    kerr_a: f32,
    gam_m1: f32,
    dx1: f32,
    dx2: f32,
    dx3: f32,
    dt: f32,
    dir: u32,
    op: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0)
var<storage, read_write> prims: F32Buffer;
@group(0) @binding(1)
var<storage, read_write> cons: F32Buffer;
@group(0) @binding(2)
var<storage, read_write> cons_new: F32Buffer;
@group(0) @binding(3)
var<storage, read_write> flux: F32Buffer;
@group(0) @binding(4)
var<storage, read_write> rhs: F32Buffer;
@group(0) @binding(5)
var<storage, read_write> gcov: F32Buffer;
@group(0) @binding(6)
var<storage, read_write> sqrt_g: F32Buffer;
@group(0) @binding(7)
var<uniform> params: Params;

fn soa_index(channel: u32, cell: u32) -> u32 {
    return channel * params.n_total + cell;
}

fn metric_index(cell: u32) -> u32 {
    var tmp: u32 = cell;
    tmp = tmp / params.n3;
    let j: u32 = tmp % params.n2;
    tmp = tmp / params.n2;
    let i: u32 = tmp;
    return i * params.n2 + j;
}

fn select3(a: f32, b: f32, c: f32, idx: u32) -> f32 {
    if (idx == 1u) {
        return b;
    }
    if (idx == 2u) {
        return c;
    }
    return a;
}

fn load_metric(met_idx: u32, component: u32) -> f32 {
    return gcov.values[5u * met_idx + component];
}

fn precompute_metric(cell: u32) {
    let total: u32 = params.n1 * params.n2;
    if (cell >= total) {
        return;
    }

    let i: u32 = cell / params.n2;
    let j: u32 = cell % params.n2;
    let xi: f32 = f32(i) / f32(params.n1 - 1u);
    let r: f32 = params.r_min * exp(xi * log(params.r_max / params.r_min));
    let th: f32 = PI * (f32(j) + 0.5) / f32(params.n2);
    let sth: f32 = sin(th);
    let cth: f32 = cos(th);
    let a: f32 = params.kerr_a;
    let sigma: f32 = r * r + a * a * cth * cth;
    let delta: f32 = r * r - 2.0 * r + a * a;
    let a_met: f32 = (r * r + a * a) * (r * r + a * a) - delta * a * a * sth * sth;
    let base: u32 = 5u * cell;

    gcov.values[base + 0u] = -(1.0 - 2.0 * r / sigma);
    gcov.values[base + 1u] = sigma / delta;
    gcov.values[base + 2u] = sigma;
    gcov.values[base + 3u] = a_met * sth * sth / sigma;
    gcov.values[base + 4u] = -2.0 * a * r * sth * sth / sigma;
    sqrt_g.values[cell] = sigma * abs(sth);
}

fn prim2con(cell: u32) {
    if (cell >= params.n_total) {
        return;
    }

    let met_idx: u32 = metric_index(cell);
    let rho: f32 = prims.values[soa_index(RHO, cell)];
    let u: f32 = prims.values[soa_index(UU, cell)];
    let v1: f32 = prims.values[soa_index(V1, cell)];
    let v2: f32 = prims.values[soa_index(V2, cell)];
    let v3: f32 = prims.values[soa_index(V3, cell)];
    let b1: f32 = prims.values[soa_index(B1, cell)];
    let b2: f32 = prims.values[soa_index(B2, cell)];
    let b3: f32 = prims.values[soa_index(B3, cell)];
    let pressure: f32 = params.gam_m1 * u;

    let g_tt: f32 = load_metric(met_idx, 0u);
    let g_rr: f32 = load_metric(met_idx, 1u);
    let g_thth: f32 = load_metric(met_idx, 2u);
    let g_phph: f32 = load_metric(met_idx, 3u);
    let g_tph: f32 = load_metric(met_idx, 4u);
    let sg: f32 = sqrt_g.values[met_idx];

    let vsq: f32 = g_rr * v1 * v1 + g_thth * v2 * v2 + g_phph * v3 * v3;
    let alpha_sq: f32 = -(g_tt + 2.0 * g_tph * v3 + vsq);
    var ut: f32 = 1.0;
    if (alpha_sq > 1.0e-20) {
        ut = inverseSqrt(alpha_sq);
    }
    let inv_ut: f32 = 1.0 / ut;

    let u_r: f32 = g_rr * ut * v1;
    let u_th: f32 = g_thth * ut * v2;
    let u_ph: f32 = g_phph * ut * v3 + g_tph * ut;
    let u_t: f32 = g_tt * ut + g_tph * ut * v3;

    let bt: f32 = (b1 * u_r + b2 * u_th + b3 * u_ph) * inv_ut;
    let bsq_raw: f32 = (b1 * b1 * g_rr + b2 * b2 * g_thth + b3 * b3 * g_phph) / (ut * ut)
        + bt * bt * (-1.0 / (ut * ut) + vsq);
    let bsq: f32 = max(bsq_raw, 0.0);
    let w: f32 = rho + u + pressure + bsq;
    let ptot: f32 = pressure + 0.5 * bsq;

    cons.values[soa_index(0u, cell)] = sg * rho * ut;
    let t_t_t: f32 = w * ut * u_t + ptot - bt * (g_tt * bt);
    cons.values[soa_index(1u, cell)] = sg * (t_t_t + rho * ut);
    cons.values[soa_index(2u, cell)] = sg * (w * ut * u_r);
    cons.values[soa_index(3u, cell)] = sg * (w * ut * u_th);
    cons.values[soa_index(4u, cell)] = sg * (w * ut * u_ph);
    cons.values[soa_index(5u, cell)] = sg * b1;
    cons.values[soa_index(6u, cell)] = sg * b2;
    cons.values[soa_index(7u, cell)] = sg * b3;
}

fn compute_flux(cell: u32) {
    if (cell >= params.n_total) {
        return;
    }

    let dir: u32 = params.dir;
    let met_idx: u32 = metric_index(cell);
    let rho: f32 = prims.values[soa_index(RHO, cell)];
    let u: f32 = prims.values[soa_index(UU, cell)];
    let v1: f32 = prims.values[soa_index(V1, cell)];
    let v2: f32 = prims.values[soa_index(V2, cell)];
    let v3: f32 = prims.values[soa_index(V3, cell)];
    let b1: f32 = prims.values[soa_index(B1, cell)];
    let b2: f32 = prims.values[soa_index(B2, cell)];
    let b3: f32 = prims.values[soa_index(B3, cell)];
    let pressure: f32 = params.gam_m1 * u;

    let g_tt: f32 = load_metric(met_idx, 0u);
    let g_rr: f32 = load_metric(met_idx, 1u);
    let g_thth: f32 = load_metric(met_idx, 2u);
    let g_phph: f32 = load_metric(met_idx, 3u);
    let g_tph: f32 = load_metric(met_idx, 4u);
    let sg: f32 = sqrt_g.values[met_idx];

    let vsq: f32 = g_rr * v1 * v1 + g_thth * v2 * v2 + g_phph * v3 * v3;
    let alpha_sq: f32 = -(g_tt + 2.0 * g_tph * v3 + vsq);
    var ut: f32 = 1.0;
    if (alpha_sq > 1.0e-20) {
        ut = inverseSqrt(alpha_sq);
    }
    let inv_ut: f32 = 1.0 / ut;
    let inv_ut_sq: f32 = inv_ut * inv_ut;

    let u_r: f32 = g_rr * ut * v1;
    let u_th: f32 = g_thth * ut * v2;
    let u_ph: f32 = g_phph * ut * v3 + g_tph * ut;
    let u_t: f32 = g_tt * ut + g_tph * ut * v3;

    let bt: f32 = (b1 * u_r + b2 * u_th + b3 * u_ph) * inv_ut;
    let bsq_raw: f32 = (b1 * b1 * g_rr + b2 * b2 * g_thth + b3 * b3 * g_phph) * inv_ut_sq
        + bt * bt * (-inv_ut_sq + vsq);
    let bsq: f32 = max(bsq_raw, 0.0);
    let w: f32 = rho + u + pressure + bsq;
    let ptot: f32 = pressure + 0.5 * bsq;

    let v_dir: f32 = select3(v1, v2, v3, dir);
    let b_dir: f32 = select3(b1, b2, b3, dir);
    let u_up_dir: f32 = ut * v_dir;
    let b_up_dir: f32 = (b_dir + bt * ut * v_dir) * inv_ut;

    flux.values[soa_index(0u, cell)] = sg * rho * u_up_dir;
    var b_cov_t: f32 = g_tt * bt;
    if (dir == 2u) {
        b_cov_t = b_cov_t + g_tph * b_up_dir;
    }
    let t_dir_t: f32 = w * u_up_dir * u_t - b_up_dir * b_cov_t;
    flux.values[soa_index(1u, cell)] = sg * (t_dir_t + rho * u_up_dir);

    for (var jj: u32 = 0u; jj < 3u; jj = jj + 1u) {
        let v_j: f32 = select3(v1, v2, v3, jj);
        let b_j: f32 = select3(b1, b2, b3, jj);
        let u_cov_j: f32 = select3(u_r, u_th, u_ph, jj);
        let g_diag_j: f32 = select3(g_rr, g_thth, g_phph, jj);
        let b_up_j: f32 = (b_j + bt * ut * v_j) * inv_ut;
        let b_cov_j: f32 = g_diag_j * b_up_j;
        var delta_jd: f32 = 0.0;
        if (jj == dir) {
            delta_jd = 1.0;
        }
        flux.values[soa_index(2u + jj, cell)] =
            sg * (w * u_up_dir * u_cov_j + ptot * delta_jd - b_up_dir * b_cov_j);
    }

    for (var jj2: u32 = 0u; jj2 < 3u; jj2 = jj2 + 1u) {
        let v_j2: f32 = select3(v1, v2, v3, jj2);
        let b_j2: f32 = select3(b1, b2, b3, jj2);
        flux.values[soa_index(5u + jj2, cell)] = sg * (v_dir * b_j2 - v_j2 * b_dir);
    }
}

fn flux_divergence(cell: u32) {
    if (cell >= params.n_total) {
        return;
    }

    let dir: u32 = params.dir;
    var tmp: u32 = cell;
    let k: u32 = tmp % params.n3;
    tmp = tmp / params.n3;
    let j: u32 = tmp % params.n2;
    tmp = tmp / params.n2;
    let i: u32 = tmp;

    var at_boundary: bool = false;
    var idx_m: u32 = cell;
    var idx_p: u32 = cell;

    if (dir == 0u) {
        let stride: u32 = params.n2 * params.n3;
        at_boundary = (i == 0u || i >= params.n1 - 1u);
        if (!at_boundary) {
            idx_m = cell - stride;
            idx_p = cell + stride;
        }
    } else if (dir == 1u) {
        let stride: u32 = params.n3;
        at_boundary = (j == 0u || j >= params.n2 - 1u);
        if (!at_boundary) {
            idx_m = cell - stride;
            idx_p = cell + stride;
        }
    } else {
        if (k == 0u) {
            idx_m = cell + params.n3 - 1u;
        } else {
            idx_m = cell - 1u;
        }
        if (k == params.n3 - 1u) {
            idx_p = cell + 1u - params.n3;
        } else {
            idx_p = cell + 1u;
        }
    }

    if (at_boundary) {
        return;
    }

    let inv_dx: f32 = select3(1.0 / params.dx1, 1.0 / params.dx2, 1.0 / params.dx3, dir);
    for (var var_idx: u32 = 0u; var_idx < NCONS; var_idx = var_idx + 1u) {
        let offset: u32 = var_idx * params.n_total;
        let f_p: f32 = flux.values[offset + idx_p];
        let f_m: f32 = flux.values[offset + idx_m];
        rhs.values[offset + cell] = rhs.values[offset + cell] - 0.5 * inv_dx * (f_p - f_m);
    }
}

fn euler_update(idx: u32) {
    let total_vars: u32 = NCONS * params.n_total;
    if (idx >= total_vars) {
        return;
    }
    cons_new.values[idx] = cons.values[idx] + params.dt * rhs.values[idx];
}

@compute @workgroup_size(256)
fn grmhd_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    if (params.op == 0u) {
        precompute_metric(idx);
    } else if (params.op == 1u) {
        prim2con(idx);
    } else if (params.op == 2u) {
        compute_flux(idx);
    } else if (params.op == 3u) {
        flux_divergence(idx);
    } else if (params.op == 4u) {
        euler_update(idx);
    }
}
"#;

#[derive(Clone, Copy, Debug)]
pub struct GrmhdVulkanConfig {
    pub n1: usize,
    pub n2: usize,
    pub n3: usize,
    pub r_min: f32,
    pub r_max: f32,
    pub kerr_a: f32,
    pub gamma: f32,
}

impl GrmhdVulkanConfig {
    pub fn new(
        n1: usize,
        n2: usize,
        n3: usize,
        r_min: f32,
        r_max: f32,
        kerr_a: f32,
        gamma: f32,
    ) -> Result<Self, String> {
        let config = Self {
            n1,
            n2,
            n3,
            r_min,
            r_max,
            kerr_a,
            gamma,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn n_total(&self) -> usize {
        self.n1 * self.n2 * self.n3
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.n1 < 2 || self.n2 < 2 || self.n3 == 0 {
            return Err(format!(
                "GRMHD grid must satisfy n1>=2, n2>=2, n3>=1, got {}x{}x{}",
                self.n1, self.n2, self.n3
            ));
        }
        let n_total = (self.n1 as u64)
            .checked_mul(self.n2 as u64)
            .and_then(|v| v.checked_mul(self.n3 as u64))
            .ok_or_else(|| "GRMHD grid cell count overflows u64".to_string())?;
        if n_total > u32::MAX as u64 {
            return Err(format!(
                "GRMHD grid cell count exceeds u32 indexing: {n_total}"
            ));
        }
        if !(self.r_min.is_finite() && self.r_min > 0.0) {
            return Err(format!(
                "r_min must be finite and positive, got {}",
                self.r_min
            ));
        }
        if !(self.r_max.is_finite() && self.r_max > self.r_min) {
            return Err(format!(
                "r_max must be finite and greater than r_min, got {} <= {}",
                self.r_max, self.r_min
            ));
        }
        if !(self.kerr_a.is_finite() && self.kerr_a.abs() <= 1.0) {
            return Err(format!(
                "kerr_a must be finite with |a| <= 1, got {}",
                self.kerr_a
            ));
        }
        if !(self.gamma.is_finite() && self.gamma > 1.0) {
            return Err(format!("gamma must be finite and > 1, got {}", self.gamma));
        }
        Ok(())
    }

    pub(crate) fn dx(&self) -> [f32; 3] {
        [
            (self.r_max / self.r_min).ln() / self.n1 as f32,
            std::f32::consts::PI / self.n2 as f32,
            2.0 * std::f32::consts::PI / self.n3 as f32,
        ]
    }
}

pub struct GrmhdVulkanPipeline {
    pipeline: ComputePipeline,
    descriptor_layout: DescriptorSetLayout,
}

pub struct GrmhdVulkanKernel;

impl GrmhdVulkanKernel {
    pub fn wgsl_source() -> &'static str {
        GRMHD_VULKAN_WGSL
    }

    pub fn is_available() -> bool {
        match Self::build_context() {
            Ok((_instance, _adapter, _device)) => true,
            Err(_) => false,
        }
    }

    pub fn advance_conserved(
        config: GrmhdVulkanConfig,
        prims_soa: &[f32],
        dt: f32,
        steps: usize,
    ) -> Result<Vec<f32>, String> {
        config.validate()?;
        validate_run_inputs(config, prims_soa, dt, steps)?;

        let (_instance, adapter, device) = Self::build_context()?;
        let pipeline = Self::build_pipeline(&device)?;
        let run = GrmhdVulkanRun::new(&device, &adapter, &pipeline, config, prims_soa)?;
        let dispatch = DispatchScope::new(&device)
            .map_err(|e| format!("GRMHD dispatch scope creation failed: {e}"))?;
        run.dispatch_op(
            &dispatch,
            &pipeline.pipeline,
            DispatchParams {
                op: OP_PRECOMPUTE_METRIC,
                dir: 0,
                dt,
                work_items: config.n1 * config.n2,
            },
        )?;

        for _ in 0..steps {
            run.rhs_buffer
                .write_f32_slice(&run.zero_cons)
                .map_err(|e| format!("GRMHD RHS reset failed: {e}"))?;
            run.dispatch_op(
                &dispatch,
                &pipeline.pipeline,
                DispatchParams {
                    op: OP_PRIM2CON,
                    dir: 0,
                    dt,
                    work_items: config.n_total(),
                },
            )?;
            for dir in 0..3u32 {
                run.dispatch_op(
                    &dispatch,
                    &pipeline.pipeline,
                    DispatchParams {
                        op: OP_COMPUTE_FLUX,
                        dir,
                        dt,
                        work_items: config.n_total(),
                    },
                )?;
                run.dispatch_op(
                    &dispatch,
                    &pipeline.pipeline,
                    DispatchParams {
                        op: OP_FLUX_DIVERGENCE,
                        dir,
                        dt,
                        work_items: config.n_total(),
                    },
                )?;
            }
            run.dispatch_op(
                &dispatch,
                &pipeline.pipeline,
                DispatchParams {
                    op: OP_EULER_UPDATE,
                    dir: 0,
                    dt,
                    work_items: NCONS * config.n_total(),
                },
            )?;
        }

        run.cons_new_buffer
            .read_f32_slice(NCONS * config.n_total())
            .map_err(|e| format!("GRMHD conservative readback failed: {e}"))
    }

    pub fn build_pipeline(device: &Device) -> Result<GrmhdVulkanPipeline, String> {
        let shader = ShaderModule::from_wgsl(device, GRMHD_VULKAN_WGSL, GRMHD_VULKAN_ENTRY_POINT)
            .map_err(|e| format!("GRMHD WGSL compile failed: {e}"))?;
        let descriptor_layout = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .storage_buffer(1)
            .storage_buffer(2)
            .storage_buffer(3)
            .storage_buffer(4)
            .storage_buffer(5)
            .storage_buffer(6)
            .uniform_buffer(7)
            .build(device)
            .map_err(|e| format!("GRMHD descriptor layout failed: {e}"))?;
        let pipeline = ComputePipelineBuilder::new(device, &shader)
            .descriptor_layout(&descriptor_layout)
            .build()
            .map_err(|e| format!("GRMHD compute pipeline build failed: {e}"))?;

        Ok(GrmhdVulkanPipeline {
            pipeline,
            descriptor_layout,
        })
    }

    fn build_context() -> Result<(Instance, Adapter, Device), String> {
        let instance = InstanceBuilder::new("grmhd_core_vulkan")
            .validation(ValidationPolicy::Disable)
            .build()
            .map_err(|e| format!("GRMHD Vulkan instance creation failed: {e}"))?;
        let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
            .map_err(|e| format!("GRMHD Vulkan adapter pick failed: {e}"))?;
        let device = DeviceBuilder::new(adapter.clone())
            .build(&instance)
            .map_err(|e| format!("GRMHD Vulkan device creation failed: {e}"))?;
        Ok((instance, adapter, device))
    }
}

pub fn advance_conserved_cpu_reference(
    config: GrmhdVulkanConfig,
    prims_soa: &[f32],
    dt: f32,
    steps: usize,
) -> Result<Vec<f32>, String> {
    config.validate()?;
    validate_run_inputs(config, prims_soa, dt, steps)?;

    let n_total = config.n_total();
    let mut gcov = vec![0.0f32; 5 * config.n1 * config.n2];
    let mut sqrt_g = vec![0.0f32; config.n1 * config.n2];
    precompute_metric_cpu(config, &mut gcov, &mut sqrt_g);

    let mut cons = vec![0.0f32; NCONS * n_total];
    let mut cons_new = vec![0.0f32; NCONS * n_total];
    let mut flux = vec![0.0f32; NCONS * n_total];
    let mut rhs = vec![0.0f32; NCONS * n_total];

    for _ in 0..steps {
        rhs.fill(0.0);
        prim2con_cpu(config, prims_soa, &gcov, &sqrt_g, &mut cons);
        for dir in 0..3 {
            compute_flux_cpu(config, prims_soa, &gcov, &sqrt_g, dir, &mut flux);
            flux_divergence_cpu(config, &flux, dir, &mut rhs);
        }
        for idx in 0..cons.len() {
            cons_new[idx] = cons[idx] + dt * rhs[idx];
        }
    }

    Ok(cons_new)
}

#[derive(Clone, Copy)]
struct DispatchParams {
    op: u32,
    dir: u32,
    dt: f32,
    work_items: usize,
}

struct GrmhdVulkanRun {
    config: GrmhdVulkanConfig,
    #[allow(dead_code)]
    descriptor_pool: DescriptorPool,
    descriptor_set: DescriptorSet,
    #[allow(dead_code)]
    prims_buffer: HostVisibleBuffer,
    #[allow(dead_code)]
    cons_buffer: HostVisibleBuffer,
    cons_new_buffer: HostVisibleBuffer,
    #[allow(dead_code)]
    flux_buffer: HostVisibleBuffer,
    rhs_buffer: HostVisibleBuffer,
    #[allow(dead_code)]
    gcov_buffer: HostVisibleBuffer,
    #[allow(dead_code)]
    sqrt_buffer: HostVisibleBuffer,
    params_buffer: HostVisibleBuffer,
    zero_cons: Vec<f32>,
}

impl GrmhdVulkanRun {
    fn new(
        device: &Device,
        adapter: &Adapter,
        pipeline: &GrmhdVulkanPipeline,
        config: GrmhdVulkanConfig,
        prims_soa: &[f32],
    ) -> Result<Self, String> {
        let descriptor_pool = DescriptorPool::for_layout(device, &pipeline.descriptor_layout, 1)
            .map_err(|e| format!("GRMHD descriptor pool allocation failed: {e}"))?;
        let descriptor_set = descriptor_pool
            .allocate_set(&pipeline.descriptor_layout)
            .map_err(|e| format!("GRMHD descriptor set allocation failed: {e}"))?;

        let buffers = GrmhdVulkanBuffers::new(device, adapter, config)?;
        let zero_cons = buffers.initialize(prims_soa, config)?;
        buffers.write_descriptors(&descriptor_set);

        Ok(Self {
            config,
            descriptor_pool,
            descriptor_set,
            prims_buffer: buffers.prims_buffer,
            cons_buffer: buffers.cons_buffer,
            cons_new_buffer: buffers.cons_new_buffer,
            flux_buffer: buffers.flux_buffer,
            rhs_buffer: buffers.rhs_buffer,
            gcov_buffer: buffers.gcov_buffer,
            sqrt_buffer: buffers.sqrt_buffer,
            params_buffer: buffers.params_buffer,
            zero_cons,
        })
    }

    fn dispatch_op(
        &self,
        dispatch: &DispatchScope,
        pipeline: &ComputePipeline,
        params: DispatchParams,
    ) -> Result<(), String> {
        let groups = u32::try_from(params.work_items)
            .map_err(|_| {
                format!(
                    "GRMHD dispatch item count exceeds u32: {}",
                    params.work_items
                )
            })?
            .div_ceil(WORKGROUP_SIZE);
        let bytes = encode_params(self.config, params.dt, params.dir, params.op)?;
        self.params_buffer
            .write_bytes(&bytes)
            .map_err(|e| format!("GRMHD params upload failed: {e}"))?;
        dispatch
            .dispatch(
                pipeline,
                self.descriptor_set.raw(),
                groups,
                1,
                1,
                DISPATCH_TIMEOUT_NS,
            )
            .map_err(|e| format!("GRMHD Vulkan dispatch failed: {e}"))
    }
}

struct GrmhdVulkanBuffers {
    prims_buffer: HostVisibleBuffer,
    cons_buffer: HostVisibleBuffer,
    cons_new_buffer: HostVisibleBuffer,
    flux_buffer: HostVisibleBuffer,
    rhs_buffer: HostVisibleBuffer,
    gcov_buffer: HostVisibleBuffer,
    sqrt_buffer: HostVisibleBuffer,
    params_buffer: HostVisibleBuffer,
}

impl GrmhdVulkanBuffers {
    fn new(device: &Device, adapter: &Adapter, config: GrmhdVulkanConfig) -> Result<Self, String> {
        let n_total = config.n_total();
        let cons_len = NCONS * n_total;
        Ok(Self {
            prims_buffer: allocate_storage_f32(device, adapter, NPRIM * n_total, "primitive")?,
            cons_buffer: allocate_storage_f32(device, adapter, cons_len, "conservative")?,
            cons_new_buffer: allocate_storage_f32(device, adapter, cons_len, "new conservative")?,
            flux_buffer: allocate_storage_f32(device, adapter, cons_len, "flux")?,
            rhs_buffer: allocate_storage_f32(device, adapter, cons_len, "RHS")?,
            gcov_buffer: allocate_storage_f32(
                device,
                adapter,
                5 * config.n1 * config.n2,
                "metric",
            )?,
            sqrt_buffer: allocate_storage_f32(device, adapter, config.n1 * config.n2, "sqrt_g")?,
            params_buffer: HostVisibleBuffer::uniform(device, adapter, 64)
                .map_err(|e| format!("GRMHD params buffer allocation failed: {e}"))?,
        })
    }

    fn initialize(&self, prims_soa: &[f32], config: GrmhdVulkanConfig) -> Result<Vec<f32>, String> {
        self.prims_buffer
            .write_f32_slice(prims_soa)
            .map_err(|e| format!("GRMHD primitive upload failed: {e}"))?;
        let zero_cons = vec![0.0f32; NCONS * config.n_total()];
        self.cons_buffer
            .write_f32_slice(&zero_cons)
            .map_err(|e| format!("GRMHD conservative initialization failed: {e}"))?;
        self.cons_new_buffer
            .write_f32_slice(&zero_cons)
            .map_err(|e| format!("GRMHD new conservative initialization failed: {e}"))?;
        self.flux_buffer
            .write_f32_slice(&zero_cons)
            .map_err(|e| format!("GRMHD flux initialization failed: {e}"))?;
        self.rhs_buffer
            .write_f32_slice(&zero_cons)
            .map_err(|e| format!("GRMHD RHS initialization failed: {e}"))?;
        Ok(zero_cons)
    }

    fn write_descriptors(&self, descriptor_set: &DescriptorSet) {
        descriptor_set.write_storage_buffer(0, &self.prims_buffer);
        descriptor_set.write_storage_buffer(1, &self.cons_buffer);
        descriptor_set.write_storage_buffer(2, &self.cons_new_buffer);
        descriptor_set.write_storage_buffer(3, &self.flux_buffer);
        descriptor_set.write_storage_buffer(4, &self.rhs_buffer);
        descriptor_set.write_storage_buffer(5, &self.gcov_buffer);
        descriptor_set.write_storage_buffer(6, &self.sqrt_buffer);
        descriptor_set.write_uniform_buffer(7, &self.params_buffer);
    }
}

fn allocate_storage_f32(
    device: &Device,
    adapter: &Adapter,
    len: usize,
    label: &str,
) -> Result<HostVisibleBuffer, String> {
    let bytes = byte_len::<f32>(len, &format!("GRMHD {label} buffer"))?;
    HostVisibleBuffer::storage(device, adapter, bytes)
        .map_err(|e| format!("GRMHD {label} buffer allocation failed: {e}"))
}

fn encode_params(
    config: GrmhdVulkanConfig,
    dt: f32,
    dir: u32,
    op: u32,
) -> Result<[u8; 64], String> {
    let dx = config.dx();
    let n1 = u32::try_from(config.n1).map_err(|_| "n1 does not fit u32".to_string())?;
    let n2 = u32::try_from(config.n2).map_err(|_| "n2 does not fit u32".to_string())?;
    let n3 = u32::try_from(config.n3).map_err(|_| "n3 does not fit u32".to_string())?;
    let n_total = u32::try_from(config.n_total())
        .map_err(|_| "GRMHD cell count does not fit u32".to_string())?;
    let words = [
        n1.to_le_bytes(),
        n2.to_le_bytes(),
        n3.to_le_bytes(),
        n_total.to_le_bytes(),
        config.r_min.to_le_bytes(),
        config.r_max.to_le_bytes(),
        config.kerr_a.to_le_bytes(),
        (config.gamma - 1.0).to_le_bytes(),
        dx[0].to_le_bytes(),
        dx[1].to_le_bytes(),
        dx[2].to_le_bytes(),
        dt.to_le_bytes(),
        dir.to_le_bytes(),
        op.to_le_bytes(),
        0u32.to_le_bytes(),
        0u32.to_le_bytes(),
    ];
    let mut bytes = [0u8; 64];
    for (idx, word) in words.iter().enumerate() {
        let start = 4 * idx;
        bytes[start..start + 4].copy_from_slice(word);
    }
    Ok(bytes)
}

fn validate_run_inputs(
    config: GrmhdVulkanConfig,
    prims_soa: &[f32],
    dt: f32,
    steps: usize,
) -> Result<(), String> {
    let expected = NPRIM * config.n_total();
    if prims_soa.len() != expected {
        return Err(format!(
            "primitive SoA length mismatch: got {}, expected {}",
            prims_soa.len(),
            expected
        ));
    }
    if !(dt.is_finite() && dt > 0.0) {
        return Err(format!("dt must be finite and positive, got {dt}"));
    }
    if steps == 0 {
        return Err("steps must be at least 1".to_string());
    }
    Ok(())
}

fn byte_len<T>(len: usize, label: &str) -> Result<u64, String> {
    len.checked_mul(std::mem::size_of::<T>())
        .map(|bytes| bytes as u64)
        .ok_or_else(|| format!("{label} byte length overflows usize"))
}

fn precompute_metric_cpu(config: GrmhdVulkanConfig, gcov: &mut [f32], sqrt_g: &mut [f32]) {
    for i in 0..config.n1 {
        for j in 0..config.n2 {
            let idx = i * config.n2 + j;
            let xi = i as f32 / (config.n1 - 1) as f32;
            let r = config.r_min * (xi * (config.r_max / config.r_min).ln()).exp();
            let th = std::f32::consts::PI * (j as f32 + 0.5) / config.n2 as f32;
            let sth = th.sin();
            let cth = th.cos();
            let a = config.kerr_a;
            let sigma = r * r + a * a * cth * cth;
            let delta = r * r - 2.0 * r + a * a;
            let a_met = (r * r + a * a) * (r * r + a * a) - delta * a * a * sth * sth;
            let base = 5 * idx;
            gcov[base] = -(1.0 - 2.0 * r / sigma);
            gcov[base + 1] = sigma / delta;
            gcov[base + 2] = sigma;
            gcov[base + 3] = a_met * sth * sth / sigma;
            gcov[base + 4] = -2.0 * a * r * sth * sth / sigma;
            sqrt_g[idx] = sigma * sth.abs();
        }
    }
}

fn prim2con_cpu(
    config: GrmhdVulkanConfig,
    prims_soa: &[f32],
    gcov: &[f32],
    sqrt_g: &[f32],
    cons: &mut [f32],
) {
    let n_total = config.n_total();
    for cell in 0..n_total {
        let met_idx = metric_index_cpu(config, cell);
        let rho = prims_soa[soa_index_cpu(n_total, 0, cell)];
        let u = prims_soa[soa_index_cpu(n_total, 1, cell)];
        let v1 = prims_soa[soa_index_cpu(n_total, 2, cell)];
        let v2 = prims_soa[soa_index_cpu(n_total, 3, cell)];
        let v3 = prims_soa[soa_index_cpu(n_total, 4, cell)];
        let b1 = prims_soa[soa_index_cpu(n_total, 5, cell)];
        let b2 = prims_soa[soa_index_cpu(n_total, 6, cell)];
        let b3 = prims_soa[soa_index_cpu(n_total, 7, cell)];
        let pressure = (config.gamma - 1.0) * u;
        let base = 5 * met_idx;
        let g_tt = gcov[base];
        let g_rr = gcov[base + 1];
        let g_thth = gcov[base + 2];
        let g_phph = gcov[base + 3];
        let g_tph = gcov[base + 4];
        let sg = sqrt_g[met_idx];

        let vsq = g_rr * v1 * v1 + g_thth * v2 * v2 + g_phph * v3 * v3;
        let alpha_sq = -(g_tt + 2.0 * g_tph * v3 + vsq);
        let ut = if alpha_sq > 1.0e-20 {
            alpha_sq.sqrt().recip()
        } else {
            1.0
        };
        let inv_ut = 1.0 / ut;
        let u_r = g_rr * ut * v1;
        let u_th = g_thth * ut * v2;
        let u_ph = g_phph * ut * v3 + g_tph * ut;
        let u_t = g_tt * ut + g_tph * ut * v3;
        let bt = (b1 * u_r + b2 * u_th + b3 * u_ph) * inv_ut;
        let bsq = ((b1 * b1 * g_rr + b2 * b2 * g_thth + b3 * b3 * g_phph) / (ut * ut)
            + bt * bt * (-1.0 / (ut * ut) + vsq))
            .max(0.0);
        let w = rho + u + pressure + bsq;
        let ptot = pressure + 0.5 * bsq;

        cons[soa_index_cpu(n_total, 0, cell)] = sg * rho * ut;
        let t_t_t = w * ut * u_t + ptot - bt * (g_tt * bt);
        cons[soa_index_cpu(n_total, 1, cell)] = sg * (t_t_t + rho * ut);
        cons[soa_index_cpu(n_total, 2, cell)] = sg * (w * ut * u_r);
        cons[soa_index_cpu(n_total, 3, cell)] = sg * (w * ut * u_th);
        cons[soa_index_cpu(n_total, 4, cell)] = sg * (w * ut * u_ph);
        cons[soa_index_cpu(n_total, 5, cell)] = sg * b1;
        cons[soa_index_cpu(n_total, 6, cell)] = sg * b2;
        cons[soa_index_cpu(n_total, 7, cell)] = sg * b3;
    }
}

fn compute_flux_cpu(
    config: GrmhdVulkanConfig,
    prims_soa: &[f32],
    gcov: &[f32],
    sqrt_g: &[f32],
    dir: usize,
    flux: &mut [f32],
) {
    let n_total = config.n_total();
    for cell in 0..n_total {
        let met_idx = metric_index_cpu(config, cell);
        let rho = prims_soa[soa_index_cpu(n_total, 0, cell)];
        let u = prims_soa[soa_index_cpu(n_total, 1, cell)];
        let v = [
            prims_soa[soa_index_cpu(n_total, 2, cell)],
            prims_soa[soa_index_cpu(n_total, 3, cell)],
            prims_soa[soa_index_cpu(n_total, 4, cell)],
        ];
        let b = [
            prims_soa[soa_index_cpu(n_total, 5, cell)],
            prims_soa[soa_index_cpu(n_total, 6, cell)],
            prims_soa[soa_index_cpu(n_total, 7, cell)],
        ];
        let pressure = (config.gamma - 1.0) * u;
        let base = 5 * met_idx;
        let g_tt = gcov[base];
        let g_diag = [gcov[base + 1], gcov[base + 2], gcov[base + 3]];
        let g_tph = gcov[base + 4];
        let sg = sqrt_g[met_idx];
        let vsq = g_diag[0] * v[0] * v[0] + g_diag[1] * v[1] * v[1] + g_diag[2] * v[2] * v[2];
        let alpha_sq = -(g_tt + 2.0 * g_tph * v[2] + vsq);
        let ut = if alpha_sq > 1.0e-20 {
            alpha_sq.sqrt().recip()
        } else {
            1.0
        };
        let inv_ut = 1.0 / ut;
        let inv_ut_sq = inv_ut * inv_ut;
        let u_cov = [
            g_diag[0] * ut * v[0],
            g_diag[1] * ut * v[1],
            g_diag[2] * ut * v[2] + g_tph * ut,
        ];
        let u_t = g_tt * ut + g_tph * ut * v[2];
        let bt = (b[0] * u_cov[0] + b[1] * u_cov[1] + b[2] * u_cov[2]) * inv_ut;
        let bsq = ((b[0] * b[0] * g_diag[0] + b[1] * b[1] * g_diag[1] + b[2] * b[2] * g_diag[2])
            * inv_ut_sq
            + bt * bt * (-inv_ut_sq + vsq))
            .max(0.0);
        let w = rho + u + pressure + bsq;
        let ptot = pressure + 0.5 * bsq;
        let v_dir = v[dir];
        let b_dir = b[dir];
        let u_up_dir = ut * v_dir;
        let b_up_dir = (b_dir + bt * ut * v_dir) * inv_ut;

        flux[soa_index_cpu(n_total, 0, cell)] = sg * rho * u_up_dir;
        let mut b_cov_t = g_tt * bt;
        if dir == 2 {
            b_cov_t += g_tph * b_up_dir;
        }
        let t_dir_t = w * u_up_dir * u_t - b_up_dir * b_cov_t;
        flux[soa_index_cpu(n_total, 1, cell)] = sg * (t_dir_t + rho * u_up_dir);
        for jj in 0..3 {
            let b_up_j = (b[jj] + bt * ut * v[jj]) * inv_ut;
            let b_cov_j = g_diag[jj] * b_up_j;
            let delta_jd = if jj == dir { 1.0 } else { 0.0 };
            flux[soa_index_cpu(n_total, 2 + jj, cell)] =
                sg * (w * u_up_dir * u_cov[jj] + ptot * delta_jd - b_up_dir * b_cov_j);
        }
        for jj in 0..3 {
            flux[soa_index_cpu(n_total, 5 + jj, cell)] = sg * (v_dir * b[jj] - v[jj] * b_dir);
        }
    }
}

fn flux_divergence_cpu(config: GrmhdVulkanConfig, flux: &[f32], dir: usize, rhs: &mut [f32]) {
    let n_total = config.n_total();
    let dx = config.dx();
    let inv_dx = 1.0 / dx[dir];
    for cell in 0..n_total {
        let (i, j, k) = decode_cell_cpu(config, cell);
        let (at_boundary, idx_m, idx_p) = match dir {
            0 => {
                let stride = config.n2 * config.n3;
                if i == 0 || i >= config.n1 - 1 {
                    (true, cell, cell)
                } else {
                    (false, cell - stride, cell + stride)
                }
            }
            1 => {
                let stride = config.n3;
                if j == 0 || j >= config.n2 - 1 {
                    (true, cell, cell)
                } else {
                    (false, cell - stride, cell + stride)
                }
            }
            _ => {
                let idx_m = if k == 0 {
                    cell + config.n3 - 1
                } else {
                    cell - 1
                };
                let idx_p = if k == config.n3 - 1 {
                    cell + 1 - config.n3
                } else {
                    cell + 1
                };
                (false, idx_m, idx_p)
            }
        };
        if at_boundary {
            continue;
        }
        for var_idx in 0..NCONS {
            let offset = var_idx * n_total;
            rhs[offset + cell] -= 0.5 * inv_dx * (flux[offset + idx_p] - flux[offset + idx_m]);
        }
    }
}

fn decode_cell_cpu(config: GrmhdVulkanConfig, cell: usize) -> (usize, usize, usize) {
    let k = cell % config.n3;
    let tmp = cell / config.n3;
    let j = tmp % config.n2;
    let i = tmp / config.n2;
    (i, j, k)
}

fn metric_index_cpu(config: GrmhdVulkanConfig, cell: usize) -> usize {
    let (i, j, _) = decode_cell_cpu(config, cell);
    i * config.n2 + j
}

fn soa_index_cpu(n_total: usize, channel: usize, cell: usize) -> usize {
    channel * n_total + cell
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn fixture_config() -> GrmhdVulkanConfig {
        GrmhdVulkanConfig::new(6, 5, 4, 2.5, 20.0, 0.0, 4.0 / 3.0).unwrap()
    }

    fn acceptance_config() -> GrmhdVulkanConfig {
        GrmhdVulkanConfig::new(32, 32, 32, 2.5, 20.0, 0.0, 4.0 / 3.0).unwrap()
    }

    fn fixture_prims(config: GrmhdVulkanConfig) -> Vec<f32> {
        let n_total = config.n_total();
        let mut prims = vec![0.0f32; NPRIM * n_total];
        for cell in 0..n_total {
            let (_, _, k) = decode_cell_cpu(config, cell);
            prims[soa_index_cpu(n_total, 0, cell)] = 1.0 + 0.01 * (cell % 7) as f32;
            prims[soa_index_cpu(n_total, 1, cell)] = 0.02 + 0.001 * (cell % 5) as f32;
            prims[soa_index_cpu(n_total, 2, cell)] = 0.0003 * (cell % 3) as f32;
            prims[soa_index_cpu(n_total, 3, cell)] = -0.0002 * (cell % 4) as f32;
            prims[soa_index_cpu(n_total, 4, cell)] = 0.0001 * k as f32;
            prims[soa_index_cpu(n_total, 5, cell)] = 0.001;
            prims[soa_index_cpu(n_total, 6, cell)] = 0.0005;
            prims[soa_index_cpu(n_total, 7, cell)] = -0.00025;
        }
        prims
    }

    #[test]
    fn grmhd_vulkan_wgsl_parses_and_emits_compute_spirv() {
        let module = naga::front::wgsl::parse_str(GrmhdVulkanKernel::wgsl_source()).unwrap();
        let override_ids: BTreeMap<&str, u32> = module
            .overrides
            .iter()
            .filter_map(|(_, override_constant)| {
                Some((
                    override_constant.name.as_deref()?,
                    u32::from(override_constant.id?),
                ))
            })
            .collect();
        assert!(override_ids.is_empty());
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: GRMHD_VULKAN_ENTRY_POINT.to_string(),
        };
        let spirv = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options::default(),
            Some(&pipeline_options),
        )
        .unwrap();
        assert!(!spirv.is_empty());
    }

    #[test]
    fn cpu_reference_returns_finite_conserved_soa() {
        let config = fixture_config();
        let prims = fixture_prims(config);
        let cons = advance_conserved_cpu_reference(config, &prims, 0.0001, 10).unwrap();
        assert_eq!(cons.len(), NCONS * config.n_total());
        assert!(cons.iter().all(|value| value.is_finite()));
        assert!(cons.iter().any(|value| value.abs() > 0.0));
    }

    #[test]
    fn invalid_inputs_are_rejected_before_dispatch() {
        let config = fixture_config();
        let prims = fixture_prims(config);
        assert!(advance_conserved_cpu_reference(config, &prims, 0.0, 1).is_err());
        assert!(advance_conserved_cpu_reference(config, &prims, 0.1, 0).is_err());
        assert!(
            advance_conserved_cpu_reference(config, &prims[..prims.len() - 1], 0.1, 1).is_err()
        );
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn grmhd_vulkan_matches_cpu_reference_for_cuda_style_advance() {
        let config = acceptance_config();
        let prims = fixture_prims(config);
        let cpu = advance_conserved_cpu_reference(config, &prims, 0.0001, 10).unwrap();
        let gpu = GrmhdVulkanKernel::advance_conserved(config, &prims, 0.0001, 10).unwrap();
        assert_eq!(cpu.len(), gpu.len());
        for (idx, (expected, observed)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let scale = expected.abs().max(1.0);
            let rel = (expected - observed).abs() / scale;
            assert!(
                rel < 1.0e-4,
                "GRMHD Vulkan mismatch at {idx}: cpu={expected}, gpu={observed}, rel={rel}"
            );
        }
    }
}
