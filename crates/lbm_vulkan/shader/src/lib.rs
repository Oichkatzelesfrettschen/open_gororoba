#![no_std]
use spirv_std::spirv;
use spirv_std::glam::{UVec3, Vec3, Vec4, Mat4};

#[derive(Copy, Clone)]
#[repr(C)]
pub struct LbmPushConstants {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub global_tau_scale: f32,
}

const CX: [i32; 19] = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
const CY: [i32; 19] = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1];
const CZ: [i32; 19] = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1];
const WF: [f32; 19] = [
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
];

#[spirv(compute(threads(8, 8, 8)))]
pub fn lbm_step(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] pc: &LbmPushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] f_in: &[f32],
    #[spirv(descriptor_set = 0, binding = 1)] f_out: &mut [f32],
    #[spirv(descriptor_set = 0, binding = 2)] rho_out: &mut [f32],
    #[spirv(descriptor_set = 0, binding = 3)] u_out: &mut [f32],
    #[spirv(descriptor_set = 0, binding = 4)] tau_in: &[f32],
    #[spirv(descriptor_set = 0, binding = 5)] force_in: &[f32],
    #[spirv(descriptor_set = 0, binding = 6)] entropy_out: &mut [f32],
) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if x >= pc.nx || y >= pc.ny || z >= pc.nz {
        return;
    }

    let idx = (x + pc.nx * (y + pc.ny * z)) as usize;
    
    // 1. Macroscopic
    let mut rho = 0.0;
    let mut momentum = Vec3::ZERO;
    let mut f_local = [0.0; 19];

    for i in 0..19 {
        let val = f_in[idx * 19 + i];
        f_local[i] = if val > 0.0 { val } else { 0.0 };
        rho += f_local[i];
        momentum += Vec3::new(CX[i] as f32, CY[i] as f32, CZ[i] as i32 as f32) * f_local[i];
    }

    let force = Vec3::new(force_in[idx * 3], force_in[idx * 3 + 1], force_in[idx * 3 + 2]) * 0.01;
    let tau = if tau_in[idx] > 0.55 { tau_in[idx] } else { 0.55 };

    let u = if rho > 1e-6 {
        let vel = (momentum + 0.5 * force) / rho;
        if vel.length() > 0.1 { vel.normalize() * 0.1 } else { vel }
    } else {
        Vec3::ZERO
    };

    rho_out[idx] = rho;
    u_out[idx * 3] = u.x;
    u_out[idx * 3 + 1] = u.y;
    u_out[idx * 3 + 2] = u.z;

    // 2. Collision
    let u_sq = u.dot(u);
    let omega = 1.0 / tau;
    let force_prefactor = 1.0 - 0.5 * omega;
    let mut entropy = 0.0;

    for i in 0..19 {
        let ci = Vec3::new(CX[i] as f32, CY[i] as f32, CZ[i] as f32);
        let cu = ci.dot(u);
        let feq = WF[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
        
        let cf = ci.dot(force);
        let uf = u.dot(force);
        let si = WF[i] * ((cf - uf) * 3.0 + (cu * cf) * 9.0);

        let f_new = f_local[i] * (1.0 - omega) + feq * omega + force_prefactor * si;
        
        let f_neq = f_local[i] - feq;
        entropy += (f_neq * f_neq) / (feq + 1e-9);

        // 3. Streaming
        let nx = pc.nx as i32;
        let ny = pc.ny as i32;
        let nz = pc.nz as i32;
        let next_x = (x as i32 + CX[i] + nx) % nx;
        let next_y = (y as i32 + CY[i] + ny) % ny;
        let next_z = (z as i32 + CZ[i] + nz) % nz;
        let next_idx = (next_x + nx * (next_y + ny * next_z)) as usize;
        
        f_out[next_idx * 19 + i] = f_new;
    }
    entropy_out[idx] = entropy;
}
