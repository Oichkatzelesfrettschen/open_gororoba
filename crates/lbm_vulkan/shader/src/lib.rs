#![no_std]
use spirv_std::{
    glam::{UVec3, Vec2, Vec3, Vec4},
    spirv,
};

#[derive(Copy, Clone)]
#[repr(C)]
pub struct LbmPushConstants {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub global_tau_scale: f32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct RenderPushConstants {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub width: u32,
    pub height: u32,
    pub time: f32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct ZdGenPushConstants {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub tau_base: f32,
    pub tau_amp: f32,
    pub lambda: f32,
}

const CX: [i32; 19] = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
const CY: [i32; 19] = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1];
const CZ: [i32; 19] = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1];
const WF: [f32; 19] = [
    1.0 / 3.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

const ASSESSORS: [u32; 84] = [
    1, 10, 1, 11, 1, 12, 1, 13, 1, 14, 1, 15, 2, 9, 2, 11, 2, 12, 2, 13, 2, 14, 2, 15, 3, 9, 3, 10,
    3, 12, 3, 13, 3, 14, 3, 15, 4, 9, 4, 10, 4, 11, 4, 13, 4, 14, 4, 15, 5, 9, 5, 10, 5, 11, 5, 12,
    5, 14, 5, 15, 6, 9, 6, 10, 6, 11, 6, 12, 6, 13, 6, 15, 7, 9, 7, 10, 7, 11, 7, 12, 7, 13, 7, 14,
];

fn fire_palette(t: f32) -> Vec3 {
    let black = Vec3::ZERO;
    let red = Vec3::new(0.5, 0.0, 0.0);
    let orange = Vec3::new(1.0, 0.5, 0.0);
    let yellow = Vec3::new(1.0, 1.0, 0.5);
    let white = Vec3::ONE;

    if t < 0.25 {
        black.lerp(red, t * 4.0)
    } else if t < 0.5 {
        red.lerp(orange, (t - 0.25) * 4.0)
    } else if t < 0.75 {
        orange.lerp(yellow, (t - 0.5) * 4.0)
    } else {
        yellow.lerp(white, (t - 0.75) * 4.0)
    }
}

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

    let mut rho = 0.0;
    let mut momentum = Vec3::ZERO;
    let mut f_local = [0.0; 19];

    for i in 0..19 {
        let val = f_in[idx * 19 + i];
        f_local[i] = if val > 0.0 { val } else { 0.0 };
        rho += f_local[i];
        momentum += Vec3::new(CX[i] as f32, CY[i] as f32, CZ[i] as f32) * f_local[i];
    }

    let force = Vec3::new(
        force_in[idx * 3],
        force_in[idx * 3 + 1],
        force_in[idx * 3 + 2],
    ) * 0.01;
    let tau = if tau_in[idx] > 0.55 {
        tau_in[idx]
    } else {
        0.55
    };

    let u = if rho > 1e-6 {
        let vel = (momentum + 0.5 * force) / rho;
        if vel.length() > 0.1 {
            vel.normalize() * 0.1
        } else {
            vel
        }
    } else {
        Vec3::ZERO
    };

    rho_out[idx] = rho;
    u_out[idx * 3] = u.x;
    u_out[idx * 3 + 1] = u.y;
    u_out[idx * 3 + 2] = u.z;

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

#[spirv(compute(threads(16, 16)))]
pub fn render_frame(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] pc: &RenderPushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] field: &[f32],
    #[spirv(descriptor_set = 0, binding = 1)] image: &spirv_std::Image!(
        2D,
        format = rgba8,
        sampled = false
    ),
) {
    let x = id.x;
    let y = id.y;

    if x >= pc.width || y >= pc.height {
        return;
    }

    let target = Vec3::new(pc.nx as f32 * 0.5, pc.ny as f32 * 0.5, pc.nz as f32 * 0.5);
    let cam_dist = pc.nx as f32 * 1.2;
    let angle = pc.time * 0.05 + 0.5;
    let ro = target
        + Vec3::new(
            angle.cos() * cam_dist,
            cam_dist * 0.4,
            angle.sin() * cam_dist,
        );

    let ww = (target - ro).normalize();
    let uu = ww.cross(Vec3::Y).normalize();
    let vv = uu.cross(ww).normalize();

    let p = (Vec2::new(x as f32, y as f32) - 0.5 * Vec2::new(pc.width as f32, pc.height as f32))
        / pc.height as f32;
    let rd = (p.x * uu + p.y * vv + 1.5 * ww).normalize();

    let mut t = 0.0;
    let mut color = Vec3::ZERO;
    let mut opacity = 0.0;
    let step_size = 0.8;

    for _ in 0..200 {
        let pos = ro + rd * t;
        if pos.x >= 0.0
            && pos.x < pc.nx as f32
            && pos.y >= 0.0
            && pos.y < pc.ny as f32
            && pos.z >= 0.0
            && pos.z < pc.nz as f32
        {
            let idx = (pos.x as u32 + pc.nx * (pos.y as u32 + pc.ny * pos.z as u32)) as usize;
            let val = field[idx];

            if val > 0.001 {
                let intensity = (val * 10.0).clamp(0.0, 1.0);
                let c = fire_palette(intensity);
                let a = intensity * 0.05;
                color += (1.0 - opacity) * c * a;
                opacity += (1.0 - opacity) * a;
            }
        }
        t += step_size;
        if opacity >= 0.98 {
            break;
        }
    }

    unsafe {
        image.write(id.xy(), Vec4::new(color.x, color.y, color.z, 1.0));
    }
}

#[spirv(compute(threads(8, 8, 8)))]
pub fn generate_zd_field(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] pc: &ZdGenPushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] tau_out: &mut [f32],
) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if x >= pc.nx || y >= pc.ny || z >= pc.nz {
        return;
    }

    let idx = (x + pc.nx * (y + pc.ny * z)) as usize;
    let p = Vec3::new(
        x as f32 / pc.nx as f32,
        y as f32 / pc.ny as f32,
        z as f32 / pc.nz as f32,
    );

    let mut psi = [0.0f32; 16];
    let mut norm_sq = 0.25;
    psi[0] = 0.5;
    for k in 1..16 {
        let fk = k as f32 * 0.5;
        psi[k] = (p.x * fk + fk).sin() * (p.y * fk - p.z * 0.2).cos();
        norm_sq += psi[k] * psi[k];
    }
    let inv_norm = norm_sq.sqrt().recip();
    for k in 0..16 {
        psi[k] *= inv_norm;
    }

    let mut max_proj = 0.0f32;
    for i in 0..42 {
        let a = ASSESSORS[2 * i] as usize;
        let b = ASSESSORS[2 * i + 1] as usize;
        max_proj = max_proj.max(psi[a].abs() + psi[b].abs());
    }
    max_proj *= 0.70710678;

    let dist_sq = (2.0 * (1.0 - max_proj)).max(0.0);
    tau_out[idx] = pc.tau_base + pc.tau_amp * (-dist_sq * pc.lambda).exp();
}
