// Volumetric ray-march renderer for LBM density/entropy fields.
//
// Camera controlled by explicit uniform parameters (azimuth, elevation,
// distance, FOV) rather than a time-driven orbit. This enables interactive
// mouse/keyboard control from the host.

struct RenderConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    width: u32,
    height: u32,
    time: f32,         // visual effect time (color phase animation)
    cam_azimuth: f32,  // camera orbit azimuth in radians
    cam_elevation: f32, // camera elevation angle in radians
    cam_distance: f32, // camera distance from grid center
    cam_fov: f32,      // field of view factor (2.0 = ~53 deg, lower = wider)
}

@group(0) @binding(2) var<uniform> pc: RenderConstants;
@group(0) @binding(0) var<storage, read> energy_field: array<f32>;
@group(0) @binding(3) var<storage, read> structure_field: array<f32>;
@group(0) @binding(1) var out_image: texture_storage_2d<rgba8unorm, write>;

fn spectral_projection(t: f32, phase: f32) -> vec3<f32> {
    let r = sin(t * 6.28 + phase) * 0.5 + 0.5;
    let g = sin(t * 6.28 + phase + 2.09) * 0.5 + 0.5;
    let b = sin(t * 6.28 + phase + 4.18) * 0.5 + 0.5;
    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= pc.width || y >= pc.height) { return; }

    // Camera setup from explicit uniforms
    let center = vec3<f32>(f32(pc.nx) * 0.5, f32(pc.ny) * 0.5, f32(pc.nz) * 0.5);

    let cos_az = cos(pc.cam_azimuth);
    let sin_az = sin(pc.cam_azimuth);
    let cos_el = cos(pc.cam_elevation);
    let sin_el = sin(pc.cam_elevation);

    // Spherical coordinates: camera orbits around center
    let ro = center + vec3<f32>(
        cos_az * cos_el * pc.cam_distance,
        sin_el * pc.cam_distance,
        sin_az * cos_el * pc.cam_distance
    );

    // Look-at camera matrix
    let ww = normalize(center - ro);
    let uu = normalize(cross(ww, vec3<f32>(0.0, 1.0, 0.0)));
    let vv = normalize(cross(uu, ww));

    // Ray direction from pixel coordinates
    let p_uv = (vec2<f32>(f32(x), f32(y)) - 0.5 * vec2<f32>(f32(pc.width), f32(pc.height))) / f32(pc.height);
    let rd = normalize(p_uv.x * uu + p_uv.y * vv + pc.cam_fov * ww);

    // Volumetric ray-march
    var color = vec3<f32>(0.002, 0.002, 0.005); // void background
    var opacity = 0.0;
    var t_dist = max(0.0, pc.cam_distance - f32(pc.nx) * 1.5);
    let step_size = 0.5;

    for (var i = 0; i < 600; i++) {
        let pos = ro + rd * t_dist;
        if (pos.x >= 0.0 && pos.x < f32(pc.nx)
            && pos.y >= 0.0 && pos.y < f32(pc.ny)
            && pos.z >= 0.0 && pos.z < f32(pc.nz))
        {
            let idx = u32(pos.x) + pc.nx * (u32(pos.y) + pc.ny * u32(pos.z));

            // Algebraic structure (tau/viscosity field)
            let tau = structure_field[idx];
            let s_dens = smoothstep(0.56, 0.75, tau);

            // Physical energy (entropy production from LBM collision)
            let e_val = energy_field[idx];
            let e_dens = log(1.0 + e_val * 500000.0) * 0.02;

            // Color synthesis
            let col_s = spectral_projection(s_dens, pc.time * 0.1) * s_dens * 0.5;
            let col_e = vec3<f32>(1.0, 0.6, 0.1) * e_dens * 2.0; // entropic plasma

            let dens = s_dens * 0.1 + e_dens;

            // Scattering with depth-dependent occlusion
            let shadow = exp(-opacity * 3.0);
            color += (1.0 - opacity) * (col_s + col_e) * shadow;
            opacity += (1.0 - opacity) * dens;
        }
        t_dist += step_size;
        if (opacity >= 0.99) { break; }
    }

    // Post-processing: Reinhard tonemap + gamma
    color = color * 1.2 / (1.0 + color);
    color = pow(color, vec3<f32>(0.4545));
    textureStore(out_image, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}
