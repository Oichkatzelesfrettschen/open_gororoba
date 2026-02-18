struct RenderConstants {
    nx: u32, ny: u32, nz: u32, width: u32, height: u32, time: f32
}
@group(0) @binding(2) var<uniform> pc: RenderConstants;
@group(0) @binding(0) var<storage, read> energy_field: array<f32>;
@group(0) @binding(3) var<storage, read> structure_field: array<f32>;
@group(0) @binding(1) var out_image: texture_storage_2d<rgba8unorm, write>;

fn spectral_projection(t: f32, phase: f32) -> vec3<f32> {
    // 16D Color mapping
    let r = sin(t * 6.28 + phase) * 0.5 + 0.5;
    let g = sin(t * 6.28 + phase + 2.09) * 0.5 + 0.5;
    let b = sin(t * 6.28 + phase + 4.18) * 0.5 + 0.5;
    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= pc.width || y >= pc.height) { return; }

    let center = vec3<f32>(f32(pc.nx)*0.5, f32(pc.ny)*0.5, f32(pc.nz)*0.5);
    let cam_angle = pc.time * 0.01;
    let ro = center + vec3<f32>(cos(cam_angle)*180.0, sin(pc.time*0.005)*60.0, sin(cam_angle)*180.0);
    
    let ww = normalize(center - ro);
    let uu = normalize(cross(ww, vec3<f32>(0.0, 1.0, 0.0)));
    let vv = normalize(cross(uu, ww));
    let p_uv = (vec2<f32>(f32(x), f32(y)) - 0.5 * vec2<f32>(f32(pc.width), f32(pc.height))) / f32(pc.height);
    let rd = normalize(p_uv.x * uu + p_uv.y * vv + 2.0 * ww);

    var color = vec3<f32>(0.002, 0.002, 0.005); // Void background
    var opacity = 0.0;
    var t_dist = 40.0;
    let step = 0.5;

    for (var i = 0; i < 600; i++) {
        let pos = ro + rd * t_dist;
        if (pos.x >= 0.0 && pos.x < f32(pc.nx) && pos.y >= 0.0 && pos.y < f32(pc.ny) && pos.z >= 0.0 && pos.z < f32(pc.nz)) {
            let idx = u32(pos.x) + pc.nx * (u32(pos.y) + pc.ny * u32(pos.z));
            
            // Algebraic Structure (Associator Tension)
            let tau = structure_field[idx];
            let s_dens = smoothstep(0.56, 0.75, tau);
            
            // Physical Energy (Entropy)
            let e_val = energy_field[idx];
            let e_dens = log(1.0 + e_val * 500000.0) * 0.02;

            // Synthesis: "The Geometric Fire"
            let col_s = spectral_projection(s_dens, pc.time * 0.1) * s_dens * 0.5; // Shimmering manifold
            let col_e = vec3<f32>(1.0, 0.6, 0.1) * e_dens * 2.0; // Golden entropic plasma
            
            let dens = s_dens * 0.1 + e_dens;
            
            // Scattering/Occlusion Heuristic
            let shadow = exp(-opacity * 3.0);
            color += (1.0 - opacity) * (col_s + col_e) * shadow;
            opacity += (1.0 - opacity) * dens;
        }
        t_dist += step;
        if (opacity >= 0.99) { break; }
    }

    // High-End Post: Reinhard + Saturation Boost
    color = color * 1.2 / (1.0 + color);
    color = pow(color, vec3<f32>(0.4545)); // Gamma
    textureStore(out_image, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}
