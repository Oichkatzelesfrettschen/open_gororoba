struct RenderConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    width: u32,
    height: u32,
    time: f32,
}

@group(0) @binding(2) var<uniform> pc: RenderConstants;
@group(0) @binding(0) var<storage, read> entropy_field: array<f32>;
@group(0) @binding(3) var<storage, read> tau_field: array<f32>;
@group(0) @binding(1) var out_image: texture_storage_2d<rgba8unorm, write>;

// --- Palettes ---

fn palette_structure(t: f32) -> vec3<f32> {
    // Neon Blue/Cyan/Purple
    let col_a = vec3<f32>(0.0, 0.0, 0.2);
    let col_b = vec3<f32>(0.0, 0.5, 1.0);
    let col_c = vec3<f32>(0.5, 0.0, 1.0);
    return mix(col_a, mix(col_b, col_c, t), t);
}

fn palette_energy(t: f32) -> vec3<f32> {
    // Black -> Red -> Gold -> White
    let black = vec3<f32>(0.0, 0.0, 0.0);
    let red = vec3<f32>(0.8, 0.1, 0.0);
    let gold = vec3<f32>(1.0, 0.8, 0.1);
    let white = vec3<f32>(1.0, 1.0, 1.0);
    
    if (t < 0.3) { return mix(black, red, t * 3.33); }
    else if (t < 0.7) { return mix(red, gold, (t - 0.3) * 2.5); }
    else { return mix(gold, white, (t - 0.7) * 3.33); }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    if (x >= pc.width || y >= pc.height) { return; }

    // Camera Orbit
    let center = vec3<f32>(f32(pc.nx)*0.5, f32(pc.ny)*0.5, f32(pc.nz)*0.5);
    let radius = f32(pc.nx) * 1.3;
    let cam_angle = pc.time * 0.02 + 1.0;
    let cam_height = sin(pc.time * 0.01) * f32(pc.ny) * 0.5;
    
    let ro = center + vec3<f32>(cos(cam_angle)*radius, cam_height, sin(cam_angle)*radius);
    let cam_target = center;
    
    let ww = normalize(cam_target - ro);
    let uu = normalize(cross(ww, vec3<f32>(0.0, 1.0, 0.0)));
    let vv = normalize(cross(uu, ww));
    
    let p_uv = (vec2<f32>(f32(x), f32(y)) - 0.5 * vec2<f32>(f32(pc.width), f32(pc.height))) / f32(pc.height);
    let rd = normalize(p_uv.x * uu + p_uv.y * vv + 1.5 * ww);

    var color_acc = vec3<f32>(0.01, 0.01, 0.02); // Dark background
    var opacity_acc = 0.0;
    let step_size = 0.6; // High fidelity
    var t = 0.0;

    // Intersection with Box (Optimization)
    // Box is [0, nx] x [0, ny] x [0, nz]
    // Simple logic: just march, but skip empty space if possible.
    // For now, linear march is fine.
    
    for (var i = 0; i < 400; i++) {
        let pos = ro + rd * t;
        
        if (pos.x >= 0.0 && pos.x < f32(pc.nx) && pos.y >= 0.0 && pos.y < f32(pc.ny) && pos.z >= 0.0 && pos.z < f32(pc.nz)) {
            let idx = u32(pos.x) + pc.nx * (u32(pos.y) + pc.ny * u32(pos.z));
            
            // 1. Structure Field (Tau)
            let tau_val = tau_field[idx]; 
            // Tau is ~0.55 to 0.75. Normalize to 0..1 for vis.
            let struc_dens = smoothstep(0.56, 0.7, tau_val) * 0.1;
            
            // 2. Energy Field (Entropy)
            let ent_val = entropy_field[idx];
            // Entropy is 1e-6 to 1.0. Log scale.
            let energy_dens = log(1.0 + ent_val * 100000.0) * 0.05;

            // Composition
            if (struc_dens > 0.001 || energy_dens > 0.001) {
                // Lighting Approximation (Simple directional)
                // To do real gradient, we need neighbors. Too expensive for now.
                // Use density-based self-shadowing heuristic:
                let shadow = exp(-opacity_acc * 4.0);
                
                let col_s = palette_structure(struc_dens * 10.0) * struc_dens; // Blue Glow
                let col_e = palette_energy(clamp(energy_dens * 2.0, 0.0, 1.0)) * energy_dens * 2.0; // Fire
                
                let src_alpha = clamp(struc_dens + energy_dens, 0.0, 1.0);
                let src_col = (col_s + col_e) * shadow;
                
                color_acc = color_acc + (1.0 - opacity_acc) * src_col;
                opacity_acc = opacity_acc + (1.0 - opacity_acc) * src_alpha;
            }
        }
        
        t = t + step_size;
        if (opacity_acc >= 0.98) { break; }
    }

    // Tone Mapping
    color_acc = color_acc / (color_acc + vec3<f32>(1.0)); // Reinhard
    color_acc = pow(color_acc, vec3<f32>(0.4545)); // Gamma

    textureStore(out_image, vec2<i32>(i32(x), i32(y)), vec4<f32>(color_acc, 1.0));
}
