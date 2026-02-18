struct RenderConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    width: u32,
    height: u32,
    time: f32,
}

@group(0) @binding(2) var<uniform> pc: RenderConstants;
@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var out_image: texture_storage_2d<rgba8unorm, write>;

fn fire_palette(t: f32) -> vec3<f32> {
    let black = vec3<f32>(0.0, 0.0, 0.0);
    let red = vec3<f32>(0.5, 0.0, 0.0);
    let orange = vec3<f32>(1.0, 0.5, 0.0);
    let yellow = vec3<f32>(1.0, 1.0, 0.5);
    let white = vec3<f32>(1.0, 1.0, 1.0);
    
    if (t < 0.25) {
        return mix(black, red, t * 4.0);
    } else if (t < 0.5) {
        return mix(red, orange, (t - 0.25) * 4.0);
    } else if (t < 0.75) {
        return mix(orange, yellow, (t - 0.5) * 4.0);
    } else {
        return mix(yellow, white, (t - 0.75) * 4.0);
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    if (x >= pc.width || y >= pc.height) {
        return;
    }

    let cam_target = vec3<f32>(f32(pc.nx)*0.5, f32(pc.ny)*0.5, f32(pc.nz)*0.5);
    let cam_dist = f32(pc.nx) * 1.2;
    let angle = pc.time * 0.05 + 0.5;
    let ro = cam_target + vec3<f32>(cos(angle)*cam_dist, cam_dist * 0.4, sin(angle)*cam_dist);

    let ww = normalize(cam_target - ro);
    let uu = normalize(cross(ww, vec3<f32>(0.0, 1.0, 0.0)));
    let vv = normalize(cross(uu, ww));
    
    let p = (vec2<f32>(f32(x), f32(y)) - 0.5 * vec2<f32>(f32(pc.width), f32(pc.height))) / f32(pc.height);
    let rd = normalize(p.x * uu + p.y * vv + 1.5 * ww);

    var t = 0.0;
    var color = vec3<f32>(0.0);
    var opacity = 0.0;
    let step_size = 0.8;

    for (var i = 0; i < 200; i++) {
        let pos = ro + rd * t;
        if (pos.x >= 0.0 && pos.x < f32(pc.nx) && pos.y >= 0.0 && pos.y < f32(pc.ny) && pos.z >= 0.0 && pos.z < f32(pc.nz)) {
            let idx = u32(pos.x) + pc.nx * (u32(pos.y) + pc.ny * u32(pos.z));
            let val = field[idx];
            
            if (val > 1e-7) {
                let intensity = clamp(val * 200000.0, 0.0, 1.0);
                let c = fire_palette(intensity);
                let a = intensity * 0.05;
                color = color + (1.0 - opacity) * c * a;
                opacity = opacity + (1.0 - opacity) * a;
            }
        }
        t = t + step_size;
        if (opacity >= 0.98) { break; }
    }

    textureStore(out_image, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}
