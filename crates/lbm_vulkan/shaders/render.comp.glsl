#version 450

layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, set = 0, binding = 0) readonly buffer Field { float data[]; };
layout(rgba8, set = 0, binding = 1) writeonly uniform image2D out_image;

layout(push_constant) uniform Constants {
    uint nx; uint ny; uint nz;
    uint width; uint height;
    float time;
} pc;

vec3 fire_palette(float t) {
    vec3 black = vec3(0.0);
    vec3 red = vec3(0.5, 0.0, 0.0);
    vec3 orange = vec3(1.0, 0.5, 0.0);
    vec3 yellow = vec3(1.0, 1.0, 0.5);
    vec3 white = vec3(1.0);
    
    if (t < 0.25) return mix(black, red, t * 4.0);
    if (t < 0.5) return mix(red, orange, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(orange, yellow, (t - 0.5) * 4.0);
    return mix(yellow, white, (t - 0.75) * 4.0);
}

void main() {
    uint x = gl_GlobalInvocationID.x; uint y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;

    vec3 target = vec3(float(pc.nx)*0.5, float(pc.ny)*0.5, float(pc.nz)*0.5);
    float cam_dist = float(pc.nx) * 1.2;
    float angle = pc.time * 0.05 + 0.5;
    vec3 ro = target + vec3(cos(angle)*cam_dist, cam_dist * 0.4, sin(angle)*cam_dist);

    vec3 ww = normalize(target - ro);
    vec3 uu = normalize(cross(ww, vec3(0, 1, 0)));
    vec3 vv = normalize(cross(uu, ww));
    vec2 p = (vec2(x, y) - 0.5 * vec2(pc.width, pc.height)) / float(pc.height);
    vec3 rd = normalize(p.x * uu + p.y * vv + 1.5 * ww);

    float t = 0.0; vec3 color = vec3(0.0); float opacity = 0.0;
    float step_size = 0.8; 
    
    for (int i = 0; i < 200; i++) {
        vec3 pos = ro + rd * t;
        if (pos.x >= 0 && pos.x < pc.nx && pos.y >= 0 && pos.y < pc.ny && pos.z >= 0 && pos.z < pc.nz) {
            uint idx = uint(pos.x) + pc.nx * (uint(pos.y) + pc.ny * uint(pos.z));
            float val = data[idx];
            
            if (val > 0.001) {
                float intensity = clamp(val * 10.0, 0.0, 1.0);
                vec3 c = fire_palette(intensity);
                float a = intensity * 0.05;
                color += (1.0 - opacity) * c * a;
                opacity += (1.0 - opacity) * a;
            }
        }
        t += step_size;
        if (opacity >= 0.98) break;
    }

    imageStore(out_image, ivec2(x, y), vec4(color, 1.0));
}
