#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, set = 0, binding = 0) readonly buffer Field { float data[]; };
layout(rgba8, set = 0, binding = 1) writeonly uniform image2D out_image;

layout(push_constant) uniform Constants {
    uint nx;
    uint ny;
    uint nz;
    uint width;
    uint height;
    float time;
} pc;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    if (x >= pc.width || y >= pc.height) return;

    // Camera setup
    vec3 ro = vec3(float(pc.nx) * 1.5, float(pc.ny) * 0.5, float(pc.nz) * 0.5);
    // Rotate camera over time
    float angle = pc.time * 0.2;
    float rx = (ro.x - float(pc.nx)*0.5) * cos(angle) - (ro.z - float(pc.nz)*0.5) * sin(angle) + float(pc.nx)*0.5;
    float rz = (ro.x - float(pc.nx)*0.5) * sin(angle) + (ro.z - float(pc.nz)*0.5) * cos(angle) + float(pc.nz)*0.5;
    ro = vec3(rx, ro.y, rz);

    vec3 target = vec3(float(pc.nx)*0.5, float(pc.ny)*0.5, float(pc.nz)*0.5);
    vec3 ww = normalize(target - ro);
    vec3 uu = normalize(cross(ww, vec3(0, 1, 0)));
    vec3 vv = normalize(cross(uu, ww));
    
    vec2 p = (vec2(x, y) - 0.5 * vec2(pc.width, pc.height)) / float(pc.height);
    vec3 rd = normalize(p.x * uu + p.y * vv + 1.5 * ww);

    // Raymarching
    float t = 0.0;
    vec3 color = vec3(0.0);
    float alpha = 0.0;
    
    // Step size
    float step_size = 1.0;
    for (int i = 0; i < 256; i++) {
        vec3 pos = ro + rd * t;
        
        // Bounds check
        if (pos.x < 0 || pos.x >= pc.nx || pos.y < 0 || pos.y >= pc.ny || pos.z < 0 || pos.z >= pc.nz) {
            t += step_size;
            if (t > 1000.0) break;
            continue;
        }
        
        // Sample field
        uint idx = uint(pos.x) + pc.nx * (uint(pos.y) + pc.ny * uint(pos.z));
        float val = data[idx];
        
        // Accumulate (Heatmap: black -> blue -> magenta -> white)
        if (val > 0.01) {
            vec3 c = mix(vec3(0, 0, 1), vec3(1, 0, 1), clamp(val * 5.0, 0.0, 1.0));
            c = mix(c, vec3(1, 1, 1), clamp(val - 0.5, 0.0, 1.0));
            
            float a = val * 0.1; // Opacity
            color += (1.0 - alpha) * c * a;
            alpha += (1.0 - alpha) * a;
        }
        
        t += step_size;
        if (alpha >= 0.95) break;
    }

    imageStore(out_image, ivec2(x, y), vec4(color, 1.0));
}
