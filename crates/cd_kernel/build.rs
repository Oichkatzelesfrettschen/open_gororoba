//! Build script: compile turboquant Vulkan compute shaders from GLSL to SPIR-V.
//!
//! Each .comp file under src/turboquant/vulkan/shaders/ is compiled to a
//! corresponding .spv file in $OUT_DIR/turboquant_vulkan/. The .rs side
//! includes the bytes via `include_bytes!(concat!(env!("OUT_DIR"), "/turboquant_vulkan/<name>.spv"))`.
//!
//! Requires `glslc` (Vulkan SDK) on PATH. If glslc is missing the build
//! is skipped with a warning so non-Vulkan-enabled builds still succeed;
//! the runtime VulkanQuantizer falls back to CPU when the SPIR-V bytes
//! are absent.

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let shader_src_dir = manifest_dir.join("src/turboquant/vulkan/shaders");
    let shader_out_dir = out_dir.join("turboquant_vulkan");
    if let Err(e) = std::fs::create_dir_all(&shader_out_dir) {
        println!(
            "cargo:warning=mkdir {} failed: {}",
            shader_out_dir.display(),
            e
        );
        return;
    }

    let glslc_available = std::process::Command::new("glslc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    for shader in &["quantize.comp", "dequant_dot.comp"] {
        let input = shader_src_dir.join(shader);
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("shader");
        let output = shader_out_dir.join(format!("{}.spv", stem));
        if !glslc_available || !input.exists() {
            // Write a 0-byte placeholder so include_bytes! does not fail
            // at compile time. The runtime VulkanQuantizer detects
            // empty bytes and falls back to CPU.
            let _ = std::fs::write(&output, []);
            continue;
        }
        let status = std::process::Command::new("glslc")
            .arg("-O")
            .arg("--target-env=vulkan1.2")
            .arg("-fshader-stage=compute")
            .arg("-o")
            .arg(&output)
            .arg(&input)
            .status();
        match status {
            Ok(s) if s.success() => {
                println!("cargo:rerun-if-changed={}", input.display());
            }
            Ok(s) => {
                println!(
                    "cargo:warning=glslc failed for {} (exit {}); writing empty placeholder",
                    input.display(),
                    s
                );
                let _ = std::fs::write(&output, []);
            }
            Err(e) => {
                println!(
                    "cargo:warning=glslc spawn failed for {}: {}; writing empty placeholder",
                    input.display(),
                    e
                );
                let _ = std::fs::write(&output, []);
            }
        }
    }
    if !glslc_available {
        println!(
            "cargo:warning=glslc not found on PATH; turboquant Vulkan SPIR-V \
             placeholders are 0 bytes. Install the Vulkan SDK for the GPU \
             quantize backend (runtime will fall back to CPU)."
        );
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/turboquant/vulkan/shaders");
}
