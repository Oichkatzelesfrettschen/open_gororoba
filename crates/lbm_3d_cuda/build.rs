//! AOT .cubin compilation for CUDA kernels.
//!
//! Compiles .cu files to .cubin via `nvcc --cubin -arch=sm_89` at build time.
//! The cubins are embedded via `include_bytes!` and loaded via `cuModuleLoadData`,
//! eliminating the ~1.5 GB VRAM spike from NVRTC JIT compilation at runtime.
//!
//! Feature-gated behind `aot-cubin`. When disabled, falls back to NVRTC.

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// CUDA kernel source files to compile AOT.
/// Only the production-critical kernels are compiled; the rest use NVRTC fallback.
const AOT_KERNELS: &[&str] = &[
    "kernels_soa.cu",      // FP32 MRT/A-A/tiled/coarsened/ephemeral
    "kernels_int8_soa.cu", // INT8 MRT/A-A (production tier)
    "kernels_bf16_soa.cu", // BF16 MRT/A-A
    "kernels_fp16_soa.cu", // FP16 MRT
    "kernels_fp8_soa.cu",  // FP8 e4m3 MRT
    "kernels_slice.cu",    // On-the-fly slice extraction
];

fn main() {
    // Only compile cubins when the aot-cubin feature is enabled
    if env::var("CARGO_FEATURE_AOT_CUBIN").is_err() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src_dir = Path::new("src");

    // Detect GPU architecture from environment or default to sm_89 (Ada)
    let arch = env::var("GOROROBA_CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());

    println!("cargo:rerun-if-env-changed=GOROROBA_CUDA_ARCH");

    for kernel in AOT_KERNELS {
        let cu_path = src_dir.join(kernel);
        let cubin_name = kernel.replace(".cu", ".cubin");
        let cubin_path = out_dir.join(&cubin_name);

        println!("cargo:rerun-if-changed=src/{kernel}");

        // nvcc --cubin -arch=sm_89 -o OUT_DIR/kernels_soa.cubin src/kernels_soa.cu
        let status = Command::new("nvcc")
            .arg("--cubin")
            .arg(format!("-arch={arch}"))
            .arg("-o")
            .arg(&cubin_path)
            .arg(&cu_path)
            // Include paths for BF16/FP16 headers
            .arg("-I/opt/cuda/include")
            .arg("-I/usr/include")
            .status();

        match status {
            Ok(s) if s.success() => {
                eprintln!("  AOT: {} -> {} ({})", kernel, cubin_name, arch);
            }
            Ok(s) => {
                panic!(
                    "nvcc failed for {} with exit code {:?}. \
                     Ensure CUDA toolkit is installed and nvcc is in PATH.",
                    kernel,
                    s.code()
                );
            }
            Err(e) => {
                panic!(
                    "Failed to run nvcc for {}: {}. \
                     Install CUDA toolkit or disable the aot-cubin feature.",
                    kernel, e
                );
            }
        }
    }

    // Write a manifest file listing all compiled cubins
    let manifest_path = out_dir.join("aot_cubins.rs");
    let mut manifest = String::new();
    manifest.push_str("// Auto-generated AOT cubin manifest. Do not edit.\n\n");
    for kernel in AOT_KERNELS {
        let const_name = kernel
            .replace(".cu", "")
            .replace("kernels_", "CUBIN_")
            .to_uppercase();
        let cubin_name = kernel.replace(".cu", ".cubin");
        manifest.push_str(&format!(
            "pub const {const_name}: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{cubin_name}\"));\n"
        ));
    }
    std::fs::write(&manifest_path, manifest).expect("Failed to write AOT cubin manifest");
}
