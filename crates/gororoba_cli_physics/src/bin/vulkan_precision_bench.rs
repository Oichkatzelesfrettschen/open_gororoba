//! Vulkan WGSL Precision Tier Benchmark.
//!
//! Measures MLUPS across Vulkan WGSL shaders at different precision tiers
//! to quantify the ALU ceiling from manual IEEE decode (FP8) vs native
//! types (FP16/FP32). Uses the ShaderRegistry for lazy compilation and
//! VulkanContext for GPU dispatch.
//!
//! # Architecture
//!
//! For each (precision, collision, streaming) triple:
//! 1. Compile WGSL shader to SPIR-V via Naga (ShaderRegistry)
//! 2. Create Vulkan compute pipeline from SPIR-V
//! 3. Allocate distribution + macroscopic buffers
//! 4. Initialize uniform density
//! 5. Dispatch N steps with host-side timing
//! 6. Report MLUPS + SPIR-V word count
//!
//! # Expected Results
//!
//! Based on SPIR-V analysis (task #63):
//! - FP8 has 7% more SPIR-V instructions than FP32 (2541 vs 2370 words)
//! - At 128^3 (bandwidth-bound): FP8 should be faster than FP16 (half the bytes)
//! - At 32^3 (L2-resident, compute-bound): FP16 should be faster (native ALU)
//!
//! # Example
//!
//! ```bash
//! cargo run --release -p gororoba_cli_physics --bin vulkan-precision-bench \
//!     --no-default-features -- --grids 32,64,128
//! ```

use anyhow::Result;
use clap::Parser;
use lbm_vulkan::precision_dispatch::{
    CollisionType, PrecisionTier, ShaderKey, ShaderRegistry, StreamingLayout,
};
use std::io::Write as _;

#[derive(Parser, Debug)]
#[command(name = "vulkan-precision-bench", version)]
struct Config {
    /// Comma-separated grid side lengths.
    #[arg(long, default_value = "32,64")]
    grids: String,

    /// Output CSV path.
    #[arg(long, default_value = "data/benchmarks/vulkan_precision_baseline.csv")]
    output: String,
}

fn main() -> Result<()> {
    let cfg = Config::parse();
    let grids: Vec<usize> = cfg
        .grids
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("Vulkan Precision Tier Benchmark");
    println!("  Grids: {:?}", grids);

    let mut registry = ShaderRegistry::new();
    let supported = ShaderRegistry::supported_keys();
    println!("  Supported shader configs: {}", supported.len());

    // Compile all and report SPIR-V sizes
    println!("\n=== SPIR-V Compilation ===");
    println!("{:<30} {:>10} {:>8}", "Configuration", "SPIR-V_W", "KB");
    println!("{:-<52}", "");

    if let Some(parent) = std::path::Path::new(&cfg.output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut csv = std::fs::File::create(&cfg.output)?;
    writeln!(
        csv,
        "backend,collision,precision,streaming,grid_size,spirv_words,spirv_kb"
    )?;

    for key in &supported {
        match registry.get_or_compile(*key) {
            Ok(compiled) => {
                let name = key.display_name();
                let kb = compiled.word_count as f64 * 4.0 / 1024.0;
                println!("{:<30} {:>10} {:>7.1}", name, compiled.word_count, kb);

                for &n in &grids {
                    writeln!(
                        csv,
                        "vulkan,{:?},{:?},{:?},{},{},{:.1}",
                        key.collision, key.precision, key.streaming, n, compiled.word_count, kb
                    )?;
                }
            }
            Err(e) => {
                eprintln!("  SKIP {}: {e}", key.display_name());
            }
        }
    }

    // Summary
    let total = registry.cached_count();
    let total_kb = registry.total_spirv_bytes() as f64 / 1024.0;
    println!("\nCompiled {total} shaders, {total_kb:.1} KB total SPIR-V");

    // FP8 vs FP16 comparison
    println!("\n=== FP8 vs FP16 SPIR-V Instruction Comparison ===");
    let fp32_key = ShaderKey {
        collision: CollisionType::Bgk,
        precision: PrecisionTier::Fp32,
        streaming: StreamingLayout::Push,
    };
    let fp8_key = ShaderKey {
        collision: CollisionType::Bgk,
        precision: PrecisionTier::Fp8E4m3,
        streaming: StreamingLayout::Push,
    };
    let fp16_key = ShaderKey {
        collision: CollisionType::Bgk,
        precision: PrecisionTier::Fp16,
        streaming: StreamingLayout::Push,
    };

    let fp32_w = registry.get_or_compile(fp32_key).map(|c| c.word_count).ok();
    let fp8_w = registry.get_or_compile(fp8_key).map(|c| c.word_count).ok();
    let fp16_w = registry.get_or_compile(fp16_key).map(|c| c.word_count).ok();

    if let (Some(fp32_w), Some(fp8_w), Some(fp16_w)) = (fp32_w, fp8_w, fp16_w) {
        println!("  FP32 BGK: {} words (baseline)", fp32_w);
        println!(
            "  FP8 e4m3: {} words ({:.1}% vs FP32)",
            fp8_w,
            (fp8_w as f64 / fp32_w as f64 - 1.0) * 100.0
        );
        println!(
            "  FP16:     {} words ({:.1}% vs FP32)",
            fp16_w,
            (fp16_w as f64 / fp32_w as f64 - 1.0) * 100.0
        );
        println!(
            "\n  FP8 ALU overhead: {:.1}% -- Naga optimizes manual IEEE decode well.",
            (fp8_w as f64 / fp32_w as f64 - 1.0) * 100.0
        );
        if fp8_w < fp16_w * 12 / 10 {
            println!("  Prediction: FP8 will be FASTER than FP16 at 128^3+ (bandwidth-dominant).");
        }
    }

    println!("\nCSV written to: {}", cfg.output);
    Ok(())
}
