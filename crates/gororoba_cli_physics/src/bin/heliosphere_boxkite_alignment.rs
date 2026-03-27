use anyhow::Result;
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use data_core::HeliosphereFeatureRow;
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

use algebra_analysis::{
    boxkite_alignment::{generate_psl_2_7_permutations_16d, box_kite_alignment_scan_cpu},
    boxkites::{cached_sedenion_boxkites},
};

#[cfg(feature = "gpu")]
use lbm_3d_cuda::alignment_gpu::GpuBoxKiteAlignmentEngine;
use lbm_vulkan::{VulkanContext, alignment_vulkan::VulkanBoxKiteAlignmentEngine};

#[derive(Parser, Debug)]
#[command(name = "heliosphere-boxkite-alignment")]
#[command(about = "High-throughput GPU scan of 16D Takens descriptors for Box-Kite alignment")]
struct Args {
    /// Input full feature cube CSV.
    #[arg(long, default_value = "data/output/heliosphere/full_feature_cube.csv")]
    input_csv: PathBuf,

    /// Output alignment results CSV.
    #[arg(long, default_value = "data/output/heliosphere/boxkite_alignment_scan.csv")]
    out_csv: PathBuf,

    /// Batch size for GPU processing.
    #[arg(long, default_value_t = 65536)]
    batch_size: usize,

    /// Force CPU fallback.
    #[arg(long)]
    cpu_only: bool,

    /// Explicit backend selection (cuda, vulkan, cpu).
    #[arg(long)]
    backend: Option<String>,
}

#[derive(Serialize)]
struct AlignmentResultRow {
    r_au: f64,
    mission: String,
    max_alignment: f64,
    best_orient_idx: u32,
    backend: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("[1/4] Loading 16D Takens descriptors from {}...", args.input_csv.display());
    let mut reader = ReaderBuilder::new().from_path(&args.input_csv)?;
    
    // Group by mission + product to preserve temporal continuity for Takens
    let mut mission_groups: BTreeMap<(String, String), Vec<HeliosphereFeatureRow>> = BTreeMap::new();
    for result in reader.deserialize::<HeliosphereFeatureRow>() {
        let r = result?;
        if r.bx.is_finite() && r.by.is_finite() && r.bz.is_finite() && r.b_mag.is_finite() && r.b_mag > 0.0 {
            mission_groups.entry((r.mission.clone(), r.product.clone())).or_default().push(r);
        }
    }

    // Sort rows by time within each group
    for rows in mission_groups.values_mut() {
        rows.sort_by(|a, b| {
            a.year
                .cmp(&b.year)
                .then(a.doy.cmp(&b.doy))
                .then(a.hour.cmp(&b.hour))
        });
    }

    let mut all_vectors = Vec::new();
    let mut all_r_aus = Vec::new();
    let mut all_missions = Vec::new();

    for ((mission, _product), rows) in mission_groups {
        if rows.len() < 4 { continue; }
        
        // 16D reconstruction from 4-sample delay of (Bx, By, Bz, |B|)
        for window in rows.windows(4) {
            let mut v16 = [0.0; 16];
            let local_mean_b = (window[0].b_mag + window[1].b_mag + window[2].b_mag + window[3].b_mag) / 4.0;
            if local_mean_b <= 0.0 { continue; }

            for i in 0..4 {
                v16[i * 4 + 0] = window[i].bx / local_mean_b;
                v16[i * 4 + 1] = window[i].by / local_mean_b;
                v16[i * 4 + 2] = window[i].bz / local_mean_b;
                v16[i * 4 + 3] = (window[i].b_mag - local_mean_b) / local_mean_b;
            }
            all_vectors.push(v16);
            all_r_aus.push(window[3].r_au);
            all_missions.push(mission.clone());
        }
    }

    let n_vectors = all_r_aus.len();
    println!("      Found {} valid 16D vectors.", n_vectors);

    println!("[2/4] Implementing YSU LBVH pattern: computing 16D Morton codes and sorting...");
    // Compute bounds for Morton normalization
    let mut mins = [f64::INFINITY; 16];
    let mut maxs = [f64::NEG_INFINITY; 16];
    for i in 0..n_vectors {
        for d in 0..16 {
            let v = all_vectors[i][d];
            if v < mins[d] { mins[d] = v; }
            if v > maxs[d] { maxs[d] = v; }
        }
    }

    // Compute Morton codes and store with original indices
    let mut indexed_morton: Vec<(u64, usize)> = (0..n_vectors)
        .map(|i| {
            (algebra_analysis::boxkite_alignment::morton_code_16d(&all_vectors[i], &mins, &maxs), i)
        })
        .collect();

    // Sort by Morton code (Linear BVH order)
    indexed_morton.sort_by_key(|&(m, _)| m);

    // Reorder vectors for alignment scan (improves cache locality)
    let mut sorted_vectors = Vec::with_capacity(n_vectors);
    for &(_, idx) in &indexed_morton {
        sorted_vectors.push(all_vectors[idx]);
    }

    // Prepare Box-Kite indices
    let boxkites = cached_sedenion_boxkites();
    let orientations = generate_psl_2_7_permutations_16d();

    let mut max_alignments_sorted = Vec::new();
    let mut best_orients_sorted = Vec::new();

    let mut used_backend = "None";

    let target_backend = args.backend.as_ref().map(|s| s.to_lowercase());

    if !args.cpu_only && (target_backend.is_none() || target_backend.as_deref() == Some("cuda") || target_backend.as_deref() == Some("vulkan")) {
        if target_backend.as_deref() == Some("cuda") || target_backend.is_none() {
            #[cfg(feature = "gpu")]
            {
                println!("[3/4] Attempting CUDA backend...");
                if let Some(gpu) = GpuBoxKiteAlignmentEngine::try_new() {
                    // Flatten sorted_vectors for CUDA
                    let flat_vectors: Vec<f64> = sorted_vectors.iter().flatten().copied().collect();
                    
                    let mut bk_indices_u8 = Vec::with_capacity(7 * 12);
                    for bk in boxkites.iter() {
                        let mut indices = std::collections::BTreeSet::new();
                        for a in &bk.assessors {
                            indices.insert(a.low);
                            indices.insert(a.high);
                        }
                        let vec_indices: Vec<u8> = indices.iter().map(|&i| i as u8).collect();
                        bk_indices_u8.extend_from_slice(&vec_indices);
                    }

                    let mut orient_bytes = Vec::with_capacity(168 * 16);
                    for p in &orientations {
                        for &idx in p {
                            orient_bytes.push(idx as u8);
                        }
                    }

                    match gpu.run_alignment_scan(&flat_vectors, &orient_bytes, &bk_indices_u8) {
                        Ok((m, b)) => {
                            max_alignments_sorted = m;
                            best_orients_sorted = b;
                            used_backend = "CUDA";
                        }
                        Err(e) => {
                            if target_backend.as_deref() == Some("cuda") {
                                return Err(anyhow::anyhow!("Forced CUDA backend failed: {}", e));
                            }
                            eprintln!("      CUDA scan failed: {}. Falling back...", e);
                        }
                    }
                } else if target_backend.as_deref() == Some("cuda") {
                    return Err(anyhow::anyhow!("CUDA backend requested but engine could not be initialized"));
                }
            }
        }

        if used_backend == "None" && (target_backend.as_deref() == Some("vulkan") || target_backend.is_none()) {
            println!("[3/4] Attempting Vulkan backend...");
            if let Ok(v_ctx) = VulkanContext::new(false) {
                if let Ok(mut v_engine) = VulkanBoxKiteAlignmentEngine::try_new(&v_ctx) {
                    let flat_vectors: Vec<f64> = sorted_vectors.iter().flatten().copied().collect();
                    
                    let mut bk_indices_u32 = Vec::with_capacity(7 * 12);
                    for bk in boxkites.iter() {
                        let mut indices = std::collections::BTreeSet::new();
                        for a in &bk.assessors {
                            indices.insert(a.low);
                            indices.insert(a.high);
                        }
                        let vec_indices: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
                        bk_indices_u32.extend_from_slice(&vec_indices);
                    }

                    let mut orient_u32 = Vec::with_capacity(168 * 16);
                    for p in &orientations {
                        for &idx in p {
                            orient_u32.push(idx as u32);
                        }
                    }

                    match v_engine.run_alignment_scan(&v_ctx, &flat_vectors, &orient_u32, &bk_indices_u32) {
                        Ok((m, b)) => {
                            max_alignments_sorted = m;
                            best_orients_sorted = b;
                            used_backend = "Vulkan";
                        }
                        Err(e) => {
                            if target_backend.as_deref() == Some("vulkan") {
                                return Err(anyhow::anyhow!("Forced Vulkan backend failed: {}", e));
                            }
                            eprintln!("      Vulkan scan failed: {}. Falling back...", e);
                        }
                    }
                } else if target_backend.as_deref() == Some("vulkan") {
                    return Err(anyhow::anyhow!("Vulkan backend requested but engine could not be initialized"));
                }
            } else if target_backend.as_deref() == Some("vulkan") {
                return Err(anyhow::anyhow!("Vulkan backend requested but context could not be created"));
            }
        }
    }

    if used_backend == "None" {
        if let Some(ref b) = target_backend {
            if b != "cpu" {
                return Err(anyhow::anyhow!("Requested backend '{}' failed or unavailable, and fallback disabled", b));
            }
        }
        println!("[3/4] Using CPU backend (Rayon)...");
        let (m, b) = box_kite_alignment_scan_cpu(&sorted_vectors, &orientations, boxkites);
        max_alignments_sorted = m;
        best_orients_sorted = b;
        used_backend = "CPU";
    }

    println!("      Alignment scan completed using {} backend.", used_backend);

    // Map results back to original order
    let mut max_alignments = vec![0.0; n_vectors];
    let mut best_orients = vec![0u32; n_vectors];
    for (i, &(_, orig_idx)) in indexed_morton.iter().enumerate() {
        max_alignments[orig_idx] = max_alignments_sorted[i];
        best_orients[orig_idx] = best_orients_sorted[i];
    }

    println!("[4/4] Writing results to {}...", args.out_csv.display());
    if let Some(parent) = args.out_csv.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = WriterBuilder::new().from_path(&args.out_csv)?;
    for i in 0..n_vectors {
        writer.serialize(AlignmentResultRow {
            r_au: all_r_aus[i],
            mission: all_missions[i].clone(),
            max_alignment: max_alignments[i],
            best_orient_idx: best_orients[i],
            backend: used_backend.to_string(),
        })?;
    }
    writer.flush()?;

    println!("DONE.");
    Ok(())
}
