use anyhow::{Context, Result};
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use data_core::HeliosphereFeatureRow;
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

use algebra_analysis::{
    boxkite_alignment::{generate_psl_2_7_permutations_16d},
    boxkites::{cached_sedenion_boxkites},
};
use lbm_3d_cuda::alignment_gpu::GpuBoxKiteAlignmentEngine;

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
}

#[derive(Serialize)]
struct AlignmentResultRow {
    r_au: f64,
    mission: String,
    max_alignment: f64,
    best_orient_idx: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("[1/3] Loading 16D Takens descriptors from {}...", args.input_csv.display());
    let mut reader = ReaderBuilder::new().from_path(&args.input_csv)?;
    
    // Group by mission to preserve temporal continuity for Takens
    let mut mission_groups: BTreeMap<String, Vec<HeliosphereFeatureRow>> = BTreeMap::new();
    for result in reader.deserialize::<HeliosphereFeatureRow>() {
        let r = result?;
        mission_groups.entry(r.mission.clone()).or_default().push(r);
    }

    let mut all_vectors = Vec::new();
    let mut all_r_aus = Vec::new();
    let mut all_missions = Vec::new();

    for (mission, rows) in mission_groups {
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
            all_vectors.extend_from_slice(&v16);
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
            let v = all_vectors[i * 16 + d];
            if v < mins[d] { mins[d] = v; }
            if v > maxs[d] { maxs[d] = v; }
        }
    }

    // Compute Morton codes and store with original indices
    let mut indexed_morton: Vec<(u64, usize)> = (0..n_vectors)
        .map(|i| {
            let mut v = [0.0; 16];
            v.copy_from_slice(&all_vectors[i * 16..(i + 1) * 16]);
            (algebra_analysis::boxkite_alignment::morton_code_16d(&v, &mins, &maxs), i)
        })
        .collect();

    // Sort by Morton code (Linear BVH order)
    indexed_morton.sort_by_key(|&(m, _)| m);

    // Reorder vectors for GPU scan (improves cache locality)
    let mut sorted_vectors = Vec::with_capacity(all_vectors.len());
    for &(_, idx) in &indexed_morton {
        sorted_vectors.extend_from_slice(&all_vectors[idx * 16..(idx + 1) * 16]);
    }

    println!("[3/4] Initializing GPU and preparing structural constants...");
    let gpu = GpuBoxKiteAlignmentEngine::try_new()
        .context("Failed to initialize GPU. Ensure CUDA is available.")?;

    // Prepare Box-Kite indices [7 * 12]
    let boxkites = cached_sedenion_boxkites();
    let mut bk_indices = Vec::with_capacity(7 * 12);
    for bk in boxkites.iter() {
        let mut indices = std::collections::BTreeSet::new();
        for a in &bk.assessors {
            indices.insert(a.low);
            indices.insert(a.high);
        }
        let vec_indices: Vec<u8> = indices.iter().map(|&i| i as u8).collect();
        bk_indices.extend_from_slice(&vec_indices);
    }

    // Prepare PSL(2,7) orientations [168 * 16]
    let orientations = generate_psl_2_7_permutations_16d();
    let mut orient_bytes = Vec::with_capacity(168 * 16);
    for p in &orientations {
        for &idx in p {
            orient_bytes.push(idx as u8);
        }
    }

    println!("[3/4] Running GPU-parallel orientation scan...");
    let (max_alignments_sorted, best_orients_sorted) = gpu.run_alignment_scan(
        &sorted_vectors,
        &orient_bytes,
        &bk_indices,
    )?;

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
        })?;
    }
    writer.flush()?;

    println!("DONE.");
    Ok(())
}
