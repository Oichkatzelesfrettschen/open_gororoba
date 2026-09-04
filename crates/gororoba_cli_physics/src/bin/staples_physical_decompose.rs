//! Attribute the six-sample THEMIS-A staple score to physical channels.
//!
//! The CD associator is a cubic mismatch on six FGM samples. This binary
//! scores the same E-239 subsample with GSE channels, lag-pair CD masks,
//! and Sonnerup-Cahill LMN quantities on that window. Rank-AUC says which
//! physical channel carries magnetopause-crossing discrimination. It does
//! not claim CD uniqueness.
//!
//! ```bash
//! cargo run --profile validation -p gororoba_cli_physics --bin staples-physical-decompose
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use gororoba_cli_physics::staple_associator::{STAPLE_DIM, staple_embedding};
use gororoba_cli_physics::staple_benchmark::{load_labeled_files, rank_auc, stratified_keep};
use gororoba_cli_physics::staple_controls::{SparseCubicTensor, six_sample_baselines};
use gororoba_cli_physics::staple_physical::{
    average_ranks, b_path_length, keep_two_lags, mag_jump, masked_normalized_score, mva_six_sample,
    octonion_two_lag_only, quaternion_lag0_only, zero_doubling_slot, zero_magnitude_channels,
};
use serde::Serialize;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser, Debug)]
#[command(about = "Physical-channel decomposition of the six-sample staple associator")]
struct Args {
    #[arg(long, default_value = "data/output/tha_matched_files.csv")]
    matched_files: PathBuf,
    #[arg(long, default_value = "data/output/cat_themis_a.csv")]
    catalog: PathBuf,
    #[arg(long, default_value = "data/output/staples_physical_decompose.json")]
    out: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0.05)]
    neg_fraction: f64,
    #[arg(long, default_value_t = 2.0)]
    label_pad_minutes: f64,
    #[arg(long, default_value_t = 500)]
    min_samples: usize,
}

#[derive(Serialize)]
struct ChannelAuc {
    name: &'static str,
    auc: f64,
    what_it_measures: &'static str,
}

#[derive(Serialize)]
struct Report {
    files: usize,
    subsample_size: usize,
    subsample_positives: usize,
    channels: Vec<ChannelAuc>,
    reading: String,
}

fn auc_of(scores: &[f64], labels: &[bool]) -> f64 {
    rank_auc(scores, labels)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    let files = load_labeled_files(
        &args.matched_files,
        &args.catalog,
        args.label_pad_minutes,
        args.min_samples,
    )?;
    anyhow::ensure!(!files.is_empty(), "no labeled files");
    let table = cd_kernel::mult_table::CdMultTable::generate(STAPLE_DIM);
    let tensor = SparseCubicTensor::from_associator(&table);

    let mut labels = Vec::new();
    let mut cd_scores = Vec::new();
    let mut rot_scores = Vec::new();
    let mut mag_scores = Vec::new();
    let mut mag0_scores = Vec::new();
    let mut e8_0_scores = Vec::new();
    let mut q_scores = Vec::new();
    let mut oct_scores = Vec::new();
    let mut lag12_scores = Vec::new();
    let mut lag23_scores = Vec::new();
    let mut lm_rot_scores = Vec::new();
    let mut delta_bl_scores = Vec::new();
    let mut gram_scores = Vec::new();
    let mut path_scores = Vec::new();

    for f in &files {
        let staples = staple_embedding(&f.rows);
        let base = six_sample_baselines(&f.rows);
        anyhow::ensure!(
            staples.len().saturating_sub(2) == f.labels.len(),
            "label alignment in {}",
            f.path
        );
        anyhow::ensure!(base.max_rotation.len() == f.labels.len());
        for k in stratified_keep(&f.labels, f.file_id, args.seed, args.neg_fraction) {
            let a = &staples[k];
            let b = &staples[k + 1];
            let c = &staples[k + 2];
            labels.push(f.labels[k]);
            cd_scores.push(tensor.normalized_score(a, b, c));
            rot_scores.push(base.max_rotation[k]);
            mag_scores.push(mag_jump(&f.rows, k));
            mag0_scores.push(masked_normalized_score(
                &tensor,
                a,
                b,
                c,
                zero_magnitude_channels,
            ));
            e8_0_scores.push(masked_normalized_score(
                &tensor,
                a,
                b,
                c,
                zero_doubling_slot,
            ));
            q_scores.push(masked_normalized_score(
                &tensor,
                a,
                b,
                c,
                quaternion_lag0_only,
            ));
            oct_scores.push(masked_normalized_score(
                &tensor,
                a,
                b,
                c,
                octonion_two_lag_only,
            ));
            lag12_scores.push(masked_normalized_score(&tensor, a, b, c, |v| {
                keep_two_lags(v, 1, 2)
            }));
            lag23_scores.push(masked_normalized_score(&tensor, a, b, c, |v| {
                keep_two_lags(v, 2, 3)
            }));
            let mva = mva_six_sample(&f.rows, k).expect("six-sample window");
            lm_rot_scores.push(mva.lm_rotation);
            delta_bl_scores.push(mva.delta_bl);
            gram_scores.push(base.max_gram_volume[k]);
            path_scores.push(b_path_length(&f.rows, k));
        }
    }

    let channels = vec![
        ChannelAuc {
            name: "cd_associator",
            auc: auc_of(&cd_scores, &labels),
            what_it_measures: "full 16D CD cubic mismatch on six samples",
        },
        ChannelAuc {
            name: "max_rotation",
            auc: auc_of(&rot_scores, &labels),
            what_it_measures: "max stepwise angle of B in the same six samples",
        },
        ChannelAuc {
            name: "mag_jump",
            auc: auc_of(&mag_scores, &labels),
            what_it_measures: "max |B| minus min |B| in the same six samples",
        },
        ChannelAuc {
            name: "cd_magnitude_zeroed",
            auc: auc_of(&mag0_scores, &labels),
            what_it_measures: "CD associator after clearing every |B| channel",
        },
        ChannelAuc {
            name: "cd_e8_zeroed",
            auc: auc_of(&e8_0_scores, &labels),
            what_it_measures: "CD associator after clearing lag-2 Bx (doubling slot)",
        },
        ChannelAuc {
            name: "cd_quaternion_lag0",
            auc: auc_of(&q_scores, &labels),
            what_it_measures: "CD associator on lag-0 only; algebraically associative, expect chance",
        },
        ChannelAuc {
            name: "cd_octonion_two_lag",
            auc: auc_of(&oct_scores, &labels),
            what_it_measures: "CD associator on lags 0-1 (octonion prefix e0..e7)",
        },
        ChannelAuc {
            name: "cd_lags_1_2",
            auc: auc_of(&lag12_scores, &labels),
            what_it_measures: "CD associator keeping only lags 1 and 2",
        },
        ChannelAuc {
            name: "cd_lags_2_3",
            auc: auc_of(&lag23_scores, &labels),
            what_it_measures: "CD associator keeping only lags 2 and 3",
        },
        ChannelAuc {
            name: "mva_lm_rotation",
            auc: auc_of(&lm_rot_scores, &labels),
            what_it_measures: "max LM-plane rotation from Sonnerup-Cahill MVA on the six samples",
        },
        ChannelAuc {
            name: "mva_delta_bl",
            auc: auc_of(&delta_bl_scores, &labels),
            what_it_measures: "max minus min B along the MVA maximum-variance axis",
        },
        ChannelAuc {
            name: "b_path_length",
            auc: auc_of(&path_scores, &labels),
            what_it_measures: "hodogram arc length in R^3: sum |dB| over the six samples",
        },
        ChannelAuc {
            name: "gram_volume",
            auc: auc_of(&gram_scores, &labels),
            what_it_measures: "max |det(dB_i, dB_{i+1}, dB_{i+2})|: 3D parallelepiped volume, zero for planar B",
        },
        ChannelAuc {
            name: "rank_rotation_plus_mag_jump",
            auc: auc_of(
                &average_ranks(&rot_scores)
                    .iter()
                    .zip(average_ranks(&mag_scores).iter())
                    .map(|(r, m)| r + m)
                    .collect::<Vec<_>>(),
                &labels,
            ),
            what_it_measures: "sum of ranks of max_rotation and mag_jump; no fitted weights",
        },
    ];

    let named = |name: &str| {
        channels
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.auc)
            .unwrap_or(f64::NAN)
    };
    let reading = format!(
        "lag-0 quaternion {q:.4} is chance (lag-mixing). adjacent lag pairs 0-1 {oct:.4}, 1-2 {l12:.4}, 2-3 {l23:.4} vs full CD {cd:.4}. MVA LM-rotation {lm:.4} is a plane reduction. 3D hodogram path {path:.4}, 3D gram volume {gram:.4}. max_rotation {rot:.4} still leads. mag_jump {magj:.4}.",
        q = named("cd_quaternion_lag0"),
        oct = named("cd_octonion_two_lag"),
        l12 = named("cd_lags_1_2"),
        l23 = named("cd_lags_2_3"),
        cd = named("cd_associator"),
        lm = named("mva_lm_rotation"),
        path = named("b_path_length"),
        gram = named("gram_volume"),
        rot = named("max_rotation"),
        magj = named("mag_jump"),
    );

    let report = Report {
        files: files.len(),
        subsample_size: labels.len(),
        subsample_positives: labels.iter().filter(|&&x| x).count(),
        channels,
        reading,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&args.out, json).with_context(|| format!("write {}", args.out.display()))?;
    eprintln!("{}", serde_json::to_string_pretty(&report)?);
    eprintln!(
        "wrote {} in {:.0} s",
        args.out.display(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
