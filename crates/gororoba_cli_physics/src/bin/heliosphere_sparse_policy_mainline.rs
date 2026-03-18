use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use gororoba_cli_physics::heliosphere_eval::{
    SeededSparsePolicySummary, SparseMaskSummary, SparsePolicyDatasetContext, SparsePolicyTransferSpec,
    build_sparse_policy_dataset_context, load_heliosphere_rows, summarize_seeded_sparse_policy_rows,
    summarize_sparse_policies,
};
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-sparse-policy-mainline",
    about = "Promote and stress-test the mainline heliosphere sparse-policy candidate across seeds and cubes"
)]
struct Cli {
    #[arg(long = "cube-csv")]
    cube_csvs: Vec<PathBuf>,

    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value_t = 24)]
    horizon_hours: i64,

    #[arg(long, default_value_t = 1024)]
    grid: usize,

    #[arg(long = "split-seed")]
    split_seeds: Vec<u64>,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyAggregate {
    policy_key: String,
    normalization_strategy: String,
    descriptor_profile: String,
    seed_count: usize,
    seeds_within_budget: usize,
    mean_active_fraction: f64,
    max_active_fraction: f64,
    mean_event_label_recall: f64,
    max_event_label_recall: f64,
    mean_event_label_precision: f64,
    mean_projected_gib: f64,
    max_projected_gib: f64,
    mean_lead_time_hours: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CubeEvaluation {
    cube_name: String,
    cube_csv: String,
    positive_sample_count: usize,
    supervised: bool,
    baseline_policies: Vec<SparseMaskSummary>,
    seeded_policies: Vec<SeededSparsePolicySummary>,
    aggregates: Vec<PolicyAggregate>,
    promoted_survives_cube: bool,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    repo_root: String,
    horizon_hours: i64,
    grid: usize,
    split_seeds: Vec<u64>,
    reference_cube: String,
    comparator_policy_key: String,
    promoted_policy_key: Option<String>,
    promotion_survives_all_cubes: bool,
    registry_update_performed: bool,
    registry_update_reason: String,
    cubes: Vec<CubeEvaluation>,
    notes: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cube_csvs = if cli.cube_csvs.is_empty() {
        vec![
            PathBuf::from("reports/heliosphere_feature_cube_modern2020_2026-03-15.csv"),
            PathBuf::from("reports/heliosphere_feature_cube_imap2026_2026-03-15.csv"),
            PathBuf::from("reports/heliosphere_feature_cube_inner1976_2026-03-15.csv"),
        ]
    } else {
        cli.cube_csvs.clone()
    };
    let split_seeds = if cli.split_seeds.is_empty() {
        vec![0, 1, 2, 3, 4]
    } else {
        cli.split_seeds.clone()
    };
    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "heliosphere_sparse_policy_mainline_{}.toml",
            Utc::now().date_naive()
        ))
    });

    let cache_root = cli.repo_root.join("data/external");
    let mut cubes = Vec::new();
    let mut raw_rows_by_cube = BTreeMap::<String, Vec<data_core::HeliosphereFeatureRow>>::new();
    let mut policy_contexts = BTreeMap::<String, SparsePolicyDatasetContext>::new();
    for cube_csv in &cube_csvs {
        let rows = load_heliosphere_rows(cube_csv)?;
        let cube_name = cube_name(cube_csv);
        let baseline_policies =
            summarize_sparse_policies(&rows, &cache_root, cli.horizon_hours, cli.grid)?;
        let (positive_sample_count, seeded_policies) = summarize_seeded_sparse_policy_rows(
            &rows,
            &cache_root,
            cli.horizon_hours,
            cli.grid,
            *split_seeds.first().unwrap_or(&0),
        )?;
        let aggregates = aggregate_seeded_policies(&seeded_policies);
        cubes.push(CubeEvaluation {
            cube_name: cube_name.clone(),
            cube_csv: cube_csv.display().to_string(),
            positive_sample_count,
            supervised: positive_sample_count > 0,
            baseline_policies,
            seeded_policies,
            aggregates,
            promoted_survives_cube: false,
            blockers: Vec::new(),
        });
        raw_rows_by_cube.insert(cube_name, rows);
    }

    for (cube_name, rows) in &raw_rows_by_cube {
        if let Ok(context) = build_sparse_policy_dataset_context(rows, &cache_root, cli.horizon_hours) {
            policy_contexts.insert(cube_name.clone(), context);
        }
    }

    let reference_cube_name = choose_reference_cube(&cubes);
    let comparator_policy_key = "mission_product_quiet|invariants_only".to_string();
    let promoted_policy_key =
        select_promoted_policy(&cubes, &reference_cube_name, &comparator_policy_key);
    if let Some(reference_context) = policy_contexts.get(&reference_cube_name) {
        for cube in &mut cubes {
            let Some(target_context) = policy_contexts.get(&cube.cube_name) else {
                continue;
            };
            cube.seeded_policies.retain(|row| {
                let key = policy_key(row);
                key != comparator_policy_key
                    && promoted_policy_key.as_deref().is_none_or(|promoted| key != promoted)
            });
            for seed in split_seeds.iter().copied() {
                if let Ok((_positive, comparator_row)) = transferred_seed_row(
                    reference_context,
                    target_context,
                    &comparator_policy_key,
                    cli.horizon_hours,
                    cli.grid,
                    seed,
                ) {
                    cube.seeded_policies.push(comparator_row);
                }
                if let Some(promoted_key) = promoted_policy_key.as_deref()
                    && let Ok((_positive, promoted_row)) = transferred_seed_row(
                        reference_context,
                        target_context,
                        promoted_key,
                        cli.horizon_hours,
                        cli.grid,
                        seed,
                    )
                {
                    cube.seeded_policies.push(promoted_row);
                }
            }
            cube.aggregates = aggregate_seeded_policies(&cube.seeded_policies);
        }
    }
    let mut promotion_survives_all_cubes = promoted_policy_key.is_some();
    for cube in &mut cubes {
        let (survives, blockers) = evaluate_cube_survival(
            cube,
            &comparator_policy_key,
            promoted_policy_key.as_deref(),
        );
        cube.promoted_survives_cube = survives;
        cube.blockers = blockers;
        promotion_survives_all_cubes &= survives;
    }
    let registry_update_performed = false;
    let registry_update_reason = if promotion_survives_all_cubes {
        "Promotion survived all cubes; registry uplift intentionally deferred to a dedicated evidence-linked tranche."
            .to_string()
    } else {
        "No registry claim/insight update performed because the promoted sparse-policy candidate did not survive all cubes/seeds under the configured criteria."
            .to_string()
    };

    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        repo_root: cli.repo_root.display().to_string(),
        horizon_hours: cli.horizon_hours,
        grid: cli.grid,
        split_seeds,
        reference_cube: reference_cube_name,
        comparator_policy_key,
        promoted_policy_key,
        promotion_survives_all_cubes,
        registry_update_performed,
        registry_update_reason,
        cubes,
        notes: vec![
            "The promoted candidate is selected on the reference supervised cube and then evaluated unchanged across all requested cubes and split seeds."
                .to_string(),
            "Supervised cubes use event-label recall under the 12 GiB hard budget as the primary promotion criterion."
                .to_string(),
            "Unsupervised cubes are treated as compression/generalization checks only; they cannot create a positive promotion by themselves."
                .to_string(),
        ],
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", out.display()))?;
    println!("reference_cube = {}", report.reference_cube);
    println!("promoted_policy_key = {}", report.promoted_policy_key.as_deref().unwrap_or("none"));
    println!(
        "promotion_survives_all_cubes = {}",
        report.promotion_survives_all_cubes
    );
    println!("out = {}", out.display());
    Ok(())
}

fn cube_name(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("cube")
        .to_string()
}

fn choose_reference_cube(cubes: &[CubeEvaluation]) -> String {
    cubes.iter()
        .find(|cube| cube.supervised && cube.cube_name.contains("modern2020"))
        .or_else(|| cubes.iter().find(|cube| cube.supervised))
        .map(|cube| cube.cube_name.clone())
        .unwrap_or_else(|| cubes.first().map(|cube| cube.cube_name.clone()).unwrap_or_else(|| "cube".to_string()))
}

fn aggregate_seeded_policies(rows: &[SeededSparsePolicySummary]) -> Vec<PolicyAggregate> {
    let mut grouped = BTreeMap::<String, Vec<&SeededSparsePolicySummary>>::new();
    for row in rows {
        grouped.entry(policy_key(row)).or_default().push(row);
    }
    let mut aggregates = grouped
        .into_iter()
        .map(|(key, rows)| {
            let first = rows[0];
            let mean_lead = mean_optional(
                &rows.iter().map(|row| row.median_lead_time_hours).collect::<Vec<_>>(),
            );
            PolicyAggregate {
                policy_key: key,
                normalization_strategy: first.normalization_strategy.clone(),
                descriptor_profile: first.descriptor_profile.clone(),
                seed_count: rows.len(),
                seeds_within_budget: rows
                    .iter()
                    .filter(|row| row.sparse_bf16_aa_projected_gib <= 12.0)
                    .count(),
                mean_active_fraction: mean(
                    &rows.iter().map(|row| row.active_fraction).collect::<Vec<_>>(),
                ),
                max_active_fraction: rows
                    .iter()
                    .map(|row| row.active_fraction)
                    .fold(0.0, f64::max),
                mean_event_label_recall: mean(
                    &rows
                        .iter()
                        .map(|row| row.event_label_recall)
                        .collect::<Vec<_>>(),
                ),
                max_event_label_recall: rows
                    .iter()
                    .map(|row| row.event_label_recall)
                    .fold(0.0, f64::max),
                mean_event_label_precision: mean(
                    &rows
                        .iter()
                        .map(|row| row.event_label_precision)
                        .collect::<Vec<_>>(),
                ),
                mean_projected_gib: mean(
                    &rows
                        .iter()
                        .map(|row| row.sparse_bf16_aa_projected_gib)
                        .collect::<Vec<_>>(),
                ),
                max_projected_gib: rows
                    .iter()
                    .map(|row| row.sparse_bf16_aa_projected_gib)
                    .fold(0.0, f64::max),
                mean_lead_time_hours: mean_lead,
            }
        })
        .collect::<Vec<_>>();
    aggregates.sort_by(|a, b| a.policy_key.cmp(&b.policy_key));
    aggregates
}

fn select_promoted_policy(
    cubes: &[CubeEvaluation],
    reference_cube_name: &str,
    comparator_policy_key: &str,
) -> Option<String> {
    let reference = cubes.iter().find(|cube| cube.cube_name == reference_cube_name)?;
    if !reference.supervised {
        return None;
    }
    let comparator = reference
        .aggregates
        .iter()
        .find(|row| row.policy_key == comparator_policy_key)?;
    reference
        .aggregates
        .iter()
        .filter(|row| row.descriptor_profile != "invariants_only")
        // Promotion is selected from the reference cube's initial seed-0 sweep.
        // Additional split seeds are evaluated only after a candidate is chosen.
        .filter(|row| row.seeds_within_budget == row.seed_count)
        .filter(|row| row.max_projected_gib <= 12.0)
        .filter(|row| row.mean_event_label_recall > comparator.mean_event_label_recall)
        .max_by(|a, b| {
            a.mean_event_label_recall
                .total_cmp(&b.mean_event_label_recall)
                .then_with(|| b.mean_projected_gib.total_cmp(&a.mean_projected_gib))
        })
        .map(|row| row.policy_key.clone())
}

fn evaluate_cube_survival(
    cube: &CubeEvaluation,
    comparator_policy_key: &str,
    promoted_policy_key: Option<&str>,
) -> (bool, Vec<String>) {
    let mut blockers = Vec::new();
    let Some(promoted_policy_key) = promoted_policy_key else {
        blockers.push("no_promoted_policy_selected".to_string());
        return (false, blockers);
    };
    let Some(comparator) = cube
        .aggregates
        .iter()
        .find(|row| row.policy_key == comparator_policy_key)
    else {
        blockers.push("missing_comparator_policy".to_string());
        return (false, blockers);
    };
    let Some(promoted) = cube
        .aggregates
        .iter()
        .find(|row| row.policy_key == promoted_policy_key)
    else {
        blockers.push("missing_promoted_policy".to_string());
        return (false, blockers);
    };
    if promoted.max_projected_gib > 12.0 {
        blockers.push(format!(
            "promoted_policy_exceeds_budget:max_projected_gib={:.6}",
            promoted.max_projected_gib
        ));
    }
    if promoted.seeds_within_budget < promoted.seed_count {
        blockers.push(format!(
            "promoted_policy_not_budget_stable:{}/{}",
            promoted.seeds_within_budget, promoted.seed_count
        ));
    }
    if cube.supervised {
        if promoted.mean_event_label_recall <= comparator.mean_event_label_recall {
            blockers.push(format!(
                "promoted_recall_not_better:{:.6}<={:.6}",
                promoted.mean_event_label_recall, comparator.mean_event_label_recall
            ));
        }
    } else {
        if promoted.mean_projected_gib > comparator.mean_projected_gib {
            blockers.push(format!(
                "unsupervised_cube_memory_regression:{:.6}>{:.6}",
                promoted.mean_projected_gib, comparator.mean_projected_gib
            ));
        }
    }
    (blockers.is_empty(), blockers)
}

fn transferred_seed_row(
    training_context: &SparsePolicyDatasetContext,
    target_context: &SparsePolicyDatasetContext,
    policy_key: &str,
    horizon_hours: i64,
    grid: usize,
    seed: u64,
) -> Result<(usize, SeededSparsePolicySummary)> {
    let (normalization_strategy, descriptor_profile) = policy_key
        .split_once('|')
        .with_context(|| format!("invalid policy key '{policy_key}'"))?;
    let spec = SparsePolicyTransferSpec {
        horizon_hours,
        grid,
        split_seed: seed,
        normalization_strategy,
        descriptor_profile,
    };
    gororoba_cli_physics::heliosphere_eval::summarize_transferred_seeded_sparse_policy_from_contexts(
        training_context,
        target_context,
        &spec,
    )
}

fn policy_key(row: &SeededSparsePolicySummary) -> String {
    format!("{}|{}", row.normalization_strategy, row.descriptor_profile)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_optional(values: &[Option<f64>]) -> Option<f64> {
    let finite = values.iter().flatten().copied().filter(|value| value.is_finite()).collect::<Vec<_>>();
    if finite.is_empty() {
        None
    } else {
        Some(mean(&finite))
    }
}
