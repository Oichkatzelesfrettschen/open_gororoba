//! nonlocal-algebraic-metamaterial: build and benchmark C-010 hybrid non-local designs.

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use csv::Writer;
use gororoba_cli_physics::nonlocal_report::{
    build_nonlocal_benchmark_report, write_nonlocal_benchmark_report,
};
use materials_core::{
    M3ProjectionConfig, SyntheticCouplingModel, find_calibration_record, load_calibration_records,
};
use optics_core::{
    absorber_benchmark::{ComparativeBenchmark, CouplingTopology, ProjectionGate, run_benchmark},
    tcmt::KerrCavity,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

const DEFAULT_CALIBRATION_CSV: &str =
    "registry/data/project_csv/canonical/PC-0006_c010_nonlocal_material_calibrations.toml";

#[derive(Parser)]
#[command(
    name = "nonlocal-algebraic-metamaterial",
    about = "Hybrid LC/Floquet synthesis for the C-010 non-local algebraic metamaterial tranche"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build assessor, LC, and Floquet matrices for one literature-backed calibration row.
    BuildMatrix {
        #[arg(long)]
        calibration_id: String,
        #[arg(long, default_value = DEFAULT_CALIBRATION_CSV)]
        calibration_csv: PathBuf,
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },
    /// Benchmark one calibration row against the standard connected C-010 baselines.
    Benchmark {
        #[arg(long)]
        calibration_id: String,
        #[arg(long, default_value = DEFAULT_CALIBRATION_CSV)]
        calibration_csv: PathBuf,
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },
    /// Rank all cited calibration rows by benchmark score and loss penalty.
    CandidateScan {
        #[arg(long, default_value = DEFAULT_CALIBRATION_CSV)]
        calibration_csv: PathBuf,
        #[arg(long, default_value_t = 5)]
        top_n: usize,
        #[arg(long)]
        output_csv: Option<PathBuf>,
    },
}

#[derive(Debug, Clone)]
struct BenchmarkBundle {
    benchmark: ComparativeBenchmark,
    gate: optics_core::absorber_benchmark::ProjectionGateResult,
    average_coupling: f64,
}

#[derive(Debug, Clone)]
struct CandidateRow {
    calibration_id: String,
    platform: String,
    provenance: String,
    benchmark_score: f64,
    pass: bool,
    zd_isolation: f64,
    best_local_isolation: f64,
    dominance_margin: f64,
    zd_crosstalk_db: f64,
    average_coupling: f64,
    citation_key: String,
    citation_url: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Command::BuildMatrix {
            calibration_id,
            calibration_csv,
            output_dir,
        } => {
            let model = load_model(&calibration_csv, &calibration_id)?;
            println!("=== Non-Local Algebraic Metamaterial Matrix Build ===");
            println!(
                "Calibration: {} [{}]",
                model.calibration.id, model.calibration.platform
            );
            println!(
                "Citation: {} <{}>",
                model.calibration.citation_key, model.calibration.citation_url
            );
            println!(
                "Topology: {} assessor nodes, {} masked edges",
                model.topology.nodes.len(),
                model.topology.edge_count()
            );
            println!(
                "Average effective coupling: {:.6}",
                model.average_nonzero_coupling()
            );

            if let Some(dir) = output_dir {
                write_model_matrices(&dir, &model)?;
                println!("Wrote matrices under {}", dir.display());
            }
        }
        Command::Benchmark {
            calibration_id,
            calibration_csv,
            output_dir,
        } => {
            let model = load_model(&calibration_csv, &calibration_id)?;
            let bundle = benchmark_model(&model);
            let report = build_nonlocal_benchmark_report(
                &model,
                &bundle.benchmark,
                &bundle.gate,
                bundle.average_coupling,
            );

            println!("=== C-010 Non-Local Benchmark ===");
            println!(
                "Calibration: {} [{}]",
                report.calibration_id, report.platform
            );
            println!(
                "Citation: {} <{}>",
                report.citation_key, report.citation_url
            );
            println!();
            println!("{}", bundle.benchmark.summary_table());
            println!();
            println!("Projection gate: {}", report.projection_gate_verdict);
            println!("  ZD isolation: {:.6}", report.zd_isolation);
            println!("  Best local isolation: {:.6}", report.best_local_isolation);
            println!("  Dominance margin: {:.6}", report.dominance_margin);
            println!("  ZD crosstalk: {:.1} dB", report.zd_crosstalk_db);
            println!("  Average coupling: {:.6}", report.average_coupling);
            println!();
            println!("Spectral sidecars:");
            println!(
                "  Reggiani partner degeneracies: {:?}",
                report
                    .spectral_crosscheck
                    .reggiani_partner_degeneracies
                    .iter()
                    .map(|row| (row.eigenvalue, row.multiplicity))
                    .collect::<Vec<_>>()
            );
            println!(
                "  ZD crystal flat-band fraction: {:.3}",
                report.spectral_crosscheck.zd_flat_band_fraction
            );
            println!(
                "  Magnonic flat bands: {:?}, min bandwidth {:.4} GHz",
                report.spectral_crosscheck.magnonic_flat_band_indices,
                report.spectral_crosscheck.magnonic_min_bandwidth_ghz
            );
            println!(
                "  Graphene valley check: C_K={:.4}, C_K'={:.4}",
                report.spectral_crosscheck.graphene_valley_chern_k,
                report.spectral_crosscheck.graphene_valley_chern_kprime
            );
            println!(
                "  Sersic layout span (n=4, non-gating): {:.4} .. {:.4}",
                report.spectral_crosscheck.sersic_layout_min,
                report.spectral_crosscheck.sersic_layout_max
            );

            if let Some(dir) = output_dir {
                fs::create_dir_all(&dir)
                    .with_context(|| format!("failed to create {}", dir.display()))?;
                write_model_matrices(&dir, &model)?;
                write_nonlocal_benchmark_report(&dir.join("benchmark_summary.toml"), &report)
                    .map_err(|err| anyhow!(err))?;
                println!("Wrote benchmark bundle under {}", dir.display());
            }
        }
        Command::CandidateScan {
            calibration_csv,
            top_n,
            output_csv,
        } => {
            let records = load_calibration_records(&calibration_csv)
                .map_err(|err| anyhow!(err))
                .with_context(|| format!("failed to load {}", calibration_csv.display()))?;
            let mut rows = Vec::new();
            for record in &records {
                let model = SyntheticCouplingModel::build(
                    &M3ProjectionConfig::c010_hybrid_default(),
                    record,
                )
                .map_err(|err| anyhow!(err))
                .with_context(|| format!("failed to build synthetic model for {}", record.id))?;
                let bundle = benchmark_model(&model);
                let benchmark_score = bundle.gate.dominance_margin + bundle.gate.zd_isolation
                    - 0.25 * record.loss_tangent
                    + 0.05 * record.nonlocality_score;
                rows.push(CandidateRow {
                    calibration_id: record.id.clone(),
                    platform: record.platform.clone(),
                    provenance: record.data_provenance_class.clone(),
                    benchmark_score,
                    pass: bundle.gate.pass,
                    zd_isolation: bundle.gate.zd_isolation,
                    best_local_isolation: bundle.gate.best_local_isolation,
                    dominance_margin: bundle.gate.dominance_margin,
                    zd_crosstalk_db: bundle.gate.zd_crosstalk_db,
                    average_coupling: bundle.average_coupling,
                    citation_key: record.citation_key.clone(),
                    citation_url: record.citation_url.clone(),
                });
            }

            rows.sort_by(|left, right| {
                right
                    .benchmark_score
                    .partial_cmp(&left.benchmark_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            println!("=== C-010 Candidate Scan ===");
            println!(
                "{:<24} {:<16} {:<12} {:>8} {:>8} {:>8} {:>8}",
                "Calibration", "Platform", "Prov.", "Score", "Isol.", "Margin", "Pass"
            );
            println!("{}", "-".repeat(96));
            for row in rows.iter().take(top_n.max(1)) {
                println!(
                    "{:<24} {:<16} {:<12} {:>8.4} {:>8.4} {:>8.4} {:>8}",
                    row.calibration_id,
                    truncate(&row.platform, 16),
                    truncate(&row.provenance, 12),
                    row.benchmark_score,
                    row.zd_isolation,
                    row.dominance_margin,
                    if row.pass { "PASS" } else { "FAIL" },
                );
                println!("  {} <{}>", row.citation_key, row.citation_url);
            }

            if let Some(path) = output_csv {
                write_candidate_scan_csv(&path, &rows)?;
                println!("Wrote {}", path.display());
            }
        }
    }

    Ok(())
}

fn load_model(calibration_csv: &Path, calibration_id: &str) -> Result<SyntheticCouplingModel> {
    let records = load_calibration_records(calibration_csv)
        .map_err(|err| anyhow!(err))
        .with_context(|| format!("failed to load {}", calibration_csv.display()))?;
    let record = find_calibration_record(&records, calibration_id)
        .map_err(|err| anyhow!(err))
        .with_context(|| format!("missing calibration {}", calibration_id))?;
    SyntheticCouplingModel::build(&M3ProjectionConfig::c010_hybrid_default(), record)
        .map_err(|err| anyhow!(err))
        .with_context(|| format!("failed to build model for {}", calibration_id))
}

fn benchmark_model(model: &SyntheticCouplingModel) -> BenchmarkBundle {
    let average_coupling = model.average_nonzero_coupling();
    let suite = vec![
        model.to_coupling_topology(&format!("{}-floquet", model.calibration.id)),
        CouplingTopology::ring(42, average_coupling),
        CouplingTopology::chain(42, average_coupling),
        CouplingTopology::complete(42, average_coupling),
        CouplingTopology::cross_cluster_bridges(7, 6, average_coupling),
    ];
    let benchmark = run_benchmark(
        &suite,
        &benchmark_cavity(),
        1.0e9,
        &[0, 1, 2, 3, 4, 5],
        1.0e-3,
        1.0e-12,
        500,
    );
    let gate = ProjectionGate::c010_default().evaluate(&benchmark);
    BenchmarkBundle {
        benchmark,
        gate,
        average_coupling,
    }
}

fn benchmark_cavity() -> KerrCavity {
    KerrCavity::new(
        2.0 * std::f64::consts::PI * 193.0e12,
        1.0e6,
        5.0e5,
        1.45,
        2.6e-20,
        1.0e-18,
    )
}

fn write_model_matrices(dir: &Path, model: &SyntheticCouplingModel) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    write_bool_matrix_csv(
        &dir.join("assessor_mask.csv"),
        &model.topology.adjacency_mask,
    )?;
    write_f64_matrix_csv(
        &dir.join("m3_assessor_weights.csv"),
        &model.normalized_assessor_weights,
    )?;
    write_f64_matrix_csv(
        &dir.join("lc_admittance_magnitude.csv"),
        &model.lc_admittance.effective_real_coupling,
    )?;
    write_f64_matrix_csv(
        &dir.join("floquet_effective_coupling.csv"),
        &model.floquet_effective.hermitian_real_coupling,
    )?;
    Ok(())
}

fn write_bool_matrix_csv(path: &Path, matrix: &[Vec<bool>]) -> Result<()> {
    let mut writer =
        Writer::from_path(path).with_context(|| format!("failed to create {}", path.display()))?;
    for row in matrix {
        let rendered: Vec<String> = row
            .iter()
            .map(|value| {
                if *value {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            })
            .collect();
        writer
            .write_record(rendered)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn write_f64_matrix_csv(path: &Path, matrix: &[Vec<f64>]) -> Result<()> {
    let mut writer =
        Writer::from_path(path).with_context(|| format!("failed to create {}", path.display()))?;
    for row in matrix {
        let rendered: Vec<String> = row.iter().map(|value| format!("{value:.12}")).collect();
        writer
            .write_record(rendered)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn write_candidate_scan_csv(path: &Path, rows: &[CandidateRow]) -> Result<()> {
    let mut writer =
        Writer::from_path(path).with_context(|| format!("failed to create {}", path.display()))?;
    writer.write_record([
        "calibration_id",
        "platform",
        "provenance",
        "benchmark_score",
        "pass",
        "zd_isolation",
        "best_local_isolation",
        "dominance_margin",
        "zd_crosstalk_db",
        "average_coupling",
        "citation_key",
        "citation_url",
    ])?;
    for row in rows {
        writer.write_record([
            row.calibration_id.as_str(),
            row.platform.as_str(),
            row.provenance.as_str(),
            &format!("{:.12}", row.benchmark_score),
            if row.pass { "true" } else { "false" },
            &format!("{:.12}", row.zd_isolation),
            &format!("{:.12}", row.best_local_isolation),
            &format!("{:.12}", row.dominance_margin),
            &format!("{:.12}", row.zd_crosstalk_db),
            &format!("{:.12}", row.average_coupling),
            row.citation_key.as_str(),
            row.citation_url.as_str(),
        ])?;
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn truncate(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }
    let keep = max_len.saturating_sub(3);
    format!("{}...", &text[..keep])
}
