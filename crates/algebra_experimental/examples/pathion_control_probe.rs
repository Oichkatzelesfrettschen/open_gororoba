use algebra_experimental::higher_cd_control::{
    HigherCdBasisReport, PathionControlReport, ZdGraphSpectrumReport,
    default_pathion_control_report,
};
use std::{fs, path::Path};

const OUT_DIR: &str = "data/results/higher_cd/pathion_control";

fn write_metric_csv(path: &Path, rows: &[(String, String)]) -> std::io::Result<()> {
    let mut out = String::from("metric,value\n");
    for (metric, value) in rows {
        out.push_str(&format!("{metric},{value}\n"));
    }
    fs::write(path, out)
}

fn write_basis_summary_csv(path: &Path, report: &HigherCdBasisReport) -> std::io::Result<()> {
    let mut rows = vec![
        ("algebra_name".to_string(), report.algebra_name.clone()),
        ("ambient_dim".to_string(), report.ambient_dim.to_string()),
        (
            "requested_rank".to_string(),
            report.requested_rank.to_string(),
        ),
        ("actual_rank".to_string(), report.actual_rank.to_string()),
        ("basis_cols".to_string(), report.basis_cols.to_string()),
        (
            "assessor_count".to_string(),
            report.assessor_count.to_string(),
        ),
        (
            "effective_rank_1e4".to_string(),
            report.effective_rank_1e4.to_string(),
        ),
        (
            "effective_rank_1e6".to_string(),
            report.effective_rank_1e6.to_string(),
        ),
        (
            "effective_rank_1e8".to_string(),
            report.effective_rank_1e8.to_string(),
        ),
        (
            "effective_rank_1e10".to_string(),
            report.effective_rank_1e10.to_string(),
        ),
        (
            "leading_singular_value".to_string(),
            format!("{:.12}", report.leading_singular_value),
        ),
        (
            "trailing_singular_value".to_string(),
            format!("{:.12}", report.trailing_singular_value),
        ),
    ];
    for (index, value) in report.singular_values.iter().enumerate() {
        rows.push((format!("singular_value_{index}"), format!("{value:.12}")));
    }
    write_metric_csv(path, &rows)
}

fn write_spectrum_csv(path: &Path, report: &ZdGraphSpectrumReport) -> std::io::Result<()> {
    let mut rows = vec![
        ("algebra_name".to_string(), report.algebra_name.clone()),
        ("ambient_dim".to_string(), report.ambient_dim.to_string()),
        ("edge_count".to_string(), report.edge_count.to_string()),
        ("degree_min".to_string(), report.degree_min.to_string()),
        ("degree_max".to_string(), report.degree_max.to_string()),
        (
            "degree_mean".to_string(),
            format!("{:.12}", report.degree_mean),
        ),
        ("n_components".to_string(), report.n_components.to_string()),
        (
            "positive_eigenvalue_count".to_string(),
            report.positive_eigenvalue_count.to_string(),
        ),
    ];
    for (index, value) in report.eigenvalues.iter().enumerate() {
        rows.push((
            format!("laplacian_eigenvalue_{index}"),
            format!("{value:.12}"),
        ));
    }
    for (index, value) in report.eigenvalues_16.iter().enumerate() {
        rows.push((format!("gpu_eigenvalue_{index}"), format!("{value:.12}")));
    }
    write_metric_csv(path, &rows)
}

fn render_summary_markdown(report: &PathionControlReport) -> String {
    format!(
        "# Pathion Control Summary\n\n\
## Summary\n\n\
- Algebra: `{}`\n\
- Ambient dimension: `{}`\n\
- Requested `V_k` rank cap: `{}`\n\
- Actual extracted rank: `{}`\n\
- Assessor count: `{}`\n\
- ZD graph edges: `{}`\n\
- Connected components: `{}`\n\
- Positive Laplacian eigenvalues: `{}`\n\
- Leading/trailing singular values: `{:.12}` / `{:.12}`\n\n\
## Method\n\n\
This control-lane bundle is derived stepwise in pure Rust from\n\
`cd_kernel` sign/associator primitives, `extract_vk_basis`, and the\n\
dimension-parametric control report assembly in\n\
`algebra_experimental::higher_cd_control`.\n\n\
## Interpretation\n\n\
Pathion remains a higher-CD control/falsification lane. These outputs are\n\
derived support artifacts for the Cayley-Dickson stack, not the primary\n\
bridge architecture.\n",
        report.summary.algebra_name,
        report.summary.ambient_dim,
        report.summary.requested_rank,
        report.summary.actual_rank,
        report.summary.assessor_count,
        report.summary.edge_count,
        report.summary.connected_components,
        report.summary.positive_eigenvalue_count,
        report.summary.leading_singular_value,
        report.summary.trailing_singular_value,
    )
}

fn write_pathion_control_artifacts(
    base: &Path,
    report: &PathionControlReport,
) -> std::io::Result<()> {
    fs::create_dir_all(base)?;
    fs::write(
        base.join("pathion_control_report.json"),
        report.to_json_pretty(),
    )?;
    write_basis_summary_csv(&base.join("vk_basis_summary.csv"), &report.basis_report)?;
    write_spectrum_csv(&base.join("zd_graph_spectrum.csv"), &report.spectrum_report)?;
    fs::write(
        base.join("pathion_control_summary.md"),
        render_summary_markdown(report),
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = default_pathion_control_report();
    let out_dir = Path::new(OUT_DIR);
    write_pathion_control_artifacts(out_dir, &report)?;

    println!("{}", report.summary_row());
    println!("artifact_dir={}", out_dir.display());
    Ok(())
}
