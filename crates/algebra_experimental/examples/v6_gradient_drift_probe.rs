use algebra_experimental::neutrino_sector::{
    BranchMapReport, BranchWallReport, LoopReport, PathScanReport, V6ProbeArtifacts,
    V6ProbeSummary, default_probe_artifacts, summarize_probe_artifacts,
};
use std::{fs, path::Path};

const OUT_DIR: &str = "data/results/neutrino_sector/v6_branch_transport";

fn write_branch_map_csv(path: &Path, report: &BranchMapReport) -> std::io::Result<()> {
    let mut out = String::from(
        "alpha_ch,alpha_nu,perm_match,perm_u,perm_d,align_g12,align_g13,align_g23,align_u_solar,align_u_atmo,abs_align_g23,abs_align_u_atmo\n",
    );
    for row in &report.rows {
        out.push_str(&format!(
            "{:.2},{:.2},{},{:?},{:?},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            row.alpha_ch,
            row.alpha_nu,
            row.perm_match,
            row.perm_u,
            row.perm_d,
            row.align_g12,
            row.align_g13,
            row.align_g23,
            row.align_u_solar,
            row.align_u_atmo,
            row.abs_align_g23,
            row.abs_align_u_atmo
        ));
    }
    fs::write(path, out)
}

fn write_branch_walls_csv(path: &Path, report: &BranchWallReport) -> std::io::Result<()> {
    let mut out = String::from(
        "kind,alpha_ch_0,alpha_nu_0,alpha_ch_1,alpha_nu_1,branch_0,branch_1,align_g23,align_u_atmo\n",
    );
    for row in &report.rows {
        out.push_str(&format!(
            "{},{:.2},{:.2},{:.2},{:.2},{},{},{:.6},{:.6}\n",
            row.kind,
            row.alpha_ch_0,
            row.alpha_nu_0,
            row.alpha_ch_1,
            row.alpha_nu_1,
            row.branch_0,
            row.branch_1,
            row.align_g23,
            row.align_u_atmo
        ));
    }
    fs::write(path, out)
}

fn write_loop_csv(path: &Path, report: &LoopReport) -> std::io::Result<()> {
    let mut out = String::from(
        "step,alpha_ch,alpha_nu,branch,wall_crossed,flip_g12,flip_g13,flip_g23,flip_u_solar,flip_u_atmo,align_g23,align_u_atmo\n",
    );
    for row in &report.steps {
        out.push_str(&format!(
            "{},{:.2},{:.2},{},{},{},{},{},{},{},{:.6},{:.6}\n",
            row.step,
            row.alpha_ch,
            row.alpha_nu,
            row.branch,
            row.wall_crossed,
            row.flip_g12,
            row.flip_g13,
            row.flip_g23,
            row.flip_u_solar,
            row.flip_u_atmo,
            row.align_g23,
            row.align_u_atmo
        ));
    }
    fs::write(path, out)
}

fn write_path_scan_csv(path: &Path, report: &PathScanReport) -> std::io::Result<()> {
    let mut out = String::from(
        "alpha_ch,alpha_nu,branch,perm_match,align_g12,align_g13,align_g23,align_u_solar,align_u_atmo\n",
    );
    for row in &report.rows {
        out.push_str(&format!(
            "{:.2},{:.2},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            row.alpha_ch,
            row.alpha_nu,
            row.branch,
            row.perm_match,
            row.align_g12,
            row.align_g13,
            row.align_g23,
            row.align_u_solar,
            row.align_u_atmo
        ));
    }
    fs::write(path, out)
}

fn write_probe_artifacts(base: &Path, artifacts: &V6ProbeArtifacts) -> std::io::Result<()> {
    fs::create_dir_all(base)?;
    fs::write(
        base.join("probe_artifacts.json"),
        serde_json::to_string_pretty(artifacts).expect("serialize V6 probe artifacts"),
    )?;
    fs::write(
        base.join("loop_summaries.json"),
        serde_json::to_string_pretty(&vec![
            &artifacts.stable_branch_loop.summary,
            &artifacts.wall_crossing_loop.summary,
        ])
        .expect("serialize loop summaries"),
    )?;
    write_branch_map_csv(&base.join("branch_map.csv"), &artifacts.branch_map)?;
    write_branch_walls_csv(&base.join("branch_walls.csv"), &artifacts.branch_walls)?;
    write_path_scan_csv(
        &base.join("fixed_alpha_ch_3_00_scan.csv"),
        &artifacts.fixed_alpha_ch_scan,
    )?;
    write_path_scan_csv(
        &base.join("fixed_alpha_nu_1_35_scan.csv"),
        &artifacts.fixed_alpha_nu_scan,
    )?;
    write_loop_csv(
        &base.join("stable_branch_loop_steps.csv"),
        &artifacts.stable_branch_loop,
    )?;
    write_loop_csv(
        &base.join("wall_crossing_loop_steps.csv"),
        &artifacts.wall_crossing_loop,
    )?;
    Ok(())
}

fn render_summary_markdown(summary: &V6ProbeSummary) -> String {
    format!(
        "# V6 Branch Transport Summary\n\n\
Base branch: `{}`\n\n\
## Key Metrics\n\n\
- Branch-map rows: `{}`\n\
- Same-branch points: `{}`\n\
- Switched-branch points: `{}`\n\
- Branch-wall edges: `{}`\n\
- Stable-loop wall crossings: `{}`\n\
- Wall-crossing-loop wall crossings: `{}`\n\
- Stable loop closes after transport: `{}`\n\
- Wall-crossing loop closes after transport: `{}`\n\
- Fixed `alpha_nu = 1.35` strip stays on base branch: `{}`\n\
- Fixed `alpha_ch = 3.00` path shows low-`alpha_nu` wall: `{}`\n\
- Singular value min/max/spread: `{:.12}` / `{:.12}` / `{:.12}`\n\n\
## Method\n\n\
These numbers are derived directly from the pure-Rust branch-transport lane in\n\
`algebra_experimental::neutrino_sector::branch_transport`. The structural `V_6`\n\
subspace is extracted once, gradient-selected frames are recomputed at each\n\
parameter point, and loop closure is evaluated after sign-consistent transport.\n\n\
## Interpretation\n\n\
The current evidence supports a stable structural `V_6` subspace and a highly\n\
stable local gradient frame in the main branch. The dominant warnings arise from\n\
discrete permutation-branch walls with associated gauge/sign flips, not from\n\
detected residual monodromy in the tested loops.\n",
        summary.base_branch,
        summary.branch_map_rows,
        summary.same_perm_points,
        summary.switched_perm_points,
        summary.branch_wall_count,
        summary.stable_loop_wall_crossings,
        summary.wall_loop_wall_crossings,
        summary.stable_loop_closes,
        summary.wall_loop_closes,
        summary.fixed_alpha_nu_scan_all_match,
        summary.fixed_alpha_ch_scan_has_low_alpha_wall,
        summary.singular_value_min,
        summary.singular_value_max,
        summary.singular_value_spread
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifacts = default_probe_artifacts();
    let summary = summarize_probe_artifacts(&artifacts);
    let out_dir = Path::new(OUT_DIR);
    write_probe_artifacts(out_dir, &artifacts)?;
    fs::write(
        out_dir.join("summary_metrics.json"),
        serde_json::to_string_pretty(&summary).expect("serialize V6 summary metrics"),
    )?;
    fs::write(
        out_dir.join("SUMMARY.md"),
        render_summary_markdown(&summary),
    )?;

    println!(
        "v6_branch_transport_artifacts branch_map_rows={} wall_count={} stable_loop_walls={} wall_loop_walls={}",
        artifacts.branch_map.rows.len(),
        artifacts.branch_walls.count,
        artifacts.stable_branch_loop.summary.wall_crossings,
        artifacts.wall_crossing_loop.summary.wall_crossings
    );
    println!("artifact_dir={}", out_dir.display());
    Ok(())
}
