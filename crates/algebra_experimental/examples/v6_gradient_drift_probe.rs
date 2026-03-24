use algebra_experimental::neutrino_sector::{
    BranchMapReport, BranchWallReport, LoopReport, PathScanReport, V6ProbeArtifacts,
    default_probe_artifacts,
};
use std::fs;
use std::path::Path;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifacts = default_probe_artifacts();
    let out_dir = Path::new(OUT_DIR);
    write_probe_artifacts(out_dir, &artifacts)?;

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
