use pathion_ellip::pathion_resonance::{ResonanceConfig, default_resonance_report};
use std::{fs, path::Path};

const OUT_DIR: &str = "data/results/higher_cd/pathion_resonance";
const DEFAULT_ORBITAL_FREQ: f64 = 1.0;
const DEFAULT_ALPHA: f64 = 1e-6;

fn write_band_csv(
    path: &Path,
    report: &pathion_ellip::pathion_resonance::PathionResonanceReport,
) -> std::io::Result<()> {
    let mut out = String::from("harmonic,zd_index,eigenvalue,detuning,coupling_strength\n");
    for band in &report.bands {
        out.push_str(&format!(
            "{},{},{:.12},{:.12},{:.12}\n",
            band.harmonic, band.zd_index, band.eigenvalue, band.detuning, band.coupling_strength
        ));
    }
    fs::write(path, out)
}

fn render_summary_markdown(
    report: &pathion_ellip::pathion_resonance::PathionResonanceReport,
) -> String {
    format!(
        "# Pathion Resonance Summary\n\n\
## Summary\n\n\
- Algebra: `{}`\n\
- Ambient dimension: `{}`\n\
- Orbital frequency: `{:.12}`\n\
- Alpha scale: `{:.12}`\n\
- Band count: `{}`\n\
- Total coupling: `{:.12}`\n\
- Perturbation: `[{:.12}, {:.12}, {:.12}]`\n\n\
## Method\n\n\
This report is derived in pure Rust from the normalized Pathion control report,\n\
then evaluated through the shared resonance helpers that consume the higher-CD\n\
Laplacian spectrum.\n",
        report.algebra_name,
        report.ambient_dim,
        report.orbital_frequency,
        report.alpha_scale,
        report.bands.len(),
        report.total_coupling,
        report.perturbation[0],
        report.perturbation[1],
        report.perturbation[2]
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ResonanceConfig::default();
    let report = default_resonance_report(DEFAULT_ORBITAL_FREQ, DEFAULT_ALPHA, &config);
    let out_dir = Path::new(OUT_DIR);
    fs::create_dir_all(out_dir)?;
    fs::write(
        out_dir.join("pathion_resonance_report.json"),
        report.to_json_pretty(),
    )?;
    write_band_csv(&out_dir.join("pathion_resonance_bands.csv"), &report)?;
    fs::write(
        out_dir.join("pathion_resonance_summary.md"),
        render_summary_markdown(&report),
    )?;

    println!("{}", report.summary_row());
    println!("artifact_dir={}", out_dir.display());
    Ok(())
}
