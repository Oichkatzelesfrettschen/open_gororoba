//! Publication-ready PGFPlots visualization of E-183/E-192 null result.
//!
//! Generates three .tex panels for inclusion in manga_zd_null_result.tex:
//!
//!   1. spatial_residuals.tex -- stacked profile with baryonic systematics annotated
//!   2. derivative_dft.tex    -- AC-coupled harmonic spectrum (d/dx removes DC offset)
//!   3. stft_spectrogram.tex  -- STFT power vs x-position (baryonic boundary at x~2)
//!
//! Input CSVs come from harmonic-halo-stacking-manga + harmonic-halo-signal-analysis.
//!
//! Reference: C-1377, E-195

use anyhow::Context;
use clap::Parser;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(name = "harmonic-halo-visualize")]
#[command(about = "Generate PGFPlots panels for E-183/E-192 null result manuscript")]
struct Cli {
    /// Stacked profile CSV (x, delta_stack, delta_stack_err, n_contributing).
    #[arg(long)]
    stacked_csv: PathBuf,

    /// STFT spectrogram CSV (x_center, k, power, phase, n_eff).
    #[arg(long)]
    stft_csv: Option<PathBuf>,

    /// Derivative DFT CSV (x_mid, d_delta_dx).
    #[arg(long)]
    deriv_csv: Option<PathBuf>,

    /// Output directory for .tex panels.
    #[arg(long, default_value = "docs/latex/figs")]
    out_dir: PathBuf,
}

/// A stacked profile row.
struct ProfileRow {
    x: f64,
    delta: f64,
    err: f64,
    n: usize,
}

/// A derivative DFT row.
struct DerivRow {
    x_mid: f64,
    d_delta: f64,
}

/// An STFT row.
struct StftRow {
    x_center: f64,
    k: f64,
    power: f64,
}

fn load_profile(path: &Path) -> anyhow::Result<Vec<ProfileRow>> {
    let mut rdr = csv::Reader::from_path(path).context("open stacked CSV")?;
    let mut rows = Vec::new();
    for result in rdr.records() {
        let rec = result?;
        let x: f64 = rec.get(0).unwrap_or("").trim().parse().unwrap_or(f64::NAN);
        let delta: f64 = rec.get(1).unwrap_or("").trim().parse().unwrap_or(f64::NAN);
        let err: f64 = rec.get(2).unwrap_or("").trim().parse().unwrap_or(f64::NAN);
        let n: usize = rec.get(3).unwrap_or("0").trim().parse().unwrap_or(0);
        if x.is_finite() && delta.is_finite() {
            rows.push(ProfileRow { x, delta, err, n });
        }
    }
    Ok(rows)
}

fn load_deriv(path: &Path) -> anyhow::Result<Vec<DerivRow>> {
    let mut rdr = csv::Reader::from_path(path).context("open deriv CSV")?;
    let mut rows = Vec::new();
    for result in rdr.records() {
        let rec = result?;
        let x: f64 = rec.get(0).unwrap_or("").trim().parse().unwrap_or(f64::NAN);
        let d: f64 = rec.get(1).unwrap_or("").trim().parse().unwrap_or(f64::NAN);
        if x.is_finite() && d.is_finite() {
            rows.push(DerivRow {
                x_mid: x,
                d_delta: d,
            });
        }
    }
    Ok(rows)
}

fn load_stft(path: &Path) -> anyhow::Result<Vec<StftRow>> {
    let mut rdr = csv::Reader::from_path(path).context("open STFT CSV")?;
    let mut rows = Vec::new();
    for result in rdr.records() {
        let rec = result?;
        let x: f64 = rec.get(0).unwrap_or("").trim().parse().unwrap_or(f64::NAN);
        let k: f64 = rec.get(1).unwrap_or("").trim().parse().unwrap_or(f64::NAN);
        let pow: f64 = rec.get(2).unwrap_or("").trim().parse().unwrap_or(f64::NAN);
        if x.is_finite() && k.is_finite() && pow.is_finite() {
            rows.push(StftRow {
                x_center: x,
                k,
                power: pow,
            });
        }
    }
    Ok(rows)
}

/// Generate the spatial residuals panel (Panel 3).
fn write_spatial_residuals(rows: &[ProfileRow], out: &Path) -> anyhow::Result<()> {
    let valid: Vec<&ProfileRow> = rows.iter().filter(|r| r.n >= 10).collect();
    if valid.is_empty() {
        anyhow::bail!("No valid rows (n >= 10) in profile");
    }

    // Build coordinate lists
    let coords: String = valid
        .iter()
        .map(|r| format!("({:.4},{:.6})", r.x, r.delta))
        .collect::<Vec<_>>()
        .join(" ");
    let upper: String = valid
        .iter()
        .map(|r| format!("({:.4},{:.6})", r.x, r.delta + 2.0 * r.err))
        .collect::<Vec<_>>()
        .join(" ");
    let lower: String = valid
        .iter()
        .map(|r| format!("({:.4},{:.6})", r.x, r.delta - 2.0 * r.err))
        .collect::<Vec<_>>()
        .join(" ");

    let tex = format!(
        r"\begin{{tikzpicture}}
\begin{{axis}}[
  xlabel={{$x = r/r_s$}},
  ylabel={{$\delta_\mathrm{{stack}}$}},
  width=8cm, height=5cm,
  grid=both, grid style={{dashed,gray!40}},
  ymin=-0.25, ymax=0.35,
  title={{MaNGA E-183 Stacked Profile ($N=6992$)}}
]
% 2-sigma error band
\addplot[fill=blue!15,draw=none,forget plot]
  coordinates {{{upper}}}
  \closedcycle;
\addplot[fill=white,draw=none,forget plot]
  coordinates {{{lower}}}
  \closedcycle;
% Stacked profile
\addplot[blue,thick] coordinates {{{coords}}};
% Zero line (NFW perfect fit)
\addplot[black,dashed,domain=0.4:1.5] {{0}};
% Annotations
\node[font=\tiny,above] at (axis cs:0.55,0.06) {{$+5\%$ bulge}};
\node[font=\tiny,below] at (axis cs:0.75,-0.12) {{$-15\%$ cusp}};
\node[font=\tiny,above] at (axis cs:0.953,0.31) {{$+29\%$ IFU}};
\end{{axis}}
\end{{tikzpicture}}
"
    );

    let mut f = fs::File::create(out)?;
    f.write_all(tex.as_bytes())?;
    eprintln!("Written {:?}", out);
    Ok(())
}

/// Generate the derivative DFT panel (Panel 2).
fn write_derivative_dft(rows: &[DerivRow], out: &Path) -> anyhow::Result<()> {
    use std::f64::consts::PI;
    if rows.is_empty() {
        anyhow::bail!("No derivative rows");
    }
    // Targeted DFT at CD-ZD wavenumbers k_n = 2*pi*n/7
    let n_modes = 7usize;
    let mut powers: Vec<(f64, f64)> = Vec::with_capacity(n_modes);
    for n in 1..=n_modes {
        let k = 2.0 * PI * n as f64 / n_modes as f64;
        let count = rows.len() as f64;
        let re: f64 = rows
            .iter()
            .map(|r| r.d_delta * (k * r.x_mid).cos())
            .sum::<f64>()
            / count;
        let im: f64 = rows
            .iter()
            .map(|r| r.d_delta * (k * r.x_mid).sin())
            .sum::<f64>()
            / count;
        powers.push((k, re * re + im * im));
    }

    // Red-noise reference: P_ref(k) ~ k^{-2} normalized to mode-1
    let p1 = powers[0].1.max(1e-15);
    let ref_coords: String = powers
        .iter()
        .enumerate()
        .map(|(i, (k, _))| {
            let p_ref = p1 / ((i + 1) as f64).powi(2);
            format!("({:.4},{:.6})", k, p_ref)
        })
        .collect::<Vec<_>>()
        .join(" ");

    let bar_coords: String = powers
        .iter()
        .map(|(k, p)| format!("({:.4},{:.6})", k, p))
        .collect::<Vec<_>>()
        .join(" ");

    let tex = format!(
        r"\begin{{tikzpicture}}
\begin{{axis}}[
  xlabel={{Mode wavenumber $k$}},
  ylabel={{DFT power}},
  width=8cm, height=5cm,
  ybar, bar width=0.2,
  xtick={{0.898,1.795,2.693,3.590,4.488,5.386,6.283}},
  xticklabels={{$k_1$,$k_2$,$k_3$,$k_4$,$k_5$,$k_6$,$k_7$}},
  title={{AC-Coupled Derivative DFT (E-192 Null)}}
]
\addplot[blue!70!black,fill=blue!30] coordinates {{{bar_coords}}};
\addplot[red,dashed,thick,mark=none] coordinates {{{ref_coords}}};
\legend{{Derivative DFT power,$k^{{-2}}$ red noise}}
\end{{axis}}
\end{{tikzpicture}}
"
    );

    let mut f = fs::File::create(out)?;
    f.write_all(tex.as_bytes())?;
    eprintln!("Written {:?}", out);
    Ok(())
}

/// Generate the STFT spectrogram panel (Panel 1) as a TikZ matrix plot.
fn write_stft_spectrogram(rows: &[StftRow], out: &Path) -> anyhow::Result<()> {
    if rows.is_empty() {
        // Write a placeholder if no STFT data
        let tex = r"\begin{tikzpicture}
\node {STFT data not provided};
\end{tikzpicture}
";
        let mut f = fs::File::create(out)?;
        f.write_all(tex.as_bytes())?;
        eprintln!("Written placeholder {:?} (no STFT data)", out);
        return Ok(());
    }

    // Collect unique x_center and k values
    let mut x_vals: Vec<f64> = rows.iter().map(|r| r.x_center).collect();
    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    x_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    let mut k_vals: Vec<f64> = rows.iter().map(|r| r.k).collect();
    k_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    k_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-8);

    // Build coordinate list for pgfplots matrix
    let coords: String = rows
        .iter()
        .map(|r| format!("({:.4},{:.4},{:.6})", r.x_center, r.k, r.power))
        .collect::<Vec<_>>()
        .join("\n  ");

    let tex = format!(
        r"\begin{{tikzpicture}}
\begin{{axis}}[
  xlabel={{$x = r/r_s$}},
  ylabel={{Mode $k$}},
  width=8cm, height=5cm,
  colormap/jet,
  title={{STFT Spectrogram: Baryonic Localization at $x < 2$}},
  point meta min=0,
]
\addplot3[surf,shader=flat] coordinates {{
  {coords}
}};
% Baryonic boundary marker
\draw[red,dashed] (axis cs:2.0,\pgfkeysvalueof{{/pgfplots/ymin}}) --
                   (axis cs:2.0,\pgfkeysvalueof{{/pgfplots/ymax}})
  node[above,font=\tiny] {{$x=2$}};
\end{{axis}}
\end{{tikzpicture}}
"
    );

    let mut f = fs::File::create(out)?;
    f.write_all(tex.as_bytes())?;
    eprintln!("Written {:?}", out);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    fs::create_dir_all(&cli.out_dir)?;

    // Load stacked profile (required)
    eprintln!("Loading stacked profile from {:?}...", cli.stacked_csv);
    let profile = load_profile(&cli.stacked_csv)?;
    eprintln!(
        "  {} rows loaded ({} valid)",
        profile.len(),
        profile.iter().filter(|r| r.n >= 10).count()
    );

    // Panel 3: spatial residuals
    let out_spatial = cli.out_dir.join("spatial_residuals.tex");
    write_spatial_residuals(&profile, &out_spatial)
        .context("failed to write spatial_residuals.tex")?;

    // Panel 2: derivative DFT
    let out_deriv = cli.out_dir.join("derivative_dft.tex");
    if let Some(deriv_path) = &cli.deriv_csv {
        let deriv = load_deriv(deriv_path)?;
        write_derivative_dft(&deriv, &out_deriv).context("failed to write derivative_dft.tex")?;
    } else {
        eprintln!("No --deriv-csv provided; writing placeholder derivative_dft.tex");
        let placeholder =
            r"\begin{tikzpicture}\node {Derivative DFT data not provided};\end{tikzpicture}";
        let mut f = fs::File::create(&out_deriv)?;
        f.write_all(placeholder.as_bytes())?;
    }

    // Panel 1: STFT spectrogram
    let out_stft = cli.out_dir.join("stft_spectrogram.tex");
    let stft_rows = if let Some(stft_path) = &cli.stft_csv {
        load_stft(stft_path)?
    } else {
        Vec::new()
    };
    write_stft_spectrogram(&stft_rows, &out_stft)
        .context("failed to write stft_spectrogram.tex")?;

    eprintln!("Done. Three panels written to {:?}", cli.out_dir);
    Ok(())
}
