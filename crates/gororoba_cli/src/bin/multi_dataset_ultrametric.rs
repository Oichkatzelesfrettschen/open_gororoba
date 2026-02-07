//! Cross-Dataset Ultrametric Analysis
//!
//! Runs Direction 2 (multi-attribute Euclidean ultrametricity) on every
//! available astrophysical catalog:
//!
//! - CHIME/FRB Cat 2: (log_DM, gl, gb)
//! - ATNF Pulsars: (log_DM, gl, gb)
//! - GWOSC GW Events: (log_chirp_mass, z, q, chi_eff) -- 219 events O1-O4a
//! - Pantheon+ SN Ia: (z, mu, x1, c)
//! - Gaia DR3 Nearby Stars: (parallax, pmra, pmdec, rv, G_mag, bp_rp)
//! - SDSS DR18 Quasars: (z, u, g, r, i)
//! - Fermi GBM GRBs: (log_t90, log_fluence, ra, dec) -- 4203 GRBs
//!
//! For each dataset, normalizes attributes to [0,1], computes Euclidean
//! distance matrices, and runs the ultrametric fraction test against a
//! column-shuffled null. Reports p-values per dataset with CSV output.
//!
//! Usage:
//!   multi-dataset-ultrametric                     # Use default data paths
//!   multi-dataset-ultrametric --data-dir /path    # Custom data directory
//!   multi-dataset-ultrametric --json              # JSON output

use clap::Parser;
use rand::prelude::*;
use rand::SeedableRng;
use std::path::{Path, PathBuf};

use stats_core::ultrametric::baire::{
    AttributeSpec, BaireEncoder, BaireTestResult, euclidean_ultrametric_test,
};

#[derive(Parser)]
#[command(name = "multi-dataset-ultrametric")]
#[command(about = "Cross-dataset ultrametric analysis on all available catalogs")]
struct Args {
    /// Data directory for downloaded files.
    #[arg(long, default_value = "data/external")]
    data_dir: String,

    /// Number of triples for ultrametric fraction test.
    #[arg(long, default_value = "100000")]
    n_triples: usize,

    /// Number of permutations for null distribution.
    #[arg(long, default_value = "200")]
    n_permutations: usize,

    /// Output CSV path.
    #[arg(long, default_value = "data/csv/c071g_multi_dataset_ultrametric.csv")]
    output: PathBuf,

    /// Output as JSON.
    #[arg(long)]
    json: bool,
}

/// Loader function type: returns (data rows, attribute specs) or None.
type LoaderFn = Box<dyn Fn(&Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)>>;

/// Result row for one dataset analysis.
struct DatasetResult {
    name: String,
    n_objects: usize,
    attributes: String,
    um_fraction: f64,
    null_mean: f64,
    null_std: f64,
    p_value: f64,
    verdict: String,
}

fn load_chime_frb(dir: &Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)> {
    let path = dir.join("chime_frb_cat2.csv");
    if !path.exists() {
        return None;
    }
    let events = data_core::catalogs::chime::parse_chime_csv(&path).ok()?;
    let mut rows = Vec::new();
    for e in &events {
        if e.bonsai_dm.is_nan() || e.gl.is_nan() || e.gb.is_nan() {
            continue;
        }
        if e.bonsai_dm <= 0.0 {
            continue;
        }
        rows.push(vec![e.bonsai_dm.log10(), e.gl, e.gb]);
    }
    if rows.len() < 10 {
        return None;
    }
    let specs = vec![
        attr("log_DM", &rows, 0, false),
        attr("gl", &rows, 1, false),
        attr("gb", &rows, 2, false),
    ];
    Some((rows, specs))
}

fn load_atnf_pulsars(dir: &Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)> {
    let path = dir.join("atnf_pulsars.csv");
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&path).ok()?;
    let mut rows = Vec::new();
    let mut header_seen = false;
    let mut idx_dm = None;
    let mut idx_gl = None;
    let mut idx_gb = None;

    // Detect delimiter: semicolon (ATNF native) or comma (our CSV)
    let first_data_line = content.lines().find(|l| {
        let t = l.trim();
        !t.is_empty() && !t.starts_with('#')
    });
    let delim = if first_data_line.is_some_and(|l| l.contains(';')) {
        ';'
    } else {
        ','
    };

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !header_seen {
            let cols: Vec<&str> = trimmed.split(delim).collect();
            for (i, c) in cols.iter().enumerate() {
                let lc = c.trim().to_lowercase();
                if lc == "dm" {
                    idx_dm = Some(i);
                } else if lc == "gl" {
                    idx_gl = Some(i);
                } else if lc == "gb" {
                    idx_gb = Some(i);
                }
            }
            header_seen = true;
            continue;
        }
        let fields: Vec<&str> = trimmed.split(delim).collect();
        let dm = idx_dm.and_then(|i| fields.get(i)).and_then(|s| s.trim().parse::<f64>().ok());
        let gl = idx_gl.and_then(|i| fields.get(i)).and_then(|s| s.trim().parse::<f64>().ok());
        let gb = idx_gb.and_then(|i| fields.get(i)).and_then(|s| s.trim().parse::<f64>().ok());
        if let (Some(dm), Some(gl), Some(gb)) = (dm, gl, gb) {
            if dm > 0.0 {
                rows.push(vec![dm.log10(), gl, gb]);
            }
        }
    }
    if rows.len() < 10 {
        return None;
    }
    let specs = vec![
        attr("log_DM", &rows, 0, false),
        attr("gl", &rows, 1, false),
        attr("gb", &rows, 2, false),
    ];
    Some((rows, specs))
}

fn load_gwtc3(dir: &Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)> {
    // Try the full GWOSC catalog first, fall back to GWTC-3 confident
    let gwosc_path = dir.join("gwosc_all_events.csv");
    let gwtc3_path = dir.join("GWTC-3_confident.csv");

    if gwosc_path.exists() {
        // Parse the full GWOSC catalog (CSV with standard column names)
        return load_gwosc_csv(&gwosc_path);
    }
    if !gwtc3_path.exists() {
        return None;
    }
    let events = data_core::catalogs::gwtc::parse_gwtc3_csv(&gwtc3_path).ok()?;
    let mut rows = Vec::new();
    for e in &events {
        if e.chirp_mass_source.is_nan() || e.redshift.is_nan()
            || e.mass_1_source.is_nan() || e.mass_2_source.is_nan()
        {
            continue;
        }
        if e.chirp_mass_source <= 0.0 || e.mass_1_source <= 0.0 {
            continue;
        }
        let q = e.mass_2_source / e.mass_1_source;
        rows.push(vec![e.chirp_mass_source.log10(), e.redshift, q]);
    }
    if rows.len() < 10 {
        return None;
    }
    let specs = vec![
        attr("log_chirp_mass", &rows, 0, false),
        attr("z", &rows, 1, false),
        attr("q", &rows, 2, false),
    ];
    Some((rows, specs))
}

/// Load the combined GWTC catalog from GWOSC (219 unique events, O1 through O4a).
fn load_gwosc_csv(path: &Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut rows = Vec::new();
    let mut header_seen = false;
    let mut idx_m1 = None;
    let mut idx_m2 = None;
    let mut idx_mc = None;
    let mut idx_z = None;
    let mut idx_chi = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !header_seen {
            let cols: Vec<&str> = trimmed.split(',').collect();
            for (i, c) in cols.iter().enumerate() {
                let lc = c.trim().to_lowercase();
                match lc.as_str() {
                    "mass_1_source" => idx_m1 = Some(i),
                    "mass_2_source" => idx_m2 = Some(i),
                    "chirp_mass_source" => idx_mc = Some(i),
                    "redshift" => idx_z = Some(i),
                    "chi_eff" => idx_chi = Some(i),
                    _ => {}
                }
            }
            header_seen = true;
            continue;
        }
        let fields: Vec<&str> = trimmed.split(',').collect();
        let parse = |idx: Option<usize>| -> Option<f64> {
            idx.and_then(|i| fields.get(i))
                .and_then(|s| {
                    let s = s.trim();
                    if s.is_empty() || s == "None" { None } else { s.parse().ok() }
                })
        };
        let mc = parse(idx_mc);
        let z = parse(idx_z);
        let m1 = parse(idx_m1);
        let m2 = parse(idx_m2);
        let chi = parse(idx_chi);

        if let (Some(mc), Some(z), Some(m1), Some(m2)) = (mc, z, m1, m2) {
            if mc > 0.0 && m1 > 0.0 {
                let q = m2 / m1;
                let chi_val = chi.unwrap_or(0.0);
                rows.push(vec![mc.log10(), z, q, chi_val]);
            }
        }
    }
    if rows.len() < 10 {
        return None;
    }
    let specs = vec![
        attr("log_chirp_mass", &rows, 0, false),
        attr("z", &rows, 1, false),
        attr("q", &rows, 2, false),
        attr("chi_eff", &rows, 3, false),
    ];
    Some((rows, specs))
}

fn load_pantheon(dir: &Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)> {
    let path = dir.join("PantheonPlusSH0ES.dat");
    if !path.exists() {
        return None;
    }
    let sne = data_core::catalogs::pantheon::parse_pantheon_dat(&path).ok()?;
    let mut rows = Vec::new();
    for s in &sne {
        if s.z_cmb.is_nan() || s.mu.is_nan() || s.x1.is_nan() || s.c.is_nan() {
            continue;
        }
        if s.z_cmb < 0.01 {
            continue;
        }
        rows.push(vec![s.z_cmb, s.mu, s.x1, s.c]);
    }
    if rows.len() < 10 {
        return None;
    }
    let specs = vec![
        attr("z_cmb", &rows, 0, false),
        attr("mu", &rows, 1, false),
        attr("x1", &rows, 2, false),
        attr("c", &rows, 3, false),
    ];
    Some((rows, specs))
}

fn load_gaia(dir: &Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)> {
    let path = dir.join("gaia_dr3_nearby.csv");
    if !path.exists() {
        return None;
    }
    let stars = data_core::catalogs::gaia::parse_gaia_csv(&path).ok()?;
    let mut rows = Vec::new();
    for s in &stars {
        if s.parallax.is_nan()
            || s.pmra.is_nan()
            || s.pmdec.is_nan()
            || s.radial_velocity.is_nan()
            || s.phot_g_mean_mag.is_nan()
            || s.bp_rp.is_nan()
        {
            continue;
        }
        rows.push(vec![
            s.parallax,
            s.pmra,
            s.pmdec,
            s.radial_velocity,
            s.phot_g_mean_mag,
            s.bp_rp,
        ]);
    }
    if rows.len() < 10 {
        return None;
    }
    let specs = vec![
        attr("parallax", &rows, 0, false),
        attr("pmra", &rows, 1, false),
        attr("pmdec", &rows, 2, false),
        attr("rv", &rows, 3, false),
        attr("G_mag", &rows, 4, false),
        attr("bp_rp", &rows, 5, false),
    ];
    Some((rows, specs))
}

/// Parse sexagesimal RA "HH MM SS.S" to decimal degrees.
fn parse_ra_sexa(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 {
        return s.trim().parse().ok();
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let sec: f64 = if parts.len() >= 3 { parts[2].parse().unwrap_or(0.0) } else { 0.0 };
    Some((h + m / 60.0 + sec / 3600.0) * 15.0)
}

/// Parse sexagesimal Dec "[+-]DD MM SS" to decimal degrees.
fn parse_dec_sexa(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 {
        return s.trim().parse().ok();
    }
    let d: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let sec: f64 = if parts.len() >= 3 { parts[2].parse().unwrap_or(0.0) } else { 0.0 };
    let sign = if d < 0.0 || parts[0].starts_with('-') { -1.0 } else { 1.0 };
    Some(sign * (d.abs() + m / 60.0 + sec / 3600.0))
}

fn load_fermi_gbm(dir: &Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)> {
    let path = dir.join("fermi_gbm_grbs.csv");
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&path).ok()?;
    let mut rows = Vec::new();
    let mut header_seen = false;
    let mut idx_t90 = None;
    let mut idx_fluence = None;
    let mut idx_ra = None;
    let mut idx_dec = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !header_seen {
            let cols: Vec<&str> = trimmed.split(',').collect();
            for (i, c) in cols.iter().enumerate() {
                let lc = c.trim().to_lowercase();
                match lc.as_str() {
                    "t90" => idx_t90 = Some(i),
                    "fluence" => idx_fluence = Some(i),
                    "ra" => idx_ra = Some(i),
                    "dec" => idx_dec = Some(i),
                    _ => {}
                }
            }
            header_seen = true;
            continue;
        }
        let fields: Vec<&str> = trimmed.split(',').collect();
        let parse_num = |idx: Option<usize>| -> Option<f64> {
            idx.and_then(|i| fields.get(i))
                .and_then(|s| {
                    let s = s.trim();
                    if s.is_empty() || s == "NaN" { None } else { s.parse().ok() }
                })
        };
        let t90 = parse_num(idx_t90);
        let fluence = parse_num(idx_fluence);
        // RA/Dec are sexagesimal in HEASARC Fermi GBM data
        let ra = idx_ra
            .and_then(|i| fields.get(i))
            .and_then(|s| parse_ra_sexa(s));
        let dec = idx_dec
            .and_then(|i| fields.get(i))
            .and_then(|s| parse_dec_sexa(s));

        if let (Some(t90), Some(fluence), Some(ra), Some(dec)) = (t90, fluence, ra, dec) {
            if t90 > 0.0 && fluence > 0.0 {
                rows.push(vec![t90.log10(), fluence.log10(), ra, dec]);
            }
        }
    }
    if rows.len() < 10 {
        return None;
    }
    let specs = vec![
        attr("log_t90", &rows, 0, false),
        attr("log_fluence", &rows, 1, false),
        attr("ra", &rows, 2, false),
        attr("dec", &rows, 3, false),
    ];
    Some((rows, specs))
}

fn load_mcgill(dir: &Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)> {
    let path = dir.join("mcgill_magnetars.csv");
    if !path.exists() {
        return None;
    }
    // McGill RA/Dec columns are sexagesimal -- the generic parser treats them
    // as NaN, so we parse the CSV directly with sexagesimal conversion.
    let content = std::fs::read_to_string(&path).ok()?;
    let mut rows = Vec::new();
    let mut header_seen = false;
    let mut idx_ra = None;
    let mut idx_dec = None;
    let mut idx_period = None;
    let mut idx_b = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !header_seen {
            let cols: Vec<&str> = trimmed.split(',').collect();
            for (i, c) in cols.iter().enumerate() {
                let lc = c.trim().to_lowercase();
                if lc == "ra" { idx_ra = Some(i); }
                if lc == "decl" || lc == "dec" { idx_dec = Some(i); }
                if lc == "period" { idx_period = Some(i); }
                if lc == "b" || lc == "b_dipole" { idx_b = Some(i); }
            }
            header_seen = true;
            continue;
        }
        let fields: Vec<&str> = trimmed.split(',').collect();
        let ra = idx_ra
            .and_then(|i| fields.get(i))
            .and_then(|s| parse_ra_sexa(s));
        let dec = idx_dec
            .and_then(|i| fields.get(i))
            .and_then(|s| parse_dec_sexa(s));
        let period = idx_period
            .and_then(|i| fields.get(i))
            .and_then(|s| { let s = s.trim(); s.parse::<f64>().ok() });
        let b = idx_b
            .and_then(|i| fields.get(i))
            .and_then(|s| { let s = s.trim(); s.parse::<f64>().ok() });

        if let (Some(ra), Some(dec), Some(period), Some(b)) = (ra, dec, period, b) {
            if period > 0.0 && b > 0.0 {
                rows.push(vec![period.log10(), b.log10(), ra, dec]);
            }
        }
    }
    if rows.len() < 10 {
        return None;
    }
    let specs = vec![
        attr("log_period", &rows, 0, false),
        attr("log_B_dipole", &rows, 1, false),
        attr("ra", &rows, 2, false),
        attr("dec", &rows, 3, false),
    ];
    Some((rows, specs))
}

fn load_sdss(dir: &Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)> {
    let path = dir.join("sdss_dr18_quasars.csv");
    if !path.exists() {
        return None;
    }
    let qsos = data_core::catalogs::sdss::parse_sdss_quasar_csv(&path).ok()?;
    let mut rows = Vec::new();
    for q in &qsos {
        if q.z.is_nan() || q.mag_u.is_nan() || q.mag_g.is_nan()
            || q.mag_r.is_nan() || q.mag_i.is_nan()
        {
            continue;
        }
        if q.z < 0.1 {
            continue;
        }
        rows.push(vec![q.z, q.mag_u, q.mag_g, q.mag_r, q.mag_i]);
    }
    if rows.len() < 10 {
        return None;
    }
    let specs = vec![
        attr("z", &rows, 0, false),
        attr("u_mag", &rows, 1, false),
        attr("g_mag", &rows, 2, false),
        attr("r_mag", &rows, 3, false),
        attr("i_mag", &rows, 4, false),
    ];
    Some((rows, specs))
}

fn load_hipparcos(dir: &Path) -> Option<(Vec<Vec<f64>>, Vec<AttributeSpec>)> {
    let path = dir.join("hip_main.dat");
    if !path.exists() {
        return None;
    }
    // Hipparcos main catalog is pipe-delimited with 78 fields.
    // Field indices (0-based after splitting on '|'):
    //   8: RA (degrees)    9: Dec (degrees)
    //  11: parallax (mas) 12: pmRA (mas/yr)  13: pmDec (mas/yr)
    //   5: Vmag
    let content = std::fs::read_to_string(&path).ok()?;
    let mut rows = Vec::new();
    for line in content.lines() {
        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() < 14 {
            continue;
        }
        let parse = |s: &str| -> Option<f64> {
            let s = s.trim();
            if s.is_empty() { None } else { s.parse().ok() }
        };
        let (Some(plx), Some(pmra), Some(pmdec), Some(vmag), Some(ra), Some(dec)) =
            (parse(fields[11]), parse(fields[12]), parse(fields[13]),
             parse(fields[5]), parse(fields[8]), parse(fields[9]))
        else {
            continue;
        };
        // Filter: positive parallax, reasonable magnitude
        if plx <= 0.0 || !(-2.0..=20.0).contains(&vmag) {
            continue;
        }
        rows.push(vec![plx, pmra, pmdec, vmag, ra, dec]);
    }
    if rows.len() < 10 {
        return None;
    }
    let specs = vec![
        attr("parallax", &rows, 0, false),
        attr("pmra", &rows, 1, false),
        attr("pmdec", &rows, 2, false),
        attr("Vmag", &rows, 3, false),
        attr("ra", &rows, 4, false),
        attr("dec", &rows, 5, false),
    ];
    Some((rows, specs))
}

/// Build an AttributeSpec from data column range.
fn attr(name: &str, data: &[Vec<f64>], col: usize, log_scale: bool) -> AttributeSpec {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for row in data {
        let v = row[col];
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    // Small epsilon to avoid zero range
    if (max - min).abs() < 1e-15 {
        max = min + 1.0;
    }
    AttributeSpec {
        name: name.to_string(),
        min,
        max,
        log_scale,
    }
}

fn run_test(
    name: &str,
    data: &[Vec<f64>],
    specs: &[AttributeSpec],
    n_triples: usize,
    n_perms: usize,
) -> DatasetResult {
    eprintln!(
        "  Testing {} ({} objects, {} attributes)...",
        name,
        data.len(),
        specs.len()
    );

    // For large datasets, subsample to keep runtime reasonable
    let max_n = 5000;
    let (test_data, actual_n) = if data.len() > max_n {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(max_n);
        let sub: Vec<Vec<f64>> = indices.iter().map(|&i| data[i].clone()).collect();
        (sub, max_n)
    } else {
        (data.to_vec(), data.len())
    };

    // Rebuild specs from subsampled data
    let sub_specs: Vec<AttributeSpec> = specs
        .iter()
        .enumerate()
        .map(|(col, s)| {
            let mut a = attr(&s.name, &test_data, col, s.log_scale);
            a.log_scale = s.log_scale;
            a
        })
        .collect();

    let encoder = BaireEncoder::new(sub_specs, 10, 4);
    let result: BaireTestResult =
        euclidean_ultrametric_test(&encoder, &test_data, n_triples, n_perms, 12345);

    let verdict = if result.p_value <= 0.005 {
        "PASS"
    } else {
        "FAIL"
    };

    eprintln!(
        "    N={}, frac={:.4}, null={:.4}, p={:.3} -> {}",
        actual_n, result.ultrametric_fraction, result.null_fraction_mean,
        result.p_value, verdict
    );

    DatasetResult {
        name: name.to_string(),
        n_objects: actual_n,
        attributes: specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join("+"),
        um_fraction: result.ultrametric_fraction,
        null_mean: result.null_fraction_mean,
        null_std: result.null_fraction_std,
        p_value: result.p_value,
        verdict: verdict.to_string(),
    }
}

fn main() {
    let args = Args::parse();
    let dir = Path::new(&args.data_dir);

    eprintln!("=== Cross-Dataset Ultrametric Analysis ===");
    eprintln!("  Direction 2: Multi-Attribute Euclidean Ultrametricity");
    eprintln!();

    let mut results: Vec<DatasetResult> = Vec::new();

    // Load and test each dataset
    let loaders: Vec<(&str, LoaderFn)> = vec![
        ("CHIME/FRB Cat 2", Box::new(load_chime_frb)),
        ("ATNF Pulsars", Box::new(load_atnf_pulsars)),
        ("McGill Magnetars", Box::new(load_mcgill)),
        ("GWOSC GW Events", Box::new(load_gwtc3)),
        ("Pantheon+ SN Ia", Box::new(load_pantheon)),
        ("Gaia DR3 Stars", Box::new(load_gaia)),
        ("SDSS DR18 Quasars", Box::new(load_sdss)),
        ("Fermi GBM GRBs", Box::new(load_fermi_gbm)),
        ("Hipparcos Stars", Box::new(load_hipparcos)),
    ];

    for (name, loader) in &loaders {
        match loader(dir) {
            Some((data, specs)) => {
                let r = run_test(name, &data, &specs, args.n_triples, args.n_permutations);
                results.push(r);
            }
            None => {
                eprintln!("  {} -- data not available, skipping", name);
            }
        }
    }

    eprintln!();

    // Output results
    if args.json {
        print_json(&results);
    } else {
        print_table(&results);
    }

    // Write CSV
    write_csv(&args.output, &results);
}

fn print_table(results: &[DatasetResult]) {
    println!(
        "{:<22} {:>7} {:<30} {:>10} {:>10} {:>8} {:>8}",
        "Dataset", "N", "Attributes", "UM_frac", "Null_mean", "p-value", "Verdict"
    );
    println!("{}", "-".repeat(100));
    for r in results {
        println!(
            "{:<22} {:>7} {:<30} {:>10.4} {:>10.4} {:>8.3} {:>8}",
            r.name, r.n_objects, r.attributes, r.um_fraction, r.null_mean,
            r.p_value, r.verdict
        );
    }
    println!();

    let pass_count = results.iter().filter(|r| r.verdict == "PASS").count();
    let total = results.len();
    println!(
        "Summary: {}/{} datasets show significant ultrametric structure (p<=0.005)",
        pass_count, total
    );
}

fn print_json(results: &[DatasetResult]) {
    println!("[");
    for (i, r) in results.iter().enumerate() {
        let comma = if i < results.len() - 1 { "," } else { "" };
        println!(
            "  {{\"dataset\": \"{}\", \"n\": {}, \"attributes\": \"{}\", \
             \"um_fraction\": {:.6}, \"null_mean\": {:.6}, \"null_std\": {:.6}, \
             \"p_value\": {:.4}, \"verdict\": \"{}\"}}{}",
            r.name, r.n_objects, r.attributes, r.um_fraction, r.null_mean,
            r.null_std, r.p_value, r.verdict, comma
        );
    }
    println!("]");
}

fn write_csv(path: &Path, results: &[DatasetResult]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out = String::from("dataset,n_objects,attributes,um_fraction,null_mean,null_std,p_value,verdict\n");
    for r in results {
        out.push_str(&format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.4},{}\n",
            r.name, r.n_objects, r.attributes, r.um_fraction, r.null_mean,
            r.null_std, r.p_value, r.verdict
        ));
    }
    std::fs::write(path, &out).unwrap_or_else(|e| {
        eprintln!("WARNING: Could not write CSV to {}: {}", path.display(), e);
    });
    eprintln!("Results written to {}", path.display());
}
