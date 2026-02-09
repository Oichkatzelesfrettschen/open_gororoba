//! Direction 2: Compact Object Multi-Attribute Ultrametricity
//!
//! Tests whether the multi-attribute parameter space of compact objects
//! (pulsars, magnetars, FRBs) exhibits ultrametric structure under
//! normalized Euclidean distances.
//!
//! # Method
//!
//! 1. Load CHIME FRBs (+ optionally ATNF pulsars, McGill magnetars)
//! 2. Extract shared attributes: DM, galactic position
//! 3. Normalize attributes (log-scale for DM, linear for position)
//! 4. Compute Euclidean distance matrix in normalized space
//! 5. Run ultrametric fraction test on Euclidean distances
//! 6. Compare to null (column-shuffled data)
//!
//! # Note on Baire distances
//!
//! The Baire metric d(x,y) = base^{-k} is ultrametric by construction,
//! so testing ultrametric fraction on Baire distances always yields 1.0.
//! The meaningful test uses Euclidean distances in the multi-attribute space.
//!
//! # Usage
//!
//! baire-compact --frbs data/external/chime_frb_cat2.csv \
//!               --pulsars data/external/atnf_pulsars.csv \
//!               --magnetars data/external/mcgill_magnetars.csv \
//!               --output data/csv/c071c_baire_compact_ultrametric.csv

use clap::Parser;
use std::path::PathBuf;

use data_core::catalogs::chime::parse_chime_csv;
use stats_core::ultrametric::baire::{euclidean_ultrametric_test, AttributeSpec, BaireEncoder};

#[derive(Parser)]
#[command(name = "baire-compact")]
#[command(
    about = "Direction 2: Test ultrametric structure in compact object multi-attribute space"
)]
struct Cli {
    /// Path to CHIME FRB CSV.
    #[arg(long, default_value = "data/external/chime_frb_cat2.csv")]
    frbs: PathBuf,

    /// Path to ATNF pulsar CSV (optional).
    #[arg(long)]
    pulsars: Option<PathBuf>,

    /// Path to McGill magnetar CSV (optional).
    #[arg(long)]
    magnetars: Option<PathBuf>,

    /// Base for Baire digit encoding (used for normalization spec).
    #[arg(long, default_value = "10")]
    base: u64,

    /// Number of digits per attribute.
    #[arg(long, default_value = "4")]
    n_digits: usize,

    /// Number of triples for ultrametric fraction test.
    #[arg(long, default_value = "100000")]
    n_triples: usize,

    /// Number of permutations for null distribution.
    #[arg(long, default_value = "200")]
    n_permutations: usize,

    /// Output CSV path.
    #[arg(long, default_value = "data/csv/c071c_baire_compact_ultrametric.csv")]
    output: PathBuf,
}

/// Extract [DM, gl, gb] from a dataset, returning (data_rows, population_label).
fn extract_frb_data(cli: &Cli) -> (Vec<Vec<f64>>, String) {
    eprintln!("Loading FRBs from {}...", cli.frbs.display());
    let frb_events = parse_chime_csv(&cli.frbs).unwrap_or_else(|e| {
        eprintln!("Failed to parse CHIME CSV: {}", e);
        std::process::exit(1);
    });

    eprintln!("Loaded {} FRB events", frb_events.len());

    let mut data: Vec<Vec<f64>> = Vec::new();

    for event in &frb_events {
        let dm = event.bonsai_dm;
        let gl = event.gl;
        let gb = event.gb;

        if dm.is_nan() || dm <= 0.0 {
            continue;
        }
        if gl.is_nan() || gb.is_nan() {
            continue;
        }

        data.push(vec![dm, gl, gb]);
    }

    eprintln!("Valid FRBs with DM + position: {}", data.len());
    (data, "FRB".to_string())
}

fn load_pulsars(path: &std::path::Path) -> Vec<Vec<f64>> {
    eprintln!("Loading pulsars from {}...", path.display());
    let pulsars = data_core::catalogs::atnf::parse_atnf_csv(path).unwrap_or_else(|e| {
        eprintln!("Failed to parse ATNF CSV: {}", e);
        Vec::new()
    });

    let mut data: Vec<Vec<f64>> = Vec::new();
    for p in &pulsars {
        if p.dm.is_nan() || p.dm <= 0.0 {
            continue;
        }
        if p.gl.is_nan() || p.gb.is_nan() {
            continue;
        }
        data.push(vec![p.dm, p.gl, p.gb]);
    }

    eprintln!("Valid pulsars with DM + position: {}", data.len());
    data
}

fn load_magnetars(path: &std::path::Path) -> Vec<Vec<f64>> {
    eprintln!("Loading magnetars from {}...", path.display());
    let magnetars = data_core::catalogs::mcgill::parse_mcgill_csv(path).unwrap_or_else(|e| {
        eprintln!("Failed to parse McGill CSV: {}", e);
        Vec::new()
    });

    let mut data: Vec<Vec<f64>> = Vec::new();
    for m in &magnetars {
        if m.dm.is_nan() || m.dm <= 0.0 {
            continue;
        }
        if m.gl.is_nan() || m.gb.is_nan() {
            continue;
        }
        data.push(vec![m.dm, m.gl, m.gb]);
    }

    eprintln!("Valid magnetars with DM + position: {}", data.len());
    data
}

fn run_test(
    label: &str,
    data: &[Vec<f64>],
    cli: &Cli,
) -> Option<stats_core::ultrametric::baire::BaireTestResult> {
    if data.len() < 10 {
        eprintln!(
            "  {} -- too few objects ({}) for analysis",
            label,
            data.len()
        );
        return None;
    }

    // Compute attribute ranges from this dataset
    let dm_range = data
        .iter()
        .map(|r| r[0])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });

    let attributes = vec![
        AttributeSpec {
            name: "DM".into(),
            min: dm_range.0,
            max: dm_range.1,
            log_scale: true,
        },
        AttributeSpec {
            name: "gl".into(),
            min: 0.0,
            max: 360.0,
            log_scale: false,
        },
        AttributeSpec {
            name: "gb".into(),
            min: -90.0,
            max: 90.0,
            log_scale: false,
        },
    ];

    let encoder = BaireEncoder::new(attributes, cli.base, cli.n_digits);

    eprintln!(
        "  {} -- testing {} objects, 3 attributes (DM, gl, gb)...",
        label,
        data.len()
    );
    let result = euclidean_ultrametric_test(&encoder, data, cli.n_triples, cli.n_permutations, 42);

    eprintln!(
        "  {} -- frac={:.4}, null={:.4}+/-{:.4}, p={:.4}",
        label,
        result.ultrametric_fraction,
        result.null_fraction_mean,
        result.null_fraction_std,
        result.p_value,
    );

    Some(result)
}

fn main() {
    let cli = Cli::parse();

    eprintln!("=== Direction 2: Multi-Attribute Euclidean Ultrametricity ===");
    eprintln!("(Testing Euclidean distances in normalized [DM, gl, gb] space)");

    // 1. Load datasets
    let (frb_data, _) = extract_frb_data(&cli);

    let pulsar_data = cli.pulsars.as_ref().map(|p| load_pulsars(p));
    let magnetar_data = cli.magnetars.as_ref().map(|p| load_magnetars(p));

    if frb_data.len() < 10 {
        eprintln!("Too few valid FRBs for analysis");
        std::process::exit(1);
    }

    // 2. Run tests on each population separately
    eprintln!("\n--- Per-population tests ---");

    let frb_result = run_test("FRB", &frb_data, &cli);

    let pulsar_result = pulsar_data
        .as_ref()
        .and_then(|d| run_test("Pulsar", d, &cli));

    let magnetar_result = magnetar_data
        .as_ref()
        .and_then(|d| run_test("Magnetar", d, &cli));

    // 3. Combined cross-population test
    eprintln!("\n--- Cross-population test ---");
    let mut combined_data = frb_data.clone();
    if let Some(ref pd) = pulsar_data {
        combined_data.extend(pd.iter().cloned());
    }
    if let Some(ref md) = magnetar_data {
        combined_data.extend(md.iter().cloned());
    }

    let combined_result = if combined_data.len() > frb_data.len() {
        run_test("Combined", &combined_data, &cli)
    } else {
        eprintln!("  No additional populations loaded; combined = FRB-only");
        None
    };

    // 4. DM-only test for comparison
    eprintln!("\n--- DM-only baseline ---");
    let dm_only_data: Vec<Vec<f64>> = frb_data.iter().map(|r| vec![r[0]]).collect();
    let dm_range = dm_only_data
        .iter()
        .map(|r| r[0])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    let dm_attrs = vec![AttributeSpec {
        name: "DM".into(),
        min: dm_range.0,
        max: dm_range.1,
        log_scale: true,
    }];
    let dm_encoder = BaireEncoder::new(dm_attrs, cli.base, cli.n_digits);
    let dm_result = euclidean_ultrametric_test(
        &dm_encoder,
        &dm_only_data,
        cli.n_triples,
        cli.n_permutations,
        42,
    );
    eprintln!(
        "  DM-only: frac={:.4}, null={:.4}, p={:.4}",
        dm_result.ultrametric_fraction, dm_result.null_fraction_mean, dm_result.p_value,
    );

    // 5. Write results CSV
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut wtr = csv::Writer::from_path(&cli.output).unwrap();
    wtr.write_record([
        "encoding",
        "n_objects",
        "n_attributes",
        "base",
        "n_digits",
        "ultrametric_fraction",
        "null_fraction_mean",
        "null_fraction_std",
        "p_value",
        "verdict",
    ])
    .unwrap();

    let write_row = |wtr: &mut csv::Writer<std::fs::File>,
                     label: &str,
                     result: &stats_core::ultrametric::baire::BaireTestResult| {
        let verdict = if result.p_value < 0.05 {
            "Pass"
        } else {
            "Fail"
        };
        wtr.write_record([
            label,
            &result.n_objects.to_string(),
            &result.n_attributes.to_string(),
            &result.base.to_string(),
            &result.n_digits.to_string(),
            &format!("{:.6}", result.ultrametric_fraction),
            &format!("{:.6}", result.null_fraction_mean),
            &format!("{:.6}", result.null_fraction_std),
            &format!("{:.6}", result.p_value),
            verdict,
        ])
        .unwrap();
    };

    if let Some(ref r) = frb_result {
        write_row(&mut wtr, "FRB_3attr", r);
    }
    if let Some(ref r) = pulsar_result {
        write_row(&mut wtr, "Pulsar_3attr", r);
    }
    if let Some(ref r) = magnetar_result {
        write_row(&mut wtr, "Magnetar_3attr", r);
    }
    if let Some(ref r) = combined_result {
        write_row(&mut wtr, "Combined_3attr", r);
    }
    write_row(&mut wtr, "FRB_DM_only", &dm_result);

    wtr.flush().unwrap();

    eprintln!("\nResults written to {}", cli.output.display());

    // Summary
    eprintln!("\n=== Gate Verdict ===");
    let frb_pass = frb_result.as_ref().is_some_and(|r| r.p_value < 0.05);
    let combined_pass = combined_result.as_ref().is_some_and(|r| r.p_value < 0.05);
    let dm_pass = dm_result.p_value < 0.05;

    if frb_pass || combined_pass {
        eprintln!("C-437: PASS -- multi-attribute space shows ultrametric structure");
    } else if dm_pass {
        eprintln!("C-437: MARGINAL -- DM-only shows signal but multi-attribute does not");
    } else {
        eprintln!("C-437: FAIL -- no ultrametric signal in multi-attribute space");
    }
}
