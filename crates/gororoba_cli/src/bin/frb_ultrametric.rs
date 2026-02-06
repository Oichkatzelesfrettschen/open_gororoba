//! frb-ultrametric: Test FRB dispersion measures for p-adic ultrametric structure.
//!
//! Implements the C-071 claim test pipeline: reads CHIME/FRB catalog CSV,
//! extracts DM columns, runs the full ultrametric analysis battery
//! (Euclidean fraction, graded defect, multi-prime p-adic clustering),
//! and outputs results as CSV + JSON gate verdict.
//!
//! Usage:
//!   frb-ultrametric --input data/external/chime_frb_cat1.csv \
//!                   --dm-column bonsai_dm \
//!                   --n-triples 100000 \
//!                   --output data/csv/c071_frb_ultrametric.csv

use clap::Parser;
use stats_core::ultrametric::{run_ultrametric_analysis, ultrametric_gate, UltrametricConfig};

#[derive(Parser)]
#[command(name = "frb-ultrametric")]
#[command(about = "Test FRB dispersion measures for p-adic ultrametric structure (C-071)")]
struct Args {
    /// Input CSV file (CHIME/FRB catalog)
    #[arg(short, long)]
    input: String,

    /// DM column name to analyze (default: bonsai_dm)
    #[arg(long, default_value = "bonsai_dm")]
    dm_column: String,

    /// Number of random triples to sample
    #[arg(long, default_value = "100000")]
    n_triples: usize,

    /// Number of permutations for null distribution
    #[arg(long, default_value = "1000")]
    n_permutations: usize,

    /// Comma-separated list of primes for p-adic tests
    #[arg(long, default_value = "2,3,5,7,11,13")]
    primes: String,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output CSV file for results
    #[arg(short, long)]
    output: Option<String>,

    /// Analyze all standard DM columns (bonsai_dm, dm_exc_ne2001, dm_exc_ymw16)
    #[arg(long)]
    all_columns: bool,
}

fn main() {
    let args = Args::parse();

    // Parse primes
    let primes: Vec<u64> = args
        .primes
        .split(',')
        .map(|s| s.trim().parse::<u64>().expect("Invalid prime"))
        .collect();

    let config = UltrametricConfig {
        n_triples: args.n_triples,
        n_permutations: args.n_permutations,
        primes: primes.clone(),
        seed: args.seed,
    };

    // Determine which columns to analyze
    let columns: Vec<String> = if args.all_columns {
        vec![
            "bonsai_dm".to_string(),
            "dm_exc_ne2001".to_string(),
            "dm_exc_ymw16".to_string(),
        ]
    } else {
        vec![args.dm_column.clone()]
    };

    // Read CSV and detect available columns
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(&args.input)
        .expect("Failed to open input CSV");

    let headers: Vec<String> = reader
        .headers()
        .expect("No headers")
        .iter()
        .map(|h| h.trim().to_string())
        .collect();

    eprintln!("Input: {}", args.input);
    eprintln!("Available columns: {:?}", headers);

    // Read all records
    let records: Vec<csv::StringRecord> = reader
        .records()
        .filter_map(|r| r.ok())
        .collect();
    eprintln!("Total records: {}", records.len());

    // Set up output CSV writer
    let mut csv_writer = args.output.as_ref().map(|path| {
        let wtr = csv::Writer::from_path(path).expect("Failed to create output CSV");
        wtr
    });

    // Write header if CSV output
    if let Some(ref mut wtr) = csv_writer {
        let mut header = vec![
            "catalog", "dm_column", "n_events", "n_triples",
            "ultrametric_fraction", "null_fraction_mean", "null_fraction_std",
            "fraction_p_value", "fraction_ci_lower", "fraction_ci_upper",
            "mean_defect", "null_mean_defect", "defect_p_value",
        ];
        // Add p-adic columns
        for p in &primes {
            // Borrow-safe: push string literals with prime numbers
            header.push("placeholder"); // Will be replaced below
            let _ = p; // Use p in the loop
        }
        // Actually write a proper header with dynamic prime columns
        let mut full_header: Vec<String> = vec![
            "catalog".into(), "dm_column".into(), "n_events".into(), "n_triples".into(),
            "ultrametric_fraction".into(), "null_fraction_mean".into(), "null_fraction_std".into(),
            "fraction_p_value".into(), "fraction_ci_lower".into(), "fraction_ci_upper".into(),
            "mean_defect".into(), "null_mean_defect".into(), "defect_p_value".into(),
        ];
        for p in &primes {
            full_header.push(format!("padic_p{}_p_value", p));
        }
        full_header.push("bonferroni_threshold".into());
        full_header.push("any_significant".into());
        full_header.push("verdict".into());

        wtr.write_record(&full_header).unwrap();
    }

    // Infer catalog name from filename
    let catalog_name = if args.input.contains("cat2") {
        "cat2"
    } else if args.input.contains("cat1") {
        "cat1"
    } else {
        "unknown"
    };

    // Analyze each column
    for col_name in &columns {
        let col_idx = match headers.iter().position(|h| h == col_name) {
            Some(idx) => idx,
            None => {
                eprintln!("WARNING: Column '{}' not found in CSV, skipping", col_name);
                continue;
            }
        };

        // Extract DM values, filtering NaN and empty
        let values: Vec<f64> = records
            .iter()
            .filter_map(|record| {
                record.get(col_idx)
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .filter(|v| v.is_finite() && *v > 0.0)
            })
            .collect();

        eprintln!("\n=== Column: {} ===", col_name);
        eprintln!("Valid events: {} / {}", values.len(), records.len());

        if values.len() < 10 {
            eprintln!("ERROR: Too few valid values ({}) for analysis, skipping", values.len());
            continue;
        }

        // Run analysis
        let analysis = run_ultrametric_analysis(&values, &config);

        // Print results
        eprintln!("Ultrametric fraction: {:.6} (null: {:.6} +/- {:.6})",
            analysis.fraction_result.ultrametric_fraction,
            analysis.fraction_result.null_fraction_mean,
            analysis.fraction_result.null_fraction_std,
        );
        eprintln!("Fraction p-value: {:.6}", analysis.fraction_result.p_value);
        eprintln!("Bootstrap 95% CI: [{:.6}, {:.6}]",
            analysis.fraction_result.bootstrap_ci.0,
            analysis.fraction_result.bootstrap_ci.1,
        );
        eprintln!("Mean defect: {:.6} (null: {:.6})",
            analysis.defect_result.mean_defect,
            analysis.defect_result.null_mean_defect,
        );
        eprintln!("Defect p-value: {:.6}", analysis.defect_result.defect_p_value);

        for pr in &analysis.padic_results {
            eprintln!("P-adic p={}: fraction={:.6} (null={:.6}), p-value={:.6}",
                pr.prime, pr.padic_ultrametric_fraction,
                pr.null_ultrametric_fraction, pr.p_value,
            );
        }

        eprintln!("Bonferroni threshold: {:.6}", analysis.bonferroni_threshold);
        eprintln!("Any significant: {}", analysis.any_significant);
        eprintln!("Verdict: {}", analysis.verdict);

        // Gate result
        let gate = ultrametric_gate(
            "C-071",
            &analysis,
            &format!("FRB DM ultrametric ({}:{})", catalog_name, col_name),
        );
        println!("{}", gate.to_json());

        // Write CSV row
        if let Some(ref mut wtr) = csv_writer {
            let mut row: Vec<String> = vec![
                catalog_name.into(),
                col_name.clone(),
                values.len().to_string(),
                config.n_triples.to_string(),
                format!("{:.6}", analysis.fraction_result.ultrametric_fraction),
                format!("{:.6}", analysis.fraction_result.null_fraction_mean),
                format!("{:.6}", analysis.fraction_result.null_fraction_std),
                format!("{:.6}", analysis.fraction_result.p_value),
                format!("{:.6}", analysis.fraction_result.bootstrap_ci.0),
                format!("{:.6}", analysis.fraction_result.bootstrap_ci.1),
                format!("{:.6}", analysis.defect_result.mean_defect),
                format!("{:.6}", analysis.defect_result.null_mean_defect),
                format!("{:.6}", analysis.defect_result.defect_p_value),
            ];
            for pr in &analysis.padic_results {
                row.push(format!("{:.6}", pr.p_value));
            }
            row.push(format!("{:.6}", analysis.bonferroni_threshold));
            row.push(analysis.any_significant.to_string());
            row.push(analysis.verdict.to_string());

            wtr.write_record(&row).unwrap();
        }
    }

    // Flush CSV
    if let Some(ref mut wtr) = csv_writer {
        wtr.flush().unwrap();
        eprintln!("\nWrote results to {}", args.output.as_ref().unwrap());
    }
}
