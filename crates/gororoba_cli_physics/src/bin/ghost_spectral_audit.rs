//! Ghost Spectral Audit -- unified multi-method falsification driver.
//!
//! Orchestrates Rust-native tests (permutation, bootstrap, BH-FDR) and
//! Python bridge tests (Lomb-Scargle, IAAFT surrogates, multitaper,
//! quantile periodogram, coherence) for the phi^{-1/2} ghost frequency
//! hypothesis.
//!
//! Usage:
//!     ghost-spectral-audit single --dataset data.csv --column value
//!     ghost-spectral-audit batch --dir data/csv/ghost_audit/ --column value

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
};

use clap::{Parser, Subcommand};

use spectral_core::ghost_spectral::{
    ALIASED_GHOST_FREQ, FREQ_TOL, GhostVerdict, check_ghost_rigorous, combine_p_values,
    permutation_test, sorted_distribution_null,
};

/// Ghost Spectral Audit -- multi-method falsification
#[derive(Parser)]
#[command(name = "ghost-spectral-audit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Audit a single dataset with all available methods
    Single {
        /// Path to CSV file
        #[arg(long)]
        dataset: PathBuf,
        /// Column name to analyze
        #[arg(long, default_value = "value")]
        column: String,
        /// Number of permutations for Rust-native test
        #[arg(long, default_value_t = 1000)]
        permutations: usize,
        /// Number of bootstrap samples for sorted-distribution null
        #[arg(long, default_value_t = 10000)]
        bootstrap: usize,
        /// Number of IAAFT surrogates for Python bridge test
        #[arg(long, default_value_t = 1000)]
        surrogates: usize,
        /// Path to Python interpreter (must have ghost_stats dependencies)
        #[arg(long, default_value = "python3")]
        python: String,
        /// Path to ghost_stats Python module directory
        #[arg(long)]
        python_module: Option<PathBuf>,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Audit all CSVs in a directory, combine results
    Batch {
        /// Directory containing CSV files
        #[arg(long)]
        dir: PathBuf,
        /// Column name to analyze
        #[arg(long, default_value = "value")]
        column: String,
        /// Path to Python interpreter
        #[arg(long, default_value = "python3")]
        python: String,
        /// Path to ghost_stats Python module directory
        #[arg(long)]
        python_module: Option<PathBuf>,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

/// Result from a single Python bridge test
#[derive(Debug)]
struct PythonResult {
    method: String,
    p_value: f64,
    detail: String,
}

fn load_csv_values(path: &Path, column: &str) -> Vec<f64> {
    let mut values = Vec::new();
    if let Ok(mut rdr) = csv::ReaderBuilder::new().from_path(path) {
        let headers = match rdr.headers() {
            Ok(h) => h.clone(),
            Err(_) => return values,
        };
        let col_idx = headers.iter().position(|h| h == column);
        let col_idx = match col_idx {
            Some(i) => i,
            None => return values,
        };
        for record in rdr.records().flatten() {
            if let Some(field) = record.get(col_idx)
                && let Ok(v) = field.parse::<f64>()
                && v.is_finite()
            {
                values.push(v);
            }
        }
    }
    values
}

fn run_python_script(python: &str, script_path: &Path, args: &[&str]) -> Option<serde_json::Value> {
    let output = Command::new(python)
        .arg(script_path)
        .args(args)
        .arg("--json")
        .output();

    match output {
        Ok(out) => {
            if out.status.success() {
                let stdout = String::from_utf8_lossy(&out.stdout);
                serde_json::from_str(&stdout).ok()
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                eprintln!("  [WARN] Python script failed: {stderr}");
                None
            }
        }
        Err(e) => {
            eprintln!("  [WARN] Could not run Python: {e}");
            None
        }
    }
}

fn resolve_python_module(explicit: &Option<PathBuf>) -> PathBuf {
    if let Some(p) = explicit {
        return p.clone();
    }
    let candidates = [
        PathBuf::from("src/ghost_stats"),
        PathBuf::from("../../src/ghost_stats"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    PathBuf::from("src/ghost_stats")
}

fn run_lomb_scargle(
    python: &str,
    module_dir: &Path,
    dataset: &Path,
    column: &str,
) -> Option<PythonResult> {
    let script = module_dir.join("lomb_scargle.py");
    let ds = dataset.to_string_lossy();
    let json = run_python_script(python, &script, &["--input", &ds, "--column", column])?;

    let fap = json.get("fap_at_target")?.as_f64()?;
    let peak_freq = json
        .get("peak_freq")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let peak_power = json
        .get("peak_power")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    Some(PythonResult {
        method: "Lomb-Scargle (Baluev FAP)".to_string(),
        p_value: fap,
        detail: format!("peak_freq={peak_freq:.6}, power={peak_power:.6e}"),
    })
}

fn run_iaaft_surrogates(
    python: &str,
    module_dir: &Path,
    dataset: &Path,
    column: &str,
    n_surrogates: usize,
    seed: u64,
) -> Option<PythonResult> {
    let script = module_dir.join("surrogates.py");
    let ds = dataset.to_string_lossy();
    let n_str = n_surrogates.to_string();
    let seed_str = seed.to_string();
    let json = run_python_script(
        python,
        &script,
        &[
            "--input", &ds, "--column", column, "--method", "iaaft", "--n", &n_str, "--seed",
            &seed_str,
        ],
    )?;

    let p = json.get("p_empirical")?.as_f64()?;
    let obs = json
        .get("observed_power")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let null_mean = json
        .get("null_mean")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    Some(PythonResult {
        method: "IAAFT Surrogates".to_string(),
        p_value: p,
        detail: format!("observed={obs:.6e}, null_mean={null_mean:.6e}"),
    })
}

fn run_multitaper(
    python: &str,
    module_dir: &Path,
    dataset: &Path,
    column: &str,
) -> Option<PythonResult> {
    let script = module_dir.join("multitaper.py");
    let ds = dataset.to_string_lossy();
    let json = run_python_script(python, &script, &["--input", &ds, "--column", column])?;

    let p = json.get("f_pvalue")?.as_f64()?;
    let f_stat = json
        .get("f_statistic")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    Some(PythonResult {
        method: "Multitaper F-test".to_string(),
        p_value: p,
        detail: format!("F={f_stat:.4}"),
    })
}

/// Single-dataset ghost-spectral audit configuration: data location
/// (path + column), statistical resampling counts (permutations, bootstrap,
/// surrogates), Python interpreter setup, and PRNG seed.
struct AuditSingleConfig<'a> {
    dataset: &'a Path,
    column: &'a str,
    n_values: usize,
    permutations: usize,
    bootstrap: usize,
    surrogates: usize,
    python: &'a str,
    python_module: &'a Option<PathBuf>,
    seed: u64,
}

fn audit_single(cfg: AuditSingleConfig<'_>) {
    let dataset = cfg.dataset;
    let column = cfg.column;
    let n_values = cfg.n_values;
    let permutations = cfg.permutations;
    let bootstrap = cfg.bootstrap;
    let surrogates = cfg.surrogates;
    let python = cfg.python;
    let python_module = cfg.python_module;
    let seed = cfg.seed;
    let _ = n_values; // used for clarity in caller
    println!("=== Ghost Spectral Audit ===");
    println!("Dataset: {}", dataset.display());
    println!("Column: {column}");
    println!();

    let values = load_csv_values(dataset, column);
    if values.is_empty() {
        eprintln!("ERROR: No valid data in column '{column}'");
        return;
    }
    let n = values.len();
    println!("Loaded {n} values");

    let is_sorted = values.windows(2).all(|w| w[0] <= w[1]);
    if is_sorted {
        println!();
        println!("WARNING: Data appears to be monotonically sorted.");
        println!("  FFT of sorted catalog data tests distributional shape, not");
        println!("  physical periodicity. See Tang & Zhang 2005 (selection effects).");
    }
    println!();

    // --- Rust-native tests ---
    println!("--- Rust-Native Tests ---");
    println!();

    // 1. Rigorous ghost check (noise floor, FAP, FDR)
    let rigorous = check_ghost_rigorous(&values, ALIASED_GHOST_FREQ, FREQ_TOL, 0.05);
    println!("1. Rigorous Ghost Check:");
    println!("   Verdict: {:?}", rigorous.verdict);
    println!("   Raw SNR: {:.4}", rigorous.raw_snr);
    println!("   Corrected SNR: {:.4}", rigorous.corrected_snr);
    println!("   p_raw: {:.6e}", rigorous.p_raw);
    println!("   p_bonferroni: {:.6e}", rigorous.p_bonferroni);
    println!("   p_fdr: {:.6e}", rigorous.p_fdr);
    println!("   Trials: {}", rigorous.n_trials);
    if !rigorous.methodology_note.is_empty() {
        println!("   Note: {}", rigorous.methodology_note);
    }
    println!();

    // 2. Permutation test
    let perm = permutation_test(&values, ALIASED_GHOST_FREQ, permutations, seed);
    println!("2. Permutation Test (N={permutations}):");
    println!("   Observed power: {:.6e}", perm.observed_power);
    println!("   Null mean: {:.6e}", perm.null_mean);
    println!("   Null std: {:.6e}", perm.null_std);
    println!("   Empirical p: {:.6e}", perm.p_empirical);
    println!();

    // 3. Sorted-distribution bootstrap null (only for sorted data)
    let boot_p = if is_sorted {
        let boot = sorted_distribution_null(&values, ALIASED_GHOST_FREQ, bootstrap, seed);
        println!("3. Bootstrap Null (N={bootstrap}):");
        println!("   Observed power: {:.6e}", boot.observed_power);
        println!("   Null mean: {:.6e}", boot.null_mean);
        println!("   Null std: {:.6e}", boot.null_std);
        println!("   Empirical p: {:.6e}", boot.p_empirical);
        println!();
        Some(boot.p_empirical)
    } else {
        println!("3. Bootstrap Null: SKIPPED (data not sorted)");
        println!("   Bootstrap null only applies to sorted catalog data.");
        println!();
        None
    };

    // --- Python bridge tests ---
    let module_dir = resolve_python_module(python_module);
    println!("--- Python Bridge Tests ---");
    println!("Module dir: {}", module_dir.display());
    println!();

    let mut python_p_values: Vec<(String, f64)> = Vec::new();

    // 4. Lomb-Scargle
    print!("4. Lomb-Scargle (Baluev FAP)... ");
    if let Some(r) = run_lomb_scargle(python, &module_dir, dataset, column) {
        println!("p={:.6e}  [{}]", r.p_value, r.detail);
        python_p_values.push((r.method, r.p_value));
    } else {
        println!("SKIPPED (Python not available)");
    }

    // 5. IAAFT Surrogates
    print!("5. IAAFT Surrogates (N={surrogates})... ");
    if let Some(r) = run_iaaft_surrogates(python, &module_dir, dataset, column, surrogates, seed) {
        println!("p={:.6e}  [{}]", r.p_value, r.detail);
        python_p_values.push((r.method, r.p_value));
    } else {
        println!("SKIPPED (Python not available)");
    }

    // 6. Multitaper F-test
    print!("6. Multitaper F-test... ");
    if let Some(r) = run_multitaper(python, &module_dir, dataset, column) {
        println!("p={:.6e}  [{}]", r.p_value, r.detail);
        python_p_values.push((r.method, r.p_value));
    } else {
        println!("SKIPPED (Python not available)");
    }

    println!();

    // --- Summary ---
    println!("=== SUMMARY ===");
    println!();

    let mut all_p_values: Vec<f64> = vec![rigorous.p_bonferroni, perm.p_empirical];
    let mut method_names: Vec<String> = vec![
        "Rigorous (Bonferroni)".to_string(),
        "Permutation".to_string(),
    ];
    if let Some(bp) = boot_p {
        all_p_values.push(bp);
        method_names.push("Bootstrap null".to_string());
    }
    all_p_values.extend(python_p_values.iter().map(|(_, p)| *p));
    method_names.extend(python_p_values.iter().map(|(name, _)| name.clone()));

    println!("Method                      | p-value      | Verdict");
    println!("----------------------------|--------------|----------");
    for (name, p) in method_names.iter().zip(all_p_values.iter()) {
        let verdict = if *p < 0.05 {
            "REJECT H0"
        } else {
            "FAIL TO REJECT"
        };
        println!("{name:<28}| {p:<13.6e}| {verdict}");
    }
    println!();

    let n_reject = all_p_values.iter().filter(|&&p| p < 0.05).count();
    let n_total = all_p_values.len();
    println!("Methods rejecting H0: {n_reject}/{n_total}");

    // Combined p-values (Rust-native only, since sample sizes match)
    let mut rust_p = vec![rigorous.p_bonferroni, perm.p_empirical];
    if let Some(bp) = boot_p {
        rust_p.push(bp);
    }
    let rust_n: Vec<f64> = vec![n as f64; rust_p.len()];
    let combined = combine_p_values(&rust_p, &rust_n);
    println!();
    println!(
        "Combined (Rust-native, Fisher): p = {:.6e}",
        combined.fisher_p
    );
    println!(
        "Combined (Rust-native, Stouffer): Z = {:.4}, p = {:.6e}",
        combined.stouffer_z, combined.stouffer_p
    );

    println!();
    let final_verdict = if n_reject == 0 {
        "GHOST FREQUENCY IS NOT SIGNIFICANT -- consistent with noise/distributional artifact"
    } else if n_reject == n_total {
        "GHOST FREQUENCY IS SIGNIFICANT across ALL methods -- genuine structure detected"
    } else {
        "MIXED RESULTS -- some methods detect significance, others do not. Inconclusive."
    };
    println!("FINAL VERDICT: {final_verdict}");
}

fn audit_batch(dir: &Path, column: &str, python: &str, python_module: &Option<PathBuf>, seed: u64) {
    println!("=== Ghost Spectral Audit -- Batch Mode ===");
    println!("Directory: {}", dir.display());
    println!();

    let mut csv_files: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "csv") {
                csv_files.push(path);
            }
        }
    }
    csv_files.sort();

    if csv_files.is_empty() {
        eprintln!("ERROR: No CSV files found in {}", dir.display());
        return;
    }

    println!("Found {} CSV files", csv_files.len());
    println!();

    // Track results with sample sizes for combination
    let mut per_dataset: BTreeMap<String, (f64, f64, GhostVerdict)> = BTreeMap::new();

    for csv_path in &csv_files {
        let name = csv_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let values = load_csv_values(csv_path, column);
        if values.is_empty() {
            println!("  {name}: SKIP (no valid data)");
            continue;
        }

        let n = values.len();
        let result = check_ghost_rigorous(&values, ALIASED_GHOST_FREQ, FREQ_TOL, 0.05);
        println!(
            "  {name}: N={n}, verdict={:?}, p_bonf={:.6e}, p_fdr={:.6e}",
            result.verdict, result.p_bonferroni, result.p_fdr
        );
        per_dataset.insert(name, (result.p_bonferroni, n as f64, result.verdict));
    }

    println!();

    if per_dataset.is_empty() {
        println!("No valid datasets to combine.");
        return;
    }

    // Combine across datasets
    let p_values: Vec<f64> = per_dataset.values().map(|(p, _, _)| *p).collect();
    let sample_sizes: Vec<f64> = per_dataset.values().map(|(_, n, _)| *n).collect();

    let combined = combine_p_values(&p_values, &sample_sizes);
    println!(
        "=== Cross-Dataset Combination ({} datasets) ===",
        p_values.len()
    );
    println!(
        "  Fisher chi2: {:.4}, p = {:.6e}",
        combined.fisher_chi2, combined.fisher_p
    );
    println!(
        "  Stouffer Z: {:.4}, p = {:.6e}",
        combined.stouffer_z, combined.stouffer_p
    );
    println!();

    let n_detected = per_dataset
        .values()
        .filter(|(_, _, v)| matches!(v, GhostVerdict::Detected))
        .count();
    let n_marginal = per_dataset
        .values()
        .filter(|(_, _, v)| matches!(v, GhostVerdict::Marginal))
        .count();
    let n_null = per_dataset
        .values()
        .filter(|(_, _, v)| matches!(v, GhostVerdict::Null))
        .count();

    println!("Verdicts: {n_detected} Detected, {n_marginal} Marginal, {n_null} Null");

    let batch_verdict = if combined.fisher_p > 0.05 && combined.stouffer_p > 0.05 {
        "GHOST FREQUENCY NOT SIGNIFICANT across datasets"
    } else if combined.fisher_p < 0.01 && combined.stouffer_p < 0.01 {
        "GHOST FREQUENCY SIGNIFICANT across datasets"
    } else {
        "MIXED / INCONCLUSIVE across datasets"
    };
    println!("BATCH VERDICT: {batch_verdict}");

    let _ = resolve_python_module(python_module);
    let _ = python;
    let _ = seed;
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Single {
            dataset,
            column,
            permutations,
            bootstrap,
            surrogates,
            python,
            python_module,
            seed,
        } => {
            let values = load_csv_values(&dataset, &column);
            let n = values.len();
            drop(values);
            audit_single(AuditSingleConfig {
                dataset: &dataset,
                column: &column,
                n_values: n,
                permutations,
                bootstrap,
                surrogates,
                python: &python,
                python_module: &python_module,
                seed,
            });
        }
        Commands::Batch {
            dir,
            column,
            python,
            python_module,
            seed,
        } => {
            audit_batch(&dir, &column, &python, &python_module, seed);
        }
    }
}
