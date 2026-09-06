//! magnonic-band-structure: 9-band magnonic crystal band topology calculator.
//!
//! Computes band structures, Chern numbers, valley Chern numbers, flat band
//! detection, parameter sweeps, domain wall edge modes, and point defect
//! localized modes for the Kaman et al. (2026) 9-band tight-binding model
//! of magnonic crystals in hexagonal YIG antidot arrays.
//!
//! Usage:
//!   magnonic-band-structure bands --n-k 100 --output bands.csv
//!   magnonic-band-structure chern --n-grid 31
//!   magnonic-band-structure flat-band --n-k 50
//!   magnonic-band-structure sweep --param delta-eps --min 0 --max 1 --steps 20
//!   magnonic-band-structure domain-wall --n-cells 20 --n-k 100
//!   magnonic-band-structure defect --radius 5

use clap::{Parser, Subcommand};
use quantum_core::{
    magnonic_crystal::{
        InversionBreakingParams, MagnonicTBParams, build_domain_wall_supercell,
        build_magnonic_9band, compute_magnonic_bands, point_defect_modes,
    },
    tight_binding::{TopologyAdmission, Valley, checked_subspace_topology, detect_flat_bands},
};

#[derive(Parser)]
#[command(name = "magnonic-band-structure")]
#[command(about = "9-band magnonic crystal band topology calculator")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compute 9-band dispersion along Gamma-M-K-Gamma
    Bands {
        /// Points per high-symmetry segment
        #[arg(long, default_value = "100")]
        n_k: usize,
        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Compute Chern and valley Chern numbers for all 9 bands
    Chern {
        /// Brillouin zone grid size
        #[arg(long, default_value = "31")]
        n_grid: usize,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Detect flat bands and compute DOS enhancement
    FlatBand {
        /// BZ sampling density
        #[arg(long, default_value = "50")]
        n_k: usize,
        /// Bandwidth threshold (GHz)
        #[arg(long, default_value = "0.05")]
        threshold: f64,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Parameter sweep with band edges and valley Chern numbers
    Sweep {
        /// Parameter to sweep: delta-eps, b-ext, or hole-ratio
        #[arg(long, default_value = "delta-eps")]
        param: String,
        /// Minimum value
        #[arg(long, default_value = "0.0")]
        min: f64,
        /// Maximum value
        #[arg(long, default_value = "0.5")]
        max: f64,
        /// Number of sweep steps
        #[arg(long, default_value = "20")]
        steps: usize,
        /// Chern grid size
        #[arg(long, default_value = "15")]
        n_grid: usize,
        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Domain wall supercell ribbon band structure
    DomainWall {
        /// Number of unit cells across ribbon
        #[arg(long, default_value = "20")]
        n_cells: usize,
        /// k-points along edge direction
        #[arg(long, default_value = "100")]
        n_k: usize,
        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Spectral-gap candidates in an open point-defect patch; localization unassessed
    Defect {
        /// Supercell radius (total side = 2*radius+1)
        #[arg(long, default_value = "3")]
        radius: usize,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Bands { n_k, output, json } => run_bands(n_k, output, json),
        Commands::Chern { n_grid, json } => run_chern(n_grid, json),
        Commands::FlatBand {
            n_k,
            threshold,
            json,
        } => run_flat_band(n_k, threshold, json),
        Commands::Sweep {
            param,
            min,
            max,
            steps,
            n_grid,
            output,
        } => run_sweep(&param, min, max, steps, n_grid, output),
        Commands::DomainWall {
            n_cells,
            n_k,
            output,
        } => run_domain_wall(n_cells, n_k, output),
        Commands::Defect { radius, json } => run_defect(radius, json),
    }
}

fn run_bands(n_k: usize, output: Option<String>, json: bool) {
    let result = compute_magnonic_bands(
        &MagnonicTBParams::kaman_default(),
        &InversionBreakingParams::none(),
        n_k,
        11, // minimal chern grid for bands mode
        400.0,
    );

    if json {
        println!("{{");
        println!("  \"n_bands\": 9,");
        println!("  \"n_k\": {},", result.k_distances.len());
        println!("  \"symmetry_labels\": [");
        for (i, (label, dist)) in result.symmetry_labels.iter().enumerate() {
            let comma = if i < result.symmetry_labels.len() - 1 {
                ","
            } else {
                ""
            };
            println!(
                "    {{\"label\": \"{}\", \"k_dist\": {:.6}}}{}",
                label, dist, comma
            );
        }
        println!("  ]");
        println!("}}");
    } else {
        println!("Magnonic Crystal 9-Band Dispersion");
        println!("===================================");
        println!();
        println!("Model: Kaman et al. (2026), symmetric (no inversion breaking)");
        println!("Path: Gamma -> M -> K -> Gamma ({} points/segment)", n_k);
        println!();
        println!("Symmetry points:");
        for (label, dist) in &result.symmetry_labels {
            println!("  {}: k_dist = {:.4}", label, dist);
        }
        println!();
        println!("Band energy ranges (GHz):");
        for (band, energies) in result.band_energies.iter().enumerate() {
            let min_e = energies.iter().cloned().fold(f64::MAX, f64::min);
            let max_e = energies.iter().cloned().fold(f64::MIN, f64::max);
            println!(
                "  Band {}: [{:.4}, {:.4}]  (BW = {:.4})",
                band,
                min_e,
                max_e,
                max_e - min_e
            );
        }
    }

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        let mut header = vec!["k_dist".to_string()];
        for b in 0..9 {
            header.push(format!("band_{}", b));
        }
        wtr.write_record(&header).unwrap();

        for (k_idx, &k_d) in result.k_distances.iter().enumerate() {
            let mut row = vec![format!("{:.8}", k_d)];
            for band in &result.band_energies {
                row.push(format!("{:.8}", band[k_idx]));
            }
            wtr.write_record(&row).unwrap();
        }
        wtr.flush().unwrap();
        eprintln!("Wrote band structure to {}", path);
    }
}

fn run_chern(n_grid: usize, json: bool) {
    let params = MagnonicTBParams::kaman_default();
    let inv_none = InversionBreakingParams::none();
    let inv_broken = InversionBreakingParams::kaman_default();

    let model_sym = build_magnonic_9band(&params, &inv_none, 400.0);
    let model_broken = build_magnonic_9band(&params, &inv_broken, 400.0);

    if json {
        println!("{{");
        println!("  \"n_grid\": {},", n_grid);
        println!("  \"symmetric\": {{");
        print_chern_json(&model_sym, n_grid);
        println!("  }},");
        println!("  \"broken\": {{");
        print_chern_json(&model_broken, n_grid);
        println!("  }}");
        println!("}}");
    } else {
        println!("Magnonic Crystal Chern Numbers");
        println!("===============================");
        println!("Grid: {} x {}", n_grid, n_grid);
        println!();

        println!("--- Symmetric (no inversion breaking) ---");
        print_chern_table(&model_sym, n_grid);
        println!();
        println!("--- Broken inversion (Kaman defaults) ---");
        print_chern_table(&model_broken, n_grid);
    }
}

fn print_chern_json(model: &quantum_core::tight_binding::TightBindingModel, n_grid: usize) {
    let bands: Vec<_> = (0..model.n_orbitals()).map(|band| {
        match checked_subspace_topology(model, band..band + 1, n_grid, TopologyAdmission::default()) {
            Ok(topology) => serde_json::json!({"band": band, "chern": topology.chern_number(), "vcn_k": topology.valley_chern_number(Valley::K), "vcn_kprime": topology.valley_chern_number(Valley::KPrime), "scope": "sampled_gap_and_links"}),
            Err(error) => serde_json::json!({"band": band, "chern": null, "vcn_k": null, "vcn_kprime": null, "admission_error": error.to_string()}),
        }
    }).collect();
    println!(
        "    \"bands\": {}",
        serde_json::to_string(&bands).expect("finite topology records")
    );
}

fn print_chern_table(model: &quantum_core::tight_binding::TightBindingModel, n_grid: usize) {
    println!("Band  Chern  VCN(K)  VCN(K')  sampled admission");
    for band in 0..model.n_orbitals() {
        match checked_subspace_topology(model, band..band + 1, n_grid, TopologyAdmission::default())
        {
            Ok(topology) => println!(
                "{band:5} {:+8.4} {:+8.4} {:+8.4} admitted",
                topology.chern_number(),
                topology.valley_chern_number(Valley::K),
                topology.valley_chern_number(Valley::KPrime)
            ),
            Err(error) => println!("{band:5} unavailable: {error}"),
        }
    }
}

fn run_flat_band(n_k: usize, threshold: f64, json: bool) {
    let model = build_magnonic_9band(
        &MagnonicTBParams::kaman_default(),
        &InversionBreakingParams::none(),
        400.0,
    );
    let info = detect_flat_bands(&model, n_k, threshold);

    if json {
        println!("{{");
        println!("  \"threshold_ghz\": {},", threshold);
        println!(
            "  \"flat_band_indices\": [{}],",
            info.flat_band_indices
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "  \"bandwidths_ghz\": [{}]",
            info.bandwidths
                .iter()
                .map(|b| format!("{:.6}", b))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("}}");
    } else {
        println!("Magnonic Crystal Flat Band Detection");
        println!("=====================================");
        println!("Threshold: {} GHz", threshold);
        println!("BZ sampling: {} x {}", n_k, n_k);
        println!();
        println!("{:>5}  {:>12}  {:>6}", "Band", "Bandwidth", "Flat?");
        println!("{:-<5}  {:-<12}  {:-<6}", "", "", "");
        for (b, &bw) in info.bandwidths.iter().enumerate() {
            let flat = if bw < threshold { "YES" } else { "no" };
            println!("{:5}  {:12.6}  {:>6}", b, bw, flat);
        }
        if !info.flat_band_indices.is_empty() {
            let avg_dispersive: f64 = info
                .bandwidths
                .iter()
                .enumerate()
                .filter(|(i, _)| !info.flat_band_indices.contains(i))
                .map(|(_, &bw)| bw)
                .sum::<f64>()
                / (info.bandwidths.len() - info.flat_band_indices.len()) as f64;
            let min_flat_bw: f64 = info
                .flat_band_indices
                .iter()
                .map(|&i| info.bandwidths[i])
                .fold(f64::MAX, f64::min);
            if min_flat_bw > 0.0 {
                println!(
                    "\nDOS enhancement ratio: ~{:.0}x",
                    avg_dispersive / min_flat_bw
                );
            } else {
                println!("\nPerfectly flat band (zero bandwidth)");
            }
        }
    }
}

fn run_sweep(param: &str, min: f64, max: f64, steps: usize, n_grid: usize, output: Option<String>) {
    println!("Magnonic Crystal Parameter Sweep");
    println!("=================================");
    println!("Parameter: {}", param);
    println!("Range: [{}, {}], {} steps", min, max, steps);
    println!("Chern grid: {}", n_grid);
    println!();

    let params = MagnonicTBParams::kaman_default();
    let mut rows: Vec<Vec<String>> = Vec::new();

    for step in 0..=steps {
        let val = min + (max - min) * step as f64 / steps as f64;
        let inv = match param {
            "delta-eps" => InversionBreakingParams {
                delta_eps_s: val,
                delta_eps_p: val * 0.5,
            },
            _ => {
                eprintln!("Unknown sweep parameter: {}", param);
                return;
            }
        };

        let model = build_magnonic_9band(&params, &inv, 400.0);
        let info = detect_flat_bands(&model, 20, 0.05);

        let mut row = vec![format!("{:.6}", val)];
        for b in 0..9 {
            row.push(format!("{:.6}", info.bandwidths[b]));
        }
        // Chern numbers for bands near Dirac point (bands 2-3 typically)
        for b in 0..9_usize.min(model.n_orbitals()) {
            match checked_subspace_topology(&model, b..b + 1, n_grid, TopologyAdmission::default())
            {
                Ok(topology) => row.push(format!("{:.4}", topology.chern_number())),
                Err(error) => row.push(format!("unavailable: {error}")),
            }
        }
        rows.push(row);

        if step % 5 == 0 {
            eprintln!("  step {}/{}", step, steps);
        }
    }

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        let mut header = vec!["param_value".to_string()];
        for b in 0..9 {
            header.push(format!("bw_{}", b));
        }
        for b in 0..9 {
            header.push(format!("chern_{}", b));
        }
        wtr.write_record(&header).unwrap();
        for row in &rows {
            wtr.write_record(row).unwrap();
        }
        wtr.flush().unwrap();
        eprintln!("Wrote sweep data to {}", path);
    } else {
        println!("Done. Use --output to save CSV.");
    }
}

fn run_domain_wall(n_cells: usize, n_k: usize, output: Option<String>) {
    println!("Magnonic Crystal Domain Wall");
    println!("============================");
    println!("Ribbon width: {} cells ({} orbitals)", n_cells, n_cells * 9);
    println!("k-points: {}", n_k);
    println!();

    let model = build_domain_wall_supercell(
        &MagnonicTBParams::kaman_default(),
        &InversionBreakingParams::kaman_default(),
        n_cells,
        400.0,
    );

    // 1D band structure along a1 direction (ribbon edge)
    let n_orb = model.n_orbitals();
    let b1 = model.lattice.b1;

    let mut all_evals: Vec<Vec<f64>> = vec![Vec::with_capacity(n_k); n_orb];
    let mut k_vals = Vec::with_capacity(n_k);

    for i in 0..n_k {
        let t = i as f64 / n_k as f64;
        let kx = b1.x * t;
        let ky = b1.y * t;
        k_vals.push(t);
        let evals = model.band_energies(kx, ky);
        for (band, &e) in evals.iter().enumerate() {
            all_evals[band].push(e);
        }
    }

    println!("Computed {} bands at {} k-points", n_orb, n_k);

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        let mut header = vec!["k_frac".to_string()];
        for b in 0..n_orb {
            header.push(format!("band_{}", b));
        }
        wtr.write_record(&header).unwrap();
        for (k_idx, &k) in k_vals.iter().enumerate() {
            let mut row = vec![format!("{:.8}", k)];
            for band in &all_evals {
                row.push(format!("{:.8}", band[k_idx]));
            }
            wtr.write_record(&row).unwrap();
        }
        wtr.flush().unwrap();
        eprintln!("Wrote domain wall bands to {}", path);
    } else {
        println!("Use --output to save CSV.");
    }
}

fn run_defect(radius: usize, json: bool) {
    let side = 2 * radius + 1;
    let n_orb = side * side * 9 - 1; // one kagome site removed

    if !json {
        println!("Magnonic Crystal Spectral-Gap Candidates");
        println!("====================================");
        println!(
            "Supercell: {}x{} = {} cells ({} orbitals)",
            side,
            side,
            side * side,
            n_orb
        );
        println!();
    }

    let modes = point_defect_modes(
        &MagnonicTBParams::kaman_default(),
        &InversionBreakingParams::kaman_default(),
        radius,
        400.0,
    );

    if json {
        println!("{{");
        println!("  \"radius\": {},", radius);
        println!("  \"supercell_side\": {},", side);
        println!("  \"n_orbitals\": {},", n_orb);
        println!("  \"localization_status\": \"unassessed\",");
        println!("  \"parameter_origin\": \"repository_reference\",");
        println!("  \"n_gap_candidates\": {},", modes.len());
        println!(
            "  \"candidate_model_frequencies_ghz\": [{}]",
            modes
                .iter()
                .map(|f| format!("{:.6}", f))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("}}");
    } else {
        println!("Candidates in sampled bulk gap: {}", modes.len());
        println!("Localization: unassessed; repository reference model frequencies");
        for (i, &f) in modes.iter().enumerate() {
            println!("  Candidate {}: {:.4} model GHz", i, f);
        }
        if modes.len() >= 2 {
            let splitting = (modes[1] - modes[0]).abs();
            println!("\nFirst candidate energy spacing: {:.4} model GHz", splitting);
        }
    }
}
