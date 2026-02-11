use clap::{Parser, ValueEnum};
use csv::Writer;
use snia_core::{
    best_parameter_set, scan_parameter_sets, write_snapshot_toml, BurnState, CalibrationResult,
    CarbonBurnModel, DdtParameterSet, HllcFlux1D, HydroState1D, NickelYieldModel, SimulationResult,
    SimulationSnapshot, SniaCoreSolver, SolverConfig, WhiteDwarfEos,
};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RunMode {
    Quick,
    Full,
}

impl RunMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "snia-ddt",
    about = "Run SN Ia DDT scaffold solver and write reproducible artifacts.",
    after_help = "Examples:\n  snia-ddt --mode quick\n  snia-ddt --mode full --scan --output-dir artifacts/snia_ddt"
)]
struct Args {
    #[arg(long, value_enum, default_value_t = RunMode::Quick)]
    mode: RunMode,
    #[arg(
        long,
        default_value = "artifacts/snia_ddt",
        help = "Base output directory. Mode-specific subdirectories are used."
    )]
    output_dir: PathBuf,
    #[arg(long, help = "Run calibration scan and emit scan CSV output.")]
    scan: bool,
    #[arg(
        long,
        default_value_t = 0.6,
        help = "Target nickel-56 mass for scan objective."
    )]
    target_ni56_msun: f64,
}

fn solver_config_for_mode(mode: RunMode) -> SolverConfig {
    match mode {
        RunMode::Quick => SolverConfig {
            n_cells: 24,
            final_time_s: 8.0e-4,
            max_steps: 96,
            ..SolverConfig::default()
        },
        RunMode::Full => SolverConfig {
            n_cells: 96,
            final_time_s: 2.0e-3,
            max_steps: 512,
            ..SolverConfig::default()
        },
    }
}

fn initial_hydro_for_mode(mode: RunMode) -> HydroState1D {
    match mode {
        RunMode::Quick => HydroState1D {
            density: 2.2e7,
            velocity: 8.0e5,
            pressure: 2.8e23,
            specific_internal_energy: 1.0e17,
        },
        RunMode::Full => HydroState1D {
            density: 2.6e7,
            velocity: 1.2e6,
            pressure: 3.1e23,
            specific_internal_energy: 1.2e17,
        },
    }
}

fn initial_temperature_for_mode(mode: RunMode) -> f64 {
    match mode {
        RunMode::Quick => 1.2e9,
        RunMode::Full => 1.8e9,
    }
}

fn scan_grid_for_mode(mode: RunMode) -> Vec<DdtParameterSet> {
    match mode {
        RunMode::Quick => vec![
            DdtParameterSet {
                transition_density: 1.8e7,
                turbulence_intensity: 0.7,
                ignition_offset_km: 80.0,
            },
            DdtParameterSet {
                transition_density: 2.2e7,
                turbulence_intensity: 1.0,
                ignition_offset_km: 60.0,
            },
            DdtParameterSet {
                transition_density: 2.8e7,
                turbulence_intensity: 1.2,
                ignition_offset_km: 40.0,
            },
        ],
        RunMode::Full => vec![
            DdtParameterSet {
                transition_density: 1.5e7,
                turbulence_intensity: 0.6,
                ignition_offset_km: 90.0,
            },
            DdtParameterSet {
                transition_density: 1.8e7,
                turbulence_intensity: 0.7,
                ignition_offset_km: 80.0,
            },
            DdtParameterSet {
                transition_density: 2.0e7,
                turbulence_intensity: 0.8,
                ignition_offset_km: 70.0,
            },
            DdtParameterSet {
                transition_density: 2.2e7,
                turbulence_intensity: 0.9,
                ignition_offset_km: 60.0,
            },
            DdtParameterSet {
                transition_density: 2.4e7,
                turbulence_intensity: 1.0,
                ignition_offset_km: 55.0,
            },
            DdtParameterSet {
                transition_density: 2.6e7,
                turbulence_intensity: 1.1,
                ignition_offset_km: 50.0,
            },
            DdtParameterSet {
                transition_density: 2.8e7,
                turbulence_intensity: 1.2,
                ignition_offset_km: 45.0,
            },
            DdtParameterSet {
                transition_density: 3.0e7,
                turbulence_intensity: 1.3,
                ignition_offset_km: 40.0,
            },
            DdtParameterSet {
                transition_density: 3.2e7,
                turbulence_intensity: 1.4,
                ignition_offset_km: 35.0,
            },
        ],
    }
}

fn write_summary_csv(
    path: &Path,
    mode: RunMode,
    scan_enabled: bool,
    target_ni56_msun: f64,
    result: &SimulationResult,
    best_scan: Option<CalibrationResult>,
    scan_rows: usize,
) -> Result<(), Box<dyn Error>> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record([
        "mode",
        "scan_enabled",
        "target_ni56_msun",
        "n_steps",
        "final_time_s",
        "burned_mass_msun",
        "nickel56_mass_msun",
        "peak_density",
        "detonation_events",
        "scan_rows",
        "scan_best_objective",
        "scan_best_transition_density",
        "scan_best_turbulence_intensity",
        "scan_best_ignition_offset_km",
        "scan_best_predicted_nickel56_msun",
    ])?;

    let (
        scan_best_objective,
        scan_best_transition_density,
        scan_best_turbulence_intensity,
        scan_best_ignition_offset_km,
        scan_best_predicted_nickel56_msun,
    ) = if let Some(best) = best_scan {
        (
            best.objective.to_string(),
            best.parameter_set.transition_density.to_string(),
            best.parameter_set.turbulence_intensity.to_string(),
            best.parameter_set.ignition_offset_km.to_string(),
            best.predicted_nickel56_msun.to_string(),
        )
    } else {
        (
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        )
    };

    writer.write_record([
        mode.as_str().to_string(),
        scan_enabled.to_string(),
        target_ni56_msun.to_string(),
        result.n_steps.to_string(),
        result.final_time_s.to_string(),
        result.burned_mass_msun.to_string(),
        result.nickel56_mass_msun.to_string(),
        result.peak_density.to_string(),
        result.detonation_events.len().to_string(),
        scan_rows.to_string(),
        scan_best_objective,
        scan_best_transition_density,
        scan_best_turbulence_intensity,
        scan_best_ignition_offset_km,
        scan_best_predicted_nickel56_msun,
    ])?;
    writer.flush()?;
    Ok(())
}

fn write_scan_csv(path: &Path, scan_results: &[CalibrationResult]) -> Result<(), Box<dyn Error>> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record([
        "transition_density",
        "turbulence_intensity",
        "ignition_offset_km",
        "predicted_nickel56_msun",
        "objective",
    ])?;
    for row in scan_results {
        writer.write_record([
            row.parameter_set.transition_density.to_string(),
            row.parameter_set.turbulence_intensity.to_string(),
            row.parameter_set.ignition_offset_km.to_string(),
            row.predicted_nickel56_msun.to_string(),
            row.objective.to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    if args.target_ni56_msun < 0.0 {
        return Err("target_ni56_msun must be non-negative".into());
    }

    let output_dir = args.output_dir.join(args.mode.as_str());
    fs::create_dir_all(&output_dir)?;

    let config = solver_config_for_mode(args.mode);
    let initial_hydro = initial_hydro_for_mode(args.mode);
    let initial_temperature = initial_temperature_for_mode(args.mode);

    let mut solver = SniaCoreSolver::new(
        config,
        WhiteDwarfEos::default(),
        HllcFlux1D::default(),
        CarbonBurnModel::default(),
        NickelYieldModel::default(),
        initial_hydro,
        initial_temperature,
        BurnState::default(),
    )?;

    let result = solver.run()?;
    let snapshot = SimulationSnapshot::from(&result);
    let snapshot_path = output_dir.join("snia_snapshot.toml");
    write_snapshot_toml(&snapshot_path, &snapshot)?;

    let (best_scan, scan_rows) = if args.scan {
        let scan_results = scan_parameter_sets(
            &scan_grid_for_mode(args.mode),
            result.burned_mass_msun,
            args.target_ni56_msun,
            NickelYieldModel::default(),
        );
        let scan_path = output_dir.join("snia_scan.csv");
        write_scan_csv(&scan_path, &scan_results)?;
        (best_parameter_set(&scan_results), scan_results.len())
    } else {
        (None, 0)
    };

    let summary_path = output_dir.join("snia_summary.csv");
    write_summary_csv(
        &summary_path,
        args.mode,
        args.scan,
        args.target_ni56_msun,
        &result,
        best_scan,
        scan_rows,
    )?;

    println!(
        "SN Ia DDT run complete: mode={}, steps={}, t={:.4e} s, Ni56={:.4e} Msun, output={}",
        args.mode.as_str(),
        result.n_steps,
        result.final_time_s,
        result.nickel56_mass_msun,
        output_dir.display()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run(Args::parse())
}
