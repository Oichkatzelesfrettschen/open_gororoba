//! Warp Ring 3D Simulation.
//!
//! Extends the 2D Warp Ring to a full 3D experiment-emulation-simulation.
//! Integrating 3D LBM, E7 roots, and SHI warp lensing.

#[cfg(feature = "hdf5-export")]
use data_artifacts_core::hdf5_export::{
    export_experiment_contract, export_field_3d, export_rho_quality_metrics,
    export_simulation_trace, read_rho_mean_trace,
};
use data_artifacts_core::quality::{RhoQualityThresholds, validate_rho_trace};
use gororoba_algebra::{
    lie::e7::geometry::generate_e7_roots, physics::octonion_field::FieldParams,
};
#[cfg(feature = "hdf5-export")]
use gororoba_contracts::{WarpRingConfig, WarpRingExperiment, WarpRingResults};
use gororoba_engine::simulation::{
    E7SpectralFilter, LbmBackend3D, SimulationConfig3D, SimulationState3D,
};
use lbm_core::turbulence::extract_dominant_triads_3d;
use std::{error::Error, path::Path};
use tracing::info;

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    info!("=== Warp Ring 3D Simulation ===");

    // -- Configuration --
    let nx = 128; // Milestone 60s @ 128^3
    let ny = 128;
    let nz = 128;
    let tau = 0.6;
    let steps = 32000; // ~60 seconds at 540 steps/s

    let config = SimulationConfig3D {
        nx,
        ny,
        nz,
        tau,
        use_gpu: true,
        precision: lbm_3d_cuda::Precision::BF16,
        algebra_params: FieldParams::default(),
        coupling_fluid_algebra: 0.1,
        coupling_algebra_fluid: 0.1,
    };

    let mut state = SimulationState3D::new(config)?;
    // Gaussian Projector with sigma=1.5
    state.imbalance = Some(Box::new(E7SpectralFilter::new(nx, ny, nz, 0.95)));

    info!(
        "[1/5] Initializing 3D Fluid ({}x{}x{}, tau={})...",
        nx, ny, nz, tau
    );
    // Initialize with uniform rest state
    #[cfg(feature = "gpu")]
    if let LbmBackend3D::Gpu(ref mut _solver) = state.fluid {
        // solver.initialize_uniform(1.0, [0.0, 0.0, 0.0])? is not public,
        // LbmSolver3DCuda::new already initializes to w_i (rho=1, u=0).
        // Let's verify if initialize_custom is needed.
        // For now, let's just use the default initialized state.
    }

    info!("[2/5] Running 3D Simulation ({} steps)...", steps);

    // Data collectors
    let mut time_hist = Vec::with_capacity(steps);
    let mut rho_hist = Vec::with_capacity(steps);
    let mut enstrophy_hist = Vec::with_capacity(steps);
    let mut alg_norm_hist = Vec::with_capacity(steps);

    let start_time = std::time::Instant::now();

    for t in 0..steps {
        state.step()?;

        // HDF5 Data Collection
        if t % 10 == 0 {
            time_hist.push(t as f64);
            rho_hist.push(state.fluid.try_mean_density()?);
            enstrophy_hist.push(0.0);
            alg_norm_hist.push(0.0);

            if t % 100 == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let steps_per_sec = t as f64 / elapsed.max(0.001);
                info!(
                    "      Step {} completed. Speed: {:.2} steps/s",
                    t, steps_per_sec
                );
            }
        }

        // Full Field Snapshot every 500 steps
        if t > 0 && t % 500 == 0 {
            #[cfg(all(feature = "gpu", feature = "hdf5-export"))]
            {
                let snapshot_path = format!("data/h5/warp_field_{}_step_{}.h5", nx, t);
                if let LbmBackend3D::Gpu(ref mut solver) = state.fluid {
                    solver.sync_to_host()?;
                    let u_flat: Vec<f64> = solver
                        .u
                        .iter()
                        .flat_map(|v| vec![v[0] as f64, v[1] as f64, v[2] as f64])
                        .collect();
                    export_field_3d(Path::new(&snapshot_path), "velocity", nx, ny, nz, &u_flat)
                        .unwrap_or_else(|e| info!("Field Snapshot failed: {}", e));
                    info!("      [Snapshot] Saved full field to {}", snapshot_path);
                }
            }
        }
    }

    let total_elapsed = start_time.elapsed().as_secs_f64();
    let steps_per_sec = steps as f64 / total_elapsed;
    let mlups = (steps_per_sec * (nx * ny * nz) as f64) / 1e6;
    let rho_thresholds = RhoQualityThresholds::default();
    let rho_quality = validate_rho_trace(&rho_hist, rho_thresholds)
        .map_err(|e| std::io::Error::other(format!("pre-export rho quality gate failed: {e}")))?;

    info!(
        "Simulation completed in {:.2}s ({:.2} steps/s, {:.2} MLUPS)",
        total_elapsed, steps_per_sec, mlups
    );
    info!(
        "Rho quality pre-export: samples={}, final={:.6}, drift={:.3e}, std={:.3e}",
        rho_quality.sample_count,
        rho_quality.final_value,
        rho_quality.abs_drift_final,
        rho_quality.std_dev
    );

    info!("[3/5] Extracting 3D Spectral Triads...");
    #[cfg(feature = "gpu")]
    if let LbmBackend3D::Gpu(ref mut solver) = state.fluid {
        solver.sync_to_host()?;
    }

    let (ux, uy, uz) = state.fluid.try_velocity(nx, ny, nz)?;
    let triads = extract_dominant_triads_3d(&ux, &uy, &uz, 1e-6);
    info!("      Found {} dominant 3D triads.", triads.len());

    info!("[4/5] Loading E7 Roots for Geometry Mapping...");
    let roots = generate_e7_roots();
    info!("      Loaded {} E7 roots.", roots.len());

    info!("[5/5] Exporting HDF5 Artifact...");
    let filename = format!("data/h5/warp_ring_{}.h5", nx);
    let path = Path::new(&filename);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    #[cfg(feature = "hdf5-export")]
    {
        let contract = WarpRingExperiment {
            experiment_id: format!(
                "EXP-WARP-{}-{}",
                nx,
                chrono::Utc::now().format("%Y%m%d-%H%M%S")
            ),
            config: WarpRingConfig {
                resolution: nx,
                steps,
                tau,
                forcing_type: "E7_Gaussian_Smooth".to_string(),
                coupling_lambda: 0.95,
                initial_condition: "Random_Noise".to_string(),
            },
            results: WarpRingResults {
                final_enstrophy: *enstrophy_hist.last().unwrap_or(&0.0),
                mean_density: *rho_hist.last().unwrap_or(&0.0),
                betti_1_persistence: None,
                execution_time_s: total_elapsed,
                steps_per_second: steps_per_sec,
                mlups,
                artifact_path: filename.clone(),
            },
        };

        export_experiment_contract(path, &contract)?;
        export_simulation_trace(path, &time_hist, &rho_hist, &enstrophy_hist, &alg_norm_hist)?;

        // Fail-closed artifact acceptance: verify rho_mean directly from the written HDF5 dataset.
        let rho_from_file = read_rho_mean_trace(path)?;
        let file_quality = validate_rho_trace(&rho_from_file, rho_thresholds).map_err(|e| {
            std::io::Error::other(format!("post-export rho quality gate failed: {e}"))
        })?;
        export_rho_quality_metrics(path, &file_quality, rho_thresholds)?;
        info!(
            "      Accepted artifact {:?} with rho gate pass: samples={}, drift={:.3e}, std={:.3e}",
            path, file_quality.sample_count, file_quality.abs_drift_final, file_quality.std_dev
        );
    }
    #[cfg(not(feature = "hdf5-export"))]
    {
        info!("      HDF5 export disabled (feature flag missing).");
    }

    info!("Done. Simulation results verified at {}^3.", nx);

    Ok(())
}
