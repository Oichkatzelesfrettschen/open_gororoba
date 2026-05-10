//! flyby-residual-audit: Quantify the classical (Newtonian + DM density) residual
//! for all 6 Anderson et al. spacecraft, establishing the baseline that the
//! Chingon non-associative model must explain.
//!
//! Uses the flyby/ library modules (EnvironmentModel trait) rather than the
//! production Chingon-Vlasov pipeline. This binary answers: "How much delta-V
//! does a Newtonian trajectory accumulate from NFW/wake DM density alone?"
//!
//! The answer should be ~0 (classical gravity is conservative), confirming
//! that the anomalous delta-V is genuinely non-Newtonian.

use anyhow::Result;
use clap::Parser;
use gororoba_cli_physics::{
    anomaly_residual::{
        FlybyObservedRecord, FlybyResidualRecord, read_csv_records, validate_flyby_catalog,
        write_csv_records,
    },
    flyby::{
        config::{self, ALPHA_CHINGON, ETA_WAKE, FlybyConfig, GM_EARTH, R_EARTH, V_WIND_GALACTIC},
        environment::{EarthOnlyNfwLike, EarthWakeModel, EnvironmentModel, State},
        geometry::{compute_soi_window, dm_wind_j2000, hyperbolic_initial_state},
        integrator::rk4_step,
        report::{FlybyResult, print_comparison_table},
    },
};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "flyby-residual-audit")]
#[command(about = "Generate governed flyby observed-vs-modeled residual artifacts")]
struct Args {
    #[arg(
        long,
        default_value = "data/external/flyby_anomaly/anderson_2008_flyby_catalog.csv"
    )]
    input: PathBuf,
    #[arg(
        long,
        default_value = "data/output/anomaly/flyby/flyby_observed_vs_modeled.csv"
    )]
    out: PathBuf,
    #[arg(long, default_value_t = 1.0)]
    dt: f64,
}

/// Run a flyby through the generic EnvironmentModel pipeline.
///
/// Returns the outbound velocity magnitude after RK4 integration
/// through the SOI window with the given environmental model.
fn run_generic_flyby<E: EnvironmentModel>(cfg: &FlybyConfig, env: &E, dt: f64, alpha: f64) -> f64 {
    let t_window = compute_soi_window(cfg);
    let (pos_init, vel_init) = hyperbolic_initial_state(cfg, t_window);

    let total_time = 2.0 * t_window;
    let steps = (total_time / dt) as usize;

    let mut state = State {
        position: pos_init,
        velocity: vel_init,
    };

    // Acceleration callback: Earth gravity + optional anomalous density-coupled force.
    // The anomalous force is alpha * rho * v (velocity-dependent drag).
    let accel = |s: &State, _t: f64, rho: f64, _tensor: &[[f64; 3]; 3]| -> [f64; 3] {
        let r_sq = s.position[0] * s.position[0]
            + s.position[1] * s.position[1]
            + s.position[2] * s.position[2];
        let r = r_sq.sqrt();
        let gm_r3 = GM_EARTH / (r_sq * r);

        let mut a = [
            -s.position[0] * gm_r3,
            -s.position[1] * gm_r3,
            -s.position[2] * gm_r3,
        ];

        // Anomalous density-coupled drag (simplified scalar model)
        if alpha > 0.0 {
            let drag = alpha * rho;
            a[0] += drag * s.velocity[0];
            a[1] += drag * s.velocity[1];
            a[2] += drag * s.velocity[2];
        }

        a
    };

    let mut t = -t_window;
    for _ in 0..steps {
        rk4_step(&mut state, t, dt, env, &accel);
        t += dt;
    }

    let v_sq = state.velocity[0] * state.velocity[0]
        + state.velocity[1] * state.velocity[1]
        + state.velocity[2] * state.velocity[2];
    v_sq.sqrt()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let observed_rows = read_csv_records::<FlybyObservedRecord>(&args.input)?;
    let flybys = config::all_flybys();
    validate_flyby_catalog(&observed_rows, &flybys)?;
    let observed_by_name: std::collections::BTreeMap<&str, &FlybyObservedRecord> = observed_rows
        .iter()
        .map(|row| (row.name.as_str(), row))
        .collect();
    let dt = args.dt;

    let v_wind = dm_wind_j2000(&V_WIND_GALACTIC);
    let v_wind_norm =
        (v_wind[0] * v_wind[0] + v_wind[1] * v_wind[1] + v_wind[2] * v_wind[2]).sqrt();

    println!("=== Flyby Residual Audit ===");
    println!("  Using flyby/ library (EnvironmentModel trait dispatch)");
    println!(
        "  v_wind (J2000) = ({:.2}, {:.2}, {:.2}) km/s  |v|={:.1}",
        v_wind[0], v_wind[1], v_wind[2], v_wind_norm
    );
    println!("  dt = {:.1} s", dt);
    println!();

    // Model 1: Pure Kepler (no DM density, alpha=0)
    let kepler_env = EarthOnlyNfwLike {
        base_density: 0.0,
        earth_radius_km: R_EARTH,
    };

    // Model 2: NFW 1/r^3 density (no wake)
    //
    // Design intent: NFW and wake models are classical (Newtonian) baselines.
    // They are NOT anomalous -- they represent conservative density-coupled
    // drag that should produce ~zero net delta-V over a symmetric trajectory.
    // The audit confirms that the observed flyby anomaly cannot arise from
    // classical density coupling alone, establishing the baseline that the
    // Chingon non-associative tensor contraction must explain.
    let nfw_env = EarthOnlyNfwLike::default();

    // Model 3: NFW + gravitational focusing wake
    let w_mag = (v_wind[0] * v_wind[0] + v_wind[1] * v_wind[1] + v_wind[2] * v_wind[2]).sqrt();
    let wake_env = EarthWakeModel {
        base_density: 1.0,
        earth_radius_km: R_EARTH,
        wind_direction: if w_mag > 1e-10 {
            [v_wind[0] / w_mag, v_wind[1] / w_mag, v_wind[2] / w_mag]
        } else {
            [1.0, 0.0, 0.0]
        },
        eta_wake: ETA_WAKE,
    };

    println!(
        "{:<28} {:>10} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "Spacecraft", "v_inf", "alt(km)", "v_kepler", "v_nfw", "v_wake", "obs(mm/s)"
    );
    println!("{}", "-".repeat(96));

    let mut results = Vec::new();
    let mut residual_rows = Vec::new();

    for cfg in &flybys {
        let observed = observed_by_name
            .get(cfg.name)
            .expect("flyby catalog validation should guarantee presence");
        let v_kepler = run_generic_flyby(cfg, &kepler_env, dt, 0.0);
        let v_nfw = run_generic_flyby(cfg, &nfw_env, dt, ALPHA_CHINGON);
        let v_wake = run_generic_flyby(cfg, &wake_env, dt, ALPHA_CHINGON);

        let residual_nfw_mm_s = (v_nfw - v_kepler) * 1e6;
        let residual_wake_mm_s = (v_wake - v_kepler) * 1e6;

        println!(
            "{:<28} {:>10.3} {:>10.0} {:>12.6} {:>12.6} {:>12.6} {:>10.2}",
            cfg.name, cfg.v_inf, cfg.perigee_alt_km, v_kepler, v_nfw, v_wake, cfg.observed_dv_mm_s,
        );

        results.push(FlybyResult {
            name: cfg.name.to_string(),
            v_out_control: v_kepler,
            v_out_anomalous: v_wake,
            observed_dv_mm_s: observed.observed_dv_mm_s,
        });

        residual_rows.push(FlybyResidualRecord {
            name: observed.name.clone(),
            perigee_date_utc: observed.perigee_date_utc.clone(),
            perigee_jed: observed.perigee_jed,
            observed_dv_mm_s: observed.observed_dv_mm_s,
            classical_dv_mm_s: 0.0,
            nfw_dv_mm_s: residual_nfw_mm_s,
            wake_dv_mm_s: residual_wake_mm_s,
            residual_wake_minus_observed_mm_s: residual_wake_mm_s - observed.observed_dv_mm_s,
            sign_match: (residual_wake_mm_s > 0.0 && observed.observed_dv_mm_s > 0.0)
                || (residual_wake_mm_s < 0.0 && observed.observed_dv_mm_s < 0.0)
                || (residual_wake_mm_s.abs() < 1e-10 && observed.observed_dv_mm_s.abs() < 1e-10),
            source_bib: observed.source_bib.clone(),
        });

        // CSV-friendly line for piping
        eprintln!(
            "{},{:.3},{:.0},{:.8},{:.8},{:.8},{:.6e},{:.6e},{:.2}",
            cfg.name,
            cfg.v_inf,
            cfg.perigee_alt_km,
            v_kepler,
            v_nfw,
            v_wake,
            residual_nfw_mm_s,
            residual_wake_mm_s,
            cfg.observed_dv_mm_s,
        );
    }

    println!();
    println!("=== Comparison Table (wake model vs observed) ===");
    print_comparison_table(&results);
    write_csv_records(&args.out, &residual_rows)?;
    println!();
    println!("artifact: {}", args.out.display());

    println!();
    println!("=== Audit Verdict ===");
    println!("  If residuals are O(1e-6) mm/s, classical density coupling");
    println!("  does NOT explain the flyby anomaly. The anomalous delta-V");
    println!("  must come from non-Newtonian physics (Chingon tensor contraction).");
    Ok(())
}
