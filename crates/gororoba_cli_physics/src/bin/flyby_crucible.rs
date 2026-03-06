//! flyby-crucible: Test the 64D Chingon non-alternative drag against
//! anomalous spacecraft flyby velocity jumps.
//!
//! Universality test: the coupling constant alpha_chingon = 8.0e-14 is LOCKED.
//! Different trajectories produce different delta-V purely from geometry.
//!
//! Spacecraft database: Galileo (1990), NEAR (1998), MESSENGER (2005),
//! Cassini (1999), Juno (2013), Rosetta-I (2005).

use algebra_core::construction::chingon::AlternativityViolationTensor;
use clap::Parser;
use gr_core::forces::chingon_drag::compute_chingon_drag;
use nalgebra::Vector3;


/// Locked coupling constant -- NOT a tunable parameter.
const ALPHA_CHINGON: f64 = 8.0e-14;

/// Earth GM in km^3/s^2.
const GM_EARTH: f64 = 398600.4418;

/// Earth radius in km.
const R_EARTH: f64 = 6371.0;

/// Dark matter wind velocity in Galactic frame (km/s).
/// Sun moves at ~220 km/s through the Milky Way; Earth's orbital motion
/// adds a seasonal modulation we ignore here.
const V_WIND: [f64; 3] = [-30.0, 220.0, 0.0];

/// Dark matter local density (normalized, dimensionless coupling absorbed into alpha).
const RHO_DM: f64 = 1.0;

/// Configuration for a single flyby event.
#[derive(Debug, Clone)]
struct FlybyConfig {
    name: &'static str,
    /// Perigee altitude above Earth surface (km).
    perigee_alt_km: f64,
    /// Hyperbolic excess velocity v_inf (km/s).
    v_inf: f64,
    /// Inbound asymptotic declination (degrees, positive = North).
    inbound_dec_deg: f64,
    /// Inbound asymptotic right ascension (degrees).
    inbound_ra_deg: f64,
    /// Observed anomalous delta-V (mm/s). Positive = speed gain.
    observed_dv_mm_s: f64,
}

fn all_flybys() -> Vec<FlybyConfig> {
    vec![
        FlybyConfig {
            name: "Galileo-I (1990-12-08)",
            perigee_alt_km: 960.0,
            v_inf: 8.949,
            inbound_dec_deg: -12.5,
            inbound_ra_deg: 263.0,
            observed_dv_mm_s: 3.92,
        },
        FlybyConfig {
            name: "NEAR (1998-01-23)",
            perigee_alt_km: 539.0,
            v_inf: 6.851,
            inbound_dec_deg: -20.8,
            inbound_ra_deg: 280.0,
            observed_dv_mm_s: 13.46,
        },
        FlybyConfig {
            name: "Cassini (1999-08-18)",
            perigee_alt_km: 1175.0,
            v_inf: 16.01,
            inbound_dec_deg: -12.9,
            inbound_ra_deg: 257.0,
            observed_dv_mm_s: -2.0,
        },
        FlybyConfig {
            name: "Rosetta-I (2005-03-04)",
            perigee_alt_km: 1956.0,
            v_inf: 3.863,
            inbound_dec_deg: -34.3,
            inbound_ra_deg: 247.0,
            observed_dv_mm_s: 1.80,
        },
        FlybyConfig {
            name: "MESSENGER (2005-08-02)",
            perigee_alt_km: 2347.0,
            v_inf: 4.056,
            inbound_dec_deg: 31.4,
            inbound_ra_deg: 232.0,
            observed_dv_mm_s: 0.02,
        },
        FlybyConfig {
            name: "Juno (2013-10-09)",
            perigee_alt_km: 559.0,
            v_inf: 9.897,
            inbound_dec_deg: -13.6,
            inbound_ra_deg: 0.0,
            observed_dv_mm_s: 0.0,
        },
    ]
}

/// Convert asymptotic direction (RA, Dec in degrees) to a unit vector.
fn radec_to_unit(ra_deg: f64, dec_deg: f64) -> Vector3<f64> {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
}

/// Compute initial position and velocity at T seconds before perigee
/// for a hyperbolic flyby in the orbital plane.
///
/// The orbital plane is defined by the inbound asymptotic direction
/// and a perpendicular in the equatorial plane.
fn hyperbolic_initial_state(
    cfg: &FlybyConfig,
    t_before_perigee: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let r_perigee = R_EARTH + cfg.perigee_alt_km;
    let v_inf = cfg.v_inf;

    // Semi-major axis (negative for hyperbola)
    let a = -GM_EARTH / (v_inf * v_inf);
    // Eccentricity
    let e = 1.0 - r_perigee / a;
    // Semi-latus rectum
    let p = a * (1.0 - e * e);

    // Mean motion
    let n = (-GM_EARTH / (a * a * a)).sqrt();

    // Mean anomaly at T before perigee (hyperbolic: M = e * sinh(H) - H)
    // We need to find H from M = -n * t_before_perigee (negative = before perigee)
    let m_target = -n * t_before_perigee;

    // Solve Kepler's equation for hyperbolic anomaly H via Newton-Raphson
    let mut h = m_target / e; // initial guess
    for _ in 0..50 {
        let f_h = e * h.sinh() - h - m_target;
        let fp_h = e * h.cosh() - 1.0;
        if fp_h.abs() < 1e-30 {
            break;
        }
        h -= f_h / fp_h;
    }

    // True anomaly from hyperbolic anomaly
    let cos_nu = (e - h.cosh()) / (1.0 - e * h.cosh());
    let sin_nu = (e * e - 1.0).sqrt() * h.sinh() / (1.0 - e * h.cosh());
    let nu = sin_nu.atan2(cos_nu);

    // Radius at this point
    let r = p / (1.0 + e * nu.cos());

    // Position in perifocal frame (x toward perigee, y perpendicular)
    let x_pf = r * nu.cos();
    let y_pf = r * nu.sin();

    // Velocity in perifocal frame
    let h_ang = (GM_EARTH * p).sqrt(); // specific angular momentum
    let vx_pf = -GM_EARTH / h_ang * nu.sin();
    let vy_pf = GM_EARTH / h_ang * (e + nu.cos());

    // Rotate perifocal frame to ECI using asymptotic direction
    let inbound_dir = radec_to_unit(cfg.inbound_ra_deg, cfg.inbound_dec_deg);

    // Build a rotation matrix: perifocal x-axis aligned with -inbound_dir
    // (perigee is approached from inbound direction)
    let x_hat = -inbound_dir;
    // Pick a perpendicular direction in the equatorial plane
    let z_ref = Vector3::new(0.0, 0.0, 1.0);
    let y_hat = z_ref.cross(&x_hat);
    let y_hat = if y_hat.norm() < 1e-10 {
        // Degenerate: inbound near pole, use x-ref instead
        let x_ref = Vector3::new(1.0, 0.0, 0.0);
        x_ref.cross(&x_hat).normalize()
    } else {
        y_hat.normalize()
    };
    let z_hat = x_hat.cross(&y_hat);

    let pos = x_hat * x_pf + y_hat * y_pf;
    let vel = x_hat * vx_pf + y_hat * vy_pf;

    // Sanity: ignore z_hat in position (2D orbital plane), but the rotation
    // embeds it in 3D space correctly.
    let _ = z_hat;

    (pos, vel)
}

/// Run a single flyby simulation with RK4 integration.
/// Returns (v_out_control, v_out_chingon, trajectory_points).
fn run_flyby(
    cfg: &FlybyConfig,
    avt: &AlternativityViolationTensor,
    dt: f64,
    t_before: f64,
    t_after: f64,
    trajectory_stride: usize,
) -> (f64, f64, Vec<[f64; 7]>) {
    let (pos_init, vel_init) = hyperbolic_initial_state(cfg, t_before);
    let total_time = t_before + t_after;
    let steps = (total_time / dt) as usize;
    let v_wind = Vector3::new(V_WIND[0], V_WIND[1], V_WIND[2]);

    let rk4_run = |use_chingon: bool, record: bool| -> (f64, Vec<[f64; 7]>) {
        let mut p = pos_init;
        let mut v = vel_init;
        let mut traj = Vec::new();

        for step in 0..steps {
            if record && step % trajectory_stride == 0 {
                let t = step as f64 * dt - t_before;
                traj.push([t, p.x, p.y, p.z, v.x, v.y, v.z]);
            }

            let accel = |p_in: Vector3<f64>, v_in: Vector3<f64>| -> Vector3<f64> {
                let r_sq = p_in.norm_squared();
                let r = r_sq.sqrt();
                let a_grav = -p_in * (GM_EARTH / (r_sq * r));
                if use_chingon {
                    a_grav
                        + compute_chingon_drag(v_in - v_wind, RHO_DM, ALPHA_CHINGON, avt)
                } else {
                    a_grav
                }
            };

            let k1_v = accel(p, v);
            let k1_p = v;

            let k2_v = accel(p + k1_p * (dt / 2.0), v + k1_v * (dt / 2.0));
            let k2_p = v + k1_v * (dt / 2.0);

            let k3_v = accel(p + k2_p * (dt / 2.0), v + k2_v * (dt / 2.0));
            let k3_p = v + k2_v * (dt / 2.0);

            let k4_v = accel(p + k3_p * dt, v + k3_v * dt);
            let k4_p = v + k3_v * dt;

            p += (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * (dt / 6.0);
            v += (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * (dt / 6.0);
        }

        if record {
            let t = steps as f64 * dt - t_before;
            traj.push([t, p.x, p.y, p.z, v.x, v.y, v.z]);
        }

        (v.norm(), traj)
    };

    let (v_ctrl, _) = rk4_run(false, false);
    let (v_chingon, traj) = rk4_run(true, true);

    (v_ctrl, v_chingon, traj)
}

#[derive(Parser)]
#[command(name = "flyby-crucible")]
#[command(about = "64D Chingon-Vlasov flyby anomaly universality test")]
struct Cli {
    /// Run only a specific spacecraft (galileo, near, cassini, rosetta, messenger, juno).
    /// If omitted, runs all spacecraft.
    #[arg(long)]
    spacecraft: Option<String>,

    /// Integration timestep in seconds.
    #[arg(long, default_value = "10.0")]
    dt: f64,

    /// Time before perigee to start integration (seconds).
    #[arg(long, default_value = "2592000.0")]
    t_before: f64,

    /// Time after perigee to end integration (seconds).
    #[arg(long, default_value = "2592000.0")]
    t_after: f64,

    /// Output trajectory CSV file prefix.
    #[arg(long)]
    csv_prefix: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    println!("=== 64D Chingon-Vlasov Flyby Crucible ===");
    println!("  alpha_chingon = {:.2e} (LOCKED)", ALPHA_CHINGON);
    println!("  v_wind = ({:.0}, {:.0}, {:.0}) km/s", V_WIND[0], V_WIND[1], V_WIND[2]);
    println!("  dt = {:.1} s", cli.dt);
    println!(
        "  window = T-{:.0} to T+{:.0} days",
        cli.t_before / 86400.0,
        cli.t_after / 86400.0
    );
    println!();

    // Compute AVT once (expensive at dim=64)
    println!("Computing 64D Alternativity Violation Tensor...");
    let avt = AlternativityViolationTensor::new(64);
    println!("  AVT: {} violations", avt.violations.len());
    println!();

    let all = all_flybys();
    let configs: Vec<&FlybyConfig> = if let Some(ref name) = cli.spacecraft {
        let key = name.to_lowercase();
        let filtered: Vec<&FlybyConfig> = all
            .iter()
            .filter(|c| c.name.to_lowercase().contains(&key))
            .collect();
        if filtered.is_empty() {
            anyhow::bail!(
                "Unknown spacecraft '{}'. Available: galileo, near, cassini, rosetta, messenger, juno",
                name
            );
        }
        filtered
    } else {
        all.iter().collect()
    };

    // Trajectory recording stride: keep ~500 points
    let total_steps = ((cli.t_before + cli.t_after) / cli.dt) as usize;
    let stride = (total_steps / 500).max(1);

    println!(
        "{:>30} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "Spacecraft", "Obs (mm/s)", "Pred (mm/s)", "Ratio", "Perigee(km)", "v_inf(km/s)"
    );
    println!("{}", "-".repeat(94));

    for cfg in &configs {
        let (v_ctrl, v_chingon, traj) = run_flyby(
            cfg,
            &avt,
            cli.dt,
            cli.t_before,
            cli.t_after,
            stride,
        );

        let delta_v_mm_s = (v_chingon - v_ctrl) * 1e6;
        let ratio = if cfg.observed_dv_mm_s.abs() > 0.001 {
            delta_v_mm_s / cfg.observed_dv_mm_s
        } else {
            f64::NAN
        };

        println!(
            "{:>30} {:>10.2} {:>10.2} {:>12.3} {:>12.0} {:>10.3}",
            cfg.name,
            cfg.observed_dv_mm_s,
            delta_v_mm_s,
            ratio,
            cfg.perigee_alt_km,
            cfg.v_inf
        );

        // Write trajectory CSV if requested
        if let Some(ref prefix) = cli.csv_prefix {
            let safe_name: String = cfg
                .name
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect();
            let path = format!("{}_{}.csv", prefix, safe_name);
            let mut wtr = csv::Writer::from_path(&path)?;
            wtr.write_record(["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])?;
            for row in &traj {
                wtr.write_record(row.iter().map(|v| format!("{:.6}", v)))?;
            }
            wtr.flush()?;
            eprintln!("  Wrote {}", path);
        }
    }

    println!();
    println!("=== Universality Verdict ===");
    println!("  If pred/obs ratio is consistent across all spacecraft,");
    println!("  the coupling constant is universal.");
    println!("  Tolerance: |pred/obs - 1| < 0.25 for each flyby.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_initial_state_sanity() {
        let cfg = &all_flybys()[0]; // Galileo
        let (pos, vel) = hyperbolic_initial_state(cfg, 86400.0); // T-1 day

        // Position should be far from Earth (> perigee)
        let r = pos.norm();
        let r_perigee = R_EARTH + cfg.perigee_alt_km;
        assert!(
            r > r_perigee,
            "At T-1 day, r={:.0} should be > r_perigee={:.0}",
            r,
            r_perigee
        );

        // Speed should be close to v_inf at large distances
        let v = vel.norm();
        assert!(
            v > cfg.v_inf * 0.8 && v < cfg.v_inf * 3.0,
            "Speed {:.3} km/s unreasonable for v_inf={:.3}",
            v,
            cfg.v_inf
        );
    }

    #[test]
    fn test_flyby_configs_physical() {
        for cfg in &all_flybys() {
            assert!(cfg.perigee_alt_km > 0.0, "{}: negative perigee", cfg.name);
            assert!(cfg.v_inf > 0.0, "{}: negative v_inf", cfg.name);
            assert!(
                cfg.inbound_dec_deg.abs() <= 90.0,
                "{}: declination out of range",
                cfg.name
            );
        }
    }

    #[test]
    fn test_radec_to_unit_normalization() {
        for ra in [0.0, 90.0, 180.0, 270.0] {
            for dec in [-90.0, -45.0, 0.0, 45.0, 90.0] {
                let u = radec_to_unit(ra, dec);
                assert!(
                    (u.norm() - 1.0).abs() < 1e-12,
                    "Unit vector norm = {} at RA={}, Dec={}",
                    u.norm(),
                    ra,
                    dec
                );
            }
        }
    }

    #[test]
    fn test_alpha_locked() {
        // The coupling constant must be exactly 8.0e-14
        assert_eq!(ALPHA_CHINGON, 8.0e-14);
    }
}
