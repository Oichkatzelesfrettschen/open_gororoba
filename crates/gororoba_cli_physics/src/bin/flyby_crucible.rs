//! flyby-crucible: Test the N-dimensional Chingon non-alternative drag against
//! anomalous spacecraft flyby velocity jumps.
//!
//! Universality test: the coupling constant alpha_chingon = 8.0e-14 is LOCKED.
//! Different trajectories produce different delta-V purely from geometry.
//!
//! Spacecraft database: Galileo (1990), NEAR (1998), MESSENGER (2005),
//! Cassini (1999), Juno (2013), Rosetta-I (2005).

use clap::Parser;
use gororoba_algebra::construction::chingon::AlternativityViolationTensor;
#[cfg(feature = "gpu")]
use gororoba_algebra::gpu::GpuPackableAvt;
use gororoba_cli_physics::{
    ephemeris_loader::{EphemerisLoader, GM_MOON, GM_SUN},
    flyby::{
        config::{
            ALPHA_CHINGON, ETA_WAKE, FlybyConfig, GM_EARTH, R_EARTH, SOI_R_EARTH, V_WIND_GALACTIC,
            all_flybys,
        },
        geometry::{
            compute_soi_window, dm_wake_density_factor, dm_wind_j2000, hyperbolic_initial_state,
            radec_to_unit,
        },
    },
};
#[cfg(feature = "gpu")]
use gr_core::forces::chingon_bivector_drag::ThreeBodyOrbitalParams;
use gr_core::forces::chingon_bivector_drag::compute_chingon_bivector_drag_3body;
#[cfg(feature = "gpu")]
use lbm_3d_cuda::chingon_gpu::ChingonGpuPipeline;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// --- Inline 3-vector helpers (replaces nalgebra::Vector3 at the CLI boundary) ---
#[inline]
fn v3_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn v3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn v3_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn v3_norm_sq(a: [f64; 3]) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}
#[inline]
fn v3_norm(a: [f64; 3]) -> f64 {
    v3_norm_sq(a).sqrt()
}
#[inline]
fn v3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn v3_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn v3_add_assign(a: &mut [f64; 3], b: [f64; 3]) {
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
}
#[inline]
fn v3_sub_assign(a: &mut [f64; 3], b: [f64; 3]) {
    a[0] -= b[0];
    a[1] -= b[1];
    a[2] -= b[2];
}
#[inline]
fn v3_add4(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> [f64; 3] {
    [
        a[0] + b[0] + c[0] + d[0],
        a[1] + b[1] + c[1] + d[1],
        a[2] + b[2] + c[2] + d[2],
    ]
}

/// Type alias for the GPU pipeline, gated behind the `gpu` feature.
/// When `gpu` is disabled, we use a unit struct so function signatures
/// remain identical (the `Option` is always `None`).
#[cfg(feature = "gpu")]
type GpuPipeline = ChingonGpuPipeline;
#[cfg(not(feature = "gpu"))]
type GpuPipeline = ();

/// Run a single flyby simulation with RK4 integration.
/// Returns (v_out_control, v_out_chingon, trajectory_points, h_trace).
///
/// When `ephem` is Some, Moon and Sun gravitational accelerations are added
/// to the RK4 closure (three-body correction). The ephemeris is queried at
/// the JED corresponding to each integration timestep.
#[allow(clippy::too_many_arguments)]
fn run_flyby(
    cfg: &FlybyConfig,
    avt: &AlternativityViolationTensor,
    dt: f64,
    t_before: f64,
    t_after: f64,
    trajectory_stride: usize,
    v_wind: &[f64; 3],
    ephem: Option<&EphemerisLoader>,
    trace_h: bool,
    eta_wake: f64,
    #[cfg_attr(not(feature = "gpu"), allow(unused_variables))] gpu_pipeline: Option<
        &Mutex<GpuPipeline>,
    >,
) -> (f64, f64, Vec<[f64; 7]>, Vec<[f64; 12]>) {
    let (pos_init, vel_init) = hyperbolic_initial_state(cfg, t_before);
    let total_time = t_before + t_after;
    let steps = (total_time / dt) as usize;

    /// Seconds per Julian day.
    const SPD: f64 = 86400.0;

    let rk4_run = |use_chingon: bool,
                   record: bool,
                   do_trace: bool|
     -> (f64, Vec<[f64; 7]>, Vec<[f64; 12]>) {
        let mut p = pos_init;
        let mut v = vel_init;
        let mut traj = Vec::new();
        let mut h_trace_out: Vec<[f64; 12]> = Vec::new();

        for step in 0..steps {
            let t_sec = step as f64 * dt - t_before;

            if record && step % trajectory_stride == 0 {
                traj.push([t_sec, p[0], p[1], p[2], v[0], v[1], v[2]]);
            }

            // Current JED for ephemeris queries (perigee + offset in days)
            let jed_now = cfg.perigee_jed + t_sec / SPD;

            // Pre-fetch three-body positions (once per step, shared across RK4 stages).
            // Moon/Sun move negligibly during a single RK4 step (~1s), so querying
            // once per step rather than per stage is both correct and 4x faster.
            let (r_moon, r_sun, _emb_offset) = if let Some(eph) = ephem {
                let state = eph.three_body_state(jed_now);
                (state.moon_pos_km, state.sun_pos_km, state.emb_offset_km)
            } else {
                ([0.0; 3], [0.0; 3], [0.0; 3])
            };

            let accel = |p_in: [f64; 3], _v_in: [f64; 3]| -> [f64; 3] {
                let r_sq = v3_norm_sq(p_in);
                let r = r_sq.sqrt();
                let mut a = v3_scale(p_in, -(GM_EARTH / (r_sq * r)));

                // Three-body: Moon and Sun gravitational perturbations
                if ephem.is_some() {
                    let dp_moon = v3_sub(p_in, r_moon);
                    let d_moon = v3_norm(dp_moon);
                    if d_moon > 1.0 {
                        v3_sub_assign(
                            &mut a,
                            v3_scale(dp_moon, GM_MOON / (d_moon * d_moon * d_moon)),
                        );
                        // Indirect term: acceleration of Earth by Moon
                        let d_moon_0 = v3_norm(r_moon);
                        if d_moon_0 > 1.0 {
                            v3_sub_assign(
                                &mut a,
                                v3_scale(r_moon, GM_MOON / (d_moon_0 * d_moon_0 * d_moon_0)),
                            );
                        }
                    }
                    let dp_sun = v3_sub(p_in, r_sun);
                    let d_sun = v3_norm(dp_sun);
                    if d_sun > 1.0 {
                        v3_sub_assign(&mut a, v3_scale(dp_sun, GM_SUN / (d_sun * d_sun * d_sun)));
                        // Indirect term: acceleration of Earth by Sun
                        let d_sun_0 = v3_norm(r_sun);
                        if d_sun_0 > 1.0 {
                            v3_sub_assign(
                                &mut a,
                                v3_scale(r_sun, GM_SUN / (d_sun_0 * d_sun_0 * d_sun_0)),
                            );
                        }
                    }
                }

                if use_chingon {
                    let alpha_eff = ALPHA_CHINGON
                        * dm_wake_density_factor(v3_norm(p_in), &p_in, v_wind, eta_wake);

                    #[cfg(feature = "gpu")]
                    let gpu_force = gpu_pipeline.and_then(|mtx| {
                        let params = ThreeBodyOrbitalParams::compute(
                            nalgebra::Vector3::from(p_in),
                            nalgebra::Vector3::from(_v_in),
                            nalgebra::Vector3::from(*v_wind),
                            nalgebra::Vector3::from(r_moon),
                            nalgebra::Vector3::from(r_sun),
                        )?;
                        let mut pipe = mtx.lock().ok()?;
                        let f = pipe
                            .compute_force_3body(
                                &params.triad_earth,
                                &params.triad_lunar,
                                &params.triad_solar,
                                &params.h_triad_earth,
                                &params.vrel_triad_lunar,
                                &params.h_triad_solar,
                                &params.vhat_triad_solar,
                                params.h_earth_norm,
                                params.v_rel_norm,
                                params.cross_sign,
                                alpha_eff,
                            )
                            .ok()?;
                        Some(f)
                    });

                    #[cfg(feature = "gpu")]
                    {
                        if let Some(f) = gpu_force {
                            v3_add_assign(&mut a, f);
                        } else {
                            let f = compute_chingon_bivector_drag_3body(
                                p_in, _v_in, *v_wind, alpha_eff, avt, r_moon, r_sun,
                            );
                            v3_add_assign(&mut a, f);
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        let f = compute_chingon_bivector_drag_3body(
                            p_in, _v_in, *v_wind, alpha_eff, avt, r_moon, r_sun,
                        );
                        v3_add_assign(&mut a, f);
                    }
                }

                a
            };

            // h(t).v_wind trace for diagnostics (geocentric h + body h norms)
            if do_trace && step % trajectory_stride == 0 {
                let h = v3_cross(p, v);
                let h_dot_vw = v3_dot(h, *v_wind);
                let cross_sign = if h_dot_vw > 0.0 { 1.0 } else { -1.0 };
                let dm_fac = dm_wake_density_factor(v3_norm(p), &p, v_wind, eta_wake);
                let h_lunar = v3_cross(v3_sub(p, r_moon), v);
                let h_solar = v3_cross(v3_sub(p, r_sun), v);
                let a_chingon_mag = if use_chingon {
                    let alpha_eff = ALPHA_CHINGON * dm_fac;
                    let f = compute_chingon_bivector_drag_3body(
                        p, v, *v_wind, alpha_eff, avt, r_moon, r_sun,
                    );
                    v3_norm(f)
                } else {
                    0.0
                };
                let a_moon_mag = if ephem.is_some() {
                    let dp = v3_sub(p, r_moon);
                    let d = v3_norm(dp);
                    if d > 1.0 { GM_MOON / (d * d) } else { 0.0 }
                } else {
                    0.0
                };
                let a_sun_mag = if ephem.is_some() {
                    let dp = v3_sub(p, r_sun);
                    let d = v3_norm(dp);
                    if d > 1.0 { GM_SUN / (d * d) } else { 0.0 }
                } else {
                    0.0
                };
                h_trace_out.push([
                    t_sec,
                    h[0],
                    h[1],
                    h[2],
                    h_dot_vw,
                    cross_sign,
                    a_chingon_mag,
                    dm_fac,
                    a_moon_mag,
                    a_sun_mag,
                    v3_norm(h_lunar),
                    v3_norm(h_solar),
                ]);
            }

            let k1_v = accel(p, v);
            let k1_p = v;

            let k2_v = accel(
                v3_add(p, v3_scale(k1_p, dt / 2.0)),
                v3_add(v, v3_scale(k1_v, dt / 2.0)),
            );
            let k2_p = v3_add(v, v3_scale(k1_v, dt / 2.0));

            let k3_v = accel(
                v3_add(p, v3_scale(k2_p, dt / 2.0)),
                v3_add(v, v3_scale(k2_v, dt / 2.0)),
            );
            let k3_p = v3_add(v, v3_scale(k2_v, dt / 2.0));

            let k4_v = accel(v3_add(p, v3_scale(k3_p, dt)), v3_add(v, v3_scale(k3_v, dt)));
            let k4_p = v3_add(v, v3_scale(k3_v, dt));

            v3_add_assign(
                &mut p,
                v3_scale(
                    v3_add4(k1_p, v3_scale(k2_p, 2.0), v3_scale(k3_p, 2.0), k4_p),
                    dt / 6.0,
                ),
            );
            v3_add_assign(
                &mut v,
                v3_scale(
                    v3_add4(k1_v, v3_scale(k2_v, 2.0), v3_scale(k3_v, 2.0), k4_v),
                    dt / 6.0,
                ),
            );
        }

        if record {
            let t = steps as f64 * dt - t_before;
            traj.push([t, p[0], p[1], p[2], v[0], v[1], v[2]]);
        }

        (v3_norm(v), traj, h_trace_out)
    };

    let (v_ctrl, _, _) = rk4_run(false, false, false);
    let (v_chingon, traj, h_trace_data) = rk4_run(true, true, trace_h);

    (v_ctrl, v_chingon, traj, h_trace_data)
}

/// Pin the current thread pool to physical cores for V-Cache locality.
fn pin_physical_cores() {
    #[cfg(target_os = "linux")]
    {
        use std::{collections::BTreeMap, fs};

        if let Ok(online) = fs::read_to_string("/sys/devices/system/cpu/online") {
            let mut core_groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
            for part in online.trim().split(',') {
                let range: Vec<&str> = part.split('-').collect();
                let (lo, hi) = if range.len() == 2 {
                    (
                        range[0].parse::<usize>().unwrap_or(0),
                        range[1].parse::<usize>().unwrap_or(0),
                    )
                } else {
                    let v = range[0].parse::<usize>().unwrap_or(0);
                    (v, v)
                };
                for cpu in lo..=hi {
                    let pkg = fs::read_to_string(format!(
                        "/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
                    ))
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                    let core = fs::read_to_string(format!(
                        "/sys/devices/system/cpu/cpu{cpu}/topology/core_id"
                    ))
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(cpu);
                    core_groups.entry((pkg, core)).or_default().push(cpu);
                }
            }
            let physical_ids: Vec<usize> = core_groups
                .values()
                .filter_map(|cpus| cpus.iter().min().copied())
                .collect();

            let n = physical_ids.len().max(1);
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .start_handler(move |idx| {
                    if idx < physical_ids.len() {
                        core_affinity::set_for_current(core_affinity::CoreId {
                            id: physical_ids[idx],
                        });
                    }
                })
                .build_global();
        }
    }
}

#[derive(Parser)]
#[command(name = "flyby-crucible")]
#[command(about = "N-dimensional Chingon-Vlasov flyby anomaly universality test")]
struct Cli {
    /// Run only a specific spacecraft (galileo, near, cassini, rosetta, messenger, juno).
    /// If omitted, runs all spacecraft.
    #[arg(long)]
    spacecraft: Option<String>,

    /// Path to JPL .bsp ephemeris file (DE440 or DE430).
    /// Enables three-body Moon/Sun gravitational correction.
    #[arg(long, default_value = "data/external/de440.bsp")]
    bsp: String,

    /// Disable three-body correction (single-body Earth-only gravity).
    #[arg(long)]
    no_threebody: bool,

    /// Output h(t).v_wind trace CSV per spacecraft (diagnostic for Rosetta-I sign crossing).
    #[arg(long)]
    trace_h: bool,

    /// Integration timestep in seconds.
    #[arg(long, default_value = "1.0")]
    dt: f64,

    /// Override time before perigee (seconds). If omitted, uses SOI-based window.
    #[arg(long)]
    t_before: Option<f64>,

    /// Override time after perigee (seconds). If omitted, uses SOI-based window.
    #[arg(long)]
    t_after: Option<f64>,

    /// Override the gravitational focusing wake amplitude (ETA_WAKE).
    /// Default: 0.10. Set to 0.0 to disable wake density modulation.
    #[arg(long)]
    eta_wake: Option<f64>,

    /// Use GPU (CUDA) for AVT tensor contraction.
    /// The RK4 loop and orbital mechanics stay on CPU in f64;
    /// only the N-D bilinear contraction runs on GPU in f32.
    #[arg(long)]
    gpu: bool,

    /// Algebra dimension (must be power of 2: 64, 128, 256).
    /// Higher dimensions provide more geometric bandwidth for the
    /// non-associative tensor contraction at the cost of more violations
    /// and GPU memory.
    ///   64D:  52K violations, 0.2 MB SSBO  (production baseline)
    ///   128D: 469K violations, 1.9 MB SSBO (127 prime: asymmetric 3-body)
    ///   256D: 3.97M violations, 15.9 MB SSBO (symmetric 3-body: 85/85/85)
    #[arg(long, default_value = "64")]
    dim: usize,

    /// Violation cap for AVT construction.
    /// At 256D: 3.97M violations (uncapped). Set lower to trade accuracy
    /// for construction speed at higher dimensions.
    #[arg(long, default_value = "10000000")]
    avt_cap: usize,

    /// Output trajectory CSV file prefix.
    #[arg(long)]
    csv_prefix: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Pin to physical cores for V-Cache locality
    pin_physical_cores();

    let v_wind = dm_wind_j2000(&V_WIND_GALACTIC);

    // Load three-body ephemeris (Moon + Sun positions from JPL DE440)
    let ephem: Option<EphemerisLoader> = if cli.no_threebody {
        println!("Three-body correction DISABLED (--no-threebody)");
        None
    } else {
        let bsp_path = std::path::Path::new(&cli.bsp);
        match EphemerisLoader::load(bsp_path) {
            Ok(loader) => {
                println!("Three-body correction ENABLED (JPL DE440)");
                Some(loader)
            }
            Err(e) => {
                println!("Three-body correction DISABLED: {}", e);
                None
            }
        }
    };

    println!("=== 64D Chingon-Vlasov Flyby Crucible ===");
    println!("  alpha_chingon = {:.2e} (LOCKED)", ALPHA_CHINGON);
    println!(
        "  v_wind (Galactic) = ({:.1}, {:.1}, {:.1}) km/s",
        V_WIND_GALACTIC[0], V_WIND_GALACTIC[1], V_WIND_GALACTIC[2]
    );
    println!(
        "  v_wind (J2000)    = ({:.2}, {:.2}, {:.2}) km/s  |v|={:.1}",
        v_wind[0],
        v_wind[1],
        v_wind[2],
        v3_norm(v_wind)
    );
    println!("  dt = {:.2} s", cli.dt);
    println!(
        "  SOI = {} R_earth = {:.0} km",
        SOI_R_EARTH,
        SOI_R_EARTH * R_EARTH
    );
    println!();

    // Validate dimension
    let dim = cli.dim;
    assert!(
        dim.is_power_of_two() && dim >= 16,
        "dim must be a power of 2 >= 16, got {}",
        dim
    );
    let (block_size, n_phases, _b1_start, _b2_start, _b3_start, block3_size) =
        gr_core::forces::chingon_bivector_drag::block_layout(dim);
    println!(
        "  dim = {}D, blocks = {}/{}/{}, phases = {}/{}",
        dim,
        block_size,
        block_size,
        block3_size,
        n_phases,
        block3_size.div_ceil(3)
    );

    // Compute AVT once (expensive at high dims)
    println!(
        "Computing {}D Alternativity Violation Tensor (cap={})...",
        dim, cli.avt_cap
    );
    let t0 = std::time::Instant::now();
    let avt = Arc::new(AlternativityViolationTensor::new_with_cap(dim, cli.avt_cap));
    println!(
        "  AVT: {} violations ({:.2}s, {:.1} MB)",
        avt.violations.len(),
        t0.elapsed().as_secs_f64(),
        avt.violations.len() as f64 * 40.0 / 1e6
    );
    println!();

    // Initialize GPU pipeline if requested
    let gpu_pipeline: Option<Arc<Mutex<GpuPipeline>>> = if cli.gpu {
        #[cfg(feature = "gpu")]
        {
            let packed = avt.pack_for_gpu();
            println!("Initializing CUDA pipeline...");
            println!(
                "  Packed AVT: {} violations, {} bits/index, {} bytes",
                packed.violation_count,
                packed.index_bits,
                packed.data.len() * 4
            );
            let t_gpu = std::time::Instant::now();
            match ChingonGpuPipeline::new(&packed) {
                Ok(pipeline) => {
                    println!(
                        "  GPU pipeline ready ({:.2}s)",
                        t_gpu.elapsed().as_secs_f64()
                    );
                    Some(Arc::new(Mutex::new(pipeline)))
                }
                Err(e) => {
                    println!("  GPU init failed: {} -- falling back to CPU", e);
                    None
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            println!("WARNING: --gpu requested but binary compiled without 'gpu' feature");
            None
        }
    } else {
        None
    };

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

    // Print orbital plane diagnostics: h_hat and h.v_wind for each spacecraft
    println!("--- Orbital plane diagnostics ---");
    println!(
        "{:>30} {:>8} {:>8} {:>12} {:>6} {:>6}",
        "Spacecraft", "h.vw_s", "turn_d", "h_dec", "obs_s", "pred_s"
    );
    for cfg in &configs {
        let inb = radec_to_unit(cfg.inbound_ra_deg, cfg.inbound_dec_deg);
        let outb = radec_to_unit(cfg.outbound_ra_deg, cfg.outbound_dec_deg);
        let inb_neg = v3_scale(inb, -1.0);
        let h_orb = v3_cross(inb_neg, outb);
        let h_n = v3_norm(h_orb);
        let h_hat = if h_n > 1e-10 {
            v3_scale(h_orb, 1.0 / h_n)
        } else {
            [0.0; 3]
        };
        let h_dot_vw = v3_dot(h_hat, v_wind);
        let turn_deg = v3_dot(inb_neg, outb).acos().to_degrees();
        let h_dec = h_hat[2].asin().to_degrees();
        let pred_s = if h_dot_vw > 0.0 { "+" } else { "-" };
        let obs_s = if cfg.observed_dv_mm_s > 0.01 {
            "+"
        } else if cfg.observed_dv_mm_s < -0.01 {
            "-"
        } else {
            "~0"
        };
        println!(
            "{:>30} {:>8.2} {:>8.1} {:>12.1} {:>6} {:>6}",
            cfg.name, h_dot_vw, turn_deg, h_dec, obs_s, pred_s
        );
    }
    println!();

    println!(
        "{:>30} {:>10} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "Spacecraft", "Obs (mm/s)", "Pred (mm/s)", "Ratio", "Perigee(km)", "v_inf", "Window(s)"
    );
    println!("{}", "-".repeat(108));

    // Run all flyby simulations in parallel
    let eta_wake = cli.eta_wake.unwrap_or(ETA_WAKE);
    println!("  eta_wake = {:.3}", eta_wake);

    let t1 = std::time::Instant::now();
    let results: Vec<_> = configs
        .par_iter()
        .map(|cfg| {
            let t_before = cli.t_before.unwrap_or_else(|| compute_soi_window(cfg));
            let t_after = cli.t_after.unwrap_or_else(|| compute_soi_window(cfg));
            let total_steps = ((t_before + t_after) / cli.dt) as usize;
            let stride = (total_steps / 500).max(1);

            let (v_ctrl, v_chingon, traj, h_trace_data) = run_flyby(
                cfg,
                &avt,
                cli.dt,
                t_before,
                t_after,
                stride,
                &v_wind,
                ephem.as_ref(),
                cli.trace_h,
                eta_wake,
                gpu_pipeline.as_deref(),
            );

            let delta_v_mm_s = (v_chingon - v_ctrl) * 1e6;
            let ratio = if cfg.observed_dv_mm_s.abs() > 0.001 {
                delta_v_mm_s / cfg.observed_dv_mm_s
            } else {
                f64::NAN
            };

            (
                *cfg,
                delta_v_mm_s,
                ratio,
                t_before + t_after,
                traj,
                h_trace_data,
            )
        })
        .collect();
    let sim_elapsed = t1.elapsed().as_secs_f64();

    for (cfg, delta_v_mm_s, ratio, window, traj, h_trace_data) in &results {
        println!(
            "{:>30} {:>10.2} {:>12.4e} {:>12.4} {:>12.0} {:>10.3} {:>10.0}",
            cfg.name,
            cfg.observed_dv_mm_s,
            delta_v_mm_s,
            ratio,
            cfg.perigee_alt_km,
            cfg.v_inf,
            window
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
            wtr.write_record([
                "t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s",
            ])?;
            for row in traj {
                wtr.write_record(row.iter().map(|v| format!("{:.6}", v)))?;
            }
            wtr.flush()?;
            eprintln!("  Wrote {}", path);
        }

        // Write h(t).v_wind trace CSV if --trace-h was given
        if cli.trace_h && !h_trace_data.is_empty() {
            let safe_name: String = cfg
                .name
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect();
            let path = format!("h_trace_{}.csv", safe_name);
            let mut wtr = csv::Writer::from_path(&path)?;
            wtr.write_record([
                "t_s",
                "h_x",
                "h_y",
                "h_z",
                "h_dot_vwind",
                "cross_sign",
                "a_chingon_mag",
                "dm_factor",
                "a_moon_mag",
                "a_sun_mag",
                "h_lunar_norm",
                "h_solar_norm",
            ])?;
            for row in h_trace_data {
                wtr.write_record(row.iter().map(|v| format!("{:.8e}", v)))?;
            }
            wtr.flush()?;
            eprintln!("  Wrote {}", path);
        }
    }

    println!();
    println!(
        "=== Universality Verdict ({:.2}s, {} physical cores) ===",
        sim_elapsed,
        rayon::current_num_threads()
    );
    println!("  If pred/obs ratio is consistent across all spacecraft,");
    println!("  the coupling constant is universal.");
    println!("  Tolerance: |pred/obs - 1| < 0.25 for each flyby.");

    // Future Capability: Sprint 73 -- Vulkan RK4 port
    // For 256D AVT contraction: 256^3 = ~16.7M ops/timestep * 4 RK4 stages
    // = ~67M ops/step * 150k steps = ~10T FLOPs per flyby. CPU freezes.
    // Port to GLSL compute shader: rk4_chingon.comp
    //   SSBO: OnceLock 256D AVT violations tensor (read-only, loaded once)
    //   UBO: JPL ephemeris data (Earth, Moon, Sun at time T, updated per step)
    //   Push Constants: alpha_eff, dm_density_factor for current step
    // Pipeline: ash 0.38.0 (already in workspace), mirror lbm_vulkan/src/compute.rs
    // RTX 4070 Ti: 48 SMs, 7680 CUDA cores, 12 GB VRAM, 504 GB/s bandwidth
    //   256D AVT violations fit in ~10 MB SSBO (trivial). Compute-bound.

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gororoba_cli_physics::flyby::geometry::galactic_to_j2000;

    #[test]
    fn test_hyperbolic_initial_state_sanity() {
        let cfg = &all_flybys()[0]; // Galileo
        let t_window = compute_soi_window(cfg);
        let (pos_arr, vel_arr) = hyperbolic_initial_state(cfg, t_window);

        let r = v3_norm(pos_arr);
        let r_perigee = R_EARTH + cfg.perigee_alt_km;
        assert!(
            r > r_perigee,
            "At SOI boundary, r={:.0} should be > r_perigee={:.0}",
            r,
            r_perigee
        );

        let v = v3_norm(vel_arr);
        assert!(
            v > cfg.v_inf * 0.8 && v < cfg.v_inf * 3.0,
            "Speed {:.3} km/s unreasonable for v_inf={:.3}",
            v,
            cfg.v_inf
        );
    }

    #[test]
    fn test_soi_window_reasonable() {
        for cfg in &all_flybys() {
            let t = compute_soi_window(cfg);
            // Window should be between 1 hour and 2 days
            assert!(
                t > 3600.0 && t < 172800.0,
                "{}: SOI window {:.0}s out of range",
                cfg.name,
                t
            );
        }
    }

    #[test]
    fn test_galactic_to_j2000_orthogonal() {
        let r = galactic_to_j2000();
        // R^T * R must be the identity matrix (rotation matrix is orthogonal).
        // For column-major layout m[row][col]: (R^T R)[i][j] = sum_k R[k][i] * R[k][j].
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| r[k][i] * r[k][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "R^T R[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_dm_wind_magnitude_preserved() {
        let v_gal_n = v3_norm(V_WIND_GALACTIC);
        let v_j2000_arr = dm_wind_j2000(&V_WIND_GALACTIC);
        let v_j2000_n = v3_norm(v_j2000_arr);
        assert!(
            (v_gal_n - v_j2000_n).abs() < 1e-6,
            "Rotation should preserve magnitude: |v_gal|={:.3}, |v_j2000|={:.3}",
            v_gal_n,
            v_j2000_n
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
                let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-12,
                    "Unit vector norm = {} at RA={}, Dec={}",
                    norm,
                    ra,
                    dec
                );
            }
        }
    }

    #[test]
    fn test_alpha_locked() {
        // With 1/r^3 NFW density scaling, alpha is recalibrated to 6e-12
        // (vs 8e-14 for uniform density) to match NEAR's +13.46 mm/s.
        assert_eq!(ALPHA_CHINGON, 6.0e-12);
    }
}
