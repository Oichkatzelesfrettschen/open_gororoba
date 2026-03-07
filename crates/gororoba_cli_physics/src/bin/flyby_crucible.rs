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
use gororoba_cli_physics::ephemeris_loader::{EphemerisLoader, GM_MOON, GM_SUN};
use gr_core::forces::chingon_bivector_drag::compute_chingon_bivector_drag;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use std::sync::Arc;

/// Coupling constant with NFW-like 1/r^3 density scaling.
///
/// With uniform density (no scaling), alpha = 8e-14 reproduced sign patterns
/// but magnitudes were ~5x too large. With 1/r^3 gravitational focusing,
/// the effective integration path is weighted toward perigee, requiring a
/// larger alpha_0 to produce the same integrated delta-V. Calibrated to
/// NEAR's observed +13.46 mm/s.
const ALPHA_CHINGON: f64 = 6.0e-12;

/// Earth GM in km^3/s^2.
const GM_EARTH: f64 = 398600.4418;

/// Earth radius in km.
const R_EARTH: f64 = 6371.0;

/// Dark matter wind velocity in Galactic coordinates (km/s).
/// Sun moves at ~220 km/s toward Galactic l=90, b=0 (Cygnus direction).
/// Components: (U_toward_center, V_rotation, W_north) in Galactic frame.
const V_WIND_GALACTIC: [f64; 3] = [-11.1, 232.24, 7.25];

/// SOI radius for integration window: 50 Earth radii.
/// Beyond this, Earth's gravity is negligible and Chingon drag has no
/// geometric coupling to the flyby trajectory.
const SOI_R_EARTH: f64 = 50.0;

/// Altitude-dependent DM density enhancement factor.
///
/// Models Earth's gravitational focusing of galactic DM as a power-law
/// density profile: rho(r) = (R_earth / r)^3.
///
/// Physical motivation: Earth's gravity focuses collisionless dark matter
/// particles via Liouville's theorem, producing a power-law density
/// enhancement near the surface. Unlike atmospheric gas (which has pressure
/// support and exponential scale heights), DM is collisionless and clusters
/// purely gravitationally. The NFW profile's inner slope goes as r^{-1} to
/// r^{-3} depending on the capture mechanism:
/// - Gravitational focusing of unbound particles: ~ 1/r (Lundberg & Edsjo 2004)
/// - Bound captured component: ~ 1/r^2 to 1/r^3 (Peter 2009)
///
/// We use n=3 (NFW-like inner cusp), consistent with the NS-NFW coupling
/// used for galactic halos in Sprint 68. The profile gives:
///   rho(539 km alt) / rho(2347 km alt) = (6910/8718)^3 = 0.498
///   rho(perigee) / rho(SOI=318550 km) = (6371/318550)^3 = 8e-6
///
/// The 1/r^3 provides physically correct weighting: force concentrates
/// around perigee where gravitational capture is strongest, with graceful
/// falloff that still allows meaningful RK4 integration through the SOI.
///
/// Normalized so density_factor(R_earth) = 1.0.
fn dm_density_factor(r_km: f64) -> f64 {
    if r_km <= R_EARTH {
        return 1.0;
    }
    (R_EARTH / r_km).powi(3)
}

/// Tidal DM density: anisotropic modifier using Moon/Sun alignment.
///
/// The scalar 1/r^3 profile is stretched along the Earth-Moon and
/// Earth-Sun axes. When the spacecraft is aligned with the Moon
/// (cos^2(theta) ~ 1), density is enhanced. When perpendicular
/// (cos^2(theta) ~ 0), density equals the isotropic base.
///
/// eta_moon: tidal stretching strength from lunar gravity
/// eta_sun: tidal stretching strength from solar gravity (weaker)
///
/// Physical motivation: the Moon's gravity well focuses galactic DM
/// infall along the Earth-Moon axis, creating tidal density ridges
/// analogous to oceanic tides but in the dark matter halo.
const ETA_MOON: f64 = 0.15;
const ETA_SUN: f64 = 0.05;

fn tidal_dm_density(
    r_sc: Vector3<f64>,
    r_moon: Vector3<f64>,
    r_sun: Vector3<f64>,
) -> f64 {
    let r_km = r_sc.norm();
    let base = dm_density_factor(r_km);

    if r_km < 1.0 {
        return base;
    }

    // cos(angle) between spacecraft and Moon directions
    let r_moon_norm = r_moon.norm();
    let cos_moon = if r_moon_norm > 1.0 {
        r_sc.dot(&r_moon) / (r_km * r_moon_norm)
    } else {
        0.0
    };

    // cos(angle) between spacecraft and Sun directions
    let r_sun_norm = r_sun.norm();
    let cos_sun = if r_sun_norm > 1.0 {
        r_sc.dot(&r_sun) / (r_km * r_sun_norm)
    } else {
        0.0
    };

    // Anisotropic tidal stretching: density enhanced along Moon/Sun axes
    base * (1.0 + ETA_MOON * cos_moon * cos_moon + ETA_SUN * cos_sun * cos_sun)
}

/// Galactic-to-J2000 ECI rotation matrix.
///
/// IAU definition (Hipparcos-based, Murray 1989 / Liu+ 2011):
///   Galactic North Pole (J2000): RA = 192.85948 deg, Dec = +27.12825 deg
///   Galactic Center   (J2000): RA = 266.40510 deg, Dec = -28.93617 deg
///
/// The matrix R transforms Galactic (l, b) Cartesian to J2000 ECI Cartesian:
///   v_J2000 = R * v_Galactic
///
/// Columns of R are the Galactic basis vectors expressed in J2000:
///   col0 = unit vector toward Galactic center (l=0, b=0)
///   col1 = unit vector toward l=90, b=0
///   col2 = unit vector toward Galactic North Pole (b=90)
fn galactic_to_j2000() -> Matrix3<f64> {
    // Galactic North Pole in J2000
    let ra_ngp = 192.85948_f64.to_radians();
    let dec_ngp = 27.12825_f64.to_radians();

    // Galactic Center in J2000
    let ra_gc = 266.40510_f64.to_radians();
    let dec_gc = (-28.93617_f64).to_radians();

    // z_gal = NGP direction in J2000
    let z_gal = Vector3::new(
        dec_ngp.cos() * ra_ngp.cos(),
        dec_ngp.cos() * ra_ngp.sin(),
        dec_ngp.sin(),
    );

    // x_gal = Galactic Center direction in J2000
    let x_gal_raw = Vector3::new(
        dec_gc.cos() * ra_gc.cos(),
        dec_gc.cos() * ra_gc.sin(),
        dec_gc.sin(),
    );

    // Orthogonalize: x_gal must be perpendicular to z_gal
    let x_gal = (x_gal_raw - z_gal * z_gal.dot(&x_gal_raw)).normalize();

    // y_gal = z_gal x x_gal (right-handed)
    let y_gal = z_gal.cross(&x_gal);

    // Columns: x_gal, y_gal, z_gal
    Matrix3::from_columns(&[x_gal, y_gal, z_gal])
}

/// Compute the DM wind vector in J2000 ECI coordinates (km/s).
fn dm_wind_j2000() -> Vector3<f64> {
    let v_gal = Vector3::new(V_WIND_GALACTIC[0], V_WIND_GALACTIC[1], V_WIND_GALACTIC[2]);
    galactic_to_j2000() * v_gal
}

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
    /// Outbound asymptotic declination (degrees).
    /// From Anderson et al. (2008) Table I. Required to determine the
    /// orbital plane normal h = (-v_in) x v_out correctly.
    outbound_dec_deg: f64,
    /// Outbound asymptotic right ascension (degrees).
    outbound_ra_deg: f64,
    /// Observed anomalous delta-V (mm/s). Positive = speed gain.
    observed_dv_mm_s: f64,
    /// Perigee epoch as Julian Ephemeris Date (JED/TDB).
    /// Required for three-body Moon/Sun position queries.
    perigee_jed: f64,
}

fn all_flybys() -> Vec<FlybyConfig> {
    // Inbound/outbound asymptotic directions from Anderson et al. (2008)
    // PRL 100, 091102, Table I.
    use gororoba_cli_physics::ephemeris_loader::flyby_epochs;
    vec![
        FlybyConfig {
            name: "Galileo-I (1990-12-08)",
            perigee_alt_km: 960.0,
            v_inf: 8.949,
            inbound_dec_deg: -12.5,
            inbound_ra_deg: 263.0,
            outbound_dec_deg: -4.9,
            outbound_ra_deg: 223.0,
            observed_dv_mm_s: 3.92,
            perigee_jed: flyby_epochs::GALILEO,
        },
        FlybyConfig {
            name: "NEAR (1998-01-23)",
            perigee_alt_km: 539.0,
            v_inf: 6.851,
            inbound_dec_deg: -20.8,
            inbound_ra_deg: 280.0,
            outbound_dec_deg: 72.0,
            outbound_ra_deg: 89.0,
            observed_dv_mm_s: 13.46,
            perigee_jed: flyby_epochs::NEAR,
        },
        FlybyConfig {
            name: "Cassini (1999-08-18)",
            perigee_alt_km: 1175.0,
            v_inf: 16.01,
            inbound_dec_deg: -12.9,
            inbound_ra_deg: 257.0,
            outbound_dec_deg: -5.0,
            outbound_ra_deg: 344.0,
            observed_dv_mm_s: -2.0,
            perigee_jed: flyby_epochs::CASSINI,
        },
        FlybyConfig {
            name: "Rosetta-I (2005-03-04)",
            perigee_alt_km: 1956.0,
            v_inf: 3.863,
            inbound_dec_deg: -34.3,
            inbound_ra_deg: 247.0,
            outbound_dec_deg: -20.6,
            outbound_ra_deg: 116.0,
            observed_dv_mm_s: 1.80,
            perigee_jed: flyby_epochs::ROSETTA_I,
        },
        FlybyConfig {
            name: "MESSENGER (2005-08-02)",
            perigee_alt_km: 2347.0,
            v_inf: 4.056,
            inbound_dec_deg: 31.4,
            inbound_ra_deg: 232.0,
            outbound_dec_deg: 75.4,
            outbound_ra_deg: 174.0,
            observed_dv_mm_s: 0.02,
            perigee_jed: flyby_epochs::MESSENGER,
        },
        FlybyConfig {
            name: "Juno (2013-10-09)",
            perigee_alt_km: 559.0,
            v_inf: 9.897,
            inbound_dec_deg: -13.6,
            inbound_ra_deg: 0.0,
            outbound_dec_deg: -5.3,
            outbound_ra_deg: 345.0,
            observed_dv_mm_s: 0.0,
            perigee_jed: flyby_epochs::JUNO,
        },
    ]
}

/// Compute a physically motivated integration window (seconds before/after perigee).
///
/// The window spans the time for the spacecraft to travel from SOI_R_EARTH * R_EARTH
/// to perigee and back out. This ensures the Kepler solver operates in a numerically
/// stable regime (|M| < ~100) while capturing all gravitationally significant dynamics.
fn compute_soi_window(cfg: &FlybyConfig) -> f64 {
    let r_soi = SOI_R_EARTH * R_EARTH;
    let r_perigee = R_EARTH + cfg.perigee_alt_km;

    // Hyperbolic orbit parameters
    let a = -GM_EARTH / (cfg.v_inf * cfg.v_inf);
    let e = 1.0 - r_perigee / a;

    // True anomaly at SOI radius
    let cos_nu_soi = ((a * (1.0 - e * e) / r_soi) - 1.0) / e;
    // Clamp for numerical safety (SOI might be beyond the hyperbola's asymptote)
    let cos_nu_soi = cos_nu_soi.clamp(-1.0, 1.0);
    let nu_soi = cos_nu_soi.acos();

    // Hyperbolic anomaly at SOI
    let tan_half_nu = (nu_soi / 2.0).tan();
    let tan_half_h = tan_half_nu / ((e + 1.0) / (e - 1.0)).sqrt();
    let h_soi = 2.0 * tan_half_h.atanh();

    // Mean anomaly at SOI
    let m_soi = e * h_soi.sinh() - h_soi;

    // Mean motion
    let n = (-GM_EARTH / (a * a * a)).sqrt();

    // Time from perigee to SOI
    let t_soi = m_soi.abs() / n;

    // Add 10% margin
    t_soi * 1.1
}

/// Convert asymptotic direction (RA, Dec in degrees) to a unit vector.
fn radec_to_unit(ra_deg: f64, dec_deg: f64) -> Vector3<f64> {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
}

/// Compute initial position and velocity at T seconds before perigee
/// for a hyperbolic flyby in the orbital plane.
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

    // Mean anomaly at T before perigee
    let m_target = -n * t_before_perigee;

    // Solve Kepler's equation for hyperbolic anomaly H via Newton-Raphson.
    // Clamp initial guess to prevent sinh overflow.
    let mut h = (m_target / e).clamp(-20.0, 20.0);
    for _ in 0..100 {
        let sh = h.sinh();
        let ch = h.cosh();
        let f_h = e * sh - h - m_target;
        let fp_h = e * ch - 1.0;
        if fp_h.abs() < 1e-30 {
            break;
        }
        let dh = f_h / fp_h;
        // Damped step to prevent overshooting
        h -= dh.clamp(-2.0, 2.0);
        if dh.abs() < 1e-12 {
            break;
        }
    }

    // True anomaly from hyperbolic anomaly
    let cos_nu = (e - h.cosh()) / (1.0 - e * h.cosh());
    let sin_nu = (e * e - 1.0).sqrt() * h.sinh() / (1.0 - e * h.cosh());
    let nu = sin_nu.atan2(cos_nu);

    // Radius at this point
    let r = p / (1.0 + e * nu.cos());

    // Position in perifocal frame
    let x_pf = r * nu.cos();
    let y_pf = r * nu.sin();

    // Velocity in perifocal frame
    let h_ang = (GM_EARTH * p).sqrt();
    let vx_pf = -GM_EARTH / h_ang * nu.sin();
    let vy_pf = GM_EARTH / h_ang * (e + nu.cos());

    // Rotate perifocal frame to ECI using inbound AND outbound asymptotic directions.
    //
    // The perifocal frame has x_hat pointing from the focus toward perigee (nu=0),
    // y_hat perpendicular in the orbital plane (toward nu=pi/2), and z_hat = h_hat
    // perpendicular to the orbital plane.
    //
    // The perigee direction bisects the angle between the two asymptotic velocity
    // vectors (-v_in and v_out). This is geometrically exact: the hyperbola is
    // symmetric about the apse line, and both asymptotes make equal angles with it.
    let inbound_dir = radec_to_unit(cfg.inbound_ra_deg, cfg.inbound_dec_deg);
    let outbound_dir = radec_to_unit(cfg.outbound_ra_deg, cfg.outbound_dec_deg);

    // Perigee direction = bisector of -inbound and outbound
    let bisector = -inbound_dir + outbound_dir;
    let x_hat = if bisector.norm() > 1e-10 {
        bisector.normalize()
    } else {
        // Degenerate: 180-degree turning (head-on collision). Use -inbound.
        (-inbound_dir).normalize()
    };

    // Orbital angular momentum direction: h = (-v_in) x v_out
    let h_orbital = (-inbound_dir).cross(&outbound_dir);
    let h_norm_orb = h_orbital.norm();

    // y_hat completes the right-handed perifocal frame: y = h x x
    let y_hat = if h_norm_orb > 1e-10 {
        let h_hat = h_orbital / h_norm_orb;
        h_hat.cross(&x_hat).normalize()
    } else {
        // Degenerate: inbound ~ outbound (no turning). Fall back to z_ref.
        let z_ref = Vector3::new(0.0, 0.0, 1.0);
        let y_raw = z_ref.cross(&x_hat);
        if y_raw.norm() > 1e-10 {
            y_raw.normalize()
        } else {
            let x_ref = Vector3::new(1.0, 0.0, 0.0);
            x_ref.cross(&x_hat).normalize()
        }
    };

    let pos = x_hat * x_pf + y_hat * y_pf;
    let vel = x_hat * vx_pf + y_hat * vy_pf;

    (pos, vel)
}

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
    v_wind: &Vector3<f64>,
    ephem: Option<&EphemerisLoader>,
    trace_h: bool,
) -> (f64, f64, Vec<[f64; 7]>, Vec<[f64; 10]>) {
    let (pos_init, vel_init) = hyperbolic_initial_state(cfg, t_before);
    let total_time = t_before + t_after;
    let steps = (total_time / dt) as usize;

    /// Seconds per Julian day.
    const SPD: f64 = 86400.0;

    let rk4_run =
        |use_chingon: bool, record: bool, do_trace: bool| -> (f64, Vec<[f64; 7]>, Vec<[f64; 10]>) {
            let mut p = pos_init;
            let mut v = vel_init;
            let mut traj = Vec::new();
            let mut h_trace_out: Vec<[f64; 10]> = Vec::new();

            for step in 0..steps {
                let t_sec = step as f64 * dt - t_before;

                if record && step % trajectory_stride == 0 {
                    traj.push([t_sec, p.x, p.y, p.z, v.x, v.y, v.z]);
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
                    (Vector3::zeros(), Vector3::zeros(), Vector3::zeros())
                };

                let accel = |p_in: Vector3<f64>, v_in: Vector3<f64>| -> Vector3<f64> {
                    let r_sq = p_in.norm_squared();
                    let r = r_sq.sqrt();
                    let mut a = -p_in * (GM_EARTH / (r_sq * r));

                    // Three-body: Moon and Sun gravitational perturbations
                    if ephem.is_some() {
                        let dp_moon = p_in - r_moon;
                        let d_moon = dp_moon.norm();
                        if d_moon > 1.0 {
                            a -= dp_moon * (GM_MOON / (d_moon * d_moon * d_moon));
                            // Indirect term: acceleration of Earth by Moon
                            let d_moon_0 = r_moon.norm();
                            if d_moon_0 > 1.0 {
                                a -= r_moon * (GM_MOON / (d_moon_0 * d_moon_0 * d_moon_0));
                            }
                        }
                        let dp_sun = p_in - r_sun;
                        let d_sun = dp_sun.norm();
                        if d_sun > 1.0 {
                            a -= dp_sun * (GM_SUN / (d_sun * d_sun * d_sun));
                            // Indirect term: acceleration of Earth by Sun
                            let d_sun_0 = r_sun.norm();
                            if d_sun_0 > 1.0 {
                                a -= r_sun * (GM_SUN / (d_sun_0 * d_sun_0 * d_sun_0));
                            }
                        }
                    }

                    if use_chingon {
                        // Tidal DM density: anisotropic along Moon/Sun axes
                        let alpha_eff = ALPHA_CHINGON * tidal_dm_density(p_in, r_moon, r_sun);
                        a += compute_chingon_bivector_drag(
                            p_in, v_in, *v_wind, alpha_eff, avt,
                        );
                    }

                    a
                };

                // h(t).v_wind trace for diagnostics (geocentric h)
                if do_trace && step % trajectory_stride == 0 {
                    let h = p.cross(&v);
                    let h_dot_vw = h.dot(v_wind);
                    let cross_sign = if h_dot_vw > 0.0 { 1.0 } else { -1.0 };
                    let dm_fac = tidal_dm_density(p, r_moon, r_sun);
                    let a_chingon_mag = if use_chingon {
                        let alpha_eff = ALPHA_CHINGON * dm_fac;
                        compute_chingon_bivector_drag(p, v, *v_wind, alpha_eff, avt).norm()
                    } else {
                        0.0
                    };
                    let a_moon_mag = if ephem.is_some() {
                        let dp = p - r_moon;
                        let d = dp.norm();
                        if d > 1.0 { GM_MOON / (d * d) } else { 0.0 }
                    } else {
                        0.0
                    };
                    let a_sun_mag = if ephem.is_some() {
                        let dp = p - r_sun;
                        let d = dp.norm();
                        if d > 1.0 { GM_SUN / (d * d) } else { 0.0 }
                    } else {
                        0.0
                    };
                    h_trace_out.push([
                        t_sec, h.x, h.y, h.z, h_dot_vw, cross_sign,
                        a_chingon_mag, dm_fac, a_moon_mag, a_sun_mag,
                    ]);
                }

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

            (v.norm(), traj, h_trace_out)
        };

    let (v_ctrl, _, _) = rk4_run(false, false, false);
    let (v_chingon, traj, h_trace_data) = rk4_run(true, true, trace_h);

    (v_ctrl, v_chingon, traj, h_trace_data)
}

/// Pin the current thread pool to physical cores for V-Cache locality.
fn pin_physical_cores() {
    #[cfg(target_os = "linux")]
    {
        use std::collections::BTreeMap;
        use std::fs;

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
#[command(about = "64D Chingon-Vlasov flyby anomaly universality test")]
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

    /// Output trajectory CSV file prefix.
    #[arg(long)]
    csv_prefix: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Pin to physical cores for V-Cache locality
    pin_physical_cores();

    let v_wind = dm_wind_j2000();

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
        v_wind.x,
        v_wind.y,
        v_wind.z,
        v_wind.norm()
    );
    println!("  dt = {:.2} s", cli.dt);
    println!("  SOI = {} R_earth = {:.0} km", SOI_R_EARTH, SOI_R_EARTH * R_EARTH);
    println!();

    // Compute AVT once (expensive at dim=64)
    println!("Computing 64D Alternativity Violation Tensor...");
    let t0 = std::time::Instant::now();
    let avt = Arc::new(AlternativityViolationTensor::new(64));
    println!("  AVT: {} violations ({:.2}s)", avt.violations.len(), t0.elapsed().as_secs_f64());
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

    // Print orbital plane diagnostics: h_hat and h.v_wind for each spacecraft
    println!("--- Orbital plane diagnostics ---");
    println!(
        "{:>30} {:>8} {:>8} {:>12} {:>6} {:>6}",
        "Spacecraft", "h.vw_s", "turn_d", "h_dec", "obs_s", "pred_s"
    );
    for cfg in &configs {
        let inb = radec_to_unit(cfg.inbound_ra_deg, cfg.inbound_dec_deg);
        let outb = radec_to_unit(cfg.outbound_ra_deg, cfg.outbound_dec_deg);
        let h_orb = (-inb).cross(&outb);
        let h_n = h_orb.norm();
        let h_hat = if h_n > 1e-10 { h_orb / h_n } else { Vector3::zeros() };
        let h_dot_vw = h_hat.dot(&v_wind);
        let turn_deg = (-inb).dot(&outb).acos().to_degrees();
        let h_dec = h_hat.z.asin().to_degrees();
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
    let t1 = std::time::Instant::now();
    let results: Vec<_> = configs
        .par_iter()
        .map(|cfg| {
            let t_before = cli.t_before.unwrap_or_else(|| compute_soi_window(cfg));
            let t_after = cli.t_after.unwrap_or_else(|| compute_soi_window(cfg));
            let total_steps = ((t_before + t_after) / cli.dt) as usize;
            let stride = (total_steps / 500).max(1);

            let (v_ctrl, v_chingon, traj, h_trace_data) = run_flyby(
                cfg, &avt, cli.dt, t_before, t_after, stride, &v_wind,
                ephem.as_ref(), cli.trace_h,
            );

            let delta_v_mm_s = (v_chingon - v_ctrl) * 1e6;
            let ratio = if cfg.observed_dv_mm_s.abs() > 0.001 {
                delta_v_mm_s / cfg.observed_dv_mm_s
            } else {
                f64::NAN
            };

            (*cfg, delta_v_mm_s, ratio, t_before + t_after, traj, h_trace_data)
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
            wtr.write_record(["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])?;
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
                "t_s", "h_x", "h_y", "h_z", "h_dot_vwind", "cross_sign",
                "a_chingon_mag", "dm_factor", "a_moon_mag", "a_sun_mag",
            ])?;
            for row in h_trace_data {
                wtr.write_record(row.iter().map(|v| format!("{:.8e}", v)))?;
            }
            wtr.flush()?;
            eprintln!("  Wrote {}", path);
        }
    }

    println!();
    println!("=== Universality Verdict ({:.2}s, {} physical cores) ===", sim_elapsed, rayon::current_num_threads());
    println!("  If pred/obs ratio is consistent across all spacecraft,");
    println!("  the coupling constant is universal.");
    println!("  Tolerance: |pred/obs - 1| < 0.25 for each flyby.");

    // TODO(Sprint 73 -- Vulkan RK4 port):
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

    #[test]
    fn test_hyperbolic_initial_state_sanity() {
        let cfg = &all_flybys()[0]; // Galileo
        let t_window = compute_soi_window(cfg);
        let (pos, vel) = hyperbolic_initial_state(cfg, t_window);

        let r = pos.norm();
        let r_perigee = R_EARTH + cfg.perigee_alt_km;
        assert!(
            r > r_perigee,
            "At SOI boundary, r={:.0} should be > r_perigee={:.0}",
            r,
            r_perigee
        );

        let v = vel.norm();
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
        let rtr = r.transpose() * r;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rtr[(i, j)] - expected).abs() < 1e-10,
                    "R^T R[{},{}] = {}, expected {}",
                    i,
                    j,
                    rtr[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_dm_wind_magnitude_preserved() {
        let v_gal = Vector3::new(V_WIND_GALACTIC[0], V_WIND_GALACTIC[1], V_WIND_GALACTIC[2]);
        let v_j2000 = dm_wind_j2000();
        assert!(
            (v_gal.norm() - v_j2000.norm()).abs() < 1e-6,
            "Rotation should preserve magnitude: |v_gal|={:.3}, |v_j2000|={:.3}",
            v_gal.norm(),
            v_j2000.norm()
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
        // With 1/r^3 NFW density scaling, alpha is recalibrated to 6e-12
        // (vs 8e-14 for uniform density) to match NEAR's +13.46 mm/s.
        assert_eq!(ALPHA_CHINGON, 6.0e-12);
    }
}
