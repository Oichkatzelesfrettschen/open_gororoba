//! Coordinate transforms and orbital geometry for flyby analysis.
//!
//! Extracted from flyby_crucible.rs: Galactic-to-J2000, RA/Dec conversions,
//! hyperbolic orbit initial state, SOI window computation.

use nalgebra::{Matrix3, Vector3};

use super::config::{FlybyConfig, GM_EARTH, R_EARTH, SOI_R_EARTH};

/// Galactic-to-J2000 ECI rotation matrix.
///
/// IAU definition (Hipparcos-based, Murray 1989 / Liu+ 2011):
///   Galactic North Pole (J2000): RA = 192.85948 deg, Dec = +27.12825 deg
///   Galactic Center   (J2000): RA = 266.40510 deg, Dec = -28.93617 deg
pub fn galactic_to_j2000() -> Matrix3<f64> {
    let ra_ngp = 192.85948_f64.to_radians();
    let dec_ngp = 27.12825_f64.to_radians();
    let ra_gc = 266.40510_f64.to_radians();
    let dec_gc = (-28.93617_f64).to_radians();

    let z_gal = Vector3::new(
        dec_ngp.cos() * ra_ngp.cos(),
        dec_ngp.cos() * ra_ngp.sin(),
        dec_ngp.sin(),
    );
    let x_gal_raw = Vector3::new(
        dec_gc.cos() * ra_gc.cos(),
        dec_gc.cos() * ra_gc.sin(),
        dec_gc.sin(),
    );
    let x_gal = (x_gal_raw - z_gal * z_gal.dot(&x_gal_raw)).normalize();
    let y_gal = z_gal.cross(&x_gal);
    Matrix3::from_columns(&[x_gal, y_gal, z_gal])
}

/// Compute the DM wind vector in J2000 ECI coordinates (km/s).
pub fn dm_wind_j2000(v_wind_galactic: &[f64; 3]) -> Vector3<f64> {
    let v_gal = Vector3::new(v_wind_galactic[0], v_wind_galactic[1], v_wind_galactic[2]);
    galactic_to_j2000() * v_gal
}

/// Convert asymptotic direction (RA, Dec in degrees) to a unit vector.
pub fn radec_to_unit(ra_deg: f64, dec_deg: f64) -> Vector3<f64> {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
}

/// Compute a physically motivated integration window (seconds before/after perigee).
///
/// The window spans the time for the spacecraft to travel from SOI_R_EARTH * R_EARTH
/// to perigee and back out, with 10% margin.
pub fn compute_soi_window(cfg: &FlybyConfig) -> f64 {
    let r_soi = SOI_R_EARTH * R_EARTH;
    let r_perigee = R_EARTH + cfg.perigee_alt_km;

    let a = -GM_EARTH / (cfg.v_inf * cfg.v_inf);
    let e = 1.0 - r_perigee / a;

    let cos_nu_soi = ((a * (1.0 - e * e) / r_soi) - 1.0) / e;
    let cos_nu_soi = cos_nu_soi.clamp(-1.0, 1.0);
    let nu_soi = cos_nu_soi.acos();

    let tan_half_nu = (nu_soi / 2.0).tan();
    let tan_half_h = tan_half_nu / ((e + 1.0) / (e - 1.0)).sqrt();
    let h_soi = 2.0 * tan_half_h.atanh();

    let m_soi = e * h_soi.sinh() - h_soi;
    let n = (-GM_EARTH / (a * a * a)).sqrt();
    let t_soi = m_soi.abs() / n;

    t_soi * 1.1
}

/// Compute initial position and velocity at T seconds before perigee
/// for a hyperbolic flyby in the orbital plane.
pub fn hyperbolic_initial_state(
    cfg: &FlybyConfig,
    t_before_perigee: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let r_perigee = R_EARTH + cfg.perigee_alt_km;
    let v_inf = cfg.v_inf;

    let a = -GM_EARTH / (v_inf * v_inf);
    let e = 1.0 - r_perigee / a;
    let p = a * (1.0 - e * e);

    let n = (-GM_EARTH / (a * a * a)).sqrt();
    let m_target = -n * t_before_perigee;

    // Solve Kepler's equation for hyperbolic anomaly H via Newton-Raphson.
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
        h -= dh.clamp(-2.0, 2.0);
        if dh.abs() < 1e-12 {
            break;
        }
    }

    let cos_nu = (e - h.cosh()) / (1.0 - e * h.cosh());
    let sin_nu = (e * e - 1.0).sqrt() * h.sinh() / (1.0 - e * h.cosh());
    let nu = sin_nu.atan2(cos_nu);

    let r = p / (1.0 + e * nu.cos());
    let x_pf = r * nu.cos();
    let y_pf = r * nu.sin();

    let h_ang = (GM_EARTH * p).sqrt();
    let vx_pf = -GM_EARTH / h_ang * nu.sin();
    let vy_pf = GM_EARTH / h_ang * (e + nu.cos());

    // Rotate perifocal frame to ECI using inbound/outbound asymptotic directions.
    let inbound_dir = radec_to_unit(cfg.inbound_ra_deg, cfg.inbound_dec_deg);
    let outbound_dir = radec_to_unit(cfg.outbound_ra_deg, cfg.outbound_dec_deg);

    let bisector = -inbound_dir + outbound_dir;
    let x_hat = if bisector.norm() > 1e-10 {
        bisector.normalize()
    } else {
        (-inbound_dir).normalize()
    };

    let h_orbital = (-inbound_dir).cross(&outbound_dir);
    let h_norm_orb = h_orbital.norm();

    let y_hat = if h_norm_orb > 1e-10 {
        let h_hat = h_orbital / h_norm_orb;
        h_hat.cross(&x_hat).normalize()
    } else {
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

/// Altitude-dependent DM density enhancement factor.
///
/// Models Earth's gravitational focusing as a power-law: rho(r) = (R_earth / r)^3.
/// Normalized so density_factor(R_earth) = 1.0.
/// See BIB-0303 (Lundberg & Edsjo 2004).
pub fn dm_density_factor(r_km: f64) -> f64 {
    if r_km <= R_EARTH {
        return 1.0;
    }
    (R_EARTH / r_km).powi(3)
}

/// DM density with gravitational focusing wake along the wind axis.
///
/// Combines the 1/r^3 NFW radial profile with the signed cos(theta) wake
/// enhancement: downstream (cos > 0) gets enhancement, upstream (cos < 0)
/// gets depletion.
pub fn dm_wake_density_factor(
    r_km: f64,
    r_pos: &Vector3<f64>,
    v_wind: &Vector3<f64>,
    eta_wake: f64,
) -> f64 {
    let base = dm_density_factor(r_km);
    if eta_wake == 0.0 {
        return base;
    }
    let r_n = r_pos.norm();
    let v_n = v_wind.norm();
    if r_n < 1.0 || v_n < 1.0 {
        return base;
    }
    let cos_wind = r_pos.dot(v_wind) / (r_n * v_n);
    base * (1.0 + eta_wake * cos_wind)
}
