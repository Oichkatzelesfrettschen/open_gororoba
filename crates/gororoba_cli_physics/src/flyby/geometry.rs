//! Coordinate transforms and orbital geometry for flyby analysis.
//!
//! Extracted from flyby_crucible.rs: Galactic-to-J2000, RA/Dec conversions,
//! hyperbolic orbit initial state, SOI window computation.

use super::config::{FlybyConfig, GM_EARTH, R_EARTH, SOI_R_EARTH};

/// Galactic-to-J2000 ECI rotation matrix.
///
/// IAU definition (Hipparcos-based, Murray 1989 / Liu+ 2011):
///   Galactic North Pole (J2000): RA = 192.85948 deg, Dec = +27.12825 deg
///   Galactic Center   (J2000): RA = 266.40510 deg, Dec = -28.93617 deg
///
/// Returns a row-major 3x3 matrix; column k is the k-th galactic basis vector
/// in J2000 ECI.  Multiply with `mat3_mul_vec3` to transform galactic -> ECI.
pub fn galactic_to_j2000() -> [[f64; 3]; 3] {
    let ra_ngp = 192.85948_f64.to_radians();
    let dec_ngp = 27.12825_f64.to_radians();
    let ra_gc = 266.40510_f64.to_radians();
    let dec_gc = (-28.93617_f64).to_radians();

    let z_gal = [
        dec_ngp.cos() * ra_ngp.cos(),
        dec_ngp.cos() * ra_ngp.sin(),
        dec_ngp.sin(),
    ];
    let x_gal_raw = [
        dec_gc.cos() * ra_gc.cos(),
        dec_gc.cos() * ra_gc.sin(),
        dec_gc.sin(),
    ];
    // Gram-Schmidt: project x_gal_raw perpendicular to z_gal, then normalize.
    let proj = dot3(z_gal, x_gal_raw);
    let x_gal = normalize3(sub3(x_gal_raw, scale3(z_gal, proj)));
    let y_gal = cross3(z_gal, x_gal);
    mat3_from_columns(x_gal, y_gal, z_gal)
}

/// Compute the DM wind vector in J2000 ECI coordinates (km/s).
pub fn dm_wind_j2000(v_wind_galactic: &[f64; 3]) -> [f64; 3] {
    mat3_mul_vec3(galactic_to_j2000(), *v_wind_galactic)
}

/// Convert asymptotic direction (RA, Dec in degrees) to a unit vector.
pub fn radec_to_unit(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    [dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]
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
) -> ([f64; 3], [f64; 3]) {
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

    let bisector = add3(neg3(inbound_dir), outbound_dir);
    let x_hat = if norm3(bisector) > 1e-10 {
        normalize3(bisector)
    } else {
        normalize3(neg3(inbound_dir))
    };

    let h_orbital = cross3(neg3(inbound_dir), outbound_dir);
    let h_norm_orb = norm3(h_orbital);

    let y_hat = if h_norm_orb > 1e-10 {
        let h_hat = scale3(h_orbital, 1.0 / h_norm_orb);
        normalize3(cross3(h_hat, x_hat))
    } else {
        let y_raw = cross3([0.0, 0.0, 1.0], x_hat);
        if norm3(y_raw) > 1e-10 {
            normalize3(y_raw)
        } else {
            normalize3(cross3([1.0, 0.0, 0.0], x_hat))
        }
    };

    let pos = add3(scale3(x_hat, x_pf), scale3(y_hat, y_pf));
    let vel = add3(scale3(x_hat, vx_pf), scale3(y_hat, vy_pf));

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
    r_pos: &[f64; 3],
    v_wind: &[f64; 3],
    eta_wake: f64,
) -> f64 {
    let base = dm_density_factor(r_km);
    if eta_wake == 0.0 {
        return base;
    }
    let r_n = norm3(*r_pos);
    let v_n = norm3(*v_wind);
    if r_n < 1.0 || v_n < 1.0 {
        return base;
    }
    let cos_wind = dot3(*r_pos, *v_wind) / (r_n * v_n);
    base * (1.0 + eta_wake * cos_wind)
}

// --- Private vec3 / mat3 helpers ---

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let n = norm3(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn neg3(a: [f64; 3]) -> [f64; 3] {
    [-a[0], -a[1], -a[2]]
}

/// Build a column-major rotation matrix from three orthonormal column vectors.
///
/// `m[row][col]` layout: the k-th column is the k-th argument.  Matches
/// `nalgebra::Matrix3::from_columns` semantics; `mat3_mul_vec3(m, v)` gives `m * v`.
#[inline]
fn mat3_from_columns(x: [f64; 3], y: [f64; 3], z: [f64; 3]) -> [[f64; 3]; 3] {
    [
        [x[0], y[0], z[0]],
        [x[1], y[1], z[1]],
        [x[2], y[2], z[2]],
    ]
}

/// Matrix-vector product for a 3x3 row-major matrix and a 3-vector.
#[inline]
fn mat3_mul_vec3(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
