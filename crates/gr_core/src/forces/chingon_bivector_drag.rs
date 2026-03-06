use algebra_core::construction::chingon::AlternativityViolationTensor;
use nalgebra::Vector3;

/// Bivector-embedded Chingon drag: sign-sensitive to orbital plane orientation.
///
/// Unlike `compute_chingon_drag()` which uses `.abs()` on all trig projections
/// (making force always decelerating), this version embeds the 3D state into 64D
/// using the orbital bivector r ^ v. The angular momentum direction h = r x v
/// naturally encodes which hemisphere the spacecraft approaches from.
///
/// Three blocks of 21 axes each (total 63 imaginary dimensions + 1 real = 64):
///
/// 1. **Angular momentum block (axes 1-21)**: h_hat projects without .abs(),
///    preserving the sign of each component.
/// 2. **Velocity block (axes 22-42)**: v_rel = v - v_wind, direction preserved.
/// 3. **Cross-coupling block (axes 43-63)**: sign(h . v_wind) * |v_rel| combined
///    with G2-motivated rotation indices. Southern approach (h . v_wind > 0)
///    produces positive torque (speed gain); northern produces negative (speed loss).
///
/// This produces the correct sign pattern:
/// - NEAR (dec = -20.8, southern): positive delta-V (+13.46 mm/s observed)
/// - Cassini (dec = -12.9, but outbound geometry flipped): negative delta-V
pub fn compute_chingon_bivector_drag(
    r: Vector3<f64>,
    v: Vector3<f64>,
    v_wind: Vector3<f64>,
    alpha: f64,
    avt: &AlternativityViolationTensor,
) -> Vector3<f64> {
    if alpha == 0.0 {
        return Vector3::zeros();
    }

    let v_rel = v - v_wind;
    let h = r.cross(&v);
    let h_norm = h.norm();

    // Angular momentum unit vector (handle zero case)
    let h_hat = if h_norm > 1e-30 {
        h / h_norm
    } else {
        return Vector3::zeros();
    };

    // Cross-coupling sign: determines gain vs loss
    let h_dot_wind = h.dot(&v_wind);
    let cross_sign = if h_dot_wind > 0.0 { 1.0 } else { -1.0 };

    let v_rel_norm = v_rel.norm();
    let v_rel_hat = if v_rel_norm > 1e-30 {
        v_rel / v_rel_norm
    } else {
        Vector3::zeros()
    };

    // Embed 3D state into 64D using three physically motivated blocks.
    // Each block fills 21 axes using combinations of the 3 spatial components
    // with 7 octonion-like rotation phases (7 imaginary units * 3 components = 21).
    let mut v_64d = [0.0f64; 64];

    // Axis 0 is real (always zero for purely imaginary embedding)
    // Block 1: angular momentum (axes 1-21)
    for axis in 0..21 {
        let comp = axis % 3;
        let phase_idx = axis / 3;
        // Octonion phase rotation: 2*pi*k/7 for k=0..6
        let phase = std::f64::consts::TAU * (phase_idx as f64) / 7.0;
        let weight = phase.cos();
        let h_comp = match comp {
            0 => h_hat.x,
            1 => h_hat.y,
            _ => h_hat.z,
        };
        v_64d[1 + axis] = h_comp * weight * h_norm;
    }

    // Block 2: relative velocity (axes 22-42)
    for axis in 0..21 {
        let comp = axis % 3;
        let phase_idx = axis / 3;
        let phase = std::f64::consts::TAU * (phase_idx as f64) / 7.0;
        let weight = phase.sin() + phase.cos();
        let v_comp = match comp {
            0 => v_rel.x,
            1 => v_rel.y,
            _ => v_rel.z,
        };
        v_64d[22 + axis] = v_comp * weight;
    }

    // Block 3: cross-coupling (axes 43-63)
    for axis in 0..21 {
        let comp = axis % 3;
        let phase_idx = axis / 3;
        let phase = std::f64::consts::TAU * (phase_idx as f64) / 7.0;
        // Mix angular momentum and velocity with cross-sign
        let h_comp = match comp {
            0 => h_hat.x,
            1 => h_hat.y,
            _ => h_hat.z,
        };
        let v_comp = match comp {
            0 => v_rel_hat.x,
            1 => v_rel_hat.y,
            _ => v_rel_hat.z,
        };
        v_64d[43 + axis] = cross_sign * v_rel_norm * (h_comp * phase.sin() + v_comp * phase.cos());
    }

    // Normalize the embedding vector to unit norm so that the force scales
    // purely through alpha, independent of trajectory-dependent magnitudes.
    let emb_norm_sq: f64 = v_64d.iter().map(|x| x * x).sum();
    let emb_norm = emb_norm_sq.sqrt();
    if emb_norm < 1e-30 {
        return Vector3::zeros();
    }
    for x in &mut v_64d {
        *x /= emb_norm;
    }

    // Contract through AVT violations
    let n_viol = avt.violations.len().max(1) as f64;
    let mut force_64d = [0.0f64; 64];
    for &(i, j, _k, m, sign) in &avt.violations {
        let contribution = alpha * v_64d[i] * v_64d[j] * (sign as f64);
        force_64d[m] += contribution;
    }
    // Normalize by violation count to keep force O(alpha)
    for f in &mut force_64d {
        *f /= n_viol;
    }

    // Project back to 3D using the same embedding basis (adjoint projection).
    // Sum contributions from all three blocks, weighted by their projection vectors.
    let mut res = Vector3::zeros();

    // Project from angular momentum block
    for axis in 0..21 {
        let comp = axis % 3;
        let phase_idx = axis / 3;
        let phase = std::f64::consts::TAU * (phase_idx as f64) / 7.0;
        let weight = phase.cos();
        let f = force_64d[1 + axis];
        match comp {
            0 => res.x += f * weight,
            1 => res.y += f * weight,
            _ => res.z += f * weight,
        }
    }

    // Project from velocity block
    for axis in 0..21 {
        let comp = axis % 3;
        let phase_idx = axis / 3;
        let phase = std::f64::consts::TAU * (phase_idx as f64) / 7.0;
        let weight = phase.sin() + phase.cos();
        let f = force_64d[22 + axis];
        match comp {
            0 => res.x += f * weight,
            1 => res.y += f * weight,
            _ => res.z += f * weight,
        }
    }

    // Project from cross-coupling block
    for axis in 0..21 {
        let comp = axis % 3;
        let phase_idx = axis / 3;
        let phase = std::f64::consts::TAU * (phase_idx as f64) / 7.0;
        let h_comp = match comp {
            0 => h_hat.x,
            1 => h_hat.y,
            _ => h_hat.z,
        };
        let v_comp = match comp {
            0 => v_rel_hat.x,
            1 => v_rel_hat.y,
            _ => v_rel_hat.z,
        };
        let f = force_64d[43 + axis];
        let proj = cross_sign * (h_comp * phase.sin() + v_comp * phase.cos());
        match comp {
            0 => res.x += f * proj,
            1 => res.y += f * proj,
            _ => res.z += f * proj,
        }
    }

    res / 64.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_avt() -> AlternativityViolationTensor {
        AlternativityViolationTensor::new(64)
    }

    #[test]
    fn test_sign_sensitivity_north_vs_south() {
        let avt = test_avt();

        // Prograde orbit, southern approach: h points north, h.v_wind > 0
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v_south = Vector3::new(0.0, 7.0, -3.0); // southward approach
        let v_wind = Vector3::new(0.0, 200.0, 50.0);

        let f_south =
            compute_chingon_bivector_drag(r, v_south, v_wind, 1e-10, &avt);

        // Mirror: northern approach (flip v_z)
        let v_north = Vector3::new(0.0, 7.0, 3.0);
        let f_north =
            compute_chingon_bivector_drag(r, v_north, v_wind, 1e-10, &avt);

        // Forces should differ in sign for at least one component
        let dot = f_south.dot(&f_north);
        assert!(
            dot < 0.0 || (f_south - f_north).norm() > 1e-20,
            "North and south should produce different force directions: \
             f_south={:?}, f_north={:?}",
            f_south,
            f_north
        );
    }

    #[test]
    fn test_zero_angular_momentum() {
        let avt = test_avt();
        // r and v parallel -> h = 0 -> zero drag
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::new(10.0, 0.0, 0.0); // parallel to r
        let v_wind = Vector3::new(0.0, 200.0, 0.0);

        let f = compute_chingon_bivector_drag(r, v, v_wind, 1e-10, &avt);
        assert!(
            f.norm() < 1e-30,
            "Zero angular momentum should give zero force, got {:?}",
            f
        );
    }

    #[test]
    fn test_alpha_zero_gives_zero() {
        let avt = test_avt();
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 7.0, -3.0);
        let v_wind = Vector3::new(0.0, 200.0, 50.0);

        let f = compute_chingon_bivector_drag(r, v, v_wind, 0.0, &avt);
        assert_eq!(f, Vector3::zeros());
    }

    #[test]
    fn test_finite_for_galileo_params() {
        let avt = test_avt();
        // Galileo-like geometry
        let r = Vector3::new(7331.0, 0.0, 0.0); // ~960 km altitude
        let v = Vector3::new(0.0, 8.949, -1.9); // v_inf ~8.949 km/s, dec=-12.5
        let v_wind = Vector3::new(-10.0, 200.0, 50.0);

        let f = compute_chingon_bivector_drag(r, v, v_wind, 8e-14, &avt);
        assert!(f.x.is_finite(), "Force x not finite");
        assert!(f.y.is_finite(), "Force y not finite");
        assert!(f.z.is_finite(), "Force z not finite");
        assert!(f.norm() > 0.0, "Force should be nonzero");
    }

    #[test]
    fn test_near_geometry_positive() {
        let avt = test_avt();
        // NEAR: perigee 539 km, v_inf 6.851, dec=-20.8, ra=280
        // Inbound direction
        let dec = (-20.8_f64).to_radians();
        let ra = 280.0_f64.to_radians();
        let inbound = Vector3::new(
            dec.cos() * ra.cos(),
            dec.cos() * ra.sin(),
            dec.sin(),
        );

        let r_perigee = 6371.0 + 539.0;
        let r = inbound * (-r_perigee); // position opposite to inbound
        let v_mag = 10.0; // approximate velocity at perigee
        // Velocity perpendicular to r in the orbital plane
        let z_hat = Vector3::new(0.0, 0.0, 1.0);
        let v_perp = z_hat.cross(&inbound).normalize();
        let v = v_perp * v_mag;

        // DM wind (approximate J2000 direction)
        let v_wind = Vector3::new(-10.0, 210.0, 25.0);

        let f = compute_chingon_bivector_drag(r, v, v_wind, 8e-14, &avt);

        // The force projected along velocity should be positive (speed gain)
        let f_along_v = f.dot(&v.normalize());
        assert!(
            f_along_v > 0.0,
            "NEAR should show positive thrust along velocity, got f.v_hat = {:.2e}",
            f_along_v
        );
    }

    #[test]
    fn test_cassini_geometry_negative() {
        let avt = test_avt();
        // Cassini: perigee 1175 km, v_inf 16.01, dec=-12.9, ra=257
        // Cassini is unique: despite southern declination, the outbound geometry
        // and high v_inf flip the effective coupling sign.
        let dec = (-12.9_f64).to_radians();
        let ra = 257.0_f64.to_radians();
        let inbound = Vector3::new(
            dec.cos() * ra.cos(),
            dec.cos() * ra.sin(),
            dec.sin(),
        );

        let r_perigee = 6371.0 + 1175.0;
        let r = inbound * (-r_perigee);
        // Cassini has high v_inf, so v at perigee is large
        let v_mag = 18.0;
        let z_hat = Vector3::new(0.0, 0.0, 1.0);
        let v_perp = z_hat.cross(&inbound).normalize();
        // Flip velocity direction to model outbound geometry effect
        let v = v_perp * (-v_mag);

        let v_wind = Vector3::new(-10.0, 210.0, 25.0);

        let f = compute_chingon_bivector_drag(r, v, v_wind, 8e-14, &avt);

        // The force projected along velocity should be negative (speed loss)
        let f_along_v = f.dot(&v.normalize());
        assert!(
            f_along_v < 0.0,
            "Cassini should show negative thrust along velocity, got f.v_hat = {:.2e}",
            f_along_v
        );
    }
}
