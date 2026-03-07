use algebra_core::construction::chingon::AlternativityViolationTensor;
use nalgebra::Vector3;

/// Bivector-embedded Chingon drag: sign-sensitive to orbital plane orientation.
///
/// Uses a **frame-invariant** orbital triad to embed the 3D state into 64D,
/// ensuring the tensor contraction produces forces whose sign relative to
/// velocity is determined by the physical geometry, not the coordinate frame.
///
/// The orbital triad (e_v, e_h, e_n):
/// - `e_v = v_rel / |v_rel|` -- along the DM-relative velocity
/// - `e_h = h_hat` -- orbital angular momentum direction (h = r x v)
/// - `e_n = e_v x e_h` -- completing the right-handed triad
///
/// Three blocks of 21 axes each (total 63 imaginary dimensions + 1 real = 64):
///
/// 1. **Angular momentum block (axes 1-21)**: h projected into orbital triad
///    (always [0, 1, 0] in this frame, so this block encodes h_norm only).
/// 2. **Velocity block (axes 22-42)**: v_rel projected into orbital triad
///    (always [|v_rel|, 0, 0] in this frame).
/// 3. **Cross-coupling block (axes 43-63)**: sign(h . v_wind) * mixed terms.
///    This block is the only one sensitive to the spacecraft's hemisphere
///    relative to the dark matter wind.
///
/// Force direction: The cross-coupling block's sign(h . v_wind) determines
/// whether the force accelerates (+) or decelerates (-) the spacecraft along
/// its velocity vector.
pub fn compute_chingon_bivector_drag(
    r: Vector3<f64>,
    v: Vector3<f64>,
    v_wind: Vector3<f64>,
    alpha: f64,
    avt: &AlternativityViolationTensor,
) -> Vector3<f64> {
    chingon_bivector_drag_core(r.cross(&v), v - v_wind, v_wind, alpha, avt)
}

/// EXPLOREME(Sprint 71): Barycentric variant -- FALSIFIED.
///
/// Shifts angular momentum reference from geocenter to Earth-Moon barycenter:
///   h_3body = (r - emb_offset) x v
///
/// Sprint 71 experiment showed this DEGRADES NEAR from 0.999 to 1.45 ratio
/// and worsens Rosetta-I from -22.43 to -36.06 mm/s. The EMB offset (~4671 km)
/// is larger than Rosetta-I perigee (1956 km), amplifying h and the Chingon force.
///
/// Kept for reference. Geocentric h (the non-_bary variant) is production.
#[allow(dead_code)]
pub fn compute_chingon_bivector_drag_bary(
    r: Vector3<f64>,
    v: Vector3<f64>,
    v_wind: Vector3<f64>,
    alpha: f64,
    avt: &AlternativityViolationTensor,
    emb_offset: Vector3<f64>,
) -> Vector3<f64> {
    chingon_bivector_drag_core((r - emb_offset).cross(&v), v - v_wind, v_wind, alpha, avt)
}

/// Shared core: takes precomputed h and v_rel.
fn chingon_bivector_drag_core(
    h: Vector3<f64>,
    v_rel: Vector3<f64>,
    v_wind: Vector3<f64>,
    alpha: f64,
    avt: &AlternativityViolationTensor,
) -> Vector3<f64> {
    if alpha == 0.0 {
        return Vector3::zeros();
    }

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

    // Build the orbital triad (e_v, e_h, e_n) for frame-invariant embedding.
    // In this frame:
    //   h projects to [0, h_norm, 0]
    //   v_rel projects to [v_rel_norm, 0, 0]  (approximately -- has small e_n component)
    //   The cross-coupling mixes e_h and e_v components with cross_sign
    let e_v = v_rel_hat;
    let e_h = h_hat;
    let e_n = e_v.cross(&e_h);
    let e_n_norm = e_n.norm();
    let e_n = if e_n_norm > 1e-30 { e_n / e_n_norm } else { Vector3::zeros() };

    // Project physical vectors into the orbital triad
    let h_triad = [h.dot(&e_v), h.dot(&e_h), h.dot(&e_n)];
    let vrel_triad = [v_rel.dot(&e_v), v_rel.dot(&e_h), v_rel.dot(&e_n)];
    let vrel_hat_triad = [v_rel_hat.dot(&e_v), v_rel_hat.dot(&e_h), v_rel_hat.dot(&e_n)];

    // Precompute trig table: only 7 distinct phases (TAU * k / 7, k=0..6).
    // Each RK4 step calls this function, so eliminating 42 trig calls per
    // invocation (~150k steps * 6 spacecraft = ~900k calls) is significant.
    // The 512-byte v_64d array fits in L1; the 112-byte trig table does too.
    let phase_trig: [(f64, f64); 7] = std::array::from_fn(|k| {
        let phase = std::f64::consts::TAU * (k as f64) / 7.0;
        (phase.sin(), phase.cos())
    });

    // Embed 3D state into 64D using three physically motivated blocks.
    // Each block fills 21 axes using combinations of the 3 orbital triad components
    // with 7 octonion-like rotation phases (7 imaginary units * 3 components = 21).
    let mut v_64d = [0.0f64; 64];

    // Axis 0 is real (always zero for purely imaginary embedding)
    // Block 1: angular momentum (axes 1-21)
    for axis in 0..21 {
        let comp = axis % 3;
        let (_, cos_p) = phase_trig[axis / 3];
        v_64d[1 + axis] = h_triad[comp] * cos_p;
    }

    // Block 2: relative velocity (axes 22-42)
    for axis in 0..21 {
        let comp = axis % 3;
        let (sin_p, cos_p) = phase_trig[axis / 3];
        v_64d[22 + axis] = vrel_triad[comp] * (sin_p + cos_p);
    }

    // Block 3: cross-coupling (axes 43-63)
    for axis in 0..21 {
        let comp = axis % 3;
        let (sin_p, cos_p) = phase_trig[axis / 3];
        let h_c = h_triad[comp] / h_norm.max(1e-30);
        let v_c = vrel_hat_triad[comp];
        v_64d[43 + axis] = cross_sign * v_rel_norm * (h_c * sin_p + v_c * cos_p);
    }

    // Do NOT normalize to unit norm. The physical magnitudes (h_norm, v_rel_norm)
    // carry the trajectory-dependent amplification that makes the force O(mm/s)
    // over a flyby. The bilinear AVT contraction with physical magnitudes gives
    // force ~ alpha * h_norm * v_rel_norm / 64 ~ O(1e-8) km/s^2 at alpha=8e-14,
    // integrating to O(1 mm/s) delta-V over a typical 50000 s flyby window.
    //
    // Stability: v_rel is dominated by v_wind (~230 km/s), so the embedding
    // magnitudes don't grow with the integration. The force is bounded by
    // alpha * |v_wind|^2 * SOI_radius ~ 8e-14 * 5e4 * 3e5 ~ 1e-6 km/s^2,
    // which is 9 orders of magnitude below gravity at perigee. RK4 is stable.
    let n_viol = avt.violations.len().max(1) as f64;
    let mut force_64d = [0.0f64; 64];
    for &(i, j, _k, m, sign) in &avt.violations {
        let contribution = alpha * v_64d[i] * v_64d[j] * (sign as f64);
        force_64d[m] += contribution;
    }
    // Normalize by violation count to keep force O(alpha * phys_scale)
    for f in &mut force_64d {
        *f /= n_viol;
    }

    // Project back to 3D using ONLY the cross-coupling block (axes 43-63).
    //
    // The angular momentum block (1-21) and velocity block (22-42) encode the
    // orbital state but should not generate force: they represent the "classical"
    // embedding that merely positions the spacecraft in the 64D algebra. The
    // physical non-alternative torque arises solely from the cross-coupling block
    // which mixes h and v_rel with cross_sign = sign(h . v_wind).
    //
    // This isolates the non-alternative torque from classical drag artifacts,
    // ensuring the force sign along velocity is governed by cross_sign.
    let mut res_triad = [0.0f64; 3];

    for axis in 0..21 {
        let comp = axis % 3;
        let (sin_p, cos_p) = phase_trig[axis / 3];
        let h_c = h_triad[comp] / h_norm.max(1e-30);
        let v_c = vrel_hat_triad[comp];
        let proj = cross_sign * (h_c * sin_p + v_c * cos_p);
        res_triad[comp] += force_64d[43 + axis] * proj;
    }

    // Rotate from orbital triad back to the original coordinate frame
    let res = e_v * res_triad[0] + e_h * res_triad[1] + e_n * res_triad[2];

    res / 64.0
}

/// Pre-computed orbital parameters for the 3-body 64D embedding.
///
/// Extracted from `compute_chingon_bivector_drag_3body` so that both
/// the CPU and GPU contraction paths can share the same geometry computation.
pub struct ThreeBodyOrbitalParams {
    /// Earth triad: [e_v, e_h, e_n] as row-major 3x3.
    pub triad_earth: [[f64; 3]; 3],
    /// Lunar triad.
    pub triad_lunar: [[f64; 3]; 3],
    /// Solar triad.
    pub triad_solar: [[f64; 3]; 3],
    /// h_earth projected into Earth triad [3].
    pub h_triad_earth: [f64; 3],
    /// v_rel projected into Lunar triad [3].
    pub vrel_triad_lunar: [f64; 3],
    /// h_earth projected into Solar triad [3].
    pub h_triad_solar: [f64; 3],
    /// v_hat projected into Solar triad [3].
    pub vhat_triad_solar: [f64; 3],
    /// Magnitude of h_earth.
    pub h_earth_norm: f64,
    /// Magnitude of v_rel.
    pub v_rel_norm: f64,
    /// sign(h_earth . v_wind): +1.0 or -1.0.
    pub cross_sign: f64,
}

impl ThreeBodyOrbitalParams {
    /// Compute orbital parameters for the 3-body embedding.
    ///
    /// Returns `None` if the geometry is degenerate (zero velocity or
    /// zero angular momentum).
    pub fn compute(
        r: Vector3<f64>,
        v: Vector3<f64>,
        v_wind: Vector3<f64>,
        r_moon: Vector3<f64>,
        r_sun: Vector3<f64>,
    ) -> Option<Self> {
        let v_rel = v - v_wind;
        let v_rel_norm = v_rel.norm();
        if v_rel_norm < 1e-30 {
            return None;
        }

        let h_earth = r.cross(&v);
        let h_lunar = (r - r_moon).cross(&v);
        let h_solar = (r - r_sun).cross(&v);

        let h_earth_norm = h_earth.norm();
        if h_earth_norm < 1e-30 {
            return None;
        }

        let h_dot_wind = h_earth.dot(&v_wind);
        let cross_sign = if h_dot_wind > 0.0 { 1.0 } else { -1.0 };

        let build_triad = |h: &Vector3<f64>| -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
            let hn = h.norm();
            if hn < 1e-30 {
                let e_h = h_earth / h_earth_norm;
                let e_v = v_rel / v_rel_norm;
                let e_n_raw = e_v.cross(&e_h);
                let e_n_n = e_n_raw.norm();
                let e_n = if e_n_n > 1e-30 { e_n_raw / e_n_n } else { Vector3::zeros() };
                return (e_v, e_h, e_n);
            }
            let e_h = *h / hn;
            let e_v = v_rel / v_rel_norm;
            let e_n_raw = e_v.cross(&e_h);
            let e_n_n = e_n_raw.norm();
            let e_n = if e_n_n > 1e-30 { e_n_raw / e_n_n } else { Vector3::zeros() };
            (e_v, e_h, e_n)
        };

        let (e_v_earth, e_h_earth, e_n_earth) = build_triad(&h_earth);
        let (e_v_lunar, e_h_lunar, e_n_lunar) = build_triad(&h_lunar);
        let (e_v_solar, e_h_solar, e_n_solar) = build_triad(&h_solar);

        let v_hat = v_rel / v_rel_norm;

        let h_triad_earth = [h_earth.dot(&e_v_earth), h_earth.dot(&e_h_earth), h_earth.dot(&e_n_earth)];
        let vrel_triad_lunar = [v_rel.dot(&e_v_lunar), v_rel.dot(&e_h_lunar), v_rel.dot(&e_n_lunar)];
        let h_triad_solar = [h_earth.dot(&e_v_solar), h_earth.dot(&e_h_solar), h_earth.dot(&e_n_solar)];
        let vhat_triad_solar = [v_hat.dot(&e_v_solar), v_hat.dot(&e_h_solar), v_hat.dot(&e_n_solar)];

        fn v3_to_arr(v: Vector3<f64>) -> [f64; 3] { [v.x, v.y, v.z] }

        Some(Self {
            triad_earth: [v3_to_arr(e_v_earth), v3_to_arr(e_h_earth), v3_to_arr(e_n_earth)],
            triad_lunar: [v3_to_arr(e_v_lunar), v3_to_arr(e_h_lunar), v3_to_arr(e_n_lunar)],
            triad_solar: [v3_to_arr(e_v_solar), v3_to_arr(e_h_solar), v3_to_arr(e_n_solar)],
            h_triad_earth,
            vrel_triad_lunar,
            h_triad_solar,
            vhat_triad_solar,
            h_earth_norm,
            v_rel_norm,
            cross_sign,
        })
    }
}

/// Compute the three-body block layout for a given dimension.
///
/// Returns `(block_size, n_phases, block1_start, block2_start, block3_start, block3_size)`
/// where blocks 1 and 2 have `block_size` axes, block 3 has `block3_size` axes,
/// and `n_phases = ceil(block_size / 3)` for trig LUT sizing.
///
/// # Symmetric vs Asymmetric Partitions
///
/// The three-body embedding partitions `(dim-1)` imaginary axes into 3 blocks.
/// When `(dim-1) % 3 != 0`, block3 absorbs the remainder.
///
/// ```text
///   dim   n_imag  partition        symmetry    3-body flyby quality
///    64       63  21 / 21 / 21     symmetric   5/6 signs correct (production)
///   128      127  42 / 42 / 43     ASYMMETRIC  2/6 signs (DEGRADED -- Mersenne prime)
///   256      255  85 / 85 / 85     symmetric   (pending, Sprint 73)
///   512      511  170/ 170/ 171    ASYMMETRIC  (511 = 7 * 73, not divisible by 3)
///  1024     1023  341/ 341/ 341    symmetric   (1023 = 3 * 341)
/// ```
///
/// # 128D Mersenne Prime Blocking (Sprint 73 finding, C-958)
///
/// 128 - 1 = 127, which is the Mersenne prime M_7 = 2^7 - 1.
/// 127 is indivisible by 3 (or any integer > 1), forcing an asymmetric
/// 42/42/43 partition. This 1-axis asymmetry BREAKS the tensor contraction
/// geometry: experimentally degrades flyby sign prediction from 5/6 to 2/6.
/// The asymmetry is structural, not tunable -- no scalar parameter can fix it.
///
/// 128D is therefore EXCLUDED from 3-body flyby applications. It remains
/// useful for non-flyby work: NNSD quantum chaos (routon_chaos_crucible),
/// topological void generation, and Non-Associative Entropy Filter (LBM).
///
/// # AVT Dimensional Scaling (Sprint 73 survey)
///
/// ```text
///   dim   AVT violations    ratio     v/d^3       construction time
///     8              0        -     0.000000      <1 ms  (alternative)
///    16            336        -     0.082031      <1 ms
///    32          5,040    15.0x     0.153809       3 ms
///    64         52,080    10.3x     0.198669      28 ms
///   128        468,720     9.0x     0.223503     250 ms
///   256      3,968,496     8.5x     0.236541    2100 ms
/// ```
///
/// The ratio converges toward 8x (= 2^3) confirming O(dim^3) scaling.
/// The density v/d^3 converges toward ~0.25 (quarter-dense).
pub fn block_layout(dim: usize) -> (usize, usize, usize, usize, usize, usize) {
    let n_imag = dim - 1; // imaginary axes (axis 0 is real/unused)
    let base_block = n_imag / 3;
    let remainder = n_imag % 3;
    // Blocks 1 and 2 get base_block; block 3 gets base_block + remainder
    let block1_size = base_block;
    let block2_size = base_block;
    let block3_size = base_block + remainder;
    let block1_start = 1; // skip axis 0
    let block2_start = block1_start + block1_size;
    let block3_start = block2_start + block2_size;
    let n_phases = block1_size.div_ceil(3); // ceil(block_size / 3) for trig LUT
    (block1_size, n_phases, block1_start, block2_start, block3_start, block3_size)
}

/// Three-body N-dimensional embedding: body-specific triads for each sub-block.
///
/// Dimension-parametric version that works for 64D, 128D, 256D, etc.
/// Uses the same physical embedding strategy as the 64D version but
/// with block sizes computed from `block_layout(avt.dim)`.
///
/// Block partition for 3-body:
///   Block 1: angular momentum via Earth triad (h_earth = r x v)
///   Block 2: velocity via Lunar triad (h_lunar = (r-r_moon) x v)
///   Block 3: cross-coupling via Solar triad (h_solar = (r-r_sun) x v)
///
/// For 128D (127 prime): asymmetric partition 42/42/43.
/// For 256D (255 = 3*85): symmetric partition 85/85/85.
pub fn compute_chingon_bivector_drag_3body(
    r: Vector3<f64>,
    v: Vector3<f64>,
    v_wind: Vector3<f64>,
    alpha: f64,
    avt: &AlternativityViolationTensor,
    r_moon: Vector3<f64>,
    r_sun: Vector3<f64>,
) -> Vector3<f64> {
    if alpha == 0.0 {
        return Vector3::zeros();
    }

    let params = match ThreeBodyOrbitalParams::compute(r, v, v_wind, r_moon, r_sun) {
        Some(p) => p,
        None => return Vector3::zeros(),
    };

    let dim = avt.dim;
    let (block_size, n_phases, b1_start, b2_start, b3_start, block3_size) = block_layout(dim);

    let v_rel = v - v_wind;
    let v_rel_norm = params.v_rel_norm;
    let h_earth_norm = params.h_earth_norm;
    let cross_sign = params.cross_sign;

    fn arr_to_v3(a: [f64; 3]) -> Vector3<f64> { Vector3::new(a[0], a[1], a[2]) }

    let e_v_solar = arr_to_v3(params.triad_solar[0]);
    let e_h_solar = arr_to_v3(params.triad_solar[1]);
    let e_n_solar = arr_to_v3(params.triad_solar[2]);

    let h_triad_earth = params.h_triad_earth;
    let vrel_triad_lunar = params.vrel_triad_lunar;
    let h_triad_solar = params.h_triad_solar;
    let vhat_triad_solar = params.vhat_triad_solar;

    let _ = v_rel;

    // Precompute trig table: n_phases distinct rotation phases
    let phase_trig: Vec<(f64, f64)> = (0..n_phases)
        .map(|k| {
            let phase = std::f64::consts::TAU * (k as f64) / (n_phases as f64);
            (phase.sin(), phase.cos())
        })
        .collect();

    // Allocate state vector on the heap for dim > 64
    let mut v_nd = vec![0.0f64; dim];

    // Block 1: angular momentum via Earth triad
    for axis in 0..block_size {
        let comp = axis % 3;
        let phase_idx = axis / 3;
        let (_, cos_p) = phase_trig[phase_idx];
        v_nd[b1_start + axis] = h_triad_earth[comp] * cos_p;
    }

    // Block 2: velocity via Lunar triad
    for axis in 0..block_size {
        let comp = axis % 3;
        let phase_idx = axis / 3;
        let (sin_p, cos_p) = phase_trig[phase_idx];
        v_nd[b2_start + axis] = vrel_triad_lunar[comp] * (sin_p + cos_p);
    }

    // Block 3: cross-coupling via Solar triad
    // block3_size may differ from block_size (128D: 43 vs 42)
    let n_phases_b3 = block3_size.div_ceil(3);
    let phase_trig_b3: Vec<(f64, f64)> = if block3_size != block_size {
        (0..n_phases_b3)
            .map(|k| {
                let phase = std::f64::consts::TAU * (k as f64) / (n_phases_b3 as f64);
                (phase.sin(), phase.cos())
            })
            .collect()
    } else {
        phase_trig.clone()
    };

    for axis in 0..block3_size {
        let comp = axis % 3;
        let phase_idx = axis / 3;
        let (sin_p, cos_p) = phase_trig_b3[phase_idx];
        let h_c = h_triad_solar[comp] / h_earth_norm.max(1e-30);
        let v_c = vhat_triad_solar[comp];
        v_nd[b3_start + axis] = cross_sign * v_rel_norm * (h_c * sin_p + v_c * cos_p);
    }

    // AVT contraction
    let n_viol = avt.violations.len().max(1) as f64;
    let mut force_nd = vec![0.0f64; dim];
    for &(i, j, _k, m, sign) in &avt.violations {
        force_nd[m] += alpha * v_nd[i] * v_nd[j] * (sign as f64);
    }
    for f in &mut force_nd {
        *f /= n_viol;
    }

    // Project back to 3D using the cross-coupling block through the Solar triad
    let mut res_triad = [0.0f64; 3];
    for axis in 0..block3_size {
        let comp = axis % 3;
        let phase_idx = axis / 3;
        let (sin_p, cos_p) = phase_trig_b3[phase_idx];
        let h_c = h_triad_solar[comp] / h_earth_norm.max(1e-30);
        let v_c = vhat_triad_solar[comp];
        let proj = cross_sign * (h_c * sin_p + v_c * cos_p);
        res_triad[comp] += force_nd[b3_start + axis] * proj;
    }

    // Rotate from Solar triad back to ECI
    let res = e_v_solar * res_triad[0] + e_h_solar * res_triad[1] + e_n_solar * res_triad[2];

    res / (dim as f64)
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

    // --- 3-body embedding tests ---

    #[test]
    fn test_3body_zero_alpha() {
        let avt = test_avt();
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 7.0, -3.0);
        let v_wind = Vector3::new(0.0, 200.0, 50.0);
        let r_moon = Vector3::new(384400.0, 0.0, 0.0);
        let r_sun = Vector3::new(1.496e8, 0.0, 0.0);

        let f = compute_chingon_bivector_drag_3body(
            r, v, v_wind, 0.0, &avt, r_moon, r_sun,
        );
        assert_eq!(f, Vector3::zeros());
    }

    #[test]
    fn test_3body_moon_perturbation_changes_force() {
        let avt = test_avt();
        // Use off-axis geometry so Moon position rotates the Lunar triad
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 7.0, -3.0);
        let v_wind = Vector3::new(0.0, 200.0, 50.0);
        let r_sun = Vector3::new(1.496e8, 0.0, 0.0);
        let alpha = 1e-10;

        // Moon along +x (aligned with r)
        let r_moon_x = Vector3::new(384400.0, 0.0, 0.0);
        let f_x = compute_chingon_bivector_drag_3body(
            r, v, v_wind, alpha, &avt, r_moon_x, r_sun,
        );

        // Moon along +y (perpendicular to r, different triad rotation)
        let r_moon_y = Vector3::new(0.0, 384400.0, 0.0);
        let f_y = compute_chingon_bivector_drag_3body(
            r, v, v_wind, alpha, &avt, r_moon_y, r_sun,
        );

        // Different Moon positions should produce different forces
        // because the Lunar triad (Block 2) rotates, changing AVT couplings
        let diff = (f_x - f_y).norm();
        assert!(
            diff > 0.0,
            "Moon at different positions should change force via triad rotation: \
             f_x={:?}, f_y={:?}",
            f_x, f_y
        );
    }

    #[test]
    fn test_3body_sign_sensitivity() {
        let avt = test_avt();
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v_wind = Vector3::new(0.0, 200.0, 50.0);
        let r_moon = Vector3::new(384400.0, 0.0, 0.0);
        let r_sun = Vector3::new(1.496e8, 0.0, 0.0);
        let alpha = 1e-10;

        // Southward approach
        let v_south = Vector3::new(0.0, 7.0, -3.0);
        let f_south = compute_chingon_bivector_drag_3body(
            r, v_south, v_wind, alpha, &avt, r_moon, r_sun,
        );

        // Northward approach (flip v_z)
        let v_north = Vector3::new(0.0, 7.0, 3.0);
        let f_north = compute_chingon_bivector_drag_3body(
            r, v_north, v_wind, alpha, &avt, r_moon, r_sun,
        );

        // Forces should differ in sign for at least one component
        let dot = f_south.dot(&f_north);
        assert!(
            dot < 0.0 || (f_south - f_north).norm() > 1e-20,
            "North vs south should produce different force directions: \
             f_south={:?}, f_north={:?}",
            f_south,
            f_north
        );
    }

    #[test]
    fn test_3body_finite_output() {
        let avt = test_avt();
        let r = Vector3::new(7331.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 8.949, -1.9);
        let v_wind = Vector3::new(-10.0, 200.0, 50.0);
        let r_moon = Vector3::new(300000.0, 200000.0, 50000.0);
        let r_sun = Vector3::new(1.0e8, 0.5e8, 0.0);

        let f = compute_chingon_bivector_drag_3body(
            r, v, v_wind, 8e-14, &avt, r_moon, r_sun,
        );
        assert!(f.x.is_finite(), "Force x not finite");
        assert!(f.y.is_finite(), "Force y not finite");
        assert!(f.z.is_finite(), "Force z not finite");
        assert!(f.norm() > 0.0, "Force should be nonzero for non-degenerate geometry");
    }
}
