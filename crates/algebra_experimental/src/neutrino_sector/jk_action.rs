#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorType {
    SU3,
    SU2,
    U1,
    Leptoquark,
    Dark,
}

pub fn classify_generator(gen_index: usize) -> GeneratorType {
    match gen_index {
        0..=7 => GeneratorType::SU3,
        8..=10 => GeneratorType::SU2,
        11 => GeneratorType::U1,
        12..=23 => GeneratorType::Leptoquark,
        _ => GeneratorType::Dark,
    }
}

// ---------------------------------------------------------------------------
// J_k complex structure -- full 16D sedenion action
// ---------------------------------------------------------------------------

/// Apply J_k complex structure to a 16D sedenion vector via octonion
/// left-multiplication on both halves independently.
///
/// # Mathematical foundation
///
/// A sedenion decomposes as `(a, b)` where `a` and `b` are octonions (the
/// lower and upper halves).  The complex structure `J_k` for `k in 1..=7`
/// acts by left-multiplication with the basis element `e_k`:
///
/// ```text
/// J_k(a, b) = (e_k * a,  e_k * b)
/// ```
///
/// This gives a **14D active subspace**: each half has 7 imaginary components
/// rotated by `e_k`, while the real (index 0) and `e_k` (index k) components
/// rotate among themselves.  Compare with the 6D perp-only action from
/// [`complex_structure()`](gororoba_algebra::lie::g2::stabilizer::complex_structure),
/// which restricts to the 6 indices perpendicular to both `e_0` and `e_k`
/// within the lower octonion only.
///
/// # Why full 16D instead of 6D perp-only
///
/// The 6D action discards contributions from indices `{0, k}` in the lower
/// block and the entire upper block `{8..15}`.  For friction profiles whose
/// support spans both halves, the 16D action captures phase angles that the
/// 6D action misses.  In practice (C-1496), friction profiles from selectors
/// `(e_7, e_8)` have zero upper-block weight, so the two actions agree --
/// but this is a property of the *profiles*, not of the action itself.
///
/// # Callers
///
/// - [`test_cp_violation_phase_only`]: primary CP violation pipeline
/// - [`test_cp_violation_jk_dimension_comparison`]: 6D-vs-16D diagnostic
/// - [`test_cp_violation_joint_3d_scan`]: 3D optimiser for J_max (C-1497 AMENDED)
/// - [`test_complex_pmns_alpha_scan`]: fine alpha_CP scan (origin of this
///   implementation, extracted from closure at former line 6508)
///
/// # Panics
///
/// Via [`Octonion::basis`]: panics if `k >= 8`.  Caller must ensure
/// `k in 1..=7` (the seven imaginary octonion units).
///
/// # Concrete example
///
/// ```text
/// k = 1,  v = (0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0)
///                      ^-- e_2 in lower block
/// J_1(v) = e_1 * e_2 = e_3  (by the Fano plane)
///        => result[3] = 1.0, all others zero
/// ```
pub fn apply_jk_full_16d(v: &[f64; 16], k: usize) -> [f64; 16] {
    use gororoba_algebra::construction::octonion::Octonion;

    // e_k is the imaginary unit that defines this complex structure.
    // Octonion::basis(k) returns the unit vector with component k = 1.0.
    let ek = Octonion::basis(k);
    let mut result = [0.0_f64; 16];

    // Lower octonion half: indices 0..7.
    // core::array::from_fn avoids an intermediate Vec allocation.
    let lower = Octonion::new(core::array::from_fn(|i| v[i]));
    // Left-multiplication: J_k(a) = e_k * a.  The product uses the
    // Fano-plane sign rule via cd_basis_mul_sign_iter internally.
    let jk_lower = ek.multiply(&lower);
    result[..8].copy_from_slice(&jk_lower.components);

    // Upper octonion half: indices 8..15.
    // The sedenion doubling construction makes the upper half an
    // independent octonion -- J_k acts on it identically.
    let upper = Octonion::new(core::array::from_fn(|i| v[i + 8]));
    let jk_upper = ek.multiply(&upper);
    result[8..16].copy_from_slice(&jk_upper.components);

    result
}
