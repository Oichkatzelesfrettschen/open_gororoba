//! Cross-validation: verified_core Sprint 59 modules vs production crates.
//!
//! Confirms that the Rocq-verified arithmetic produces identical results
//! to the corresponding production implementations in gr_core, algebra_core,
//! and optics_core.

const TOL: f64 = 1e-12;

// --- C-886..C-892: Brans-Dicke PPN cross-validation ---

#[test]
fn cross_validate_ppn_gamma_bd() {
    for &omega in &[0.1, 1.0, 10.0, 100.0, 1000.0, 43476.0] {
        let vc = verified_core::brans_dicke::ppn_gamma_bd(omega);
        let gc = gr_core::ppn_constraints::ppn_gamma_bd(omega);
        assert!(
            (vc - gc).abs() < TOL,
            "gamma_bd mismatch at omega={omega}: vc={vc}, gc={gc}"
        );
    }
}

#[test]
fn cross_validate_nordtvedt_bd() {
    for &omega in &[0.1, 1.0, 10.0, 100.0, 1000.0] {
        let vc = verified_core::brans_dicke::nordtvedt_bd(omega);
        let gc = gr_core::ppn_constraints::nordtvedt_parameter_bd(omega);
        assert!(
            (vc - gc).abs() < TOL,
            "nordtvedt mismatch at omega={omega}: vc={vc}, gc={gc}"
        );
    }
}

// --- C-893..C-899: CD complex/quaternion vs algebra_core ---

#[test]
fn cross_validate_complex_mul() {
    use verified_core::cayley_dickson::{Complex, complex_mul};

    let z = Complex { re: 3.0, im: 4.0 };
    let w = Complex { re: 1.0, im: -2.0 };
    let vc = complex_mul(&z, &w);

    // algebra_core uses Vec<f64> Cayley-Dickson
    let ac = algebra_core::construction::cayley_dickson::cd_multiply(&[3.0, 4.0], &[1.0, -2.0]);
    assert!((vc.re - ac[0]).abs() < TOL, "re: {} vs {}", vc.re, ac[0]);
    assert!((vc.im - ac[1]).abs() < TOL, "im: {} vs {}", vc.im, ac[1]);
}

#[test]
fn cross_validate_quat_mul() {
    use verified_core::quaternion::{Quat, quat_mul};

    let p = Quat {
        qw: 1.0,
        qx: 2.0,
        qy: 3.0,
        qz: 4.0,
    };
    let q = Quat {
        qw: 5.0,
        qx: 6.0,
        qy: 7.0,
        qz: 8.0,
    };
    let vc = quat_mul(&p, &q);

    let ac = algebra_core::construction::cayley_dickson::cd_multiply(
        &[1.0, 2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0, 8.0],
    );
    assert!((vc.qw - ac[0]).abs() < TOL, "w: {} vs {}", vc.qw, ac[0]);
    assert!((vc.qx - ac[1]).abs() < TOL, "x: {} vs {}", vc.qx, ac[1]);
    assert!((vc.qy - ac[2]).abs() < TOL, "y: {} vs {}", vc.qy, ac[2]);
    assert!((vc.qz - ac[3]).abs() < TOL, "z: {} vs {}", vc.qz, ac[3]);
}

#[test]
fn cross_validate_quat_norm_multiplicative() {
    use verified_core::quaternion::{Quat, quat_mul, quat_norm_sq};

    let p = Quat {
        qw: 1.0,
        qx: 2.0,
        qy: 3.0,
        qz: 4.0,
    };
    let q = Quat {
        qw: 5.0,
        qx: 6.0,
        qy: 7.0,
        qz: 8.0,
    };

    let prod_norm = quat_norm_sq(&quat_mul(&p, &q));
    let norm_prod = quat_norm_sq(&p) * quat_norm_sq(&q);
    assert!(
        (prod_norm - norm_prod).abs() < 1e-8,
        "|pq|^2 = {prod_norm} vs |p|^2*|q|^2 = {norm_prod}"
    );
}

// --- C-906: Quaternion inverse cross-validation ---

#[test]
fn cross_validate_quat_inverse() {
    use verified_core::{
        cayley_dickson::quat_inv,
        quaternion::{Quat, quat_mul},
    };

    let q = Quat {
        qw: 0.5,
        qx: 0.5,
        qy: 0.5,
        qz: 0.5,
    };
    let qi = quat_inv(&q);
    let prod = quat_mul(&q, &qi);
    assert!((prod.qw - 1.0).abs() < TOL, "w = {}", prod.qw);
    assert!(prod.qx.abs() < TOL, "x = {}", prod.qx);
    assert!(prod.qy.abs() < TOL, "y = {}", prod.qy);
    assert!(prod.qz.abs() < TOL, "z = {}", prod.qz);
}

// --- C-915..C-917: Binary entropy self-consistency ---

#[test]
fn cross_validate_binary_entropy_properties() {
    use verified_core::binary_entropy::binary_entropy;

    // Symmetry
    for &p in &[0.1, 0.2, 0.3, 0.4] {
        assert!(
            (binary_entropy(p) - binary_entropy(1.0 - p)).abs() < TOL,
            "H({p}) != H({})",
            1.0 - p
        );
    }

    // Max at 1/2 equals ln(2)
    assert!(
        (binary_entropy(0.5) - 2.0_f64.ln()).abs() < TOL,
        "H(0.5) should be ln(2)"
    );

    // Nonnegative
    for &p in &[0.01, 0.25, 0.5, 0.75, 0.99] {
        assert!(binary_entropy(p) >= 0.0);
    }
}

// --- C-913..C-914: Fano/TCMT cross-validation ---

#[test]
fn cross_validate_tcmt_antiresonance() {
    use verified_core::fano_resonance::tcmt_transmission_coeff;

    // At resonance (delta=0) with no loss: T = 0
    let t = tcmt_transmission_coeff(0.0, 1.0, 0.0);
    assert!(t.abs() < TOL, "T at resonance should be 0, got {t}");
}

#[test]
fn cross_validate_tcmt_unitarity() {
    use verified_core::fano_resonance::{tcmt_reflection_coeff, tcmt_transmission_coeff};

    for &delta in &[0.0, 0.5, 1.0, 2.0, 10.0] {
        let gamma_e = 1.5;
        let t = tcmt_transmission_coeff(delta, gamma_e, 0.0);
        let r = tcmt_reflection_coeff(delta, gamma_e);
        assert!(
            (t + r - 1.0).abs() < TOL,
            "T+R should be 1 at delta={delta}"
        );
    }
}
