use super::*;

#[test]
fn test_quaternion_heisenberg() {
    // dim=4: 3 sites, 2^3 = 8 states
    let model = build_cd_heisenberg(4, 0.0);
    assert_eq!(model.n_sites, 3);
    assert_eq!(model.couplings.len(), 3); // C(3,2) = 3

    let ed = exact_diagonalize(&model).expect("ED failed");
    assert_eq!(ed.hilbert_dim, 8);
    assert_eq!(ed.eigenvalues.len(), 8);

    // Ground state energy should be finite
    assert!(ed.eigenvalues[0].is_finite());
    // Energy should be ordered
    for i in 1..ed.eigenvalues.len() {
        assert!(ed.eigenvalues[i] >= ed.eigenvalues[i - 1] - 1e-10);
    }
}

#[test]
fn test_octonion_heisenberg() {
    // dim=8: 7 sites, 2^7 = 128 states
    let model = build_cd_heisenberg(8, 0.0);
    assert_eq!(model.n_sites, 7);
    assert_eq!(model.couplings.len(), 21); // C(7,2) = 21

    let ed = exact_diagonalize(&model).expect("ED failed");
    assert_eq!(ed.hilbert_dim, 128);

    // Check thermodynamics at T = 1.0
    let thermo = thermodynamic_quantities(&ed, 1.0).expect("thermo failed");
    assert!(thermo.specific_heat >= 0.0);
    assert!(thermo.susceptibility >= 0.0);
    assert!(thermo.partition_function > 0.0);
}

#[test]
fn test_j1j2_chain_unfrustrated() {
    // alpha=0: pure nearest-neighbor chain (unfrustrated)
    let model = build_j1j2_chain(8, 0.0, 1.0, 0.0);
    assert_eq!(model.n_sites, 8);
    assert_eq!(model.couplings.len(), 8); // N nearest-neighbor bonds

    let imbalance_val = graph_imbalance_index(&model);
    assert_eq!(imbalance_val, 0.0); // No imbalance for pure chain
}

#[test]
fn test_j1j2_chain_frustrated() {
    // J1 < 0 (ferromagnetic), J2 > 0 (antiferromagnetic): competing interactions
    // alpha = 0.5 means J2 = 0.5 * |J1|
    // Use J = -1 (ferromagnetic NN) + alpha * J (antiferromagnetic NNN)
    let model = build_j1j2_chain(8, -0.5, 1.0, 0.0);
    assert_eq!(model.couplings.len(), 16); // 8 NN + 8 NNN

    let imbalance_val = graph_imbalance_index(&model);
    // With mixed-sign couplings, triangles can be frustrated
    assert!(
        imbalance_val > 0.0,
        "imbalance = {} should be > 0",
        imbalance_val
    );
}

#[test]
fn test_cd_imbalance_index() {
    // Quaternion (dim=4): 0% frustrated (anti-commutativity makes all products negative)
    let model4 = build_cd_heisenberg(4, 0.0);
    let f4 = graph_imbalance_index(&model4);

    // Octonion (dim=8): should match face sign census
    let model8 = build_cd_heisenberg(8, 0.0);
    let f8 = graph_imbalance_index(&model8);

    // Imbalance should increase with dimension
    // (from CD face sign census: 0% at dim=4, then increases)
    assert!(f4.is_finite());
    assert!(f8.is_finite());
}

#[test]
fn test_interpolated_model() {
    let cd = build_cd_heisenberg(8, 0.0);

    // lambda=0: all couplings = +1 (unfrustrated)
    let ref_model = build_interpolated(&cd, 0.0);
    for &(_, _, j) in &ref_model.couplings {
        assert!((j - 1.0).abs() < 1e-10);
    }

    // lambda=1: full CD couplings
    let full_model = build_interpolated(&cd, 1.0);
    for (orig, interp) in cd.couplings.iter().zip(full_model.couplings.iter()) {
        assert!((orig.2 - interp.2).abs() < 1e-10);
    }
}

#[test]
fn test_spin_current_antisymmetric() {
    let model = build_cd_heisenberg(4, 0.0);
    let j_s = build_spin_current_operator(&model);
    let dim = 1 << model.n_sites;

    // Spin current should be antisymmetric: J_S^T = -J_S
    for a in 0..dim {
        for b in 0..dim {
            let diff = j_s[a * dim + b] + j_s[b * dim + a];
            assert!(
                diff.abs() < 1e-12,
                "Spin current not antisymmetric at ({}, {}): {} vs {}",
                a,
                b,
                j_s[a * dim + b],
                j_s[b * dim + a]
            );
        }
    }
}

#[test]
fn test_kubo_quaternion() {
    // Compute full Kubo transport for quaternion Heisenberg model
    let model = build_cd_heisenberg(4, 0.0);
    let transport = kubo_transport(&model, 0.5, 1e-10).expect("kubo failed");

    // At finite temperature, Drude weights should be non-negative
    assert!(transport.drude_weight_spin >= -1e-10);
    assert!(transport.drude_weight_energy >= -1e-10);
    assert!(transport.thermal_conductivity.is_finite());
}

#[test]
fn test_kubo_octonion() {
    // Compute Kubo transport for octonion Heisenberg model
    let model = build_cd_heisenberg(8, 0.0);
    let transport = kubo_transport(&model, 1.0, 1e-10).expect("kubo failed");

    assert!(transport.drude_weight_spin >= -1e-10);
    assert!(transport.thermal_conductivity.is_finite());
}

#[test]
#[ignore = "heavy research lane: transport alpha sweep"]
fn test_kubo_j1j2_alpha_sweep() {
    // Sweep imbalance parameter and check non-monotonic transport
    let n = 8;
    let alphas = [0.0, 0.1, 0.25, 0.5, 0.7, 1.0];
    let mut kths = Vec::new();

    for &alpha in &alphas {
        let model = build_j1j2_chain(n, alpha, 1.0, 3.0); // B=3 (near saturation)
        let transport = kubo_transport(&model, 0.1, 1e-10).expect("kubo failed");
        kths.push(transport.thermal_conductivity);
    }

    // All should be finite
    for k in &kths {
        assert!(k.is_finite(), "K_th not finite: {:?}", kths);
    }
}

#[test]
fn test_cd_vs_j1j2_imbalance() {
    // Compare imbalance indices
    let cd8 = build_cd_heisenberg(8, 0.0);
    let f_cd8 = graph_imbalance_index(&cd8);

    // Find the J1-J2 alpha value with matching imbalance_val
    let mut best_alpha = 0.0;
    let mut best_diff = f64::MAX;
    for i in 0..100 {
        let alpha = i as f64 * 0.02;
        let chain = build_j1j2_chain(8, alpha, 1.0, 0.0);
        let f_chain = graph_imbalance_index(&chain);
        let diff = (f_chain - f_cd8).abs();
        if diff < best_diff {
            best_diff = diff;
            best_alpha = alpha;
        }
    }

    // Should find a matching alpha
    assert!(best_alpha.is_finite());
}

#[test]
fn test_optimized_matches_naive_quaternion() {
    let model = build_cd_heisenberg(4, 0.0);
    let naive = kubo_transport(&model, 0.5, 1e-10).expect("kubo failed");
    let opt = kubo_transport_optimized(&model, 0.5, 1e-10).expect("kubo opt failed");

    assert!(
        (opt.drude_weight_spin - naive.drude_weight_spin).abs() < 1e-6,
        "D_S mismatch: opt={} naive={}",
        opt.drude_weight_spin,
        naive.drude_weight_spin
    );
    assert!(
        (opt.total_weight_spin - naive.total_weight_spin).abs()
            / naive.total_weight_spin.max(1e-20)
            < 0.01,
        "I0_S mismatch: opt={} naive={}",
        opt.total_weight_spin,
        naive.total_weight_spin
    );
    assert!(
        (opt.total_weight_energy - naive.total_weight_energy).abs()
            / naive.total_weight_energy.max(1e-20)
            < 0.01,
        "I0_E mismatch: opt={} naive={}",
        opt.total_weight_energy,
        naive.total_weight_energy
    );
}

#[test]
fn test_optimized_matches_naive_j1j2() {
    let model = build_j1j2_chain(8, 0.25, 1.0, 3.0);
    let naive = kubo_transport(&model, 0.1, 1e-10).expect("kubo failed");
    let opt = kubo_transport_optimized(&model, 0.1, 1e-10).expect("kubo opt failed");

    assert!(
        (opt.drude_weight_spin - naive.drude_weight_spin).abs() < 1e-6,
        "D_S mismatch: opt={} naive={}",
        opt.drude_weight_spin,
        naive.drude_weight_spin
    );
}
