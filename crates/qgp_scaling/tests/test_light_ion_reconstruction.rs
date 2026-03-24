use qgp_scaling::{
    data_tables::{alice_oo_5360_pi0_raa, atlas_nene_5360_v2, cms_oo_5360_raa},
    glauber::{SigmaNN, compute_centrality_bins},
    multiplicity::{CollisionSystem, multiplicity_table},
    nucleus::NucleusParams,
};

#[test]
fn test_oo_5360_reconstruction_consistency() {
    // 1. Verify multiplicity data exists for O-O
    let mult = multiplicity_table(CollisionSystem::OO5360);
    assert!(!mult.is_empty());
    assert_eq!(mult[0].dnch_deta, 151.0); // Central value check

    // 2. Verify R_AA reference data exists
    let raa_ref = cms_oo_5360_raa();
    assert!(!raa_ref.is_empty());
    let pt6 = raa_ref.iter().find(|r| (r.pt - 6.0).abs() < 0.1).unwrap();
    assert!((pt6.raa - 0.69).abs() < 1e-6);

    // 3. Run Glauber calculation for O-O at 5.36 TeV
    let sigma = SigmaNN::lhc_5360();
    let o16 = NucleusParams::o16();
    let edges = vec![0.0, 0.05, 0.10];
    let bins = compute_centrality_bins(&edges, &sigma, &o16, 40, 100);

    assert_eq!(bins.len(), 2);
    // Central O-O should have Npart ~ 25-28
    assert!(bins[0].n_part > 20.0 && bins[0].n_part < 30.0);
}

#[test]
fn test_nene_5360_deformation_anchor() {
    // Verify ATLAS v2 reference for Ne-Ne prolate shape
    let v2_ref = atlas_nene_5360_v2();
    assert!(!v2_ref.is_empty());

    // Central bin (0-5%) should show flow signature
    let central = &v2_ref[0];
    assert_eq!(central.n, 2);
    assert!(central.v_n > 0.04);
}

#[test]
fn test_pi0_oo_comparison() {
    let pi0_raa = alice_oo_5360_pi0_raa();
    assert!(!pi0_raa.is_empty());

    // ALICE pi0 R_AA should be significantly suppressed (< 1.0)
    for point in pi0_raa {
        assert!(point.raa < 0.9);
    }
}
