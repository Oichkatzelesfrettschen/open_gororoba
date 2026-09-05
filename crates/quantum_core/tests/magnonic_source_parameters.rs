use quantum_core::magnonic_crystal::{
    InversionBreakingParams, MagnonicTBParams, build_magnonic_9band,
};

#[test]
fn source_table_ii_sublattice_difference_preserves_published_sign() {
    let parameters = MagnonicTBParams::kaman_table_ii();
    let inversion = InversionBreakingParams::kaman_table_ii();
    let model = build_magnonic_9band(&parameters, &inversion, 333.0);
    // Kaman et al., Table II and section V.A define Delta=(epsilon_B-epsilon_A)/2.
    let source_differences = [(0, 3, 0.12), (1, 4, 0.35), (2, 5, 0.35)];
    for (orbital_a, orbital_b, source_delta) in source_differences {
        let observed = (model.orbitals[orbital_b].on_site_energy
            - model.orbitals[orbital_a].on_site_energy)
            / 2.0;
        assert!((observed - source_delta).abs() < 1e-14);
    }
}

#[test]
fn reference_and_source_parameters_remain_distinct() {
    let reference = MagnonicTBParams::kaman_default();
    let source = MagnonicTBParams::kaman_table_i();
    let reference_values = [
        reference.eps_s,
        reference.eps_p,
        reference.eps_k,
        reference.t_sk,
        reference.t_pk,
    ];
    let source_values = [
        source.eps_s,
        source.eps_p,
        source.eps_k,
        source.t_sk,
        source.t_pk,
    ];
    assert!(
        reference_values
            .iter()
            .zip(source_values)
            .all(|(reference, source)| *reference != source)
    );
    assert!(source.t_sk < 0.0 && source.t_pk < 0.0);
    let model = build_magnonic_9band(&source, &InversionBreakingParams::none(), 333.0);
    assert!(
        model
            .hoppings
            .iter()
            .filter(|hopping| hopping.from == 0)
            .all(|hopping| hopping.amplitude.re < 0.0)
    );
}
