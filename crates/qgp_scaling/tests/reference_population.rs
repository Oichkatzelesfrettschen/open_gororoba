use qgp_scaling::competing_models::{
    CurveSource, ReferencePopulation, all_cujet3_curves, all_langevin_curves,
    require_matched_population, retained_bic_population_admission,
};

#[test]
fn every_population_axis_discriminates_a_mismatch() {
    let population = ReferencePopulation {
        collision_system: "Pb-Pb",
        energy_gev: 5020,
        species: "charged hadron",
        centrality_percent: (0, 5),
    };
    assert!(require_matched_population(&population, &population).is_ok());
    let mismatches = [
        ReferencePopulation {
            collision_system: "Xe-Xe",
            ..population.clone()
        },
        ReferencePopulation {
            energy_gev: 2760,
            ..population.clone()
        },
        ReferencePopulation {
            species: "D meson",
            ..population.clone()
        },
        ReferencePopulation {
            centrality_percent: (0, 10),
            ..population.clone()
        },
    ];
    for mismatch in mismatches {
        assert!(require_matched_population(&population, &mismatch).is_err());
    }
    let invalid = ReferencePopulation {
        energy_gev: 0,
        ..population
    };
    assert!(require_matched_population(&invalid, &invalid).is_err());
}

#[test]
fn source_quarantine_survives_display_relabeling() {
    for mut curve in all_cujet3_curves().into_iter().chain(all_langevin_curves()) {
        curve.name = "Matched population";
        assert!(curve.source.require_identified_vector().is_err());
    }
    assert!(CurveSource::Constructed.require_identified_vector().is_ok());
    let error = retained_bic_population_admission().unwrap_err();
    for source_witness in [
        "Jai Prakash",
        "Figure 4",
        "static charm-quark",
        "6 fm/c",
        "250/350 MeV",
        "Figure 3(a)",
    ] {
        assert!(error.contains(source_witness), "missing {source_witness}");
    }
}

#[test]
fn source_repair_preserves_all_retained_numerical_vectors() {
    let cujet_values = [
        vec![
            0.13, 0.16, 0.21, 0.29, 0.34, 0.40, 0.44, 0.47, 0.50, 0.54, 0.57,
        ],
        vec![
            0.18, 0.22, 0.28, 0.35, 0.40, 0.47, 0.51, 0.54, 0.56, 0.60, 0.62,
        ],
        vec![
            0.30, 0.35, 0.41, 0.49, 0.53, 0.58, 0.62, 0.65, 0.67, 0.70, 0.72,
        ],
    ];
    for (curve, expected) in all_cujet3_curves().into_iter().zip(cujet_values) {
        assert_eq!(curve.source, CurveSource::QuarantinedCujet);
        assert_eq!(curve.n_params, 5);
        assert_eq!(
            curve.pt,
            vec![
                5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0
            ]
        );
        assert_eq!(curve.raa, expected);
    }
    let langevin_values = [
        vec![
            0.60, 0.38, 0.28, 0.22, 0.20, 0.21, 0.24, 0.28, 0.33, 0.40, 0.46, 0.50, 0.54, 0.57,
        ],
        vec![
            0.72, 0.55, 0.46, 0.42, 0.41, 0.43, 0.47, 0.50, 0.55, 0.62, 0.67, 0.71, 0.74, 0.76,
        ],
    ];
    for (curve, expected) in all_langevin_curves().into_iter().zip(langevin_values) {
        assert_eq!(curve.source, CurveSource::QuarantinedLangevin);
        assert_eq!(curve.n_params, 3);
        assert_eq!(
            curve.pt,
            vec![
                2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0
            ]
        );
        assert_eq!(curve.raa, expected);
    }
}

#[test]
fn retained_comparison_reports_both_incompatible_competitors() {
    let error = retained_bic_population_admission().expect_err("retained populations differ");
    for predicate in [
        "CUJET3.0",
        "collision energy",
        "fractional Langevin",
        "particle species",
        "centrality",
    ] {
        assert!(error.contains(predicate));
    }
}
