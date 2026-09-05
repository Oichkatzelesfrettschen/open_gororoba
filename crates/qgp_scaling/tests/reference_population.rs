use qgp_scaling::competing_models::{
    ReferencePopulation, require_matched_population, retained_bic_population_admission,
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
