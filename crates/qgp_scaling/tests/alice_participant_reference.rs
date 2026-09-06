use qgp_scaling::data_tables::{alice_pbpb_5020_npart, alice_public_2018_011_table1_participants};

#[test]
fn source_table_preserves_population_and_distinct_uncertainties() {
    let table = alice_public_2018_011_table1_participants();
    assert_eq!(table.source_report, "ALICE-PUBLIC-2018-011");
    assert_eq!(table.source_table, "Table 1");
    assert_eq!(table.source_pdf_page, 7);
    assert_eq!(
        table.source_pdf_sha256,
        "de4d98816c22c0991e13e1669d9d708ffea7b470b56a020bbe90e38625ef7a6c"
    );
    assert_eq!(table.collision_system, "Pb-Pb");
    assert_eq!(table.sqrt_s_nn_tev, 5.02);
    assert_eq!(
        table.centrality_selection,
        "Sharp cuts in simulated V0M multiplicity"
    );
    assert_eq!(
        table.distribution_model,
        "NBD-Glauber fit and Glauber Monte Carlo"
    );
    assert_eq!(
        table.systematic_definition,
        "Absolute mean uncertainty from independent Glauber parameter variations; source contributions combined in quadrature"
    );
    let source_rows: Vec<_> = include_str!(
        "../../../data/output/audit/qgp-participant-reference-intake/table1-participant-rows.csv"
    )
    .lines()
    .skip(1)
    .map(|line| {
        line.split(',')
            .map(|value| value.parse::<f64>().expect("source-table numeric field"))
            .collect::<Vec<_>>()
    })
    .collect();
    assert_eq!(source_rows.len(), 9);
    for (row, source) in table.rows.iter().zip(source_rows) {
        assert_eq!(source.len(), 5);
        assert_eq!(row.cent_lo, source[0] / 100.0);
        assert_eq!(row.cent_hi, source[1] / 100.0);
        assert_eq!(row.mean, source[2]);
        assert_eq!(row.rms, source[3]);
        assert_eq!(row.mean_systematic, source[4]);
        assert!(row.rms > row.mean_systematic);
    }
}

#[test]
fn legacy_values_remain_a_distinct_reference() {
    let historical = alice_pbpb_5020_npart();
    let admitted = alice_public_2018_011_table1_participants();
    let historical_means = [382.8, 329.7, 260.5, 186.4, 128.9, 85.0, 52.8, 30.0, 15.8];
    let historical_errors = [3.1, 4.6, 4.4, 3.8, 3.3, 2.6, 2.0, 1.3, 0.6];
    assert_eq!(historical.len(), 9);
    for (index, (old, source)) in historical.iter().zip(admitted.rows).enumerate() {
        assert_eq!(old.n_part, historical_means[index]);
        assert_eq!(old.n_part_err, historical_errors[index]);
        assert_eq!(old.cent_lo, source.cent_lo);
        assert_eq!(old.cent_hi, source.cent_hi);
        assert_ne!(old.n_part, source.mean);
        assert_ne!(old.n_part_err, source.mean_systematic);
    }
}
