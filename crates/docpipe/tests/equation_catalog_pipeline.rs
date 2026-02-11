use docpipe::equation_catalog::{
    build_catalog, catalog_to_toml, classify_category, classify_domain,
    convert_historical_csv_reader, write_catalog_toml, EquationCategory, EquationDomain,
    EquationSourceStream,
};

#[test]
fn dedupe_and_classification_behavior() {
    let fixture = include_str!("fixtures/historical_equations.csv");
    let input_rows =
        convert_historical_csv_reader(fixture.as_bytes(), EquationSourceStream::Text).unwrap();
    let known_modules = vec![
        "synthesis/modules/equations/eq_ge001_auto.tex".to_string(),
        "synthesis/modules/equations/eq_orphan_auto.tex".to_string(),
    ];
    let catalog = build_catalog(input_rows, &known_modules);

    assert_eq!(catalog.rows.len(), 3);

    let first = catalog
        .rows
        .iter()
        .find(|row| row.eq_id == "AE001")
        .unwrap();
    let duplicate = catalog
        .rows
        .iter()
        .find(|row| row.eq_id == "AE002")
        .unwrap();
    let genesis = catalog
        .rows
        .iter()
        .find(|row| row.eq_id == "GE001")
        .unwrap();

    assert_eq!(first.related_eqs, "");
    assert_eq!(duplicate.related_eqs, "AE001");
    assert_eq!(genesis.domain, EquationDomain::Qm);
    assert_eq!(genesis.category, EquationCategory::Fundamental);
}

#[test]
fn parity_and_gap_reports_are_stable() {
    let fixture = include_str!("fixtures/historical_equations.csv");
    let input_rows =
        convert_historical_csv_reader(fixture.as_bytes(), EquationSourceStream::Text).unwrap();
    let known_modules = vec![
        "synthesis/modules/equations/eq_ge001_auto.tex".to_string(),
        "synthesis/modules/equations/eq_orphan_auto.tex".to_string(),
    ];
    let catalog = build_catalog(input_rows, &known_modules);

    assert_eq!(catalog.parity_report.row_count, 3);
    assert_eq!(catalog.parity_report.modules_indexed, 2);
    assert_eq!(
        catalog.parity_report.rows_without_module_link,
        vec!["AE001".to_string(), "AE002".to_string()]
    );
    assert_eq!(
        catalog.parity_report.unreferenced_modules,
        vec!["synthesis/modules/equations/eq_orphan_auto.tex".to_string()]
    );

    assert_eq!(catalog.gap_report.total_missing_module_links, 2);
    assert_eq!(catalog.gap_report.buckets.len(), 1);
    let bucket = &catalog.gap_report.buckets[0];
    assert_eq!(bucket.framework, "Aether");
    assert_eq!(bucket.domain, EquationDomain::General);
    assert_eq!(bucket.missing_rows[0].eq_id, "AE001");
    assert_eq!(bucket.missing_rows[1].eq_id, "AE002");
}

#[test]
fn csv_conversion_and_toml_writer() {
    let fixture = include_str!("fixtures/historical_equations.csv");
    let input_rows =
        convert_historical_csv_reader(fixture.as_bytes(), EquationSourceStream::Text).unwrap();
    let catalog = build_catalog(input_rows, &[]);
    let toml_text = catalog_to_toml(&catalog).unwrap();

    assert!(toml_text.contains("[[rows]]"));
    assert!(toml_text.contains("[parity_report]"));
    assert!(toml_text.contains("[gap_report]"));

    let dir = tempfile::tempdir().unwrap();
    let out_path = dir.path().join("equation_catalog.toml");
    write_catalog_toml(&out_path, &catalog).unwrap();
    assert!(out_path.exists());
}

#[test]
fn classification_helpers_follow_script_rules() {
    let domain = classify_domain("H psi = E psi", "quantum wavefunction");
    let category = classify_category("fundamental hamiltonian relation");
    assert_eq!(domain, EquationDomain::Qm);
    assert_eq!(category, EquationCategory::Fundamental);
}
