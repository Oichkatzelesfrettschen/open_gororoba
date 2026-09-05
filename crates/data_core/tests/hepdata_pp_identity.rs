use data_core::catalogs::hepdata_table::{HepDataTableContract, admit_table_json};

const TABLE: &str =
    include_str!("../../../data/output/audit/qgp-pp-spectrum-calibration/pp-table4.json");
const YAML_REFERENCE: &str = include_str!(
    "../../../data/output/audit/qgp-pp-spectrum-calibration/pp-table4-yaml-reference.csv"
);

#[test]
fn both_pp_populations_match_complete_yaml_reference() {
    let mut comparisons = 0;
    for (group, energy) in [(0, "5020.0 GeV"), (1, "2760.0 GeV")] {
        let qualifiers = [
            ("SQRT(S)", energy),
            ("ETARAP", "-0.8 - 0.8"),
            ("RE", "P P --> CHARGED X"),
        ];
        let contract = HepDataTableContract {
            doi: "10.17182/hepdata.86210.v1/t4",
            independent_header: "PT [GEV]",
            dependent_header: "(1/(Nevt))*D2(N)/DETARAP/DPT [C/GEV]",
            group,
            qualifiers: &qualifiers,
            required_errors: &["stat", "sys"],
        };
        let rows = admit_table_json(TABLE, &contract).unwrap();
        assert_eq!(rows.len(), 39);
        assert_eq!(rows[0].low, "0.15");
        assert_eq!(rows[0].high, "0.20");
        assert_eq!(rows[38].high, "50.00");
        let mut reader = csv::Reader::from_reader(YAML_REFERENCE.as_bytes());
        let reference: Vec<_> = reader
            .records()
            .map(Result::unwrap)
            .filter(|row| row[0].parse::<usize>().unwrap() == group)
            .collect();
        assert_eq!(reference.len(), rows.len());
        for (actual, expected) in rows.iter().zip(&reference) {
            assert_eq!(&expected[1], energy);
            let stat = &actual
                .errors
                .iter()
                .find(|error| error.label == "stat")
                .unwrap()
                .symerror;
            let syst = &actual
                .errors
                .iter()
                .find(|error| error.label == "sys")
                .unwrap()
                .symerror;
            for (field, column) in [&actual.low, &actual.high, &actual.value, stat, syst]
                .into_iter()
                .zip(2..7)
            {
                assert_eq!(field.as_str(), &expected[column]);
                comparisons += 1;
            }
        }
    }
    println!("provider representation: 2 groups, 39 rows each, {comparisons} numeric fields agree");
    assert_eq!(comparisons, 390);
}

#[test]
fn energy_exchange_cannot_pass_selected_population() {
    let contract = HepDataTableContract {
        doi: "10.17182/hepdata.86210.v1/t4",
        independent_header: "PT [GEV]",
        dependent_header: "(1/(Nevt))*D2(N)/DETARAP/DPT [C/GEV]",
        group: 0,
        qualifiers: &[("SQRT(S)", "2760.0 GeV")],
        required_errors: &["stat", "sys"],
    };
    assert!(admit_table_json(TABLE, &contract).is_err());
    let mut altered: serde_json::Value = serde_json::from_str(TABLE).unwrap();
    altered["values"][0]["y"].as_array_mut().unwrap().reverse();
    assert!(admit_table_json(&altered.to_string(), &contract).is_err());
}
