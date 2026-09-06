use qgp_scaling::fragmentation::{
    DssGrid, DssGridIdentity, DssHadron, FragmentationValues, HadronCharge, PerturbativeOrder,
    SourceRealPrecision,
};
use std::collections::BTreeSet;

const LO: &str = include_str!(
    "../../../data/output/audit/qgp-fragmentation-grid-intake/pinned-mirrors/jeffersonlab-HLO.GRID"
);
const NLO: &str = include_str!(
    "../../../data/output/audit/qgp-fragmentation-grid-intake/pinned-mirrors/jeffersonlab-HNLO.GRID"
);
const REAL32: &str =
    include_str!("../../../data/output/audit/qgp-fragmentation-interpolation/default_real32.csv");
const REAL64: &str = include_str!(
    "../../../data/output/audit/qgp-fragmentation-interpolation/default_real64_knots_and_exponent_amendment.csv"
);
const Z: [f64; 16] = [
    0.05, 0.095, 0.1, 0.2, 0.225, 0.35, 0.5, 0.7, 0.93, 0.999, 1.0, 0.053, 0.137, 0.333, 0.777,
    0.975,
];
const Q2: [f64; 9] = [1.0, 1.25, 10.0, 100.0, 100000.0, 1.1, 37.0, 1777.0, 99999.0];
const CHARGES: [HadronCharge; 4] = [
    HadronCharge::Plus,
    HadronCharge::Minus,
    HadronCharge::Average,
    HadronCharge::Sum,
];

fn identity(order: usize) -> DssGridIdentity {
    DssGridIdentity {
        hadron: DssHadron::ChargedHadron,
        order: if order == 0 {
            PerturbativeOrder::Lo
        } else {
            PerturbativeOrder::Nlo
        },
        provenance: format!(
            "retained mirrored charged grid order={order}; hash admission external"
        ),
    }
}
fn grid(order: usize, precision: SourceRealPrecision) -> DssGrid {
    DssGrid::parse(
        if order == 0 { LO } else { NLO },
        identity(order),
        precision,
    )
    .unwrap()
}
fn channels(value: FragmentationValues) -> [f64; 9] {
    [
        value.u,
        value.ubar,
        value.d,
        value.dbar,
        value.s,
        value.sbar,
        value.charm,
        value.bottom,
        value.gluon,
    ]
}
fn gate(reference: f64) -> f64 {
    2e-12 + 2e-11 * reference.abs()
}

struct ReferenceRow {
    order: usize,
    charge: usize,
    z: usize,
    q2: usize,
    values: [f64; 9],
}

fn reference_rows(source: &str) -> Result<Vec<ReferenceRow>, String> {
    let mut lines = source.lines();
    if lines.next() != Some("order,charge,z,q2,u,ub,d,db,s,sb,c,b,g") {
        return Err("wrong reference header/channel identity".into());
    }
    let mut keys = BTreeSet::new();
    let mut rows = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 13 {
            return Err("wrong reference field count".into());
        }
        let order = match fields[0] {
            "0" => 0,
            "1" => 1,
            _ => return Err("invalid order identity".into()),
        };
        let charge = match fields[1] {
            "positive" => 0,
            "negative" => 1,
            "average" => 2,
            "sum" => 3,
            _ => return Err("invalid charge identity".into()),
        };
        let numbers: Vec<f64> = fields[2..]
            .iter()
            .map(|field| {
                field
                    .parse::<f64>()
                    .map_err(|_| "invalid reference decimal".to_string())
            })
            .collect::<Result<_, _>>()?;
        if numbers.iter().any(|value| !value.is_finite()) {
            return Err("nonfinite reference".into());
        }
        let z = Z
            .iter()
            .position(|value| value.to_bits() == numbers[0].to_bits())
            .ok_or("z outside frozen mesh")?;
        let q2 = Q2
            .iter()
            .position(|value| value.to_bits() == numbers[1].to_bits())
            .ok_or("Q2 outside frozen mesh")?;
        if !keys.insert((order, charge, z, q2)) {
            return Err("duplicate reference key".into());
        }
        rows.push(ReferenceRow {
            order,
            charge,
            z,
            q2,
            values: numbers[2..].try_into().unwrap(),
        });
    }
    let expected: BTreeSet<_> = (0..2)
        .flat_map(|order| {
            (0..4).flat_map(move |charge| {
                (0..16).flat_map(move |z| (0..9).map(move |q2| (order, charge, z, q2)))
            })
        })
        .collect();
    if keys != expected || rows.len() != 1152 {
        return Err("reference mesh coverage mismatch".into());
    }
    Ok(rows)
}

#[test]
fn both_precision_variants_match_every_frozen_reference_channel() {
    for (label, source, precision, wrong_precision) in [
        (
            "default_real32",
            REAL32,
            SourceRealPrecision::DefaultReal32,
            SourceRealPrecision::DefaultReal64,
        ),
        (
            "default_real64_knots_and_exponent_amendment",
            REAL64,
            SourceRealPrecision::DefaultReal64,
            SourceRealPrecision::DefaultReal32,
        ),
    ] {
        let rows = reference_rows(source).unwrap();
        let grids = [grid(0, precision), grid(1, precision)];
        let wrong = [grid(0, wrong_precision), grid(1, wrong_precision)];
        let mut maximum_absolute = 0.0_f64;
        let mut maximum_gate_ratio = 0.0_f64;
        let mut failures = 0;
        let mut wrong_precision_failures = 0;
        let mut wrong_charge_factor_failures = 0;
        for row in &rows {
            let actual = channels(
                grids[row.order]
                    .evaluate(Z[row.z], Q2[row.q2], CHARGES[row.charge])
                    .unwrap(),
            );
            let wrong_actual = channels(
                wrong[row.order]
                    .evaluate(Z[row.z], Q2[row.q2], CHARGES[row.charge])
                    .unwrap(),
            );
            let average = channels(
                grids[row.order]
                    .evaluate(Z[row.z], Q2[row.q2], HadronCharge::Average)
                    .unwrap(),
            );
            for channel in 0..9 {
                let difference = (actual[channel] - row.values[channel]).abs();
                maximum_absolute = maximum_absolute.max(difference);
                maximum_gate_ratio = maximum_gate_ratio.max(difference / gate(row.values[channel]));
                failures += usize::from(difference > gate(row.values[channel]));
                wrong_precision_failures += usize::from(
                    (wrong_actual[channel] - row.values[channel]).abs() > gate(row.values[channel]),
                );
                if row.charge == 3 {
                    wrong_charge_factor_failures += usize::from(
                        (average[channel] - row.values[channel]).abs() > gate(row.values[channel]),
                    );
                }
            }
        }
        println!(
            "DSS variant={label} rows={} channels={} maxabs={maximum_absolute:.17e} max_gate_ratio={maximum_gate_ratio:.17e} failures={failures} wrong_precision_failures={wrong_precision_failures} wrong_charge_factor_failures={wrong_charge_factor_failures}",
            rows.len(),
            rows.len() * 9
        );
        assert_eq!(failures, 0, "{label}: unchanged numerical gate");
        assert!(wrong_precision_failures > 0);
        assert!(wrong_charge_factor_failures > 0);
    }
}

#[test]
fn immutable_sets_charge_identities_and_endpoint_hold() {
    for precision in [
        SourceRealPrecision::DefaultReal32,
        SourceRealPrecision::DefaultReal64,
    ] {
        let forward = [grid(0, precision), grid(1, precision)];
        let reverse_nlo = grid(1, precision);
        let reverse_lo = grid(0, precision);
        let reverse = [reverse_lo, reverse_nlo];
        let mut comparisons = 0;
        for order in [1, 0] {
            for z in Z.into_iter().rev() {
                for q2 in Q2.into_iter().rev() {
                    let positive =
                        channels(forward[order].evaluate(z, q2, HadronCharge::Plus).unwrap());
                    let negative =
                        channels(forward[order].evaluate(z, q2, HadronCharge::Minus).unwrap());
                    let sum = channels(forward[order].evaluate(z, q2, HadronCharge::Sum).unwrap());
                    let average = channels(
                        forward[order]
                            .evaluate(z, q2, HadronCharge::Average)
                            .unwrap(),
                    );
                    for charge in CHARGES {
                        if z == 1.0 {
                            assert!(
                                channels(forward[order].evaluate(z, q2, charge).unwrap())
                                    .iter()
                                    .all(|value| *value == 0.0)
                            );
                        }
                        let first = channels(forward[order].evaluate(z, q2, charge).unwrap())
                            .map(f64::to_bits);
                        let second = channels(reverse[order].evaluate(z, q2, charge).unwrap())
                            .map(f64::to_bits);
                        assert_eq!(first, second);
                        comparisons += 9;
                    }
                    for channel in 0..9 {
                        assert!(
                            (sum[channel] - positive[channel] - negative[channel]).abs()
                                <= gate(sum[channel])
                        );
                        assert_eq!(sum[channel], 2.0 * average[channel]);
                        if z == 1.0 {
                            assert_eq!(sum[channel], 0.0);
                        }
                    }
                }
            }
        }
        println!("immutable precision={precision:?} reverse_order_bit_comparisons={comparisons}");
    }
}

#[test]
fn reference_coverage_validator_rejects_mutations() {
    let mut lines: Vec<&str> = REAL32.lines().collect();
    lines.pop();
    assert!(reference_rows(&lines.join("\n")).is_err());
    lines.push(lines[1]);
    assert!(reference_rows(&lines.join("\n")).is_err());
    assert!(reference_rows(&REAL32.replacen("order,charge", "charge,order", 1)).is_err());
    assert!(reference_rows(&REAL32.replacen("positive", "unspecified", 1)).is_err());
}

#[test]
fn malformed_grids_and_outside_queries_reject() {
    let parse =
        |source: &str| DssGrid::parse(source, identity(0), SourceRealPrecision::DefaultReal32);
    let lines: Vec<&str> = LO.lines().collect();
    assert!(parse(&lines[..815].join("\n")).is_err());
    assert!(parse(&format!("{LO}{}\n", lines[0])).is_err());
    assert!(parse(&LO.replacen(lines[0], &lines[0][1..], 1)).is_err());
    for token in [
        "       NaN",
        "       inf",
        " 1.00e-999",
        " 1.00e+999",
        "  1.2.3E00",
        "\t1.000E+00",
    ] {
        assert_eq!(token.len(), 10);
        let mut mutated = LO.to_owned();
        mutated.replace_range(..10, token);
        assert!(parse(&mutated).is_err(), "{token}");
    }
    let mut missing = identity(0);
    let mut overflow = LO.to_owned();
    overflow.replace_range(..10, " 1.000e308");
    match parse(&overflow) {
        Err(error) => assert_eq!(error.0, "grid endpoint normalization overflow"),
        Ok(_) => panic!("endpoint normalization overflow admitted"),
    }
    missing.provenance = "  ".into();
    assert!(DssGrid::parse(LO, missing, SourceRealPrecision::DefaultReal32).is_err());
    let admitted = grid(0, SourceRealPrecision::DefaultReal32);
    for (z, q2) in [
        (0.05_f64.next_down(), 1.0),
        (1.0_f64.next_up(), 1.0),
        (0.5, 1.0_f64.next_down()),
        (0.5, 1e5_f64.next_up()),
        (f64::NAN, 1.0),
        (0.5, f64::NAN),
        (f64::INFINITY, 1.0),
        (0.5, f64::NEG_INFINITY),
    ] {
        assert!(admitted.evaluate(z, q2, HadronCharge::Sum).is_err());
    }
    for (z, q2) in [(0.05, 1.0), (1.0, 1e5)] {
        assert!(admitted.evaluate(z, q2, HadronCharge::Sum).is_ok());
    }
}

#[test]
fn synthetic_log_cell_matches_independent_bilinear_expression() {
    let mut source = String::new();
    for _ in 0..34 {
        for q_index in 0..24 {
            source.push_str(&format!("{:10.3e}", q_index as f64 + 1.0));
            source.push_str(&" 0.000E+00".repeat(8));
            source.push('\n');
        }
    }
    let grid = DssGrid::parse(&source, identity(0), SourceRealPrecision::DefaultReal64).unwrap();
    let z = (0.5_f64 * 0.55).sqrt();
    let q2 = (100.0_f64 * 180.0).sqrt();
    let factor =
        |value: f64| (1.0 - value) * (1.0 - value) * (1.0 - value) * (1.0 - value) * value.sqrt();
    // Geometric midpoints carry equal log-cell weights; columns at Q2=100,180 are12,13.
    let expected = 12.5 * 0.5 * (1.0 / factor(0.5) + 1.0 / factor(0.55)) * factor(z);
    let actual = grid.evaluate(z, q2, HadronCharge::Sum).unwrap();
    assert!((actual.u - expected).abs() <= gate(expected));
    assert_eq!(actual.u, actual.ubar);
    assert_eq!(actual.gluon, 0.0);
    println!(
        "synthetic log-cell expected={expected:.17e} actual={:.17e} absolute_error={:.17e}",
        actual.u,
        (actual.u - expected).abs()
    );
}
