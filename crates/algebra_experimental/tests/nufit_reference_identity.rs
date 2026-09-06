//! Fixed-angle membership in source-identified normal-ordering reference intervals.

use algebra_experimental::experimental_predictions::{NuFit60, SigmaContour};
use serde::Deserialize;
use std::{collections::BTreeSet, error::Error, process::Command};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    release: String,
    analysis: String,
    source: String,
    sha256: String,
    one_sigma: [f64; 2],
    three_sigma: [f64; 2],
    inside_one_sigma: bool,
    inside_three_sigma: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    theta23_degrees: f64,
    ordering: String,
    references: Vec<Reference>,
    nufit60: Vec<CompleteReference>,
    historical_composite_contours: [[f64; 5]; 6],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompleteReference {
    analysis: String,
    contours: [[f64; 5]; 6],
    cp_a_one_sigma: bool,
    cp_a_three_sigma: bool,
    cp_b_one_sigma: bool,
    cp_b_three_sigma: bool,
}

fn contour_values(contour: SigmaContour) -> [f64; 5] {
    [
        contour.best,
        contour.one_sigma_low,
        contour.one_sigma_high,
        contour.three_sigma_low,
        contour.three_sigma_high,
    ]
}

fn complete_values(reference: &NuFit60) -> [[f64; 5]; 6] {
    [
        reference.theta_12,
        reference.theta_13,
        reference.theta_23,
        reference.delta_cp,
        reference.dm21_sq,
        reference.dm31_sq,
    ]
    .map(contour_values)
}

fn periodic_contains(interval: [f64; 2], angle: f64) -> bool {
    (-3..=3).any(|turn| contains(interval, angle + 360.0 * f64::from(turn)).unwrap())
}

fn contains(interval: [f64; 2], value: f64) -> Result<bool, &'static str> {
    if !value.is_finite()
        || !interval.iter().all(|bound| bound.is_finite())
        || interval[0] > interval[1]
    {
        return Err("invalid ordered finite interval");
    }
    Ok(interval[0] <= value && value <= interval[1])
}

fn check(fixture: &Fixture) -> Result<(), Box<dyn Error>> {
    if fixture.ordering != "normal" || fixture.theta23_degrees != 48.99 {
        return Err("reference ordering or fixed comparison angle changed".into());
    }
    let expected = BTreeSet::from([
        ("6.0", "IC19 without SK atmospheric data"),
        ("6.0", "IC24 with SK atmospheric data"),
        ("6.1", "IC23 without SK atmospheric data"),
        ("6.1", "IC24 with SK atmospheric data"),
    ]);
    let actual: BTreeSet<_> = fixture
        .references
        .iter()
        .map(|reference| (reference.release.as_str(), reference.analysis.as_str()))
        .collect();
    if actual != expected || fixture.references.len() != expected.len() {
        return Err("reference variants are missing, duplicated or mislabeled".into());
    }
    for reference in &fixture.references {
        if contains(reference.one_sigma, fixture.theta23_degrees)? != reference.inside_one_sigma
            || contains(reference.three_sigma, fixture.theta23_degrees)?
                != reference.inside_three_sigma
        {
            return Err("recorded membership differs from the declared interval".into());
        }
    }
    Ok(())
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/nufit_reference_intervals.json")).unwrap()
}

#[test]
fn fixed_angle_reference_membership_and_pdf_receipts_agree() -> Result<(), Box<dyn Error>> {
    let fixture = fixture();
    check(&fixture)?;
    let mut checked_sources = BTreeSet::new();
    for reference in &fixture.references {
        if checked_sources.insert((&reference.source, &reference.sha256)) {
            // Coreutils supplies an independent byte-identity check without changing fitting dependencies.
            let output = Command::new("sha256sum")
                .arg("--")
                .arg(repo_root::path!(&reference.source))
                .output()?;
            if !output.status.success() {
                return Err("required sha256sum source verification failed".into());
            }
            let text = std::str::from_utf8(&output.stdout)?;
            if text.split_whitespace().next() != Some(reference.sha256.as_str()) {
                return Err("reference PDF checksum differs from the inspected source".into());
            }
        }
        println!(
            "release={} analysis={} theta23={} inside_one_sigma={} inside_three_sigma={}",
            reference.release,
            reference.analysis,
            fixture.theta23_degrees,
            reference.inside_one_sigma,
            reference.inside_three_sigma
        );
    }
    assert_eq!(
        fixture
            .references
            .iter()
            .filter(|row| row.inside_one_sigma)
            .count(),
        1
    );
    assert!(fixture.references.iter().all(|row| row.inside_three_sigma));
    Ok(())
}

#[test]
fn source_variant_and_historical_universal_exclusion_mutations_fail() {
    let mut changed = fixture();
    changed.references[0].inside_one_sigma = false;
    assert!(check(&changed).is_err());
    let mut changed = fixture();
    changed.references[0].analysis = "IC24 with SK atmospheric data".to_owned();
    assert!(check(&changed).is_err());
    let mut changed = fixture();
    changed.references[0].one_sigma.swap(0, 1);
    assert!(check(&changed).is_err());
    assert!(contains([f64::NAN, 49.2], 48.99).is_err());
    assert_eq!(contains([47.6, 49.2], 47.6), Ok(true));
    assert_eq!(contains([47.6, 49.2], 49.2), Ok(true));
}

#[test]
fn complete_nufit60_source_rows_cp_targets_and_mass_ratio() {
    let fixture = fixture();
    assert_eq!(fixture.nufit60.len(), 2);
    for reference in &fixture.nufit60 {
        let production = match reference.analysis.as_str() {
            "IC24 with SK atmospheric data" => NuFit60::normal_ordering_ic24_with_sk(),
            "IC19 without SK atmospheric data" => NuFit60::normal_ordering_ic19_without_sk(),
            _ => panic!("unexpected complete-table variant"),
        };
        assert_eq!(complete_values(&production), reference.contours);
        let phase = reference.contours[3];
        for (angle, expected_one, expected_three) in [
            (165.0, reference.cp_a_one_sigma, reference.cp_a_three_sigma),
            (93.0, reference.cp_b_one_sigma, reference.cp_b_three_sigma),
        ] {
            assert_eq!(periodic_contains([phase[1], phase[2]], angle), expected_one);
            assert_eq!(
                periodic_contains([phase[3], phase[4]], angle),
                expected_three
            );
            for shifted in [
                angle - 720.0,
                angle - 360.0,
                angle,
                angle + 360.0,
                angle + 720.0,
            ] {
                assert_eq!(
                    production.delta_cp.in_one_sigma_degrees(shifted),
                    expected_one
                );
                assert_eq!(
                    production.delta_cp.in_three_sigma_degrees(shifted),
                    expected_three
                );
            }
            println!(
                "cp analysis={} angle={angle} one_sigma={expected_one} three_sigma={expected_three}",
                reference.analysis
            );
        }
        let numerator = reference.contours[4];
        let denominator = reference.contours[5];
        let ratio = numerator[0] / denominator[0];
        let one_sigma_box = [numerator[1] / denominator[2], numerator[2] / denominator[1]];
        let three_sigma_box = [numerator[3] / denominator[4], numerator[4] / denominator[3]];
        assert_eq!(ratio, production.dm21_sq.best / production.dm31_sq.best);
        assert!(contains(one_sigma_box, 0.0304).unwrap());
        assert!(contains(three_sigma_box, 0.0304).unwrap());
        println!(
            "ratio analysis={} best={ratio:.17} prediction=0.0304 difference={:.17} marginal_one_sigma_box=[{:.17},{:.17}] marginal_three_sigma_box=[{:.17},{:.17}] joint_confidence=unassessed",
            reference.analysis,
            0.0304 - ratio,
            one_sigma_box[0],
            one_sigma_box[1],
            three_sigma_box[0],
            three_sigma_box[1]
        );
    }
    assert_eq!(
        complete_values(&NuFit60::normal_ordering()),
        fixture.nufit60[0].contours
    );
}

#[test]
fn historical_composite_values_and_comparisons_remain_separate() {
    let fixture = fixture();
    let historical = NuFit60::historical_composite_reference();
    assert_eq!(
        complete_values(&historical),
        fixture.historical_composite_contours
    );
    assert!(historical.delta_cp.in_one_sigma_degrees(165.0));
    assert!(!historical.delta_cp.in_one_sigma_degrees(93.0));
    assert!(historical.delta_cp.in_three_sigma_degrees(93.0));
    assert_ne!(complete_values(&historical), fixture.nufit60[0].contours);
    assert_ne!(complete_values(&historical), fixture.nufit60[1].contours);
    println!(
        "historical_composite cp_a_one_sigma=true cp_b_one_sigma=false cp_b_three_sigma=true mislabeled_slot_ratio={:.17} source_population=unresolved",
        historical.dm21_sq.best / historical.dm31_sq.best
    );
}
