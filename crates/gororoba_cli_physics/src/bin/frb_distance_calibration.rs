//! Evaluate hash-bound source-level distance calibration without label leakage.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use cosmology_core::{
    distances::{comoving_distance, dm_excess_to_redshift, planck2018},
    frb_calibration::{
        ConformalQuantile, DistancePredictionInterval, assess_coverage, calibrate_absolute_errors,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    protocol: PathBuf,
    #[arg(long)]
    expected_protocol_sha256: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    repo_root: Option<PathBuf>,
}

#[derive(Clone, Deserialize)]
struct Protocol {
    schema_version: u32,
    data_path: String,
    data_sha256: String,
    source_ids: Vec<String>,
    split_salt: String,
    calibration_count: usize,
    coverages: Vec<f64>,
    primary_coverage: f64,
    primary_halo_dm: f64,
    primary_observer_host_dm: f64,
    halo_scenarios: Vec<f64>,
    observer_host_scenarios: Vec<f64>,
    historical_neighborhood_radii_mpc: Vec<f64>,
    undercoverage_alpha: f64,
    source_grouping_path: String,
    source_grouping_sha256: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct InputRow {
    source_id: String,
    dm_observed: f64,
    dm_error: f64,
    dm_disk_ne2001: f64,
    redshift: f64,
    telescope: String,
    frequency_mhz: f64,
    redshift_kind: String,
    reference: String,
    discovery_utc_as_reported: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct Prediction {
    redshift: f64,
    distance_mpc: f64,
}

#[derive(Serialize)]
struct EvaluatedRow {
    raw: InputRow,
    partition: &'static str,
    prediction: Option<Prediction>,
    prediction_error: Option<String>,
    catalog_distance_mpc: f64,
    signed_error_mpc: Option<f64>,
    absolute_error_mpc: Option<f64>,
    calibration_score_kind: &'static str,
}

fn digest(bytes: &[u8]) -> String {
    let alphabet = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        encoded.push(char::from(alphabet[usize::from(byte >> 4)]));
        encoded.push(char::from(alphabet[usize::from(byte & 15)]));
    }
    encoded
}

fn verify_digest(bytes: &[u8], expected: &str, label: &str) -> Result<()> {
    ensure!(
        expected.len() == 64 && expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "Invalid {label} SHA256"
    );
    ensure!(digest(bytes) == expected, "{label} SHA256 mismatch");
    Ok(())
}

fn validate_protocol(protocol: &Protocol) -> Result<()> {
    ensure!(protocol.schema_version == 1, "Unsupported protocol schema");
    ensure!(
        protocol.source_ids.len() == 21
            && protocol.source_ids.iter().collect::<BTreeSet<_>>().len() == 21,
        "Protocol requires 21 unique source IDs"
    );
    ensure!(
        protocol.calibration_count == 11,
        "Protocol requires an 11/10 split"
    );
    ensure!(
        protocol.coverages == [0.8, 0.9, 0.95] && protocol.primary_coverage == 0.9,
        "Unsupported coverage specification"
    );
    ensure!(
        protocol.halo_scenarios == [0.0, 50.0, 100.0]
            && protocol.observer_host_scenarios == [0.0, 50.0, 100.0],
        "Unsupported foreground grid"
    );
    ensure!(
        protocol.primary_halo_dm == 50.0 && protocol.primary_observer_host_dm == 50.0,
        "Unsupported primary foreground"
    );
    ensure!(
        protocol.historical_neighborhood_radii_mpc == [50.0, 100.0, 200.0, 500.0]
            && protocol.undercoverage_alpha == 0.05,
        "Unsupported assessment thresholds"
    );
    ensure!(!protocol.split_salt.is_empty(), "Split salt is required");
    Ok(())
}

fn validate_rows(rows: &[InputRow], protocol: &Protocol) -> Result<()> {
    ensure!(rows.len() == 21, "Expected exactly 21 cohort rows");
    let actual: BTreeSet<_> = rows.iter().map(|row| row.source_id.as_str()).collect();
    ensure!(actual.len() == rows.len(), "Duplicate cohort source IDs");
    let expected: BTreeSet<_> = protocol.source_ids.iter().map(String::as_str).collect();
    ensure!(
        actual == expected,
        "Cohort membership differs from protocol"
    );
    for row in rows {
        ensure!(
            [
                row.dm_observed,
                row.dm_error,
                row.dm_disk_ne2001,
                row.redshift,
                row.frequency_mhz
            ]
            .iter()
            .all(|value| value.is_finite()),
            "Nonfinite numeric field for {}",
            row.source_id
        );
        ensure!(
            row.redshift > 0.0
                && row.dm_observed >= 0.0
                && row.dm_error >= 0.0
                && row.dm_disk_ne2001 >= 0.0
                && row.frequency_mhz > 0.0,
            "Invalid numeric field for {}",
            row.source_id
        );
    }
    Ok(())
}

fn read_rows(bytes: &[u8], protocol: &Protocol) -> Result<Vec<InputRow>> {
    verify_digest(bytes, &protocol.data_sha256, "cohort")?;
    let rows: Vec<InputRow> = csv::Reader::from_reader(bytes)
        .deserialize()
        .collect::<std::result::Result<_, _>>()?;
    validate_rows(&rows, protocol)?;
    Ok(rows)
}

fn ordered_rows(rows: &[InputRow], protocol: &Protocol) -> Vec<InputRow> {
    let mut ordered = rows.to_vec();
    ordered.sort_by_cached_key(|row| {
        (
            digest(format!("{}:{}", protocol.split_salt, row.source_id).as_bytes()),
            row.source_id.clone(),
        )
    });
    ordered
}

/// The measured redshift label cannot enter the four-component predictor.
fn predict(observed: f64, disk: f64, halo: f64, observer_host: f64) -> Result<Prediction> {
    ensure!(
        [observed, disk, halo, observer_host]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "Invalid DM component"
    );
    // Summing the two fixed foregrounds first preserves equal-total identity.
    let cosmic_dm = observed - disk - (halo + observer_host);
    ensure!(cosmic_dm > 0.0, "Nonpositive cosmic-DM residual");
    let redshift = dm_excess_to_redshift(
        cosmic_dm,
        planck2018::OMEGA_M,
        planck2018::OMEGA_B,
        planck2018::H0,
    )?;
    let distance_mpc = comoving_distance(redshift, planck2018::OMEGA_M, planck2018::H0);
    ensure!(
        redshift.is_finite() && redshift > 0.0 && distance_mpc.is_finite() && distance_mpc > 0.0,
        "Nonfinite or underflowed positive-DM prediction"
    );
    Ok(Prediction {
        redshift,
        distance_mpc,
    })
}

fn evaluate(rows: &[InputRow], protocol: &Protocol) -> Result<Vec<Value>> {
    validate_protocol(protocol)?;
    validate_rows(rows, protocol)?;
    let ordered = ordered_rows(rows, protocol);
    let mut scenarios = Vec::new();
    for &halo in &protocol.halo_scenarios {
        for &host in &protocol.observer_host_scenarios {
            let mut evaluated = Vec::new();
            for (index, row) in ordered.iter().enumerate() {
                let truth = comoving_distance(row.redshift, planck2018::OMEGA_M, planck2018::H0);
                ensure!(
                    truth.is_finite() && truth > 0.0,
                    "Invalid catalog distance for {}",
                    row.source_id
                );
                let (prediction, prediction_error) =
                    match predict(row.dm_observed, row.dm_disk_ne2001, halo, host) {
                        Ok(prediction) => (Some(prediction), None),
                        Err(error) => (None, Some(error.to_string())),
                    };
                let signed = prediction
                    .as_ref()
                    .map(|prediction| prediction.distance_mpc - truth);
                ensure!(
                    signed.is_none_or(f64::is_finite),
                    "Nonfinite distance error for {}",
                    row.source_id
                );
                evaluated.push(EvaluatedRow {
                    raw: row.clone(),
                    partition: if index < protocol.calibration_count {
                        "calibration"
                    } else {
                        "heldout"
                    },
                    prediction,
                    prediction_error,
                    catalog_distance_mpc: truth,
                    signed_error_mpc: signed,
                    absolute_error_mpc: signed.map(f64::abs),
                    calibration_score_kind: if signed.is_some() {
                        "finite_absolute_error"
                    } else {
                        "positive_infinity_prediction_failure"
                    },
                });
            }
            let calibration_rows = &evaluated[..protocol.calibration_count];
            let heldout_rows = &evaluated[protocol.calibration_count..];
            let scores: Vec<f64> = calibration_rows
                .iter()
                .map(|row| row.absolute_error_mpc.unwrap_or(f64::INFINITY))
                .collect();
            let mut assessments = Vec::new();
            for &coverage in &protocol.coverages {
                let calibration = calibrate_absolute_errors(&scores, coverage)?;
                let quantile = match calibration.quantile() {
                    ConformalQuantile::Finite { radius_mpc } => {
                        json!({"kind":"finite", "radius_mpc":radius_mpc})
                    }
                    ConformalQuantile::Unbounded => json!({"kind":"unbounded", "radius_mpc":null}),
                };
                let mut intervals = Vec::new();
                let mut covered = 0;
                for row in heldout_rows {
                    let (is_covered, interval) = if let Some(prediction) = &row.prediction {
                        match calibration.interval(prediction.distance_mpc) {
                            Ok(DistancePredictionInterval::Finite {
                                lower_mpc,
                                upper_mpc,
                            }) => (
                                row.catalog_distance_mpc >= lower_mpc
                                    && row.catalog_distance_mpc <= upper_mpc,
                                json!({"kind":"finite", "lower_mpc":lower_mpc, "upper_mpc":upper_mpc, "width_mpc":upper_mpc-lower_mpc}),
                            ),
                            Ok(DistancePredictionInterval::Unbounded) => (
                                true,
                                json!({"kind":"unbounded", "lower_mpc":0.0, "upper_mpc":null, "width_mpc":null, "precision_established":false}),
                            ),
                            Err(error) => (
                                false,
                                json!({"kind":"interval_failure", "error":error.to_string()}),
                            ),
                        }
                    } else {
                        (
                            false,
                            json!({"kind":"prediction_failure", "error":row.prediction_error}),
                        )
                    };
                    covered += u64::from(is_covered);
                    intervals.push(json!({"source_id":row.raw.source_id, "covered":is_covered, "interval":interval}));
                }
                let assessment = assess_coverage(covered, heldout_rows.len() as u64, coverage)?;
                let primary = halo == protocol.primary_halo_dm
                    && host == protocol.primary_observer_host_dm
                    && coverage == protocol.primary_coverage;
                let inference = if primary {
                    if assessment.undercoverage_p_value < protocol.undercoverage_alpha {
                        "undercoverage_detected_under_independent_common_coverage_model"
                    } else {
                        "undercoverage_not_detected"
                    }
                } else {
                    "descriptive_sensitivity_only"
                };
                assessments.push(json!({
                    "nominal_coverage":coverage, "primary":primary,
                    "calibration":{"count":calibration.count(),"rank":calibration.rank(),"quantile":quantile,"prediction_failures":calibration_rows.iter().filter(|row| row.prediction.is_none()).count()},
                    "heldout":{"covered":assessment.covered,"total":assessment.total,"empirical_coverage":assessment.empirical_coverage,"prediction_failures":heldout_rows.iter().filter(|row| row.prediction.is_none()).count(),"clopper_pearson_two_sided_95":[assessment.clopper_pearson_lower,assessment.clopper_pearson_upper],"binomial_lower_tail_p_value":assessment.undercoverage_p_value},
                    "inference":inference, "intervals":intervals,
                    "precision_boundary":"An unbounded quantile establishes no finite calibration precision; neighborhood scales are descriptive rather than usefulness thresholds."
                }));
            }
            let scales: Vec<Value> = protocol.historical_neighborhood_radii_mpc.iter().map(|radius| json!({
                "radius_mpc":radius,
                "calibration_total":calibration_rows.len(),
                "heldout_total":heldout_rows.len(),
                "calibration_finite_errors_exceeding_radius":calibration_rows.iter().filter(|row| row.absolute_error_mpc.is_some_and(|error|error > *radius)).count(),
                "heldout_finite_errors_exceeding_radius":heldout_rows.iter().filter(|row| row.absolute_error_mpc.is_some_and(|error|error > *radius)).count(),
                "calibration_prediction_failures":calibration_rows.iter().filter(|row|row.prediction.is_none()).count(),
                "heldout_prediction_failures":heldout_rows.iter().filter(|row|row.prediction.is_none()).count()
            })).collect();
            scenarios.push(json!({"halo_dm":halo,"observer_host_dm":host,"summed_foreground_dm":halo+host,"rows":evaluated,"interval_assessments":assessments,"historical_scale_comparisons":scales}));
        }
    }
    Ok(scenarios)
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = args
        .repo_root
        .unwrap_or(std::env::current_dir()?)
        .canonicalize()?;
    let protocol_bytes =
        fs::read(resolve(&root, &args.protocol)).context("Read frozen protocol")?;
    verify_digest(&protocol_bytes, &args.expected_protocol_sha256, "protocol")?;
    let protocol_text = std::str::from_utf8(&protocol_bytes)?;
    let protocol: Protocol = toml::from_str(protocol_text)?;
    validate_protocol(&protocol)?;
    let full_protocol: toml::Value = toml::from_str(protocol_text)?;
    let grouping_bytes = fs::read(resolve(&root, Path::new(&protocol.source_grouping_path)))?;
    verify_digest(
        &grouping_bytes,
        &protocol.source_grouping_sha256,
        "source grouping",
    )?;
    let grouping: toml::Value = toml::from_str(std::str::from_utf8(&grouping_bytes)?)?;
    ensure!(
        grouping.get("cohort_sha256").and_then(toml::Value::as_str)
            == Some(protocol.data_sha256.as_str()),
        "Grouping cohort identity mismatch"
    );
    let data_bytes = fs::read(resolve(&root, Path::new(&protocol.data_path)))?;
    let rows = read_rows(&data_bytes, &protocol)?;
    let ordered = ordered_rows(&rows, &protocol);
    let scenarios = evaluate(&rows, &protocol)?;
    let output = json!({
        "schema_version":1,
        "protocol":full_protocol,
        "protocol_sha256":digest(&protocol_bytes),
        "cohort_sha256":digest(&data_bytes),
        "source_grouping_sha256":digest(&grouping_bytes),
        "generator_source_sha256":digest(include_bytes!("frb_distance_calibration.rs")),
        "calibration_source_ids":ordered[..protocol.calibration_count].iter().map(|row| &row.source_id).collect::<Vec<_>>(),
        "heldout_source_ids":ordered[protocol.calibration_count..].iter().map(|row| &row.source_id).collect::<Vec<_>>(),
        "model":{"omega_m":planck2018::OMEGA_M,"omega_b":planck2018::OMEGA_B,"h0":planck2018::H0},
        "signed_error_convention":"Predicted comoving distance minus catalog-redshift comoving distance",
        "scenarios":scenarios,
        "boundary":"Selected measured-redshift cohort audit; exchangeability and host-label validity remain assumptions. CHIME transport and hierarchy specificity are unestablished."
    });
    let destination = resolve(&root, &args.output);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .context("Create fresh output artifact")?;
    serde_json::to_writer_pretty(&mut file, &output)?;
    file.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Protocol, Vec<InputRow>) {
        let rows: Vec<_> = (0..21)
            .map(|index| InputRow {
                source_id: format!("synthetic-{index:02}"),
                dm_observed: 300.0 + index as f64 * 10.0,
                dm_error: 0.2,
                dm_disk_ne2001: 30.0,
                redshift: 0.1 + index as f64 * 0.01,
                telescope: "synthetic".into(),
                frequency_mhz: 1000.0,
                redshift_kind: "synthetic".into(),
                reference: "analytic_fixture".into(),
                discovery_utc_as_reported: "synthetic".into(),
            })
            .collect();
        let protocol = Protocol {
            schema_version: 1,
            data_path: String::new(),
            data_sha256: String::new(),
            source_ids: rows.iter().map(|row| row.source_id.clone()).collect(),
            split_salt: "synthetic-only".into(),
            calibration_count: 11,
            coverages: vec![0.8, 0.9, 0.95],
            primary_coverage: 0.9,
            primary_halo_dm: 50.0,
            primary_observer_host_dm: 50.0,
            halo_scenarios: vec![0.0, 50.0, 100.0],
            observer_host_scenarios: vec![0.0, 50.0, 100.0],
            historical_neighborhood_radii_mpc: vec![50.0, 100.0, 200.0, 500.0],
            undercoverage_alpha: 0.05,
            source_grouping_path: String::new(),
            source_grouping_sha256: String::new(),
        };
        (protocol, rows)
    }

    #[test]
    fn hashes_membership_duplicates_and_nonfinite_values_fail_closed() {
        let (mut protocol, rows) = fixture();
        let mut writer = csv::Writer::from_writer(Vec::new());
        for row in &rows {
            writer.serialize(row).unwrap();
        }
        let mut body = writer.into_inner().unwrap();
        protocol.data_sha256 = digest(&body);
        assert_eq!(read_rows(&body, &protocol).unwrap().len(), 21);
        body.push(b' ');
        assert!(read_rows(&body, &protocol).is_err());
        assert!(verify_digest(b"changed", &digest(b"original"), "fixture").is_err());
        assert!(read_rows(b"untrusted", &protocol).is_err());
        let mut wrong = rows.clone();
        wrong[0].source_id = "unexpected".into();
        assert!(validate_rows(&wrong, &protocol).is_err());
        wrong[0].source_id = wrong[1].source_id.clone();
        assert!(validate_rows(&wrong, &protocol).is_err());
        let mut wrong = rows;
        wrong[0].redshift = f64::NAN;
        assert!(validate_rows(&wrong, &protocol).is_err());
    }

    #[test]
    fn source_order_preserves_every_result() {
        let (protocol, mut rows) = fixture();
        let original = evaluate(&rows, &protocol).unwrap();
        assert_eq!(
            original
                .iter()
                .flat_map(|scenario| scenario["interval_assessments"].as_array().unwrap())
                .filter(|assessment| assessment["primary"] == true)
                .count(),
            1
        );
        rows.reverse();
        assert_eq!(evaluate(&rows, &protocol).unwrap(), original);
    }

    #[test]
    fn heldout_label_mutation_preserves_predictions_quantiles_and_membership() {
        let (protocol, rows) = fixture();
        let original = evaluate(&rows, &protocol).unwrap();
        let mut mutated = ordered_rows(&rows, &protocol);
        for row in &mut mutated[11..] {
            row.redshift *= 2.0;
        }
        let changed = evaluate(&mutated, &protocol).unwrap();
        assert_ne!(
            original[0]["rows"][11]["catalog_distance_mpc"],
            changed[0]["rows"][11]["catalog_distance_mpc"]
        );
        for (before, after) in original.iter().zip(&changed) {
            for (left, right) in before["rows"]
                .as_array()
                .unwrap()
                .iter()
                .zip(after["rows"].as_array().unwrap())
            {
                for field in ["prediction", "partition", "prediction_error"] {
                    assert_eq!(left[field], right[field]);
                }
                assert_eq!(left["raw"]["source_id"], right["raw"]["source_id"]);
            }
            for (left, right) in before["interval_assessments"]
                .as_array()
                .unwrap()
                .iter()
                .zip(after["interval_assessments"].as_array().unwrap())
            {
                assert_eq!(left["calibration"], right["calibration"]);
            }
        }
    }

    #[test]
    fn failures_retain_denominators_and_unbounded_intervals_remain_explicit() {
        let (protocol, rows) = fixture();
        let mut ordered = ordered_rows(&rows, &protocol);
        ordered[0].dm_observed = 1.0;
        ordered[11].dm_observed = 1.0;
        let results = evaluate(&ordered, &protocol).unwrap();
        for scenario in results {
            for (index, rank) in [10, 11, 12].into_iter().enumerate() {
                assert_eq!(
                    scenario["interval_assessments"][index]["calibration"]["rank"],
                    rank
                );
            }
            let assessment = &scenario["interval_assessments"][1];
            assert_eq!(assessment["calibration"]["count"], 11);
            assert_eq!(assessment["calibration"]["prediction_failures"], 1);
            assert_eq!(assessment["heldout"]["total"], 10);
            assert_eq!(assessment["heldout"]["prediction_failures"], 1);
            assert_eq!(assessment["heldout"]["covered"], 9);
            assert_eq!(assessment["calibration"]["quantile"]["kind"], "unbounded");
        }
        assert!(predict(130.0, 30.0, 50.0, 50.0).is_err());
    }

    #[test]
    fn equal_foreground_totals_preserve_predictions_and_calibration() {
        let (protocol, rows) = fixture();
        let results = evaluate(&rows, &protocol).unwrap();
        for left in &results {
            for right in &results {
                if left["summed_foreground_dm"] == right["summed_foreground_dm"] {
                    assert_eq!(left["rows"], right["rows"]);
                    for (left_assessment, right_assessment) in left["interval_assessments"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(right["interval_assessments"].as_array().unwrap())
                    {
                        assert_eq!(
                            left_assessment["calibration"],
                            right_assessment["calibration"]
                        );
                    }
                }
            }
        }
    }
}
