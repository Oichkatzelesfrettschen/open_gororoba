//! Verify bounded insight-reference coverage against a canonical snapshot.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs, path::PathBuf};

const EXPECTED_INSIGHTS: [&str; 8] = [
    "I-001", "I-002", "I-094", "I-095", "I-096", "I-207", "I-209", "I-212",
];

#[derive(Debug, Parser)]
#[command(about = "Verify eight insight records and their canonical claim-reference coverage")]
struct Args {
    #[arg(long)]
    findings: PathBuf,
    #[arg(long)]
    snapshot: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Findings {
    records: Vec<InsightFinding>,
}

#[derive(Debug, Deserialize)]
struct InsightFinding {
    id: String,
    disposition: String,
    preserve: String,
    falsifier: String,
    successor: String,
    independent_test: String,
    residual: String,
    atomic_propositions: Vec<Proposition>,
    references: Vec<ReferenceRole>,
}

#[derive(Debug, Deserialize)]
struct Proposition {
    id: String,
    proposition: String,
    assessment: String,
    evidence: String,
}

#[derive(Debug, Deserialize)]
struct ReferenceRole {
    claim_id: String,
    role: String,
    predicate: String,
    boundary: String,
}

#[derive(Debug, Deserialize)]
struct Snapshot {
    insights: Vec<SnapshotInsight>,
    references: Vec<SnapshotReference>,
}

#[derive(Debug, Deserialize)]
struct SnapshotInsight {
    id: String,
    claim_refs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SnapshotReference {
    insight_id: String,
    claim_id: String,
    claim_status: String,
    statement: String,
}

#[derive(Debug, Serialize, PartialEq)]
struct Coverage {
    status: &'static str,
    insights: usize,
    reference_pairs: usize,
    distinct_claims: usize,
}

fn require_text(value: &str, owner: &str, field: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "{owner}: empty {field}");
    Ok(())
}

fn validate(findings: &Findings, snapshot: &Snapshot) -> Result<Coverage> {
    let expected: BTreeSet<&str> = EXPECTED_INSIGHTS.into_iter().collect();
    let finding_ids: BTreeSet<&str> = findings
        .records
        .iter()
        .map(|record| record.id.as_str())
        .collect();
    let snapshot_ids: BTreeSet<&str> = snapshot
        .insights
        .iter()
        .map(|record| record.id.as_str())
        .collect();
    ensure!(
        findings.records.len() == 8 && finding_ids == expected,
        "Findings must contain exactly the eight pilot insight IDs without duplicates"
    );
    ensure!(
        snapshot.insights.len() == 8 && snapshot_ids == expected,
        "Snapshot must contain exactly the eight pilot insight IDs without duplicates"
    );

    let mut actual_pairs = BTreeSet::new();
    for record in &findings.records {
        for (field, value) in [
            ("disposition", &record.disposition),
            ("preserve", &record.preserve),
            ("falsifier", &record.falsifier),
            ("successor", &record.successor),
            ("independent_test", &record.independent_test),
            ("residual", &record.residual),
        ] {
            require_text(value, &record.id, field)?;
        }
        ensure!(
            !record.atomic_propositions.is_empty(),
            "{}: missing atomic propositions",
            record.id
        );
        let mut proposition_ids = BTreeSet::new();
        for proposition in &record.atomic_propositions {
            for (field, value) in [
                ("proposition id", &proposition.id),
                ("proposition", &proposition.proposition),
                ("assessment", &proposition.assessment),
                ("evidence", &proposition.evidence),
            ] {
                require_text(value, &record.id, field)?;
            }
            ensure!(
                proposition_ids.insert(&proposition.id),
                "{}: duplicate proposition {}",
                record.id,
                proposition.id
            );
        }
        for reference in &record.references {
            for (field, value) in [
                ("claim_id", &reference.claim_id),
                ("role", &reference.role),
                ("predicate", &reference.predicate),
                ("boundary", &reference.boundary),
            ] {
                require_text(value, &record.id, field)?;
            }
            ensure!(
                actual_pairs.insert((record.id.as_str(), reference.claim_id.as_str())),
                "{}: duplicate reference {}",
                record.id,
                reference.claim_id
            );
        }
    }

    let mut declared_pairs = BTreeSet::new();
    for insight in &snapshot.insights {
        for claim in &insight.claim_refs {
            require_text(claim, &insight.id, "claim_refs")?;
            ensure!(
                declared_pairs.insert((insight.id.as_str(), claim.as_str())),
                "{}: duplicate declared reference {claim}",
                insight.id
            );
        }
    }
    let mut canonical_pairs = BTreeSet::new();
    for reference in &snapshot.references {
        ensure!(
            expected.contains(reference.insight_id.as_str()),
            "Unknown snapshot insight {}",
            reference.insight_id
        );
        require_text(&reference.claim_id, &reference.insight_id, "claim_id")?;
        require_text(&reference.claim_status, &reference.claim_id, "claim_status")?;
        require_text(&reference.statement, &reference.claim_id, "statement")?;
        ensure!(
            canonical_pairs.insert((reference.insight_id.as_str(), reference.claim_id.as_str())),
            "Duplicate snapshot reference {}/{}",
            reference.insight_id,
            reference.claim_id
        );
    }
    ensure!(
        declared_pairs == canonical_pairs,
        "Snapshot reference rows differ from declared insight claim_refs"
    );
    ensure!(
        actual_pairs == canonical_pairs,
        "Findings reference pairs differ from canonical snapshot"
    );
    ensure!(
        actual_pairs.len() == 37,
        "Expected exactly 37 reference pairs"
    );
    let distinct_claims = actual_pairs
        .iter()
        .map(|(_, claim)| *claim)
        .collect::<BTreeSet<_>>()
        .len();
    ensure!(distinct_claims == 26, "Expected exactly 26 distinct claims");
    Ok(Coverage {
        status: "pass",
        insights: 8,
        reference_pairs: actual_pairs.len(),
        distinct_claims,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let findings: Findings = serde_json::from_slice(
        &fs::read(&args.findings).with_context(|| format!("Read {}", args.findings.display()))?,
    )
    .context("Parse findings")?;
    let snapshot: Snapshot = serde_json::from_slice(
        &fs::read(&args.snapshot).with_context(|| format!("Read {}", args.snapshot.display()))?,
    )
    .context("Parse snapshot")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&validate(&findings, &snapshot)?)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn fixture() -> (Value, Value) {
        let mut records = Vec::new();
        let mut insights = Vec::new();
        let mut references = Vec::new();
        for (index, insight) in EXPECTED_INSIGHTS.iter().enumerate() {
            let claims: Vec<String> = (index..37)
                .step_by(8)
                .map(|number| format!("C-{:03}", number % 26 + 1))
                .collect();
            records.push(json!({
                "id": insight, "disposition": "retain", "preserve": "bounded result",
                "falsifier": "matched control", "successor": "narrow result", "independent_test": "held-out comparison", "residual": "source admission",
                "atomic_propositions": [{"id":"representation", "proposition":"rank statistic", "assessment":"bounded", "evidence":"producer source"}],
                "references": claims.iter().map(|claim| json!({"claim_id":claim,"role":"supports_with_limits","predicate":"representation","boundary":"source inspection"})).collect::<Vec<_>>()
            }));
            insights.push(json!({"id":insight,"claim_refs":claims}));
            references.extend(claims.iter().map(|claim| json!({"insight_id":insight,"claim_id":claim,"claim_status":"Verified","statement":"bounded statement"})));
        }
        (
            json!({"records":records}),
            json!({"insights":insights,"references":references}),
        )
    }

    fn check(findings: Value, snapshot: Value) -> Result<Coverage> {
        validate(
            &serde_json::from_value(findings)?,
            &serde_json::from_value(snapshot)?,
        )
    }

    #[test]
    fn accepts_exact_coverage_independent_of_order() {
        let (mut findings, mut snapshot) = fixture();
        findings["records"].as_array_mut().unwrap().reverse();
        snapshot["references"].as_array_mut().unwrap().reverse();
        let result = check(findings, snapshot).unwrap();
        assert_eq!(
            result,
            Coverage {
                status: "pass",
                insights: 8,
                reference_pairs: 37,
                distinct_claims: 26
            }
        );
    }

    #[test]
    fn rejects_missing_duplicate_and_substituted_references() {
        for mutation in ["missing", "duplicate", "substituted"] {
            let (mut findings, snapshot) = fixture();
            let references = findings["records"][0]["references"].as_array_mut().unwrap();
            match mutation {
                "missing" => {
                    references.pop();
                }
                "duplicate" => references.push(references[0].clone()),
                _ => references[0]["claim_id"] = json!("C-999"),
            }
            assert!(check(findings, snapshot).is_err(), "{mutation}");
        }
    }

    #[test]
    fn rejects_missing_duplicate_and_unknown_insights() {
        for mutation in ["missing", "duplicate", "unknown"] {
            let (mut findings, snapshot) = fixture();
            let records = findings["records"].as_array_mut().unwrap();
            match mutation {
                "missing" => {
                    records.pop();
                }
                "duplicate" => records[0] = records[1].clone(),
                _ => records[0]["id"] = json!("I-999"),
            }
            assert!(check(findings, snapshot).is_err(), "{mutation}");
        }
    }

    #[test]
    fn rejects_missing_and_blank_required_fields() {
        for field in [
            "disposition",
            "preserve",
            "falsifier",
            "successor",
            "independent_test",
            "residual",
        ] {
            for missing in [false, true] {
                let (mut findings, snapshot) = fixture();
                if missing {
                    findings["records"][0]
                        .as_object_mut()
                        .unwrap()
                        .remove(field);
                } else {
                    findings["records"][0][field] = json!(" \n\t");
                }
                assert!(
                    check(findings, snapshot).is_err(),
                    "{field}, missing={missing}"
                );
            }
        }
    }

    #[test]
    fn rejects_empty_or_duplicate_propositions_and_empty_roles() {
        for mutation in [
            "empty",
            "duplicate",
            "blank-id",
            "blank-evidence",
            "blank-role",
        ] {
            let (mut findings, snapshot) = fixture();
            match mutation {
                "empty" => findings["records"][0]["atomic_propositions"] = json!([]),
                "duplicate" => {
                    let propositions = findings["records"][0]["atomic_propositions"]
                        .as_array_mut()
                        .unwrap();
                    propositions.push(propositions[0].clone());
                }
                "blank-id" => findings["records"][0]["atomic_propositions"][0]["id"] = json!(" "),
                "blank-evidence" => {
                    findings["records"][0]["atomic_propositions"][0]["evidence"] = json!(" ")
                }
                _ => findings["records"][0]["references"][0]["role"] = json!(" "),
            }
            assert!(check(findings, snapshot).is_err(), "{mutation}");
        }
    }

    #[test]
    fn rejects_incomplete_inconsistent_or_duplicate_snapshot_rows() {
        for mutation in [
            "missing",
            "duplicate",
            "dangling",
            "unknown",
            "undeclared",
            "blank-status",
        ] {
            let (findings, mut snapshot) = fixture();
            match mutation {
                "missing" => {
                    snapshot["references"].as_array_mut().unwrap().pop();
                }
                "duplicate" => {
                    let references = snapshot["references"].as_array_mut().unwrap();
                    references.push(references[0].clone());
                }
                "dangling" => snapshot["references"][0]["claim_status"] = Value::Null,
                "unknown" => snapshot["references"][0]["insight_id"] = json!("I-999"),
                "undeclared" => snapshot["insights"][0]["claim_refs"] = json!([]),
                _ => snapshot["references"][0]["claim_status"] = json!(" "),
            }
            assert!(check(findings, snapshot).is_err(), "{mutation}");
        }
    }

    #[test]
    fn rejects_duplicate_json_fields() {
        assert!(serde_json::from_str::<Findings>(r#"{"records":[],"records":[]}"#).is_err());
        let (findings, _) = fixture();
        let record = findings["records"][0].to_string();
        let duplicate = record.replacen('{', r#"{"id":"I-001","#, 1);
        assert!(serde_json::from_str::<InsightFinding>(&duplicate).is_err());
    }
}
