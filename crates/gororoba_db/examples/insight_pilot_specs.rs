//! Generate pinned insight admissions from retained review decisions and a baseline database.
use anyhow::{Context, Result, ensure};
use provenance_store::{InsightAdmissionSpec, ProvenanceStore};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::PathBuf};

fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        arguments.len() == 2,
        "usage: insight_pilot_specs REPO_ROOT BASELINE_DB"
    );
    let root = PathBuf::from(&arguments[0]);
    let store = ProvenanceStore::open_read_only(&PathBuf::from(&arguments[1]))?;
    let directory = root.join("data/output/audit/insight-canonical-admission");
    let review_path = "data/output/audit/insight-reference-role-pilot/findings.json";
    let review: Value = serde_json::from_slice(&fs::read(root.join(review_path))?)?;
    let corrections: Vec<Value> =
        serde_json::from_slice(&fs::read(directory.join("corrections.json"))?)?;
    let decisions: Value =
        serde_json::from_slice(&fs::read(directory.join("predicate-decisions.json"))?)?;
    let evidence: Value =
        serde_json::from_slice(&fs::read(directory.join("evidence-files.json"))?)?;
    let mut seen = BTreeSet::new();
    fs::create_dir_all(directory.join("specs"))?;
    for correction in corrections {
        let id = correction["id"].as_str().context("correction ID")?;
        ensure!(seen.insert(id.to_owned()), "duplicate insight");
        let (record_index, record) = review["records"]
            .as_array()
            .context("review records")?
            .iter()
            .enumerate()
            .find(|(_, record)| record["id"] == id)
            .context("reviewed insight")?;
        let expected = store.insight_admission_state(id)?;
        let mut revised = serde_json::to_value(&expected)?;
        for field in ["title", "status", "summary"] {
            revised[field] = correction[field].clone();
        }
        revised["status_note"] = correction["note"].clone();
        if correction.get("confidence").is_some() {
            revised["confidence"] = correction["confidence"].clone();
        }
        let mut predicates = Vec::new();
        for (predicate_index, predicate) in record["atomic_propositions"]
            .as_array()
            .context("review predicates")?
            .iter()
            .enumerate()
        {
            let predicate_id = predicate["id"].as_str().context("predicate ID")?;
            let decision = &decisions[id]["predicates"][predicate_id];
            ensure!(
                decision.as_array().is_some_and(|values| values.len() == 3),
                "missing decision {id}/{predicate_id}"
            );
            predicates.push(json!({"id":predicate_id,"proposition":predicate["proposition"],"kind":predicate.get("kind").and_then(Value::as_str).unwrap_or("asserted_proposition"),"evidence_layer":decision[0],"outcome":decision[1],"execution_status":decision[2],"boundary":predicate["assessment"],"evidence_anchors":[predicate["evidence"],format!("{review_path}#{id}/{predicate_id}")],"evidence_bindings":[{"path":review_path,"json_pointer":format!("/records/{record_index}/atomic_propositions/{predicate_index}")}]}));
        }
        let mut references = Vec::new();
        for reference in record["references"]
            .as_array()
            .context("review references")?
        {
            let claim = reference["claim_id"].as_str().context("claim ID")?;
            let decision = &decisions[id]["links"][claim];
            ensure!(
                decision.as_array().is_some_and(|values| values.len() == 2),
                "missing role {id}/{claim}"
            );
            references.push(json!({"claim_id":claim,"predicate_id":decision[0],"role":decision[1],"provenance":format!("Pilot role {}: {}",reference["role"].as_str().context("role")?,reference["predicate"].as_str().context("reference predicate")?),"boundary":reference["boundary"]}));
        }
        if id == "I-212" {
            let followups: Value = serde_json::from_slice(&fs::read(
                root.join("data/output/audit/insight-reference-role-pilot/followup-claims.json"),
            )?)?;
            for claim in [
                "C-1731", "C-1734", "C-1738", "C-1739", "C-1741", "C-1743", "C-1744", "C-1754",
            ] {
                let historical = claim == "C-1731";
                let boundary = if historical {
                    "Superseded historical follow-up; its uniqueness and matched-context wording is refined by the admitted descendants.".to_owned()
                } else {
                    followups
                        .as_array()
                        .context("followup rows")?
                        .iter()
                        .find(|row| row["id"] == claim)
                        .context("followup claim")?["statement"]
                        .as_str()
                        .context("followup statement")?
                        .to_owned()
                };
                revised["claim_refs"]
                    .as_array_mut()
                    .context("references")?
                    .push(json!(claim));
                references.push(json!({"claim_id":claim,"predicate_id":if claim=="C-1738" {"matched-context"} else {"unique-information"},"role":if historical {"historical_context"} else {"followup_result"},"provenance":"Canonical follow-up statement retained in followup-claims.json; the original eight-insight snapshot retains the C-1731 status-note reference.","boundary":boundary}));
            }
        }
        let spec: InsightAdmissionSpec = serde_json::from_value(
            json!({"admission_id":format!("insight-reference-predicate-admission:{id}"),"insight_id":id,"expected":expected,"revised":revised,"evidence_files":evidence,"predicates":predicates,"references":references,"disposition":record["disposition"],"falsifier":record["falsifier"],"successor":record["successor"],"residual":record["residual"]}),
        )?;
        spec.validate()?;
        fs::write(
            directory.join("specs").join(format!("{id}.toml")),
            toml::to_string_pretty(&spec)?,
        )?;
    }
    ensure!(seen.len() == 8, "requires exactly eight reviewed insights");
    println!("Generated eight pinned insight admission specifications.");
    Ok(())
}
