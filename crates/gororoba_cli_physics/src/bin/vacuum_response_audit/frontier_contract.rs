//! Exact coverage and dependency checks for the vacuum integration ledger.
//! Declaration lookup measures source linkage; Rocq owns theorem execution.
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path},
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Contract {
    title: String,
    source_ledger: String,
    leaf_count: usize,
    scope: String,
    proof_boundary: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Leaf {
    id: String,
    disposition: String,
    formal: Vec<String>,
    implementation: Vec<String>,
    dependencies: Vec<String>,
    physical_validation: String,
    next_gate: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ledger {
    contract: Contract,
    leaf: Vec<Leaf>,
}

fn validate_reference(root: &Path, reference: &str, formal: bool) -> Result<(), String> {
    let (file, symbol) = reference
        .split_once("::")
        .map_or((reference, None), |(file, symbol)| (file, Some(symbol)));
    let path = Path::new(file);
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "reference escapes its relative evidence boundary: {reference}"
        ));
    }
    let metadata =
        fs::symlink_metadata(root.join(path)).map_err(|error| format!("{reference}: {error}"))?;
    if !metadata.is_file() {
        return Err(format!(
            "reference is not a regular evidence file: {reference}"
        ));
    }
    if formal
        && (!file.starts_with("proofs/theories/") || !file.ends_with(".v") || symbol.is_none())
    {
        return Err(format!(
            "formal reference requires a theory and declaration: {reference}"
        ));
    }
    if let Some(symbol) = symbol {
        if symbol.is_empty()
            || !symbol
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(format!("malformed declaration in {reference}"));
        }
        let source = fs::read_to_string(root.join(path)).map_err(|error| error.to_string())?;
        let prefixes: &[&str] = if formal {
            &["Theorem ", "Lemma ", "Corollary "]
        } else {
            &["fn "]
        };
        let exists = prefixes.iter().any(|prefix| {
            source
                .match_indices(&format!("{prefix}{symbol}"))
                .any(|(offset, matched)| {
                    source[offset + matched.len()..]
                        .chars()
                        .next()
                        .is_some_and(|character| {
                            character.is_whitespace() || matches!(character, ':' | '(' | '<')
                        })
                })
        });
        if !exists {
            return Err(format!("missing declaration: {reference}"));
        }
        if formal {
            let project = fs::read_to_string(root.join("proofs/_RocqProject"))
                .map_err(|error| error.to_string())?;
            if !project
                .lines()
                .any(|line| Some(line) == file.strip_prefix("proofs/"))
            {
                return Err(format!(
                    "formal source is outside the compiled corpus: {reference}"
                ));
            }
        }
    }
    Ok(())
}

fn visit<'a>(
    id: &'a str,
    rows: &BTreeMap<&'a str, &'a Leaf>,
    visiting: &mut BTreeSet<&'a str>,
    complete: &mut BTreeSet<&'a str>,
) -> Result<(), String> {
    if complete.contains(id) {
        return Ok(());
    }
    if !visiting.insert(id) {
        return Err(format!("cyclic integration dependency at {id}"));
    }
    let row = rows
        .get(id)
        .ok_or_else(|| format!("unknown integration dependency {id}"))?;
    for dependency in &row.dependencies {
        visit(dependency, rows, visiting, complete)?;
    }
    visiting.remove(id);
    complete.insert(id);
    Ok(())
}

fn validate(root: &Path, expected: &BTreeSet<String>, ledger: &Ledger) -> Result<(), String> {
    let contract = &ledger.contract;
    if contract.source_ledger != "plans/vacuum_energy_falsification.toml"
        || contract.leaf_count != expected.len()
        || [&contract.title, &contract.scope, &contract.proof_boundary]
            .iter()
            .any(|value| value.trim().is_empty())
    {
        return Err("integration contract denominator or boundary is inconsistent".to_owned());
    }
    let actual: BTreeSet<_> = ledger.leaf.iter().map(|leaf| leaf.id.clone()).collect();
    if actual != *expected || ledger.leaf.len() != actual.len() {
        return Err("integration ledger must cover each original leaf exactly once".to_owned());
    }
    let dispositions = [
        "formal_model",
        "formal_counterexample",
        "numerical_counterexample",
        "source_reconciled",
        "source_open",
        "measurement_open",
        "implementation_repaired",
        "model_composed_measurement_open",
        "analytic_counterexample_recorded",
        "definition_open",
        "source_reconciled_model_open",
    ];
    for leaf in &ledger.leaf {
        if !dispositions.contains(&leaf.disposition.as_str())
            || leaf.next_gate.trim().is_empty()
            || leaf.physical_validation != "unperformed"
            || leaf.implementation.is_empty()
        {
            return Err(format!(
                "{} lacks an admissible evidence disposition, implementation link or remaining gate",
                leaf.id
            ));
        }
        if leaf.disposition.starts_with("formal_") && leaf.formal.is_empty() {
            return Err(format!(
                "{} claims formal coverage without a declaration",
                leaf.id
            ));
        }
        for reference in &leaf.formal {
            validate_reference(root, reference, true)?;
        }
        for reference in &leaf.implementation {
            validate_reference(root, reference, false)?;
        }
        if leaf.dependencies.iter().collect::<BTreeSet<_>>().len() != leaf.dependencies.len() {
            return Err(format!("{} repeats a dependency", leaf.id));
        }
    }
    let rows: BTreeMap<_, _> = ledger
        .leaf
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect();
    let mut complete = BTreeSet::new();
    for id in rows.keys() {
        visit(id, &rows, &mut BTreeSet::new(), &mut complete)?;
    }
    Ok(())
}

fn fixture() -> (std::path::PathBuf, BTreeSet<String>, Ledger) {
    let root = repo_root::discover().expect("run inside the workspace or set GOROROBA_REPO_ROOT");
    let source: toml::Value = toml::from_str(
        &fs::read_to_string(root.join("plans/vacuum_energy_falsification.toml")).unwrap(),
    )
    .unwrap();
    let original_rows = source["leaf"].as_array().unwrap();
    let expected: BTreeSet<_> = original_rows
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        expected.len(),
        original_rows.len(),
        "original ledger IDs are unique"
    );
    let ledger = toml::from_str(
        &fs::read_to_string(root.join("plans/vacuum_integration_contract.toml")).unwrap(),
    )
    .unwrap();
    (root, expected, ledger)
}

#[test]
fn every_original_leaf_has_one_checked_integration_and_acyclic_dependencies() {
    let (root, expected, ledger) = fixture();
    validate(&root, &expected, &ledger).unwrap();
}

#[test]
fn missing_duplicate_or_invented_leaf_rejects() {
    let (root, expected, ledger) = fixture();
    let mut missing = ledger.clone();
    missing.leaf.pop();
    assert!(validate(&root, &expected, &missing).is_err());
    let mut duplicate = ledger.clone();
    duplicate.leaf.push(duplicate.leaf[0].clone());
    assert!(validate(&root, &expected, &duplicate).is_err());
    let mut invented = ledger.clone();
    invented.leaf[0].id = "unregistered-inference".to_owned();
    assert!(validate(&root, &expected, &invented).is_err());
}

#[test]
fn cycles_unknown_dependencies_and_unearned_physical_closure_reject() {
    let (root, expected, ledger) = fixture();
    let mut cycle = ledger.clone();
    let first_id = cycle.leaf[0].id.clone();
    cycle.leaf[0].dependencies.push(first_id);
    assert!(validate(&root, &expected, &cycle).is_err());
    let mut unknown = ledger.clone();
    unknown.leaf[0].dependencies.push("unknown".to_owned());
    assert!(validate(&root, &expected, &unknown).is_err());
    let mut closed = ledger.clone();
    closed.leaf[0].physical_validation = "established".to_owned();
    assert!(validate(&root, &expected, &closed).is_err());
    let mut empty = ledger.clone();
    empty.leaf[0].next_gate.clear();
    assert!(validate(&root, &expected, &empty).is_err());
}

#[test]
fn stale_declarations_path_escape_and_missing_proofs_reject() {
    let (root, expected, ledger) = fixture();
    let mut stale = ledger.clone();
    stale.leaf[0].formal[0].push_str("_invented");
    assert!(validate(&root, &expected, &stale).is_err());
    let mut unproved = ledger.clone();
    unproved.leaf[0].formal.clear();
    assert!(validate(&root, &expected, &unproved).is_err());
    assert!(validate_reference(&root, "../AGENTS.md", false).is_err());
    assert!(validate_reference(&root, "proofs/theories/Absent.v::invented", true).is_err());
}
