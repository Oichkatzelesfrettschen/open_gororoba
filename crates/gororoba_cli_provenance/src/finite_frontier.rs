//! Exact finite-set verification for the Casimir/optics discrimination frontier.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    path::Path,
};

pub const EXPECTED_FRONTIER_ID: &str = "casimir-optics-discrimination";
pub const EXPECTED_STABLE_KEY: &str = "frontier_id";

pub const EXPECTED_FRONTIER_IDS: [&str; 30] = [
    "bounded-nuisance-profile-distance",
    "calibrated-efficient-information",
    "casimir-optics-rust-audit-producer",
    "ci-owned-broad-validation-resource-boundary",
    "classical-commutator-loop-control",
    "claim-experiment-scope-reconciliation",
    "common-cap-finite-temperature-contrast",
    "declared-frontier-denominator-proof",
    "derived-feature-joint-covariance",
    "discrimination-error-margin",
    "gold-specimen-source-admission",
    "hnls-operator-span-scope",
    "lifshitz-planar-polar-normalization",
    "lifshitz-sphere-pfa-energy-conversion",
    "material-derivation-evidence-classes",
    "material-measurement-quantity-semantics",
    "material-state-specimen-identity",
    "material-typed-missingness-featurizer",
    "materials-lifshitz-matsubara-prefactor",
    "matched-active-sham-null-identity",
    "mineral-metadata-sentinel-elimination",
    "multilayer-au-oxide-pressure-replay",
    "optics-finite-film-state-meter",
    "optics-passive-fourier-convention",
    "primary-source-intake",
    "physical-calibration-priority-replay",
    "physical-nuisance-jacobian-replay",
    "quantum-memory-complete-positivity-gate",
    "selected-schedule-independent-monte-carlo",
    "uv-background-model-sensitivity",
];

pub const EXPECTED_OPEN_FRONTIER_IDS: [&str; 2] = [
    "declared-frontier-denominator-proof",
    "selected-schedule-independent-monte-carlo",
];

#[derive(Clone, Debug, Deserialize)]
struct FrontierDocument {
    frontier: FrontierHeader,
    row: Vec<FrontierRow>,
}

#[derive(Clone, Debug, Deserialize)]
struct FrontierHeader {
    id: String,
    stable_key: String,
    declared_rows: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct FrontierRow {
    frontier_id: String,
    completion_state: String,
    #[serde(default)]
    ordering_dependencies: Vec<String>,
    #[serde(default)]
    required_generator: String,
    #[serde(default)]
    next_action: Option<String>,
    #[serde(default)]
    completion_witness: Option<String>,
    #[serde(default)]
    completion_verifier: Option<String>,
}

impl FrontierRow {
    fn residual_action(&self) -> Option<&str> {
        self.next_action
            .as_deref()
            .filter(|action| !action.trim().is_empty())
            .or_else(|| {
                (!self.required_generator.trim().is_empty())
                    .then_some(self.required_generator.as_str())
            })
    }

    fn has_completion_witness(&self) -> bool {
        self.completion_witness
            .as_deref()
            .is_some_and(|witness| !witness.trim().is_empty())
    }

    fn has_completion_verifier(&self) -> bool {
        self.completion_verifier
            .as_deref()
            .is_some_and(|verifier| !verifier.trim().is_empty())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ResidualAction {
    pub frontier_id: String,
    pub next_action: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FiniteFrontierReport {
    pub frontier_id: String,
    pub stable_key: String,
    pub declared_count: usize,
    pub closed_count: usize,
    pub open_count: usize,
    pub declared_keys: Vec<String>,
    pub closed_keys: Vec<String>,
    pub open_keys: Vec<String>,
    pub residual_actions: Vec<ResidualAction>,
}

impl FiniteFrontierReport {
    pub fn render_text(&self) -> String {
        let mut output = format!(
            "frontier={} stable_key={} declared={} closed={} open={}\n",
            self.frontier_id,
            self.stable_key,
            self.declared_count,
            self.closed_count,
            self.open_count
        );
        output.push_str(&format!("declared_keys={}\n", self.declared_keys.join(",")));
        output.push_str(&format!("closed_keys={}\n", self.closed_keys.join(",")));
        output.push_str(&format!("open_keys={}\n", self.open_keys.join(",")));
        output.push_str("residual_actions:\n");
        for residual in &self.residual_actions {
            output.push_str(&format!(
                "- {}: {}\n",
                residual.frontier_id, residual.next_action
            ));
        }
        output
    }
}

pub fn verify_finite_frontier_path(path: &Path) -> Result<FiniteFrontierReport> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("read finite frontier {}", path.display()))?;
    verify_finite_frontier_source(&source)
        .with_context(|| format!("verify finite frontier {}", path.display()))
}

pub fn verify_finite_frontier_source(source: &str) -> Result<FiniteFrontierReport> {
    let document: FrontierDocument =
        toml::from_str(source).context("parse finite frontier TOML")?;
    verify_document(&document)
}

fn verify_document(document: &FrontierDocument) -> Result<FiniteFrontierReport> {
    let expected_ids: BTreeSet<&str> = EXPECTED_FRONTIER_IDS.into_iter().collect();
    let expected_open_ids: BTreeSet<&str> = EXPECTED_OPEN_FRONTIER_IDS.into_iter().collect();
    let mut diagnostics = Vec::new();

    if document.frontier.id != EXPECTED_FRONTIER_ID {
        diagnostics.push(format!(
            "frontier id {:?} does not equal {:?}",
            document.frontier.id, EXPECTED_FRONTIER_ID
        ));
    }
    if document.frontier.stable_key != EXPECTED_STABLE_KEY {
        diagnostics.push(format!(
            "stable key {:?} does not equal {:?}",
            document.frontier.stable_key, EXPECTED_STABLE_KEY
        ));
    }
    if document.frontier.declared_rows != EXPECTED_FRONTIER_IDS.len() {
        diagnostics.push(format!(
            "declared count {} does not equal compiled denominator {}",
            document.frontier.declared_rows,
            EXPECTED_FRONTIER_IDS.len()
        ));
    }
    if document.row.len() != document.frontier.declared_rows {
        diagnostics.push(format!(
            "declared count {} does not equal parsed row count {}",
            document.frontier.declared_rows,
            document.row.len()
        ));
    }

    let mut observed_ids = BTreeSet::new();
    let mut duplicate_ids = BTreeSet::new();
    for row in &document.row {
        if !observed_ids.insert(row.frontier_id.as_str()) {
            duplicate_ids.insert(row.frontier_id.as_str());
        }
    }
    if !duplicate_ids.is_empty() {
        diagnostics.push(format!(
            "duplicate frontier IDs: {}",
            join_set(&duplicate_ids)
        ));
    }

    let missing_ids: BTreeSet<_> = expected_ids.difference(&observed_ids).copied().collect();
    let unexpected_ids: BTreeSet<_> = observed_ids.difference(&expected_ids).copied().collect();
    if !missing_ids.is_empty() {
        diagnostics.push(format!("missing frontier IDs: {}", join_set(&missing_ids)));
    }
    if !unexpected_ids.is_empty() {
        diagnostics.push(format!(
            "unexpected frontier IDs: {}",
            join_set(&unexpected_ids)
        ));
    }

    let mut rows_by_id = BTreeMap::new();
    for row in &document.row {
        rows_by_id.entry(row.frontier_id.as_str()).or_insert(row);
        let expected_state = if expected_open_ids.contains(row.frontier_id.as_str()) {
            "open"
        } else {
            "closed"
        };
        if expected_ids.contains(row.frontier_id.as_str())
            && row.completion_state != expected_state
        {
            diagnostics.push(format!(
                "row {} has state {:?}, expected {:?}",
                row.frontier_id, row.completion_state, expected_state
            ));
        }
        match row.completion_state.as_str() {
            "closed" => {
                if !row.has_completion_witness() {
                    diagnostics.push(format!(
                        "closed row {} lacks completion_witness",
                        row.frontier_id
                    ));
                }
                if !row.has_completion_verifier() {
                    diagnostics.push(format!(
                        "closed row {} lacks completion_verifier",
                        row.frontier_id
                    ));
                }
            }
            "open" => {
                if row.residual_action().is_none() {
                    diagnostics.push(format!(
                        "open row {} lacks a next action or required generator",
                        row.frontier_id
                    ));
                }
            }
            invalid => diagnostics.push(format!(
                "row {} has invalid completion state {:?}",
                row.frontier_id, invalid
            )),
        }

        let mut dependencies = BTreeSet::new();
        for dependency in &row.ordering_dependencies {
            if !dependencies.insert(dependency.as_str()) {
                diagnostics.push(format!(
                    "row {} repeats dependency {}",
                    row.frontier_id, dependency
                ));
            }
            if !expected_ids.contains(dependency.as_str()) {
                diagnostics.push(format!(
                    "row {} references unknown dependency {}",
                    row.frontier_id, dependency
                ));
            }
        }
    }

    if missing_ids.is_empty()
        && unexpected_ids.is_empty()
        && duplicate_ids.is_empty()
        && let Some(cycle_keys) = dependency_cycle_keys(&rows_by_id)
    {
        diagnostics.push(format!(
            "dependency cycle includes: {}",
            cycle_keys.join(",")
        ));
    }

    if !diagnostics.is_empty() {
        bail!(
            "finite frontier validation failed:\n- {}",
            diagnostics.join("\n- ")
        );
    }

    let mut closed_keys = Vec::new();
    let mut open_keys = Vec::new();
    let mut residual_actions = Vec::new();
    for row in rows_by_id.values() {
        match row.completion_state.as_str() {
            "closed" => closed_keys.push(row.frontier_id.clone()),
            "open" => {
                open_keys.push(row.frontier_id.clone());
                residual_actions.push(ResidualAction {
                    frontier_id: row.frontier_id.clone(),
                    next_action: row
                        .residual_action()
                        .expect("validated open row has residual action")
                        .to_owned(),
                });
            }
            _ => unreachable!("completion state validated above"),
        }
    }

    Ok(FiniteFrontierReport {
        frontier_id: document.frontier.id.clone(),
        stable_key: document.frontier.stable_key.clone(),
        declared_count: document.frontier.declared_rows,
        closed_count: closed_keys.len(),
        open_count: open_keys.len(),
        declared_keys: rows_by_id.keys().map(|key| (*key).to_owned()).collect(),
        closed_keys,
        open_keys,
        residual_actions,
    })
}

fn dependency_cycle_keys(rows_by_id: &BTreeMap<&str, &FrontierRow>) -> Option<Vec<String>> {
    let mut remaining_dependencies: BTreeMap<&str, usize> = rows_by_id
        .iter()
        .map(|(frontier_id, row)| (*frontier_id, row.ordering_dependencies.len()))
        .collect();
    let mut dependents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (frontier_id, row) in rows_by_id {
        for dependency in &row.ordering_dependencies {
            dependents
                .entry(dependency.as_str())
                .or_default()
                .push(frontier_id);
        }
    }

    let mut ready: VecDeque<&str> = remaining_dependencies
        .iter()
        .filter_map(|(frontier_id, count)| (*count == 0).then_some(*frontier_id))
        .collect();
    while let Some(frontier_id) = ready.pop_front() {
        if let Some(rows) = dependents.get(frontier_id) {
            for dependent in rows {
                let count = remaining_dependencies
                    .get_mut(dependent)
                    .expect("dependent belongs to exact denominator");
                *count -= 1;
                if *count == 0 {
                    ready.push_back(dependent);
                }
            }
        }
    }

    let cycle_keys: Vec<String> = remaining_dependencies
        .into_iter()
        .filter_map(|(frontier_id, count)| (count > 0).then_some(frontier_id.to_owned()))
        .collect();
    (!cycle_keys.is_empty()).then_some(cycle_keys)
}

fn join_set<T>(values: &BTreeSet<T>) -> String
where
    T: AsRef<str> + Ord,
{
    values
        .iter()
        .map(|value| value.as_ref())
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline_document() -> FrontierDocument {
        FrontierDocument {
            frontier: FrontierHeader {
                id: EXPECTED_FRONTIER_ID.to_owned(),
                stable_key: EXPECTED_STABLE_KEY.to_owned(),
                declared_rows: EXPECTED_FRONTIER_IDS.len(),
            },
            row: EXPECTED_FRONTIER_IDS
                .iter()
                .map(|frontier_id| {
                    let is_open = EXPECTED_OPEN_FRONTIER_IDS.contains(frontier_id);
                    FrontierRow {
                        frontier_id: (*frontier_id).to_owned(),
                        completion_state: if is_open { "open" } else { "closed" }.to_owned(),
                        ordering_dependencies: Vec::new(),
                        required_generator: format!("verify {frontier_id}"),
                        next_action: None,
                        completion_witness: (!is_open).then(|| format!("witness/{frontier_id}")),
                        completion_verifier: (!is_open).then(|| format!("verify {frontier_id}")),
                    }
                })
                .collect(),
        }
    }

    fn error_text(document: &FrontierDocument) -> String {
        verify_document(document).unwrap_err().to_string()
    }

    #[test]
    fn accepts_exact_denominator_and_reports_bounded_partitions() {
        let report = verify_document(&baseline_document()).unwrap();
        assert_eq!(report.declared_count, EXPECTED_FRONTIER_IDS.len());
        assert_eq!(
            report.closed_count,
            EXPECTED_FRONTIER_IDS.len() - EXPECTED_OPEN_FRONTIER_IDS.len()
        );
        assert_eq!(report.open_count, EXPECTED_OPEN_FRONTIER_IDS.len());
        assert_eq!(report.declared_keys.len(), EXPECTED_FRONTIER_IDS.len());
        assert_eq!(report.residual_actions.len(), EXPECTED_OPEN_FRONTIER_IDS.len());
    }

    #[test]
    fn rejects_declared_count_mutation() {
        let mut document = baseline_document();
        document.frontier.declared_rows -= 1;
        let error = error_text(&document);
        assert!(error.contains(&format!(
            "declared count {} does not equal compiled denominator {}",
            EXPECTED_FRONTIER_IDS.len() - 1,
            EXPECTED_FRONTIER_IDS.len()
        )));
        assert!(error.contains(&format!(
            "declared count {} does not equal parsed row count {}",
            EXPECTED_FRONTIER_IDS.len() - 1,
            EXPECTED_FRONTIER_IDS.len()
        )));
    }

    #[test]
    fn rejects_duplicate_and_missing_key_mutation() {
        let mut document = baseline_document();
        let missing = document.row[1].frontier_id.clone();
        document.row[1].frontier_id = document.row[0].frontier_id.clone();
        let error = error_text(&document);
        assert!(error.contains("duplicate frontier IDs"));
        assert!(error.contains(&format!("missing frontier IDs: {missing}")));
    }

    #[test]
    fn rejects_missing_row_mutation() {
        let mut document = baseline_document();
        let missing = document.row.pop().unwrap().frontier_id;
        let error = error_text(&document);
        assert!(error.contains(&format!(
            "does not equal parsed row count {}",
            EXPECTED_FRONTIER_IDS.len() - 1
        )));
        assert!(error.contains(&format!("missing frontier IDs: {missing}")));
    }

    #[test]
    fn rejects_unexpected_key_mutation() {
        let mut document = baseline_document();
        let missing = document.row[0].frontier_id.clone();
        document.row[0].frontier_id = "unreviewed-frontier-expansion".to_owned();
        let error = error_text(&document);
        assert!(error.contains(&format!("missing frontier IDs: {missing}")));
        assert!(error.contains("unexpected frontier IDs: unreviewed-frontier-expansion"));
    }

    #[test]
    fn rejects_witnessless_closed_row_mutation() {
        let mut document = baseline_document();
        document.row[0].completion_witness = None;
        document.row[0].completion_verifier = None;
        let error = error_text(&document);
        assert!(error.contains("lacks completion_witness"));
        assert!(error.contains("lacks completion_verifier"));
    }

    #[test]
    fn rejects_actionless_open_row_mutation() {
        let mut document = baseline_document();
        let open_row = document
            .row
            .iter_mut()
            .find(|row| EXPECTED_OPEN_FRONTIER_IDS.contains(&row.frontier_id.as_str()))
            .unwrap();
        open_row.required_generator.clear();
        let error = error_text(&document);
        assert!(error.contains("lacks a next action or required generator"));
    }

    #[test]
    fn rejects_invalid_completion_state_mutation() {
        let mut document = baseline_document();
        document.row[0].completion_state = "complete".to_owned();
        let error = error_text(&document);
        assert!(error.contains("has invalid completion state \"complete\""));
    }

    #[test]
    fn rejects_unknown_dependency_mutation() {
        let mut document = baseline_document();
        document.row[0]
            .ordering_dependencies
            .push("outside-compiled-denominator".to_owned());
        let error = error_text(&document);
        assert!(error.contains("references unknown dependency outside-compiled-denominator"));
    }

    #[test]
    fn rejects_dependency_cycle_mutation() {
        let mut document = baseline_document();
        let first = document.row[0].frontier_id.clone();
        let second = document.row[1].frontier_id.clone();
        document.row[0].ordering_dependencies.push(second.clone());
        document.row[1].ordering_dependencies.push(first.clone());
        let error = error_text(&document);
        assert!(error.contains("dependency cycle includes"));
        assert!(error.contains(&first));
        assert!(error.contains(&second));
    }

    #[test]
    fn rejects_expected_state_partition_mutations() {
        let mut document = baseline_document();
        document.row[0].completion_state = "open".to_owned();
        let error = error_text(&document);
        assert!(error.contains("expected \"closed\""));

        let mut document = baseline_document();
        let open_row = document
            .row
            .iter_mut()
            .find(|row| EXPECTED_OPEN_FRONTIER_IDS.contains(&row.frontier_id.as_str()))
            .unwrap();
        open_row.completion_state = "closed".to_owned();
        open_row.completion_witness = Some("witness/invalid-state".to_owned());
        open_row.completion_verifier = Some("verify invalid-state".to_owned());
        let error = error_text(&document);
        assert!(error.contains("expected \"open\""));
    }
}
