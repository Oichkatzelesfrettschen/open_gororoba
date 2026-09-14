//! Exact finite-set verification for the Casimir/optics discrimination frontier.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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

// Each tuple binds a frontier ID to SHA-256 of the exact UTF-8 witness and
// verifier field values in the canonical frontier document.
const EXPECTED_CLOSED_ROW_IDENTITIES: [(&str, &str, &str); 28] = [
    (
        "lifshitz-planar-polar-normalization",
        "69d9a2c8016fba4803b3c6527317440450f3f97ff21ac8c3e6336e3dac706cb2",
        "9e1cd64e63ae2ed09b51510bf0f13910f5fb98b7565335d6f5a94c37df8a5f73",
    ),
    (
        "lifshitz-sphere-pfa-energy-conversion",
        "d62d93a4315e8f7b12796f6a5e5103dbcb8624cbac542659a1d8c92197161ad1",
        "5e3283f2c6e44d073955670bb3baec789efef9786aa5dd8de6b34377b3c792e9",
    ),
    (
        "materials-lifshitz-matsubara-prefactor",
        "99ab4b0c55777c22a4a42d95f8f1dee0128ad867fa8d1d186b97eb452d174ba9",
        "916c65483111bdf7814ec0c1352a3668810a9ac63e9f3049b01a116ac798e236",
    ),
    (
        "optics-passive-fourier-convention",
        "d3a7bf093834b3dcdb0af6c1f1cb982188a7492003d67bfa8ffacf6191f9c0bb",
        "97549ac3f3b1f61d02e398ae295df278edb01fd9997e65e92fc4182dc512cbb2",
    ),
    (
        "optics-finite-film-state-meter",
        "5fec20d532a4bd9c9e267ed249590a918f7734d7d7bb3a9cca4109f538c4145a",
        "916c65483111bdf7814ec0c1352a3668810a9ac63e9f3049b01a116ac798e236",
    ),
    (
        "calibrated-efficient-information",
        "d2727a84429061b73c35cc23279e9f73d52f2f4ce4c45aae631380b0436000a0",
        "d15849ca9c0451b6297e61522d0c761dfd55266dd78fd1cf117bfa8260d31704",
    ),
    (
        "bounded-nuisance-profile-distance",
        "a8a612a3bae94342c3712c30ebfa11946cf0f1508d39165bc40b0932324937ff",
        "d15849ca9c0451b6297e61522d0c761dfd55266dd78fd1cf117bfa8260d31704",
    ),
    (
        "derived-feature-joint-covariance",
        "3b6d4cc9affdcd0fad1954d3baa7de3af60fb6d35e65eb20477daf4729d24598",
        "d15849ca9c0451b6297e61522d0c761dfd55266dd78fd1cf117bfa8260d31704",
    ),
    (
        "discrimination-error-margin",
        "49b3e689d7ae8357da8733eded2c8c6ff29e207483a5bc40070cbaa35716a65b",
        "0a66219fcfbf824297a347abacfd967ab265514ed15dfca5cf089cf55fbfebb1",
    ),
    (
        "quantum-memory-complete-positivity-gate",
        "2a06fc933bf6a17c4bf603f692325ce4246dd09e0230caabc12891e445dc8065",
        "640f8d6688f2cd880cf0a5734c87cc3734bb08b44cc6563765485cb09b0bb937",
    ),
    (
        "hnls-operator-span-scope",
        "af4a3a76e36c5a40f4d35e884227bc6dda403cc802481ca8d60cbd5740362c08",
        "60311da45247006701474cb9be7d527439f988d409f14e88b51fd756f70b91a9",
    ),
    (
        "primary-source-intake",
        "d2fbfb5ade08bf9293615c8255fba2200c64201734b1f09f577fc4d108743a47",
        "f863b4b130a5d678a2e623034f5aead38cccadf2d31b872b264cd8636a1a9c6f",
    ),
    (
        "casimir-optics-rust-audit-producer",
        "4e974928ffd1f89f1cff4da9f7c3d87ff2685f191e20cf724e710921e3633599",
        "bd8cd763b987f63a1ce6bfe463e299476002ab4d0241605909a7fc46d7edb5b3",
    ),
    (
        "claim-experiment-scope-reconciliation",
        "b71443d1ff98b9650ac6852d5eeb32e52f1409575495eec6ab4c3eb2843c8ce2",
        "fb00f61ebacf1686d900d81775c2fe1c911d6810854d547e5b02b3286a69db04",
    ),
    (
        "material-state-specimen-identity",
        "f6366b40cbd8dc549687f6f443113c6eee59be26205925788682d8e4c64f0f8c",
        "2509c04ab696d293d04b3692991a5e94ca1c71a0f40feb014be74a21678f8d09",
    ),
    (
        "material-measurement-quantity-semantics",
        "45080d146123784fbbae40dfe87afcc652939aa4ed8574b2273eede26e3ec76d",
        "2509c04ab696d293d04b3692991a5e94ca1c71a0f40feb014be74a21678f8d09",
    ),
    (
        "material-derivation-evidence-classes",
        "ba1a5a9b358084f368ff4522afefd86968f597f34fa6e1229faae894fa3ef2e2",
        "2509c04ab696d293d04b3692991a5e94ca1c71a0f40feb014be74a21678f8d09",
    ),
    (
        "material-typed-missingness-featurizer",
        "ac7d813c8b2014d1ed583bd14a6a26a87ee9790d626f3bb86fdb2390cf9c75e8",
        "d9d259e83977a59547f2bcd41ceeaffddabdee1125ee18b56caf985f241779e8",
    ),
    (
        "mineral-metadata-sentinel-elimination",
        "f75e03ef69616c6d5d2ad7fee253784f0b91f5b829557d344d778fbc573feda8",
        "916c65483111bdf7814ec0c1352a3668810a9ac63e9f3049b01a116ac798e236",
    ),
    (
        "gold-specimen-source-admission",
        "b1ce398837650c815c968807c6218c50f685d105449b8d2d77d1cb80ab68c09c",
        "827fcba491a30d70f30e8360fa6bf03f3305e5ae85210291307e93e99b2588e7",
    ),
    (
        "ci-owned-broad-validation-resource-boundary",
        "018c043cc862df6b48a6423fb9833c9f6a7cab9e7235d4109d29682f4ea76484",
        "ceeb44aba9d10360550385dbbfb14075b8a3af137034368aa01c59f2cf7e5809",
    ),
    (
        "multilayer-au-oxide-pressure-replay",
        "a1052486bded65344c3e3d0db047c30b03f32ae82ff2f9037116c2f62c5ea5e9",
        "f897e1ddbe66d82179f4990b1025bd33675b4144b45b0a4d541b28462220246f",
    ),
    (
        "common-cap-finite-temperature-contrast",
        "2dcacfd27e417c691de1c4561cc1e582ef85280c5c332734693f4fe001044815",
        "f897e1ddbe66d82179f4990b1025bd33675b4144b45b0a4d541b28462220246f",
    ),
    (
        "uv-background-model-sensitivity",
        "24677cf3d67fee44e441e6da14044b38443a01b0ebdcb008c13aa0f01ca5ca65",
        "a9949b6f78d1dedeaaae1d95655492d8a1bb50e98e67b9377fea5bce08c6d22c",
    ),
    (
        "physical-nuisance-jacobian-replay",
        "e78ba6534993767ae1bed29c437b11daff844a3e7588b5f10947e072b099492c",
        "bd8cd763b987f63a1ce6bfe463e299476002ab4d0241605909a7fc46d7edb5b3",
    ),
    (
        "physical-calibration-priority-replay",
        "9c4fb8c5e28eadba87cdd704dcd39cb3d6130b3200b454db310fe16325ca9fcf",
        "3691ef55161dbcb3defe94cca40bdf6619a6eae5eadecdd0525a1995d3565fd8",
    ),
    (
        "classical-commutator-loop-control",
        "b4bbcd7ac0d32076e644a8b7170fb9247cedabc1005fe9a6a891077319566fda",
        "bd8cd763b987f63a1ce6bfe463e299476002ab4d0241605909a7fc46d7edb5b3",
    ),
    (
        "matched-active-sham-null-identity",
        "c18e61ecce18751ecfb5e638133eeeeda0ed698f3ace498faa833cdefdb5e675",
        "bd8cd763b987f63a1ce6bfe463e299476002ab4d0241605909a7fc46d7edb5b3",
    ),
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
    let expected_closed_ids: BTreeSet<&str> = expected_ids
        .difference(&expected_open_ids)
        .copied()
        .collect();
    let expected_identity_ids: BTreeSet<&str> = EXPECTED_CLOSED_ROW_IDENTITIES
        .iter()
        .map(|(frontier_id, _, _)| *frontier_id)
        .collect();
    let mut diagnostics = Vec::new();

    if expected_identity_ids.len() != EXPECTED_CLOSED_ROW_IDENTITIES.len() {
        diagnostics.push("compiled closed-row identities contain duplicate IDs".to_owned());
    }
    let missing_identity_ids: BTreeSet<_> = expected_closed_ids
        .difference(&expected_identity_ids)
        .copied()
        .collect();
    let unexpected_identity_ids: BTreeSet<_> = expected_identity_ids
        .difference(&expected_closed_ids)
        .copied()
        .collect();
    if !missing_identity_ids.is_empty() {
        diagnostics.push(format!(
            "closed frontier IDs lack compiled witness identities: {}",
            join_set(&missing_identity_ids)
        ));
    }
    if !unexpected_identity_ids.is_empty() {
        diagnostics.push(format!(
            "compiled witness identities name non-closed frontier IDs: {}",
            join_set(&unexpected_identity_ids)
        ));
    }

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
                let expected_identity = EXPECTED_CLOSED_ROW_IDENTITIES
                    .iter()
                    .find(|(frontier_id, _, _)| *frontier_id == row.frontier_id.as_str());
                if expected_identity.is_none() && expected_ids.contains(row.frontier_id.as_str()) {
                    diagnostics.push(format!(
                        "closed row {} lacks a compiled witness identity",
                        row.frontier_id
                    ));
                }
                match (expected_identity, row.completion_witness.as_deref()) {
                    (_, None) => diagnostics.push(format!(
                        "closed row {} lacks completion_witness",
                        row.frontier_id
                    )),
                    (_, Some(witness)) if witness.trim().is_empty() => diagnostics.push(format!(
                        "closed row {} lacks completion_witness",
                        row.frontier_id
                    )),
                    (Some((_, expected_witness_sha256, _)), Some(witness))
                        if sha256_hex(witness) != *expected_witness_sha256 =>
                    {
                        diagnostics.push(format!(
                            "closed row {} completion_witness identity mismatch",
                            row.frontier_id
                        ));
                    }
                    _ => {}
                }
                match (expected_identity, row.completion_verifier.as_deref()) {
                    (_, None) => diagnostics.push(format!(
                        "closed row {} lacks completion_verifier",
                        row.frontier_id
                    )),
                    (_, Some(verifier)) if verifier.trim().is_empty() => diagnostics.push(format!(
                        "closed row {} lacks completion_verifier",
                        row.frontier_id
                    )),
                    (Some((_, _, expected_verifier_sha256)), Some(verifier))
                        if sha256_hex(verifier) != *expected_verifier_sha256 =>
                    {
                        diagnostics.push(format!(
                            "closed row {} completion_verifier identity mismatch",
                            row.frontier_id
                        ));
                    }
                    _ => {}
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

    for row in &document.row {
        if row.completion_state != "closed" {
            continue;
        }
        for dependency in &row.ordering_dependencies {
            if rows_by_id
                .get(dependency.as_str())
                .is_some_and(|dependency_row| dependency_row.completion_state == "open")
            {
                diagnostics.push(format!(
                    "closed row {} depends on open row {}",
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

fn sha256_hex(value: &str) -> String {
    Sha256::digest(value.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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
        toml::from_str(include_str!(
            "../../../plans/casimir_optics_discrimination_frontier.toml"
        ))
        .unwrap()
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
    fn rejects_closed_row_witness_and_verifier_identity_mutations() {
        let baseline = baseline_document();
        let closed_ids: Vec<_> = baseline
            .row
            .iter()
            .filter(|row| row.completion_state == "closed")
            .map(|row| row.frontier_id.clone())
            .collect();
        assert_eq!(closed_ids.len(), EXPECTED_CLOSED_ROW_IDENTITIES.len());

        for frontier_id in closed_ids {
            let mut document = baseline.clone();
            let closed_row = document
                .row
                .iter_mut()
                .find(|row| row.frontier_id == frontier_id)
                .unwrap();
            closed_row.completion_witness = Some("fabricated witness".to_owned());
            let error = error_text(&document);
            assert!(error.contains(&format!(
                "closed row {frontier_id} completion_witness identity mismatch"
            )));

            let mut document = baseline.clone();
            let closed_row = document
                .row
                .iter_mut()
                .find(|row| row.frontier_id == frontier_id)
                .unwrap();
            closed_row.completion_verifier = Some("fabricated verifier".to_owned());
            let error = error_text(&document);
            assert!(error.contains(&format!(
                "closed row {frontier_id} completion_verifier identity mismatch"
            )));
        }
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
    fn rejects_closed_row_with_open_dependency() {
        let mut document = baseline_document();
        let open_id = document
            .row
            .iter()
            .find(|row| row.completion_state == "open")
            .unwrap()
            .frontier_id
            .clone();
        let closed_row = document
            .row
            .iter_mut()
            .find(|row| row.completion_state == "closed")
            .unwrap();
        let closed_id = closed_row.frontier_id.clone();
        closed_row.ordering_dependencies.push(open_id.clone());

        let error = error_text(&document);
        assert!(error.contains(&format!(
            "closed row {closed_id} depends on open row {open_id}"
        )));
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
