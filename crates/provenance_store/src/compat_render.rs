//! Registry compat-export rendering helpers.
//!
//! These functions render claim, insight, experiment, binary, and
//! external-source rows into
//! the canonical TOML/Markdown compatibility exports consumed by
//! `registry/*.toml` and `docs/generated/*.md`.
//!
//! All output starts with the standard
//! AUTO-GENERATED / READ-ONLY COMPATIBILITY EXPORT banner. The
//! `splice_compat_toml_overrides` helper re-projects live SQLite
//! columns (status, formal_proof, status_note) into cached compat_toml_text
//! rows so mutations applied via the canonical SQLite write path
//! reach every downstream consumer.

use std::{fs, path::Path};

use anyhow::{Context, Result};
use provenance_core::{BinaryRecord, ClaimRecord};
use toml::Value;

use super::{
    migrations::{
        CONTROL_PLANE_DB_PATH, CONTROL_PLANE_EXPORT_COMMAND, EXTERNAL_SOURCES_EXPORT_COMMAND,
    },
    types::{ExperimentCompatRecord, InsightCompatRecord},
};

pub(crate) fn render_claims_registry(claims: &[ClaimRecord]) -> String {
    render_array_of_tables_registry("claims", "claim", claims.iter().map(render_claim_row))
}

pub(crate) fn render_insights_registry(insights: &[InsightCompatRecord]) -> String {
    render_array_of_tables_registry(
        "insights",
        "insight",
        insights.iter().map(render_insight_row),
    )
}

pub(crate) fn render_experiments_registry(
    header_toml: &str,
    experiments: &[ExperimentCompatRecord],
) -> String {
    let mut lines = compat_toml_export_header("experiments");
    let header = rebuild_experiments_header_toml(header_toml, experiments);
    if !header.trim().is_empty() {
        lines.push("[experiments]".to_string());
        lines.push(header);
        lines.push(String::new());
    }
    for row in experiments {
        lines.push("[[experiment]]".to_string());
        lines.push(render_experiment_row(row));
        lines.push(String::new());
    }
    lines.join("\n")
}

pub(crate) fn rebuild_experiments_header_toml(
    header_toml: &str,
    experiments: &[ExperimentCompatRecord],
) -> String {
    // Use toml::from_str rather than .parse::<Value>(): in toml 1.1 the FromStr
    // implementation rejects multi-line key-value documents.
    let mut table = toml::from_str::<Value>(header_toml.trim())
        .ok()
        .and_then(|value| value.as_table().cloned())
        .unwrap_or_default();
    table.insert("authoritative".to_string(), Value::Boolean(true));
    table.insert(
        "experiment_count".to_string(),
        Value::Integer(experiments.len() as i64),
    );
    table.insert(
        "deterministic_count".to_string(),
        Value::Integer(
            experiments
                .iter()
                .filter(|row| experiment_row_flag(row, "deterministic").unwrap_or(false))
                .count() as i64,
        ),
    );
    table.insert(
        "gpu_count".to_string(),
        Value::Integer(
            experiments
                .iter()
                .filter(|row| experiment_row_flag(row, "gpu").unwrap_or(false))
                .count() as i64,
        ),
    );
    table.insert(
        "seeded_count".to_string(),
        Value::Integer(
            experiments
                .iter()
                .filter(|row| experiment_row_has_seed(row))
                .count() as i64,
        ),
    );
    // status_allowlist is a constant policy field consumed by execution-planning --verify.
    // It must be present in every compat export so the verify allowlist check has something to read.
    table.insert(
        "status_allowlist".to_string(),
        Value::Array(vec![
            Value::String("active".to_string()),
            Value::String("deprecated".to_string()),
            Value::String("planned".to_string()),
            Value::String("blocked".to_string()),
        ]),
    );
    toml::to_string(&table)
        .unwrap_or_default()
        .trim()
        .to_string()
}

pub(crate) fn experiment_row_table(row: &ExperimentCompatRecord) -> Option<toml::value::Table> {
    // Use toml::from_str rather than .parse::<Value>(): in toml 1.1 the FromStr
    // implementation rejects multi-line key-value documents (e.g. compat_toml_text).
    toml::from_str::<Value>(row.compat_toml_text.trim())
        .ok()
        .and_then(|value| value.as_table().cloned())
}

pub(crate) fn experiment_row_flag(row: &ExperimentCompatRecord, key: &str) -> Option<bool> {
    experiment_row_table(row).and_then(|table| table.get(key).and_then(Value::as_bool))
}

pub(crate) fn experiment_row_has_seed(row: &ExperimentCompatRecord) -> bool {
    experiment_row_table(row)
        .and_then(|table| table.get("seed").cloned())
        .is_some()
}

pub(crate) fn render_array_of_tables_registry(
    kind: &str,
    array_key: &str,
    rows: impl IntoIterator<Item = String>,
) -> String {
    let mut lines = compat_toml_export_header(kind);
    for row in rows {
        lines.push(format!("[[{array_key}]]"));
        lines.push(row.trim().to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

pub(crate) fn normalized_export_text(body: &str) -> String {
    format!("{}\n", body.trim_end_matches('\n'))
}

pub(crate) fn render_claim_row(row: &ClaimRecord) -> String {
    if row.compat_toml_text.trim().is_empty() {
        let mut lines = vec![
            format!("id = {:?}", row.id),
            format!("statement = {:?}", row.statement),
            format!("status = {:?}", row.status),
            format!("where_stated = {:?}", row.where_stated),
            format!("last_verified = {:?}", row.last_verified),
        ];
        if let Some(formal_proof) = &row.formal_proof {
            lines.push(format!("formal_proof = {:?}", formal_proof));
        }
        if let Some(status_note) = &row.status_note {
            lines.push(format!("status_note = {:?}", status_note));
        }
        lines.join("\n")
    } else {
        // Splice live SQLite columns (status, formal_proof, status_note) into the
        // cached compat_toml_text. Without this, mutations applied via
        // `gororoba-db claim update-*` land in the database but never reach
        // the registry/claims.toml consumer surface.
        splice_compat_toml_overrides(
            &row.compat_toml_text,
            &[
                ("status", Some(row.status.as_str())),
                ("formal_proof", row.formal_proof.as_deref()),
                ("status_note", row.status_note.as_deref()),
            ],
        )
    }
}

pub(crate) fn render_insight_row(row: &InsightCompatRecord) -> String {
    if row.compat_toml_text.trim().is_empty() {
        let mut lines = vec![
            format!("id = {:?}", row.id),
            format!("title = {:?}", row.title),
            format!("status = {:?}", row.status),
        ];
        if !row.claim_refs.is_empty() {
            lines.push(format!("claims = {:?}", row.claim_refs));
        }
        if let Some(status_note) = &row.status_note {
            lines.push(format!("status_note = {:?}", status_note));
        }
        lines.join("\n")
    } else {
        splice_compat_toml_overrides(
            &row.compat_toml_text,
            &[("status_note", row.status_note.as_deref())],
        )
    }
}

pub(crate) fn render_experiment_row(row: &ExperimentCompatRecord) -> String {
    if row.compat_toml_text.trim().is_empty() {
        let mut lines = vec![
            format!("id = {:?}", row.id),
            format!("title = {:?}", row.title),
            format!("status = {:?}", row.status),
        ];
        if let Some(binary) = &row.binary {
            lines.push(format!("binary = {:?}", binary));
        }
        if !row.claim_refs.is_empty() {
            lines.push(format!("claim_refs = {:?}", row.claim_refs));
        }
        if let Some(status_note) = &row.status_note {
            lines.push(format!("status_note = {:?}", status_note));
        }
        lines.join("\n")
    } else {
        splice_compat_toml_overrides(
            &row.compat_toml_text,
            &[("status_note", row.status_note.as_deref())],
        )
    }
}

/// Splice live SQLite column values into a cached compat-export TOML row.
///
/// `compat_toml_text` is the verbatim TOML body that was captured when the row
/// was originally bootstrapped into SQLite. The canonical mutation surface
/// (`gororoba-db claim/insight/experiment update-*`) writes new values to the
/// dedicated SQLite columns but does NOT rewrite `compat_toml_text`. Without
/// re-projecting those live columns into the emitted compat TOML, mutations
/// would be invisible to every consumer that reads `registry/*.toml`.
///
/// Each `(key, value)` pair is applied as follows:
/// - `Some(v)`: insert the key with that value, replacing any existing value.
/// - `None`:    remove the key from the TOML if present (column was cleared).
///
/// Falls back to the original text on parse failure so a single malformed row
/// cannot regress the entire export.
pub(crate) fn splice_compat_toml_overrides(
    compat_toml_text: &str,
    overrides: &[(&str, Option<&str>)],
) -> String {
    let trimmed = compat_toml_text.trim();
    if overrides.iter().all(|(_, v)| v.is_none()) {
        return trimmed.to_string();
    }
    let Ok(parsed) = trimmed.parse::<toml_edit::DocumentMut>() else {
        return trimmed.to_string();
    };
    let mut doc = parsed;
    for (key, value) in overrides {
        match value {
            Some(v) => {
                doc[*key] = toml_edit::value(*v);
            }
            None => {
                doc.remove(key);
            }
        }
    }
    // toml_edit appends a key the cached body never had, so a value that arrived
    // through a `gororoba-db update-*` mutation would land after the
    // mirror-sorted keys while the same value re-imported from the mirror lands
    // in sorted position. Sorting the top-level keys makes the emitted row a
    // function of its content alone, so the export is identical whichever path
    // wrote the value.
    doc.sort_values();
    doc.to_string().trim().to_string()
}

pub(crate) fn render_binaries_registry(binaries: &[BinaryRecord]) -> String {
    let mut lines = compat_toml_export_header("binaries");
    lines.extend([
        "# CLI binaries registry -- generated from the canonical SQLite control plane.".to_string(),
        String::new(),
    ]);
    for binary in binaries {
        lines.push("[[binary]]".to_string());
        lines.push(format!("name = {:?}", binary.name));
        lines.push(format!("crate = {:?}", binary.crate_name));
        lines.push(format!("description = {:?}", binary.description));
        if let Some(experiment) = &binary.experiment {
            lines.push(format!("experiment = {:?}", experiment));
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

pub(crate) fn compat_toml_export_header(kind: &str) -> Vec<String> {
    vec![
        "# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.".to_string(),
        format!("# Canonical write path: {CONTROL_PLANE_DB_PATH}"),
        format!("# Regenerate with: {CONTROL_PLANE_EXPORT_COMMAND}"),
        format!("# Compatibility export lane: {kind}"),
        String::new(),
    ]
}

pub(crate) fn compat_markdown_export_header(source_label: &str) -> Vec<String> {
    vec![
        "<!-- AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT. -->".to_string(),
        format!("<!-- Source of truth: {CONTROL_PLANE_DB_PATH} -->"),
        format!("<!-- Canonical write path: {CONTROL_PLANE_DB_PATH} -->"),
        format!("<!-- Source label: {source_label} -->"),
        format!("<!-- Regenerate with: {CONTROL_PLANE_EXPORT_COMMAND} -->"),
        String::new(),
    ]
}

pub(crate) fn external_sources_compat_toml_export_header(kind: &str) -> Vec<String> {
    vec![
        "# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.".to_string(),
        format!("# Canonical write path: {CONTROL_PLANE_DB_PATH}"),
        format!("# Regenerate with: {EXTERNAL_SOURCES_EXPORT_COMMAND}"),
        format!("# Compatibility export lane: {kind}"),
    ]
}

pub(crate) fn external_sources_markdown_export_header(source_label: &str) -> Vec<String> {
    vec![
        "<!-- AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT. -->".to_string(),
        "<!-- Source of truth: registry/external_sources.toml -->".to_string(),
        format!("<!-- Canonical write path: {CONTROL_PLANE_DB_PATH} -->"),
        format!("<!-- Source label: {source_label} -->"),
        format!("<!-- Regenerate with: {EXTERNAL_SOURCES_EXPORT_COMMAND} -->"),
        String::new(),
    ]
}

pub(crate) fn bool_toml(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

pub(crate) fn write_text(path: &Path, body: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create parent directory {}", parent.display()))?;
    }
    fs::write(path, normalized_export_text(body))
        .with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::render_claim_row;
    use provenance_core::ClaimRecord;

    #[test]
    fn render_claim_row_reprojects_live_status_over_cached_text() {
        let row = ClaimRecord {
            id: "C-820".to_string(),
            statement: "A claim".to_string(),
            status: "Provisional".to_string(),
            where_stated: "source".to_string(),
            last_verified: "2026-08-04".to_string(),
            formal_proof: None,
            status_note: Some("transition applied".to_string()),
            compat_toml_text: "id = \"C-820\"\nstatus = \"Verified\"\n".to_string(),
        };

        let rendered = render_claim_row(&row);
        let table: toml::Table = toml::from_str(&rendered).expect("rendered claim is valid TOML");
        assert_eq!(
            table.get("status").and_then(toml::Value::as_str),
            Some("Provisional")
        );
        assert_eq!(
            table.get("status_note").and_then(toml::Value::as_str),
            Some("transition applied")
        );
    }
}
