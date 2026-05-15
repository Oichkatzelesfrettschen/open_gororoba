//! External-source contracts/dossiers loaders and compat-export
//! renderers.
//!
//! - `load_external_source_contracts_from_registry`: parses the
//!   `registry/external_sources.toml` [external_sources] meta header
//!   and [[source]] rows into (ExternalSourceContractsMeta,
//!   Vec<ExternalSourceContractRecord>).
//! - `load_external_source_dossiers_from_registry`: parses the
//!   external_source_dossiers.toml file into (ExternalSourceDossiersMeta,
//!   Vec<ExternalSourceDossierRecord>), defaulting `source_markdown`
//!   via `default_external_source_markdown_path` when the row omits it.
//! - `render_external_source_contracts_registry`: emits the compat TOML
//!   for the contracts surface.
//! - `render_external_source_dossiers_registry`: emits the compat TOML
//!   for the dossiers surface (including body_markdown as a TOML
//!   triple-quoted multi-line literal).
//! - `render_external_source_dossier_markdown`: emits the dossier
//!   Markdown body with the standard AUTO-GENERATED banner prepended.
//! - `render_string_array_lines`: helper used by both registry
//!   renderers for `key = [...]` TOML arrays.
//! - `render_toml_multiline`: encodes a body string as a TOML
//!   triple-quoted multi-line literal (escapes embedded `'''`).
//! - `default_external_source_markdown_path`: stable fallback path
//!   used when a dossier row omits source_markdown.

use anyhow::{Context, Result};
use provenance_core::{
    ExternalSourceContractRecord, ExternalSourceContractsMeta, ExternalSourceDossierRecord,
    ExternalSourceDossiersMeta,
};
use toml::Value;

use super::{
    compat_render::{
        bool_toml, external_sources_compat_toml_export_header,
        external_sources_markdown_export_header,
    },
    toml_helpers::{
        bool_field, optional_integer_field, optional_string_field, string_array_field, string_field,
    },
};

pub(crate) fn load_external_source_contracts_from_registry(
    raw: &str,
) -> Result<(
    ExternalSourceContractsMeta,
    Vec<ExternalSourceContractRecord>,
)> {
    let value: Value = toml::from_str(raw).context("parse external source contracts registry")?;
    let meta_table = value
        .get("external_sources")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let meta = ExternalSourceContractsMeta {
        updated: string_field(&meta_table, "updated"),
        authoritative: bool_field(&meta_table, "authoritative"),
        policy_version: string_field(&meta_table, "policy_version"),
    };
    let rows = value
        .get("source")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| row.as_table().cloned())
        .map(|table| ExternalSourceContractRecord {
            id: string_field(&table, "id"),
            path_glob: string_field(&table, "path_glob"),
            canonical_url: string_field(&table, "canonical_url"),
            mirror_urls: string_array_field(&table, "mirror_urls"),
            access_class: string_field(&table, "access_class"),
            status: string_field(&table, "status"),
            retrieval_method: string_field(&table, "retrieval_method"),
            attempt_deadline_utc: string_field(&table, "attempt_deadline_utc"),
            resolution_deadline_utc: string_field(&table, "resolution_deadline_utc"),
            blocker_note: string_field(&table, "blocker_note"),
            evidence_refs: string_array_field(&table, "evidence_refs"),
            manual_manifest_refs: string_array_field(&table, "manual_manifest_refs"),
            blocked_action_plan: string_array_field(&table, "blocked_action_plan"),
            scientific_validator_refs: string_array_field(&table, "scientific_validator_refs"),
        })
        .collect::<Vec<_>>();
    Ok((meta, rows))
}

pub(crate) fn load_external_source_dossiers_from_registry(
    raw: &str,
) -> Result<(ExternalSourceDossiersMeta, Vec<ExternalSourceDossierRecord>)> {
    let value: Value = toml::from_str(raw).context("parse external source dossiers registry")?;
    let meta_table = value
        .get("external_sources")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let rows = value
        .get("document")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| row.as_table().cloned())
        .map(|table| {
            let id = string_field(&table, "id");
            let slug = string_field(&table, "slug");
            let source_markdown = optional_string_field(&table, "source_markdown")
                .unwrap_or_else(|| default_external_source_markdown_path(&id, &slug));
            ExternalSourceDossierRecord {
                id,
                source_markdown,
                slug,
                title: string_field(&table, "title"),
                status_token: string_field(&table, "status_token"),
                content_kind: string_field(&table, "content_kind"),
                authority_level: string_field(&table, "authority_level"),
                verification_level: string_field(&table, "verification_level"),
                operational_role: string_field(&table, "operational_role"),
                source_lineage_summary: string_field(&table, "source_lineage_summary"),
                truth_surfaces: string_array_field(&table, "truth_surfaces"),
                artifact_contract_paths: string_array_field(&table, "artifact_contract_paths"),
                has_full_transcript: bool_field(&table, "has_full_transcript"),
                claim_refs: string_array_field(&table, "claim_refs"),
                url_refs: string_array_field(&table, "url_refs"),
                path_refs: string_array_field(&table, "path_refs"),
                line_count: optional_integer_field(&table, "line_count").unwrap_or_default()
                    as usize,
                notes: string_field(&table, "notes"),
                body_markdown: string_field(&table, "body_markdown"),
            }
        })
        .collect::<Vec<_>>();
    let meta = ExternalSourceDossiersMeta {
        updated: string_field(&meta_table, "updated"),
        authoritative: bool_field(&meta_table, "authoritative"),
        source_markdown_glob: string_field(&meta_table, "source_markdown_glob"),
        document_count: optional_integer_field(&meta_table, "document_count")
            .unwrap_or(rows.len() as i64) as usize,
    };
    Ok((meta, rows))
}

pub(crate) fn render_external_source_contracts_registry(
    meta: &ExternalSourceContractsMeta,
    rows: &[ExternalSourceContractRecord],
) -> String {
    let mut lines = external_sources_compat_toml_export_header("source_contracts");
    lines.push(String::new());
    lines.push("[external_sources]".to_string());
    lines.push(format!("updated = {:?}", meta.updated));
    lines.push(format!("authoritative = {}", bool_toml(meta.authoritative)));
    lines.push(format!("policy_version = {:?}", meta.policy_version));
    lines.push(String::new());
    for row in rows {
        lines.push("[[source]]".to_string());
        lines.push(format!("id = {:?}", row.id));
        lines.push(format!("path_glob = {:?}", row.path_glob));
        lines.push(format!("canonical_url = {:?}", row.canonical_url));
        render_string_array_lines(&mut lines, "mirror_urls", &row.mirror_urls);
        lines.push(format!("access_class = {:?}", row.access_class));
        lines.push(format!("status = {:?}", row.status));
        lines.push(format!("retrieval_method = {:?}", row.retrieval_method));
        lines.push(format!(
            "attempt_deadline_utc = {:?}",
            row.attempt_deadline_utc
        ));
        lines.push(format!(
            "resolution_deadline_utc = {:?}",
            row.resolution_deadline_utc
        ));
        lines.push(format!("blocker_note = {:?}", row.blocker_note));
        render_string_array_lines(&mut lines, "evidence_refs", &row.evidence_refs);
        render_string_array_lines(
            &mut lines,
            "manual_manifest_refs",
            &row.manual_manifest_refs,
        );
        render_string_array_lines(
            &mut lines,
            "scientific_validator_refs",
            &row.scientific_validator_refs,
        );
        render_string_array_lines(&mut lines, "blocked_action_plan", &row.blocked_action_plan);
        lines.push(String::new());
    }
    lines.join("\n")
}

pub(crate) fn render_external_source_dossiers_registry(
    meta: &ExternalSourceDossiersMeta,
    rows: &[ExternalSourceDossierRecord],
) -> String {
    let mut lines = external_sources_compat_toml_export_header("source_dossiers");
    lines.push(String::new());
    lines.push("[external_sources]".to_string());
    lines.push(format!("updated = {:?}", meta.updated));
    lines.push(format!("authoritative = {}", bool_toml(meta.authoritative)));
    lines.push(format!(
        "source_markdown_glob = {:?}",
        meta.source_markdown_glob
    ));
    lines.push(format!("document_count = {}", rows.len()));
    lines.push(String::new());
    for row in rows {
        lines.push("[[document]]".to_string());
        lines.push(format!("id = {:?}", row.id));
        lines.push(format!("source_markdown = {:?}", row.source_markdown));
        lines.push(format!("slug = {:?}", row.slug));
        lines.push(format!("title = {:?}", row.title));
        lines.push(format!("status_token = {:?}", row.status_token));
        lines.push(format!("content_kind = {:?}", row.content_kind));
        lines.push(format!("authority_level = {:?}", row.authority_level));
        lines.push(format!("verification_level = {:?}", row.verification_level));
        lines.push(format!("operational_role = {:?}", row.operational_role));
        lines.push(format!(
            "source_lineage_summary = {:?}",
            row.source_lineage_summary
        ));
        render_string_array_lines(&mut lines, "truth_surfaces", &row.truth_surfaces);
        render_string_array_lines(
            &mut lines,
            "artifact_contract_paths",
            &row.artifact_contract_paths,
        );
        lines.push(format!(
            "has_full_transcript = {}",
            bool_toml(row.has_full_transcript)
        ));
        render_string_array_lines(&mut lines, "claim_refs", &row.claim_refs);
        render_string_array_lines(&mut lines, "url_refs", &row.url_refs);
        render_string_array_lines(&mut lines, "path_refs", &row.path_refs);
        lines.push(format!("line_count = {}", row.line_count));
        lines.push(format!("notes = {:?}", row.notes));
        lines.push(format!(
            "body_markdown = {}",
            render_toml_multiline(&row.body_markdown)
        ));
        lines.push(String::new());
    }
    lines.join("\n")
}

pub(crate) fn render_external_source_dossier_markdown(row: &ExternalSourceDossierRecord) -> String {
    let mut lines = external_sources_markdown_export_header(&row.id);
    lines.push(row.body_markdown.trim_end().to_string());
    lines.join("\n")
}

fn render_string_array_lines(lines: &mut Vec<String>, key: &str, values: &[String]) {
    if values.is_empty() {
        lines.push(format!("{key} = []"));
        return;
    }
    lines.push(format!("{key} = ["));
    for value in values {
        lines.push(format!("  {:?},", value));
    }
    lines.push("]".to_string());
}

fn render_toml_multiline(body: &str) -> String {
    let sanitized = body.replace("'''", "'''\"\"\"'''");
    format!("'''\n{}\n'''", sanitized.trim_end())
}

fn default_external_source_markdown_path(id: &str, slug: &str) -> String {
    let stem = if !slug.trim().is_empty() {
        slug.trim().to_ascii_uppercase()
    } else if !id.trim().is_empty() {
        id.trim().to_ascii_uppercase()
    } else {
        "UNNAMED_EXTERNAL_SOURCE".to_string()
    };
    format!("docs/external_sources/{stem}.md")
}
