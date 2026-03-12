#!/usr/bin/env python3
"""
Bootstrap external source markdown documents into a TOML-first registry.

Input:
- docs/external_sources/*.md

Output:
- registry/external_sources.toml
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

CLAIM_RE = re.compile(r"\bC-\d{3}\b")
URL_RE = re.compile(r"https?://[^\s)>\"]+")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", flags=re.M)
GENERATED_HEADER_PREFIX = (
    "<!-- AUTO-GENERATED: DO NOT EDIT -->\n"
    "<!-- Source of truth: registry/external_sources.toml -->\n"
)
DEFAULT_META = {
    "operational_role": "reference_capture",
    "source_lineage_summary": "",
    "truth_surfaces": [],
    "artifact_contract_paths": [],
}

SOURCE_META = {
    "C010_NONLOCAL_ALGEBRAIC_METAMATERIALS_SOURCES.md": {
        "status_token": "ACTIVE",
        "content_kind": "claim_dataset_provenance",
        "authority_level": "primary_paper_bridge",
        "verification_level": "design_stage_replication",
        "operational_role": "claim_bridge",
        "truth_surfaces": ["algebra_topology", "design_stage_benchmark"],
        "artifact_contract_paths": [
            "crates/materials_core/src/nonlocal_metamaterial.rs",
            "crates/gororoba_cli_physics/src/bin/nonlocal_algebraic_metamaterial.rs",
            "data/csv/c010_nonlocal_material_calibrations.csv",
        ],
        "notes": (
            "Sources pack for the non-local C-010 recovery lane with explicit "
            "literature-backed LC, Floquet, graphene, and magnonic calibration anchors."
        ),
    },
    "C053_PATHION_METAMATERIAL_MAPPING_SOURCES.md": {
        "status_token": "REFERENCE",
        "content_kind": "claim_dataset_provenance",
        "authority_level": "project_claim_repair",
        "verification_level": "deterministic_toy_reproduction",
        "operational_role": "claim_bridge",
        "truth_surfaces": ["toy_mapping", "degeneracy_disclosure"],
        "artifact_contract_paths": [
            "crates/materials_core/src/pathion_toy_mapping.rs",
            "crates/gororoba_cli_physics/src/bin/c053_pathion_metamaterial_mapping.rs",
            "data/csv/c053_pathion_tmm_summary.csv",
            "crates/gororoba_cli_physics/tests/c053_pathion_metamaterial_mapping.rs",
        ],
        "notes": (
            "Repair dossier for the narrow C-053 toy lane. Keeps the pathion-to-TMM "
            "mapping reproducible while making its diagonal-only degeneracy explicit."
        ),
    },
    "C071_FRB_ULTRAMETRIC_SOURCES.md": {
        "status_token": "REFUTED",
        "content_kind": "claim_dataset_provenance",
        "authority_level": "primary_dataset_index",
        "verification_level": "computed_refutation",
        "notes": "Claim C-071 outcome is explicitly refuted with reproducible dataset hashes.",
    },
    "DATASET_MANIFEST.md": {
        "status_token": "ACTIVE",
        "content_kind": "dataset_manifest",
        "authority_level": "provider_manifest",
        "verification_level": "operational",
        "operational_role": "provider_manifest",
        "notes": "Operational provider and source manifest for fetch-datasets registry alignment.",
    },
    "DE_MARRAIS_BOXKITES_III.md": {
        "status_token": "REFERENCE",
        "content_kind": "paper_transcript",
        "authority_level": "primary_cached_paper",
        "verification_level": "source_capture",
        "notes": (
            "Cached transcript mirror for source auditability; not itself a verification result."
        ),
    },
    "DE_MARRAIS_CATAMARAN.md": {
        "status_token": "REFERENCE",
        "content_kind": "paper_summary",
        "authority_level": "primary_paper_summary",
        "verification_level": "source_capture",
        "notes": "Structured summary of a primary paper with claims-supporting context.",
    },
    "DE_MARRAIS_FLYING_HIGHER.md": {
        "status_token": "REFERENCE",
        "content_kind": "paper_transcript",
        "authority_level": "primary_cached_paper",
        "verification_level": "source_capture",
        "notes": "Contains large transcript and summary notes for downstream claims triage.",
    },
    "DE_MARRAIS_PLACEHOLDER_I.md": {
        "status_token": "REFERENCE",
        "content_kind": "paper_transcript",
        "authority_level": "primary_cached_paper",
        "verification_level": "source_capture",
        "notes": "Revision-aware source capture for placeholder substructure literature.",
    },
    "DE_MARRAIS_PLACEHOLDER_III.md": {
        "status_token": "REFERENCE",
        "content_kind": "paper_summary",
        "authority_level": "primary_paper_summary",
        "verification_level": "source_capture",
        "notes": "Short-form summary and key-result extraction from placeholder III source.",
    },
    "DE_MARRAIS_PRESTO_DIGITIZATION.md": {
        "status_token": "REFERENCE",
        "content_kind": "paper_transcript",
        "authority_level": "primary_cached_paper",
        "verification_level": "source_capture",
        "notes": "Primary transcript with explicit rule extraction for Cayley-Dickson references.",
    },
    "DE_MARRAIS_WOLFRAM_SLIDES.md": {
        "status_token": "REFERENCE",
        "content_kind": "slides_transcript",
        "authority_level": "primary_cached_slides",
        "verification_level": "source_capture",
        "notes": "Slide transcript capture; interpretation remains separate from source capture.",
    },
    "HELIOSPHERE_DATASET_PROGRESS_2026-03-09.md": {
        "status_token": "ACTIVE",
        "content_kind": "claims_inbox_index",
        "authority_level": "project_tracking_index",
        "verification_level": "workflow_control",
        "operational_role": "chronology_pack_status",
        "truth_surfaces": ["chronology_control"],
        "notes": (
            "Operational heliosphere chronology and"
            " staged-lane status note for experiment scoping."
        ),
    },
    "INDEX.md": {
        "status_token": "ACTIVE",
        "content_kind": "generated_index",
        "authority_level": "auto_generated",
        "verification_level": "operational",
        "operational_role": "generated_index",
        "notes": (
            "Auto-generated index of external source dossiers;"
            " source of truth is registry/external_sources.toml."
        ),
    },
    "INVERSE_CD_FORMALISM.md": {
        "status_token": "UNVERIFIED",
        "content_kind": "conversation_extraction",
        "authority_level": "derived_conversation_note",
        "verification_level": "unverified_hypothesis",
        "notes": "Conversation-derived formalism notes flagged as unverified in-file.",
    },
    "OMNI_DATA_AUDIT_2026-03-10.md": {
        "status_token": "ACTIVE",
        "content_kind": "claim_dataset_provenance",
        "authority_level": "primary_dataset_index",
        "verification_level": "source_capture",
        "operational_role": "dataset_lineage_audit",
        "source_lineage_summary": (
            "Mixed lineage: governed AMDA omni-hour-all fallback for 1997-2019; "
            "canonical SPDF OMNI2 yearly ASCII for 2020-2025."
        ),
        "truth_surfaces": ["environment_context", "lineage_transition"],
        "artifact_contract_paths": [
            "crates/data_core/src/catalogs/omni.rs",
            "crates/gororoba_cli_physics/src/bin/solar_wind_ic.rs",
        ],
        "notes": "Documents the staged OMNI mixed-lineage lane and year-by-year local coverage.",
    },
    "OPEN_CLAIMS_SOURCES.md": {
        "status_token": "ACTIVE",
        "content_kind": "claims_inbox_index",
        "authority_level": "project_tracking_index",
        "verification_level": "workflow_control",
        "operational_role": "claims_inbox",
        "notes": "Inbox index for open claims pending dedicated source dossiers.",
    },
    "PIONEER_FLYBY_ANOMALY_SOURCES.md": {
        "status_token": "ACTIVE",
        "content_kind": "claim_dataset_provenance",
        "authority_level": "primary_dataset_index",
        "verification_level": "source_capture",
        "operational_role": "falsification_contract",
        "truth_surfaces": ["observation_benchmark", "environment_context"],
        "artifact_contract_paths": [
            "crates/gororoba_cli_physics/src/anomaly_residual.rs",
            "crates/gororoba_cli_physics/src/bin/flyby_residual_audit.rs",
            "crates/gororoba_cli_physics/src/bin/pioneer_residual_audit.rs",
            "crates/gororoba_cli_physics/src/bin/fractal_metric_fit.rs",
        ],
        "notes": (
            "Primary-source and benchmark index for"
            " Pioneer anomaly and Earth-flyby audit inputs."
        ),
    },
    "REGGIANI_MANIFOLD_CLAIMS.md": {
        "status_token": "PARTIALLY_VERIFIED",
        "content_kind": "paper_claim_bridge",
        "authority_level": "primary_paper_bridge",
        "verification_level": "partial_replication",
        "operational_role": "claim_bridge",
        "notes": "Distinguishes paper-asserted manifold claims from replicated algebraic checks.",
    },
    "SEDENION_ZD_EXPERIMENTAL.md": {
        "status_token": "MIXED",
        "content_kind": "evidence_synthesis",
        "authority_level": "mixed_primary_and_conversation",
        "verification_level": "mixed",
        "notes": "Combines primary-source statements with codebase verification references.",
    },
    "SOHO_ARCHIVE_MIRROR_AUDIT_2026-03-09.md": {
        "status_token": "ACTIVE",
        "content_kind": "claim_dataset_provenance",
        "authority_level": "primary_dataset_index",
        "verification_level": "source_capture",
        "operational_role": "mirror_audit",
        "notes": (
            "Endpoint audit for SOHO archive retrieval"
            " surfaces and host-dependent mirror behavior."
        ),
    },
    "WHEEL_ALGEBRA_TAXONOMY.md": {
        "status_token": "UNVERIFIED",
        "content_kind": "conversation_extraction",
        "authority_level": "derived_conversation_note",
        "verification_level": "unverified_hypothesis",
        "notes": "Conversation-extracted taxonomy with explicit naming-collision warning.",
    },
    "WOW_SIGNAL_SOURCES.md": {
        "status_token": "ACTIVE",
        "content_kind": "claim_dataset_provenance",
        "authority_level": "primary_dataset_index",
        "verification_level": "source_capture",
        "notes": "Provenance chain for Wow! signal archival data and BL 6EQUJ5 follow-up.",
    },
    "ysu_engine_gpu_patterns.md": {
        "status_token": "REFERENCE",
        "content_kind": "technical_reference",
        "authority_level": "external_codebase_reference",
        "verification_level": "source_capture",
        "operational_role": "claim_bridge",
        "notes": (
            "Technique-reference capture mapping external"
            " GPU optimization patterns onto local CUDA/LBM work."
        ),
    },
}


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    source_markdown: str
    slug: str
    title: str
    status_token: str
    content_kind: str
    authority_level: str
    verification_level: str
    operational_role: str
    source_lineage_summary: str
    truth_surfaces: list[str]
    artifact_contract_paths: list[str]
    has_full_transcript: bool
    claim_refs: list[str]
    url_refs: list[str]
    path_refs: list[str]
    line_count: int
    notes: str
    body_markdown: str


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _ascii_sanitize(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    out_chars: list[str] = []
    for ch in text:
        mapped = replacements.get(ch, ch)
        for item in mapped:
            code = ord(item)
            if item in {"\n", "\r", "\t"}:
                out_chars.append(item)
            elif code < 32:
                out_chars.append(" ")
            elif code <= 127:
                out_chars.append(item)
            else:
                out_chars.append(f"<U+{code:04X}>")
    return "".join(out_chars)


def _escape(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _render_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_escape(value) for value in values) + "]"


def _meta_list(meta: dict[str, object], key: str) -> list[str]:
    raw = meta.get(key, [])
    if raw in ("", None):
        return []
    if not isinstance(raw, list):
        raise SystemExit(f"ERROR: {key} must be a list in SOURCE_META")
    return [str(item) for item in raw if str(item).strip()]


def _render_multiline(text: str) -> str:
    if "'''" not in text:
        return "'''\n" + text + "\n'''"
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _extract_paths(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in BACKTICK_RE.findall(text):
        token = raw.strip()
        if not token:
            continue
        if token.startswith("http://") or token.startswith("https://"):
            continue
        if "/" not in token and "." not in token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for raw in URL_RE.findall(text):
        token = raw.rstrip(".,;")
        if token in seen:
            continue
        seen.add(token)
        urls.append(token)
    return urls


def _title_from_text(text: str, fallback: str) -> str:
    match = HEADING_RE.search(text)
    if match:
        return match.group(1).strip()
    return fallback


def _strip_generated_header(text: str) -> str:
    stripped = text.lstrip()
    while stripped.startswith(GENERATED_HEADER_PREFIX):
        stripped = stripped[len(GENERATED_HEADER_PREFIX) :].lstrip("\n")
    return stripped


def _to_slug(filename: str) -> str:
    return filename.replace(".md", "").lower()


def _parse_doc(index: int, path: Path, text: str) -> SourceDoc:
    name = path.name
    meta_row = SOURCE_META.get(name)
    if meta_row is None:
        raise SystemExit(f"ERROR: Missing SOURCE_META entry for {name}")
    meta = dict(DEFAULT_META)
    meta.update(meta_row)
    sanitized = _strip_generated_header(_ascii_sanitize(text))
    claim_refs = sorted(set(CLAIM_RE.findall(sanitized)))
    title = _title_from_text(sanitized, path.stem)
    return SourceDoc(
        doc_id=f"XS-{index:03d}",
        source_markdown=path.as_posix(),
        slug=_to_slug(name),
        title=title,
        status_token=str(meta["status_token"]),
        content_kind=str(meta["content_kind"]),
        authority_level=str(meta["authority_level"]),
        verification_level=str(meta["verification_level"]),
        operational_role=str(meta["operational_role"]),
        source_lineage_summary=str(meta["source_lineage_summary"]),
        truth_surfaces=_meta_list(meta, "truth_surfaces"),
        artifact_contract_paths=_meta_list(meta, "artifact_contract_paths"),
        has_full_transcript=("## Full Transcript" in sanitized),
        claim_refs=claim_refs,
        url_refs=_extract_urls(sanitized),
        path_refs=_extract_paths(sanitized),
        line_count=len(sanitized.splitlines()),
        notes=str(meta["notes"]),
        body_markdown=sanitized.rstrip("\n"),
    )


def _render_toml(records: list[SourceDoc]) -> str:
    lines: list[str] = []
    lines.append("# External source dossiers normalized into TOML-first registry.")
    lines.append("# Generated by src/scripts/analysis/normalize_external_sources_registry.py")
    lines.append("")
    lines.append("[external_sources]")
    lines.append('updated = "2026-02-09"')
    lines.append("authoritative = true")
    lines.append('source_markdown_glob = "docs/external_sources/*.md"')
    lines.append(f"document_count = {len(records)}")
    lines.append("")
    for rec in records:
        lines.append("[[document]]")
        lines.append(f"id = {_escape(rec.doc_id)}")
        lines.append(f"source_markdown = {_escape(rec.source_markdown)}")
        lines.append(f"slug = {_escape(rec.slug)}")
        lines.append(f"title = {_escape(rec.title)}")
        lines.append(f"status_token = {_escape(rec.status_token)}")
        lines.append(f"content_kind = {_escape(rec.content_kind)}")
        lines.append(f"authority_level = {_escape(rec.authority_level)}")
        lines.append(f"verification_level = {_escape(rec.verification_level)}")
        lines.append(f"operational_role = {_escape(rec.operational_role)}")
        lines.append(f"source_lineage_summary = {_escape(rec.source_lineage_summary)}")
        lines.append(f"truth_surfaces = {_render_list(rec.truth_surfaces)}")
        lines.append(f"artifact_contract_paths = {_render_list(rec.artifact_contract_paths)}")
        lines.append(f"has_full_transcript = {'true' if rec.has_full_transcript else 'false'}")
        lines.append(f"claim_refs = {_render_list(rec.claim_refs)}")
        lines.append(f"url_refs = {_render_list(rec.url_refs)}")
        lines.append(f"path_refs = {_render_list(rec.path_refs)}")
        lines.append(f"line_count = {rec.line_count}")
        lines.append(f"notes = {_escape(rec.notes)}")
        lines.append(f"body_markdown = {_render_multiline(rec.body_markdown)}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--bootstrap-from-markdown",
        action="store_true",
        help="Required flag to ingest markdown into TOML registry.",
    )
    parser.add_argument(
        "--glob",
        default="docs/external_sources/*.md",
        help="Markdown source glob.",
    )
    parser.add_argument(
        "--out",
        default="registry/external_sources.toml",
        help="Output TOML registry path.",
    )
    args = parser.parse_args()

    if not args.bootstrap_from_markdown:
        raise SystemExit("ERROR: pass --bootstrap-from-markdown to ingest markdown sources")

    root = Path(args.repo_root).resolve()
    files = sorted(root.glob(args.glob))
    if not files:
        raise SystemExit(f"ERROR: no files matched {args.glob!r}")

    records: list[SourceDoc] = []
    for idx, file_path in enumerate(files, start=1):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        records.append(_parse_doc(idx, file_path.relative_to(root), text))

    rendered = _render_toml(records)
    out_path = root / args.out
    _assert_ascii(rendered, str(out_path))
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path} with {len(records)} documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
