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

SOURCE_META = {
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
        "notes": "Operational provider and source manifest for fetch-datasets registry alignment.",
    },
    "DE_MARRAIS_BOXKITES_III.md": {
        "status_token": "REFERENCE",
        "content_kind": "paper_transcript",
        "authority_level": "primary_cached_paper",
        "verification_level": "source_capture",
        "notes": (
            "Cached transcript mirror for source auditability; "
            "not itself a verification result."
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
    "INVERSE_CD_FORMALISM.md": {
        "status_token": "UNVERIFIED",
        "content_kind": "conversation_extraction",
        "authority_level": "derived_conversation_note",
        "verification_level": "unverified_hypothesis",
        "notes": "Conversation-derived formalism notes flagged as unverified in-file.",
    },
    "OPEN_CLAIMS_SOURCES.md": {
        "status_token": "ACTIVE",
        "content_kind": "claims_inbox_index",
        "authority_level": "project_tracking_index",
        "verification_level": "workflow_control",
        "notes": "Inbox index for open claims pending dedicated source dossiers.",
    },
    "REGGIANI_MANIFOLD_CLAIMS.md": {
        "status_token": "PARTIALLY_VERIFIED",
        "content_kind": "paper_claim_bridge",
        "authority_level": "primary_paper_bridge",
        "verification_level": "partial_replication",
        "notes": "Distinguishes paper-asserted manifold claims from replicated algebraic checks.",
    },
    "SEDENION_ZD_EXPERIMENTAL.md": {
        "status_token": "MIXED",
        "content_kind": "evidence_synthesis",
        "authority_level": "mixed_primary_and_conversation",
        "verification_level": "mixed",
        "notes": "Combines primary-source statements with codebase verification references.",
    },
    "WHEEL_ALGEBRA_TAXONOMY.md": {
        "status_token": "UNVERIFIED",
        "content_kind": "conversation_extraction",
        "authority_level": "derived_conversation_note",
        "verification_level": "unverified_hypothesis",
        "notes": "Conversation-extracted taxonomy with explicit naming-collision warning.",
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


def _to_slug(filename: str) -> str:
    return filename.replace(".md", "").lower()


def _parse_doc(index: int, path: Path, text: str) -> SourceDoc:
    name = path.name
    meta = SOURCE_META.get(name)
    if meta is None:
        raise SystemExit(f"ERROR: Missing SOURCE_META entry for {name}")
    sanitized = _ascii_sanitize(text)
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
    lines.append('authoritative = true')
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
