#!/usr/bin/env python3
"""
Bootstrap research narrative markdown into a TOML-first registry.

Inputs:
- docs/theory/*.md
- docs/engineering/*.md

Output:
- registry/research_narratives.toml
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


@dataclass(frozen=True)
class NarrativeDoc:
    doc_id: str
    source_markdown: str
    domain: str
    slug: str
    title: str
    status_token: str
    content_kind: str
    verification_level: str
    claim_refs: list[str]
    url_refs: list[str]
    path_refs: list[str]
    line_count: int
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


def _extract_urls(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in URL_RE.findall(text):
        token = raw.rstrip(".,;")
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


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


def _title_from_text(text: str, fallback: str) -> str:
    match = HEADING_RE.search(text)
    if match:
        return match.group(1).strip()
    return fallback


def _slug(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def _status_token(filename: str) -> str:
    upper = filename.upper()
    if upper.startswith("PHASE_"):
        return "PHASE_REPORT"
    if "AUDIT" in upper:
        return "AUDIT"
    if "REPORT" in upper:
        return "REPORT"
    if "SPEC" in upper or "PROTOCOL" in upper:
        return "SPECIFICATION"
    if "METHODOLOGY" in upper:
        return "METHODOLOGY"
    if "RECONCILIATION" in upper:
        return "RECONCILIATION"
    return "NARRATIVE"


def _content_kind(filename: str) -> str:
    upper = filename.upper()
    if upper.startswith("PHASE_"):
        return "phase_execution_report"
    if "AUDIT" in upper:
        return "audit_note"
    if "REPORT" in upper:
        return "engineering_report"
    if "SPEC" in upper or "PROTOCOL" in upper:
        return "specification"
    if "THEORY" in upper or "RECONCILIATION" in upper or "METHODOLOGY" in upper:
        return "theory_note"
    return "research_note"


def _verification_level(domain: str, filename: str) -> str:
    upper = filename.upper()
    if "VALIDATION" in upper or "VERIFICATION" in upper:
        return "validation_summary"
    if domain == "theory":
        return "theoretical"
    return "engineering_narrative"


def _parse_doc(index: int, rel_path: Path, text: str) -> NarrativeDoc:
    sanitized = _ascii_sanitize(text)
    domain = rel_path.parts[1] if len(rel_path.parts) > 1 else "docs"
    filename = rel_path.name
    return NarrativeDoc(
        doc_id=f"RN-{index:03d}",
        source_markdown=rel_path.as_posix(),
        domain=domain,
        slug=_slug(rel_path),
        title=_title_from_text(sanitized, rel_path.stem),
        status_token=_status_token(filename),
        content_kind=_content_kind(filename),
        verification_level=_verification_level(domain, filename),
        claim_refs=sorted(set(CLAIM_RE.findall(sanitized))),
        url_refs=_extract_urls(sanitized),
        path_refs=_extract_paths(sanitized),
        line_count=len(sanitized.splitlines()),
        body_markdown=sanitized.rstrip("\n"),
    )


def _render_toml(records: list[NarrativeDoc]) -> str:
    lines: list[str] = []
    lines.append("# Research narrative registry (TOML-first).")
    lines.append("# Generated by src/scripts/analysis/normalize_research_narratives_registry.py")
    lines.append("")
    lines.append("[research_narratives]")
    lines.append('updated = "2026-02-09"')
    lines.append("authoritative = true")
    lines.append('source_markdown_globs = ["docs/theory/*.md", "docs/engineering/*.md"]')
    lines.append(f"document_count = {len(records)}")
    lines.append("")
    for rec in records:
        lines.append("[[document]]")
        lines.append(f"id = {_escape(rec.doc_id)}")
        lines.append(f"source_markdown = {_escape(rec.source_markdown)}")
        lines.append(f"domain = {_escape(rec.domain)}")
        lines.append(f"slug = {_escape(rec.slug)}")
        lines.append(f"title = {_escape(rec.title)}")
        lines.append(f"status_token = {_escape(rec.status_token)}")
        lines.append(f"content_kind = {_escape(rec.content_kind)}")
        lines.append(f"verification_level = {_escape(rec.verification_level)}")
        lines.append(f"claim_refs = {_render_list(rec.claim_refs)}")
        lines.append(f"url_refs = {_render_list(rec.url_refs)}")
        lines.append(f"path_refs = {_render_list(rec.path_refs)}")
        lines.append(f"line_count = {rec.line_count}")
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
        "--out",
        default="registry/research_narratives.toml",
        help="Output TOML registry path.",
    )
    args = parser.parse_args()

    if not args.bootstrap_from_markdown:
        raise SystemExit("ERROR: pass --bootstrap-from-markdown to ingest markdown sources")

    root = Path(args.repo_root).resolve()
    files = sorted(
        [*root.glob("docs/theory/*.md"), *root.glob("docs/engineering/*.md")]
    )
    if not files:
        raise SystemExit("ERROR: no research narrative markdown files found")

    records: list[NarrativeDoc] = []
    for idx, path in enumerate(files, start=1):
        text = path.read_text(encoding="utf-8", errors="ignore")
        records.append(_parse_doc(idx, path.relative_to(root), text))

    rendered = _render_toml(records)
    out_path = root / args.out
    _assert_ascii(rendered, str(out_path))
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path} with {len(records)} documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
