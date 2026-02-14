#!/usr/bin/env python3
"""
Build a central TOML index for tracked markdown knowledge sources.

This index does not replace narrative docs. It classifies markdown files,
records provenance metadata, and maps markdown mirrors to TOML registries.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

CLAIM_ID_RE = re.compile(r"\bC-\d{3}\b")
INSIGHT_ID_RE = re.compile(r"\bI-\d{3}\b")
EXPERIMENT_ID_RE = re.compile(r"\bE-\d{3}\b")
HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", flags=re.M)
BACKTICK_RE = re.compile(r"`([^`\n]+)`")

MIRROR_TO_TOML = {
    "AGENTS.md": "registry/entrypoint_docs.toml",
    "CLAUDE.md": "registry/entrypoint_docs.toml",
    "GEMINI.md": "registry/entrypoint_docs.toml",
    "README.md": "registry/entrypoint_docs.toml",
    "curated/README.md": "registry/entrypoint_docs.toml",
    "curated/01_theory_frameworks/README_COQ.md": "registry/entrypoint_docs.toml",
    "data/csv/README.md": "registry/entrypoint_docs.toml",
    "data/artifacts/README.md": "registry/entrypoint_docs.toml",
    "NAVIGATOR.md": "registry/navigator.toml",
    "REQUIREMENTS.md": "registry/requirements.toml",
    "docs/REQUIREMENTS.md": "registry/requirements.toml",
    "docs/CLAIMS_EVIDENCE_MATRIX.md": "registry/claims.toml",
    "docs/BIBLIOGRAPHY.md": "registry/bibliography.toml",
    "docs/INSIGHTS.md": "registry/insights.toml",
    "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md": "registry/experiments.toml",
    "docs/generated/REPORTS_NARRATIVES_REGISTRY_MIRROR.md": "registry/reports_narratives.toml",
    "docs/generated/DOCS_CONVOS_REGISTRY_MIRROR.md": "registry/docs_convos.toml",
    "docs/generated/DATA_ARTIFACT_NARRATIVES_REGISTRY_MIRROR.md": "registry/data_artifact_narratives.toml",
    "data/artifacts/ALGEBRAIC_FOUNDATIONS.md": "registry/data_artifact_narratives.toml",
    "data/artifacts/BIBLIOGRAPHY.md": "registry/data_artifact_narratives.toml",
    "data/artifacts/FINAL_REPORT.md": "registry/data_artifact_narratives.toml",
    "data/artifacts/QUANTUM_REPORT.md": "registry/data_artifact_narratives.toml",
    "data/artifacts/SIMULATION_REPORT.md": "registry/data_artifact_narratives.toml",
    "data/artifacts/extracted_equations.md": "registry/data_artifact_narratives.toml",
    "data/artifacts/reality_check_and_synthesis.md": "registry/data_artifact_narratives.toml",
    "docs/book/src/registry/claims.md": "registry/claims.toml",
    "docs/book/src/registry/insights.md": "registry/insights.toml",
    "docs/book/src/registry/experiments.md": "registry/experiments.toml",
}

IMMUTABLE_AGENT_OVERLAYS = {"CLAUDE.md", "GEMINI.md"}


def _toml_backing_for_path(path: str) -> str:
    if path in MIRROR_TO_TOML:
        return MIRROR_TO_TOML[path]
    if path.startswith("reports/"):
        return "registry/reports_narratives.toml"
    if path.startswith("docs/convos/"):
        return "registry/docs_convos.toml"
    if path.startswith("docs/") and path.count("/") == 1:
        return "registry/docs_root_narratives.toml"
    if path.startswith("docs/book/src/"):
        return "registry/book_docs.toml"
    if path.startswith("docs/external_sources/"):
        return "registry/external_sources.toml"
    if path.startswith("docs/theory/") or path.startswith("docs/engineering/"):
        return "registry/research_narratives.toml"
    return ""


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    path: str
    title: str
    kind: str
    authoring_mode: str
    generated: bool
    status: str
    migration_priority: str
    toml_backing: str
    sha256: str
    size_bytes: int
    line_count: int
    claim_ref_count: int
    insight_ref_count: int
    experiment_ref_count: int
    link_count: int
    link_sample: list[str]


def _tracked_markdown_files(repo_root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "*.md"],
        cwd=repo_root,
        text=True,
    )
    files = [line.strip() for line in output.splitlines() if line.strip()]
    existing = [rel for rel in files if (repo_root / rel).exists()]
    return sorted(existing)


def _escape_toml_string(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _title_from_markdown(text: str, fallback: str) -> str:
    match = HEADING_RE.search(text)
    if match:
        return match.group(1).strip()
    return fallback


def _kind_for_path(path: str, text: str) -> tuple[str, str, bool]:
    if path in IMMUTABLE_AGENT_OVERLAYS:
        return ("manual_source", "manual", False)
    if path in MIRROR_TO_TOML and path.startswith("data/artifacts/"):
        return ("markdown_mirror", "generated", True)
    if path.startswith("reports/"):
        return ("markdown_mirror", "generated", True)
    if path.startswith("docs/convos/"):
        return ("markdown_mirror", "generated", True)
    if path.startswith("docs/") and path.count("/") == 1:
        return ("markdown_mirror", "generated", True)
    if path.startswith("docs/book/src/"):
        return ("markdown_mirror", "generated", True)
    if path.startswith("docs/theory/") or path.startswith("docs/engineering/"):
        return ("markdown_mirror", "generated", True)
    if path.startswith("docs/external_sources/"):
        return ("markdown_mirror", "generated", True)
    if path in MIRROR_TO_TOML:
        return ("markdown_mirror", "generated", True)
    if path.startswith("convos/"):
        return ("transcript_input", "manual", False)
    if path.startswith("data/artifacts/"):
        return ("artifact_report", "generated", True)
    if path.startswith("docs/claims/by_domain/"):
        return ("generated_markdown", "generated", True)
    if path.startswith("docs/tickets/") and path.endswith("_claims_audit.md"):
        return ("generated_markdown", "generated", True)
    if path.startswith("docs/book/src/registry/"):
        return ("generated_markdown", "generated", True)
    if "auto-generated" in text.lower() or "do not edit" in text.lower():
        return ("generated_markdown", "generated", True)
    return ("manual_source", "manual", False)


def _status_for_path(path: str) -> str:
    if path.startswith("docs/archive/") or path.startswith("archive/"):
        return "archived"
    return "active"


def _migration_priority(kind: str, path: str) -> str:
    if kind == "markdown_mirror":
        return "critical"
    if kind == "manual_source" and path.startswith("docs/"):
        return "high"
    if kind == "manual_source":
        return "medium"
    return "none"


def _extract_link_sample(text: str) -> list[str]:
    links: set[str] = set()
    for raw in BACKTICK_RE.findall(text):
        token = raw.strip()
        if "/" not in token and "." not in token:
            continue
        if token.startswith("http://") or token.startswith("https://"):
            continue
        if len(token) > 200:
            continue
        if " " in token and "/" not in token:
            continue
        links.add(token)
    return sorted(links)[:8]


def _record_for_path(index: int, repo_root: Path, rel_path: str) -> DocumentRecord:
    path = repo_root / rel_path
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = _title_from_markdown(text, Path(rel_path).stem)
    kind, authoring_mode, generated = _kind_for_path(rel_path, text[:4000])
    status = _status_for_path(rel_path)
    migration_priority = _migration_priority(kind, rel_path)
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    line_count = text.count("\n") + (0 if not text else 1)
    claim_ref_count = len(set(CLAIM_ID_RE.findall(text)))
    insight_ref_count = len(set(INSIGHT_ID_RE.findall(text)))
    experiment_ref_count = len(set(EXPERIMENT_ID_RE.findall(text)))
    link_sample = _extract_link_sample(text)
    link_count = len(set(BACKTICK_RE.findall(text)))
    return DocumentRecord(
        doc_id=f"DOC-{index:04d}",
        path=rel_path,
        title=title,
        kind=kind,
        authoring_mode=authoring_mode,
        generated=generated,
        status=status,
        migration_priority=migration_priority,
        toml_backing=_toml_backing_for_path(rel_path),
        sha256=sha256,
        size_bytes=path.stat().st_size,
        line_count=line_count,
        claim_ref_count=claim_ref_count,
        insight_ref_count=insight_ref_count,
        experiment_ref_count=experiment_ref_count,
        link_count=link_count,
        link_sample=link_sample,
    )


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII content in {context}: {sample!r}")


def _render_toml(records: list[DocumentRecord]) -> str:
    kinds = {}
    for rec in records:
        kinds[rec.kind] = kinds.get(rec.kind, 0) + 1

    lines: list[str] = []
    lines.append("# Knowledge source index for markdown assets.")
    lines.append("# Auto-generated by src/scripts/analysis/build_knowledge_sources_registry.py")
    lines.append(
        "# Regenerate with: python3 src/scripts/analysis/build_knowledge_sources_registry.py"
    )
    lines.append("")
    lines.append("[knowledge_sources]")
    lines.append('generated_at = "deterministic"')
    lines.append(f"tracked_markdown_count = {len(records)}")
    lines.append(f"manual_source_count = {kinds.get('manual_source', 0)}")
    lines.append(f"markdown_mirror_count = {kinds.get('markdown_mirror', 0)}")
    lines.append(f"generated_markdown_count = {kinds.get('generated_markdown', 0)}")
    lines.append(f"artifact_report_count = {kinds.get('artifact_report', 0)}")
    lines.append(f"transcript_input_count = {kinds.get('transcript_input', 0)}")
    lines.append("")

    for rec in records:
        lines.append("[[document]]")
        lines.append(f"id = {_escape_toml_string(rec.doc_id)}")
        lines.append(f"path = {_escape_toml_string(rec.path)}")
        lines.append(f"title = {_escape_toml_string(rec.title)}")
        lines.append(f"kind = {_escape_toml_string(rec.kind)}")
        lines.append(f"authoring_mode = {_escape_toml_string(rec.authoring_mode)}")
        lines.append(f"generated = {'true' if rec.generated else 'false'}")
        lines.append(f"status = {_escape_toml_string(rec.status)}")
        lines.append(f"migration_priority = {_escape_toml_string(rec.migration_priority)}")
        if rec.toml_backing:
            lines.append(f"toml_backing = {_escape_toml_string(rec.toml_backing)}")
        lines.append(f"sha256 = {_escape_toml_string(rec.sha256)}")
        lines.append(f"size_bytes = {rec.size_bytes}")
        lines.append(f"line_count = {rec.line_count}")
        lines.append(f"claim_ref_count = {rec.claim_ref_count}")
        lines.append(f"insight_ref_count = {rec.insight_ref_count}")
        lines.append(f"experiment_ref_count = {rec.experiment_ref_count}")
        lines.append(f"link_count = {rec.link_count}")
        if rec.link_sample:
            joined = ", ".join(_escape_toml_string(item) for item in rec.link_sample)
            lines.append(f"link_sample = [{joined}]")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--out",
        default="registry/knowledge_sources.toml",
        help="Output TOML path relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = (repo_root / args.out).resolve()
    files = _tracked_markdown_files(repo_root)
    records = [_record_for_path(i + 1, repo_root, rel_path) for i, rel_path in enumerate(files)]
    rendered = _render_toml(records)
    _assert_ascii(rendered, str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path} with {len(records)} markdown records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
