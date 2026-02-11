#!/usr/bin/env python3
"""
Build strict structured-TOML payload registries from embedded body_markdown fields.

This converts markdown text embedded inside canonical TOML registries into typed
TOML units so deleted markdown files remain representable as pure TOML content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


TARGET_PREFIXES = ("docs/", "reports/", "data/artifacts/")
ROOT_TARGETS = {
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "README.md",
    "PANTHEON_PHYSICSFORGE_90_POINT_MIGRATION_PLAN.md",
    "PHASE10_11_ULTIMATE_ROADMAP.md",
    "PYTHON_REFACTORING_ROADMAP.md",
    "SYNTHESIS_PIPELINE_PROGRESS.md",
    "crates/vacuum_frustration/IMPLEMENTATION_NOTES.md",
    "curated/README.md",
    "curated/01_theory_frameworks/README_COQ.md",
    "data/csv/README.md",
    "data/artifacts/README.md",
    "NAVIGATOR.md",
    "REQUIREMENTS.md",
    "docs/REQUIREMENTS.md",
}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_RE = re.compile(r"^(?:[-*+]\s+|\d+\.\s+)(.+)$")


@dataclass(frozen=True)
class Unit:
    kind: str
    text_ascii: str
    line_start: int
    line_end: int
    heading_level: int


@dataclass(frozen=True)
class Candidate:
    path: str
    body: str
    source_registry: str
    source_document_id: str
    source_title: str


def _q(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _render_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_q(item) for item in values) + "]"


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _ascii_clean(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch in {"\n", "\r", "\t"}:
            out.append(ch)
        elif code < 32:
            out.append(" ")
        elif code <= 127:
            out.append(ch)
        elif code <= 0xFFFF:
            out.append(f"\\u{code:04X}")
        else:
            out.append(f"\\U{code:08X}")
    return "".join(out)


def _collapse(text: str) -> str:
    return " ".join(_ascii_clean(text).split())


def _is_target_path(path: str) -> bool:
    if not path.endswith(".md"):
        return False
    if path in ROOT_TARGETS:
        return True
    return path.startswith(TARGET_PREFIXES)


def _pick_path(row: dict) -> str:
    candidates = [
        str(row.get("source_markdown", "")).strip(),
        str(row.get("path", "")).strip(),
        str(row.get("markdown", "")).strip(),
    ]
    for candidate in candidates:
        if candidate and candidate.endswith(".md"):
            return candidate
    return ""


def _is_table_separator(stripped: str) -> bool:
    core = stripped.strip("|").strip()
    if not core:
        return False
    return all(ch in "-: " for ch in core)


def _parse_markdown_units(text: str) -> list[Unit]:
    lines = _ascii_clean(text).splitlines()
    units: list[Unit] = []

    paragraph_lines: list[str] = []
    paragraph_start = 0

    in_code = False
    code_start = 0
    code_lines: list[str] = []

    def flush_paragraph(line_end: int) -> None:
        nonlocal paragraph_lines, paragraph_start
        if not paragraph_lines:
            return
        payload = _collapse(" ".join(paragraph_lines))
        if payload:
            units.append(
                Unit(
                    kind="paragraph",
                    text_ascii=payload,
                    line_start=paragraph_start,
                    line_end=max(paragraph_start, line_end),
                    heading_level=0,
                )
            )
        paragraph_lines = []
        paragraph_start = 0

    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()

        if in_code:
            if stripped.startswith("```"):
                payload = "\n".join(code_lines).rstrip("\n")
                payload = _ascii_clean(payload)
                units.append(
                    Unit(
                        kind="code_block",
                        text_ascii=payload,
                        line_start=code_start,
                        line_end=idx,
                        heading_level=0,
                    )
                )
                in_code = False
                code_start = 0
                code_lines = []
            else:
                code_lines.append(raw)
            continue

        if stripped.startswith("```"):
            flush_paragraph(idx - 1)
            in_code = True
            code_start = idx
            code_lines = []
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph(idx - 1)
            level = len(heading_match.group(1))
            payload = _collapse(heading_match.group(2))
            if payload:
                units.append(
                    Unit(
                        kind="heading",
                        text_ascii=payload,
                        line_start=idx,
                        line_end=idx,
                        heading_level=level,
                    )
                )
            continue

        if stripped.startswith("|") and stripped.count("|") >= 2:
            flush_paragraph(idx - 1)
            if not _is_table_separator(stripped):
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                payload = _collapse(" | ".join(cells))
                if payload:
                    units.append(
                        Unit(
                            kind="table_row",
                            text_ascii=payload,
                            line_start=idx,
                            line_end=idx,
                            heading_level=0,
                        )
                    )
            continue

        list_match = LIST_RE.match(stripped)
        if list_match:
            flush_paragraph(idx - 1)
            payload = _collapse(list_match.group(1))
            if payload:
                units.append(
                    Unit(
                        kind="list_item",
                        text_ascii=payload,
                        line_start=idx,
                        line_end=idx,
                        heading_level=0,
                    )
                )
            continue

        if not stripped:
            flush_paragraph(idx - 1)
            continue

        if not paragraph_lines:
            paragraph_start = idx
        paragraph_lines.append(raw)

    if in_code:
        payload = "\n".join(code_lines).rstrip("\n")
        payload = _ascii_clean(payload)
        units.append(
            Unit(
                kind="code_block",
                text_ascii=payload,
                line_start=code_start,
                line_end=max(code_start, len(lines)),
                heading_level=0,
            )
        )

    flush_paragraph(len(lines))
    return units


def _write(path: Path, text: str) -> None:
    _assert_ascii(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _collect_candidates(repo_root: Path) -> dict[str, Candidate]:
    registry_dir = repo_root / "registry"
    files = sorted(p for p in registry_dir.glob("*.toml") if p.is_file())
    best_by_path: dict[str, Candidate] = {}
    duplicate_sources: dict[str, list[str]] = {}

    for file_path in files:
        raw = tomllib.loads(file_path.read_text(encoding="utf-8"))
        rows = raw.get("document", [])
        if not isinstance(rows, list):
            continue
        rel_registry = file_path.relative_to(repo_root).as_posix()
        for row in rows:
            if not isinstance(row, dict):
                continue
            body = str(row.get("body_markdown", ""))
            if not body.strip():
                continue
            path = _pick_path(row)
            if not path or not _is_target_path(path):
                continue
            candidate = Candidate(
                path=path,
                body=body,
                source_registry=rel_registry,
                source_document_id=str(row.get("id", "")).strip(),
                source_title=str(row.get("title", "")).strip(),
            )
            prev = best_by_path.get(path)
            if prev is None:
                best_by_path[path] = candidate
                duplicate_sources[path] = []
                continue
            duplicate_sources[path].append(candidate.source_registry)
            if len(candidate.body) > len(prev.body):
                best_by_path[path] = candidate

    # attach duplicate sources in deterministic order via side-channel metadata later
    return best_by_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--payload-out",
        default="registry/embedded_markdown_payloads.toml",
        help="Output payload metadata path.",
    )
    parser.add_argument(
        "--chunks-out",
        default="registry/embedded_markdown_chunks.toml",
        help="Output chunk path.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    candidates = _collect_candidates(repo_root)

    documents: list[dict] = []
    chunks: list[dict] = []
    kind_counts = {"heading": 0, "paragraph": 0, "list_item": 0, "table_row": 0, "code_block": 0}

    for doc_index, (path, candidate) in enumerate(sorted(candidates.items()), start=1):
        doc_id = f"EMB-{doc_index:05d}"
        body = _ascii_clean(candidate.body)
        body_bytes = body.encode("utf-8")
        content_sha256 = hashlib.sha256(body_bytes).hexdigest()
        line_count = body.count("\n") + (1 if body and not body.endswith("\n") else 0)
        units = _parse_markdown_units(body)
        unit_ids: list[str] = []
        exists_on_disk = (repo_root / path).exists()

        heading_count = 0
        paragraph_count = 0
        list_item_count = 0
        table_row_count = 0
        code_block_count = 0

        for part_index, unit in enumerate(units, start=1):
            chunk_id = f"{doc_id}-U{part_index:04d}"
            unit_ids.append(chunk_id)
            text_sha256 = hashlib.sha256(unit.text_ascii.encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "id": chunk_id,
                    "document_id": doc_id,
                    "chunk_index": part_index,
                    "kind": unit.kind,
                    "line_start": unit.line_start,
                    "line_end": unit.line_end,
                    "heading_level": unit.heading_level,
                    "text_ascii": unit.text_ascii,
                    "text_sha256": text_sha256,
                }
            )
            kind_counts[unit.kind] = kind_counts.get(unit.kind, 0) + 1
            if unit.kind == "heading":
                heading_count += 1
            elif unit.kind == "paragraph":
                paragraph_count += 1
            elif unit.kind == "list_item":
                list_item_count += 1
            elif unit.kind == "table_row":
                table_row_count += 1
            elif unit.kind == "code_block":
                code_block_count += 1

        scope = "root"
        if path.startswith("docs/"):
            scope = "docs"
        elif path.startswith("reports/"):
            scope = "reports"
        elif path.startswith("data/artifacts/"):
            scope = "data_artifacts"

        documents.append(
            {
                "id": doc_id,
                "path": path,
                "scope": scope,
                "exists_on_disk": exists_on_disk,
                "source_registry": candidate.source_registry,
                "source_document_id": candidate.source_document_id,
                "source_title": candidate.source_title,
                "line_count": line_count,
                "size_bytes": len(body_bytes),
                "content_sha256": content_sha256,
                "content_encoding": "structured_toml_units",
                "chunk_count": len(unit_ids),
                "heading_count": heading_count,
                "paragraph_count": paragraph_count,
                "list_item_count": list_item_count,
                "table_row_count": table_row_count,
                "code_block_count": code_block_count,
                "chunk_ids": unit_ids,
            }
        )

    payload_lines: list[str] = [
        "# Embedded markdown structured payload registry (pure TOML units).",
        "# Generated by src/scripts/analysis/build_embedded_markdown_structured_registry.py",
        "",
        "[embedded_markdown_payloads]",
        'updated = "deterministic"',
        "authoritative = true",
        'representation = "structured_toml_units"',
        f"document_count = {len(documents)}",
        f"heading_count = {kind_counts.get('heading', 0)}",
        f"paragraph_count = {kind_counts.get('paragraph', 0)}",
        f"list_item_count = {kind_counts.get('list_item', 0)}",
        f"table_row_count = {kind_counts.get('table_row', 0)}",
        f"code_block_count = {kind_counts.get('code_block', 0)}",
        "",
    ]
    for row in documents:
        payload_lines.extend(
            [
                "[[document]]",
                f"id = {_q(row['id'])}",
                f"path = {_q(row['path'])}",
                f"scope = {_q(row['scope'])}",
                f"exists_on_disk = {'true' if row['exists_on_disk'] else 'false'}",
                f"source_registry = {_q(row['source_registry'])}",
                f"source_document_id = {_q(row['source_document_id'])}",
                f"source_title = {_q(row['source_title'])}",
                f"line_count = {int(row['line_count'])}",
                f"size_bytes = {int(row['size_bytes'])}",
                f"content_sha256 = {_q(row['content_sha256'])}",
                f"content_encoding = {_q(row['content_encoding'])}",
                f"chunk_count = {int(row['chunk_count'])}",
                f"heading_count = {int(row['heading_count'])}",
                f"paragraph_count = {int(row['paragraph_count'])}",
                f"list_item_count = {int(row['list_item_count'])}",
                f"table_row_count = {int(row['table_row_count'])}",
                f"code_block_count = {int(row['code_block_count'])}",
                f"chunk_ids = {_render_list(row['chunk_ids'])}",
                "",
            ]
        )

    chunks.sort(key=lambda item: (item["document_id"], int(item["chunk_index"])))
    chunk_lines: list[str] = [
        "# Embedded markdown structured chunks (pure TOML units).",
        "# Generated by src/scripts/analysis/build_embedded_markdown_structured_registry.py",
        "",
        "[embedded_markdown_chunks]",
        'updated = "deterministic"',
        "authoritative = true",
        'representation = "structured_toml_units"',
        f"chunk_count = {len(chunks)}",
        f"document_count = {len(documents)}",
        "",
    ]
    for row in chunks:
        chunk_lines.extend(
            [
                "[[chunk]]",
                f"id = {_q(row['id'])}",
                f"document_id = {_q(row['document_id'])}",
                f"chunk_index = {int(row['chunk_index'])}",
                f"kind = {_q(row['kind'])}",
                f"line_start = {int(row['line_start'])}",
                f"line_end = {int(row['line_end'])}",
                f"heading_level = {int(row['heading_level'])}",
                f"text_ascii = {_q(row['text_ascii'])}",
                f"text_sha256 = {_q(row['text_sha256'])}",
                "",
            ]
        )

    payload_text = "\n".join(payload_lines)
    chunks_text = "\n".join(chunk_lines)
    _write(repo_root / args.payload_out, payload_text)
    _write(repo_root / args.chunks_out, chunks_text)

    print(
        "Wrote embedded markdown structured registries: "
        f"documents={len(documents)} chunks={len(chunks)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
