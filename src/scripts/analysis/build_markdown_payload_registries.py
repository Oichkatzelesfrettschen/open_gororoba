#!/usr/bin/env python3
"""
Structured markdown-to-TOML conversion registry builder.

Outputs:
- registry/markdown_payloads.toml (document metadata + unit summaries)
- registry/markdown_payload_chunks.toml (typed textual units)

Design goal:
- Preserve markdown document content in pure TOML-native structured units,
  not binary/base64 payload blobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_RE = re.compile(r"^(?:[-*+]\s+|\d+\.\s+)(.+)$")


@dataclass(frozen=True)
class Unit:
    kind: str
    text_ascii: str
    line_start: int
    line_end: int
    heading_level: int


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


def _git_paths(repo_root: Path, args: list[str]) -> set[str]:
    cmd = ["git", *args, "--", "*.md"]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(line.strip() for line in proc.stdout.splitlines() if line.strip())


def _discover_markdown_files(repo_root: Path) -> list[str]:
    paths: list[str] = []
    for path in repo_root.rglob("*.md"):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith(".git/"):
            continue
        paths.append(rel)
    paths.sort()
    return paths


def _origin_class(path: str, generated: bool, third_party: bool) -> str:
    lowered = path.lower()
    if (
        third_party
        or "/venv/" in lowered
        or "/site-packages/" in lowered
        or lowered.startswith(".pytest_cache/")
        or "/.pytest_cache/" in lowered
    ):
        return "third_party_cache"
    if generated:
        return "project_generated"
    return "project_manual"


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
            text_ascii = _collapse(heading_match.group(2))
            units.append(
                Unit(
                    kind="heading",
                    text_ascii=text_ascii,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--inventory-path",
        default="registry/markdown_inventory.toml",
        help="Markdown inventory registry.",
    )
    parser.add_argument(
        "--owner-map-path",
        default="registry/markdown_owner_map.toml",
        help="Markdown owner map registry.",
    )
    parser.add_argument(
        "--payload-out",
        default="registry/markdown_payloads.toml",
        help="Output markdown payload metadata path.",
    )
    parser.add_argument(
        "--chunks-out",
        default="registry/markdown_payload_chunks.toml",
        help="Output markdown payload chunks path.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    inventory_raw = tomllib.loads((repo_root / args.inventory_path).read_text(encoding="utf-8"))
    owner_raw = tomllib.loads((repo_root / args.owner_map_path).read_text(encoding="utf-8"))

    inventory_by_path = {
        str(row.get("path", "")): row for row in inventory_raw.get("document", []) if row.get("path")
    }
    owner_by_path = {
        str(row.get("path", "")): row for row in owner_raw.get("owner", []) if row.get("path")
    }

    tracked = _git_paths(repo_root, ["ls-files"])
    untracked = _git_paths(repo_root, ["ls-files", "--others", "--exclude-standard"])
    ignored = _git_paths(
        repo_root,
        ["ls-files", "--others", "--ignored", "--exclude-standard"],
    )

    markdown_paths = _discover_markdown_files(repo_root)
    documents: list[dict] = []
    chunks: list[dict] = []
    status_counts = {"tracked": 0, "untracked": 0, "ignored": 0, "filesystem_only": 0}
    origin_counts = {"project_manual": 0, "project_generated": 0, "third_party_cache": 0}
    kind_counts = {"heading": 0, "paragraph": 0, "list_item": 0, "table_row": 0, "code_block": 0}

    for idx, rel_path in enumerate(markdown_paths, start=1):
        abs_path = repo_root / rel_path
        raw = abs_path.read_bytes()
        decoded_text = raw.decode("utf-8", errors="ignore")

        sha256 = hashlib.sha256(raw).hexdigest()
        size_bytes = len(raw)
        line_count = decoded_text.count("\n") + (1 if decoded_text and not decoded_text.endswith("\n") else 0)

        inv_row = inventory_by_path.get(rel_path, {})
        owner_row = owner_by_path.get(rel_path, {})
        generated = bool(inv_row.get("generated", False))
        third_party = bool(inv_row.get("third_party", False))
        origin_class = _origin_class(rel_path, generated=generated, third_party=third_party)

        if rel_path in tracked:
            git_status = "tracked"
        elif rel_path in untracked:
            git_status = "untracked"
        elif rel_path in ignored:
            git_status = "ignored"
        else:
            git_status = "filesystem_only"
        status_counts[git_status] = status_counts.get(git_status, 0) + 1
        origin_counts[origin_class] = origin_counts.get(origin_class, 0) + 1

        units = _parse_markdown_units(decoded_text)
        doc_id = f"MPY-{idx:05d}"
        chunk_ids: list[str] = []

        heading_count = 0
        paragraph_count = 0
        list_item_count = 0
        table_row_count = 0
        code_block_count = 0

        for part_idx, unit in enumerate(units, start=1):
            chunk_id = f"{doc_id}-C{part_idx:04d}"
            chunk_ids.append(chunk_id)
            text_sha256 = hashlib.sha256(unit.text_ascii.encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "id": chunk_id,
                    "document_id": doc_id,
                    "chunk_index": part_idx,
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

        documents.append(
            {
                "id": doc_id,
                "path": rel_path,
                "git_status": git_status,
                "origin_class": origin_class,
                "generated": generated,
                "third_party": third_party,
                "canonical_toml_owner": str(owner_row.get("canonical_toml", "")),
                "size_bytes": size_bytes,
                "line_count": line_count,
                "content_sha256": sha256,
                "content_encoding": "structured_toml_units",
                "chunk_count": len(chunk_ids),
                "heading_count": heading_count,
                "paragraph_count": paragraph_count,
                "list_item_count": list_item_count,
                "table_row_count": table_row_count,
                "code_block_count": code_block_count,
                "chunk_ids": chunk_ids,
            }
        )

    payload_lines: list[str] = [
        "# Structured markdown payload registry (pure TOML textual units).",
        "# Generated by src/scripts/analysis/build_markdown_payload_registries.py",
        "",
        "[markdown_payloads]",
        'updated = "deterministic"',
        "authoritative = true",
        "representation = \"structured_toml_units\"",
        f"document_count = {len(documents)}",
        f"tracked_count = {status_counts.get('tracked', 0)}",
        f"untracked_count = {status_counts.get('untracked', 0)}",
        f"ignored_count = {status_counts.get('ignored', 0)}",
        f"filesystem_only_count = {status_counts.get('filesystem_only', 0)}",
        f"project_manual_count = {origin_counts.get('project_manual', 0)}",
        f"project_generated_count = {origin_counts.get('project_generated', 0)}",
        f"third_party_cache_count = {origin_counts.get('third_party_cache', 0)}",
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
                f"git_status = {_q(row['git_status'])}",
                f"origin_class = {_q(row['origin_class'])}",
                f"generated = {'true' if row['generated'] else 'false'}",
                f"third_party = {'true' if row['third_party'] else 'false'}",
                f"canonical_toml_owner = {_q(row['canonical_toml_owner'])}",
                f"size_bytes = {int(row['size_bytes'])}",
                f"line_count = {int(row['line_count'])}",
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

    chunk_lines: list[str] = [
        "# Structured markdown units (pure TOML textual chunks).",
        "# Generated by src/scripts/analysis/build_markdown_payload_registries.py",
        "",
        "[markdown_payload_chunks]",
        'updated = "deterministic"',
        "authoritative = true",
        "representation = \"structured_toml_units\"",
        f"chunk_count = {len(chunks)}",
        f"document_count = {len(documents)}",
        "",
    ]
    chunks.sort(key=lambda item: (item["document_id"], int(item["chunk_index"])))
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
    chunk_text = "\n".join(chunk_lines)
    _write(repo_root / args.payload_out, payload_text)
    _write(repo_root / args.chunks_out, chunk_text)

    print(
        "Wrote structured markdown payload registries: "
        f"documents={len(documents)} chunks={len(chunks)} "
        f"third_party_cache={origin_counts.get('third_party_cache', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
