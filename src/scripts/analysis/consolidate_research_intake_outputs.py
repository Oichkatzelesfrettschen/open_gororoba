#!/usr/bin/env python3
"""
Consolidate research intake outputs into canonical resolved artifacts.

Outputs:
- data/external/intake/.../index_resolved.tsv
- data/external/intake/.../provenance_resolved.toml
- reports/research_intake_artifact_audit_YYYY_MM_DD.toml
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class IndexRow:
    entry_id: str
    topic: str
    resource_class: str
    status: str
    method: str
    size_bytes: int
    sha256: str
    output_path: str
    error: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _escape(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _retry_priority(path: Path) -> int:
    name = path.name
    if "pyo3" in name:
        return 400
    if "python" in name:
        return 350
    if "rust" in name:
        return 300
    if "firefox_profile" in name:
        return 200
    if "firefox" in name:
        return 150
    return 100


def _load_index(path: Path) -> list[IndexRow]:
    rows: list[IndexRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "id",
            "topic",
            "resource_class",
            "status",
            "method",
            "size_bytes",
            "sha256",
            "output_path",
            "error",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"ERROR: {path} missing required index columns: {sorted(missing)}"
            )
        for row in reader:
            entry_id = str(row.get("id", "")).strip()
            if not entry_id:
                continue
            size_raw = str(row.get("size_bytes", "0")).strip()
            try:
                size_val = int(size_raw)
            except ValueError as exc:
                raise SystemExit(
                    f"ERROR: {path} has non-integer size_bytes for id={entry_id}: {size_raw!r}"
                ) from exc
            rows.append(
                IndexRow(
                    entry_id=entry_id,
                    topic=str(row.get("topic", "")).strip(),
                    resource_class=str(row.get("resource_class", "")).strip(),
                    status=str(row.get("status", "")).strip(),
                    method=str(row.get("method", "")).strip(),
                    size_bytes=size_val,
                    sha256=str(row.get("sha256", "")).strip(),
                    output_path=str(row.get("output_path", "")).strip(),
                    error=str(row.get("error", "")).strip(),
                )
            )
    return rows


def _load_registry_order_and_urls(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = list(data.get("entry", []))
    order: list[str] = []
    meta: dict[str, dict[str, str]] = {}
    for row in entries:
        entry_id = str(row.get("id", "")).strip()
        if not entry_id:
            continue
        order.append(entry_id)
        meta[entry_id] = {
            "topic": str(row.get("topic", "")).strip(),
            "resource_class": str(row.get("resource_class", "")).strip(),
            "normalized_url": str(row.get("normalized_url", "")).strip(),
            "primary_source_url": str(row.get("primary_source_url", "")).strip(),
        }
    return order, meta


def _render_index_resolved(
    ordered_ids: list[str],
    resolved: dict[str, IndexRow],
    resolved_from: dict[str, str],
) -> str:
    header = (
        "id\ttopic\tresource_class\tstatus\tmethod\tsize_bytes\tsha256\toutput_path"
        "\terror\tresolved_from"
    )
    lines = [header]
    for entry_id in ordered_ids:
        row = resolved[entry_id]
        error = row.error.replace("\t", " ").replace("\n", " ")
        lines.append(
            "\t".join(
                [
                    row.entry_id,
                    row.topic,
                    row.resource_class,
                    row.status,
                    row.method,
                    str(row.size_bytes),
                    row.sha256,
                    row.output_path,
                    error,
                    resolved_from.get(entry_id, "unknown"),
                ]
            )
        )
    rendered = "\n".join(lines) + "\n"
    _assert_ascii(rendered, "index_resolved.tsv")
    return rendered


def _render_provenance_resolved(
    batch_id: str,
    registry_rel: str,
    output_root_rel: str,
    ordered_ids: list[str],
    resolved: dict[str, IndexRow],
    resolved_from: dict[str, str],
    registry_meta: dict[str, dict[str, str]],
    primary_index_rel: str,
    retry_index_rels: list[str],
    selected_retry_index_rels: list[str],
) -> str:
    success_count = sum(1 for item in ordered_ids if resolved[item].status == "ok")
    failure_count = len(ordered_ids) - success_count

    lines: list[str] = []
    lines.append("# Generated by src/scripts/analysis/consolidate_research_intake_outputs.py")
    lines.append("[batch]")
    lines.append(f"id = {_escape(batch_id)}")
    lines.append(f"generated_at_utc = {_escape(_utc_now())}")
    lines.append(f"registry_path = {_escape(registry_rel)}")
    lines.append(f"output_root = {_escape(output_root_rel)}")
    lines.append(f"primary_index = {_escape(primary_index_rel)}")
    lines.append(
        "retry_indices = ["
        + ", ".join(_escape(item) for item in retry_index_rels)
        + "]"
    )
    lines.append(
        "selected_retry_indices = ["
        + ", ".join(_escape(item) for item in selected_retry_index_rels)
        + "]"
    )
    lines.append(f"entry_count = {len(ordered_ids)}")
    lines.append(f"success_count = {success_count}")
    lines.append(f"failure_count = {failure_count}")
    lines.append("")

    for entry_id in ordered_ids:
        row = resolved[entry_id]
        meta = registry_meta.get(entry_id, {})
        lines.append("[[artifact]]")
        lines.append(f"id = {_escape(entry_id)}")
        lines.append(f"topic = {_escape(row.topic)}")
        lines.append(f"resource_class = {_escape(row.resource_class)}")
        lines.append(f"source_url = {_escape(meta.get('normalized_url', ''))}")
        lines.append(f"primary_source_url = {_escape(meta.get('primary_source_url', ''))}")
        lines.append(f"status = {_escape(row.status)}")
        lines.append(f"method = {_escape(row.method)}")
        lines.append(f"output_path = {_escape(row.output_path)}")
        lines.append(f"sha256 = {_escape(row.sha256)}")
        lines.append(f"size_bytes = {row.size_bytes}")
        lines.append(f"error = {_escape(row.error)}")
        lines.append(f"resolved_from = {_escape(resolved_from.get(entry_id, 'unknown'))}")
        lines.append("")

    rendered = "\n".join(lines)
    _assert_ascii(rendered, "provenance_resolved.toml")
    return rendered


def _render_artifact_audit(
    intake_id: str,
    intake_root_rel: str,
    primary_index_rel: str,
    retry_index_rels: list[str],
    smoke_index_rels: list[str],
    selected_retry_index_rels: list[str],
    selected_ids_by_retry_rel: dict[str, list[str]],
    selected_failure_ids: list[str],
) -> str:
    all_index_rels = [primary_index_rel] + retry_index_rels + smoke_index_rels
    lines: list[str] = []
    lines.append("# Consolidated artifact classification for research intake reruns.")
    lines.append("[audit]")
    lines.append(f"id = {_escape(f'{intake_id}-artifact-audit')}")
    lines.append(f"generated_at_utc = {_escape(_utc_now())}")
    lines.append(f"intake_root = {_escape(intake_root_rel)}")
    lines.append(f"file_count = {len(all_index_rels)}")
    lines.append(
        f"selected_retry_count = {len(selected_retry_index_rels)}"
    )
    resolved_failure_count = sum(len(ids) for ids in selected_ids_by_retry_rel.values())
    lines.append(f"resolved_failure_count = {resolved_failure_count}")
    lines.append("")

    for rel in all_index_rels:
        if rel == primary_index_rel:
            artifact_class = "primary_canonical"
            action = "retain"
        elif rel in smoke_index_rels:
            artifact_class = "smoke_validation_only"
            action = "retain_optional"
        elif rel in selected_retry_index_rels:
            artifact_class = "retry_resolution_canonical"
            action = "retain"
        elif "rust" in rel:
            artifact_class = "retry_obsolete_legacy_rust"
            action = "archive_keep"
        else:
            artifact_class = "retry_obsolete"
            action = "archive_keep"
        lines.append("[[artifact_file]]")
        lines.append(f"path = {_escape(rel)}")
        lines.append(f"class = {_escape(artifact_class)}")
        lines.append(f"recommended_action = {_escape(action)}")
        lines.append("")

    for rel in sorted(selected_ids_by_retry_rel.keys()):
        for entry_id in selected_ids_by_retry_rel[rel]:
            lines.append("[[resolution]]")
            lines.append(f"id = {_escape(entry_id)}")
            lines.append(f"resolved_by = {_escape(rel)}")
            lines.append("")

    for entry_id in selected_failure_ids:
        lines.append("[[unresolved_failure]]")
        lines.append(f"id = {_escape(entry_id)}")
        lines.append("status = \"failed\"")
        lines.append("")

    rendered = "\n".join(lines)
    _assert_ascii(rendered, "research_intake_artifact_audit.toml")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--registry",
        default="registry/research_intake_2026_02_14.toml",
        help="Research intake registry TOML.",
    )
    parser.add_argument(
        "--intake-root",
        default="data/external/intake/2026_02_14_hypercomplex_news",
        help="Intake root directory containing index/provenance artifacts.",
    )
    parser.add_argument(
        "--primary-index",
        default="index.tsv",
        help="Primary index file within intake root.",
    )
    parser.add_argument(
        "--retry-glob",
        default="index_retry*.tsv",
        help="Glob for retry index files.",
    )
    parser.add_argument(
        "--smoke-glob",
        default="index_smoke*.tsv",
        help="Glob for smoke index files.",
    )
    parser.add_argument(
        "--resolved-index-out",
        default="index_resolved.tsv",
        help="Output filename for resolved index within intake root.",
    )
    parser.add_argument(
        "--resolved-provenance-out",
        default="provenance_resolved.toml",
        help="Output filename for resolved provenance within intake root.",
    )
    parser.add_argument(
        "--artifact-audit-out",
        default="reports/research_intake_artifact_audit_2026_02_14.toml",
        help="Output path for artifact-classification audit TOML.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    registry_path = repo_root / args.registry
    intake_root = repo_root / args.intake_root
    if not registry_path.exists():
        raise SystemExit(f"ERROR: missing registry file: {registry_path}")
    if not intake_root.is_dir():
        raise SystemExit(f"ERROR: missing intake root directory: {intake_root}")

    registry_data = tomllib.loads(registry_path.read_text(encoding="utf-8"))
    intake_meta = registry_data.get("research_intake", {})
    intake_id = str(intake_meta.get("id", "unknown-intake"))

    ordered_ids, registry_meta = _load_registry_order_and_urls(registry_path)
    primary_index_path = intake_root / args.primary_index
    if not primary_index_path.exists():
        raise SystemExit(f"ERROR: primary index missing: {primary_index_path}")
    primary_rows = _load_index(primary_index_path)
    primary_by_id = {row.entry_id: row for row in primary_rows}

    retry_paths = sorted(intake_root.glob(args.retry_glob))
    smoke_paths = sorted(intake_root.glob(args.smoke_glob))

    retry_rows_by_id: dict[str, list[tuple[int, float, IndexRow, Path]]] = {}
    for path in retry_paths:
        rows = _load_index(path)
        for row in rows:
            bucket = retry_rows_by_id.setdefault(row.entry_id, [])
            bucket.append((_retry_priority(path), path.stat().st_mtime, row, path))

    resolved: dict[str, IndexRow] = {}
    resolved_from: dict[str, str] = {}
    selected_ids_by_retry_rel: dict[str, list[str]] = {}
    unresolved_failure_ids: list[str] = []

    for entry_id in ordered_ids:
        row = primary_by_id.get(entry_id)
        if row is None:
            candidates = retry_rows_by_id.get(entry_id, [])
            ok_candidates = [item for item in candidates if item[2].status == "ok"]
            if not ok_candidates:
                continue
            best = sorted(ok_candidates, key=lambda item: (item[0], item[1]), reverse=True)[0]
            resolved_row = best[2]
            retry_rel = best[3].relative_to(repo_root).as_posix()
            resolved[entry_id] = resolved_row
            resolved_from[entry_id] = retry_rel
            selected_ids_by_retry_rel.setdefault(retry_rel, []).append(entry_id)
            continue

        resolved_row = row
        source = primary_index_path.relative_to(repo_root).as_posix()
        if row.status != "ok":
            candidates = retry_rows_by_id.get(entry_id, [])
            ok_candidates = [item for item in candidates if item[2].status == "ok"]
            if ok_candidates:
                best = sorted(ok_candidates, key=lambda item: (item[0], item[1]), reverse=True)[0]
                resolved_row = best[2]
                source = best[3].relative_to(repo_root).as_posix()
                selected_ids_by_retry_rel.setdefault(source, []).append(entry_id)
            else:
                unresolved_failure_ids.append(entry_id)
        resolved[entry_id] = resolved_row
        resolved_from[entry_id] = source

    selected_retry_index_rels = sorted(selected_ids_by_retry_rel.keys())

    ordered_resolved_ids = [entry_id for entry_id in ordered_ids if entry_id in resolved]
    for entry_id in sorted(resolved.keys()):
        if entry_id not in ordered_resolved_ids:
            ordered_resolved_ids.append(entry_id)

    resolved_index_text = _render_index_resolved(
        ordered_ids=ordered_resolved_ids,
        resolved=resolved,
        resolved_from=resolved_from,
    )
    resolved_provenance_text = _render_provenance_resolved(
        batch_id=intake_id,
        registry_rel=registry_path.relative_to(repo_root).as_posix(),
        output_root_rel=intake_root.relative_to(repo_root).as_posix(),
        ordered_ids=ordered_resolved_ids,
        resolved=resolved,
        resolved_from=resolved_from,
        registry_meta=registry_meta,
        primary_index_rel=primary_index_path.relative_to(repo_root).as_posix(),
        retry_index_rels=[path.relative_to(repo_root).as_posix() for path in retry_paths],
        selected_retry_index_rels=selected_retry_index_rels,
    )
    artifact_audit_text = _render_artifact_audit(
        intake_id=intake_id,
        intake_root_rel=intake_root.relative_to(repo_root).as_posix(),
        primary_index_rel=primary_index_path.relative_to(repo_root).as_posix(),
        retry_index_rels=[path.relative_to(repo_root).as_posix() for path in retry_paths],
        smoke_index_rels=[path.relative_to(repo_root).as_posix() for path in smoke_paths],
        selected_retry_index_rels=selected_retry_index_rels,
        selected_ids_by_retry_rel=selected_ids_by_retry_rel,
        selected_failure_ids=sorted(unresolved_failure_ids),
    )

    (intake_root / args.resolved_index_out).write_text(
        resolved_index_text,
        encoding="utf-8",
    )
    (intake_root / args.resolved_provenance_out).write_text(
        resolved_provenance_text,
        encoding="utf-8",
    )
    artifact_audit_path = repo_root / args.artifact_audit_out
    artifact_audit_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_audit_path.write_text(artifact_audit_text, encoding="utf-8")

    print(
        "Consolidated intake artifacts: "
        f"entries={len(ordered_resolved_ids)} "
        f"success={sum(1 for item in ordered_resolved_ids if resolved[item].status == 'ok')} "
        f"failure={sum(1 for item in ordered_resolved_ids if resolved[item].status != 'ok')} "
        f"selected_retry_files={len(selected_retry_index_rels)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
