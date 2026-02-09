#!/usr/bin/env python3
"""
Build a full CSV inventory registry (tracked, untracked, ignored, archived).

This complements markdown inventory work by establishing CSV corpus visibility,
including gitignored and archived files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CsvDoc:
    path: str
    git_status: str
    zone: str
    archived: bool
    generated: bool
    size_bytes: int
    line_count: int
    sha256: str
    migration_action: str
    migration_priority: str
    rationale: str


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _esc(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _git_paths(root: Path, args: list[str]) -> set[str]:
    out = subprocess.check_output(["git", *args], cwd=root, text=True)
    return {line.strip() for line in out.splitlines() if line.strip()}


def _all_filesystem_csv(root: Path) -> set[str]:
    out: set[str] = set()
    for path in root.rglob("*.csv"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        out.add(rel)
    return out


def _status(path: str, tracked: set[str], untracked: set[str], ignored: set[str]) -> str:
    if path in tracked:
        return "tracked"
    if path in untracked:
        return "untracked"
    if path in ignored:
        return "ignored"
    return "unknown"


def _zone(path: str) -> str:
    if path.startswith("data/csv/legacy/"):
        return "legacy_csv"
    if path.startswith("data/csv/"):
        return "project_csv"
    if path.startswith("data/external/"):
        return "external_csv"
    if path.startswith("curated/"):
        return "curated_csv"
    if path.startswith("archive/") or path.startswith("docs/archive/"):
        return "archive_csv"
    return "other_csv"


def _policy(path: str, zone: str) -> tuple[str, str, str]:
    if zone == "legacy_csv":
        return (
            "migrate_to_toml_canonical",
            "critical",
            "Legacy CSV should become TOML-native canonical data.",
        )
    if zone == "project_csv":
        return (
            "evaluate_for_toml_canonical",
            "high",
            "Project CSV may be generated artifacts or transition candidates.",
        )
    if zone == "curated_csv":
        return (
            "plan_curated_ingest",
            "high",
            "Curated observational CSV should move under central TOML data policy.",
        )
    if zone == "external_csv":
        return (
            "track_provenance_only",
            "medium",
            "External CSV remains provenance-managed input unless explicitly curated.",
        )
    if zone == "archive_csv":
        return (
            "review_archive_policy",
            "low",
            "Archived CSV may be retained as historical snapshots.",
        )
    return (
        "manual_triage",
        "medium",
        "CSV outside expected zones; requires classification.",
    )


def _load_canonical_source_paths(root: Path, index_rel: str, table_name: str) -> set[str]:
    index_path = root / index_rel
    if not index_path.exists():
        return set()
    parsed = tomllib.loads(index_path.read_text(encoding="utf-8"))
    # Registry metadata lives under [table_name], while records are emitted
    # as top-level [[dataset]] arrays by migration tooling.
    rows = parsed.get("dataset", [])
    if not rows:
        table_value = parsed.get(table_name, [])
        if isinstance(table_value, list):
            rows = table_value
    out: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source_path", row.get("source_csv", row.get("path", "")))).strip()
        if source:
            out.add(source)
    return out


def _load_project_split_classification(root: Path) -> dict[str, str]:
    policy_path = root / "registry/project_csv_split_policy.toml"
    if not policy_path.exists():
        return {}
    parsed = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for row in parsed.get("dataset", []):
        if not isinstance(row, dict):
            continue
        path = str(row.get("path", "")).strip()
        classification = str(row.get("classification", "")).strip()
        if path and classification:
            out[path] = classification
    return out


def _policy_with_progress(
    path: str,
    zone: str,
    legacy_canonical_paths: set[str],
    curated_canonical_paths: set[str],
    project_canonical_paths: set[str],
    project_generated_paths: set[str],
    external_holding_paths: set[str],
    archive_holding_paths: set[str],
    external_holding_scroll_paths: set[str],
    archive_holding_scroll_paths: set[str],
    project_split_classification: dict[str, str],
) -> tuple[str, str, str]:
    if zone == "legacy_csv" and path in legacy_canonical_paths:
        return (
            "canonicalized_to_toml",
            "complete",
            "Legacy CSV is already canonicalized under registry/data/legacy_csv.",
        )
    if zone == "curated_csv" and path in curated_canonical_paths:
        return (
            "canonicalized_to_toml",
            "complete",
            "Curated CSV is already canonicalized under registry/data/curated_csv.",
        )
    if zone == "project_csv" and path in project_canonical_paths:
        return (
            "canonicalized_to_toml",
            "complete",
            "Project canonical dataset is represented in registry/data/project_csv/canonical.",
        )
    if zone == "project_csv" and path in project_generated_paths:
        return (
            "canonicalized_to_toml_generated_artifact",
            "complete",
            "Project generated artifact is represented in registry/data/project_csv/generated.",
        )
    if zone == "external_csv" and path in external_holding_scroll_paths:
        return (
            "canonicalized_to_toml_holding",
            "complete",
            "External CSV holding source is represented in registry/data/external_csv_holding.",
        )
    if zone == "archive_csv" and path in archive_holding_scroll_paths:
        return (
            "canonicalized_to_toml_holding",
            "complete",
            "Archive CSV holding source is represented in registry/data/archive_csv_holding.",
        )
    if zone == "external_csv" and path in external_holding_paths:
        return (
            "queued_for_scroll_holding",
            "high",
            "External CSV is queued in holding registry for TOML scroll conversion.",
        )
    if zone == "archive_csv" and path in archive_holding_paths:
        return (
            "queued_for_scroll_holding",
            "high",
            "Archive CSV is queued in holding registry for TOML scroll conversion.",
        )
    if zone == "project_csv":
        classification = project_split_classification.get(path, "")
        if classification == "generated_artifact":
            return (
                "preserve_generated_artifact",
                "high",
                "Project CSV classified as generated artifact and pending/retained under scroll policy.",
            )
        if classification == "canonical_dataset":
            return (
                "migrate_to_toml_canonical",
                "high",
                "Project CSV classified as canonical dataset pending/active TOML migration.",
            )
    return _policy(path, zone)


def _render(docs: list[CsvDoc]) -> str:
    tracked_count = sum(1 for d in docs if d.git_status == "tracked")
    untracked_count = sum(1 for d in docs if d.git_status == "untracked")
    ignored_count = sum(1 for d in docs if d.git_status == "ignored")
    archived_count = sum(1 for d in docs if d.archived)
    legacy_count = sum(1 for d in docs if d.zone == "legacy_csv")
    curated_count = sum(1 for d in docs if d.zone == "curated_csv")
    canonicalized_count = sum(1 for d in docs if d.migration_action == "canonicalized_to_toml")
    generated_scroll_count = sum(
        1 for d in docs if d.migration_action == "canonicalized_to_toml_generated_artifact"
    )
    holding_scroll_count = sum(
        1 for d in docs if d.migration_action == "canonicalized_to_toml_holding"
    )
    holding_queue_count = sum(1 for d in docs if d.migration_action == "queued_for_scroll_holding")

    lines: list[str] = []
    lines.append("# Full CSV inventory registry (tracked/untracked/ignored/archived).")
    lines.append("# Generated by src/scripts/analysis/build_csv_inventory_registry.py")
    lines.append("")
    lines.append("[csv_inventory]")
    lines.append('updated = "2026-02-09"')
    lines.append("authoritative = true")
    lines.append(f"document_count = {len(docs)}")
    lines.append(f"tracked_count = {tracked_count}")
    lines.append(f"untracked_count = {untracked_count}")
    lines.append(f"ignored_count = {ignored_count}")
    lines.append(f"archived_count = {archived_count}")
    lines.append(f"legacy_count = {legacy_count}")
    lines.append(f"curated_count = {curated_count}")
    lines.append(f"canonicalized_count = {canonicalized_count}")
    lines.append(f"generated_scroll_count = {generated_scroll_count}")
    lines.append(f"holding_scroll_count = {holding_scroll_count}")
    lines.append(f"holding_queue_count = {holding_queue_count}")
    lines.append("")

    for doc in docs:
        lines.append("[[document]]")
        lines.append(f"path = {_esc(doc.path)}")
        lines.append(f"git_status = {_esc(doc.git_status)}")
        lines.append(f"zone = {_esc(doc.zone)}")
        lines.append(f"archived = {str(doc.archived).lower()}")
        lines.append(f"generated = {str(doc.generated).lower()}")
        lines.append(f"size_bytes = {doc.size_bytes}")
        lines.append(f"line_count = {doc.line_count}")
        lines.append(f"sha256 = {_esc(doc.sha256)}")
        lines.append(f"migration_action = {_esc(doc.migration_action)}")
        lines.append(f"migration_priority = {_esc(doc.migration_priority)}")
        lines.append(f"rationale = {_esc(doc.rationale)}")
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
        "--out",
        default="registry/csv_inventory.toml",
        help="Output TOML path relative to repo root.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()

    tracked = _git_paths(root, ["ls-files", "*.csv"])
    untracked = _git_paths(root, ["ls-files", "--others", "--exclude-standard", "*.csv"])
    ignored = _git_paths(
        root,
        ["ls-files", "--others", "--ignored", "--exclude-standard", "*.csv"],
    )
    legacy_canonical_paths = _load_canonical_source_paths(
        root, "registry/legacy_csv_datasets.toml", "legacy_csv_datasets"
    )
    curated_canonical_paths = _load_canonical_source_paths(
        root, "registry/curated_csv_datasets.toml", "curated_csv_datasets"
    )
    project_canonical_paths = _load_canonical_source_paths(
        root,
        "registry/project_csv_canonical_datasets.toml",
        "project_csv_canonical_datasets",
    )
    project_generated_paths = _load_canonical_source_paths(
        root,
        "registry/project_csv_generated_artifacts.toml",
        "project_csv_generated_artifacts",
    )
    external_holding_paths = _load_canonical_source_paths(
        root, "registry/external_csv_holding.toml", "external_csv_holding"
    )
    archive_holding_paths = _load_canonical_source_paths(
        root, "registry/archive_csv_holding.toml", "archive_csv_holding"
    )
    external_holding_scroll_paths = _load_canonical_source_paths(
        root, "registry/external_csv_holding_datasets.toml", "external_csv_holding_datasets"
    )
    archive_holding_scroll_paths = _load_canonical_source_paths(
        root, "registry/archive_csv_holding_datasets.toml", "archive_csv_holding_datasets"
    )
    project_split_classification = _load_project_split_classification(root)

    files = sorted(_all_filesystem_csv(root))

    docs: list[CsvDoc] = []
    for rel in files:
        path = root / rel
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="ignore")
        zone = _zone(rel)
        action, priority, rationale = _policy_with_progress(
            rel,
            zone,
            legacy_canonical_paths,
            curated_canonical_paths,
            project_canonical_paths,
            project_generated_paths,
            external_holding_paths,
            archive_holding_paths,
            external_holding_scroll_paths,
            archive_holding_scroll_paths,
            project_split_classification,
        )
        classification = project_split_classification.get(rel, "")
        docs.append(
            CsvDoc(
                path=rel,
                git_status=_status(rel, tracked, untracked, ignored),
                zone=zone,
                archived=rel.startswith("archive/") or rel.startswith("docs/archive/"),
                generated=(
                    classification == "generated_artifact"
                    if zone == "project_csv"
                    else rel.startswith("data/csv/") and not rel.startswith("data/csv/legacy/")
                ),
                size_bytes=len(raw),
                line_count=text.count("\n") + (1 if text else 0),
                sha256=hashlib.sha256(raw).hexdigest(),
                migration_action=action,
                migration_priority=priority,
                rationale=rationale,
            )
        )

    out_path = root / args.out
    rendered = _render(docs)
    _assert_ascii(rendered, str(out_path))
    out_path.write_text(rendered, encoding="utf-8")

    print(f"Wrote {out_path} with {len(docs)} CSV entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
