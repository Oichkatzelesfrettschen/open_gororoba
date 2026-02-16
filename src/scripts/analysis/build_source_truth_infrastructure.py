#!/usr/bin/env python3
"""
Build modular source infrastructure lanes from the authoritative artifact source-of-truth.

Input:
- registry/artifact_source_of_truth.toml

Outputs:
- registry/source_infrastructure.toml
- registry/source_lanes/papers_pdf.toml
- registry/source_lanes/datasets.toml
- registry/source_lanes/slides_artifacts.toml
- registry/source_lanes/web_references.toml
- reports/source_infrastructure_reconciliation_2026_02_15.toml
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


DATASET_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".parquet",
    ".h5",
    ".hdf5",
    ".nc",
    ".npy",
    ".npz",
    ".feather",
    ".xlsx",
    ".xls",
}

SLIDE_ARTIFACT_EXTENSIONS = {
    ".ppt",
    ".pptx",
    ".odp",
    ".key",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".ipynb",
    ".doc",
    ".docx",
}

PDF_EXTENSIONS = {".pdf"}

LANE_ORDER = [
    "datasets",
    "slides_artifacts",
    "papers_pdf",
    "web_references",
]

LANE_DESCRIPTIONS = {
    "datasets": "Numerical or tabular research datasets and machine-readable data artifacts.",
    "slides_artifacts": "Slides, decks, archives, notebooks, and non-dataset non-paper binary artifacts.",
    "papers_pdf": "Paper-oriented references with PDF documents or PDF mirrors.",
    "web_references": "Reference URLs without locally identified PDF/data/artifact files.",
}

BEST_PRACTICE_SOURCES = [
    "https://www.w3.org/TR/prov-overview/",
    "https://www.nature.com/articles/sdata201618",
    "https://doi.org/10.25490/a97f-egyk",
    "https://schema.datacite.org/meta/kernel-4.5/",
    "https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files",
    "https://openlineage.io/docs/",
]


def _ascii_sanitize(text: str) -> str:
    out: list[str] = []
    for ch in str(text):
        code = ord(ch)
        if code >= 128:
            out.append(" ")
            continue
        if code < 32 and ch not in {"\n", "\r", "\t"}:
            out.append(" ")
            continue
        if code == 127:
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        raise SystemExit(f"ERROR: non-ASCII output in {context}: {''.join(bad[:20])!r}")


def _escape_toml(text: str) -> str:
    text = _ascii_sanitize(text)
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
    return "[" + ", ".join(_escape_toml(v) for v in values) + "]"


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _value_endswith_any(value: str, extensions: set[str]) -> bool:
    lowered = value.lower()
    # Match both direct suffix and common URL forms with query strings.
    for ext in extensions:
        if lowered.endswith(ext):
            return True
        needle = f"{ext}?"
        if needle in lowered:
            return True
    return False


def classify_lane(artifact: dict) -> tuple[str, list[str]]:
    values = []
    values.extend(str(v).strip() for v in artifact.get("all_links", []) if str(v).strip())
    values.extend(str(v).strip() for v in artifact.get("downloaded_paths", []) if str(v).strip())
    canonical_url = str(artifact.get("canonical_functional_url", "")).strip()
    canonical_path = str(artifact.get("canonical_download_path", "")).strip()
    if canonical_url:
        values.append(canonical_url)
    if canonical_path:
        values.append(canonical_path)

    has_dataset = any(_value_endswith_any(v, DATASET_EXTENSIONS) for v in values)
    has_slide_artifact = any(_value_endswith_any(v, SLIDE_ARTIFACT_EXTENSIONS) for v in values)
    has_pdf = any(_value_endswith_any(v, PDF_EXTENSIONS) for v in values)

    tags: list[str] = []
    if has_dataset:
        tags.append("datasets")
    if has_slide_artifact:
        tags.append("slides_artifacts")
    if has_pdf:
        tags.append("papers_pdf")
    if not tags:
        tags.append("web_references")

    if has_dataset:
        primary = "datasets"
    elif has_slide_artifact:
        primary = "slides_artifacts"
    elif has_pdf:
        primary = "papers_pdf"
    else:
        primary = "web_references"

    return primary, _dedupe(tags)


def render_lane(name: str, artifacts: list[dict], generated_at: str) -> str:
    counts = Counter(str(a.get("status", "")).strip() for a in artifacts)
    missing_minimum = sum(1 for a in artifacts if not bool(a.get("minimum_requirement_met", False)))

    lines: list[str] = []
    lines.append(f"# Lane: {name}")
    lines.append("")
    lines.append("[lane]")
    lines.append(f"id = {_escape_toml(f'SLANE-{name.upper()}-2026-02-15')}")
    lines.append(f"name = {_escape_toml(name)}")
    lines.append(f"description = {_escape_toml(LANE_DESCRIPTIONS[name])}")
    lines.append(f"generated_at = {_escape_toml(generated_at)}")
    lines.append(f"artifact_count = {len(artifacts)}")
    lines.append(f"downloaded_count = {counts.get('downloaded', 0)}")
    lines.append(f"downloadable_count = {counts.get('downloadable', 0)}")
    lines.append(f"blocked_count = {counts.get('blocked', 0)}")
    lines.append(f"citation_only_no_link_count = {counts.get('citation_only_no_link', 0)}")
    lines.append(f"unverified_count = {counts.get('unverified', 0)}")
    lines.append(f"missing_minimum_requirement_count = {missing_minimum}")
    lines.append("")

    for artifact in artifacts:
        lines.append("[[artifact_ref]]")
        lines.append(f"id = {_escape_toml(str(artifact.get('id', '')).strip())}")
        lines.append(f"key = {_escape_toml(str(artifact.get('key', '')).strip())}")
        lines.append(f"title = {_escape_toml(str(artifact.get('title', '')).strip())}")
        lines.append(f"status = {_escape_toml(str(artifact.get('status', '')).strip())}")
        lines.append(
            f"minimum_requirement_met = {str(bool(artifact.get('minimum_requirement_met', False))).lower()}"
        )
        lines.append(
            f"canonical_functional_url = {_escape_toml(str(artifact.get('canonical_functional_url', '')).strip())}"
        )
        lines.append(
            f"canonical_download_path = {_escape_toml(str(artifact.get('canonical_download_path', '')).strip())}"
        )
        lines.append(
            f"source_refs = {_render_list([str(v).strip() for v in artifact.get('source_refs', []) if str(v).strip()])}"
        )
        lines.append("")

    return "\n".join(lines)


def render_infrastructure(
    source_path: str,
    lane_files: dict[str, str],
    lane_counts: dict[str, int],
    total_artifacts: int,
    generated_at: str,
) -> str:
    lines: list[str] = []
    lines.append("# Canonical source infrastructure manifest.")
    lines.append("")
    lines.append("[source_infrastructure]")
    lines.append('id = "SINFRA-2026-02-15"')
    lines.append(f"generated_at = {_escape_toml(generated_at)}")
    lines.append("authoritative = true")
    lines.append("policy_version = 1")
    lines.append(
        'policy = "artifact_source_of_truth.toml is authoritative; lane files are deterministic projections and must never diverge from master."'
    )
    lines.append(
        'best_practice = "single authoritative master, deterministic generated lanes, explicit blocked/manual intervention tracking, provenance-preserving mirrors, reproducible verification gates."'
    )
    lines.append(f"best_practice_sources = {_render_list(BEST_PRACTICE_SOURCES)}")
    lines.append(f"master_registry = {_escape_toml(source_path)}")
    lines.append(f"lane_count = {len(LANE_ORDER)}")
    lines.append(f"total_artifact_count = {total_artifacts}")
    lines.append("")

    for lane in LANE_ORDER:
        lines.append("[[lane]]")
        lines.append(f"name = {_escape_toml(lane)}")
        lines.append(f"description = {_escape_toml(LANE_DESCRIPTIONS[lane])}")
        lines.append(f"path = {_escape_toml(lane_files[lane])}")
        lines.append(f"artifact_count = {lane_counts.get(lane, 0)}")
        lines.append("")

    return "\n".join(lines)


def render_report(lane_counts: dict[str, int], total_artifacts: int, generated_at: str) -> str:
    lines: list[str] = []
    lines.append("# Reconciliation report for source infrastructure lanes.")
    lines.append("")
    lines.append("[report]")
    lines.append('id = "SINFRA-RECON-2026-02-15"')
    lines.append(f"generated_at = {_escape_toml(generated_at)}")
    lines.append("authoritative = true")
    lines.append(f"total_artifact_count = {total_artifacts}")
    lines.append(f"lane_total_count = {sum(lane_counts.values())}")
    lines.append("")

    for lane in LANE_ORDER:
        lines.append("[[lane_summary]]")
        lines.append(f"name = {_escape_toml(lane)}")
        lines.append(f"artifact_count = {lane_counts.get(lane, 0)}")
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
        "--source",
        default="registry/artifact_source_of_truth.toml",
        help="Authoritative master registry path.",
    )
    parser.add_argument(
        "--out-infrastructure",
        default="registry/source_infrastructure.toml",
        help="Output source infrastructure manifest.",
    )
    parser.add_argument(
        "--lane-dir",
        default="registry/source_lanes",
        help="Directory for lane outputs.",
    )
    parser.add_argument(
        "--out-report",
        default="reports/source_infrastructure_reconciliation_2026_02_15.toml",
        help="Output reconciliation report.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source_path = repo_root / args.source
    out_infrastructure = repo_root / args.out_infrastructure
    lane_dir = repo_root / args.lane_dir
    out_report = repo_root / args.out_report

    if not source_path.exists():
        raise SystemExit(f"ERROR: missing source registry: {source_path}")

    data = _load_toml(source_path)
    artifacts = list(data.get("artifact", []))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lane_map: dict[str, list[dict]] = {lane: [] for lane in LANE_ORDER}
    for artifact in artifacts:
        primary, tags = classify_lane(artifact)
        artifact["lane_primary"] = primary
        artifact["lane_tags"] = tags
        lane_map[primary].append(artifact)

    lane_dir.mkdir(parents=True, exist_ok=True)
    lane_files: dict[str, str] = {}
    lane_counts: dict[str, int] = {}

    for lane in LANE_ORDER:
        lane_artifacts = lane_map[lane]
        lane_artifacts.sort(key=lambda a: str(a.get("id", "")))
        lane_text = render_lane(lane, lane_artifacts, generated_at)
        lane_path = lane_dir / f"{lane}.toml"
        _assert_ascii(lane_text, str(lane_path))
        lane_path.write_text(lane_text, encoding="utf-8")
        lane_files[lane] = lane_path.relative_to(repo_root).as_posix()
        lane_counts[lane] = len(lane_artifacts)

    infrastructure_text = render_infrastructure(
        source_path=source_path.relative_to(repo_root).as_posix(),
        lane_files=lane_files,
        lane_counts=lane_counts,
        total_artifacts=len(artifacts),
        generated_at=generated_at,
    )
    report_text = render_report(
        lane_counts=lane_counts,
        total_artifacts=len(artifacts),
        generated_at=generated_at,
    )

    _assert_ascii(infrastructure_text, str(out_infrastructure))
    _assert_ascii(report_text, str(out_report))

    out_infrastructure.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_infrastructure.write_text(infrastructure_text, encoding="utf-8")
    out_report.write_text(report_text, encoding="utf-8")

    print(
        "Wrote source infrastructure manifest: "
        f"{out_infrastructure.relative_to(repo_root).as_posix()}"
    )
    for lane in LANE_ORDER:
        print(
            "Wrote lane: "
            f"{lane_files[lane]} artifacts={lane_counts[lane]}"
        )
    print(
        "Wrote source infrastructure reconciliation report: "
        f"{out_report.relative_to(repo_root).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
