#!/usr/bin/env python3
"""Audit the off-site Cayley-Dickson document cache.

This script inventories the cache under ~/Documents/Projects/CayleyDickson,
checks drift in the off-site chronology and manifest files, and emits a
Markdown report that can be kept in-repo for reproducible follow-up work.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CACHE_ROOT = Path("/home/eirikr/Documents/Projects/CayleyDickson")
DEFAULT_REPO_ROOT = Path("/home/eirikr/Github/open_gororoba")


@dataclass(frozen=True)
class AuditData:
    total_files: int
    pdf_count: int
    markdown_count: int
    html_count: int
    text_count: int
    top_level_counts: list[tuple[str, int]]
    tier1_pdf_counts: list[tuple[str, int]]
    flagged_paths: list[str]
    manifest_total_paths: int
    manifest_missing_paths: list[str]
    manifest_empty_paths: int
    chronology_entry_count: int
    chronology_on_disk_like: int
    chronology_missing: int
    chronology_mislabeled: int
    proofs_theories: int
    proofs_verified: int
    crate_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Path to the Cayley-Dickson cache root.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Path to the open_gororoba repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown output path. Defaults to stdout.",
    )
    return parser.parse_args()


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def read_manifest_local_pdfs(manifest_path: Path) -> list[str]:
    text = manifest_path.read_text(encoding="utf-8")
    return re.findall(r'^local_pdf\s*=\s*"([^"]*)"', text, flags=re.MULTILINE)


def parse_chronology_rows(chronology_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in chronology_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        if cells[0] in {"#", "ID", "Family"}:
            continue
        if set("".join(cells)) <= {"-"}:
            continue
        rows.append(cells[:7])
    return rows


def collect_flagged_paths(paths: Iterable[Path], root: Path) -> list[str]:
    flags = []
    for path in paths:
        rel = str(path.relative_to(root))
        rel_lower = rel.lower()
        if any(
            token in rel_lower
            for token in (
                "_dup.",
                "_mirror.",
                "blocked.html",
                "captcha.html",
                "placeholder",
                "_full.pdf",
                "_full.tex",
            )
        ):
            flags.append(rel)
    return sorted(flags)


def build_audit(cache_root: Path, repo_root: Path) -> AuditData:
    files = iter_files(cache_root)
    pdfs = [path for path in files if path.suffix.lower() == ".pdf"]
    markdown = [path for path in files if path.suffix.lower() == ".md"]
    html = [path for path in files if path.suffix.lower() == ".html"]
    text = [path for path in files if path.suffix.lower() == ".txt"]

    top_level_counts = Counter(path.relative_to(cache_root).parts[0] for path in files)
    tier1_pdf_counts = Counter(
        path.relative_to(cache_root).parts[1]
        for path in pdfs
        if len(path.relative_to(cache_root).parts) >= 2
        and path.relative_to(cache_root).parts[0] == "tier1_core_cd_algebra"
    )

    manifest_path = cache_root / "metadata/repo_extracted_metadata/MANIFEST.toml"
    manifest_local_pdfs = read_manifest_local_pdfs(manifest_path)
    manifest_missing_paths = sorted(
        local_pdf
        for local_pdf in manifest_local_pdfs
        if local_pdf and not (cache_root / local_pdf).exists()
    )
    manifest_empty_paths = sum(1 for local_pdf in manifest_local_pdfs if not local_pdf)

    chronology_rows = parse_chronology_rows(cache_root / "CHRONOLOGICAL_REFERENCE_MATRIX.md")
    chronology_on_disk_like = 0
    chronology_missing = 0
    chronology_mislabeled = 0
    for row in chronology_rows:
        status = row[-1]
        if status.startswith("[ON DISK]") or status.startswith("[FORMALIZED]"):
            chronology_on_disk_like += 1
        elif status.startswith("[MISSING]") or status.startswith("[AUDIT]"):
            chronology_missing += 1
        elif status.startswith("[MISLABELED]"):
            chronology_mislabeled += 1

    proofs_theories = len(list((repo_root / "proofs/theories").glob("*.v")))
    proofs_verified = len(list((repo_root / "proofs/verified").glob("*.v")))
    crate_count = len(list((repo_root / "crates").glob("*/Cargo.toml")))

    return AuditData(
        total_files=len(files),
        pdf_count=len(pdfs),
        markdown_count=len(markdown),
        html_count=len(html),
        text_count=len(text),
        top_level_counts=top_level_counts.most_common(),
        tier1_pdf_counts=tier1_pdf_counts.most_common(),
        flagged_paths=collect_flagged_paths(files, cache_root),
        manifest_total_paths=len(manifest_local_pdfs),
        manifest_missing_paths=manifest_missing_paths,
        manifest_empty_paths=manifest_empty_paths,
        chronology_entry_count=len(chronology_rows),
        chronology_on_disk_like=chronology_on_disk_like,
        chronology_missing=chronology_missing,
        chronology_mislabeled=chronology_mislabeled,
        proofs_theories=proofs_theories,
        proofs_verified=proofs_verified,
        crate_count=crate_count,
    )


def render_markdown(data: AuditData, cache_root: Path, repo_root: Path) -> str:
    lines: list[str] = []
    lines.append("# Cayley-Dickson Cache Audit")
    lines.append("")
    lines.append(f"- Cache root: `{cache_root}`")
    lines.append(f"- Repo root: `{repo_root}`")
    lines.append("")
    lines.append("## Corpus Snapshot")
    lines.append("")
    lines.append(f"- Total files: {data.total_files}")
    lines.append(f"- PDFs: {data.pdf_count}")
    lines.append(f"- Markdown notes: {data.markdown_count}")
    lines.append(f"- HTML traces: {data.html_count}")
    lines.append(f"- Plain-text notes: {data.text_count}")
    lines.append("")
    lines.append("### Top-Level Layout")
    lines.append("")
    for name, count in data.top_level_counts:
        lines.append(f"- `{name}`: {count}")
    lines.append("")
    lines.append("### Tier 1 PDF Density")
    lines.append("")
    for name, count in data.tier1_pdf_counts:
        lines.append(f"- `{name}`: {count} PDFs")
    lines.append("")
    lines.append("## Drift Findings")
    lines.append("")
    lines.append(
        f"- `CHRONOLOGICAL_REFERENCE_MATRIX.md` currently has {data.chronology_entry_count} table entries."
    )
    lines.append(
        "- Chronology row statuses normalize to "
        f"{data.chronology_on_disk_like} on-disk/formalized, "
        f"{data.chronology_missing} missing/audit-needed, and "
        f"{data.chronology_mislabeled} mislabeled."
    )
    lines.append(
        "- `metadata/repo_extracted_metadata/MANIFEST.toml` tracks "
        f"{data.manifest_total_paths} `local_pdf` entries; "
        f"{len(data.manifest_missing_paths)} currently fail to resolve under the cache root and "
        f"{data.manifest_empty_paths} are blank placeholders."
    )
    lines.append(
        "- The highest-density actionable source lanes remain `cd_tower_structure`, "
        "`zero_divisors_geometry`, and `quaternion_family`."
    )
    lines.append("")
    lines.append("### Flagged Files")
    lines.append("")
    for rel in data.flagged_paths:
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("### Missing Manifest Paths")
    lines.append("")
    if data.manifest_missing_paths:
        for rel in data.manifest_missing_paths[:40]:
            lines.append(f"- `{rel}`")
        remaining_missing = len(data.manifest_missing_paths) - 40
        if remaining_missing > 0:
            lines.append(f"- ... and {remaining_missing} more")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Repo Surface Crosswalk")
    lines.append("")
    lines.append(f"- Rocq theory files: {data.proofs_theories}")
    lines.append(f"- Rocq verified files: {data.proofs_verified}")
    lines.append(f"- Rust crates: {data.crate_count}")
    lines.append("")
    lines.append("### Highest-Leverage Mining Lanes")
    lines.append("")
    lines.append(
        "- `tier1_core_cd_algebra/cd_tower_structure/` -> "
        "`proofs/theories/DeMarraisAssessors.v`, `proofs/theories/WilmotCDStructure.v`, "
        "`crates/de_marrais_2000/`"
    )
    lines.append(
        "- `tier1_core_cd_algebra/zero_divisors_geometry/` -> "
        "`proofs/theories/ZDGraph.v`, `proofs/theories/ZD_Criterion.v`, `crates/brown_1972/`"
    )
    lines.append(
        "- `tier1_core_cd_algebra/foundational_legacy/` and "
        "`tier1_core_cd_algebra/foundational_followups/` -> "
        "`proofs/theories/HurwitzTheorem.v`, `proofs/theories/BrownGeneralizedCD.v`, "
        "`crates/dickson_1919/`, `crates/brown_1967/`"
    )
    lines.append(
        "- `tier1_core_cd_algebra/g2_su3_fano_validation/` and "
        "`tier1_core_cd_algebra/interleaved_generation_physics/` -> "
        "`proofs/theories/G2StabilizerDimension.v`, `proofs/theories/SU3StructureConstants.v`, "
        "`docs/physics/sedenion_standard_model.md`"
    )
    lines.append("")
    lines.append("## Recommended Next Cleanup")
    lines.append("")
    if data.manifest_missing_paths:
        lines.append(
            "- Repoint the remaining unresolved `local_pdf` entries in "
            "`metadata/repo_extracted_metadata/MANIFEST.toml` to cache-relative paths or mark them blank."
        )
    else:
        lines.append(
            "- Preserve manifest normalization by keeping `local_pdf` cache-relative and blank only when the PDF is truly absent from the snapshot."
        )
    lines.append(
        "- Keep chronology totals derived from the table rows rather than hand-maintained summary bullets."
    )
    lines.append(
        "- Separate Tier 1 algebra sources from `unrelated_physics_engineering/` so mining scripts can stay focused."
    )
    lines.append(
        "- Promote blocked/captcha traces into explicit provenance records so fetch failures remain auditable."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    audit = build_audit(args.cache_root, args.repo_root)
    markdown = render_markdown(audit, args.cache_root, args.repo_root)
    if args.output is None:
        print(markdown, end="")
    else:
        args.output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
