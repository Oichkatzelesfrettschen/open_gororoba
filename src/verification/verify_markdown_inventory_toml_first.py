#!/usr/bin/env python3
"""
Verify that markdown inventory remains TOML-first.

Policy:
- Tracked project markdown must classify as toml_published_markdown.
- Ignored project markdown may classify as toml_destination_exists_manual_markdown
  while migration is in progress, but must have explicit TOML destination.
- Non-project markdown may classify as third_party_markdown.
- No unbacked manual markdown is allowed.
"""

from __future__ import annotations

from pathlib import Path
import tomllib


ALLOWED = {
    "toml_published_markdown",
    "toml_destination_exists_manual_markdown",
    "third_party_markdown",
    "generated_artifact",
}
TRACKED_ALLOWLIST = {
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "README.md",
    "PANTHEON_PHYSICSFORGE_90_POINT_MIGRATION_PLAN.md",
    "PHASE10_11_ULTIMATE_ROADMAP.md",
    "PYTHON_REFACTORING_ROADMAP.md",
    "docs/ALGEBRA_CONSTRUCTION_COMPARISON.md",
    "docs/ALGEBRA_CRATES_SURVEY.md",
    "docs/ALGEBRA_FAMILY_TAXONOMY.md",
    "docs/BIBLIOGRAPHY.md",
    "docs/CLAIMS_EVIDENCE_MATRIX.md",
    "docs/CLAIMS_EVIDENCE_MATRIX_NOTES_2026-02-01.md",
    "docs/COMPLETE_ALGEBRA_FAMILY_TAXONOMY.md",
    "docs/CONVOS_CONCEPTS_STATUS_INDEX.md",
    "docs/CORE_ABSTRACTIONS.md",
    "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md",
    "docs/FINAL_MANUSCRIPT.md",
    "docs/GRAND_SYNTHESIS.md",
    "docs/GRAND_SYNTHESIS_PLAN.md",
    "docs/INSIGHTS.md",
    "docs/PHASE4_UNIFIED_PHYSICS_FRAMEWORK.md",
    "docs/PHASE_9_TESSARINES_RESEARCH.md",
    "docs/ROADMAP.md",
    "docs/claims/by_domain/holography.md",
    "docs/claims/by_domain/meta.md",
    "docs/convos/audit_1_read_nonuser_lines_cont.md",
    "docs/external_sources/DE_MARRAIS_BOXKITES_III.md",
    "docs/external_sources/DE_MARRAIS_FLYING_HIGHER.md",
    "docs/external_sources/DE_MARRAIS_PLACEHOLDER_I.md",
    "docs/external_sources/DE_MARRAIS_PRESTO_DIGITIZATION.md",
    "docs/external_sources/DE_MARRAIS_WOLFRAM_SLIDES.md",
    "reports/convos_claim_candidates.md",
    "curated/README.md",
    "curated/01_theory_frameworks/README_COQ.md",
    "data/artifacts/README.md",
    "data/csv/README.md",
}

POLICY_PREFIXES = (
    "docs/",
    "reports/",
    "data/artifacts/",
)


def _is_policy_path(path: str) -> bool:
    if any(path.startswith(prefix) for prefix in POLICY_PREFIXES):
        return True
    return path in TRACKED_ALLOWLIST


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    inv_path = repo_root / "registry/markdown_inventory.toml"
    data = tomllib.loads(inv_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    summary = data.get("markdown_inventory", {})
    for row in data.get("document", []):
        path = str(row.get("path", "")).strip()
        if not _is_policy_path(path):
            continue
        git_status = str(row.get("git_status", "")).strip()
        classification = str(row.get("classification", "")).strip()
        destination = str(row.get("toml_destination", "")).strip()
        generated_declared = bool(row.get("generated_declared", False))
        if git_status in {"untracked", "filesystem_only"}:
            failures.append(f"{path}: policy markdown must be tracked (git_status={git_status})")
            continue
        if classification not in ALLOWED:
            failures.append(f"{path}: disallowed classification={classification}")
        if classification == "generated_artifact":
            if not path.startswith("build/docs/generated/"):
                failures.append(
                    f"{path}: generated_artifact allowed only under build/docs/generated/"
                )
            if git_status == "tracked":
                failures.append(f"{path}: generated_artifact must not be tracked")
            continue
        if classification == "toml_destination_exists_manual_markdown":
            if git_status == "tracked":
                failures.append(
                    f"{path}: manual markdown with TOML destination must not be tracked"
                )
            if not destination:
                failures.append(f"{path}: missing toml_destination for manual markdown")
            elif not (repo_root / destination).is_file():
                failures.append(f"{path}: missing toml_destination file {destination}")
            continue
        if git_status == "tracked" and path not in TRACKED_ALLOWLIST:
            failures.append(f"{path}: tracked markdown outside allowlist")
        if classification == "toml_published_markdown":
            if not generated_declared and not path.startswith("build/docs/generated/"):
                failures.append(
                    f"{path}: toml_published_markdown without explicit generated marker header"
                )
            if not destination:
                failures.append(f"{path}: toml_published_markdown without toml_destination")
            elif not (repo_root / destination).is_file():
                failures.append(f"{path}: missing toml_destination file {destination}")

    if failures:
        print("ERROR: markdown inventory violates TOML-first policy.")
        for item in failures:
            print(f"- {item}")
        return 1

    print("OK: markdown inventory is TOML-first (project markdown) with only third-party exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
