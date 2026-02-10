#!/usr/bin/env python3
"""
Verify canonical artifact scroll registries.

Hard gates:
- registry/artifact_scrolls.toml exists and is internally consistent.
- Every document in registry/data_artifact_narratives.toml has a scroll entry.
- Per-scroll counts match on-disk structured tables.
- Equation/proof reference counts reconcile with knowledge atom registries.
"""

from __future__ import annotations

import argparse
import tomllib
from collections import defaultdict
from pathlib import Path


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII content in {path}: {sample!r}")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--index-path",
        default="registry/artifact_scrolls.toml",
        help="Artifact scroll index path.",
    )
    parser.add_argument(
        "--source-registry",
        default="registry/data_artifact_narratives.toml",
        help="Narratives source registry path.",
    )
    parser.add_argument(
        "--equation-registry",
        default="registry/knowledge/equation_atoms.toml",
        help="Equation atoms registry path.",
    )
    parser.add_argument(
        "--proof-registry",
        default="registry/knowledge/proof_atoms.toml",
        help="Proof atoms registry path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    index_path = root / args.index_path
    source_path = root / args.source_registry
    equation_path = root / args.equation_registry
    proof_path = root / args.proof_registry

    failures: list[str] = []
    for path in (index_path, source_path, equation_path, proof_path):
        if not path.exists():
            failures.append(f"missing registry: {path}")
            continue
        _assert_ascii(path)
    if failures:
        print("ERROR: artifact scroll verification failed.")
        for item in failures:
            print(f"- {item}")
        return 1

    index = _load(index_path)
    source = _load(source_path)
    equation = _load(equation_path)
    proof = _load(proof_path)

    index_meta = index.get("artifact_scrolls", {})
    index_rows = index.get("scroll", [])
    source_docs = source.get("document", [])
    equation_atoms = equation.get("atom", [])
    proof_atoms = proof.get("atom", [])

    eq_ids = {str(row.get("id", "")).strip() for row in equation_atoms}
    proof_ids = {str(row.get("id", "")).strip() for row in proof_atoms}

    eq_counts_by_source: dict[str, int] = defaultdict(int)
    for row in equation_atoms:
        source_markdown = str(row.get("source_path", "")).strip()
        if source_markdown:
            eq_counts_by_source[source_markdown] += 1

    proof_counts_by_source: dict[str, int] = defaultdict(int)
    for row in proof_atoms:
        source_markdown = str(row.get("source_path", "")).strip()
        if source_markdown:
            proof_counts_by_source[source_markdown] += 1

    if int(index_meta.get("scroll_count", -1)) != len(index_rows):
        failures.append("artifact_scrolls.scroll_count metadata mismatch")

    index_by_source: dict[str, dict[str, object]] = {}
    for row in index_rows:
        source_markdown = str(row.get("source_markdown", "")).strip()
        if not source_markdown:
            failures.append("index row missing source_markdown")
            continue
        if source_markdown in index_by_source:
            failures.append(f"duplicate source_markdown in index: {source_markdown}")
        index_by_source[source_markdown] = row

    for doc in source_docs:
        source_markdown = str(doc.get("source_markdown", "")).strip()
        if source_markdown and source_markdown not in index_by_source:
            failures.append(f"missing scroll index for source document: {source_markdown}")

    for row in index_rows:
        source_markdown = str(row.get("source_markdown", "")).strip()
        scroll_rel = str(row.get("scroll_path", "")).strip()
        scroll_file = root / scroll_rel
        if not scroll_rel:
            failures.append(f"{source_markdown}: empty scroll_path in index")
            continue
        if not scroll_file.exists():
            failures.append(f"{source_markdown}: missing scroll file {scroll_rel}")
            continue
        _assert_ascii(scroll_file)

        scroll = _load(scroll_file)
        scroll_meta = scroll.get("scroll", {})
        section_rows = scroll.get("section", [])
        claim_rows = scroll.get("claim_ref", [])
        equation_rows = scroll.get("equation_ref", [])
        proof_rows = scroll.get("proof_ref", [])
        source_rows = scroll.get("source_ref", [])

        if str(scroll_meta.get("source_markdown", "")).strip() != source_markdown:
            failures.append(f"{source_markdown}: scroll metadata source_markdown mismatch")
        if str(scroll_meta.get("canonical_registry", "")).strip() != "registry/artifact_scrolls.toml":
            failures.append(f"{source_markdown}: canonical_registry must be registry/artifact_scrolls.toml")
        if bool(scroll_meta.get("authoritative", False)) is not True:
            failures.append(f"{source_markdown}: authoritative flag must be true")

        if int(scroll_meta.get("section_count", -1)) != len(section_rows):
            failures.append(f"{source_markdown}: section_count mismatch")
        if int(scroll_meta.get("claim_ref_count", -1)) != len(claim_rows):
            failures.append(f"{source_markdown}: claim_ref_count mismatch")
        if int(scroll_meta.get("equation_ref_count", -1)) != len(equation_rows):
            failures.append(f"{source_markdown}: equation_ref_count mismatch")
        if int(scroll_meta.get("proof_ref_count", -1)) != len(proof_rows):
            failures.append(f"{source_markdown}: proof_ref_count mismatch")
        if int(scroll_meta.get("source_ref_count", -1)) != len(source_rows):
            failures.append(f"{source_markdown}: source_ref_count mismatch")

        if len(section_rows) == 0:
            failures.append(f"{source_markdown}: scroll has zero sections")

        for section in section_rows:
            if not str(section.get("body_text", "")).strip():
                failures.append(f"{source_markdown}: empty section body_text in {section.get('id')}")
                break

        if int(row.get("section_count", -1)) != len(section_rows):
            failures.append(f"{source_markdown}: index section_count mismatch")
        if int(row.get("claim_ref_count", -1)) != len(claim_rows):
            failures.append(f"{source_markdown}: index claim_ref_count mismatch")
        if int(row.get("equation_ref_count", -1)) != len(equation_rows):
            failures.append(f"{source_markdown}: index equation_ref_count mismatch")
        if int(row.get("proof_ref_count", -1)) != len(proof_rows):
            failures.append(f"{source_markdown}: index proof_ref_count mismatch")
        if int(row.get("source_ref_count", -1)) != len(source_rows):
            failures.append(f"{source_markdown}: index source_ref_count mismatch")

        if int(row.get("equation_ref_count", 0)) != eq_counts_by_source.get(source_markdown, 0):
            failures.append(
                f"{source_markdown}: equation_ref_count {row.get('equation_ref_count')} != "
                f"knowledge atom count {eq_counts_by_source.get(source_markdown, 0)}"
            )
        if int(row.get("proof_ref_count", 0)) != proof_counts_by_source.get(source_markdown, 0):
            failures.append(
                f"{source_markdown}: proof_ref_count {row.get('proof_ref_count')} != "
                f"knowledge atom count {proof_counts_by_source.get(source_markdown, 0)}"
            )

        for eq in equation_rows:
            eq_id = str(eq.get("id", "")).strip()
            if eq_id and eq_id not in eq_ids:
                failures.append(f"{source_markdown}: unknown equation_ref id {eq_id}")
                break
        for prf in proof_rows:
            prf_id = str(prf.get("id", "")).strip()
            if prf_id and prf_id not in proof_ids:
                failures.append(f"{source_markdown}: unknown proof_ref id {prf_id}")
                break

    total_sections = sum(int(row.get("section_count", 0)) for row in index_rows)
    total_claims = sum(int(row.get("claim_ref_count", 0)) for row in index_rows)
    total_equations = sum(int(row.get("equation_ref_count", 0)) for row in index_rows)
    total_proofs = sum(int(row.get("proof_ref_count", 0)) for row in index_rows)
    total_sources = sum(int(row.get("source_ref_count", 0)) for row in index_rows)

    if int(index_meta.get("total_section_count", -1)) != total_sections:
        failures.append("index total_section_count mismatch")
    if int(index_meta.get("total_claim_ref_count", -1)) != total_claims:
        failures.append("index total_claim_ref_count mismatch")
    if int(index_meta.get("total_equation_ref_count", -1)) != total_equations:
        failures.append("index total_equation_ref_count mismatch")
    if int(index_meta.get("total_proof_ref_count", -1)) != total_proofs:
        failures.append("index total_proof_ref_count mismatch")
    if int(index_meta.get("total_source_ref_count", -1)) != total_sources:
        failures.append("index total_source_ref_count mismatch")

    if failures:
        print("ERROR: artifact scroll verification failed.")
        for item in failures[:200]:
            print(f"- {item}")
        if len(failures) > 200:
            print(f"- ... and {len(failures) - 200} more failures")
        return 1

    print(
        "OK: artifact scroll registry verified. "
        f"scrolls={len(index_rows)} sections={total_sections} "
        f"claims={total_claims} equations={total_equations} proofs={total_proofs} "
        f"source_refs={total_sources}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
