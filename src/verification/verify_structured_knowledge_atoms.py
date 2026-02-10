#!/usr/bin/env python3
"""
Verify structured high-information knowledge atom registries.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII content in {path}: {sample!r}")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _unique_ids(rows: list[dict], key: str, failures: list[str], label: str) -> None:
    values = [str(row.get(key, "")) for row in rows]
    unique = set(values)
    if len(values) != len(unique):
        failures.append(f"{label}: duplicate {key} values detected.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--claim-registry",
        default="registry/knowledge/claim_atoms.toml",
        help="Claim atoms registry path.",
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
    parser.add_argument(
        "--summary-registry",
        default="registry/knowledge/structured_corpora.toml",
        help="Structured corpora summary path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    claim_path = root / args.claim_registry
    equation_path = root / args.equation_registry
    proof_path = root / args.proof_registry
    summary_path = root / args.summary_registry

    for path in (claim_path, equation_path, proof_path, summary_path):
        if not path.exists():
            raise SystemExit(f"ERROR: missing structured registry: {path}")
        _assert_ascii(path)

    claim = _load(claim_path)
    equation = _load(equation_path)
    proof = _load(proof_path)
    summary = _load(summary_path)

    claim_atoms = claim.get("atom", [])
    equation_atoms = equation.get("atom", [])
    proof_atoms = proof.get("atom", [])
    source_rows = summary.get("source", [])

    failures: list[str] = []

    if int(claim.get("knowledge_claim_atoms", {}).get("atom_count", -1)) != len(
        claim_atoms
    ):
        failures.append("claim_atoms atom_count metadata mismatch.")
    if int(
        equation.get("knowledge_equation_atoms", {}).get("atom_count", -1)
    ) != len(equation_atoms):
        failures.append("equation_atoms atom_count metadata mismatch.")
    if int(proof.get("knowledge_proof_atoms", {}).get("atom_count", -1)) != len(
        proof_atoms
    ):
        failures.append("proof_atoms atom_count metadata mismatch.")
    if int(summary.get("structured_corpora", {}).get("source_count", -1)) != len(
        source_rows
    ):
        failures.append("structured_corpora source_count metadata mismatch.")

    _unique_ids(claim_atoms, "id", failures, "claim_atoms")
    _unique_ids(equation_atoms, "id", failures, "equation_atoms")
    _unique_ids(proof_atoms, "id", failures, "proof_atoms")
    _unique_ids(source_rows, "id", failures, "structured_corpora")

    if len(claim_atoms) < 300:
        failures.append(f"claim_atoms too small: {len(claim_atoms)} < 300")
    if len(equation_atoms) < 150:
        failures.append(f"equation_atoms too small: {len(equation_atoms)} < 150")
    if len(proof_atoms) < 40:
        failures.append(f"proof_atoms too small: {len(proof_atoms)} < 40")

    claim_groups = {str(row.get("source_group", "")) for row in claim_atoms}
    if "doc_claim_matrix" not in claim_groups:
        failures.append("claim_atoms missing doc_claim_matrix coverage.")

    eq_groups = {str(row.get("source_group", "")) for row in equation_atoms}
    for required in ("doc_claim_matrix", "research_narrative", "data_artifact_narrative"):
        if required not in eq_groups:
            failures.append(f"equation_atoms missing source_group={required}")

    proof_groups = {str(row.get("source_group", "")) for row in proof_atoms}
    if "research_narrative" not in proof_groups:
        failures.append("proof_atoms missing research_narrative coverage.")

    for atom in claim_atoms[:]:
        claim_id = str(atom.get("claim_id", ""))
        if not claim_id.startswith("C-"):
            failures.append(f"invalid claim_id format: {claim_id}")
            break
        if not isinstance(atom.get("hypothesis_block_present", None), bool):
            failures.append(f"claim atom missing bool hypothesis flag: {atom.get('id')}")
            break

    for atom in equation_atoms[:]:
        if not str(atom.get("expression", "")).strip():
            failures.append("equation atom has empty expression.")
            break
        if not str(atom.get("relation_operator", "")).strip():
            failures.append(f"equation atom missing relation_operator: {atom.get('id')}")
            break
        if not str(atom.get("lhs_expression", "")).strip():
            failures.append(f"equation atom missing lhs_expression: {atom.get('id')}")
            break

    for atom in proof_atoms[:]:
        if int(atom.get("step_count", 0)) <= 0:
            failures.append(f"proof atom has invalid step_count: {atom.get('id')}")
            break
        for key in (
            "assumption_lines",
            "decision_lines",
            "conclusion_lines",
            "inference_markers",
        ):
            if not isinstance(atom.get(key, None), list):
                failures.append(f"proof atom missing list field {key}: {atom.get('id')}")
                break

    for row in source_rows:
        source_path = root / str(row.get("source_path", ""))
        if not source_path.exists():
            failures.append(f"structured source path missing: {source_path}")
        if row.get("narrative_compaction_recommended") is not True:
            failures.append(
                f"source row missing compaction recommendation: {row.get('source_uid')}"
            )
        if int(row.get("target_summary_max_lines", 0)) <= 0:
            failures.append(
                f"source row has invalid target_summary_max_lines: {row.get('source_uid')}"
            )

    if failures:
        print("ERROR: structured knowledge atom verification failed.")
        for item in failures[:200]:
            print(f"- {item}")
        if len(failures) > 200:
            print(f"- ... and {len(failures) - 200} more failures")
        return 1

    print(
        "OK: structured knowledge atoms verified. "
        f"claims={len(claim_atoms)} equations={len(equation_atoms)} "
        f"proofs={len(proof_atoms)} sources={len(source_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
