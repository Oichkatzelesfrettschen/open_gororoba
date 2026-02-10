#!/usr/bin/env python3
"""
Verify Wave 5 Batch 1 registries:
- claims_atoms
- claims_evidence_edges
- equation_atoms_v2
- equation_symbol_table
- proof_skeletons
- markdown_payloads
- markdown_payload_chunks
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import tomllib
from pathlib import Path


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: non-ASCII content in {path}: {sample!r}")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _discover_markdown_files(repo_root: Path) -> set[str]:
    out: set[str] = set()
    for path in repo_root.rglob("*.md"):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith(".git/"):
            continue
        out.add(rel)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--claims-path",
        default="registry/claims.toml",
        help="Canonical claims path.",
    )
    parser.add_argument(
        "--claims-atoms-path",
        default="registry/claims_atoms.toml",
        help="Claims atom path.",
    )
    parser.add_argument(
        "--claims-edges-path",
        default="registry/claims_evidence_edges.toml",
        help="Claims evidence edges path.",
    )
    parser.add_argument(
        "--equation-atoms-path",
        default="registry/knowledge/equation_atoms_v2.toml",
        help="Equation atoms v2 path.",
    )
    parser.add_argument(
        "--equation-symbols-path",
        default="registry/knowledge/equation_symbol_table.toml",
        help="Equation symbol table path.",
    )
    parser.add_argument(
        "--proof-skeletons-path",
        default="registry/knowledge/proof_skeletons.toml",
        help="Proof skeleton path.",
    )
    parser.add_argument(
        "--payload-path",
        default="registry/markdown_payloads.toml",
        help="Markdown payload metadata path.",
    )
    parser.add_argument(
        "--payload-chunks-path",
        default="registry/markdown_payload_chunks.toml",
        help="Markdown payload chunk path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    paths = [
        root / args.claims_path,
        root / args.claims_atoms_path,
        root / args.claims_edges_path,
        root / args.equation_atoms_path,
        root / args.equation_symbols_path,
        root / args.proof_skeletons_path,
        root / args.payload_path,
        root / args.payload_chunks_path,
    ]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"ERROR: missing registry {path}")

    for path in (
        root / args.claims_atoms_path,
        root / args.claims_edges_path,
        root / args.equation_atoms_path,
        root / args.equation_symbols_path,
        root / args.proof_skeletons_path,
        root / args.payload_path,
        root / args.payload_chunks_path,
    ):
        _assert_ascii(path)

    canonical_claims = _load(root / args.claims_path).get("claim", [])
    claim_atoms_raw = _load(root / args.claims_atoms_path)
    claim_edges_raw = _load(root / args.claims_edges_path)
    equation_atoms_raw = _load(root / args.equation_atoms_path)
    equation_symbols_raw = _load(root / args.equation_symbols_path)
    proof_raw = _load(root / args.proof_skeletons_path)
    payload_raw = _load(root / args.payload_path)
    chunk_raw = _load(root / args.payload_chunks_path)

    claim_atoms = claim_atoms_raw.get("atom", [])
    claim_edges = claim_edges_raw.get("edge", [])
    equation_atoms = equation_atoms_raw.get("atom", [])
    equation_symbols = equation_symbols_raw.get("symbol", [])
    proof_rows = proof_raw.get("skeleton", [])
    payload_docs = payload_raw.get("document", [])
    payload_chunks = chunk_raw.get("chunk", [])

    failures: list[str] = []

    # W5-007
    canonical_claim_ids = sorted(str(row.get("id", "")) for row in canonical_claims)
    atom_claim_ids = sorted(str(row.get("claim_id", "")) for row in claim_atoms)
    if canonical_claim_ids != atom_claim_ids:
        failures.append("claims_atoms claim_id set does not match canonical claims set.")
    if int(claim_atoms_raw.get("claims_atoms", {}).get("atom_count", -1)) != len(claim_atoms):
        failures.append("claims_atoms metadata atom_count mismatch.")
    status_tokens = {str(row.get("status_token", "")) for row in claim_atoms}
    if "" in status_tokens:
        failures.append("claims_atoms contains empty status_token.")

    # W5-008
    if int(claim_edges_raw.get("claims_evidence_edges", {}).get("edge_count", -1)) != len(
        claim_edges
    ):
        failures.append("claims_evidence_edges metadata edge_count mismatch.")
    edges_by_claim: dict[str, int] = {}
    for edge in claim_edges:
        claim_id = str(edge.get("claim_id", ""))
        edges_by_claim[claim_id] = edges_by_claim.get(claim_id, 0) + 1
    for claim_id in canonical_claim_ids:
        if edges_by_claim.get(claim_id, 0) <= 0:
            failures.append(f"claim has no evidence edges: {claim_id}")

    # W5-009
    if len(equation_atoms) < 150:
        failures.append(f"equation_atoms_v2 too small: {len(equation_atoms)}")
    if int(
        equation_atoms_raw.get("knowledge_equation_atoms_v2", {}).get("atom_count", -1)
    ) != len(equation_atoms):
        failures.append("equation_atoms_v2 metadata atom_count mismatch.")
    if int(equation_symbols_raw.get("equation_symbol_table", {}).get("symbol_count", -1)) != len(
        equation_symbols
    ):
        failures.append("equation_symbol_table metadata symbol_count mismatch.")

    symbol_ids = {str(row.get("id", "")) for row in equation_symbols}
    symbol_usage: dict[str, int] = {str(row.get("id", "")): 0 for row in equation_symbols}
    equation_ids: set[str] = set()
    for atom in equation_atoms:
        atom_id = str(atom.get("id", ""))
        if atom_id in equation_ids:
            failures.append(f"duplicate equation atom id: {atom_id}")
            break
        equation_ids.add(atom_id)
        refs = [str(item) for item in atom.get("symbol_refs", [])]
        if not refs:
            failures.append(f"equation atom has empty symbol_refs: {atom_id}")
            continue
        for ref in refs:
            if ref not in symbol_ids:
                failures.append(f"equation atom references unknown symbol id: {atom_id} -> {ref}")
            else:
                symbol_usage[ref] = symbol_usage.get(ref, 0) + 1
    for row in equation_symbols:
        sid = str(row.get("id", ""))
        expected = int(row.get("usage_count", 0))
        observed = int(symbol_usage.get(sid, 0))
        if expected != observed:
            failures.append(f"symbol usage mismatch: {sid} expected={expected} observed={observed}")

    # W5-010
    if len(proof_rows) < len(canonical_claims):
        failures.append(
            f"proof_skeletons too small: {len(proof_rows)} < canonical claims {len(canonical_claims)}"
        )
    if int(proof_raw.get("knowledge_proof_skeletons", {}).get("skeleton_count", -1)) != len(
        proof_rows
    ):
        failures.append("proof_skeletons metadata skeleton_count mismatch.")
    proof_ids: set[str] = set()
    for row in proof_rows:
        pid = str(row.get("id", ""))
        if pid in proof_ids:
            failures.append(f"duplicate proof skeleton id: {pid}")
            break
        proof_ids.add(pid)
        if not isinstance(row.get("assumptions", None), list):
            failures.append(f"proof skeleton assumptions not list: {pid}")
            break
        if not isinstance(row.get("derivation_steps", None), list):
            failures.append(f"proof skeleton derivation_steps not list: {pid}")
            break
        if not str(row.get("skeleton_kind", "")).strip():
            failures.append(f"proof skeleton missing skeleton_kind: {pid}")
            break

    # Full markdown payload migration
    if int(payload_raw.get("markdown_payloads", {}).get("document_count", -1)) != len(payload_docs):
        failures.append("markdown_payloads document_count metadata mismatch.")
    if int(chunk_raw.get("markdown_payload_chunks", {}).get("chunk_count", -1)) != len(
        payload_chunks
    ):
        failures.append("markdown_payload_chunks chunk_count metadata mismatch.")

    discovered_md = _discover_markdown_files(root)
    payload_paths = {str(row.get("path", "")) for row in payload_docs}
    if discovered_md != payload_paths:
        missing = sorted(discovered_md - payload_paths)
        extra = sorted(payload_paths - discovered_md)
        if missing:
            failures.append(f"markdown_payloads missing paths: {len(missing)}")
            for item in missing[:20]:
                failures.append(f"  missing: {item}")
        if extra:
            failures.append(f"markdown_payloads extra paths: {len(extra)}")
            for item in extra[:20]:
                failures.append(f"  extra: {item}")

    chunk_by_id = {str(row.get("id", "")): row for row in payload_chunks}
    if len(chunk_by_id) != len(payload_chunks):
        failures.append("duplicate markdown payload chunk ids detected.")

    third_party_count = 0
    for row in payload_docs:
        doc_id = str(row.get("id", ""))
        rel_path = str(row.get("path", ""))
        origin_class = str(row.get("origin_class", ""))
        if origin_class == "third_party_cache":
            third_party_count += 1
        chunk_ids = [str(item) for item in row.get("chunk_ids", [])]
        if int(row.get("chunk_count", -1)) != len(chunk_ids):
            failures.append(f"chunk_count mismatch for {doc_id}")
            continue
        parts: list[str] = []
        expected_next = 1
        for chunk_id in chunk_ids:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                failures.append(f"missing chunk id {chunk_id} for {doc_id}")
                break
            if str(chunk.get("document_id", "")) != doc_id:
                failures.append(f"chunk document_id mismatch: {chunk_id}")
            idx = int(chunk.get("chunk_index", 0))
            if idx != expected_next:
                failures.append(f"chunk index sequence mismatch for {doc_id}: got {idx} expected {expected_next}")
            expected_next += 1
            parts.append(str(chunk.get("payload_b64", "")))
        try:
            raw = base64.b64decode("".join(parts).encode("ascii"), validate=True)
        except Exception as exc:
            failures.append(f"base64 decode failed for {doc_id}: {exc}")
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest != str(row.get("content_sha256", "")):
            failures.append(f"sha mismatch for {doc_id} ({rel_path})")

    if third_party_count <= 0:
        failures.append("expected third_party_cache markdown documents, found none.")

    if failures:
        print("ERROR: Wave5 Batch1 registry verification failed.")
        for item in failures[:300]:
            print(f"- {item}")
        if len(failures) > 300:
            print(f"- ... and {len(failures) - 300} more failures")
        return 1

    print(
        "OK: Wave5 Batch1 registries verified. "
        f"claims={len(claim_atoms)} edges={len(claim_edges)} "
        f"equations={len(equation_atoms)} symbols={len(equation_symbols)} "
        f"proof_skeletons={len(proof_rows)} markdown_docs={len(payload_docs)} "
        f"markdown_chunks={len(payload_chunks)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
