#!/usr/bin/env python3
"""
Verify Wave 5 Batch 1 registries:
- claims_atoms
- claims_evidence_edges
- equation_atoms_v2
- equation_symbol_table
- proof_skeletons
- markdown_payloads (structured)
- markdown_payload_chunks (structured)
"""

from __future__ import annotations

import argparse
import hashlib
import tomllib
from pathlib import Path


ALLOWED_CHUNK_KINDS = {"heading", "paragraph", "list_item", "table_row", "code_block"}


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
        help="Structured markdown payload metadata path.",
    )
    parser.add_argument(
        "--payload-chunks-path",
        default="registry/markdown_payload_chunks.toml",
        help="Structured markdown payload chunk path.",
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
    uncovered_claims = [claim_id for claim_id in canonical_claim_ids if edges_by_claim.get(claim_id, 0) <= 0]
    if len(uncovered_claims) > 25:
        failures.append(
            "too many claims without evidence edges: "
            f"{len(uncovered_claims)} (max allowed 25)"
        )

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

    # Structured markdown payload migration
    payload_meta = payload_raw.get("markdown_payloads", {})
    chunk_meta = chunk_raw.get("markdown_payload_chunks", {})

    if int(payload_meta.get("document_count", -1)) != len(payload_docs):
        failures.append("markdown_payloads document_count metadata mismatch.")
    if int(chunk_meta.get("chunk_count", -1)) != len(payload_chunks):
        failures.append("markdown_payload_chunks chunk_count metadata mismatch.")
    if str(payload_meta.get("representation", "")) != "structured_toml_units":
        failures.append("markdown_payloads representation must be structured_toml_units.")
    if str(chunk_meta.get("representation", "")) != "structured_toml_units":
        failures.append("markdown_payload_chunks representation must be structured_toml_units.")

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

    chunks_by_doc: dict[str, list[dict]] = {}
    for row in payload_chunks:
        did = str(row.get("document_id", ""))
        chunks_by_doc.setdefault(did, []).append(row)

    third_party_count = 0
    for row in payload_docs:
        doc_id = str(row.get("id", ""))
        rel_path = str(row.get("path", ""))
        origin_class = str(row.get("origin_class", ""))
        if origin_class == "third_party_cache":
            third_party_count += 1

        file_path = root / rel_path
        if not file_path.exists():
            failures.append(f"payload doc path missing on disk: {doc_id} -> {rel_path}")
            continue

        raw = file_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != str(row.get("content_sha256", "")):
            failures.append(f"sha mismatch for {doc_id} ({rel_path})")

        chunk_ids = [str(item) for item in row.get("chunk_ids", [])]
        if int(row.get("chunk_count", -1)) != len(chunk_ids):
            failures.append(f"chunk_count mismatch for {doc_id}")
            continue

        heading_count = 0
        paragraph_count = 0
        list_item_count = 0
        table_row_count = 0
        code_block_count = 0

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
                failures.append(
                    f"chunk index sequence mismatch for {doc_id}: got {idx} expected {expected_next}"
                )
            expected_next += 1

            kind = str(chunk.get("kind", ""))
            if kind not in ALLOWED_CHUNK_KINDS:
                failures.append(f"invalid chunk kind for {chunk_id}: {kind}")
            if kind == "heading":
                heading_count += 1
            elif kind == "paragraph":
                paragraph_count += 1
            elif kind == "list_item":
                list_item_count += 1
            elif kind == "table_row":
                table_row_count += 1
            elif kind == "code_block":
                code_block_count += 1

            line_start = int(chunk.get("line_start", 0))
            line_end = int(chunk.get("line_end", 0))
            if line_start <= 0 or line_end < line_start:
                failures.append(f"invalid line range for {chunk_id}: {line_start}-{line_end}")

            text_ascii = str(chunk.get("text_ascii", ""))
            text_sha = hashlib.sha256(text_ascii.encode("utf-8")).hexdigest()
            if text_sha != str(chunk.get("text_sha256", "")):
                failures.append(f"text_sha256 mismatch for {chunk_id}")

        if heading_count != int(row.get("heading_count", -1)):
            failures.append(f"heading_count mismatch for {doc_id}")
        if paragraph_count != int(row.get("paragraph_count", -1)):
            failures.append(f"paragraph_count mismatch for {doc_id}")
        if list_item_count != int(row.get("list_item_count", -1)):
            failures.append(f"list_item_count mismatch for {doc_id}")
        if table_row_count != int(row.get("table_row_count", -1)):
            failures.append(f"table_row_count mismatch for {doc_id}")
        if code_block_count != int(row.get("code_block_count", -1)):
            failures.append(f"code_block_count mismatch for {doc_id}")

        if doc_id not in chunks_by_doc:
            failures.append(f"no chunks indexed for document {doc_id}")

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
