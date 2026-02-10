#!/usr/bin/env python3
"""
Verify Wave 5 Batch 2 registries:
- registry/knowledge/derivation_steps.toml
- registry/bibliography_normalized.toml
- registry/provenance_sources.toml
- registry/narrative_paragraph_atoms.toml
"""

from __future__ import annotations

import argparse
import re
import tomllib
from collections import defaultdict
from pathlib import Path


CLAIM_ID_RE = re.compile(r"\bC-\d{3}\b")
SHA_RE = re.compile(r"^[a-f0-9]{64}$")


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: non-ASCII content in {path}: {sample!r}")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _claim_set(claim_rows: list[dict]) -> set[str]:
    return {str(row.get("id", "")) for row in claim_rows if str(row.get("id", "")).startswith("C-")}


def _nonempty_body_doc_ids(rows: list[dict]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for row in rows:
        doc_id = str(row.get("id", ""))
        body = str(row.get("body_markdown", ""))
        if doc_id and body.strip():
            out.add((doc_id, str(row.get("source_markdown", ""))))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--derivation-path",
        default="registry/knowledge/derivation_steps.toml",
        help="Derivation step registry path.",
    )
    parser.add_argument(
        "--bibliography-normalized-path",
        default="registry/bibliography_normalized.toml",
        help="Bibliography normalized registry path.",
    )
    parser.add_argument(
        "--provenance-path",
        default="registry/provenance_sources.toml",
        help="Provenance source registry path.",
    )
    parser.add_argument(
        "--paragraph-path",
        default="registry/narrative_paragraph_atoms.toml",
        help="Narrative paragraph atom registry path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    paths = [
        root / "registry/claims.toml",
        root / "registry/knowledge/proof_skeletons.toml",
        root / "registry/bibliography.toml",
        root / "registry/external_sources.toml",
        root / "registry/docs_root_narratives.toml",
        root / "registry/research_narratives.toml",
        root / "registry/data_artifact_narratives.toml",
        root / args.derivation_path,
        root / args.bibliography_normalized_path,
        root / args.provenance_path,
        root / args.paragraph_path,
    ]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"ERROR: missing required registry: {path}")

    for path in (
        root / args.derivation_path,
        root / args.bibliography_normalized_path,
        root / args.provenance_path,
        root / args.paragraph_path,
    ):
        _assert_ascii(path)

    claims_raw = _load(root / "registry/claims.toml")
    proof_raw = _load(root / "registry/knowledge/proof_skeletons.toml")
    bib_raw = _load(root / "registry/bibliography.toml")
    external_raw = _load(root / "registry/external_sources.toml")
    docs_root_raw = _load(root / "registry/docs_root_narratives.toml")
    research_raw = _load(root / "registry/research_narratives.toml")
    artifact_raw = _load(root / "registry/data_artifact_narratives.toml")

    derivation_raw = _load(root / args.derivation_path)
    bib_norm_raw = _load(root / args.bibliography_normalized_path)
    provenance_raw = _load(root / args.provenance_path)
    paragraph_raw = _load(root / args.paragraph_path)

    failures: list[str] = []

    claim_ids = _claim_set(claims_raw.get("claim", []))
    proof_rows = proof_raw.get("skeleton", [])
    proof_ids = {str(row.get("id", "")) for row in proof_rows}

    # Derivation steps
    derivation_meta = derivation_raw.get("knowledge_derivation_steps", {})
    derivation_rows = derivation_raw.get("step", [])
    if int(derivation_meta.get("step_count", -1)) != len(derivation_rows):
        failures.append("derivation step_count metadata mismatch")
    if int(derivation_meta.get("skeleton_count", -1)) != len(proof_rows):
        failures.append("derivation skeleton_count metadata mismatch")
    seen_step_ids: set[str] = set()
    by_skeleton: defaultdict[str, list[dict]] = defaultdict(list)
    for row in derivation_rows:
        step_id = str(row.get("id", ""))
        if step_id in seen_step_ids:
            failures.append(f"duplicate derivation step id: {step_id}")
            break
        seen_step_ids.add(step_id)
        skeleton_id = str(row.get("skeleton_id", ""))
        if skeleton_id not in proof_ids:
            failures.append(f"derivation step references unknown skeleton_id: {skeleton_id}")
        claim_id = str(row.get("claim_id", ""))
        if claim_id and claim_id not in claim_ids:
            failures.append(f"derivation step references unknown claim_id: {step_id} -> {claim_id}")
        for cref in row.get("claim_refs", []):
            if str(cref) not in claim_ids:
                failures.append(f"derivation step contains unknown claim ref: {step_id} -> {cref}")
        by_skeleton[skeleton_id].append(row)

    for skeleton_id, rows in by_skeleton.items():
        rows_sorted = sorted(rows, key=lambda item: int(item.get("step_index", 0)))
        expected = 1
        prev_id = ""
        for row in rows_sorted:
            idx = int(row.get("step_index", 0))
            if idx != expected:
                failures.append(f"derivation step_index gap in {skeleton_id}: got {idx} expected {expected}")
                break
            deps = [str(v) for v in row.get("depends_on_step_ids", [])]
            if expected == 1 and deps:
                failures.append(f"first derivation step should have no depends_on: {row.get('id')}")
            if expected > 1 and deps != [prev_id]:
                failures.append(f"derivation dependency mismatch: {row.get('id')} expected [{prev_id}] got {deps}")
            prev_id = str(row.get("id", ""))
            expected += 1

    # Bibliography normalized
    bib_entries = bib_raw.get("entry", [])
    bib_norm_meta = bib_norm_raw.get("bibliography_normalized", {})
    bib_norm_entries = bib_norm_raw.get("entry", [])
    if int(bib_norm_meta.get("entry_count", -1)) != len(bib_norm_entries):
        failures.append("bibliography_normalized entry_count metadata mismatch")
    if len(bib_norm_entries) != len(bib_entries):
        failures.append(
            f"bibliography_normalized count mismatch: {len(bib_norm_entries)} vs source {len(bib_entries)}"
        )
    source_bib_ids = {str(row.get("id", "")) for row in bib_entries}
    seen_norm_source: set[str] = set()
    for row in bib_norm_entries:
        source_id = str(row.get("source_entry_id", ""))
        if source_id not in source_bib_ids:
            failures.append(f"bibliography_normalized unknown source_entry_id: {source_id}")
        if source_id in seen_norm_source:
            failures.append(f"bibliography_normalized duplicate source_entry_id: {source_id}")
        seen_norm_source.add(source_id)
        year = int(row.get("publication_year", 0))
        if year != 0 and (year < 1500 or year > 2100):
            failures.append(f"bibliography_normalized invalid publication_year: {source_id} -> {year}")
        if str(row.get("document_type", "")) == "arxiv_preprint" and not str(row.get("arxiv_id", "")):
            failures.append(f"bibliography_normalized arxiv_preprint missing arxiv_id: {source_id}")
        for cref in row.get("claim_refs", []):
            if str(cref) not in claim_ids:
                failures.append(f"bibliography_normalized unknown claim_ref: {source_id} -> {cref}")

    # Provenance
    external_docs = external_raw.get("document", [])
    external_doc_ids = {str(row.get("id", "")) for row in external_docs}
    provenance_meta = provenance_raw.get("provenance_sources", {})
    provenance_rows = provenance_raw.get("record", [])
    if int(provenance_meta.get("record_count", -1)) != len(provenance_rows):
        failures.append("provenance_sources record_count metadata mismatch")
    if int(provenance_meta.get("document_count", -1)) != len(external_doc_ids):
        failures.append("provenance_sources document_count metadata mismatch")
    kind_counts: defaultdict[str, int] = defaultdict(int)
    by_doc: defaultdict[str, int] = defaultdict(int)
    for row in provenance_rows:
        kind = str(row.get("source_kind", ""))
        kind_counts[kind] += 1
        doc_id = str(row.get("document_id", ""))
        by_doc[doc_id] += 1
        if doc_id not in external_doc_ids:
            failures.append(f"provenance_sources unknown document_id: {doc_id}")
        if kind not in {"url", "path", "sha256"}:
            failures.append(f"provenance_sources invalid source_kind: {kind}")
        if kind == "sha256":
            digest = str(row.get("sha256", ""))
            if not SHA_RE.fullmatch(digest):
                failures.append(f"provenance_sources invalid sha256 digest: {row.get('id')}")
            if str(row.get("source_ref", "")) != digest:
                failures.append(f"provenance_sources sha256 source_ref mismatch: {row.get('id')}")
        for cref in row.get("claim_refs", []):
            if str(cref) not in claim_ids:
                failures.append(f"provenance_sources unknown claim_ref: {row.get('id')} -> {cref}")
    if int(provenance_meta.get("url_record_count", -1)) != kind_counts["url"]:
        failures.append("provenance_sources url_record_count metadata mismatch")
    if int(provenance_meta.get("path_record_count", -1)) != kind_counts["path"]:
        failures.append("provenance_sources path_record_count metadata mismatch")
    if int(provenance_meta.get("hash_record_count", -1)) != kind_counts["sha256"]:
        failures.append("provenance_sources hash_record_count metadata mismatch")
    for doc_id in external_doc_ids:
        if by_doc.get(doc_id, 0) == 0:
            failures.append(f"provenance_sources missing records for external source doc: {doc_id}")

    # Narrative paragraph atoms
    paragraph_meta = paragraph_raw.get("narrative_paragraph_atoms", {})
    paragraph_rows = paragraph_raw.get("paragraph", [])
    if int(paragraph_meta.get("paragraph_count", -1)) != len(paragraph_rows):
        failures.append("narrative_paragraph_atoms paragraph_count metadata mismatch")
    seen_paragraph_ids: set[str] = set()
    allowed_source_registries = {
        "registry/docs_root_narratives.toml",
        "registry/research_narratives.toml",
        "registry/external_sources.toml",
        "registry/data_artifact_narratives.toml",
    }
    paragraph_by_doc: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    paragraph_doc_coverage: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in paragraph_rows:
        pid = str(row.get("id", ""))
        if pid in seen_paragraph_ids:
            failures.append(f"duplicate narrative paragraph id: {pid}")
            break
        seen_paragraph_ids.add(pid)
        source_registry = str(row.get("source_registry", ""))
        if source_registry not in allowed_source_registries:
            failures.append(f"narrative paragraph invalid source_registry: {source_registry}")
        doc_id = str(row.get("document_id", ""))
        source_markdown = str(row.get("source_markdown", ""))
        paragraph_by_doc[(source_registry, doc_id)].append(row)
        paragraph_doc_coverage[source_registry].add((doc_id, source_markdown))
        line_start = int(row.get("line_start", 0))
        line_end = int(row.get("line_end", 0))
        if line_start <= 0 or line_end <= 0 or line_end < line_start:
            failures.append(f"narrative paragraph invalid line span: {pid}")
        for cref in row.get("claim_refs", []):
            if str(cref) not in claim_ids:
                failures.append(f"narrative paragraph unknown claim_ref: {pid} -> {cref}")

    for key, rows in paragraph_by_doc.items():
        expected = 1
        for row in sorted(rows, key=lambda item: int(item.get("paragraph_index", 0))):
            idx = int(row.get("paragraph_index", 0))
            if idx != expected:
                failures.append(
                    f"narrative paragraph_index gap for {key[0]}::{key[1]} got {idx} expected {expected}"
                )
                break
            expected += 1

    docs_root_expected = _nonempty_body_doc_ids(docs_root_raw.get("document", []))
    research_expected = _nonempty_body_doc_ids(research_raw.get("document", []))
    external_expected = _nonempty_body_doc_ids(external_raw.get("document", []))
    artifact_expected = _nonempty_body_doc_ids(artifact_raw.get("document", []))
    expected_map = {
        "registry/docs_root_narratives.toml": docs_root_expected,
        "registry/research_narratives.toml": research_expected,
        "registry/external_sources.toml": external_expected,
        "registry/data_artifact_narratives.toml": artifact_expected,
    }
    for source_registry, expected_docs in expected_map.items():
        observed_docs = paragraph_doc_coverage.get(source_registry, set())
        if expected_docs != observed_docs:
            missing = sorted(expected_docs - observed_docs)
            extra = sorted(observed_docs - expected_docs)
            if missing:
                failures.append(f"narrative paragraph coverage missing docs in {source_registry}: {len(missing)}")
            if extra:
                failures.append(f"narrative paragraph coverage extra docs in {source_registry}: {len(extra)}")

    if int(paragraph_meta.get("document_count", -1)) != sum(len(v) for v in expected_map.values()):
        failures.append("narrative_paragraph_atoms document_count metadata mismatch")

    if failures:
        print("ERROR: Wave5 Batch2 registry verification failed.")
        for item in failures[:300]:
            print(f"- {item}")
        if len(failures) > 300:
            print(f"- ... and {len(failures) - 300} more failures")
        return 1

    print(
        "OK: Wave5 Batch2 registries verified. "
        f"derivations={len(derivation_rows)} "
        f"bibliography_normalized={len(bib_norm_entries)} "
        f"provenance_records={len(provenance_rows)} "
        f"narrative_paragraphs={len(paragraph_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
