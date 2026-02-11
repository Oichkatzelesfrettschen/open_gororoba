#!/usr/bin/env python3
"""
Verify embedded markdown structured payload/chunk registries.

Policy:
- Every targeted embedded markdown narrative has structured-TOML units.
- Metadata counts are consistent.
- Target markdown files do not exist on disk in strict TOML mode.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


TARGET_PREFIXES = ("docs/", "reports/", "data/artifacts/")
ROOT_TARGETS = {
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "README.md",
    "PANTHEON_PHYSICSFORGE_90_POINT_MIGRATION_PLAN.md",
    "PHASE10_11_ULTIMATE_ROADMAP.md",
    "PYTHON_REFACTORING_ROADMAP.md",
    "SYNTHESIS_PIPELINE_PROGRESS.md",
    "crates/vacuum_frustration/IMPLEMENTATION_NOTES.md",
    "curated/README.md",
    "curated/01_theory_frameworks/README_COQ.md",
    "data/csv/README.md",
    "data/artifacts/README.md",
    "NAVIGATOR.md",
    "REQUIREMENTS.md",
    "docs/REQUIREMENTS.md",
}


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: non-ASCII content in {path}: {sample!r}")


def _is_target_path(path: str) -> bool:
    if not path.endswith(".md"):
        return False
    if path in ROOT_TARGETS:
        return True
    return path.startswith(TARGET_PREFIXES)


def _pick_path(row: dict) -> str:
    candidates = [
        str(row.get("source_markdown", "")).strip(),
        str(row.get("path", "")).strip(),
        str(row.get("markdown", "")).strip(),
    ]
    for candidate in candidates:
        if candidate and candidate.endswith(".md"):
            return candidate
    return ""


def _collect_expected_paths(repo_root: Path) -> set[str]:
    expected: set[str] = set()
    for registry_path in sorted((repo_root / "registry").glob("*.toml")):
        raw = _load(registry_path)
        rows = raw.get("document", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            body = str(row.get("body_markdown", ""))
            if not body.strip():
                continue
            path = _pick_path(row)
            if path and _is_target_path(path):
                expected.add(path)
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--payload-path",
        default="registry/embedded_markdown_payloads.toml",
        help="Embedded markdown payload registry path.",
    )
    parser.add_argument(
        "--chunks-path",
        default="registry/embedded_markdown_chunks.toml",
        help="Embedded markdown chunks registry path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    payload_path = root / args.payload_path
    chunks_path = root / args.chunks_path

    if not payload_path.exists():
        raise SystemExit(f"ERROR: missing payload registry: {payload_path}")
    if not chunks_path.exists():
        raise SystemExit(f"ERROR: missing chunk registry: {chunks_path}")

    _assert_ascii(payload_path)
    _assert_ascii(chunks_path)

    payload_raw = _load(payload_path)
    chunks_raw = _load(chunks_path)

    payload_meta = payload_raw.get("embedded_markdown_payloads", {})
    chunk_meta = chunks_raw.get("embedded_markdown_chunks", {})
    docs = payload_raw.get("document", [])
    chunks = chunks_raw.get("chunk", [])

    failures: list[str] = []
    if int(payload_meta.get("document_count", -1)) != len(docs):
        failures.append("payload document_count metadata mismatch")
    if int(chunk_meta.get("chunk_count", -1)) != len(chunks):
        failures.append("chunks chunk_count metadata mismatch")
    if str(payload_meta.get("representation", "")) != "structured_toml_units":
        failures.append("payload representation must be structured_toml_units")
    if str(chunk_meta.get("representation", "")) != "structured_toml_units":
        failures.append("chunks representation must be structured_toml_units")

    chunk_by_id = {}
    chunk_ids_seen: set[str] = set()
    for row in chunks:
        chunk_id = str(row.get("id", ""))
        if chunk_id in chunk_ids_seen:
            failures.append(f"duplicate chunk id: {chunk_id}")
            continue
        chunk_ids_seen.add(chunk_id)
        chunk_by_id[chunk_id] = row

    payload_paths: set[str] = set()
    for row in docs:
        path = str(row.get("path", ""))
        payload_paths.add(path)
        if str(row.get("content_encoding", "")) != "structured_toml_units":
            failures.append(f"document not structured_toml_units: {path}")
        unit_ids = [str(value) for value in row.get("chunk_ids", [])]
        if len(unit_ids) != int(row.get("chunk_count", -1)):
            failures.append(f"chunk_count mismatch: {path}")
        for unit_id in unit_ids:
            if unit_id not in chunk_by_id:
                failures.append(f"missing chunk for document {path}: {unit_id}")
        if (root / path).exists():
            failures.append(f"strict-toml target markdown still exists on disk: {path}")

    expected_paths = _collect_expected_paths(root)
    missing = sorted(expected_paths - payload_paths)
    extra = sorted(payload_paths - expected_paths)
    if missing:
        failures.append(f"missing structured docs for embedded markdown paths: {len(missing)}")
        for item in missing[:25]:
            failures.append(f"  missing: {item}")
    if extra:
        failures.append(f"unexpected extra payload paths: {len(extra)}")
        for item in extra[:25]:
            failures.append(f"  extra: {item}")

    if failures:
        print("ERROR: embedded markdown structured registry verification failed:")
        for line in failures:
            print(f"- {line}")
        return 1

    print(
        "OK: embedded markdown structured registries verified. "
        f"documents={len(docs)} chunks={len(chunks)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
