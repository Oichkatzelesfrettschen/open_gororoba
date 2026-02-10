#!/usr/bin/env python3
"""
Verify schema signatures in registry/schema_signatures.toml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: non-ASCII content in {path}: {sample!r}")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _shape_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "table", "keys": sorted(value.keys())}
    if isinstance(value, list):
        if not value:
            return {"type": "array", "row_count": 0, "entry_kind": "empty"}
        if all(isinstance(item, dict) for item in value):
            key_sets = [set(item.keys()) for item in value]
            return {
                "type": "array",
                "row_count": len(value),
                "entry_kind": "table",
                "required_keys": sorted(set.intersection(*key_sets)) if key_sets else [],
                "union_keys": sorted(set.union(*key_sets)) if key_sets else [],
            }
        return {
            "type": "array",
            "row_count": len(value),
            "entry_kind": "scalar_or_mixed",
            "entry_types": sorted({type(item).__name__ for item in value}),
        }
    return {"type": type(value).__name__}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--schema-path",
        default="registry/schema_signatures.toml",
        help="Schema signature registry path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    schema_path = root / args.schema_path
    if not schema_path.exists():
        raise SystemExit(f"ERROR: missing schema signature registry: {schema_path}")
    _assert_ascii(schema_path)

    raw = _load(schema_path)
    meta = raw.get("schema_signatures", {})
    rows = raw.get("signature", [])
    failures: list[str] = []

    if int(meta.get("signature_count", -1)) != len(rows):
        failures.append("schema_signatures signature_count metadata mismatch")
    if int(meta.get("version", 0)) <= 0:
        failures.append("schema_signatures version must be positive")

    seen_paths: set[str] = set()
    for row in rows:
        rel = str(row.get("path", ""))
        if rel in seen_paths:
            failures.append(f"duplicate schema signature path: {rel}")
            continue
        seen_paths.add(rel)
        path = root / rel
        if not path.exists():
            failures.append(f"signed registry path missing: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        data = tomllib.loads(text)
        top_level = sorted(data.keys())
        shapes = {key: _shape_summary(data[key]) for key in top_level}
        payload = {"path": rel, "top_level_keys": top_level, "shapes": shapes}
        normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        schema_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if schema_sha != str(row.get("schema_sha256", "")):
            failures.append(f"schema_sha mismatch for {rel}")
        if content_sha != str(row.get("content_sha256", "")):
            failures.append(f"content_sha mismatch for {rel}")
        if normalized != str(row.get("shape_json", "")):
            failures.append(f"shape_json mismatch for {rel}")
        if top_level != [str(v) for v in row.get("top_level_keys", [])]:
            failures.append(f"top_level_keys mismatch for {rel}")

    declared_paths = sorted(str(v) for v in meta.get("registry_paths", []))
    if sorted(seen_paths) != declared_paths:
        missing = sorted(set(declared_paths) - seen_paths)
        extra = sorted(set(seen_paths) - set(declared_paths))
        if missing:
            failures.append(f"schema metadata registry_paths missing signatures: {len(missing)}")
        if extra:
            failures.append(f"schema signatures contain undeclared paths: {len(extra)}")

    if failures:
        print("ERROR: registry schema signature verification failed.")
        for item in failures[:250]:
            print(f"- {item}")
        if len(failures) > 250:
            print(f"- ... and {len(failures) - 250} more failures")
        return 1

    print(f"OK: schema signatures verified for {len(rows)} registries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
