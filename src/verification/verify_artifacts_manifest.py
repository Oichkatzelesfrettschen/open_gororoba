#!/usr/bin/env python3
"""
Verify data/artifacts manifest coverage and provenance sidecar hygiene.

Goals:
- Establish a stable convention for artifact paths:
  data/artifacts/<domain>/<name>.<ext>
- Require any artifact referenced by canonical "claims" docs (and core artifact
  verifiers) to be listed in data/artifacts/ARTIFACTS_MANIFEST.csv.
- Optionally require a provenance sidecar for selected artifacts.

Sidecar convention:
- For an artifact at path P (including extension), provenance sidecar is
  P + ".PROVENANCE.json" (example: foo.png.PROVENANCE.json)

This verifier is offline and deterministic.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path("data/artifacts/ARTIFACTS_MANIFEST.csv")
SCAN_FILES = [
    Path("README.md"),
    Path("docs/CLAIMS_EVIDENCE_MATRIX.md"),
    Path("docs/CLAIMS_TASKS.md"),
    Path("src/verification/verify_generated_artifacts.py"),
]

ARTIFACT_RX = re.compile(r"data/artifacts/[A-Za-z0-9_./-]+")


@dataclass(frozen=True)
class ManifestRow:
    artifact_path: str
    domain: str
    kind: str
    provenance_required: bool
    generator: str


def _error(msg: str) -> int:
    print(f"ERROR: {msg}")
    return 2


def _load_manifest(repo_root: Path) -> dict[str, ManifestRow]:
    manifest_path = repo_root / MANIFEST_PATH
    if not manifest_path.exists():
        raise SystemExit(f"ERROR: Missing {MANIFEST_PATH}")

    rows: dict[str, ManifestRow] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = ["artifact_path", "domain", "kind", "provenance_required", "generator"]
        if reader.fieldnames != expected:
            raise SystemExit(
                f"ERROR: {MANIFEST_PATH}: expected header {expected}, got {reader.fieldnames}"
            )
        for i, r in enumerate(reader, start=2):
            artifact_path = (r.get("artifact_path") or "").strip()
            domain = (r.get("domain") or "").strip()
            kind = (r.get("kind") or "").strip()
            prov_raw = (r.get("provenance_required") or "").strip().lower()
            generator = (r.get("generator") or "").strip()

            if not artifact_path:
                raise SystemExit(f"ERROR: {MANIFEST_PATH}:{i}: missing artifact_path")
            if artifact_path.startswith("/"):
                raise SystemExit(
                    f"ERROR: {MANIFEST_PATH}:{i}: absolute path not allowed: {artifact_path}"
                )
            if not artifact_path.startswith("data/artifacts/"):
                raise SystemExit(
                    f"ERROR: {MANIFEST_PATH}:{i}: expected data/artifacts prefix: {artifact_path}"
                )
            if not domain:
                raise SystemExit(f"ERROR: {MANIFEST_PATH}:{i}: missing domain for {artifact_path}")
            if not kind:
                raise SystemExit(f"ERROR: {MANIFEST_PATH}:{i}: missing kind for {artifact_path}")

            if prov_raw in {"1", "true", "yes"}:
                provenance_required = True
            elif prov_raw in {"0", "false", "no"}:
                provenance_required = False
            else:
                raise SystemExit(
                    f"ERROR: {MANIFEST_PATH}:{i}: provenance_required must be 0/1, got {prov_raw!r}"
                )

            if artifact_path in rows:
                raise SystemExit(
                    f"ERROR: {MANIFEST_PATH}:{i}: duplicate artifact_path: {artifact_path}"
                )

            rows[artifact_path] = ManifestRow(
                artifact_path=artifact_path,
                domain=domain,
                kind=kind,
                provenance_required=provenance_required,
                generator=generator,
            )
    return rows


def _extract_referenced_artifacts(repo_root: Path) -> set[str]:
    referenced: set[str] = set()
    for rel in SCAN_FILES:
        p = repo_root / rel
        if not p.exists():
            raise SystemExit(f"ERROR: Missing scan file: {rel}")
        text = p.read_text(encoding="utf-8")
        for m in ARTIFACT_RX.finditer(text):
            s = m.group(0).rstrip(".,);]\"'")
            if s.endswith("/"):
                continue
            referenced.add(s)
    return referenced


def _validate_convention(row: ManifestRow) -> None:
    parts = row.artifact_path.split("/")
    # data/artifacts/<domain>/<name>.<ext> (allow optional subdirs under domain)
    if len(parts) < 4:
        raise SystemExit(f"ERROR: manifest path too short: {row.artifact_path}")
    if parts[0:2] != ["data", "artifacts"]:
        raise SystemExit(f"ERROR: manifest path missing data/artifacts prefix: {row.artifact_path}")
    if parts[2] != row.domain:
        raise SystemExit(
            f"ERROR: manifest domain mismatch for {row.artifact_path}: domain={row.domain!r}"
        )
    filename = parts[-1]
    if "." not in filename:
        raise SystemExit(f"ERROR: artifact filename missing extension: {row.artifact_path}")
    ext = filename.rsplit(".", 1)[1].lower()
    if ext != row.kind.lower():
        raise SystemExit(
            f"ERROR: manifest kind mismatch for {row.artifact_path}: kind={row.kind!r} ext={ext!r}"
        )


def _validate_provenance_json(path: Path, *, expected_artifact_path: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: invalid JSON: {path}: {e}") from e
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: provenance JSON must be an object: {path}")
    _require_str(payload, "artifact_path", path)
    if payload["artifact_path"] != expected_artifact_path:
        raise SystemExit(
            "ERROR: "
            f"{path}: artifact_path mismatch: {payload['artifact_path']!r} "
            f"!= {expected_artifact_path!r}"
        )
    _require_str(payload, "generator", path)
    _require_int(payload, "provenance_version", path)
    _require_list(payload, "inputs", path)
    _require_dict(payload, "parameters", path)


def _require_str(obj: dict[str, Any], key: str, path: Path) -> None:
    v = obj.get(key)
    if not isinstance(v, str) or not v:
        raise SystemExit(f"ERROR: {path}: missing/invalid {key} (expected non-empty string)")


def _require_int(obj: dict[str, Any], key: str, path: Path) -> None:
    v = obj.get(key)
    if not isinstance(v, int):
        raise SystemExit(f"ERROR: {path}: missing/invalid {key} (expected int)")


def _require_list(obj: dict[str, Any], key: str, path: Path) -> None:
    v = obj.get(key)
    if not isinstance(v, list):
        raise SystemExit(f"ERROR: {path}: missing/invalid {key} (expected list)")


def _require_dict(obj: dict[str, Any], key: str, path: Path) -> None:
    v = obj.get(key)
    if not isinstance(v, dict):
        raise SystemExit(f"ERROR: {path}: missing/invalid {key} (expected object)")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = _load_manifest(repo_root)
    referenced = _extract_referenced_artifacts(repo_root)

    # Coverage: canonical references must be present in the manifest.
    missing = sorted(referenced - set(manifest))
    if missing:
        for p in missing[:100]:
            print(f"ERROR: referenced artifact missing from manifest: {p}")
        if len(missing) > 100:
            print(f"ERROR: ... plus {len(missing) - 100} more")
        return 2

    # Basic manifest hygiene + optional provenance.
    for row in manifest.values():
        _validate_convention(row)
        artifact_path = repo_root / row.artifact_path
        if not artifact_path.exists():
            return _error(f"manifest artifact missing on disk: {row.artifact_path}")
        if row.provenance_required:
            sidecar = repo_root / (row.artifact_path + ".PROVENANCE.json")
            if not sidecar.exists():
                return _error(f"missing provenance sidecar: {sidecar.relative_to(repo_root)}")
            _validate_provenance_json(sidecar, expected_artifact_path=row.artifact_path)

    print("OK: artifacts manifest coverage + provenance hygiene")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
