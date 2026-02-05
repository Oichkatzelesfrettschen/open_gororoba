from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class FileHash:
    path: str
    size_bytes: int
    mtime_utc: str
    sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_utc_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record local hashes for manifest-tracked artifacts under data/artifacts."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repo root (default: current directory).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/artifacts/ARTIFACTS_MANIFEST.csv"),
        help="Artifacts manifest CSV (default: data/artifacts/ARTIFACTS_MANIFEST.csv).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/artifacts"),
        help="Artifacts root directory (default: data/artifacts).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/artifacts/PROVENANCE.local.json"),
        help="Output JSON path (default: data/artifacts/PROVENANCE.local.json).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest_path = (repo_root / args.manifest).resolve()
    artifacts_root = (repo_root / args.root).resolve()
    out_path = (repo_root / args.output).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not artifacts_root.exists():
        raise FileNotFoundError(f"Artifacts root not found: {artifacts_root}")

    artifact_paths: list[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "artifact_path" not in (reader.fieldnames or []):
            raise ValueError("ARTIFACTS_MANIFEST.csv missing required column: artifact_path")
        for row in reader:
            p = (row.get("artifact_path") or "").strip()
            if p:
                artifact_paths.append(p)

    entries: list[FileHash] = []
    missing: list[str] = []
    for ap in sorted(set(artifact_paths)):
        abs_path = (repo_root / ap).resolve()
        if not abs_path.exists():
            missing.append(ap)
            continue
        if not abs_path.is_file():
            continue
        if artifacts_root not in abs_path.parents and abs_path != artifacts_root:
            raise ValueError(f"Manifest path is outside artifacts root: {ap}")
        rel = abs_path.relative_to(artifacts_root).as_posix()
        st = abs_path.stat()
        entries.append(
            FileHash(
                path=rel,
                size_bytes=st.st_size,
                mtime_utc=iso_utc_from_timestamp(st.st_mtime),
                sha256=sha256_file(abs_path),
            )
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": Path("data/artifacts").as_posix(),
        "hashes": [e.__dict__ for e in entries],
        "manifest": Path("data/artifacts/ARTIFACTS_MANIFEST.csv").as_posix(),
        "missing_manifest_paths": sorted(missing),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path} ({len(entries)} files; missing={len(missing)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

