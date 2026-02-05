from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Record local hashes for data/external artifacts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/external"),
        help="Root directory to hash (default: data/external).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/PROVENANCE.local.json"),
        help="Output JSON path (default: data/external/PROVENANCE.local.json).",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        raise FileNotFoundError(f"External data root not found: {root}")

    entries: list[FileHash] = []
    for path in sorted([p for p in root.rglob("*") if p.is_file()]):
        rel = path.relative_to(root).as_posix()
        st = path.stat()
        entries.append(
            FileHash(
                path=rel,
                size_bytes=st.st_size,
                mtime_utc=iso_utc_from_timestamp(st.st_mtime),
                sha256=sha256_file(path),
            )
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": root.as_posix(),
        "hashes": [e.__dict__ for e in entries],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote: {args.output} ({len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

