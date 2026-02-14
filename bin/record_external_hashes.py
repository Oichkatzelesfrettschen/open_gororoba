from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:  # Optional pyo3 bridge.
    import gororoba_py  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    gororoba_py = None  # type: ignore


@dataclass(frozen=True)
class FileHash:
    path: str
    size_bytes: int
    mtime_utc: str
    sha256: str


def sha256_file(path: Path, backend: str) -> str:
    if backend not in {"auto", "pyo3", "python"}:
        raise ValueError("sha backend must be one of: auto, pyo3, python")

    if backend in {"auto", "pyo3"}:
        if gororoba_py is not None and hasattr(gororoba_py, "py_sha256_file"):
            try:
                out = str(gororoba_py.py_sha256_file(str(path)))
                if out and len(out) == 64:
                    return out
            except Exception as exc:
                if backend == "pyo3":
                    raise RuntimeError(
                        f"pyo3 sha256 backend failed for {path}: {exc}"
                    ) from exc
        elif backend == "pyo3":
            raise RuntimeError("pyo3 backend selected but gororoba_py is unavailable")

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
    parser.add_argument(
        "--sha-backend",
        default="auto",
        help="Checksum backend: auto, pyo3, or python.",
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
                sha256=sha256_file(path, args.sha_backend),
            )
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": root.as_posix(),
        "hash_backend": args.sha_backend,
        "hashes": [e.__dict__ for e in entries],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote: {args.output} ({len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
