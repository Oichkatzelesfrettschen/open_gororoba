#!/usr/bin/env python3
"""Fetch governed hourly OMNI solar-wind + IMF data.

Primary source:
    NASA SPDF low-resolution OMNI yearly ASCII files.

Operational fallback on this host:
    AMDA/CDPP HAPI dataset `omni-hour-all`, staged as yearly CSV slices.

The repo's Rust OMNI parser now accepts both the canonical SPDF fixed-width
ASCII and the governed AMDA HAPI CSV fallback, so chronology and solver code
do not depend on a single network origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

SPDF_BASE = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_"
AMDA_HAPI = "https://amda.irap.omp.eu/service/hapi"
AMDA_DATASET_ID = "omni-hour-all"
OUT_DIR = Path("data/external/omni2")
USER_AGENT = "gororoba-fetch/0.1 (research)"
SPDF_SOURCE_CONTRACT = "SRC-OMNI2-HOURLY-SPDF-CANONICAL"
AMDA_SOURCE_CONTRACT = "SRC-OMNI-HOURLY-AMDA-FALLBACK"
MANIFEST_SOURCE_CONTRACT = "SRC-OMNI-HOURLY-GOVERNED-MANIFESTS"


def build_file_entry(
    url: str, path: Path, status: str, *, source: str, reason: str | None = None
) -> dict[str, object]:
    source_contract = (
        SPDF_SOURCE_CONTRACT if source == "spdf" else AMDA_SOURCE_CONTRACT
    )
    entry: dict[str, object] = {
        "url": url,
        "path": str(path),
        "status": status,
        "source": source,
        "source_contract": source_contract,
    }
    if reason:
        entry["reason"] = reason
    if not path.exists():
        return entry

    data = path.read_bytes()
    entry["filename"] = path.name
    entry["bytes"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    entry["lines"] = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
    return entry


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def fetch_text(url: str, *, timeout: int = 60) -> str:
    return fetch_bytes(url, timeout=timeout).decode("utf-8", errors="replace")


def write_manifest(start: int, end: int, payload: dict[str, object]) -> None:
    manifest_path = OUT_DIR / f"omni_{start}_{end}_manifest.json"
    manifest = {
        "product": "omni_hourly",
        "start_year": start,
        "end_year": end,
        "years": list(range(start, end + 1)),
        "source_contract": MANIFEST_SOURCE_CONTRACT,
        **payload,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  manifest: {manifest_path}")


def fetch_spdf_year(year: int, *, skip_existing: bool) -> dict[str, object]:
    out = OUT_DIR / f"omni2_{year}.dat"
    url = f"{SPDF_BASE}{year}.dat"

    if skip_existing and out.exists():
        return build_file_entry(url, out, "skipped", source="spdf")

    try:
        data = fetch_bytes(url)
    except (URLError, OSError) as exc:
        return {
            "url": url,
            "path": str(out),
            "status": "failed",
            "source": "spdf",
            "reason": str(exc),
        }

    if len(data) < 1024:
        return {
            "url": url,
            "path": str(out),
            "status": "failed",
            "source": "spdf",
            "reason": f"unexpectedly_small:{len(data)}",
        }

    out.write_bytes(data)
    entry = build_file_entry(url, out, "fetched", source="spdf")
    print(f"  {out.name}: OK ({entry['lines']} lines, {entry['bytes']} bytes) [SPDF]")
    return entry


def fetch_amda_info() -> dict[str, object]:
    url = f"{AMDA_HAPI}/info?id={AMDA_DATASET_ID}"
    return json.loads(fetch_text(url))


def fetch_amda_year(
    year: int, *, skip_existing: bool, amda_info: dict[str, object] | None
) -> dict[str, object]:
    out = OUT_DIR / f"omni2_{year}_amda_hourly.csv"
    start = f"{year:04d}-01-01T00:00:00Z"
    end = f"{year:04d}-12-31T23:59:59Z"
    url = (
        f"{AMDA_HAPI}/data?id={AMDA_DATASET_ID}"
        f"&time.min={start}&time.max={end}&format=csv"
    )

    if skip_existing and out.exists():
        return build_file_entry(url, out, "skipped", source="amda")

    if amda_info is not None:
        dataset_start = str(amda_info.get("startDate", ""))
        dataset_stop = str(amda_info.get("stopDate", ""))
        if dataset_start and start < dataset_start:
            return {
                "url": url,
                "path": str(out),
                "status": "failed",
                "source": "amda",
                "reason": f"year_before_dataset_start:{dataset_start}",
            }
        if dataset_stop and start > dataset_stop:
            return {
                "url": url,
                "path": str(out),
                "status": "failed",
                "source": "amda",
                "reason": f"year_after_dataset_stop:{dataset_stop}",
            }

    try:
        text = fetch_text(url)
    except (URLError, OSError) as exc:
        return {
            "url": url,
            "path": str(out),
            "status": "failed",
            "source": "amda",
            "reason": str(exc),
        }

    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if len(lines) < 24:
        return {
            "url": url,
            "path": str(out),
            "status": "failed",
            "source": "amda",
            "reason": f"unexpectedly_small:{len(lines)}",
        }

    out.write_text(text, encoding="utf-8")
    entry = build_file_entry(url, out, "fetched", source="amda")
    print(f"  {out.name}: OK ({entry['lines']} lines, {entry['bytes']} bytes) [AMDA]")
    return entry


def has_existing_year(year: int) -> Path | None:
    candidates = [
        OUT_DIR / f"omni2_{year}.dat",
        OUT_DIR / f"omni2_{year}_amda_hourly.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def fetch_year_auto(
    year: int, *, skip_existing: bool, amda_info: dict[str, object] | None
) -> dict[str, object]:
    existing = has_existing_year(year)
    if skip_existing and existing is not None:
        source = "spdf" if existing.suffix == ".dat" else "amda"
        return build_file_entry(
            "local://existing",
            existing,
            "skipped",
            source=source,
            reason="skip_existing",
        )

    spdf_entry = fetch_spdf_year(year, skip_existing=False)
    if spdf_entry["status"] in {"fetched", "skipped"}:
        return spdf_entry
    return fetch_amda_year(year, skip_existing=False, amda_info=amda_info)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch governed hourly OMNI solar wind + IMF data."
    )
    parser.add_argument(
        "--start", type=int, default=2020, help="Start year (inclusive, default 2020)"
    )
    parser.add_argument(
        "--end", type=int, default=2025, help="End year (inclusive, default 2025)"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "spdf", "amda"],
        default="auto",
        help="Preferred source ladder (default: auto)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if a governed file for that year already exists",
    )
    args = parser.parse_args()

    if args.end < args.start:
        print("--end must be >= --start", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    amda_info: dict[str, object] | None = None
    if args.source in {"auto", "amda"}:
        try:
            amda_info = fetch_amda_info()
        except (URLError, OSError, json.JSONDecodeError) as exc:
            if args.source == "amda":
                print(f"AMDA info fetch failed: {exc}", file=sys.stderr)
                return 1
            print(f"AMDA info unavailable for fallback: {exc}")

    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    for year in range(args.start, args.end + 1):
        if args.source == "spdf":
            entry = fetch_spdf_year(year, skip_existing=args.skip_existing)
        elif args.source == "amda":
            entry = fetch_amda_year(year, skip_existing=args.skip_existing, amda_info=amda_info)
        else:
            entry = fetch_year_auto(year, skip_existing=args.skip_existing, amda_info=amda_info)

        counts[str(entry["status"])] += 1
        files.append(entry)

    write_manifest(
        args.start,
        args.end,
        {
            "status": (
                "complete"
                if counts["failed"] == 0
                else "failed"
                if counts["fetched"] == 0 and counts["skipped"] == 0
                else "partial"
            ),
            "counts": counts,
            "source_mode": args.source,
            "source_contracts": sorted(
                {
                    str(file["source_contract"])
                    for file in files
                    if "source_contract" in file
                }
            ),
            "spdf_base": SPDF_BASE,
            "amda_hapi": f"{AMDA_HAPI}/catalog",
            "amda_dataset_id": AMDA_DATASET_ID,
            "amda_info": amda_info or {},
            "files": files,
        },
    )

    print()
    print(
        f"OMNI hourly fetch: {counts['fetched']} fetched, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )
    print(f"Data directory: {OUT_DIR}")
    return 1 if counts["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
