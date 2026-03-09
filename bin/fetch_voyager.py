#!/usr/bin/env python3
"""Fetch Voyager 1 & 2 merged hourly data from NASA SPDF.

Voyager spacecraft provide the deepest heliospheric penetration:
  V1: launched 1977, 157 AU (2024), crossed termination shock 94 AU (2004)
  V2: launched 1977, 134 AU (2024), crossed termination shock 84 AU (2007)

SPDF merged hourly format: Year, DOY, Hour, Distance(AU), Lat, Lon,
|B|, Bx/By/Bz (SE), density, speed, temperature.

Source: https://spdf.gsfc.nasa.gov/pub/data/voyager/

Usage:
    python3 bin/fetch_voyager.py                              # fetch V1 2020
    python3 bin/fetch_voyager.py --spacecraft both --start 2000 --end 2024
    python3 bin/fetch_voyager.py --spacecraft v2 --start 2007 --end 2010
    python3 bin/fetch_voyager.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

VOYAGER1_BASE = "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/merged/"
VOYAGER2_BASE = "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/merged/"
OUTPUT_DIR = Path("data/external/voyager")
USER_AGENT = "gororoba-fetch/0.1 (research)"


def fetch_file(url: str, out: Path, *, skip_existing: bool) -> str:
    """Fetch a single file. Returns status string."""
    if skip_existing and out.exists():
        return "skipped"

    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=60) as resp:  # noqa: S310
            data = resp.read()
    except (URLError, OSError):
        return "failed"

    if len(data) < 100:
        return "failed"

    out.write_bytes(data)
    lines = sum(1 for _ in out.open())
    size = out.stat().st_size
    print(f"  {out.name}: OK ({lines} lines, {size} bytes)")
    return "fetched"


def fetch_voyager(
    spacecraft: str, years: range, *, skip_existing: bool
) -> dict[str, int]:
    """Fetch Voyager merged hourly files for the given spacecraft and years."""
    if spacecraft == "v1":
        base = VOYAGER1_BASE
        label = "1"
        subdir = OUTPUT_DIR / "voyager1"
    else:
        base = VOYAGER2_BASE
        label = "2"
        subdir = OUTPUT_DIR / "voyager2"

    subdir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for year in years:
        fname = f"vy{label}_{year}_merged_hourly.asc"
        url = f"{base}{fname}"
        out = subdir / fname
        status = fetch_file(url, out, skip_existing=skip_existing)
        counts[status] += 1
        if status == "failed":
            print(f"  Voyager {label} {year}: FAILED (not found)")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Voyager 1 & 2 merged hourly data from NASA SPDF."
    )
    parser.add_argument(
        "--spacecraft",
        choices=["v1", "v2", "both"],
        default="v1",
        help="Which spacecraft (default: v1)",
    )
    parser.add_argument(
        "--start", type=int, default=2020, help="Start year (inclusive, default 2020)"
    )
    parser.add_argument(
        "--end", type=int, default=2020, help="End year (inclusive, default 2020)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )
    args = parser.parse_args()

    years = range(args.start, args.end + 1)
    targets = ["v1", "v2"] if args.spacecraft == "both" else [args.spacecraft]

    total: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for sc in targets:
        label = "Voyager 1" if sc == "v1" else "Voyager 2"
        print(f"Fetching {label} merged hourly {args.start}-{args.end}...")
        result = fetch_voyager(sc, years, skip_existing=args.skip_existing)
        for k in total:
            total[k] += result[k]

    print()
    print(
        f"Voyager fetch: {total['fetched']} fetched, "
        f"{total['skipped']} skipped, {total['failed']} failed"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    return 1 if total["failed"] > 0 and total["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
