#!/usr/bin/env python3
"""Fetch Ulysses merged hourly data from NASA SPDF.

Ulysses is the ONLY spacecraft with polar solar wind measurements,
providing fast latitude scans from -80 to +80 deg heliographic.
Range: 1.3-5.4 AU. Operated 1990-2009.

Key science periods:
  First fast latitude scan:  1994-1995 (solar minimum)
  Second fast latitude scan: 2000-2001 (solar maximum)
  Third fast latitude scan:  2007-2008 (deep solar minimum)

Source: https://spdf.gsfc.nasa.gov/pub/data/ulysses/

Usage:
    python3 bin/fetch_ulysses.py                              # fetch 1994-1995
    python3 bin/fetch_ulysses.py --start 1990 --end 2009      # full mission
    python3 bin/fetch_ulysses.py --start 2007 --end 2008      # third orbit
    python3 bin/fetch_ulysses.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ULYSSES_MERGED_BASE = "https://spdf.gsfc.nasa.gov/pub/data/ulysses/merged/"
OUTPUT_DIR = Path("data/external/ulysses")
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


def fetch_ulysses(years: range, *, skip_existing: bool) -> dict[str, int]:
    """Fetch Ulysses merged hourly files for the given years."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for year in years:
        fname = f"uly_{year}_merged_hourly.asc"
        url = f"{ULYSSES_MERGED_BASE}{fname}"
        out = OUTPUT_DIR / fname
        status = fetch_file(url, out, skip_existing=skip_existing)
        counts[status] += 1
        if status == "failed":
            print(f"  Ulysses {year}: FAILED (not found)")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Ulysses merged hourly data from NASA SPDF."
    )
    parser.add_argument(
        "--start", type=int, default=1994, help="Start year (inclusive, default 1994)"
    )
    parser.add_argument(
        "--end", type=int, default=1995, help="End year (inclusive, default 1995)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )
    args = parser.parse_args()

    years = range(args.start, args.end + 1)
    print(f"Fetching Ulysses merged hourly {args.start}-{args.end}...")
    result = fetch_ulysses(years, skip_existing=args.skip_existing)

    print()
    print(
        f"Ulysses fetch: {result['fetched']} fetched, "
        f"{result['skipped']} skipped, {result['failed']} failed"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    return 1 if result["failed"] > 0 and result["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
