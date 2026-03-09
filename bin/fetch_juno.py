#!/usr/bin/env python3
"""Fetch Juno cruise phase merged hourly data from NASA SPDF.

Juno JADE/MAG cruise data covers 1-5 AU (2011-2016).

Source: https://spdf.gsfc.nasa.gov/pub/data/juno/

Usage:
    python3 bin/fetch_juno.py                        # fetch 2011-2016
    python3 bin/fetch_juno.py --start 2013 --end 2015
    python3 bin/fetch_juno.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

JUNO_CRUISE_BASE = "https://spdf.gsfc.nasa.gov/pub/data/juno/merged/"
OUTPUT_DIR = Path("data/external/juno")
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Juno cruise merged hourly data from NASA SPDF."
    )
    parser.add_argument(
        "--start", type=int, default=2011, help="Start year (inclusive, default 2011)"
    )
    parser.add_argument(
        "--end", type=int, default=2016, help="End year (inclusive, default 2016)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    years = range(args.start, args.end + 1)
    total: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    print(f"Fetching Juno cruise hourly {args.start}-{args.end}...")
    for year in years:
        fname = f"juno_{year}_merged_hourly.asc"
        url = f"{JUNO_CRUISE_BASE}{fname}"
        out = OUTPUT_DIR / fname
        status = fetch_file(url, out, skip_existing=args.skip_existing)
        total[status] += 1
        if status == "failed":
            print(f"  Juno {year}: FAILED (not found)")

    print()
    print(
        f"Juno fetch: {total['fetched']} fetched, "
        f"{total['skipped']} skipped, {total['failed']} failed"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    return 1 if total["failed"] > 0 and total["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
