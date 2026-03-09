#!/usr/bin/env python3
"""Fetch Cassini cruise phase merged hourly data from NASA SPDF.

Cassini CAPS/MAG cruise data covers 1-9.5 AU (1997-2004).
Saturn orbit data (2004-2017) is mostly magnetospheric.

Source: https://spdf.gsfc.nasa.gov/pub/data/cassini/

Usage:
    python3 bin/fetch_cassini.py                         # fetch 1997-2004
    python3 bin/fetch_cassini.py --start 2000 --end 2004
    python3 bin/fetch_cassini.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

CASSINI_CRUISE_BASE = "https://spdf.gsfc.nasa.gov/pub/data/cassini/merged/"
OUTPUT_DIR = Path("data/external/cassini")
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
        description="Fetch Cassini cruise merged hourly data from NASA SPDF."
    )
    parser.add_argument(
        "--start", type=int, default=1997, help="Start year (inclusive, default 1997)"
    )
    parser.add_argument(
        "--end", type=int, default=2004, help="End year (inclusive, default 2004)"
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

    print(f"Fetching Cassini cruise hourly {args.start}-{args.end}...")
    for year in years:
        fname = f"cassini_{year}_merged_hourly.asc"
        url = f"{CASSINI_CRUISE_BASE}{fname}"
        out = OUTPUT_DIR / fname
        status = fetch_file(url, out, skip_existing=args.skip_existing)
        total[status] += 1
        if status == "failed":
            print(f"  Cassini {year}: FAILED (not found)")

    print()
    print(
        f"Cassini fetch: {total['fetched']} fetched, "
        f"{total['skipped']} skipped, {total['failed']} failed"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    return 1 if total["failed"] > 0 and total["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
