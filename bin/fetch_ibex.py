#!/usr/bin/env python3
"""Fetch IBEX ENA sky map data.

IBEX detects Energetic Neutral Atoms from the heliospheric boundary.
NOT in-situ solar wind -- used for heliospheric geometry constraints.

Source: https://ibex.princeton.edu/

Usage:
    python3 bin/fetch_ibex.py                      # fetch orbits 1-10
    python3 bin/fetch_ibex.py --start 1 --end 50
    python3 bin/fetch_ibex.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

IBEX_BASE = "https://ibex.princeton.edu/DataAction"
OUTPUT_DIR = Path("data/external/ibex")
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
    size = out.stat().st_size
    print(f"  {out.name}: OK ({size} bytes)")
    return "fetched"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch IBEX ENA sky map data."
    )
    parser.add_argument(
        "--start", type=int, default=1, help="Start orbit (inclusive, default 1)"
    )
    parser.add_argument(
        "--end", type=int, default=10, help="End orbit (inclusive, default 10)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    print(f"Fetching IBEX orbits {args.start}-{args.end}...")
    for orbit in range(args.start, args.end + 1):
        fname = f"ibex_hi_orbit{orbit:03}.csv"
        url = f"{IBEX_BASE}?id=orbit{orbit}"
        out = OUTPUT_DIR / fname
        status = fetch_file(url, out, skip_existing=args.skip_existing)
        total[status] += 1
        if status == "failed":
            print(f"  IBEX orbit {orbit}: FAILED (not found)")

    print()
    print(
        f"IBEX fetch: {total['fetched']} fetched, "
        f"{total['skipped']} skipped, {total['failed']} failed"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    return 1 if total["failed"] > 0 and total["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
