#!/usr/bin/env python3
"""Fetch NASA OMNI2 hourly solar wind + IMF data.

Downloads yearly ASCII files from NASA/GSFC SPDF.
Each file is ~2.8 MB (8760 rows, 57 columns per row).

Source: King & Papitashvili (2005), J. Geophys. Res., 110, A02104.
URL: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/

Usage:
    python3 bin/fetch_omni.py                    # fetch 2020-2025 (default)
    python3 bin/fetch_omni.py --start 1990 --end 2024
    python3 bin/fetch_omni.py --skip-existing    # skip if files exist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_"
OUT_DIR = Path("data/external/omni2")


def fetch_year(year: int, *, skip_existing: bool) -> str:
    """Fetch a single yearly OMNI2 file. Returns status string."""
    fname = f"omni2_{year}.dat"
    out = OUT_DIR / fname

    if skip_existing and out.exists():
        return "skipped"

    url = f"{BASE_URL}{year}.dat"
    try:
        urlretrieve(url, out)  # noqa: S310 -- trusted NASA URL
        lines = sum(1 for _ in out.open())
        size = out.stat().st_size
        print(f"  {fname}: OK ({lines} lines, {size} bytes)")
        return "fetched"
    except (URLError, OSError) as exc:
        print(f"  {fname}: FAILED ({exc})")
        out.unlink(missing_ok=True)
        return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch NASA OMNI2 hourly solar wind + IMF data."
    )
    parser.add_argument(
        "--start", type=int, default=2020, help="Start year (inclusive, default 2020)"
    )
    parser.add_argument(
        "--end", type=int, default=2025, help="End year (inclusive, default 2025)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    for year in range(args.start, args.end + 1):
        status = fetch_year(year, skip_existing=args.skip_existing)
        counts[status] += 1

    print()
    print(
        f"OMNI2 fetch: {counts['fetched']} fetched, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )
    print(f"Data directory: {OUT_DIR}")
    return 1 if counts["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
