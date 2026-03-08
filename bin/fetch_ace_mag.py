#!/usr/bin/env python3
"""Fetch ACE MAG Level 2 browse data (16-second cadence).

Downloads daily ASCII files from Caltech ACE Science Center.
Each file is ~300 KB (5400 rows, 12 columns per row).

Source: Smith et al. (1998), Space Sci. Rev. 86, 613
URL: https://izw1.caltech.edu/ACE/ASC/DATA/mag16_browse/

Usage:
    python3 bin/fetch_ace_mag.py                         # fetch 2024 (default)
    python3 bin/fetch_ace_mag.py --start 2020 --end 2024
    python3 bin/fetch_ace_mag.py --skip-existing
    python3 bin/fetch_ace_mag.py --start 2024 --doy-start 120 --doy-end 140
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE_URL = "https://izw1.caltech.edu/ACE/ASC/DATA/mag16_browse/"
OUT_DIR = Path("data/external/ace_mag")
USER_AGENT = "gororoba-fetch/0.1 (research)"


def days_in_year(year: int) -> int:
    """Return 366 for leap years, 365 otherwise."""
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        return 366
    return 365


def fetch_day(year: int, doy: int, *, skip_existing: bool) -> str:
    """Fetch a single daily ACE MAG file. Returns status string."""
    txt_name = f"ACE_MAG16_{year}-{doy:03d}_V4-0.txt"
    zip_name = f"ACE_MAG16_{year}-{doy:03d}_V4-0.zip"
    out = OUT_DIR / txt_name

    if skip_existing and out.exists():
        return "skipped"

    # Try raw ASCII first, then zipped.
    for fname in (txt_name, zip_name):
        url = f"{BASE_URL}{year}/{fname}"
        req = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=30) as resp:  # noqa: S310
                data = resp.read()
        except (URLError, OSError):
            continue

        if len(data) < 100:
            continue

        # If zip, extract the .txt inside.
        if fname.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    members = [m for m in zf.namelist() if m.endswith(".txt")]
                    if not members:
                        continue
                    txt_data = zf.read(members[0])
                    out.write_bytes(txt_data)
            except zipfile.BadZipFile:
                continue
        else:
            out.write_bytes(data)

        lines = sum(1 for _ in out.open())
        size = out.stat().st_size
        print(f"  {txt_name}: OK ({lines} lines, {size} bytes)")
        return "fetched"

    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch ACE MAG L2 browse data (16-sec cadence)."
    )
    parser.add_argument(
        "--start", type=int, default=2024, help="Start year (inclusive, default 2024)"
    )
    parser.add_argument(
        "--end", type=int, default=2024, help="End year (inclusive, default 2024)"
    )
    parser.add_argument(
        "--doy-start", type=int, default=1, help="Start day of year (default 1)"
    )
    parser.add_argument(
        "--doy-end", type=int, default=0, help="End day of year (default 0 = all)"
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
        n_days = days_in_year(year)
        doy_end = args.doy_end if args.doy_end > 0 else n_days
        for doy in range(args.doy_start, min(doy_end, n_days) + 1):
            status = fetch_day(year, doy, skip_existing=args.skip_existing)
            counts[status] += 1
            if status == "failed":
                print(f"  ACE_MAG16_{year}-{doy:03d}: FAILED (not found)")

    print()
    print(
        f"ACE MAG fetch: {counts['fetched']} fetched, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )
    print(f"Data directory: {OUT_DIR}")
    return 1 if counts["failed"] > 0 and counts["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
