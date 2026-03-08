#!/usr/bin/env python3
"""Fetch WIND SWE (plasma) and MFI (magnetic field) data from NASA SPDF.

WIND SWE key-parameter: yearly files with proton density, bulk speed,
thermal speed, flow direction at ~92-second cadence.
WIND MFI key-parameter: monthly hourly-averaged magnetic field in GSE.

Source (SWE): Ogilvie et al. (1995), Space Sci. Rev. 71, 55
Source (MFI): Lepping et al. (1995), Space Sci. Rev. 71, 207
URL (SWE): https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/
URL (MFI): https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/ascii/

Usage:
    python3 bin/fetch_wind.py                         # fetch 2024 (default)
    python3 bin/fetch_wind.py --start 2020 --end 2024
    python3 bin/fetch_wind.py --skip-existing
    python3 bin/fetch_wind.py --swe-only
    python3 bin/fetch_wind.py --mfi-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

SWE_BASE = "https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/"
MFI_BASE = "https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/ascii/"
SWE_DIR = Path("data/external/wind_swe")
MFI_DIR = Path("data/external/wind_mfi")
USER_AGENT = "gororoba-fetch/0.1 (research)"


def fetch_file(url: str, out: Path, *, skip_existing: bool) -> str:
    """Fetch a single file from SPDF. Returns status string."""
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


def fetch_swe(years: range, *, skip_existing: bool) -> dict[str, int]:
    """Fetch WIND SWE key-parameter yearly files."""
    SWE_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for year in years:
        # SPDF naming: wind_kp_unspike{YYYY}.txt
        fname = f"wind_kp_unspike{year}.txt"
        url = f"{SWE_BASE}{fname}"
        out = SWE_DIR / fname
        status = fetch_file(url, out, skip_existing=skip_existing)
        counts[status] += 1
        if status == "failed":
            print(f"  {fname}: FAILED (not found)")

    return counts


def fetch_mfi(years: range, *, skip_existing: bool) -> dict[str, int]:
    """Fetch WIND MFI key-parameter monthly hourly files."""
    MFI_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for year in years:
        for month in range(1, 13):
            # SPDF naming: YYYYMM_wind_mag_1hour.asc
            fname = f"{year}{month:02d}_wind_mag_1hour.asc"
            url = f"{MFI_BASE}{fname}"
            out = MFI_DIR / fname
            status = fetch_file(url, out, skip_existing=skip_existing)
            counts[status] += 1
            if status == "failed":
                print(f"  {fname}: FAILED (not found)")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch WIND SWE + MFI data from NASA SPDF."
    )
    parser.add_argument(
        "--start", type=int, default=2024, help="Start year (inclusive, default 2024)"
    )
    parser.add_argument(
        "--end", type=int, default=2024, help="End year (inclusive, default 2024)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )
    parser.add_argument(
        "--swe-only", action="store_true", help="Fetch only SWE (plasma) data"
    )
    parser.add_argument(
        "--mfi-only", action="store_true", help="Fetch only MFI (magnetic field) data"
    )
    args = parser.parse_args()

    years = range(args.start, args.end + 1)
    do_swe = not args.mfi_only
    do_mfi = not args.swe_only

    total: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    if do_swe:
        print(f"Fetching WIND SWE {args.start}-{args.end}...")
        swe = fetch_swe(years, skip_existing=args.skip_existing)
        for k in total:
            total[k] += swe[k]

    if do_mfi:
        print(f"Fetching WIND MFI {args.start}-{args.end}...")
        mfi = fetch_mfi(years, skip_existing=args.skip_existing)
        for k in total:
            total[k] += mfi[k]

    print()
    print(
        f"WIND fetch: {total['fetched']} fetched, "
        f"{total['skipped']} skipped, {total['failed']} failed"
    )
    if do_swe:
        print(f"SWE directory: {SWE_DIR}")
    if do_mfi:
        print(f"MFI directory: {MFI_DIR}")
    return 1 if total["failed"] > 0 and total["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
