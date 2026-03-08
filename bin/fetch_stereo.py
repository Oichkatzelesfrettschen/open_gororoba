#!/usr/bin/env python3
"""Fetch STEREO-A PLASTIC (plasma) and IMPACT/MAG data from NASA SPDF.

STEREO-A PLASTIC provides proton density, bulk speed, thermal speed,
and RTN velocity components at 1-minute cadence.
STEREO-A IMPACT/MAG provides magnetic field in RTN coordinates.

Source (PLASTIC): Galvin et al. (2008), Space Sci. Rev. 136, 437
Source (IMPACT): Luhmann et al. (2008), Space Sci. Rev. 136, 401
URL: https://spdf.gsfc.nasa.gov/pub/data/stereo/ahead/l2/

Usage:
    python3 bin/fetch_stereo.py                         # fetch 2024 (default)
    python3 bin/fetch_stereo.py --start 2020 --end 2024
    python3 bin/fetch_stereo.py --skip-existing
    python3 bin/fetch_stereo.py --plastic-only
    python3 bin/fetch_stereo.py --mag-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# STEREO-A PLASTIC daily files from STEREO Science Center
PLASTIC_BASE = (
    "https://stereo-ssc.nascom.nasa.gov/data/ins_data/plastic/level2/Protons/Derived/"
)
# STEREO-A IMPACT/MAG from SPDF (RTN coordinates, 1-minute averages)
MAG_BASE = "https://spdf.gsfc.nasa.gov/pub/data/stereo/ahead/l2/impact/magplasma/"
PLASTIC_DIR = Path("data/external/stereo_plastic")
MAG_DIR = Path("data/external/stereo_impact")
USER_AGENT = "gororoba-fetch/0.1 (research)"


def days_in_year(year: int) -> int:
    """Return 366 for leap years, 365 otherwise."""
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        return 366
    return 365


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


def fetch_plastic(
    years: range, *, skip_existing: bool, doy_start: int, doy_end: int
) -> dict[str, int]:
    """Fetch STEREO-A PLASTIC daily proton files."""
    PLASTIC_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for year in years:
        n_days = days_in_year(year)
        end = doy_end if doy_end > 0 else n_days
        for doy in range(doy_start, min(end, n_days) + 1):
            # PLASTIC daily file naming convention
            fname = f"STA_L2_PLA_1DMax_1min_{year}{doy:03d}_V13.txt"
            url = f"{PLASTIC_BASE}{year}/{fname}"
            out = PLASTIC_DIR / fname
            status = fetch_file(url, out, skip_existing=skip_existing)
            counts[status] += 1
            if status == "failed":
                # Try alternate version numbers
                for ver in ("V12", "V11", "V10"):
                    alt = fname.replace("V13", ver)
                    url2 = f"{PLASTIC_BASE}{year}/{alt}"
                    status = fetch_file(url2, PLASTIC_DIR / alt, skip_existing=skip_existing)
                    if status == "fetched":
                        counts["fetched"] += 1
                        counts["failed"] -= 1
                        break
                else:
                    print(f"  PLASTIC_{year}-{doy:03d}: FAILED (not found)")

    return counts


def fetch_mag(years: range, *, skip_existing: bool) -> dict[str, int]:
    """Fetch STEREO-A IMPACT/MAG monthly files from SPDF."""
    MAG_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for year in years:
        for month in range(1, 13):
            # SPDF MAGPLASMA ASCII naming
            fname = f"sta_l2_magplasma_1min_{year}{month:02d}01_v01.asc"
            url = f"{MAG_BASE}{year}/{fname}"
            out = MAG_DIR / fname
            status = fetch_file(url, out, skip_existing=skip_existing)
            counts[status] += 1
            if status == "failed":
                print(f"  MAGPLASMA_{year}-{month:02d}: FAILED (not found)")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch STEREO-A PLASTIC + IMPACT/MAG data."
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
    parser.add_argument(
        "--plastic-only", action="store_true", help="Fetch only PLASTIC (plasma) data"
    )
    parser.add_argument(
        "--mag-only", action="store_true", help="Fetch only IMPACT/MAG data"
    )
    args = parser.parse_args()

    years = range(args.start, args.end + 1)
    do_plastic = not args.mag_only
    do_mag = not args.plastic_only

    total: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    if do_plastic:
        print(f"Fetching STEREO-A PLASTIC {args.start}-{args.end}...")
        plastic = fetch_plastic(
            years,
            skip_existing=args.skip_existing,
            doy_start=args.doy_start,
            doy_end=args.doy_end,
        )
        for k in total:
            total[k] += plastic[k]

    if do_mag:
        print(f"Fetching STEREO-A IMPACT/MAG {args.start}-{args.end}...")
        mag = fetch_mag(years, skip_existing=args.skip_existing)
        for k in total:
            total[k] += mag[k]

    print()
    print(
        f"STEREO-A fetch: {total['fetched']} fetched, "
        f"{total['skipped']} skipped, {total['failed']} failed"
    )
    if do_plastic:
        print(f"PLASTIC directory: {PLASTIC_DIR}")
    if do_mag:
        print(f"MAG directory: {MAG_DIR}")
    return 1 if total["failed"] > 0 and total["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
