#!/usr/bin/env python3
"""Fetch Pioneer 10 & 11 merged hourly data from NASA SPDF.

Pioneer spacecraft fill the 5-80 AU gap in heliospheric coverage:
  P10: launched 1972, data to ~80 AU (signal lost 2003). Ecliptic plane.
  P11: launched 1973, Jupiter + Saturn flyby. Data to ~45 AU.

Pioneer anomaly context: unexplained sunward acceleration at r > 20 AU.

Source: https://spdf.gsfc.nasa.gov/pub/data/pioneer/

Usage:
    python3 bin/fetch_pioneer.py                                # fetch P10 1980-1985
    python3 bin/fetch_pioneer.py --spacecraft both --start 1972 --end 2000
    python3 bin/fetch_pioneer.py --spacecraft p11 --start 1973 --end 1990
    python3 bin/fetch_pioneer.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

PIONEER10_BASE = "https://spdf.gsfc.nasa.gov/pub/data/pioneer/pioneer10/merged/"
PIONEER11_BASE = "https://spdf.gsfc.nasa.gov/pub/data/pioneer/pioneer11/merged/"
OUTPUT_DIR = Path("data/external/pioneer")
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


def fetch_pioneer(
    spacecraft: str, years: range, *, skip_existing: bool
) -> dict[str, int]:
    """Fetch Pioneer merged hourly files for the given spacecraft and years."""
    if spacecraft == "p10":
        base = PIONEER10_BASE
        label = "10"
        subdir = OUTPUT_DIR / "pioneer10"
    else:
        base = PIONEER11_BASE
        label = "11"
        subdir = OUTPUT_DIR / "pioneer11"

    subdir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for year in years:
        fname = f"p{label}_{year}_merged_hourly.asc"
        url = f"{base}{fname}"
        out = subdir / fname
        status = fetch_file(url, out, skip_existing=skip_existing)
        counts[status] += 1
        if status == "failed":
            print(f"  Pioneer {label} {year}: FAILED (not found)")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Pioneer 10 & 11 merged hourly data from NASA SPDF."
    )
    parser.add_argument(
        "--spacecraft",
        choices=["p10", "p11", "both"],
        default="p10",
        help="Which spacecraft (default: p10)",
    )
    parser.add_argument(
        "--start", type=int, default=1980, help="Start year (inclusive, default 1980)"
    )
    parser.add_argument(
        "--end", type=int, default=1985, help="End year (inclusive, default 1985)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )
    args = parser.parse_args()

    years = range(args.start, args.end + 1)
    targets = ["p10", "p11"] if args.spacecraft == "both" else [args.spacecraft]

    total: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for sc in targets:
        label = "Pioneer 10" if sc == "p10" else "Pioneer 11"
        print(f"Fetching {label} merged hourly {args.start}-{args.end}...")
        result = fetch_pioneer(sc, years, skip_existing=args.skip_existing)
        for k in total:
            total[k] += result[k]

    print()
    print(
        f"Pioneer fetch: {total['fetched']} fetched, "
        f"{total['skipped']} skipped, {total['failed']} failed"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    return 1 if total["failed"] > 0 and total["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
