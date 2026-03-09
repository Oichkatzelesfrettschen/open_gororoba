#!/usr/bin/env python3
"""Fetch Voyager CRS (Cosmic Ray Subsystem) daily ASCII files from SPDF.

CRS data provides energetic particle count rates at MeV energies from both
Voyager spacecraft, enabling Parker modulation potential phi(r) validation.

URL pattern:
  https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager{1|2}/particle/crs/

File naming (inferred from SPDF convention):
  vy{N}crs_{YEAR}.asc

Output:
  data/external/voyager/voyager{1|2}/crs/vy{N}crs_{YEAR}.asc
"""

import argparse
import os
import sys
import urllib.error
import urllib.request

BASE_URLS = {
    1: "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/particle/crs/",
    2: "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/particle/crs/",
}

USER_AGENT = "gororoba-fetch/0.1 (research)"


def fetch_crs_file(spacecraft: int, year: int, out_dir: str) -> str:
    """Fetch one CRS file. Returns 'fetched', 'skipped', or 'failed'."""
    label = str(spacecraft)
    fname = f"vy{label}crs_{year}.asc"
    url = BASE_URLS[spacecraft] + fname
    dest = os.path.join(out_dir, fname)

    if os.path.exists(dest):
        return "skipped"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 100:
            print(f"  [warn] {fname}: response too small ({len(data)} bytes), skipping")
            return "failed"
        os.makedirs(out_dir, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(data)
        print(f"  [ok]   {fname}")
        return "fetched"
    except (urllib.error.URLError, OSError) as exc:
        print(f"  [fail] {fname}: {exc}")
        return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spacecraft",
        type=int,
        nargs="+",
        default=[1, 2],
        choices=[1, 2],
        help="Voyager spacecraft numbers (default: 1 2)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="1977-2024",
        help="Year range as START-END (default: 1977-2024)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/external/voyager",
        help="Output base directory (default: data/external/voyager)",
    )
    args = parser.parse_args()

    start_year, end_year = (int(y) for y in args.years.split("-"))
    years = range(start_year, end_year + 1)

    total_fetched = 0
    total_failed = 0

    for sc in args.spacecraft:
        out_dir = os.path.join(args.out_dir, f"voyager{sc}", "crs")
        print(f"Fetching Voyager {sc} CRS ({start_year}-{end_year}) -> {out_dir}")
        for year in years:
            status = fetch_crs_file(sc, year, out_dir)
            if status == "fetched":
                total_fetched += 1
            elif status == "failed":
                total_failed += 1

    print(f"\nSummary: fetched={total_fetched}, failed={total_failed}")
    if total_failed > 0 and total_fetched == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
