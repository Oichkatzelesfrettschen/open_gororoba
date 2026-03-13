#!/usr/bin/env python3
"""Fetch MaNGA DR17 kinematic summary catalog.

Downloads the MaNGA (Mapping Nearby Galaxies at Apache Point Observatory)
Data Release 17 Data Analysis Pipeline (DAP) summary catalog, which
contains stellar and gas kinematics for ~10,000 unique galaxies observed
with integral field units (IFUs).

The key file is the DAPall summary table, which contains per-galaxy
measurements including:
  - Stellar velocity dispersion profiles
  - H-alpha emission-line velocity fields (rotation curves)
  - NSA (NASA-Sloan Atlas) photometric properties
  - Effective radii, ellipticities, redshifts

For harmonic halo stacking, we use the H-alpha velocity fields to
construct rotation curves, then normalize by NFW-fitted r_s.

Sources (tried in order):
  1. SDSS SAS (Science Archive Server): data.sdss.org/sas/dr17/manga/spectro/analysis/
  2. SDSS mirror (rsync via HTTP): dr17.sdss.org
  3. SDSS SkyServer TAP (catalog metadata only, no kinematic maps; dapall_summary_tap.csv)

Files:
  - dapall-v3_1_1-3.1.0.fits  -- DAP summary table (~50 MB FITS)
  - drpall-v3_1_1.fits         -- DRP summary table (~30 MB FITS, cross-ref)

Output: data/external/manga/

Reference: Westfall et al., AJ 158 (2019) 231 (DAP).
           Bundy et al., ApJ 798 (2015) 7 (MaNGA survey design).
           SDSS DR17: https://www.sdss.org/dr17/manga/
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("data/external/manga")
USER_AGENT = "open_gororoba/0.1 (research)"
TIMEOUT = 300  # Large files, generous timeout

# SDSS SAS DR17 paths
SAS_BASE = "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0"
DAPALL_URL = f"{SAS_BASE}/dapall-v3_1_1-3.1.0.fits"

SAS_DRP_BASE = "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1"
DRPALL_URL = f"{SAS_DRP_BASE}/drpall-v3_1_1.fits"

# SDSS mirror
MIRROR_BASE = "https://dr17.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0"
DAPALL_MIRROR = f"{MIRROR_BASE}/dapall-v3_1_1-3.1.0.fits"

MIRROR_DRP_BASE = "https://dr17.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1"
DRPALL_MIRROR = f"{MIRROR_DRP_BASE}/drpall-v3_1_1.fits"

# File manifest with expected approximate sizes
FILES = [
    {
        "name": "dapall-v3_1_1-3.1.0.fits",
        "primary": DAPALL_URL,
        "mirror": DAPALL_MIRROR,
        "description": "DAP summary table (kinematics, emission lines)",
        "expected_mb": 50,
    },
    {
        "name": "drpall-v3_1_1.fits",
        "primary": DRPALL_URL,
        "mirror": DRPALL_MIRROR,
        "description": "DRP summary table (photometry, cross-ref)",
        "expected_mb": 30,
    },
]


def _try_skyserver_tap() -> bool:
    """Partial fallback: query SDSS SkyServer TAP for DAPall catalog metadata.

    This does NOT provide kinematic maps -- only the summary catalog columns
    used to build dapall_selection.csv. The result is saved as
    dapall_summary_tap.csv and is clearly incomplete relative to the full FITS.

    REST endpoint: skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch
    """
    dest = OUTPUT_DIR / "dapall_summary_tap.csv"
    if dest.exists():
        size_kb = dest.stat().st_size / 1024
        print(f"  Already exists: {dest.name} ({size_kb:.1f} KB)")
        return True

    sql = (
        "SELECT TOP 11000 "
        "plate, ifudsgn, plateifu, mangaid, objra, objdec, "
        "nsa_z, nsa_elpetro_th50_r, nsa_elpetro_absmag_r, "
        "stellar_vel_1re, stellar_sigma_1re, ha_gvel_1re, ha_gsigma_1re "
        "FROM mangaDAPall "
        "WHERE nsa_z > 0.001 AND nsa_z < 0.15 AND ha_gvel_1re > -9999"
    )
    params = urllib.parse.urlencode({"cmd": sql, "format": "csv"})
    url = (
        f"https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch?{params}"
    )
    print("  Trying SDSS SkyServer TAP (catalog metadata only)...")
    if download_url(url, dest):
        print("  NOTE: TAP result contains summary columns only, not kinematic maps.")
        print("  For full rotation curves the FITS dapall file is required.")
        return True
    return False


def download_url(url: str, dest: Path) -> bool:
    """Download url to dest. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
            if b"<!DOCTYPE" in data[:200] or b"<html" in data[:200].lower():
                print(f"  WARNING: {url} returned HTML error page, skipping")
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            size_mb = len(data) / (1024 * 1024)
            print(f"  OK: {dest.name} ({size_mb:.1f} MB)")
            return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  FAILED: {url} ({e})")
        return False


def fetch_file(file_info: dict) -> bool:
    """Download a single file with primary + mirror fallback."""
    dest = OUTPUT_DIR / file_info["name"]
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Already exists: {dest.name} ({size_mb:.1f} MB)")
        return True

    print(f"  Fetching {file_info['name']} ({file_info['description']})...")
    print(f"    Expected size: ~{file_info['expected_mb']} MB")

    print("    Trying SDSS SAS primary...")
    if download_url(file_info["primary"], dest):
        return True

    print("    Trying SDSS mirror...")
    if download_url(file_info["mirror"], dest):
        return True

    # Partial catalog fallback for dapall only (SkyServer TAP, no kinematic maps)
    if "dapall" in file_info["name"]:
        print("    Trying SkyServer TAP (catalog metadata, no kinematic maps)...")
        _try_skyserver_tap()

    return False


def validate(output_dir: Path) -> bool:
    """Basic validation of downloaded FITS files."""
    dapall = output_dir / "dapall-v3_1_1-3.1.0.fits"
    if not dapall.exists():
        print("  VALIDATION FAILED: dapall FITS not found")
        return False

    size_mb = dapall.stat().st_size / (1024 * 1024)
    print(f"  dapall: {size_mb:.1f} MB")

    # Check FITS magic bytes
    with open(dapall, "rb") as f:
        header = f.read(80)
    if not header.startswith(b"SIMPLE"):
        print("  WARNING: dapall does not start with FITS magic 'SIMPLE'")
        return False

    if size_mb < 10:
        print(f"  WARNING: dapall only {size_mb:.1f} MB, expected ~50 MB")

    drpall = output_dir / "drpall-v3_1_1.fits"
    if drpall.exists():
        drp_mb = drpall.stat().st_size / (1024 * 1024)
        print(f"  drpall: {drp_mb:.1f} MB")
    else:
        print("  WARNING: drpall not downloaded (optional but recommended)")

    return True


def main() -> int:
    print("=== MaNGA DR17 Kinematic Catalog ===")
    print(f"Output: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for i, file_info in enumerate(FILES, 1):
        print(f"[{i}/{len(FILES)}] {file_info['name']}")
        if not fetch_file(file_info):
            all_ok = False

    print()
    print(f"[{len(FILES) + 1}/{len(FILES) + 1}] Validating...")
    valid = validate(OUTPUT_DIR)

    print()
    if valid:
        print("=== MaNGA DR17 download complete ===")
        if not all_ok:
            print("  (some optional files missing)")
        return 0
    else:
        print("=== MaNGA DR17 download FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
