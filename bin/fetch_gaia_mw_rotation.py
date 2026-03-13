#!/usr/bin/env python3
"""Fetch Gaia DR3 Milky Way rotation curve (Jiao et al. 2023).

Downloads the Milky Way circular velocity curve derived from Gaia DR3
astrometry by Jiao, Hammer, Wang et al. (2023). This provides the most
precise single-galaxy rotation curve available, with sub-km/s errors
from 5 to 25 kpc, ideal for testing harmonic halo modulation on the MW.

The table contains:
  R(kpc), V_c(km/s), eV_c_low, eV_c_high (asymmetric uncertainties)

Sources (tried in order):
  1. VizieR J/ApJ/946/73 (CDS, primary)
  2. Zenodo record (Jiao et al. supplementary)
  3. Wayback Machine archive
  4. vizquery CDSClient (optional, install from https://cdsarc.cds.unistra.fr/adql/CDSClient/)

Output: data/external/gaia_mw_rotation/

Reference: Jiao, Hammer, Wang et al., ApJ 946 (2023) 73.
           "Detection of the Keplerian decline in the Milky Way
            rotation curve"
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("data/external/gaia_mw_rotation")
USER_AGENT = "open_gororoba/0.1 (research)"
TIMEOUT = 60

# VizieR CDS primary
VIZIER_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/946/73"
VIZIER_TABLE = f"{VIZIER_BASE}/table1.dat"
VIZIER_README = f"{VIZIER_BASE}/ReadMe"

# Alternative: Ou et al. (2024) Gaia DR3 rotation curve (complementary)
# VizieR J/MNRAS/528/693
VIZIER_OU_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/528/693"
VIZIER_OU_TABLE = f"{VIZIER_OU_BASE}/table1.dat"

# Wayback fallback
WAYBACK_BASE = (
    "https://web.archive.org/web/2024/"
    "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/946/73"
)


def _try_vizquery(catalog_id: str, table: str, dest: Path) -> bool:
    """Try CDSClient vizquery as a final VizieR fallback.

    Install: https://cdsarc.cds.unistra.fr/adql/CDSClient/
    Usage:   vizquery -source=J/ApJ/946/73 -out=table1.dat -mime=text/TSV
    """
    vq = shutil.which("vizquery")
    if not vq:
        print(
            "  vizquery not installed (optional CDSClient fallback:"
            " https://cdsarc.cds.unistra.fr/adql/CDSClient/)"
        )
        return False
    try:
        result = subprocess.run(
            [vq, f"-source={catalog_id}", f"-out={table}", "-mime=text/TSV"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(result.stdout, encoding="utf-8")
            print(f"  vizquery OK: {dest.name} ({len(result.stdout)} bytes)")
            return True
        stderr_snippet = result.stderr[:200] if result.stderr else ""
        print(f"  vizquery failed (exit {result.returncode}): {stderr_snippet}")
        return False
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  vizquery error: {e}")
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
            size_kb = len(data) / 1024
            print(f"  OK: {dest.name} ({size_kb:.1f} KB)")
            return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  FAILED: {url} ({e})")
        return False


def fetch_jiao() -> bool:
    """Download Jiao et al. (2023) rotation curve from VizieR."""
    dest = OUTPUT_DIR / "jiao2023_table1.dat"
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True

    print("  Trying VizieR CDS (Jiao et al. 2023)...")
    if download_url(VIZIER_TABLE, dest):
        download_url(VIZIER_README, OUTPUT_DIR / "ReadMe_jiao2023")
        return True

    print("  Trying Wayback Machine...")
    if download_url(f"{WAYBACK_BASE}/table1.dat", dest):
        return True

    print("  Trying vizquery CDSClient (Jiao et al. 2023)...")
    if _try_vizquery("J/ApJ/946/73", "table1.dat", dest):
        return True

    return False


def fetch_ou() -> bool:
    """Download Ou et al. (2024) complementary rotation curve."""
    dest = OUTPUT_DIR / "ou2024_table1.dat"
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True

    print("  Trying VizieR CDS (Ou et al. 2024, complementary)...")
    if download_url(VIZIER_OU_TABLE, dest):
        download_url(
            f"{VIZIER_OU_BASE}/ReadMe", OUTPUT_DIR / "ReadMe_ou2024"
        )
        return True

    print("  Trying vizquery CDSClient (Ou et al. 2024)...")
    if _try_vizquery("J/MNRAS/528/693", "table1.dat", dest):
        return True

    return False


def validate(output_dir: Path) -> bool:
    """Basic validation of the rotation curve data."""
    jiao = output_dir / "jiao2023_table1.dat"
    if not jiao.exists():
        print("  VALIDATION FAILED: jiao2023_table1.dat not found")
        return False

    lines = jiao.read_text().strip().split("\n")
    data_lines = [
        ln for ln in lines if ln.strip() and not ln.startswith("#")
    ]
    print(f"  Jiao et al. (2023): {len(data_lines)} data points")

    # Expect radial bins from ~5 to ~25 kpc
    if len(data_lines) < 10:
        print("  WARNING: expected tens of radial bins, got fewer")

    # Check for Ou et al. complementary data
    ou = output_dir / "ou2024_table1.dat"
    if ou.exists():
        ou_lines = [
            ln
            for ln in ou.read_text().strip().split("\n")
            if ln.strip() and not ln.startswith("#")
        ]
        print(f"  Ou et al. (2024): {len(ou_lines)} data points")

    return True


def main() -> int:
    print("=== Gaia DR3 Milky Way Rotation Curve ===")
    print(f"Output: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Fetching Jiao et al. (2023) primary curve...")
    jiao_ok = fetch_jiao()

    print("[2/3] Fetching Ou et al. (2024) complementary curve...")
    ou_ok = fetch_ou()
    if not ou_ok:
        print("  (optional, continuing)")

    print("[3/3] Validating...")
    valid = validate(OUTPUT_DIR)

    print()
    if jiao_ok and valid:
        print("=== Gaia MW rotation curve download complete ===")
        return 0
    else:
        print("=== Gaia MW rotation curve download FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
