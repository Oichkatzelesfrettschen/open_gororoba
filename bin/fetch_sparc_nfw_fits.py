#!/usr/bin/env python3
"""Fetch published NFW fits for SPARC galaxies (Li et al. 2020).

Downloads the mass-model decomposition results from Li, Lelli, McGaugh
& Schombert (2020), which provide NFW halo parameters (rho_s, r_s, c200,
V200) for 175 SPARC galaxies. These pre-computed NFW fits are essential
for the harmonic halo stacking analysis: they provide the r_s values
needed to normalize rotation curves to dimensionless r/r_s coordinates.

The published Table 2 contains per-galaxy best-fit parameters:
  Galaxy, D(Mpc), i(deg), Y_disk, Y_bulge, rho_s(Msun/kpc^3), r_s(kpc),
  V200(km/s), c200, chi2red

Sources (tried in order):
  1. VizieR J/ApJS/247/31 (CDS, primary)
  2. Wayback Machine archive
  3. vizquery CDSClient (optional, install from https://cdsarc.cds.unistra.fr/adql/CDSClient/)

Also downloads the supplementary MCMC chains (Table 3) if available,
which provide posterior uncertainties on r_s.

Output: data/external/sparc_nfw/

Reference: Li, Lelli, McGaugh & Schombert, ApJS 247 (2020) 31.
           "A comprehensive catalog of dark matter halo models for
            SPARC galaxies"
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("data/external/sparc_nfw")
USER_AGENT = "open_gororoba/0.1 (research)"
TIMEOUT = 60

# VizieR CDS primary
VIZIER_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/247/31"

# Tables to download
VIZIER_TABLES = {
    "table2.dat": "NFW best-fit parameters (rho_s, r_s, c200, V200, chi2)",
    "table3.dat": "MCMC posterior chains (uncertainties on fits)",
    "table1.dat": "Galaxy sample properties",
    "ReadMe": "Column definitions and format description",
}

# Wayback fallback
WAYBACK_BASE = (
    "https://web.archive.org/web/2024/"
    "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/247/31"
)


def _try_vizquery(catalog_id: str, table: str, dest: Path) -> bool:
    """Try CDSClient vizquery as a final VizieR fallback.

    Install: https://cdsarc.cds.unistra.fr/adql/CDSClient/
    Usage:   vizquery -source=J/ApJS/247/31 -out=table2.dat -mime=text/TSV
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


def fetch_from_vizier() -> bool:
    """Download all tables from VizieR CDS."""
    all_ok = True
    for fname, desc in VIZIER_TABLES.items():
        dest = OUTPUT_DIR / fname
        if dest.exists():
            print(f"  Already exists: {dest.name}")
            continue
        url = f"{VIZIER_BASE}/{fname}"
        print(f"  Fetching {fname} ({desc})...")
        if not download_url(url, dest):
            # Table 3 (MCMC chains) may not always be available
            if fname == "table2.dat":
                all_ok = False
            else:
                print(f"  (optional: {fname})")
    return all_ok


def fetch_from_wayback() -> bool:
    """Download critical tables from Wayback Machine."""
    # Only try for the essential table2.dat (NFW fits)
    dest = OUTPUT_DIR / "table2.dat"
    if dest.exists():
        return True
    url = f"{WAYBACK_BASE}/table2.dat"
    print(f"  Trying Wayback for table2.dat...")
    return download_url(url, dest)


def validate(output_dir: Path) -> bool:
    """Validate the NFW fit table."""
    table2 = output_dir / "table2.dat"
    if not table2.exists():
        print("  VALIDATION FAILED: table2.dat (NFW fits) not found")
        return False

    lines = table2.read_text().strip().split("\n")
    data_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    print(f"  NFW fit table: {len(data_lines)} lines")

    if len(data_lines) < 100:
        print("  WARNING: expected ~175 galaxy fits, got fewer lines")
        print("  (file may include header lines; check ReadMe for format)")

    # Spot-check: look for numeric content
    for ln in data_lines[:3]:
        parts = ln.split()
        if len(parts) < 5:
            print(f"  WARNING: line has fewer than 5 columns: {ln[:60]}...")

    return True


def main() -> int:
    print("=== SPARC NFW Halo Fits (Li et al. 2020) ===")
    print(f"Output: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Trying VizieR CDS...")
    if fetch_from_vizier():
        print("  VizieR OK")
    else:
        print("[2/4] Trying Wayback Machine fallback...")
        if not fetch_from_wayback():
            print("[3/4] Trying vizquery CDSClient...")
            _try_vizquery("J/ApJS/247/31", "table2.dat", OUTPUT_DIR / "table2.dat")

    print("[4/4] Validating...")
    valid = validate(OUTPUT_DIR)

    print()
    if valid:
        print("=== SPARC NFW fits download complete ===")
        return 0
    else:
        print("=== SPARC NFW fits download FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
