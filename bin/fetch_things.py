#!/usr/bin/env python3
"""Fetch THINGS catalog tables (Walter et al. 2008).

Downloads The HI Nearby Galaxy Survey (THINGS) catalog from VizieR.
THINGS observed 34 nearby galaxies in 21-cm HI with the VLA at high
angular (~6") and spectral (~5 km/s) resolution.

The VizieR catalog J/AJ/136/2563 contains:
  - table1.dat: Galaxy properties (positions, distances, morphology)
  - table4.dat: HI properties (velocity, linewidth, flux, HI mass)
  - refs.dat:   Reference codes

NOTE: These are catalog summary tables, NOT the per-galaxy rotation
curves. The actual tilted-ring rotation curves (de Blok et al. 2008)
and full data products (cubes, moment maps) are hosted at the MPIA
THINGS Data Products page: https://www2.mpia-hd.mpg.de/THINGS/Data.html

Sources (tried in order):
  1. CDSARC FTP J/AJ/136/2563 (primary, CDS Strasbourg)
  2. VizieR TSV export (fallback)
  3. VizieR mirrors: IUCAA, INASAN
  4. vizquery CDSClient (optional, install from https://cdsarc.cds.unistra.fr/adql/CDSClient/)

Output: data/external/things/

Reference: Walter, Brinks, de Blok et al., AJ 136 (2008) 2563.
           "THINGS: The HI Nearby Galaxy Survey"
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("data/external/things")
USER_AGENT = "open_gororoba/0.1 (research)"
TIMEOUT = 60

# CDSARC FTP primary -- raw catalog files
VIZIER_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/AJ/136/2563"

# Catalog tables to download
VIZIER_TABLES = {
    "table1.dat": "Galaxy properties (positions, distances, morphology)",
    "table4.dat": "HI properties (velocity, linewidth, flux, HI mass)",
    "refs.dat": "Reference codes",
}

# VizieR TSV export (fallback, same data in tab-separated format)
VIZIER_TSV = {
    "table1.tsv": (
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
        "?-source=J/AJ/136/2563/table1&-out.max=unlimited"
    ),
    "table4.tsv": (
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
        "?-source=J/AJ/136/2563/table4&-out.max=unlimited"
    ),
}

# VizieR mirrors (second fallback tier)
VIZIER_MIRRORS = [
    "https://vizier.iucaa.in/viz-bin/asu-tsv",
    "https://vizier.inasan.ru/viz-bin/asu-tsv",
]


def _try_vizquery(catalog_id: str, table: str, dest: Path) -> bool:
    """Try CDSClient vizquery as a final VizieR fallback.

    Install: https://cdsarc.cds.unistra.fr/adql/CDSClient/
    Usage:   vizquery -source=J/AJ/136/2563 -out=table1.dat -mime=text/TSV
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


def fetch_from_cdsarc() -> bool:
    """Download raw catalog files from CDSARC FTP."""
    all_ok = True
    for fname, desc in VIZIER_TABLES.items():
        dest = OUTPUT_DIR / fname
        if dest.exists():
            print(f"  Already exists: {dest}")
            continue
        url = f"{VIZIER_BASE}/{fname}"
        print(f"  Fetching {fname} ({desc})...")
        if not download_url(url, dest):
            all_ok = False
    # Also grab the ReadMe for column definitions
    readme_dest = OUTPUT_DIR / "ReadMe"
    if not readme_dest.exists():
        download_url(f"{VIZIER_BASE}/ReadMe", readme_dest)
    return all_ok


def fetch_from_vizier_tsv() -> bool:
    """Download tables via VizieR TSV export (fallback)."""
    all_ok = True
    for fname, url in VIZIER_TSV.items():
        dest = OUTPUT_DIR / fname
        if dest.exists():
            print(f"  Already exists: {dest}")
            continue
        print(f"  Fetching {fname} via VizieR TSV export...")
        if not download_url(url, dest):
            all_ok = False
    return all_ok


def fetch_from_mirrors() -> bool:
    """Download tables via VizieR mirrors (IUCAA, INASAN)."""
    for mirror_base in VIZIER_MIRRORS:
        print(f"  Trying mirror: {mirror_base}...")
        all_ok = True
        for tname in ["table1", "table4"]:
            dest = OUTPUT_DIR / f"{tname}_mirror.tsv"
            if dest.exists():
                continue
            url = f"{mirror_base}?-source=J/AJ/136/2563/{tname}&-out.max=unlimited"
            if not download_url(url, dest):
                all_ok = False
        if all_ok:
            return True
    return False


def validate(output_dir: Path) -> bool:
    """Basic validation of downloaded catalog tables."""
    ok = True

    # Check table1.dat (galaxy properties, ~34 galaxies)
    table1 = output_dir / "table1.dat"
    if table1.exists():
        lines = table1.read_text().strip().split("\n")
        data_lines = [ln for ln in lines if ln.strip() and not ln.startswith(("#", "-"))]
        print(f"  table1.dat (galaxy properties): {len(data_lines)} data lines")
        if len(data_lines) < 10:
            print(f"  WARNING: expected ~34 galaxies, got {len(data_lines)}")
    else:
        # Check for TSV fallback
        table1_tsv = output_dir / "table1.tsv"
        if table1_tsv.exists():
            lines = table1_tsv.read_text().strip().split("\n")
            print(f"  table1.tsv (galaxy properties): {len(lines)} lines")
        else:
            print("  WARNING: neither table1.dat nor table1.tsv found")
            ok = False

    # Check table4.dat (HI properties)
    table4 = output_dir / "table4.dat"
    if table4.exists():
        lines = table4.read_text().strip().split("\n")
        data_lines = [ln for ln in lines if ln.strip() and not ln.startswith(("#", "-"))]
        print(f"  table4.dat (HI properties): {len(data_lines)} data lines")
    else:
        table4_tsv = output_dir / "table4.tsv"
        if table4_tsv.exists():
            lines = table4_tsv.read_text().strip().split("\n")
            print(f"  table4.tsv (HI properties): {len(lines)} lines")
        else:
            print("  WARNING: neither table4.dat nor table4.tsv found")
            ok = False

    return ok


def main() -> int:
    print("=== THINGS Catalog (Walter et al. 2008, J/AJ/136/2563) ===")
    print(f"Output: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Trying CDSARC FTP (primary)...")
    if fetch_from_cdsarc():
        print("  CDSARC OK")
    else:
        print("[2/5] Trying VizieR TSV export (fallback)...")
        if fetch_from_vizier_tsv():
            print("  VizieR TSV OK")
        else:
            print("[3/5] Trying VizieR mirrors (IUCAA, INASAN)...")
            if not fetch_from_mirrors():
                print("[4/5] Trying vizquery CDSClient...")
                _try_vizquery(
                    "J/AJ/136/2563", "table1.dat", OUTPUT_DIR / "table1.dat"
                )

    print("[5/5] Validating...")
    valid = validate(OUTPUT_DIR)

    print()
    if valid:
        print("=== THINGS catalog download complete ===")
        print("NOTE: For rotation curves, see MPIA THINGS Data Products:")
        print("  https://www2.mpia-hd.mpg.de/THINGS/Data.html")
        return 0
    else:
        print("=== THINGS catalog download FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
