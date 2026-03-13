#!/usr/bin/env python3
"""Fetch SPARC rotation curve database (Lelli, McGaugh & Schombert 2016).

Downloads the Spitzer Photometry and Accurate Rotation Curves database:
175 late-type galaxies with high-quality HI/H-alpha rotation curves and
Spitzer [3.6] surface photometry.

Sources (tried in order):
  1. astroweb.case.edu/SPARC/ (primary, maintained by Federico Lelli)
  2. VizieR J/AJ/152/157 (CDS mirror of the published tables)

Files downloaded:
  - SPARC_Lelli2016c.mrt  -- Master table (galaxy properties: distance,
    inclination, luminosity, V_flat, R_eff, etc.)
  - Rotmod_LTG/           -- Individual rotation curve files (175 .dat files)
    Each file has columns: Rad(kpc) Vobs(km/s) errV Vgas Vdisk Vbul SBdisk SBbul

Output: data/external/sparc/

Reference: Lelli, McGaugh & Schombert, AJ 152 (2016) 157.
           http://astroweb.case.edu/SPARC/
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("data/external/sparc")
USER_AGENT = "open_gororoba/0.1 (research)"
TIMEOUT = 60

# Primary source: astroweb.case.edu
PRIMARY_BASE = "http://astroweb.case.edu/SPARC"
MASTER_TABLE_URL = f"{PRIMARY_BASE}/SPARC_Lelli2016c.mrt"

# VizieR mirror: CDS Strasbourg
VIZIER_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/AJ/152/157"
VIZIER_MASTER = f"{VIZIER_BASE}/table1.dat"

# Rotation curve tarball (primary bundles all 175 files)
ROTMOD_TAR_URL = f"{PRIMARY_BASE}/Rotmod_LTG.tar.gz"


def _try_vizquery(catalog_id: str, table: str, dest: Path) -> bool:
    """Try CDSClient vizquery as a final VizieR fallback.

    Install: https://cdsarc.cds.unistra.fr/adql/CDSClient/
    Usage:   vizquery -source=J/AJ/152/157 -out=table1.dat -mime=text/TSV
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


def fetch_master_table() -> bool:
    """Download the SPARC master properties table."""
    dest = OUTPUT_DIR / "SPARC_Lelli2016c.mrt"
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True

    print("  Trying primary (astroweb.case.edu)...")
    if download_url(MASTER_TABLE_URL, dest):
        return True

    print("  Trying VizieR mirror (CDS)...")
    vizier_dest = OUTPUT_DIR / "table1.dat"
    if download_url(VIZIER_MASTER, vizier_dest):
        return True

    print("  Trying vizquery CDSClient...")
    if _try_vizquery("J/AJ/152/157", "table1.dat", OUTPUT_DIR / "table1.dat"):
        return True

    return False


def fetch_rotation_curves() -> bool:
    """Download the rotation curve tarball and extract."""
    tar_dest = OUTPUT_DIR / "Rotmod_LTG.tar.gz"
    rotmod_dir = OUTPUT_DIR / "Rotmod_LTG"

    if rotmod_dir.exists() and any(rotmod_dir.glob("*.dat")):
        n = len(list(rotmod_dir.glob("*.dat")))
        print(f"  Already exists: {rotmod_dir} ({n} .dat files)")
        return True

    print("  Downloading rotation curve tarball...")
    if not download_url(ROTMOD_TAR_URL, tar_dest):
        # Try individual galaxy files from VizieR as fallback
        print("  Tarball failed. VizieR does not bundle individual curves.")
        print("  Try downloading manually from http://astroweb.case.edu/SPARC/")
        return False

    # Extract tarball
    import tarfile

    try:
        with tarfile.open(tar_dest, "r:gz") as tf:
            # Security: only extract .dat files, no path traversal
            safe_members = [
                m
                for m in tf.getmembers()
                if m.name.endswith(".dat")
                and not m.name.startswith("/")
                and ".." not in m.name
            ]
            tf.extractall(OUTPUT_DIR, members=safe_members)
        n = len(list(rotmod_dir.glob("*.dat"))) if rotmod_dir.exists() else 0
        print(f"  Extracted {n} rotation curve files")
        return n > 0
    except (tarfile.TarError, OSError) as e:
        print(f"  Extraction failed: {e}")
        return False


def validate(output_dir: Path) -> bool:
    """Basic validation of downloaded files."""
    master = output_dir / "SPARC_Lelli2016c.mrt"
    if not master.exists():
        master = output_dir / "table1.dat"
    if not master.exists():
        print("  VALIDATION FAILED: no master table found")
        return False

    lines = master.read_text().strip().split("\n")
    # Filter out header/comment lines
    data_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    print(f"  Master table: {len(data_lines)} lines")
    if len(data_lines) < 100:
        print("  WARNING: expected ~175 galaxies, got fewer lines")

    rotmod_dir = output_dir / "Rotmod_LTG"
    if rotmod_dir.exists():
        n_curves = len(list(rotmod_dir.glob("*.dat")))
        print(f"  Rotation curves: {n_curves} files")
        if n_curves < 170:
            print(f"  WARNING: expected ~175 files, got {n_curves}")
    else:
        print("  WARNING: no Rotmod_LTG directory")

    return True


def main() -> int:
    print("=== SPARC Rotation Curve Database ===")
    print(f"Output: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Fetching master table...")
    master_ok = fetch_master_table()

    print("[2/3] Fetching rotation curves...")
    curves_ok = fetch_rotation_curves()

    print("[3/3] Validating...")
    valid = validate(OUTPUT_DIR)

    print()
    if master_ok and curves_ok and valid:
        print("=== SPARC download complete ===")
        return 0
    elif master_ok:
        print("=== Partial success: master table OK, curves may be incomplete ===")
        return 0
    else:
        print("=== SPARC download FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
