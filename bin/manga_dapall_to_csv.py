#!/usr/bin/env python3
"""Convert MaNGA DAPall + DRPall FITS tables to CSV for Rust ingestion.

Reads the MaNGA DR17 DAPall summary FITS table (kinematic measurements)
and cross-references DRPall (photometric properties) to produce a unified
selection catalog.

DAPall provides:
  - Kinematic measurements (stellar and H-alpha velocity dispersion)
  - Emission-line properties (H-alpha surface brightness, EW)
  - Quality flags (DAPQUAL, DAPDONE)

DRPall provides:
  - NSA photometric properties (stellar mass, absolute magnitude)
  - Galaxy identification (PLATEIFU, MANGAID)
  - NSA axis ratio, Sersic index, effective radius

The output CSV is used by the Rust pipeline to:
  1. Select galaxies for MAPS file download (quality + property cuts)
  2. Provide NFW fitting priors (stellar mass -> halo mass via SMHM)

Usage:
  python3 bin/manga_dapall_to_csv.py [--dapall PATH] [--drpall PATH] [--output PATH]

Reference: Westfall et al., AJ 158 (2019) 231 (DAP).
           Law et al., AJ 152 (2016) 83 (DRP).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    from astropy.io import fits
    import numpy as np
except ImportError:
    print("ERROR: requires astropy and numpy", file=sys.stderr)
    print("  pip install astropy numpy", file=sys.stderr)
    sys.exit(1)

# DAPall extension name for the SPX-MILESHC-MASTARSSP analysis
# In DR17, each DAPTYPE has its own extension
DAPTYPE_EXT = "SPX-MILESHC-MASTARSSP"
FALLBACK_EXT = "DAPALL"

# H-alpha channel index in emission-line arrays (0-indexed: 23 in DR17)
HA_CHANNEL = 23

# Output columns for the CSV
OUTPUT_COLUMNS = [
    "plateifu",
    "mangaid",
    "plate",
    "ifudesign",
    "z",
    "nsa_elpetro_ba",       # axis ratio b/a -> inclination
    "nsa_elpetro_phi",      # position angle
    "nsa_elpetro_th50_r",   # effective radius (arcsec)
    "nsa_sersic_n",         # Sersic index
    "nsa_sersic_ba",        # Sersic b/a
    "nsa_elpetro_absmag_r", # absolute r-band magnitude (from DRPall)
    "nsa_elpetro_mass",     # log stellar mass (log Msun, from DRPall)
    "stellar_sigma_1re",    # stellar velocity dispersion within 1 R_e (km/s)
    "ha_gsb_1re",           # H-alpha surface brightness within 1 R_e
    "ha_gew_1re",           # H-alpha equivalent width within 1 R_e
    "ha_z",                 # H-alpha redshift
    "ha_gvel_lo",           # H-alpha velocity 2.5% percentile (km/s)
    "ha_gvel_hi",           # H-alpha velocity 97.5% percentile (km/s)
    "ha_gsigma_1re",        # H-alpha gas velocity dispersion within 1 R_e (km/s)
    "emline_rchi2_1re",     # emission-line fit reduced chi2 within 1 R_e
    "dapqual",              # DAP quality bitmask
]


def load_dapall(dapall_path: str) -> np.ndarray:
    """Load the DAPall table from FITS."""
    with fits.open(dapall_path) as hdul:
        for ext_name in [DAPTYPE_EXT, FALLBACK_EXT]:
            try:
                data = hdul[ext_name].data
                print(f"  Loaded extension '{ext_name}': {len(data)} rows")
                return data
            except KeyError:
                continue
        data = hdul[1].data
        print(f"  Loaded extension index 1: {len(data)} rows")
        return data


def load_drpall(drpall_path: str) -> dict:
    """Load DRPall and build PLATEIFU -> photometric properties lookup.

    DRPall DR17 uses lowercase column names:
      - plateifu (identifier)
      - nsa_elpetro_absmag: 7-band ugrizJK absolute magnitude array
      - nsa_elpetro_mass: scalar total stellar mass in linear M_sun
    """
    lookup = {}
    with fits.open(drpall_path) as hdul:
        data = hdul[1].data
        print(f"  DRPall: {len(data)} rows")

        col_names = [c.name for c in data.columns]
        # DRPall uses lowercase column names
        plateifu_col = "plateifu" if "plateifu" in col_names else "PLATEIFU"
        absmag_col = next((c for c in col_names
                           if c.lower() == "nsa_elpetro_absmag"), None)
        mass_col = next((c for c in col_names
                         if c.lower() == "nsa_elpetro_mass"), None)
        print(f"  plateifu={plateifu_col}, absmag={absmag_col}, mass={mass_col}")

        for i in range(len(data)):
            plateifu = str(data[i][plateifu_col]).strip()

            absmag_r = float("nan")
            if absmag_col:
                try:
                    absmag_arr = data[i][absmag_col]
                    # 7-band ugrizJK: r-band is index 4
                    absmag_r = float(absmag_arr[4])
                except (IndexError, ValueError):
                    pass

            log_mass = float("nan")
            if mass_col:
                try:
                    # Scalar total mass in linear M_sun
                    m = float(data[i][mass_col])
                    log_mass = float(np.log10(m)) if m > 0 else float("nan")
                except (ValueError, IndexError):
                    pass

            lookup[plateifu] = {
                "absmag_r": absmag_r,
                "log_mass": log_mass,
            }

    n_valid_mass = sum(1 for v in lookup.values() if not np.isnan(v["log_mass"]))
    print(f"  Valid stellar masses: {n_valid_mass}/{len(lookup)}")
    return lookup


def safe_get(row, col, default=float("nan")):
    """Safely extract a scalar column value, returning default if missing."""
    try:
        val = row[col]
        if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
            return val  # array column, return as-is
        return float(val) if val is not None else default
    except (KeyError, ValueError, IndexError):
        return default


def extract_row(row, drpall_lookup: dict) -> dict | None:
    """Extract one galaxy's data from a DAPall row + DRPall photometry.

    Returns None if the galaxy fails basic quality cuts.
    """
    # DAPDONE flag: skip incomplete analyses
    # DR17 stores DAPDONE as numpy bool, not character 'T'/'F'
    try:
        dapdone = row["DAPDONE"]
        if not bool(dapdone):
            return None
    except (KeyError, ValueError):
        pass

    # Quality bitmask: bit 30 = CRITICAL
    dapqual = int(row["DAPQUAL"])
    if dapqual & (1 << 30):
        return None

    plateifu = str(row["PLATEIFU"]).strip()
    mangaid = str(row["MANGAID"]).strip()

    z = safe_get(row, "Z")
    if not (0.0 < z < 0.15):
        return None

    # NSA photometric properties (from DAPall)
    nsa_ba = safe_get(row, "NSA_ELPETRO_BA")
    nsa_phi = safe_get(row, "NSA_ELPETRO_PHI")
    nsa_th50 = safe_get(row, "NSA_ELPETRO_TH50_R")
    nsa_sersic_n = safe_get(row, "NSA_SERSIC_N")
    nsa_sersic_ba = safe_get(row, "NSA_SERSIC_BA")

    # Photometric properties from DRPall cross-reference
    drp = drpall_lookup.get(plateifu, {})
    absmag_r = drp.get("absmag_r", float("nan"))
    log_mass = drp.get("log_mass", float("nan"))

    # Stellar kinematics (from DAPall)
    stellar_sigma = safe_get(row, "STELLAR_SIGMA_1RE")

    # H-alpha emission line properties (from DAPall)
    ha_gsb = float("nan")
    try:
        emline_gsb = row["EMLINE_GSB_1RE"]
        ha_gsb = float(emline_gsb[HA_CHANNEL])
    except (KeyError, IndexError, ValueError):
        pass

    ha_gew = float("nan")
    try:
        emline_gew = row["EMLINE_GEW_1RE"]
        ha_gew = float(emline_gew[HA_CHANNEL])
    except (KeyError, IndexError, ValueError):
        pass

    emline_rchi2 = safe_get(row, "EMLINE_RCHI2_1RE")
    ha_z = safe_get(row, "HA_Z")
    ha_gvel_lo = safe_get(row, "HA_GVEL_LO")
    ha_gvel_hi = safe_get(row, "HA_GVEL_HI")
    ha_gsigma = safe_get(row, "HA_GSIGMA_1RE")

    plate = int(plateifu.split("-")[0]) if "-" in plateifu else 0
    ifudesign = plateifu.split("-")[1] if "-" in plateifu else ""

    return {
        "plateifu": plateifu,
        "mangaid": mangaid,
        "plate": plate,
        "ifudesign": ifudesign,
        "z": z,
        "nsa_elpetro_ba": nsa_ba,
        "nsa_elpetro_phi": nsa_phi,
        "nsa_elpetro_th50_r": nsa_th50,
        "nsa_sersic_n": nsa_sersic_n,
        "nsa_sersic_ba": nsa_sersic_ba,
        "nsa_elpetro_absmag_r": absmag_r,
        "nsa_elpetro_mass": log_mass,
        "stellar_sigma_1re": stellar_sigma,
        "ha_gsb_1re": ha_gsb,
        "ha_gew_1re": ha_gew,
        "ha_z": ha_z,
        "ha_gvel_lo": ha_gvel_lo,
        "ha_gvel_hi": ha_gvel_hi,
        "ha_gsigma_1re": ha_gsigma,
        "emline_rchi2_1re": emline_rchi2,
        "dapqual": dapqual,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert MaNGA DAPall+DRPall FITS to CSV")
    parser.add_argument(
        "--dapall",
        default="data/external/manga/dapall-v3_1_1-3.1.0.fits",
        help="Path to DAPall FITS file",
    )
    parser.add_argument(
        "--drpall",
        default="data/external/manga/drpall-v3_1_1.fits",
        help="Path to DRPall FITS file (provides stellar mass, abs mag)",
    )
    parser.add_argument(
        "--output",
        default="data/external/manga/dapall_selection.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    dapall_path = Path(args.dapall)
    if not dapall_path.exists():
        print(f"ERROR: {dapall_path} not found", file=sys.stderr)
        print("  Run: python3 bin/fetch_manga_kinematics.py", file=sys.stderr)
        return 1

    drpall_path = Path(args.drpall)
    drpall_lookup = {}
    if drpall_path.exists():
        print(f"Reading DRPall from {drpall_path}...")
        drpall_lookup = load_drpall(str(drpall_path))
    else:
        print(f"WARNING: {drpall_path} not found, stellar masses will be NaN")
        print("  Run: python3 bin/fetch_manga_kinematics.py")

    print(f"Reading DAPall from {dapall_path}...")
    data = load_dapall(str(dapall_path))

    print("Extracting galaxy properties...")
    rows = []
    n_critical = 0
    n_incomplete = 0
    n_z_cut = 0
    for i in range(len(data)):
        result = extract_row(data[i], drpall_lookup)
        if result is None:
            try:
                if not bool(data[i]["DAPDONE"]):
                    n_incomplete += 1
                    continue
            except (KeyError, ValueError):
                pass
            dq = int(data[i]["DAPQUAL"])
            if dq & (1 << 30):
                n_critical += 1
            else:
                n_z_cut += 1
            continue
        rows.append(result)

    print(f"  Total rows: {len(data)}")
    print(f"  Passed: {len(rows)}")
    print(f"  Rejected (incomplete DAPDONE): {n_incomplete}")
    print(f"  Rejected (CRITICAL quality): {n_critical}")
    print(f"  Rejected (z cut / other): {n_z_cut}")

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} galaxies to {output_path}")

    # Summary statistics
    masses = [r["nsa_elpetro_mass"] for r in rows
              if isinstance(r["nsa_elpetro_mass"], (int, float))
              and not np.isnan(r["nsa_elpetro_mass"])]
    sigmas = [r["stellar_sigma_1re"] for r in rows
              if isinstance(r["stellar_sigma_1re"], (int, float))
              and not np.isnan(r["stellar_sigma_1re"]) and r["stellar_sigma_1re"] > 0]
    zs = [r["z"] for r in rows]

    print(f"\nSummary:")
    if zs:
        print(f"  Redshift: {np.median(zs):.4f} (median), [{min(zs):.4f}, {max(zs):.4f}]")
    if masses:
        print(f"  log(M*/Msun): {np.median(masses):.2f} (median), [{min(masses):.2f}, {max(masses):.2f}]")
    else:
        print("  log(M*/Msun): NO VALID MASSES (DRPall missing or no cross-match)")
    if sigmas:
        print(f"  sigma_* (km/s): {np.median(sigmas):.1f} (median), [{min(sigmas):.1f}, {max(sigmas):.1f}]")

    # DRPall cross-match statistics
    n_with_mass = sum(1 for r in rows
                      if isinstance(r["nsa_elpetro_mass"], (int, float))
                      and not np.isnan(r["nsa_elpetro_mass"]))
    print(f"  DRPall cross-match: {n_with_mass}/{len(rows)} with stellar mass")

    return 0


if __name__ == "__main__":
    sys.exit(main())
