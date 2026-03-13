#!/usr/bin/env python3
"""Cross-match Euclid Q1 morphology with MaNGA DR17 DAPall by sky position.

Identifies the subset of MaNGA galaxies also observed by Euclid Q1,
attaches GalaxyZoo morphology fractions from the Euclid morphology
catalogue, and writes a selection CSV suitable for use with the
harmonic-halo-stacking-manga binary via --euclid-morphology-csv.

Survey overlap note:
  Euclid Q1 covers 3 deep fields (EDF-North ~RA 270 Dec 66, EDF-Fornax
  ~RA 54 Dec -34, EDF-South ~RA 60 Dec -48). MaNGA DR17 follows the
  SDSS footprint (mostly Dec > -30). Expected overlap: ~10-20 galaxies
  in EDF-North. This script is designed for Euclid DR2+ when coverage
  will substantially overlap the SDSS/MaNGA footprint.

Morphology selection criteria applied:
  featured_fraction  > featured_min   (GZ: smooth-or-featured_featured-or-disk)
  spiral_fraction    > spiral_min     (GZ: has-spiral-arms_yes)
  face_on_fraction   > face_on_min    (GZ: disk-edge-on_no)
  non_merging_frac   > non_merge_min  (GZ: merging_none)

Inputs:
  data/external/euclid/zenodo/15106473/morphology_catalogue.parquet
  data/external/manga/dapall-v3_1_1-3.1.0.fits

Output:
  data/external/euclid/euclid_manga_crossmatch.csv

Reference: Euclid Collaboration, Q1 release (2025), Zenodo 15106473.
           Westfall et al., AJ 158 (2019) 231 (MaNGA DAP).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MORPH_PARQUET = Path(
    "data/external/euclid/zenodo/15106473/morphology_catalogue.parquet"
)
DAPALL_FITS = Path("data/external/manga/dapall-v3_1_1-3.1.0.fits")
OUTPUT_CSV = Path("data/external/euclid/euclid_manga_crossmatch.csv")

# Matching radius in arcseconds
MATCH_RADIUS_ARCSEC = 2.0

# Default morphology thresholds for late-type spiral selection
DEFAULT_FEATURED_MIN = 0.5    # P(featured/disk) > 0.5
DEFAULT_SPIRAL_MIN = 0.5      # P(has spiral arms) > 0.5
DEFAULT_FACE_ON_MIN = 0.5     # P(not edge-on) > 0.5
DEFAULT_NON_MERGE_MIN = 0.7   # P(non-merging) > 0.7

# Euclid morphology columns relevant to rotation-curve galaxy selection
MORPH_COLS = [
    "smooth-or-featured_smooth_fraction",
    "smooth-or-featured_featured-or-disk_fraction",
    "disk-edge-on_yes_fraction",
    "disk-edge-on_no_fraction",
    "has-spiral-arms_yes_fraction",
    "has-spiral-arms_no_fraction",
    "bar_strong_fraction",
    "bar_weak_fraction",
    "bar_no_fraction",
    "merging_none_fraction",
    "merging_major-disturbance_fraction",
    "merging_merger_fraction",
    "clumps_yes_fraction",
]


def load_euclid_morphology() -> pd.DataFrame:
    """Load Euclid Q1 morphology catalogue (ra, dec, GZ fractions)."""
    if not MORPH_PARQUET.exists():
        print(f"  ERROR: Euclid morphology not found: {MORPH_PARQUET}")
        print("  Run: python3 bin/fetch_euclid_zenodo.py --catalog morphology")
        sys.exit(1)

    morph = pd.read_parquet(
        MORPH_PARQUET,
        columns=["object_id", "ra", "dec"] + MORPH_COLS,
        engine="pyarrow",
    )
    print(f"  Euclid morphology: {len(morph)} objects")
    return morph


def load_manga_dapall() -> pd.DataFrame:
    """Load MaNGA DAPall FITS, return key columns with coordinates."""
    if not DAPALL_FITS.exists():
        print(f"  ERROR: MaNGA DAPall not found: {DAPALL_FITS}")
        print("  Run: python3 bin/fetch_manga_kinematics.py")
        sys.exit(1)

    try:
        from astropy.io import fits
    except ImportError:
        print("  ERROR: astropy not installed (pip install astropy)")
        sys.exit(1)

    with fits.open(DAPALL_FITS) as hdul:
        # Use HYB10-MILESHC-MASTARSSP extension (best quality, default)
        ext_names = [h.name for h in hdul]
        ext = "HYB10-MILESHC-MASTARSSP"
        if ext not in ext_names:
            ext = ext_names[1]  # fallback to first data extension
        data = hdul[ext].data
        df = pd.DataFrame(
            {
                "plateifu": data["PLATEIFU"].astype(str),
                "mangaid": data["MANGAID"].astype(str),
                "objra": data["OBJRA"].astype(float),
                "objdec": data["OBJDEC"].astype(float),
                "z": data["NSA_Z"].astype(float),
                "nsa_th50_r": data["NSA_ELPETRO_TH50_R"].astype(float),
            }
        )
    print(f"  MaNGA DAPall: {len(df)} galaxies")
    return df


def crossmatch_sky(
    manga: pd.DataFrame, euclid: pd.DataFrame, radius_arcsec: float
) -> pd.DataFrame:
    """Match MaNGA galaxies to Euclid counterparts by sky position.

    Uses a simple great-circle approximation valid for separations < 1 deg.
    Returns a DataFrame with MaNGA rows joined with their nearest Euclid
    match within radius_arcsec, plus separation column sep_arcsec.
    """
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        print("  ERROR: astropy not installed (pip install astropy)")
        sys.exit(1)

    manga_coords = SkyCoord(
        ra=manga["objra"].values * u.deg,
        dec=manga["objdec"].values * u.deg,
    )
    euclid_coords = SkyCoord(
        ra=euclid["ra"].values * u.deg,
        dec=euclid["dec"].values * u.deg,
    )

    idx, sep, _ = manga_coords.match_to_catalog_sky(euclid_coords)
    sep_arcsec = sep.arcsec

    mask = sep_arcsec < radius_arcsec
    n_matches = mask.sum()
    print(
        f"  Matches within {radius_arcsec}\" radius: {n_matches} / {len(manga)} MaNGA galaxies"
    )

    manga_matched = manga[mask].copy().reset_index(drop=True)
    manga_matched["sep_arcsec"] = sep_arcsec[mask]
    euclid_matched = euclid.iloc[idx[mask]].reset_index(drop=True)

    # Prefix Euclid columns to avoid collisions
    euclid_matched = euclid_matched.rename(
        columns={c: f"euclid_{c}" for c in euclid_matched.columns if c != "object_id"}
    )
    euclid_matched = euclid_matched.rename(columns={"object_id": "euclid_object_id"})

    result = pd.concat([manga_matched, euclid_matched], axis=1)
    return result


def apply_morphology_cuts(
    matched: pd.DataFrame,
    featured_min: float,
    spiral_min: float,
    face_on_min: float,
    non_merge_min: float,
) -> pd.DataFrame:
    """Apply GalaxyZoo morphology cuts for late-type spiral selection."""
    feat_col = "euclid_smooth-or-featured_featured-or-disk_fraction"
    spiral_col = "euclid_has-spiral-arms_yes_fraction"
    face_on_col = "euclid_disk-edge-on_no_fraction"
    no_merge_col = "euclid_merging_none_fraction"

    mask = (
        (matched[feat_col] > featured_min)
        & (matched[spiral_col] > spiral_min)
        & (matched[face_on_col] > face_on_min)
        & (matched[no_merge_col] > non_merge_min)
    )
    selected = matched[mask].copy()
    print(
        f"  After morphology cuts: {len(selected)} galaxies"
        f" (featured>{featured_min}, spiral>{spiral_min},"
        f" face-on>{face_on_min}, non-merge>{non_merge_min})"
    )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-match Euclid Q1 morphology with MaNGA DR17 DAPall"
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=MATCH_RADIUS_ARCSEC,
        metavar="ARCSEC",
        help=f"Sky matching radius in arcsec (default: {MATCH_RADIUS_ARCSEC})",
    )
    parser.add_argument(
        "--featured-min",
        type=float,
        default=DEFAULT_FEATURED_MIN,
        help=f"Min featured/disk fraction (default: {DEFAULT_FEATURED_MIN})",
    )
    parser.add_argument(
        "--spiral-min",
        type=float,
        default=DEFAULT_SPIRAL_MIN,
        help=f"Min spiral arms fraction (default: {DEFAULT_SPIRAL_MIN})",
    )
    parser.add_argument(
        "--face-on-min",
        type=float,
        default=DEFAULT_FACE_ON_MIN,
        help=f"Min face-on (not edge-on) fraction (default: {DEFAULT_FACE_ON_MIN})",
    )
    parser.add_argument(
        "--non-merge-min",
        type=float,
        default=DEFAULT_NON_MERGE_MIN,
        help=f"Min non-merging fraction (default: {DEFAULT_NON_MERGE_MIN})",
    )
    parser.add_argument(
        "--no-morphology-cuts",
        action="store_true",
        help="Output all matches without applying morphology cuts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    args = parser.parse_args()

    print("=== Euclid Q1 x MaNGA DR17 Cross-match ===")
    print()

    print("[1/4] Loading Euclid morphology catalogue...")
    euclid = load_euclid_morphology()

    print("[2/4] Loading MaNGA DAPall...")
    manga = load_manga_dapall()

    print("[3/4] Cross-matching by sky position...")
    matched = crossmatch_sky(manga, euclid, args.match_radius)

    if matched.empty:
        print()
        print("  NOTE: No matches found. Euclid Q1 covers 3 small deep fields")
        print("  with minimal overlap with the MaNGA/SDSS footprint.")
        print("  Expected behavior with current data -- see task #27 notes.")
        print("  Rerun with Euclid DR2+ for broader sky coverage.")
        return 0

    if not args.no_morphology_cuts:
        print("[4/4] Applying morphology cuts...")
        selected = apply_morphology_cuts(
            matched,
            args.featured_min,
            args.spiral_min,
            args.face_on_min,
            args.non_merge_min,
        )
    else:
        print("[4/4] Skipping morphology cuts (--no-morphology-cuts)...")
        selected = matched

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    print(f"  Wrote {len(selected)} rows -> {args.output}")
    print()
    print("=== Cross-match complete ===")
    print(f"  Pass --euclid-morphology-csv {args.output}")
    print("  to harmonic-halo-stacking-manga for morphologically-selected stacking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
