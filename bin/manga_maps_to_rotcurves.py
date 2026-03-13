#!/usr/bin/env python3
"""Extract pseudo-slit rotation curves from MaNGA MAPS files.

Two-stage MaNGA pipeline for harmonic halo stacking:
  Stage 1 (manga_dapall_to_csv.py): DAPall Guillotine -- morphological + quality cuts
  Stage 2 (THIS SCRIPT): MAPS Extraction -- download + pseudo-slit rotation curves

Given a DAPall selection CSV (from Stage 1), this script:
  1. Applies disk galaxy cuts (Sersic n < 2.5, 30 < i < 70 deg, H-alpha S/N)
  2. Downloads the per-galaxy MAPS FITS files (~20 MB each) in parallel
  3. Extracts a 1D pseudo-slit rotation curve along the kinematic major axis
     from the 2D EMLINE_GVEL (H-alpha Gaussian velocity) map
  4. Outputs SPARC-compatible CSV rotation curves for Rust ingestion

The pseudo-slit approach mirrors SPARC's tilted-ring long-slit methodology:
  - Take a slice along the kinematic PA through the velocity field center
  - Average over a slit width of ~2 spaxels (3" for MaNGA)
  - Deproject using known inclination: V_circ = V_los / sin(i)
  - Bin radially in arcsec, convert to kpc using angular diameter distance

Parallelism:
  Downloads are I/O bound (urllib releases the GIL). --workers (default 8)
  runs that many download+extract tasks concurrently via ThreadPoolExecutor.
  Resume: on restart, galaxies already in master CSV are skipped.

Output format (per galaxy): name, r_kpc, v_obs_km_s, v_err_km_s
This is directly parseable by the Rust MaNGA catalog parser.

Usage:
  python3 bin/manga_maps_to_rotcurves.py --selection data/external/manga/dapall_selection.csv \
      --output-dir data/external/manga/rotcurves/ --n-max 3000 --workers 8

Reference: Westfall et al., AJ 158 (2019) 231 (DAP).
           Law et al., AJ 152 (2016) 83 (DRP).
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import numpy as np
    from astropy.cosmology import FlatLambdaCDM
    from astropy.io import fits
except ImportError:
    print("ERROR: requires astropy and numpy", file=sys.stderr)
    print("  pip install astropy numpy", file=sys.stderr)
    sys.exit(1)

USER_AGENT = "open_gororoba/0.1 (research)"
TIMEOUT = 120

# MaNGA DR17 MAPS file URL template
# DAPTYPE for DR17: SPX-MILESHC-MASTARSSP (spaxel-level, no binning)
MAPS_URL_TEMPLATE = (
    "https://data.sdss.org/sas/dr17/manga/spectro/analysis/"
    "v3_1_1/3.1.0/SPX-MILESHC-MASTARSSP/{plate}/{ifudesign}/"
    "manga-{plate}-{ifudesign}-MAPS-SPX-MILESHC-MASTARSSP.fits.gz"
)

# H-alpha channel index in EMLINE arrays (0-indexed: 23 in DR17)
HA_CHANNEL = 23

# MaNGA spaxel size in arcsec
SPAXEL_SIZE_ARCSEC = 0.5

# Slit half-width in spaxels for pseudo-slit extraction
SLIT_HALF_WIDTH = 2  # 2 spaxels = 1 arcsec half-width

# Cosmology for angular diameter distance (Planck 2018)
COSMO = FlatLambdaCDM(H0=67.36, Om0=0.3153)


_da_cache: dict[float, float] = {}


def angular_diameter_distance_kpc(z: float) -> float:
    """Angular diameter distance in kpc at redshift z (memoized per-galaxy z)."""
    if z not in _da_cache:
        _da_cache[z] = COSMO.angular_diameter_distance(z).to("kpc").value
    return _da_cache[z]


def arcsec_to_kpc(arcsec: float, d_a_kpc: float) -> float:
    """Convert angular size in arcsec to physical kpc using precomputed d_A."""
    return arcsec * d_a_kpc * np.pi / (180.0 * 3600.0)


def download_maps(plate: int, ifudesign: str, cache_dir: Path) -> Path | None:
    """Download a MaNGA MAPS file if not cached. Returns local path or None."""
    url = MAPS_URL_TEMPLATE.format(plate=plate, ifudesign=ifudesign)
    fname = f"manga-{plate}-{ifudesign}-MAPS-SPX-MILESHC-MASTARSSP.fits.gz"
    dest = cache_dir / fname

    if dest.exists():
        return dest

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
            if b"<!DOCTYPE" in data[:200] or b"<html" in data[:200].lower():
                return None
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return dest
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def extract_pseudo_slit(maps_path: Path, pa_deg: float, incl_deg: float,
                        z: float) -> list[dict] | None:
    """Extract a 1D pseudo-slit rotation curve from a MAPS file.

    Vectorized: builds all slit pixel indices as numpy arrays before any
    masking or averaging, so the inner loop is eliminated.

    Args:
        maps_path: Path to the MAPS FITS file
        pa_deg: Kinematic position angle in degrees (E of N)
        incl_deg: Inclination in degrees
        z: Redshift for angular-to-physical conversion

    Returns:
        List of dicts with keys: r_kpc, v_obs_km_s, v_err_km_s
        or None if extraction fails.
    """
    sin_i = np.sin(np.radians(incl_deg))
    if sin_i < 0.1:
        return None  # Too face-on, deprojection blows up

    try:
        with fits.open(maps_path, memmap=False) as hdul:
            # H-alpha velocity map (copy to avoid keeping file open)
            vel_map = np.array(hdul["EMLINE_GVEL"].data[HA_CHANNEL], dtype=np.float32)
            vel_ivar = np.array(hdul["EMLINE_GVEL_IVAR"].data[HA_CHANNEL], dtype=np.float32)
            vel_mask = np.array(hdul["EMLINE_GVEL_MASK"].data[HA_CHANNEL], dtype=np.int32)
    except (KeyError, IndexError, OSError) as e:
        print(f"    MAPS read error: {e}", file=sys.stderr)
        return None

    ny, nx = vel_map.shape
    cx, cy = nx // 2, ny // 2  # Center of IFU

    pa_rad = np.radians(pa_deg)
    cos_pa = float(np.cos(pa_rad))
    sin_pa = float(np.sin(pa_rad))

    # Maximum radius in spaxels (stays within IFU boundary)
    max_r_spaxels = min(cx, cy) - 1
    r_arr = np.arange(1, max_r_spaxels + 1)  # shape (R,)

    # Slit offset indices: w in [-W..W], shape (2W+1,)
    w_arr = np.arange(-SLIT_HALF_WIDTH, SLIT_HALF_WIDTH + 1)  # shape (W,)

    # Precompute d_A once per galaxy
    d_a_kpc = angular_diameter_distance_kpc(z)
    # arcsec_per_spaxel to kpc scale factor (constant for this galaxy)
    spaxel_kpc = arcsec_to_kpc(SPAXEL_SIZE_ARCSEC, d_a_kpc)

    # For each side (+1 approaching, -1 receding) build pixel coords as arrays
    # ix[r,w] = cx + sign * r * sin_pa + w * (-cos_pa)
    # iy[r,w] = cy + sign * r * cos_pa + w * sin_pa
    points = []
    for sign in (1.0, -1.0):
        # (R, W) integer pixel indices
        ix = np.rint(cx + sign * r_arr[:, None] * sin_pa
                     + w_arr[None, :] * (-cos_pa)).astype(np.int32)
        iy = np.rint(cy + sign * r_arr[:, None] * cos_pa
                     + w_arr[None, :] * sin_pa).astype(np.int32)

        # Boundary mask
        in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

        # Clamp out-of-bounds indices for safe gather (values will be masked out)
        ix_safe = np.clip(ix, 0, nx - 1)
        iy_safe = np.clip(iy, 0, ny - 1)

        # Vectorized gather
        v_vals = vel_map[iy_safe, ix_safe].astype(np.float64)
        iv_vals = vel_ivar[iy_safe, ix_safe].astype(np.float64)
        m_vals = vel_mask[iy_safe, ix_safe]

        # Combined good-pixel mask: in_bounds + mask==0 + ivar>0
        good = in_bounds & (m_vals == 0) & (iv_vals > 0.0)

        # Weighted sum per radius: sum over w axis
        n_good = good.sum(axis=1)  # (R,)
        v_sum = np.where(good, v_vals, 0.0).sum(axis=1)  # (R,)
        var_sum = np.where(good, 1.0 / np.where(iv_vals > 0, iv_vals, 1.0), 0.0).sum(axis=1)

        valid = n_good >= 1
        if not valid.any():
            continue

        r_valid = r_arr[valid]
        v_los = v_sum[valid] / n_good[valid]
        v_err_los = np.sqrt(var_sum[valid]) / n_good[valid]
        v_circ = np.abs(v_los) / sin_i
        v_err = v_err_los / sin_i
        r_kpc_arr = r_valid * spaxel_kpc

        for r_kpc, v_c, v_e in zip(
            r_kpc_arr.tolist(), v_circ.tolist(), v_err.tolist(), strict=True
        ):
            points.append({"r_kpc": r_kpc, "v_obs_km_s": v_c, "v_err_km_s": v_e})

    if len(points) < 5:
        return None

    points.sort(key=lambda p: p["r_kpc"])
    binned = _bin_rotation_curve(points)
    return binned if len(binned) >= 5 else None


def _bin_rotation_curve(points: list[dict], n_bins: int = 30) -> list[dict]:
    """Bin rotation curve points into radial bins, averaging A+R sides."""
    if not points:
        return []

    r_min = min(p["r_kpc"] for p in points)
    r_max = max(p["r_kpc"] for p in points)
    if r_max <= r_min:
        return points

    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    binned = []

    for i in range(n_bins):
        in_bin = [p for p in points
                  if bin_edges[i] <= p["r_kpc"] < bin_edges[i + 1]]
        if not in_bin:
            continue

        r_mean = np.mean([p["r_kpc"] for p in in_bin])
        # Inverse-variance weighted mean
        weights = [1.0 / (p["v_err_km_s"] ** 2) if p["v_err_km_s"] > 0 else 0.0
                   for p in in_bin]
        w_sum = sum(weights)
        if w_sum <= 0:
            continue

        v_mean = sum(
            w * p["v_obs_km_s"] for w, p in zip(weights, in_bin, strict=True)
        ) / w_sum
        v_err = 1.0 / np.sqrt(w_sum)

        binned.append({
            "r_kpc": float(r_mean),
            "v_obs_km_s": float(v_mean),
            "v_err_km_s": float(v_err),
        })

    return binned


def apply_selection_cuts(row: dict) -> bool:
    """Apply Stage 2 disk galaxy cuts for rotation curve extraction.

    Criteria:
      - Sersic index n < 2.5 (disk-dominated, reject ellipticals)
      - 30 < inclination < 70 deg (avoid face-on and edge-on)
      - H-alpha equivalent width > 2 A (emission line detected)
      - Stellar mass log(M*) > 8.5 (reject dwarf irregulars)
    """
    try:
        sersic_n = float(row.get("nsa_sersic_n", "nan"))
        ba = float(row.get("nsa_elpetro_ba", "nan"))
        ha_gew = float(row.get("ha_gew_1re", "nan"))
        log_mass = float(row.get("nsa_elpetro_mass", "nan"))
    except (ValueError, TypeError):
        return False

    # Sersic cut: disk-dominated
    if not (0.1 < sersic_n < 2.5):
        return False

    # Inclination from axis ratio: cos(i) = b/a
    if not (0.0 < ba < 1.0):
        return False
    incl_deg = np.degrees(np.arccos(ba))
    if not (30.0 < incl_deg < 70.0):
        return False

    # H-alpha emission detected
    if not (ha_gew > 2.0):
        return False

    # Minimum stellar mass
    if not (log_mass > 8.5):
        return False

    return True


def _load_completed_plateifus(master_csv: Path) -> set:
    """Read plateifu values already written to master CSV (for resume)."""
    completed = set()
    if not master_csv.exists():
        return completed
    try:
        with open(master_csv) as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    completed.add(row[0].strip())
    except OSError:
        pass
    return completed


def _process_galaxy(gal: dict, maps_cache: Path) -> tuple[str, list[dict]] | None:
    """Download and extract one galaxy. Returns (plateifu, points) or None.

    Designed to run in a thread -- urllib releases the GIL during I/O.
    """
    plateifu = gal["plateifu"]
    plate = int(gal["plate"])
    ifudesign = gal["ifudesign"]
    z = float(gal["z"])
    ba = float(gal.get("nsa_elpetro_ba", "0.5"))
    pa = float(gal.get("nsa_elpetro_phi", "0"))
    incl_deg = np.degrees(np.arccos(max(0.0, min(1.0, ba))))

    maps_path = download_maps(plate, ifudesign, maps_cache)
    if maps_path is None:
        return None  # download failed

    points = extract_pseudo_slit(maps_path, pa, incl_deg, z)
    if points is None:
        return None  # extraction failed

    return (plateifu, points)


def main():
    parser = argparse.ArgumentParser(
        description="Extract MaNGA pseudo-slit rotation curves from MAPS files"
    )
    parser.add_argument(
        "--selection",
        default="data/external/manga/dapall_selection.csv",
        help="DAPall selection CSV from Stage 1",
    )
    parser.add_argument(
        "--maps-cache",
        default="data/external/manga/maps_cache",
        help="Directory to cache downloaded MAPS files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/external/manga/rotcurves",
        help="Output directory for rotation curve CSVs",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=3000,
        help="Maximum number of galaxies to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download+extract workers (default 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only apply cuts and report counts, do not download",
    )
    args = parser.parse_args()

    selection_path = Path(args.selection)
    if not selection_path.exists():
        print(f"ERROR: {selection_path} not found", file=sys.stderr)
        print("  Run: python3 bin/manga_dapall_to_csv.py first", file=sys.stderr)
        return 1

    # Read selection catalog
    print(f"Reading selection catalog from {selection_path}...")
    with open(selection_path) as f:
        reader = csv.DictReader(f)
        all_galaxies = list(reader)
    print(f"  {len(all_galaxies)} galaxies in catalog")

    # Apply Stage 2 cuts
    print("Applying disk galaxy cuts (Sersic < 2.5, 30 < i < 70, H-alpha, mass)...")
    selected = [g for g in all_galaxies if apply_selection_cuts(g)]
    print(f"  {len(selected)} galaxies pass cuts ({len(selected)/len(all_galaxies)*100:.1f}%)")

    if len(selected) > args.n_max:
        print(f"  Capping at {args.n_max} (use --n-max to change)")
        selected = selected[:args.n_max]

    if args.dry_run:
        print("\n=== DRY RUN: would download MAPS for these galaxies ===")
        est_gb = len(selected) * 20 / 1024
        print(f"  Estimated download: ~{est_gb:.0f} GB ({len(selected)} x ~20 MB)")
        return 0

    # Create output directories
    maps_cache = Path(args.maps_cache)
    output_dir = Path(args.output_dir)
    maps_cache.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_csv = output_dir / "manga_rotcurves_all.csv"

    # Resume: skip galaxies already in master CSV
    completed_before = _load_completed_plateifus(master_csv)
    if completed_before:
        print(f"  Resuming: {len(completed_before)} galaxies already in {master_csv}")
    to_process = [g for g in selected if g["plateifu"] not in completed_before]
    print(f"  Processing {len(to_process)} galaxies with {args.workers} workers...")

    # Thread-safe counters and CSV writer
    counters = {"success": 0, "download_fail": 0, "extract_fail": 0}
    counter_lock = threading.Lock()
    csv_lock = threading.Lock()

    # Open master CSV in append mode (so resume works)
    csv_mode = "a" if completed_before else "w"
    with open(master_csv, csv_mode, newline="") as master_f:
        master_writer = csv.writer(master_f)
        if csv_mode == "w":
            master_writer.writerow(["name", "r_kpc", "v_obs_km_s", "v_err_km_s"])

        total = len(to_process)
        done = len(completed_before)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_galaxy, gal, maps_cache): gal["plateifu"]
                for gal in to_process
            }

            for future in as_completed(futures):
                plateifu = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = None
                    print(f"  ERROR {plateifu}: {exc}", file=sys.stderr)

                with counter_lock:
                    done += 1
                    n_done = done

                if result is None:
                    # Distinguish download vs extract fail by checking cache
                    maps_cache_check = maps_cache / (
                        f"manga-{plateifu.replace('-', '-')}"
                        f"-MAPS-SPX-MILESHC-MASTARSSP.fits.gz"
                    )
                    if maps_cache_check.exists():
                        with counter_lock:
                            counters["extract_fail"] += 1
                    else:
                        with counter_lock:
                            counters["download_fail"] += 1
                else:
                    p_ifu, points = result
                    with csv_lock:
                        for pt in points:
                            master_writer.writerow([
                                p_ifu,
                                f"{pt['r_kpc']:.4f}",
                                f"{pt['v_obs_km_s']:.2f}",
                                f"{pt['v_err_km_s']:.2f}",
                            ])
                        master_f.flush()
                    with counter_lock:
                        counters["success"] += 1

                # Progress every 50 completions
                if n_done % 50 == 0 or n_done == total + len(completed_before):
                    with counter_lock:
                        s = counters["success"] + len(completed_before)
                        df = counters["download_fail"]
                        ef = counters["extract_fail"]
                    pct = 100.0 * n_done / (total + len(completed_before))
                    print(f"  [{n_done}/{total + len(completed_before)}] {pct:.1f}%"
                          f"  ok={s} dl_fail={df} ex_fail={ef}")

    print("\n=== MaNGA Rotation Curve Extraction Complete ===")
    n_total_success = counters["success"] + len(completed_before)
    print(f"  Total in CSV: {n_total_success}")
    print(f"  This run:  success={counters['success']}, "
          f"dl_fail={counters['download_fail']}, ex_fail={counters['extract_fail']}")
    print(f"  Master CSV: {master_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
