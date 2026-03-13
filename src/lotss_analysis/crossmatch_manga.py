"""LoTSS-MaNGA spatial cross-match.

E-198: LoTSS-MaNGA Kinematic Bisection -- first stage.

Loads a LoTSS source catalog (FITS) and the MaNGA rotation curve master CSV,
builds a k-d tree on sky coordinates, and performs a 3 arcsec nearest-neighbour
match.  Outputs manga_lotss_xmatch.csv with one row per MaNGA galaxy annotating
whether a LoTSS counterpart was found within the match radius.

Usage:
    python crossmatch_manga.py \\
        --lotss  ../../data/external/radio_surveys/lotss_dr3_manga_region.fits \\
        --manga  ../../data/external/manga/manga_rotcurves_all.csv \\
        --output ../../data/external/manga/manga_lotss_xmatch.csv \\
        [--radius-arcsec 3.0]

The LoTSS FITS file is expected to contain at least:
    Source_Name, RA (deg), DEC (deg), Total_flux (mJy), Spectral_index, S_Code

The MaNGA CSV is expected to contain at least:
    plateifu, objra, objdec  (IFU target coordinates in degrees, J2000)
"""

import argparse
import sys

import astropy.io.fits as fits
import numpy as np
import scipy.spatial


DEFAULT_MATCH_RADIUS_ARCSEC = 3.0


def load_lotss_fits(path: str) -> dict:
    """Return arrays of RA, Dec, flux, spectral_index, s_code from a LoTSS FITS file."""
    with fits.open(path, memmap=True) as hdul:
        # LoTSS srl.fits: data is in the first extension (HDU 1)
        data = hdul[1].data
        names_upper = [c.upper() for c in data.names]

        def col(key: str):
            # Case-insensitive column lookup
            for n in data.names:
                if n.upper() == key.upper():
                    return data[n]
            raise KeyError(f"Column '{key}' not found in {path}. Available: {data.names}")

        return {
            "source_name": np.array(col("Source_Name"), dtype=str),
            "ra":          np.asarray(col("RA"), dtype=float),
            "dec":         np.asarray(col("DEC"), dtype=float),
            "flux_mjy":    np.asarray(col("Total_flux"), dtype=float),
            "spectral_index": np.where(
                np.isfinite(np.asarray(col("Spectral_index"), dtype=float)),
                np.asarray(col("Spectral_index"), dtype=float),
                np.nan,
            ),
            "s_code": np.array(col("S_Code"), dtype=str),
        }


def load_manga_ra_dec(path: str):
    """Return (plateifu, ra, dec) arrays from the MaNGA rotation curve CSV.

    Handles both the full manga_rotcurves_all.csv (multiple rows per galaxy,
    deduplicated on plateifu) and a pre-deduplicated catalog.
    """
    import csv

    plateifus, ras, decs = [], [], []
    seen = set()

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = row.get("plateifu", "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            try:
                ra = float(row["objra"])
                dec = float(row["objdec"])
            except (KeyError, ValueError):
                continue
            plateifus.append(pid)
            ras.append(ra)
            decs.append(dec)

    return np.array(plateifus), np.array(ras, dtype=float), np.array(decs, dtype=float)


def build_lotss_kdtree(ra: np.ndarray, dec: np.ndarray) -> scipy.spatial.KDTree:
    """Build a 2D k-d tree using (RA*cos(Dec), Dec) in degrees.

    The cosine correction projects the RA axis onto the local sky plane,
    making Euclidean distance approximate great-circle distance for small
    separations (< ~1 degree).  At Dec=70 the uncorrected RA axis is
    compressed by cos(70)~0.34, so without this correction a 3 arcsec
    match radius in RA corresponds to ~8.8 arcsec on the sky.
    """
    dec_rad = np.deg2rad(dec)
    x = ra * np.cos(dec_rad)
    y = dec.copy()
    coords = np.column_stack([x, y])
    return scipy.spatial.KDTree(coords)


def crossmatch(
    manga_ra: np.ndarray,
    manga_dec: np.ndarray,
    lotss: dict,
    radius_arcsec: float,
) -> tuple:
    """Match each MaNGA galaxy to the nearest LoTSS source within radius_arcsec.

    Returns (matched_idx, matched_dist_arcsec) where matched_idx[i] is the
    index into lotss arrays for galaxy i, or -1 if no match within radius.
    matched_dist_arcsec[i] is NaN for unmatched galaxies.
    """
    radius_deg = radius_arcsec / 3600.0
    tree = build_lotss_kdtree(lotss["ra"], lotss["dec"])

    dec_rad = np.deg2rad(manga_dec)
    qx = manga_ra * np.cos(dec_rad)
    qy = manga_dec.copy()
    query_pts = np.column_stack([qx, qy])

    dists, idxs = tree.query(query_pts, k=1, distance_upper_bound=radius_deg)

    n = len(manga_ra)
    matched_idx = np.where(idxs < len(lotss["ra"]), idxs, -1).astype(int)
    matched_dist = np.where(matched_idx >= 0, dists * 3600.0, np.nan)

    return matched_idx, matched_dist


def write_xmatch_csv(
    path: str,
    plateifus: np.ndarray,
    manga_ra: np.ndarray,
    manga_dec: np.ndarray,
    matched_idx: np.ndarray,
    matched_dist: np.ndarray,
    lotss: dict,
) -> None:
    import csv

    fields = [
        "plateifu", "objra", "objdec",
        "lotss_detected", "lotss_separation_arcsec",
        "lotss_source_name", "lotss_flux_mjy",
        "lotss_spectral_index", "lotss_s_code",
    ]

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, pid in enumerate(plateifus):
            idx = matched_idx[i]
            detected = idx >= 0
            row = {
                "plateifu": pid,
                "objra": f"{manga_ra[i]:.6f}",
                "objdec": f"{manga_dec[i]:.6f}",
                "lotss_detected": "1" if detected else "0",
                "lotss_separation_arcsec": f"{matched_dist[i]:.3f}" if detected else "",
                "lotss_source_name": lotss["source_name"][idx] if detected else "",
                "lotss_flux_mjy": f"{lotss['flux_mjy'][idx]:.4f}" if detected else "",
                "lotss_spectral_index": (
                    f"{lotss['spectral_index'][idx]:.3f}"
                    if detected and np.isfinite(lotss["spectral_index"][idx])
                    else ""
                ),
                "lotss_s_code": lotss["s_code"][idx] if detected else "",
            }
            writer.writerow(row)


def print_statistics(
    plateifus: np.ndarray,
    matched_idx: np.ndarray,
    lotss: dict,
) -> None:
    n_total = len(plateifus)
    n_detected = int(np.sum(matched_idx >= 0))
    n_quiet = n_total - n_detected
    frac = n_detected / n_total if n_total > 0 else 0.0

    print(f"MaNGA galaxies:      {n_total}")
    print(f"LoTSS matches:       {n_detected}  ({100*frac:.1f}%)")
    print(f"Undetected (quiet):  {n_quiet}  ({100*(1-frac):.1f}%)")

    if n_detected > 0:
        det_flux = lotss["flux_mjy"][matched_idx[matched_idx >= 0]]
        print(f"Flux min/median/max: "
              f"{np.nanmin(det_flux):.3f} / {np.nanmedian(det_flux):.3f} / {np.nanmax(det_flux):.3f} mJy")
        si_vals = lotss["spectral_index"][matched_idx[matched_idx >= 0]]
        si_finite = si_vals[np.isfinite(si_vals)]
        if len(si_finite):
            print(f"Spectral index mean +/- std: "
                  f"{np.mean(si_finite):.3f} +/- {np.std(si_finite):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoTSS-MaNGA cross-match (E-198)")
    parser.add_argument("--lotss", required=True, help="LoTSS source catalog FITS file")
    parser.add_argument("--manga", required=True, help="MaNGA rotation curve CSV")
    parser.add_argument("--output", required=True, help="Output cross-match CSV")
    parser.add_argument(
        "--radius-arcsec",
        type=float,
        default=DEFAULT_MATCH_RADIUS_ARCSEC,
        help=f"Match radius in arcsec (default {DEFAULT_MATCH_RADIUS_ARCSEC})",
    )
    args = parser.parse_args()

    print(f"Loading LoTSS catalog: {args.lotss}")
    lotss = load_lotss_fits(args.lotss)
    print(f"  {len(lotss['ra'])} LoTSS sources loaded")

    print(f"Loading MaNGA catalog: {args.manga}")
    plateifus, manga_ra, manga_dec = load_manga_ra_dec(args.manga)
    print(f"  {len(plateifus)} unique MaNGA galaxies")

    print(f"Cross-matching with radius = {args.radius_arcsec} arcsec ...")
    matched_idx, matched_dist = crossmatch(manga_ra, manga_dec, lotss, args.radius_arcsec)

    print_statistics(plateifus, matched_idx, lotss)

    write_xmatch_csv(args.output, plateifus, manga_ra, manga_dec, matched_idx, matched_dist, lotss)
    print(f"Cross-match written to: {args.output}")


if __name__ == "__main__":
    main()
