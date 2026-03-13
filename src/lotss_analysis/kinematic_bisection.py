"""LoTSS-MaNGA Kinematic Bisection -- E-198 primary analysis.

Splits the MaNGA rotation-curve sample into Radio-Loud (LoTSS-detected) and
Radio-Quiet (undetected) subsets, then compares NFW residual RMS in the inner
core (x = 0.5 to 1.25 r/r_s) between the two populations.

The inner-core window 0.5 to 1.25 r/r_s is motivated by E-183/E-184: that is
the populated IFU region where the baryonic red-noise continuum dominates and
where a ZD or AGN feedback signal would first appear.

Statistical tests:
  - Mann-Whitney U (non-parametric, no normality assumption)
  - Pearson r between 144 MHz flux density and inner-core RMS
  - Bootstrap 95% CI on the RMS ratio Radio-Loud / Radio-Quiet

Usage:
    python kinematic_bisection.py \\
        --xmatch ../../data/external/manga/manga_lotss_xmatch.csv \\
        --rotcurves ../../data/external/manga/manga_rotcurves_all.csv \\
        [--output reports/lotss_manga_bisection_YYYY_MM_DD.toml]
        [--inner-lo 0.5] [--inner-hi 1.25] [--n-bootstrap 1000]
"""

import argparse
import csv
import datetime
import sys
from collections import defaultdict

import numpy as np
import scipy.stats


INNER_LO_DEFAULT = 0.5
INNER_HI_DEFAULT = 1.25
N_BOOTSTRAP_DEFAULT = 1000


def load_xmatch(path: str) -> dict:
    """Return dict plateifu -> {'detected': bool, 'flux_mjy': float or nan}."""
    result = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = row["plateifu"].strip()
            detected = row["lotss_detected"].strip() == "1"
            flux = float(row["lotss_flux_mjy"]) if row["lotss_flux_mjy"].strip() else float("nan")
            result[pid] = {"detected": detected, "flux_mjy": flux}
    return result


def load_inner_rms(rotcurve_path: str, inner_lo: float, inner_hi: float) -> dict:
    """Return dict plateifu -> inner-core residual RMS.

    Reads manga_rotcurves_all.csv and computes, for each galaxy, the RMS of
    the delta_v column (NFW rotation curve residual) restricted to radial bins
    with inner_lo <= x <= inner_hi (dimensionless radius x = r/r_s).
    Galaxies with fewer than 3 points in the window are excluded.
    """
    raw = defaultdict(list)
    with open(rotcurve_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = row.get("plateifu", "").strip()
            if not pid:
                continue
            try:
                x = float(row["x"])
                delta_v = float(row["delta_v"])
            except (KeyError, ValueError):
                continue
            if inner_lo <= x <= inner_hi:
                raw[pid].append(delta_v)

    rms_map = {}
    for pid, vals in raw.items():
        if len(vals) >= 3:
            arr = np.array(vals, dtype=float)
            rms_map[pid] = float(np.sqrt(np.mean(arr ** 2)))
    return rms_map


def bootstrap_rms_ratio(loud_rms: np.ndarray, quiet_rms: np.ndarray, n: int) -> tuple:
    """Bootstrap 95% CI on median(loud_rms) / median(quiet_rms).

    Returns (ratio_observed, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed=42)
    obs_ratio = np.median(loud_rms) / np.median(quiet_rms) if np.median(quiet_rms) > 0 else float("nan")

    boot_ratios = []
    for _ in range(n):
        bl = rng.choice(loud_rms, size=len(loud_rms), replace=True)
        bq = rng.choice(quiet_rms, size=len(quiet_rms), replace=True)
        m_q = np.median(bq)
        boot_ratios.append(np.median(bl) / m_q if m_q > 0 else float("nan"))

    boot_arr = np.array([v for v in boot_ratios if np.isfinite(v)])
    ci_lo = float(np.percentile(boot_arr, 2.5))
    ci_hi = float(np.percentile(boot_arr, 97.5))
    return obs_ratio, ci_lo, ci_hi


def write_toml_report(path: str, results: dict) -> None:
    lines = [
        "# LoTSS-MaNGA Kinematic Bisection Report",
        f"# Generated: {datetime.date.today().isoformat()}",
        "",
        "[experiment]",
        'id = "E-198"',
        'title = "LoTSS-MaNGA Kinematic Bisection"',
        "",
        "[sample]",
        f'n_manga_total = {results["n_total"]}',
        f'n_radio_loud = {results["n_loud"]}',
        f'n_radio_quiet = {results["n_quiet"]}',
        f'detection_fraction = {results["detect_frac"]:.4f}',
        f'inner_lo = {results["inner_lo"]}',
        f'inner_hi = {results["inner_hi"]}',
        "",
        "[inner_core_rms]",
        f'rms_radio_loud_median = {results["rms_loud_median"]:.6f}',
        f'rms_radio_quiet_median = {results["rms_quiet_median"]:.6f}',
        f'rms_ratio_loud_over_quiet = {results["rms_ratio"]:.4f}',
        f'rms_ratio_ci_lo = {results["rms_ratio_ci_lo"]:.4f}',
        f'rms_ratio_ci_hi = {results["rms_ratio_ci_hi"]:.4f}',
        f'n_bootstrap = {results["n_bootstrap"]}',
        "",
        "[mann_whitney_u]",
        f'statistic = {results["mw_stat"]:.4f}',
        f'p_value = {results["mw_p"]:.6f}',
        f'effect_direction = "{results["effect_dir"]}"',
        "",
        "[pearson_flux_rms]",
        f'r = {results["pearson_r"]:.4f}',
        f'p_value = {results["pearson_p"]:.6f}',
        f'n_pairs = {results["pearson_n"]}',
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    today = datetime.date.today().isoformat()
    parser = argparse.ArgumentParser(description="LoTSS-MaNGA kinematic bisection (E-198)")
    parser.add_argument("--xmatch", required=True,
                        help="Cross-match CSV from crossmatch_manga.py")
    parser.add_argument("--rotcurves", required=True,
                        help="MaNGA rotation curve master CSV")
    parser.add_argument("--output",
                        default=f"../../reports/lotss_manga_bisection_{today}.toml",
                        help="Output TOML report path")
    parser.add_argument("--inner-lo", type=float, default=INNER_LO_DEFAULT,
                        help="Inner radial window lower bound in r/r_s")
    parser.add_argument("--inner-hi", type=float, default=INNER_HI_DEFAULT,
                        help="Inner radial window upper bound in r/r_s")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT,
                        help="Bootstrap iterations for CI on RMS ratio")
    args = parser.parse_args()

    print(f"Loading cross-match: {args.xmatch}")
    xmatch = load_xmatch(args.xmatch)

    print(f"Loading rotation curves: {args.rotcurves}")
    print(f"  Inner window: {args.inner_lo} <= x <= {args.inner_hi} r/r_s")
    rms_map = load_inner_rms(args.rotcurves, args.inner_lo, args.inner_hi)

    # Build Radio-Loud and Radio-Quiet RMS arrays for galaxies present in both.
    loud_rms, quiet_rms, loud_flux = [], [], []
    for pid, info in xmatch.items():
        if pid not in rms_map:
            continue
        rms = rms_map[pid]
        if info["detected"]:
            loud_rms.append(rms)
            loud_flux.append(info["flux_mjy"])
        else:
            quiet_rms.append(rms)

    loud_rms = np.array(loud_rms, dtype=float)
    quiet_rms = np.array(quiet_rms, dtype=float)
    loud_flux = np.array(loud_flux, dtype=float)

    n_total = len(xmatch)
    n_loud = len(loud_rms)
    n_quiet = len(quiet_rms)
    detect_frac = n_loud / n_total if n_total > 0 else 0.0

    print(f"Radio-Loud (detected):   {n_loud}")
    print(f"Radio-Quiet (undetected): {n_quiet}")
    print(f"Detection fraction:       {100*detect_frac:.1f}%")

    if n_loud < 2 or n_quiet < 2:
        print("ERROR: Insufficient matched galaxies for statistical tests.")
        sys.exit(1)

    # Mann-Whitney U test (non-parametric, no normality assumption)
    mw_stat, mw_p = scipy.stats.mannwhitneyu(loud_rms, quiet_rms, alternative="two-sided")
    effect_dir = "loud > quiet" if np.median(loud_rms) > np.median(quiet_rms) else "loud <= quiet"

    print(f"\nMann-Whitney U: stat={mw_stat:.1f}, p={mw_p:.4e}")
    print(f"  Median RMS Radio-Loud:  {np.median(loud_rms):.6f}")
    print(f"  Median RMS Radio-Quiet: {np.median(quiet_rms):.6f}")
    print(f"  Direction: {effect_dir}")

    # Bootstrap CI on ratio
    rms_ratio, ratio_ci_lo, ratio_ci_hi = bootstrap_rms_ratio(loud_rms, quiet_rms, args.n_bootstrap)
    print(f"  RMS ratio (loud/quiet): {rms_ratio:.4f}  95% CI [{ratio_ci_lo:.4f}, {ratio_ci_hi:.4f}]")

    # Pearson r between 144 MHz flux and inner-core RMS
    valid = np.isfinite(loud_flux)
    if np.sum(valid) >= 3:
        r_val, p_val = scipy.stats.pearsonr(loud_flux[valid], loud_rms[valid])
        n_pairs = int(np.sum(valid))
        print(f"\nPearson r (flux vs RMS): r={r_val:.4f}, p={p_val:.4e}, N={n_pairs}")
    else:
        r_val, p_val, n_pairs = float("nan"), float("nan"), 0
        print("\nPearson r: insufficient valid flux measurements")

    results = {
        "n_total": n_total,
        "n_loud": n_loud,
        "n_quiet": n_quiet,
        "detect_frac": detect_frac,
        "inner_lo": args.inner_lo,
        "inner_hi": args.inner_hi,
        "rms_loud_median": float(np.median(loud_rms)),
        "rms_quiet_median": float(np.median(quiet_rms)),
        "rms_ratio": rms_ratio,
        "rms_ratio_ci_lo": ratio_ci_lo,
        "rms_ratio_ci_hi": ratio_ci_hi,
        "n_bootstrap": args.n_bootstrap,
        "mw_stat": float(mw_stat),
        "mw_p": float(mw_p),
        "effect_dir": effect_dir,
        "pearson_r": r_val,
        "pearson_p": p_val,
        "pearson_n": n_pairs,
    }

    write_toml_report(args.output, results)
    print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
