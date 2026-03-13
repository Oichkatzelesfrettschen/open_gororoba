"""LoTSS N(S) Flux Distribution DR1 -> DR2 -> DR3 -- E-200.

Computes differential source counts N(S) normalized by survey sky area and
fits a power-law dN/dS ~ S^{-gamma} to each release.

Physical motivation: the differential source count slope gamma encodes the
mixture of AGN and star-forming galaxy populations at 144 MHz.  Comparing
DR1 (325K, 2% sky), DR2 (4.4M, 13% sky), and DR3 (13.7M, 88% sky) tracks
how the faint-source population grows as sensitivity improves.

Outputs:
  - reports/lotss_flux_distribution_YYYY_MM_DD.toml (power-law fits per release)
  - reports/lotss_ns_comparison.png (if matplotlib available)

Usage:
    python flux_distribution.py \\
        --dr1 ../../data/external/radio_surveys/lotss_dr1.fits \\
        [--dr2 ../../data/external/radio_surveys/lotss_dr2.fits] \\
        [--dr3 ../../data/external/radio_surveys/lotss_dr3_manga_region.fits] \\
        [--output reports/lotss_flux_distribution_YYYY_MM_DD.toml]
"""

import argparse
import datetime
import os

import astropy.io.fits as fits
import numpy as np
import scipy.optimize


# Approximate survey solid angles in steradians.
# DR1: 424 sq deg, DR2: 5720 sq deg, DR3: ~36000 sq deg (88% of northern sky above Dec=0)
_ARCDEG_PER_SR = (180.0 / np.pi) ** 2
SURVEY_AREA_SR = {
    "DR1": 424.0 / _ARCDEG_PER_SR,
    "DR2": 5720.0 / _ARCDEG_PER_SR,
    "DR3": 36000.0 / _ARCDEG_PER_SR,
}


def load_flux(path: str) -> np.ndarray:
    """Return array of integrated 144 MHz flux densities in mJy from a LoTSS FITS."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        # Case-insensitive column lookup
        for n in data.names:
            if n.upper() == "TOTAL_FLUX":
                flux = np.asarray(data[n], dtype=float)
                # Remove non-finite and zero/negative values
                return flux[np.isfinite(flux) & (flux > 0)]
        raise KeyError(f"Column 'Total_flux' not found in {path}. Available: {data.names}")


def differential_source_count(flux_mjy: np.ndarray, omega_sr: float, n_bins: int = 40) -> tuple:
    """Compute the normalised differential source count dN/dS * S^{2.5}.

    The Euclidean normalisation S^{2.5} removes the dominant 1/S^{2.5} slope
    expected for a uniform, non-evolving population, making evolutionary
    deviations visible.

    Returns (S_mid, dN_dS_normalised, dS) where S_mid is bin centres in mJy.
    """
    log_s = np.log10(flux_mjy)
    log_bins = np.linspace(np.nanmin(log_s), np.nanmax(log_s), n_bins + 1)
    counts, edges = np.histogram(flux_mjy, bins=10 ** log_bins)
    s_mid = 0.5 * (10 ** edges[:-1] + 10 ** edges[1:])  # mJy
    delta_s = 10 ** edges[1:] - 10 ** edges[:-1]         # mJy

    # dN/dS in sr^{-1} mJy^{-1}
    dn_ds = counts / (omega_sr * delta_s)
    # Euclidean normalisation
    dn_ds_norm = dn_ds * s_mid ** 2.5

    # Only return bins with at least 5 sources for power-law fit
    valid = counts >= 5
    return s_mid[valid], dn_ds_norm[valid], delta_s[valid]


def fit_power_law(s_mid: np.ndarray, dn_ds_norm: np.ndarray) -> tuple:
    """Fit log(dN/dS_norm) = A + gamma * log(S) via scipy.optimize.curve_fit.

    Returns (gamma, gamma_err, A, A_err).  The slope gamma here is the
    departure from the Euclidean -2.5 slope; the effective N(S) slope is
    -(gamma + 2.5) for the un-normalised count.
    """
    def model(log_s, A, gamma):
        return A + gamma * log_s

    log_s = np.log10(s_mid)
    log_y = np.log10(np.abs(dn_ds_norm) + 1e-20)

    try:
        popt, pcov = scipy.optimize.curve_fit(model, log_s, log_y, p0=[10.0, -0.5])
        perr = np.sqrt(np.diag(pcov))
        return popt[1], perr[1], popt[0], perr[0]
    except (RuntimeError, ValueError):
        return float("nan"), float("nan"), float("nan"), float("nan")


def write_toml_report(path: str, results: list) -> None:
    lines = [
        "# LoTSS N(S) Flux Distribution Report",
        f"# Generated: {datetime.date.today().isoformat()}",
        "",
        "[experiment]",
        'id = "E-200"',
        'title = "LoTSS N(S) Flux Distribution DR1->DR2->DR3"',
        "",
    ]
    for r in results:
        release = r["release"]
        lines += [
            f"[{release.lower()}]",
            f'n_sources = {r["n"]}',
            f'area_sq_deg = {r["area_sq_deg"]:.1f}',
            f'flux_min_mjy = {r["flux_min"]:.4f}',
            f'flux_max_mjy = {r["flux_max"]:.4f}',
            f'flux_median_mjy = {r["flux_median"]:.4f}',
            f'power_law_gamma = {r["gamma"]:.4f}',
            f'power_law_gamma_err = {r["gamma_err"]:.4f}',
            f'effective_ns_slope = {r["ns_slope"]:.4f}',
            "",
        ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def plot_comparison(releases: list, path: str) -> None:
    """Save a comparison N(S) plot.  Silently skips if matplotlib unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"DR1": "#1f77b4", "DR2": "#ff7f0e", "DR3": "#2ca02c"}
    for r in releases:
        if r.get("s_mid") is None:
            continue
        s = r["s_mid"]
        y = r["dn_norm"]
        label = f"{r['release']} (N={r['n']:,}, gamma={r['gamma']:.2f})"
        ax.plot(s, y, ".", color=colors.get(r["release"], "k"), label=label, alpha=0.7, markersize=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("S (mJy)")
    ax.set_ylabel(r"dN/dS * S^{2.5}  (sr^{-1} mJy^{1.5})")
    ax.set_title("LoTSS 144 MHz Differential Source Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {path}")


def main() -> None:
    today = datetime.date.today().isoformat()
    parser = argparse.ArgumentParser(description="LoTSS N(S) flux distribution (E-200)")
    parser.add_argument("--dr1", default=None, help="LoTSS DR1 FITS file")
    parser.add_argument("--dr2", default=None, help="LoTSS DR2 FITS file")
    parser.add_argument("--dr3", default=None, help="LoTSS DR3 FITS file (cone search region)")
    parser.add_argument("--output",
                        default=f"../../reports/lotss_flux_distribution_{today}.toml",
                        help="Output TOML report")
    parser.add_argument("--plot",
                        default=f"../../reports/lotss_ns_comparison_{today}.png",
                        help="Output PNG comparison plot")
    args = parser.parse_args()

    catalog_paths = []
    if args.dr1:
        catalog_paths.append(("DR1", args.dr1))
    if args.dr2:
        catalog_paths.append(("DR2", args.dr2))
    if args.dr3:
        catalog_paths.append(("DR3", args.dr3))

    if not catalog_paths:
        print("ERROR: At least one of --dr1, --dr2, --dr3 must be provided.")
        return

    toml_rows = []
    plot_data = []

    for release, path in catalog_paths:
        if not os.path.isfile(path):
            print(f"WARNING: {release} file not found: {path} -- skipping")
            continue

        print(f"Loading {release}: {path}")
        flux = load_flux(path)
        n = len(flux)
        omega = SURVEY_AREA_SR[release]
        area_sq_deg = omega * _ARCDEG_PER_SR
        print(f"  N={n}  area~{area_sq_deg:.0f} sq deg")
        print(f"  Flux: min={np.min(flux):.4f}  median={np.median(flux):.4f}  max={np.max(flux):.4f} mJy")

        s_mid, dn_norm, _ = differential_source_count(flux, omega)
        gamma, gamma_err, _, _ = fit_power_law(s_mid, dn_norm)
        ns_slope = -(gamma + 2.5)
        print(f"  Power-law gamma (deviation from Euclidean): {gamma:.4f} +/- {gamma_err:.4f}")
        print(f"  Effective N(S) slope: {ns_slope:.4f}")

        row = {
            "release": release,
            "n": n,
            "area_sq_deg": area_sq_deg,
            "flux_min": float(np.min(flux)),
            "flux_max": float(np.max(flux)),
            "flux_median": float(np.median(flux)),
            "gamma": gamma,
            "gamma_err": gamma_err,
            "ns_slope": ns_slope,
        }
        toml_rows.append(row)
        plot_data.append({**row, "s_mid": s_mid, "dn_norm": dn_norm})

    if toml_rows:
        write_toml_report(args.output, toml_rows)
        print(f"\nReport written to: {args.output}")
        plot_comparison(plot_data, args.plot)
    else:
        print("No catalogs loaded -- nothing to report.")


if __name__ == "__main__":
    main()
