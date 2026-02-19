#!/usr/bin/env python3
"""Thomson multitaper spectral estimation with F-test for line components.

Multitaper method (Thomson 1982) provides optimal bias-variance tradeoff for
spectral estimation by averaging over K orthogonal DPSS (Slepian) tapers.
The F-test detects discrete line components at specific frequencies.

NW (time-bandwidth product) = 4 with K = 2*NW - 1 = 7 tapers is the standard
choice balancing resolution and variance reduction.

Usage:
    python multitaper.py --input data.csv --column value --nw 4
"""
import argparse
import csv
import sys

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats


ALIASED_GHOST_FREQ = 1.0 - 0.786_151_377_757_423
FREQ_TOL = 0.02


def load_csv_column(path: str, column: str) -> np.ndarray:
    """Load a single column from a CSV file."""
    values = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row[column])
                if np.isfinite(v):
                    values.append(v)
            except (ValueError, KeyError):
                continue
    return np.array(values)


def dpss_tapers(n: int, nw: float, k: int) -> np.ndarray:
    """Compute discrete prolate spheroidal sequences (Slepian tapers).

    Uses scipy.signal.windows.dpss which computes the DPSS via
    tridiagonal eigenproblem (efficient and numerically stable).
    """
    return scipy_signal.windows.dpss(n, nw, Kmax=k, return_ratios=False)


def multitaper_psd(
    signal: np.ndarray, nw: float = 4.0, k: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute adaptive multitaper PSD estimate.

    Returns (frequencies, psd, eigencoefficients).
    Frequencies are normalized to [0, 0.5] (Nyquist = 0.5).
    """
    n = len(signal)
    centered = signal - np.mean(signal)

    tapers = dpss_tapers(n, nw, k)

    # Compute eigencoefficients: FFT of each tapered signal
    nfft = n
    eigencoeffs = np.zeros((k, nfft // 2 + 1), dtype=complex)
    for i in range(k):
        tapered = centered * tapers[i]
        eigencoeffs[i] = np.fft.rfft(tapered, n=nfft)

    # Simple average (non-adaptive) PSD estimate
    psd = np.mean(np.abs(eigencoeffs) ** 2, axis=0) / n

    freqs = np.fft.rfftfreq(nfft)
    return freqs, psd, eigencoeffs


def f_test_line(eigencoeffs: np.ndarray, tapers: np.ndarray, n: int) -> np.ndarray:
    """Thomson F-test for line components at each frequency.

    Tests whether energy is concentrated in the zeroth-order eigencoefficient
    (which a line component would produce) versus spread across all tapers
    (which broadband noise would produce).

    Returns array of F-statistic values at each frequency bin.
    """
    k = eigencoeffs.shape[0]
    nfreq = eigencoeffs.shape[1]

    # Sum of tapers at DC (zeroth-order moment)
    taper_sums = np.sum(tapers, axis=1)  # shape (k,)

    f_stat = np.zeros(nfreq)
    for j in range(nfreq):
        # Estimate of line component amplitude
        numerator_complex = np.sum(taper_sums * eigencoeffs[:, j])
        denom_norm = np.sum(taper_sums**2)

        if denom_norm < 1e-30:
            f_stat[j] = 0.0
            continue

        mu_hat = numerator_complex / denom_norm

        # Residual power (broadband component)
        residual = eigencoeffs[:, j] - mu_hat * taper_sums
        residual_power = np.sum(np.abs(residual) ** 2)

        # Line power
        line_power = np.abs(mu_hat) ** 2 * denom_norm

        if residual_power < 1e-30:
            f_stat[j] = float("inf")
        else:
            # F-statistic with (2, 2*(k-1)) degrees of freedom
            # Factor of 2 for real and imaginary parts
            f_stat[j] = (k - 1) * line_power / residual_power

    return f_stat


def fit_red_white_noise(freqs: np.ndarray, psd: np.ndarray) -> dict:
    """Fit red noise (AR(1)) + white noise background model.

    Model: S(f) = sigma^2 / (1 - 2*rho*cos(2*pi*f) + rho^2) + white_floor

    Fits by matching lag-0 and lag-1 autocorrelation of the signal.
    Returns dict with rho, sigma2, white_floor, and background array.
    """
    # Estimate AR(1) parameter from PSD ratio at low frequencies
    # Use first 10% of frequency bins (excluding DC)
    n_low = max(2, len(freqs) // 10)
    low_psd = psd[1 : n_low + 1]
    low_freqs = freqs[1 : n_low + 1]

    # Log-linear fit to estimate spectral slope
    with np.errstate(divide="ignore", invalid="ignore"):
        log_psd = np.log(low_psd + 1e-30)
        log_freq = np.log(low_freqs + 1e-30)

    valid = np.isfinite(log_psd) & np.isfinite(log_freq)
    if valid.sum() < 2:
        # Cannot fit -- assume white noise
        white_floor = float(np.median(psd[1:]))
        return {
            "rho": 0.0,
            "sigma2": 0.0,
            "white_floor": white_floor,
            "background": np.full_like(psd, white_floor),
        }

    slope, intercept = np.polyfit(log_freq[valid], log_psd[valid], 1)

    # AR(1) rho from spectral slope: for AR(1), slope ~ -2 at high freq
    # rho ~ exp(-1/(f_knee * N)) where f_knee is the knee frequency
    rho = np.clip(1.0 - 10.0 ** (-slope / 4.0), 0.0, 0.99)

    # White noise floor from high-frequency end
    n_high = max(2, len(freqs) // 5)
    white_floor = float(np.median(psd[-n_high:]))

    # AR(1) variance
    sigma2 = max(0.0, float(np.median(psd[1:n_low])) - white_floor)

    # Background model
    cos_term = np.cos(2.0 * np.pi * freqs)
    denom = 1.0 - 2.0 * rho * cos_term + rho**2
    denom = np.maximum(denom, 1e-30)
    background = sigma2 / denom + white_floor

    return {
        "rho": float(rho),
        "sigma2": float(sigma2),
        "white_floor": float(white_floor),
        "background": background,
    }


def run_multitaper_test(
    signal: np.ndarray,
    target_freq: float = ALIASED_GHOST_FREQ,
    freq_tol: float = FREQ_TOL,
    nw: float = 4.0,
    k: int = 7,
) -> dict:
    """Run multitaper spectral analysis with F-test at target frequency."""
    n = len(signal)
    freqs, psd, eigencoeffs = multitaper_psd(signal, nw=nw, k=k)

    # Get tapers for F-test
    tapers = dpss_tapers(n, nw, k)

    # F-test for line components
    f_stat = f_test_line(eigencoeffs, tapers, n)

    # F-test p-value at target frequency
    mask = np.abs(freqs - target_freq) < freq_tol
    if not mask.any():
        return {
            "peak_freq": None,
            "peak_psd": None,
            "f_statistic": None,
            "f_pvalue": 1.0,
            "nw": nw,
            "k": k,
            "n_samples": n,
            "background_model": "none",
        }

    # Find peak near target
    target_psd = psd[mask]
    target_freqs = freqs[mask]
    peak_idx = np.argmax(target_psd)
    peak_freq = float(target_freqs[peak_idx])
    peak_psd_val = float(target_psd[peak_idx])

    # F-statistic at peak
    global_idx = np.where(mask)[0][peak_idx]
    f_val = float(f_stat[global_idx])

    # p-value from F-distribution with (2, 2*(k-1)) degrees of freedom
    f_pvalue = float(1.0 - scipy_stats.f.cdf(f_val, dfn=2, dfd=2 * (k - 1)))

    # Red+white noise background
    bg = fit_red_white_noise(freqs, psd)

    # SNR relative to background
    bg_at_peak = float(bg["background"][global_idx])
    snr_vs_bg = peak_psd_val / bg_at_peak if bg_at_peak > 0 else float("inf")

    return {
        "peak_freq": peak_freq,
        "peak_psd": peak_psd_val,
        "f_statistic": f_val,
        "f_pvalue": f_pvalue,
        "snr_vs_background": float(snr_vs_bg),
        "background_rho": bg["rho"],
        "background_white_floor": bg["white_floor"],
        "nw": nw,
        "k": k,
        "n_samples": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Thomson multitaper spectral analysis")
    parser.add_argument("--input", required=True, help="CSV file path")
    parser.add_argument("--column", default="value", help="Column name")
    parser.add_argument("--nw", type=float, default=4.0, help="Time-bandwidth product")
    parser.add_argument("--k", type=int, default=7, help="Number of tapers (default: 2*NW-1)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    values = load_csv_column(args.input, args.column)
    if len(values) == 0:
        print(f"ERROR: No valid data in column '{args.column}'", file=sys.stderr)
        sys.exit(1)

    result = run_multitaper_test(values, nw=args.nw, k=args.k)

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print(f"Multitaper Spectral Analysis (NW={args.nw}, K={args.k})")
        print(f"  Input: {args.input} [{args.column}]")
        print(f"  N = {result['n_samples']}")
        print(f"  Target: f={ALIASED_GHOST_FREQ:.6f}")
        print(f"  Peak near target: f={result['peak_freq']}, PSD={result['peak_psd']:.6e}")
        print(f"  F-statistic: {result['f_statistic']:.4f}")
        print(f"  F-test p-value: {result['f_pvalue']:.6e}")
        if "snr_vs_background" in result:
            print(f"  SNR vs background: {result['snr_vs_background']:.2f}")
            print(f"  Background AR(1) rho: {result['background_rho']:.4f}")
        verdict = "SIGNIFICANT" if result["f_pvalue"] < 0.05 else "NOT SIGNIFICANT"
        print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
