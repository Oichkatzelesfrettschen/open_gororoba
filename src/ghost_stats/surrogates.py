#!/usr/bin/env python3
"""IAAFT and phase-randomization surrogate testing.

IAAFT (Iteratively Adjusted Amplitude Adjusted Fourier Transform) preserves
both the amplitude distribution AND the power spectrum while randomizing phases.
This is the KEY test: if the peak at f=0.214 is NOT significant relative to
IAAFT surrogates, then it is fully explained by linear statistics.

Reference: Theiler et al. 1992, Physica D 58:77

Usage:
    python surrogates.py --input data.csv --column value --method iaaft --n 1000
"""
import argparse
import csv
import sys

import numpy as np


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


def phase_randomize(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomization surrogate: preserve power spectrum, randomize phases."""
    n = len(signal)
    ft = np.fft.rfft(signal)
    phases = rng.uniform(0, 2 * np.pi, size=len(ft))
    # DC and Nyquist components must be real
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    ft_surrogate = np.abs(ft) * np.exp(1j * phases)
    return np.fft.irfft(ft_surrogate, n=n)


def iaaft_surrogate(
    signal: np.ndarray, rng: np.random.Generator, max_iter: int = 100, tol: float = 1e-8
) -> np.ndarray:
    """IAAFT surrogate: preserve both amplitude distribution AND power spectrum.

    Algorithm (Schreiber & Schmitz 1996):
    1. Start with phase-randomized surrogate
    2. Iterate:
       a. Replace amplitudes with original amplitudes (rank-order matching)
       b. Impose original power spectrum via FFT amplitude replacement
    3. Converge when amplitude matching stops changing
    """
    n = len(signal)
    sorted_signal = np.sort(signal)
    original_ft = np.fft.rfft(signal)
    original_amplitudes = np.abs(original_ft)

    # Start with phase-randomized version
    surrogate = phase_randomize(signal, rng)

    for _ in range(max_iter):
        # Step A: rank-order matching (impose original distribution)
        rank = np.argsort(np.argsort(surrogate))
        amplitude_matched = sorted_signal[rank]

        # Step B: impose original power spectrum
        ft = np.fft.rfft(amplitude_matched)
        phases = np.angle(ft)
        ft_corrected = original_amplitudes * np.exp(1j * phases)
        surrogate_new = np.fft.irfft(ft_corrected, n=n)

        # Check convergence
        if np.max(np.abs(surrogate_new - surrogate)) < tol:
            break
        surrogate = surrogate_new

    # Final rank-order matching to ensure exact distribution match
    rank = np.argsort(np.argsort(surrogate))
    return sorted_signal[rank]


def compute_power_at_freq(signal: np.ndarray, target_freq: float) -> float:
    """Compute normalized power at target frequency."""
    n = len(signal)
    # Remove mean, apply Hann window
    centered = signal - np.mean(signal)
    window = np.hanning(n)
    windowed = centered * window

    ft = np.fft.rfft(windowed)
    power = np.abs(ft) ** 2 / n**2
    freqs = np.fft.rfftfreq(n)

    # Find closest bin to target
    idx = np.argmin(np.abs(freqs - target_freq))
    return float(power[idx])


def run_surrogate_test(
    signal: np.ndarray,
    method: str = "iaaft",
    n_surrogates: int = 1000,
    target_freq: float = ALIASED_GHOST_FREQ,
    seed: int = 42,
) -> dict:
    """Run surrogate data test at target frequency."""
    rng = np.random.default_rng(seed)
    observed_power = compute_power_at_freq(signal, target_freq)

    surrogate_powers = []
    for i in range(n_surrogates):
        if method == "iaaft":
            surr = iaaft_surrogate(signal, rng)
        else:
            surr = phase_randomize(signal, rng)
        surr_power = compute_power_at_freq(surr, target_freq)
        surrogate_powers.append(surr_power)

    surrogate_powers = np.array(surrogate_powers)
    count_ge = np.sum(surrogate_powers >= observed_power)
    p_empirical = (count_ge + 1) / (n_surrogates + 1)

    return {
        "method": method,
        "n_surrogates": n_surrogates,
        "target_freq": target_freq,
        "observed_power": float(observed_power),
        "null_mean": float(surrogate_powers.mean()),
        "null_std": float(surrogate_powers.std()),
        "null_median": float(np.median(surrogate_powers)),
        "null_95th": float(np.percentile(surrogate_powers, 95)),
        "null_99th": float(np.percentile(surrogate_powers, 99)),
        "p_empirical": float(p_empirical),
        "n_samples": len(signal),
    }


def main():
    parser = argparse.ArgumentParser(description="Surrogate data testing")
    parser.add_argument("--input", required=True, help="CSV file path")
    parser.add_argument("--column", default="value", help="Column name")
    parser.add_argument(
        "--method",
        default="iaaft",
        choices=["iaaft", "phase"],
        help="Surrogate method",
    )
    parser.add_argument("--n", type=int, default=1000, help="Number of surrogates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    values = load_csv_column(args.input, args.column)
    if len(values) == 0:
        print(f"ERROR: No valid data in column '{args.column}'", file=sys.stderr)
        sys.exit(1)

    result = run_surrogate_test(
        values,
        method=args.method,
        n_surrogates=args.n,
        seed=args.seed,
    )

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print(f"Surrogate Test ({args.method.upper()}, N={args.n})")
        print(f"  Input: {args.input} [{args.column}]")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Target freq: {result['target_freq']:.6f}")
        print(f"  Observed power: {result['observed_power']:.6e}")
        print(
            f"  Null: mean={result['null_mean']:.6e} +/- {result['null_std']:.6e}"
        )
        print(
            f"  Null: 95th={result['null_95th']:.6e}, 99th={result['null_99th']:.6e}"
        )
        print(f"  Empirical p-value: {result['p_empirical']:.6e}")
        verdict = "SIGNIFICANT" if result["p_empirical"] < 0.05 else "NOT SIGNIFICANT"
        print(f"  Verdict: {verdict}")
        if result["p_empirical"] >= 0.05:
            print(
                "  -> Peak is CONSISTENT with surrogates. Linear statistics explain it."
            )
        else:
            print(
                "  -> Peak EXCEEDS surrogate null. Non-linear structure may be present."
            )


if __name__ == "__main__":
    main()
