#!/usr/bin/env python3
"""Lomb-Scargle periodogram with Baluev (2008) analytic FAP.

Uses astropy.timeseries.LombScargle for production-grade implementation.
Baluev FAP is the gold standard for irregularly-sampled data.

Usage:
    python lomb_scargle.py --input data.csv --column value --fap-method baluev
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


def run_lomb_scargle(
    values: np.ndarray,
    target_freq: float = ALIASED_GHOST_FREQ,
    freq_tol: float = FREQ_TOL,
    fap_method: str = "baluev",
    n_bootstrap: int = 10000,
) -> dict:
    """Run Lomb-Scargle periodogram on sorted catalog data.

    For sorted catalog data, the "time" axis is the index (0, 1, 2, ...N-1)
    and the "value" axis is the sorted catalog value. This tests whether the
    quantile function has periodic structure.
    """
    from astropy.timeseries import LombScargle

    n = len(values)
    t = np.arange(n, dtype=float)

    # Generalized Lomb-Scargle with floating mean
    ls = LombScargle(t, values, fit_mean=True, center_data=True)

    # Frequency grid: 0 to Nyquist (0.5) with oversampling factor 5
    freq_min = 1.0 / n
    freq_max = 0.5
    frequency, power = ls.autopower(
        minimum_frequency=freq_min,
        maximum_frequency=freq_max,
        samples_per_peak=5,
    )

    # Find peak near target frequency
    mask = np.abs(frequency - target_freq) < freq_tol
    if not mask.any():
        return {
            "peak_freq": None,
            "peak_power": None,
            "fap_at_target": 1.0,
            "fap_method": fap_method,
            "max_power": float(power.max()),
            "max_freq": float(frequency[np.argmax(power)]),
            "n_samples": n,
        }

    target_power = float(power[mask].max())
    target_freq_actual = float(frequency[mask][np.argmax(power[mask])])

    # False alarm probability
    if fap_method == "baluev":
        fap = float(ls.false_alarm_probability(target_power, method="baluev"))
    elif fap_method == "bootstrap":
        fap = float(
            ls.false_alarm_probability(
                target_power, method="bootstrap", n_bootstraps=n_bootstrap
            )
        )
    else:
        fap = float(ls.false_alarm_probability(target_power, method=fap_method))

    return {
        "peak_freq": target_freq_actual,
        "peak_power": target_power,
        "fap_at_target": fap,
        "fap_method": fap_method,
        "max_power": float(power.max()),
        "max_freq": float(frequency[np.argmax(power)]),
        "n_samples": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Lomb-Scargle with Baluev FAP")
    parser.add_argument("--input", required=True, help="CSV file path")
    parser.add_argument("--column", default="value", help="Column name")
    parser.add_argument(
        "--fap-method",
        default="baluev",
        choices=["baluev", "bootstrap", "naive"],
        help="FAP computation method",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000, help="Bootstrap iterations"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    values = load_csv_column(args.input, args.column)
    if len(values) == 0:
        print(f"ERROR: No valid data in column '{args.column}'", file=sys.stderr)
        sys.exit(1)

    result = run_lomb_scargle(
        values,
        fap_method=args.fap_method,
        n_bootstrap=args.n_bootstrap,
    )

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print(f"Lomb-Scargle Periodogram ({args.fap_method} FAP)")
        print(f"  Input: {args.input} [{args.column}]")
        print(f"  N = {result['n_samples']}")
        print(f"  Target: f={ALIASED_GHOST_FREQ:.6f}")
        print(f"  Peak near target: f={result['peak_freq']}, power={result['peak_power']}")
        print(f"  FAP at target: {result['fap_at_target']:.6e}")
        print(f"  Global max: f={result['max_freq']:.6f}, power={result['max_power']:.6e}")
        verdict = "SIGNIFICANT" if result["fap_at_target"] < 0.05 else "NOT SIGNIFICANT"
        print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
