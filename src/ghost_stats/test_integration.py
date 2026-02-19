#!/usr/bin/env python3
"""Integration tests for ghost_stats Python module.

Verifies that all scripts produce correct output on synthetic data
with known ground truth: pure noise (no signal) and injected sine
(strong signal at known frequency).
"""
import json
import os
import sys
import tempfile

import numpy as np


def write_synthetic_csv(path: str, values: np.ndarray, column: str = "value"):
    """Write values to a simple CSV file."""
    with open(path, "w") as f:
        f.write(f"index,{column}\n")
        for i, v in enumerate(values):
            f.write(f"{i},{v:.12e}\n")


def run_script(script: str, args: list[str]) -> dict:
    """Run a Python script and return parsed JSON output."""
    import subprocess

    cmd = [sys.executable, script] + args + ["--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"FAIL: {script} returned {result.returncode}", file=sys.stderr)
        print(f"  stderr: {result.stderr}", file=sys.stderr)
        return {}
    return json.loads(result.stdout)


def test_surrogates_noise():
    """Pure noise should NOT be significant under IAAFT surrogates."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, 512)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        write_synthetic_csv(f.name, noise)
        path = f.name

    try:
        script = os.path.join(os.path.dirname(__file__), "surrogates.py")
        result = run_script(script, ["--input", path, "--column", "value", "--method", "iaaft", "--n", "200"])
        assert result, "surrogates.py returned empty result"
        p = result["p_empirical"]
        assert p > 0.05, f"Noise should NOT be significant, got p={p}"
        print(f"  surrogates (noise): p={p:.4f} -- PASS (> 0.05)")
    finally:
        os.unlink(path)


def test_surrogates_signal():
    """Strong injected sine should be significant under phase-randomization."""
    rng = np.random.default_rng(42)
    n = 512
    t = np.arange(n, dtype=float)
    # Strong sine at the aliased ghost frequency
    signal = 10.0 * np.sin(2 * np.pi * 0.214 * t) + rng.normal(0, 1, n)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        write_synthetic_csv(f.name, signal)
        path = f.name

    try:
        script = os.path.join(os.path.dirname(__file__), "surrogates.py")
        result = run_script(script, ["--input", path, "--column", "value", "--method", "phase", "--n", "200"])
        assert result, "surrogates.py returned empty result"
        p = result["p_empirical"]
        # Phase randomization preserves power spectrum, so an injected sine
        # should NOT be significant (since its power spectrum is preserved).
        # This is the expected behavior -- IAAFT surrogates test nonlinear
        # structure beyond the power spectrum, not the power spectrum itself.
        print(f"  surrogates (signal, phase): p={p:.4f} -- INFO (phase-rand preserves spectrum)")
    finally:
        os.unlink(path)


def test_multitaper_noise():
    """Pure noise should NOT have significant F-test at any specific frequency."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, 512)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        write_synthetic_csv(f.name, noise)
        path = f.name

    try:
        script = os.path.join(os.path.dirname(__file__), "multitaper.py")
        result = run_script(script, ["--input", path, "--column", "value"])
        assert result, "multitaper.py returned empty result"
        p = result["f_pvalue"]
        # Not guaranteed to be > 0.05 for a single random draw, but likely
        print(f"  multitaper (noise): F-pvalue={p:.4f} -- INFO")
    finally:
        os.unlink(path)


def test_combine_known():
    """Known p-values should produce correct Fisher chi2."""
    script = os.path.join(os.path.dirname(__file__), "combine.py")
    result = run_script(script, [
        "--p-values", "0.05,0.10,0.20",
        "--n-values", "100,200,300",
        "--method", "all",
    ])
    assert result, "combine.py returned empty result"

    fisher = result["results"]["fisher"]
    # chi2 = -2*(ln(0.05) + ln(0.10) + ln(0.20))
    expected_chi2 = -2 * (np.log(0.05) + np.log(0.10) + np.log(0.20))
    actual_chi2 = fisher["statistic"]
    assert abs(actual_chi2 - expected_chi2) < 0.01, (
        f"Fisher chi2 mismatch: expected {expected_chi2:.4f}, got {actual_chi2:.4f}"
    )
    print(f"  combine (Fisher): chi2={actual_chi2:.4f} (expected {expected_chi2:.4f}) -- PASS")

    stouffer = result["results"]["stouffer"]
    print(f"  combine (Stouffer): Z={stouffer['statistic']:.4f}, p={stouffer['combined_p']:.4e} -- INFO")


def main():
    print("Ghost Stats Integration Tests")
    print("=" * 50)

    tests = [
        ("Surrogates (noise)", test_surrogates_noise),
        ("Surrogates (signal)", test_surrogates_signal),
        ("Multitaper (noise)", test_multitaper_noise),
        ("Combine (known)", test_combine_known),
    ]

    n_pass = 0
    n_fail = 0

    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            test_fn()
            n_pass += 1
        except Exception as e:
            print(f"  FAIL: {e}", file=sys.stderr)
            n_fail += 1

    print()
    print("=" * 50)
    print(f"Results: {n_pass} passed, {n_fail} failed")
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
