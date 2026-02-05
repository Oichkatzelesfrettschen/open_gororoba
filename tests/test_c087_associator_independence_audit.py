"""
Tests for C-087: Associator norm independence.

Validates:
  1. Quaternions (dim=4) are associative: E[||A||^2] = 0.
  2. Sedenions (dim=16): correlation < 0.15 (cross-term decaying).
  3. dim=64: E[||A||^2] within 5% of 2.0.
  4. Correlation monotonically decreases from dim=8 onward.
"""
from __future__ import annotations

from scripts.analysis.c087_associator_independence_audit import (
    compute_associator_stats,
    run_sweep,
)


def test_quaternions_associative() -> None:
    """Quaternions (dim=4) should have zero associator."""
    stats = compute_associator_stats(dim=4, n_trials=500, seed=42)
    assert stats["mean_assoc_sq"] < 1e-20, (
        f"Quaternion associator {stats['mean_assoc_sq']:.2e} not zero"
    )


def test_sedenion_cross_term_decaying() -> None:
    """At dim=16, cross-term correlation should be < 0.15."""
    stats = compute_associator_stats(dim=16, n_trials=1000, seed=42)
    assert abs(stats["correlation_coeff"]) < 0.15, (
        f"Sedenion corr={stats['correlation_coeff']:.4f}, expected < 0.15"
    )


def test_high_dim_assoc_norm_near_two() -> None:
    """At dim=64, E[||A||^2] should be within 5% of 2.0."""
    stats = compute_associator_stats(dim=64, n_trials=1000, seed=42)
    rel_err = abs(stats["mean_assoc_sq"] - 2.0) / 2.0
    assert rel_err < 0.05, (
        f"E[||A||^2]={stats['mean_assoc_sq']:.4f}, {rel_err:.2%} from 2.0"
    )


def test_correlation_decay_monotone() -> None:
    """Absolute correlation should decrease from dim=8 onward."""
    results = run_sweep(dims=[8, 16, 32, 64], n_trials=500, seed=42)
    corrs = [abs(r["correlation_coeff"]) for r in results]
    for i in range(len(corrs) - 1):
        assert corrs[i] > corrs[i + 1], (
            f"|corr| at dim={results[i]['dim']} ({corrs[i]:.4f}) "
            f"<= dim={results[i+1]['dim']} ({corrs[i+1]:.4f})"
        )
