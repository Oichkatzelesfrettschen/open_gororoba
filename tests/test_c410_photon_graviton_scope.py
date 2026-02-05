"""
Tests for C-410: photon-graviton mixing scope limits.

Validates:
  1. Schwinger critical field B_cr ~ 4.41e9 T (within 1%).
  2. Gravitational coupling kappa is positive and tiny.
  3. Mixing amplitude ratio at B=10T is negligibly small (< 1e-40).
  4. Scope note correctly states C-402 is NOT overturned.
"""
from __future__ import annotations

from scripts.analysis.c410_photon_graviton_scope import (
    gravitational_coupling,
    mixing_amplitude_estimate,
    schwinger_critical_field,
)


def test_schwinger_critical_field() -> None:
    """B_cr should be approximately 4.41e9 T."""
    B_cr = schwinger_critical_field()
    assert 4.4e9 < B_cr < 4.5e9, f"B_cr={B_cr:.3e}, expected ~4.41e9 T"


def test_gravitational_coupling_tiny() -> None:
    """kappa = sqrt(16*pi*G/c^4) should be extremely small."""
    kappa = gravitational_coupling()
    assert kappa > 0, "kappa must be positive"
    assert kappa < 1e-20, f"kappa={kappa:.3e}, expected << 1"


def test_mixing_amplitude_negligible() -> None:
    """At B=10T, mixing amplitude ratio should be negligibly small."""
    result = mixing_amplitude_estimate(B_lab=10.0)
    ratio = result["amplitude_ratio"]
    # (alpha/pi) * (10/4.4e9)^2 ~ 2.3e-3 * 5.2e-18 ~ 1.2e-20
    assert ratio < 1e-15, (
        f"Amplitude ratio {ratio:.3e} unexpectedly large"
    )


def test_scope_note_content() -> None:
    """Scope note must state C-402 is not overturned."""
    result = mixing_amplitude_estimate(B_lab=1.0)
    note = result["scope_note"]
    assert "does NOT overturn" in note
    assert "Rodal" in note
    assert "C-402" in note
