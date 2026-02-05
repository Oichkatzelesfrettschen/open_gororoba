"""
Tests for gravastar TOV sweep CSV outputs.

Validates that the sweep script produced well-formed CSVs with
physically meaningful results (sub-horizon compactness, at least
some equilibrium solutions).
"""

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "data" / "csv"

SWEEP_PATH = CSV_DIR / "gravastar_ligo_mass_sweep.csv"
STABILITY_PATH = CSV_DIR / "gravastar_radial_stability.csv"


@pytest.fixture(scope="module")
def sweep_df():
    """Load the mass-sweep CSV; skip if file is missing."""
    if not SWEEP_PATH.exists():
        pytest.skip(f"Sweep CSV not found: {SWEEP_PATH}")
    return pd.read_csv(SWEEP_PATH)


@pytest.fixture(scope="module")
def stability_df():
    """Load the radial-stability CSV; skip if file is missing."""
    if not STABILITY_PATH.exists():
        pytest.skip(f"Stability CSV not found: {STABILITY_PATH}")
    return pd.read_csv(STABILITY_PATH)


class TestSweepCSV:
    """Validate gravastar_ligo_mass_sweep.csv."""

    def test_file_exists(self):
        assert SWEEP_PATH.exists(), f"Missing: {SWEEP_PATH}"

    def test_has_required_columns(self, sweep_df):
        expected = {
            "M_target",
            "core_compactness",
            "R1",
            "R2",
            "M_total",
            "equilibrium_satisfied",
            "compactness_2M_R2",
        }
        assert expected.issubset(set(sweep_df.columns)), (
            f"Missing columns: {expected - set(sweep_df.columns)}"
        )

    def test_nonempty(self, sweep_df):
        assert len(sweep_df) > 0, "Sweep CSV is empty"

    def test_compactness_below_one(self, sweep_df):
        """All solutions must have compactness 2M/R2 < 1.0 (sub-horizon)."""
        violations = sweep_df[sweep_df["compactness_2M_R2"] >= 1.0]
        assert len(violations) == 0, (
            f"{len(violations)} solutions violate compactness < 1.0:\n"
            f"{violations.to_string()}"
        )

    def test_at_least_some_equilibrium(self, sweep_df):
        """At least some solutions must satisfy hydrostatic equilibrium."""
        n_eq = sweep_df["equilibrium_satisfied"].sum()
        assert n_eq > 0, "No solutions satisfy equilibrium"

    def test_positive_radii(self, sweep_df):
        """R1 and R2 must be positive, and R2 > R1."""
        assert (sweep_df["R1"] > 0).all(), "Non-positive R1 found"
        assert (sweep_df["R2"] > 0).all(), "Non-positive R2 found"
        assert (sweep_df["R2"] > sweep_df["R1"]).all(), "R2 <= R1 found"

    def test_positive_mass(self, sweep_df):
        """M_total must be positive."""
        assert (sweep_df["M_total"] > 0).all(), "Non-positive M_total found"


class TestStabilityCSV:
    """Validate gravastar_radial_stability.csv."""

    def test_file_exists(self):
        assert STABILITY_PATH.exists(), f"Missing: {STABILITY_PATH}"

    def test_has_required_columns(self, stability_df):
        expected = {
            "M_target",
            "core_compactness",
            "R1",
            "R2",
            "M_total",
            "rho_v",
            "dM_drho_c",
            "harrison_wheeler_stable",
        }
        assert expected.issubset(set(stability_df.columns)), (
            f"Missing columns: {expected - set(stability_df.columns)}"
        )

    def test_nonempty(self, stability_df):
        assert len(stability_df) > 0, "Stability CSV is empty"

    def test_stability_column_is_boolean(self, stability_df):
        """harrison_wheeler_stable should parse as boolean-like values."""
        valid = stability_df["harrison_wheeler_stable"].isin([True, False])
        assert valid.all(), "Non-boolean values in harrison_wheeler_stable"

    def test_perturbation_masses_present(self, stability_df):
        """Check that perturbation mass columns exist and have finite values."""
        for col in ["M_minus_1pct", "M_center", "M_plus_1pct"]:
            assert col in stability_df.columns, f"Missing column: {col}"
        # At least some rows should have finite perturbation results
        finite_count = stability_df["M_minus_1pct"].notna().sum()
        assert finite_count > 0, "All M_minus_1pct values are NaN"
