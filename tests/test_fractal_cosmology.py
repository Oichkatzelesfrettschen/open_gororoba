"""
Tests for fractal cosmology and spectral dimension analysis.

Validates G2.1 (Calcagni fractal cosmology) and G2.2 (spectral exponent comparison).
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_physics.fractal_cosmology import (
    calcagni_spectral_dimension,
    calcagni_spectral_density,
    k_minus_3_spectrum,
    kolmogorov_spectrum,
    kraichnan_enstrophy_spectrum,
    parisi_sourlas_effective_dimension,
    parisi_sourlas_spectrum_exponent,
    cdt_spectral_dimension,
    compare_spectra,
    compute_rms_deviation,
    analyze_k_minus_3_origin,
    run_fractal_cosmology_analysis,
)


class TestCalcagniSpectralDimension:
    """Tests for Calcagni's running spectral dimension."""

    def test_uv_limit_approaches_2(self):
        """d_S(k) -> 2 as k -> infinity (UV, Planck scale)."""
        k_uv = np.array([100.0, 1000.0, 10000.0])
        d_S = calcagni_spectral_dimension(k_uv)
        # At UV, d_S should approach 2
        for d in d_S:
            assert d < 2.5, f"UV limit should approach 2, got {d}"
        # Higher k should be closer to 2
        assert d_S[-1] < d_S[0], "Higher k should give d_S closer to 2"

    def test_ir_limit_approaches_4(self):
        """d_S(k) -> 4 as k -> 0 (IR, macroscopic scale)."""
        k_ir = np.array([0.001, 0.01, 0.1])
        d_S = calcagni_spectral_dimension(k_ir)
        # At IR, d_S should approach 4
        for d in d_S:
            assert d > 3.0, f"IR limit should approach 4, got {d}"
        # Lower k should be closer to 4
        assert d_S[0] > d_S[-1], "Lower k should give d_S closer to 4"

    def test_monotonic_flow(self):
        """d_S should flow monotonically from UV to IR."""
        k = np.geomspace(0.01, 100.0, 100)
        d_S = calcagni_spectral_dimension(k)
        # d_S should decrease as k increases (from 4 at IR to 2 at UV)
        diffs = np.diff(d_S)
        assert np.all(diffs < 0), "d_S should decrease monotonically with increasing k"

    def test_transition_scale(self):
        """At k = k_* = 1, d_S should be near 3 (midpoint)."""
        k_star = np.array([1.0])
        d_S = calcagni_spectral_dimension(k_star)
        assert 2.5 < d_S[0] < 3.5, f"At transition scale, d_S should be ~3, got {d_S[0]}"

    def test_alpha_parameter_effect(self):
        """Different alpha values should change the interpolation rate."""
        k = np.geomspace(0.1, 10.0, 50)
        d_S_low_alpha = calcagni_spectral_dimension(k, alpha=0.3)
        d_S_high_alpha = calcagni_spectral_dimension(k, alpha=0.7)
        # Both should have same endpoints (approximately)
        # but different transition sharpness
        assert not np.allclose(d_S_low_alpha, d_S_high_alpha), \
            "Different alpha should give different flows"


class TestCDTSpectralDimension:
    """Tests for CDT spectral dimension (Ambjorn et al.)."""

    def test_uv_limit(self):
        """CDT d_S -> 2 at UV (like Calcagni)."""
        k_uv = np.array([100.0, 1000.0])
        d_S = cdt_spectral_dimension(k_uv)
        for d in d_S:
            assert d < 2.5, f"CDT UV limit should approach 2, got {d}"

    def test_ir_limit(self):
        """CDT d_S -> 4 at IR (like Calcagni)."""
        k_ir = np.array([0.001, 0.01])
        d_S = cdt_spectral_dimension(k_ir)
        for d in d_S:
            assert d > 3.5, f"CDT IR limit should approach 4, got {d}"

    def test_qualitative_agreement_with_calcagni(self):
        """CDT and Calcagni should have qualitatively similar behavior."""
        k = np.geomspace(0.01, 100.0, 50)
        d_S_cdt = cdt_spectral_dimension(k)
        d_S_calcagni = calcagni_spectral_dimension(k)
        # Both should be monotonically non-increasing (CDT saturates at 4 for low k)
        assert np.all(np.diff(d_S_cdt) <= 1e-10)
        assert np.all(np.diff(d_S_calcagni) < 0)
        # Both should span similar range
        assert d_S_cdt[0] > 3.0 and d_S_cdt[-1] < 3.0
        assert d_S_calcagni[0] > 3.0 and d_S_calcagni[-1] < 3.0


class TestSpectralExponents:
    """Tests for different spectral exponent frameworks."""

    def test_k_minus_3_normalization(self):
        """k^{-3} spectrum should be normalized at k=1."""
        k = np.geomspace(0.1, 10.0, 100)
        P = k_minus_3_spectrum(k)
        idx_k1 = np.argmin(np.abs(k - 1.0))
        assert np.isclose(P[idx_k1], 1.0, rtol=0.01), \
            f"P(k=1) should be 1.0, got {P[idx_k1]}"

    def test_kolmogorov_is_minus_5_3(self):
        """Kolmogorov spectrum E(k) ~ k^{-5/3}."""
        k = np.array([1.0, 2.0, 4.0, 8.0])
        P = kolmogorov_spectrum(k)
        # Check power law: P(2k)/P(k) = 2^{-5/3}
        expected_ratio = 2 ** (-5.0 / 3.0)
        actual_ratio = P[1] / P[0]
        assert np.isclose(actual_ratio, expected_ratio, rtol=0.01), \
            f"Kolmogorov ratio should be {expected_ratio}, got {actual_ratio}"

    def test_kraichnan_equals_k_minus_3(self):
        """Kraichnan enstrophy cascade E(k) ~ k^{-3} should match k^{-3}."""
        k = np.geomspace(0.1, 10.0, 100)
        P_kraichnan = kraichnan_enstrophy_spectrum(k)
        P_k_minus_3 = k_minus_3_spectrum(k)
        # Should be identical
        assert np.allclose(P_kraichnan, P_k_minus_3), \
            "Kraichnan enstrophy cascade should equal k^{-3}"

    def test_kolmogorov_differs_from_k_minus_3(self):
        """Kolmogorov k^{-5/3} should NOT match k^{-3}."""
        k = np.geomspace(0.1, 10.0, 100)
        P_kolmogorov = kolmogorov_spectrum(k)
        P_k_minus_3 = k_minus_3_spectrum(k)
        # Should NOT be identical
        assert not np.allclose(P_kolmogorov, P_k_minus_3), \
            "Kolmogorov should differ from k^{-3}"
        # Quantify the difference
        rms = compute_rms_deviation(P_kolmogorov, P_k_minus_3)
        assert rms > 0.1, f"RMS deviation should be significant, got {rms}"


class TestParisiSourlas:
    """Tests for Parisi-Sourlas dimensional reduction."""

    def test_d_minus_2_reduction(self):
        """Parisi-Sourlas: D -> D-2."""
        assert parisi_sourlas_effective_dimension(4) == 2
        assert parisi_sourlas_effective_dimension(5) == 3
        assert parisi_sourlas_effective_dimension(6) == 4
        assert parisi_sourlas_effective_dimension(2) == 0
        assert parisi_sourlas_effective_dimension(1) == 0  # max(D-2, 0)

    def test_ps_does_not_produce_k_minus_3(self):
        """Parisi-Sourlas does NOT produce k^{-3} exponent."""
        # For D=4, D_eff=2, spectral exponent = D_eff - 1 = 1 (positive!)
        exp_D4 = parisi_sourlas_spectrum_exponent(4)
        assert exp_D4 == 1.0, f"D=4 should give exponent +1, got {exp_D4}"
        # k^{-3} requires exponent = -3, so PS does NOT match
        assert exp_D4 != -3.0, "Parisi-Sourlas should NOT produce k^{-3}"


class TestOriginAnalysis:
    """Tests for the comprehensive k^{-3} origin analysis."""

    def test_kraichnan_matches(self):
        """Analysis should identify Kraichnan as exact match."""
        results = analyze_k_minus_3_origin()
        fc = results["framework_comparison"]
        assert fc["kraichnan_enstrophy"]["matches_k_minus_3"] is True
        assert fc["kraichnan_enstrophy"]["exponent"] == -3

    def test_kolmogorov_does_not_match(self):
        """Analysis should identify Kolmogorov as non-match."""
        results = analyze_k_minus_3_origin()
        fc = results["framework_comparison"]
        assert fc["kolmogorov"]["matches_k_minus_3"] is False
        assert np.isclose(fc["kolmogorov"]["exponent"], -5/3, rtol=0.01)

    def test_calcagni_does_not_match(self):
        """Analysis should identify Calcagni as non-match."""
        results = analyze_k_minus_3_origin()
        fc = results["framework_comparison"]
        assert fc["calcagni"]["matches_k_minus_3"] is False

    def test_parisi_sourlas_does_not_match(self):
        """Analysis should identify Parisi-Sourlas as non-match."""
        results = analyze_k_minus_3_origin()
        fc = results["framework_comparison"]
        assert fc["parisi_sourlas_D4"]["matches_k_minus_3"] is False

    def test_conclusion_mentions_kraichnan(self):
        """Conclusion should identify Kraichnan 2D enstrophy cascade."""
        results = analyze_k_minus_3_origin()
        conclusion = results["conclusion"].lower()
        assert "kraichnan" in conclusion or "enstrophy" in conclusion


class TestCompareSpectra:
    """Tests for spectrum comparison utility."""

    def test_all_spectra_present(self):
        """compare_spectra should return all expected spectra."""
        spectra = compare_spectra()
        expected_keys = [
            "k", "k_minus_3", "kolmogorov", "kraichnan_enstrophy",
            "calcagni", "calcagni_d_S", "cdt_d_S"
        ]
        for key in expected_keys:
            assert key in spectra, f"Missing key: {key}"

    def test_array_lengths_match(self):
        """All arrays should have same length as k."""
        spectra = compare_spectra()
        n = len(spectra["k"])
        for key, arr in spectra.items():
            assert len(arr) == n, f"{key} has wrong length: {len(arr)} vs {n}"


class TestRMSDeviation:
    """Tests for RMS deviation computation."""

    def test_identical_spectra_zero_deviation(self):
        """Identical spectra should have zero RMS deviation."""
        k = np.geomspace(0.1, 10.0, 50)
        P = k_minus_3_spectrum(k)
        rms = compute_rms_deviation(P, P)
        assert np.isclose(rms, 0.0, atol=1e-10)

    def test_different_spectra_nonzero_deviation(self):
        """Different spectra should have nonzero RMS deviation."""
        k = np.geomspace(0.1, 10.0, 50)
        P1 = k_minus_3_spectrum(k)
        P2 = kolmogorov_spectrum(k)
        rms = compute_rms_deviation(P1, P2)
        assert rms > 0.0


class TestFullAnalysis:
    """Tests for the complete analysis runner."""

    def test_run_analysis_returns_results(self):
        """run_fractal_cosmology_analysis should return complete results."""
        results = run_fractal_cosmology_analysis()
        assert "spectra" in results
        assert "origin_analysis" in results
        assert "uv_ir_analysis" in results

    def test_uv_ir_limits_correct(self):
        """UV/IR limits should be close to expected values."""
        results = run_fractal_cosmology_analysis()
        uv_ir = results["uv_ir_analysis"]
        # Calcagni
        assert uv_ir["calcagni_d_S_UV"] < 3.0, "Calcagni UV should approach 2"
        assert uv_ir["calcagni_d_S_IR"] > 3.5, "Calcagni IR should approach 4"
        # CDT
        assert uv_ir["cdt_d_S_UV"] < 3.0, "CDT UV should approach 2"
        assert uv_ir["cdt_d_S_IR"] > 3.5, "CDT IR should approach 4"

    def test_csv_output(self, tmp_path):
        """Analysis should save CSV files when output_dir provided."""
        results = run_fractal_cosmology_analysis(output_dir=tmp_path)
        # Check CSV files were created
        csv_path = tmp_path / "fractal_cosmology_comparison.csv"
        assert csv_path.exists(), f"Missing {csv_path}"
        dev_path = tmp_path / "spectral_exponent_comparisons.csv"
        assert dev_path.exists(), f"Missing {dev_path}"


# Key scientific validation tests
class TestKeyScientificResults:
    """Tests validating key scientific conclusions."""

    def test_k_minus_3_is_enstrophy_cascade(self):
        """
        KEY RESULT: k^{-3} matches the 2D enstrophy cascade spectrum.

        This is Kraichnan 1967's prediction for the forward enstrophy cascade
        in 2D turbulence at small scales. The project's vacuum dynamics model
        may have an underlying 2D structure or be related to enstrophy conservation.
        """
        k = np.geomspace(0.01, 100.0, 1000)
        P_project = k_minus_3_spectrum(k)
        P_kraichnan = kraichnan_enstrophy_spectrum(k)
        rms = compute_rms_deviation(P_project, P_kraichnan)
        # Exact match expected
        assert rms < 1e-10, \
            f"k^{{-3}} should exactly match Kraichnan enstrophy cascade, RMS={rms}"

    def test_k_minus_3_not_kolmogorov(self):
        """
        k^{-3} is NOT the Kolmogorov 3D turbulence spectrum (k^{-5/3}).

        This rules out 3D homogeneous isotropic turbulence as the origin.
        """
        k = np.geomspace(0.01, 100.0, 1000)
        P_project = k_minus_3_spectrum(k)
        P_kolm = kolmogorov_spectrum(k)
        rms = compute_rms_deviation(P_project, P_kolm)
        # Should be significantly different
        assert rms > 0.1, \
            f"k^{{-3}} should differ from Kolmogorov k^{{-5/3}}, RMS={rms}"

    def test_spectral_dimension_uv_ir_flow(self):
        """
        Both Calcagni and CDT predict d_S: 2 (UV) -> 4 (IR).

        This is a key prediction of quantum gravity approaches:
        spacetime becomes effectively 2D at Planck scale.
        """
        k_uv = np.array([1000.0])
        k_ir = np.array([0.001])

        # Calcagni
        d_S_calc_uv = calcagni_spectral_dimension(k_uv)[0]
        d_S_calc_ir = calcagni_spectral_dimension(k_ir)[0]
        assert d_S_calc_uv < 2.5, f"Calcagni UV d_S should be ~2, got {d_S_calc_uv}"
        assert d_S_calc_ir > 3.8, f"Calcagni IR d_S should be ~4, got {d_S_calc_ir}"

        # CDT
        d_S_cdt_uv = cdt_spectral_dimension(k_uv)[0]
        d_S_cdt_ir = cdt_spectral_dimension(k_ir)[0]
        assert d_S_cdt_uv < 2.5, f"CDT UV d_S should be ~2, got {d_S_cdt_uv}"
        assert d_S_cdt_ir > 3.8, f"CDT IR d_S should be ~4, got {d_S_cdt_ir}"
