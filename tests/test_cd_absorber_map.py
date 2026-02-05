"""
Tests for CD Zero-Divisor to Metamaterial Absorber Mapping (C-010).
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_physics.cd_absorber_map import (
    map_zd_to_refractive_index,
    map_zd_norm_to_thickness,
    classify_material_type,
    map_zd_pair_to_layer,
    build_absorber_stack,
    verify_physical_realizability,
    run_cd_absorber_mapping,
    _fallback_sedenion_zd_pairs,
)


class TestRefractiveIndexMapping:
    """Tests for refractive index mapping from ZD indices."""

    def test_refractive_index_positive_real_part(self):
        """Refractive index real part should be positive for physical layers."""
        for i in range(1, 10):
            for j in range(i + 1, 10):
                n = map_zd_to_refractive_index(i, j, j + 1, j + 2)
                assert n.real > 0, f"n.real should be > 0, got {n.real}"

    def test_refractive_index_nonnegative_imag_part(self):
        """Refractive index imaginary part (extinction) should be non-negative."""
        for i in range(1, 10):
            for j in range(i + 1, 10):
                n = map_zd_to_refractive_index(i, j, j + 1, j + 2)
                assert n.imag >= 0, f"n.imag should be >= 0, got {n.imag}"

    def test_base_n_affects_result(self):
        """Different base_n values produce different results."""
        n1 = map_zd_to_refractive_index(1, 2, 4, 8, base_n=1.0)
        n2 = map_zd_to_refractive_index(1, 2, 4, 8, base_n=2.0)
        assert n1.real != n2.real


class TestThicknessMapping:
    """Tests for ZD norm to thickness mapping."""

    def test_thickness_positive(self):
        """Thickness should always be positive."""
        for norm in [0.0, 0.1, 0.5, 1.0, 10.0]:
            t = map_zd_norm_to_thickness(norm)
            assert t > 0, f"Thickness should be > 0, got {t}"

    def test_thickness_bounded(self):
        """Thickness should be within specified bounds."""
        min_t, max_t = 10.0, 200.0
        for norm in [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]:
            t = map_zd_norm_to_thickness(norm, min_thickness=min_t, max_thickness=max_t)
            assert min_t <= t <= max_t, f"Thickness {t} not in [{min_t}, {max_t}]"

    def test_lower_norm_thinner_layer(self):
        """Lower ZD norm should produce thinner layers (inverse scaling)."""
        t_low = map_zd_norm_to_thickness(0.0, inverse_scaling=True)
        t_high = map_zd_norm_to_thickness(1.0, inverse_scaling=True)
        assert t_low < t_high, "Lower norm should give thinner layer"


class TestMaterialClassification:
    """Tests for material type classification."""

    def test_dielectric_classification(self):
        """Low-loss positive index -> dielectric."""
        n = complex(1.5, 0.01)
        assert classify_material_type(n) == "dielectric"

    def test_plasmonic_classification(self):
        """High-loss -> plasmonic."""
        n = complex(1.5, 0.8)
        assert classify_material_type(n) == "plasmonic"

    def test_hyperbolic_classification(self):
        """Negative real index -> hyperbolic."""
        n = complex(-0.5, 0.1)
        assert classify_material_type(n) == "hyperbolic"


class TestLayerMapping:
    """Tests for complete ZD pair to layer mapping."""

    def test_mapping_produces_valid_layer(self):
        """Mapping should produce a valid layer structure."""
        zd = (1, 2, 4, 8, 0.0)
        mapping = map_zd_pair_to_layer(zd, layer_id=0)

        assert mapping.layer.layer_id == 0
        assert mapping.zd_indices == (1, 2, 4, 8)
        assert mapping.product_norm == 0.0

    def test_mapping_physical_realizability(self):
        """Typical ZD pairs should produce physical layers."""
        zd = (1, 2, 4, 8, 0.0)
        mapping = map_zd_pair_to_layer(zd, layer_id=0)

        # Check physical constraints
        assert mapping.layer.n_real > 0
        assert mapping.layer.n_imag >= 0
        assert mapping.layer.thickness_nm > 0
        assert mapping.is_physical


class TestAbsorberStack:
    """Tests for building absorber stacks from ZD pairs."""

    def test_stack_respects_max_layers(self):
        """Stack should not exceed max_layers."""
        zd_pairs = _fallback_sedenion_zd_pairs()
        stack = build_absorber_stack(zd_pairs, max_layers=5)
        assert len(stack) <= 5

    def test_stack_sorted_by_norm(self):
        """Stack should have best annihilators (lowest norm) first."""
        zd_pairs = _fallback_sedenion_zd_pairs()
        stack = build_absorber_stack(zd_pairs, max_layers=10)

        norms = [m.product_norm for m in stack]
        assert norms == sorted(norms), "Stack should be sorted by norm"


class TestVerification:
    """Tests for physical realizability verification."""

    def test_verification_counts_correctly(self):
        """Verification should correctly count layer types."""
        zd_pairs = _fallback_sedenion_zd_pairs()
        stack = build_absorber_stack(zd_pairs, max_layers=10)
        verification = verify_physical_realizability(stack)

        assert verification["n_total"] == len(stack)
        assert verification["n_physical"] <= verification["n_total"]
        assert (
            verification["n_dielectric"] +
            verification["n_plasmonic"] +
            verification["n_hyperbolic"]
        ) == verification["n_total"]

    def test_all_physical_check(self):
        """all_physical should be True iff all layers are physical."""
        zd_pairs = _fallback_sedenion_zd_pairs()
        stack = build_absorber_stack(zd_pairs, max_layers=10)
        verification = verify_physical_realizability(stack)

        expected = verification["n_physical"] == verification["n_total"]
        assert verification["all_physical"] == expected


class TestFullAnalysis:
    """Integration tests for the complete analysis."""

    def test_analysis_runs(self):
        """Full analysis completes without errors."""
        results = run_cd_absorber_mapping(dim=16)
        assert "stack" in results
        assert "verification" in results

    def test_analysis_produces_physical_layers(self):
        """Analysis should produce mostly physical layers."""
        results = run_cd_absorber_mapping(dim=16)
        verification = results["verification"]

        # At least 80% should be physically realizable
        ratio = verification["n_physical"] / max(verification["n_total"], 1)
        assert ratio >= 0.8, f"Physical layer ratio {ratio} < 0.8"

    def test_analysis_outputs_csv(self, tmp_path):
        """Analysis saves CSV when output_dir provided."""
        results = run_cd_absorber_mapping(dim=16, output_dir=tmp_path)
        csv_path = tmp_path / "cd_zd_absorber_mapping.csv"
        assert csv_path.exists()


class TestPhysicalConstraints:
    """Tests for physical realizability constraints (C-010 key requirement)."""

    def test_all_n_positive(self):
        """C-010: All layers must have n > 0 (real refractive index)."""
        results = run_cd_absorber_mapping(dim=16)
        verification = results["verification"]
        assert verification["all_n_positive"], "All n must be > 0"

    def test_all_k_nonnegative(self):
        """C-010: All layers must have k >= 0 (extinction coefficient)."""
        results = run_cd_absorber_mapping(dim=16)
        verification = results["verification"]
        assert verification["all_k_nonnegative"], "All k must be >= 0"

    def test_all_thickness_positive(self):
        """C-010: All layers must have thickness > 0."""
        results = run_cd_absorber_mapping(dim=16)
        verification = results["verification"]
        assert verification["all_thickness_positive"], "All thickness must be > 0"
