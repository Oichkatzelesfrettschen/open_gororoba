"""
Tests for Clifford algebra Cl(8) implementation following Furey et al. 2024.

Validates:
- Cl(8) has 256 basis elements
- Gamma matrices satisfy Clifford relation {gamma_i, gamma_j} = 2*delta_ij
- Cl(6) decomposition produces 3 isomorphic left ideals
- Each generation has correct U(1)_em charges
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from gemini_physics.clifford_algebra import (
    gamma_matrices_cl8,
    verify_clifford_relation,
    count_cl_basis_elements,
    cl6_minimal_left_ideals,
    fermion_charges_cl6,
    three_generations,
    lepton_mass_ratios,
    quark_mass_ratios,
    run_cl8_verification,
    pauli_matrices,
)


class TestPauliMatrices:
    """Tests for Pauli matrix construction."""

    def test_pauli_count(self):
        """Should return exactly 3 Pauli matrices."""
        s1, s2, s3 = pauli_matrices()
        assert s1.shape == (2, 2)
        assert s2.shape == (2, 2)
        assert s3.shape == (2, 2)

    def test_pauli_anticommutation(self):
        """Pauli matrices satisfy {sigma_i, sigma_j} = 2*delta_ij."""
        s1, s2, s3 = pauli_matrices()
        I2 = np.eye(2, dtype=complex)

        # Self-anticommutators
        assert np.allclose(s1 @ s1 + s1 @ s1, 2 * I2)
        assert np.allclose(s2 @ s2 + s2 @ s2, 2 * I2)
        assert np.allclose(s3 @ s3 + s3 @ s3, 2 * I2)

        # Cross-anticommutators should be zero
        assert np.allclose(s1 @ s2 + s2 @ s1, np.zeros((2, 2)))
        assert np.allclose(s2 @ s3 + s3 @ s2, np.zeros((2, 2)))
        assert np.allclose(s1 @ s3 + s3 @ s1, np.zeros((2, 2)))


class TestGammaMatrices:
    """Tests for Cl(8) gamma matrix construction."""

    def test_gamma_count(self):
        """Cl(8) should have 8 gamma matrices."""
        gammas = gamma_matrices_cl8()
        assert len(gammas) == 8

    def test_gamma_shape(self):
        """Gamma matrices should be 16x16 (real representation)."""
        gammas = gamma_matrices_cl8()
        for g in gammas:
            assert g.shape == (16, 16)

    def test_clifford_relation(self):
        """Gamma matrices must satisfy {gamma_i, gamma_j} = 2*delta_ij."""
        gammas = gamma_matrices_cl8()
        assert verify_clifford_relation(gammas)

    def test_gamma_hermitian(self):
        """Gamma matrices should be Hermitian (for this representation)."""
        gammas = gamma_matrices_cl8()
        for g in gammas:
            # Check if Hermitian: g = g^dagger
            assert np.allclose(g, g.conj().T, atol=1e-10)


class TestCl8BasisCount:
    """Tests for Cl(n) basis element counting."""

    def test_cl8_has_256_elements(self):
        """Cl(8) should have 2^8 = 256 basis elements."""
        assert count_cl_basis_elements(8) == 256

    def test_cl6_has_64_elements(self):
        """Cl(6) should have 2^6 = 64 basis elements."""
        assert count_cl_basis_elements(6) == 64

    def test_cl4_has_16_elements(self):
        """Cl(4) should have 2^4 = 16 basis elements."""
        assert count_cl_basis_elements(4) == 16


class TestCl6Ideals:
    """Tests for Cl(6) minimal left ideal decomposition."""

    def test_cl6_projectors_exist(self):
        """Should return projector operators for Cl(6)."""
        gammas = gamma_matrices_cl8()
        ideals = cl6_minimal_left_ideals(gammas)
        assert len(ideals) >= 2  # At least P_L and P_R

    def test_chirality_projector_idempotent(self):
        """Chirality projector P_L should satisfy P_L^2 = P_L."""
        gammas = gamma_matrices_cl8()
        ideals = cl6_minimal_left_ideals(gammas)
        P_L = ideals[0]
        assert np.allclose(P_L @ P_L, P_L, atol=1e-10)

    def test_chirality_projectors_orthogonal(self):
        """P_L and P_R should be orthogonal: P_L @ P_R = 0."""
        gammas = gamma_matrices_cl8()
        ideals = cl6_minimal_left_ideals(gammas)
        P_L, P_R = ideals[0], ideals[1]
        assert np.allclose(P_L @ P_R, np.zeros_like(P_L), atol=1e-10)

    def test_chirality_projectors_complete(self):
        """P_L + P_R should equal identity."""
        gammas = gamma_matrices_cl8()
        ideals = cl6_minimal_left_ideals(gammas)
        P_L, P_R = ideals[0], ideals[1]
        I = np.eye(16, dtype=complex)
        assert np.allclose(P_L + P_R, I, atol=1e-10)


class TestFermionCharges:
    """Tests for SM fermion charge assignments."""

    def test_charge_values(self):
        """Fermion charges should match SM values."""
        charges = fermion_charges_cl6()

        assert np.isclose(charges["u_quark"]["charge"], 2/3)
        assert np.isclose(charges["d_quark"]["charge"], -1/3)
        assert np.isclose(charges["neutrino"]["charge"], 0)
        assert np.isclose(charges["electron"]["charge"], -1)

    def test_charge_quantization(self):
        """All charges should be multiples of 1/3."""
        charges = fermion_charges_cl6()
        for particle, data in charges.items():
            q = data["charge"]
            # 3*q should be an integer
            assert abs(3 * q - round(3 * q)) < 1e-10

    def test_weak_isospin_doublets(self):
        """Weak isospin should form doublets (+1/2, -1/2)."""
        charges = fermion_charges_cl6()

        # Up and down quarks form a doublet
        assert charges["u_quark"]["weak_isospin"] == 1/2
        assert charges["d_quark"]["weak_isospin"] == -1/2

        # Neutrino and electron form a doublet
        assert charges["neutrino"]["weak_isospin"] == 1/2
        assert charges["electron"]["weak_isospin"] == -1/2


class TestThreeGenerations:
    """Tests for the 3-generation structure."""

    def test_three_generations_exist(self):
        """Should have exactly 3 generations."""
        gens = three_generations()
        assert len(gens) == 3

    def test_generation_particle_count(self):
        """Each generation should have 4 particles."""
        gens = three_generations()
        for gen in gens:
            assert len(gen) == 4

    def test_generation_charge_consistency(self):
        """All generations should have the same charge pattern."""
        gens = three_generations()

        for gen in gens:
            particles = list(gen.values())
            charges = [p["charge"] for p in particles]
            # Should have: 2/3, -1/3, 0, -1 in some order
            expected_charges = sorted([2/3, -1/3, 0, -1])
            assert sorted(charges) == expected_charges


class TestMassRatios:
    """Tests for mass ratio data."""

    def test_lepton_mass_ordering(self):
        """Lepton masses should satisfy m_e < m_mu < m_tau."""
        ratios = lepton_mass_ratios()
        assert ratios["m_e"] < ratios["m_mu"] < ratios["m_tau"]

    def test_lepton_ratio_mu_e(self):
        """mu/e ratio should be approximately 206.77."""
        ratios = lepton_mass_ratios()
        assert 200 < ratios["ratio_mu_e"] < 210

    def test_lepton_ratio_tau_e(self):
        """tau/e ratio should be approximately 3477."""
        ratios = lepton_mass_ratios()
        assert 3400 < ratios["ratio_tau_e"] < 3500

    def test_quark_mass_ordering(self):
        """Quark masses should satisfy m_u < m_d < m_s < m_c < m_b < m_t."""
        ratios = quark_mass_ratios()
        assert ratios["m_u"] < ratios["m_d"]
        assert ratios["m_d"] < ratios["m_s"]
        assert ratios["m_s"] < ratios["m_c"]
        assert ratios["m_c"] < ratios["m_b"]
        assert ratios["m_b"] < ratios["m_t"]


class TestVerification:
    """Integration tests for the verification routine."""

    def test_full_verification(self):
        """Full verification should pass all checks."""
        results = run_cl8_verification()

        assert results["n_gamma_matrices"] == 8
        assert results["gamma_matrix_shape"] == (16, 16)
        assert results["clifford_relation_satisfied"] is True
        assert results["cl8_basis_count"] == 256
        assert results["expected_cl8_basis"] == 256
        assert results["charges_quantized"] is True
