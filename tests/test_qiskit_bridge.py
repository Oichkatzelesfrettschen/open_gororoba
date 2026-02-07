# Tests for IBM Qiskit bridge module
#
# Note: Tests requiring IBM hardware are marked with @pytest.mark.ibm
# and require QISKIT_IBM_TOKEN environment variable.

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Skip entire module if qiskit not installed
pytest.importorskip("qiskit")

from src.quantum_runtime.ibm_bridge import (
    JobResult,
    QiskitService,
)


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_creation(self) -> None:
        result = JobResult(
            job_id="test-123",
            backend_name="ibm_brisbane",
            status="COMPLETED",
            counts={"00": 1000, "11": 1000},
        )
        assert result.job_id == "test-123"
        assert result.backend_name == "ibm_brisbane"
        assert result.counts == {"00": 1000, "11": 1000}

    def test_optional_fields(self) -> None:
        result = JobResult(
            job_id="test-456",
            backend_name="ibm_sherbrooke",
            status="QUEUED",
        )
        assert result.counts is None
        assert result.quasi_dists is None
        assert result.metadata is None


class TestQiskitServiceInit:
    """Tests for QiskitService initialization."""

    def test_missing_token_raises(self) -> None:
        # Ensure token is not set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("QISKIT_IBM_TOKEN", None)
            with pytest.raises(ValueError, match="QISKIT_IBM_TOKEN"):
                QiskitService()


class TestGroverCircuit:
    """Tests for Grover circuit construction (no hardware)."""

    def test_grover_oracle_construction(self) -> None:
        """Test that Grover oracle circuit is constructed correctly."""
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import MCMT, ZGate

        n_qubits = 3
        marked_states = [5]  # Binary: 101

        # Build oracle manually (same logic as in ibm_bridge)
        oracle = QuantumCircuit(n_qubits, name="oracle")
        for state in marked_states:
            for i in range(n_qubits):
                if not (state >> i) & 1:
                    oracle.x(i)
            mcz = MCMT(ZGate(), n_qubits - 1, 1)
            oracle.append(mcz, range(n_qubits))
            for i in range(n_qubits):
                if not (state >> i) & 1:
                    oracle.x(i)

        # Verify circuit structure
        assert oracle.num_qubits == 3
        # Check that X gates are applied for bits that are 0 in state 5 (101)
        # Bit 1 is 0, so we expect X on qubit 1
        x_count = sum(1 for instr in oracle.data if instr.operation.name == "x")
        assert x_count == 2  # One X before MCZ, one after (for bit 1)

    def test_optimal_iterations_calculation(self) -> None:
        """Test optimal Grover iteration calculation."""
        import math

        n_qubits = 4
        n_states = 2**n_qubits  # 16
        n_marked = 1

        expected = int(round(math.pi / 4 * math.sqrt(n_states / n_marked)))
        assert expected == 3  # pi/4 * sqrt(16) ~ 3.14


@pytest.mark.skipif(
    "QISKIT_IBM_TOKEN" not in os.environ,
    reason="QISKIT_IBM_TOKEN not set",
)
class TestQiskitServiceLive:
    """Live tests requiring IBM Quantum access."""

    def test_get_backends(self) -> None:
        """Test fetching available backends."""
        service = QiskitService()
        backends = service.get_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0
        # All backend names should be strings
        assert all(isinstance(b, str) for b in backends)


class TestHybridWorkflow:
    """Tests for hybrid classical-quantum workflow."""

    def test_rust_simulation_available(self) -> None:
        """Test that Rust bindings are available for simulation."""
        try:
            import gororoba_py

            # Test grid search
            ranges = [(0.0, 1.0, 4), (0.0, 1.0, 4)]
            solutions, speedup, calls = gororoba_py.py_quantum_grid_search(
                ranges, 1.5, "sum"
            )
            assert isinstance(solutions, list)
            assert isinstance(speedup, float)
            assert isinstance(calls, int)
        except ImportError:
            pytest.skip("gororoba_py not built")

    def test_hardware_profiles(self) -> None:
        """Test that hardware profile functions work."""
        try:
            import gororoba_py

            # Neutral atom
            n, t1, t2, e1, e2 = gororoba_py.py_neutral_atom_profile(20)
            assert n == 20
            assert t1 > 0  # T1 should be positive
            assert t2 > 0  # T2 should be positive

            # IBM superconducting
            n, vendor, t1, t2, e1, e2 = gororoba_py.py_superconducting_ibm_profile(127)
            assert n == 127
            assert vendor == "ibm"

            # Trapped ion
            n, t1, t2, e1, e2 = gororoba_py.py_trapped_ion_profile(32)
            assert n == 32

        except ImportError:
            pytest.skip("gororoba_py not built")
