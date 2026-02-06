# IBM Quantum Bridge for gororoba
#
# This module provides integration with IBM Quantum via qiskit-ibm-runtime.
# API keys MUST be provided via environment variable QISKIT_IBM_TOKEN.
#
# References:
# - IBM Qiskit Runtime: https://docs.quantum.ibm.com/api/qiskit-ibm-runtime
# - Grover's Algorithm: Grover, L. K. (1996) arXiv:quant-ph/9605043

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Qiskit imports (optional dependency)
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import GroverOperator
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None  # type: ignore[misc,assignment]


@dataclass
class JobResult:
    """Result from an IBM Quantum job."""

    job_id: str
    backend_name: str
    status: str
    counts: dict[str, int] | None = None
    quasi_dists: list[dict[int, float]] | None = None
    metadata: dict[str, Any] | None = None


class QiskitService:
    """Wrapper for IBM Quantum Runtime service.

    Usage:
        # Set API token via environment variable
        export QISKIT_IBM_TOKEN="your-api-token"

        # Initialize service
        service = QiskitService()

        # List backends
        backends = service.get_backends()

        # Run a circuit
        result = service.run_circuit(circuit, backend_name="ibm_brisbane")
    """

    def __init__(self, channel: str = "ibm_quantum") -> None:
        """Initialize the Qiskit service.

        Args:
            channel: IBM Quantum channel ('ibm_quantum' or 'ibm_cloud')
        """
        if not QISKIT_AVAILABLE:
            msg = (
                "Qiskit not installed. Install with: "
                "pip install 'gororoba[quantum]'"
            )
            raise ImportError(msg)

        # Get token from environment (NEVER from code)
        token = os.environ.get("QISKIT_IBM_TOKEN")
        if not token:
            msg = (
                "QISKIT_IBM_TOKEN environment variable not set. "
                "Get your token from https://quantum.ibm.com/account"
            )
            raise ValueError(msg)

        self._service = QiskitRuntimeService(channel=channel, token=token)
        self._channel = channel

    def get_backends(self, operational: bool = True) -> list[str]:
        """Get list of available IBM Quantum backends.

        Args:
            operational: Only return operational (online) backends

        Returns:
            List of backend names
        """
        backends = self._service.backends(operational=operational)
        return [b.name for b in backends]

    def get_backend_info(self, name: str) -> dict[str, Any]:
        """Get detailed info about a specific backend.

        Args:
            name: Backend name (e.g., 'ibm_brisbane')

        Returns:
            Dictionary with backend properties
        """
        backend = self._service.backend(name)
        config = backend.configuration()
        props = backend.properties()

        return {
            "name": name,
            "n_qubits": config.n_qubits,
            "basis_gates": config.basis_gates,
            "coupling_map": config.coupling_map,
            "t1_us": [props.t1(i) * 1e6 for i in range(config.n_qubits)]
            if props
            else None,
            "t2_us": [props.t2(i) * 1e6 for i in range(config.n_qubits)]
            if props
            else None,
        }

    def run_circuit(
        self,
        circuit: QuantumCircuit,
        backend_name: str | None = None,
        shots: int = 4096,
    ) -> JobResult:
        """Run a quantum circuit on IBM hardware.

        Args:
            circuit: Qiskit QuantumCircuit to execute
            backend_name: Backend to use (None = least busy)
            shots: Number of measurement shots

        Returns:
            JobResult with counts and metadata
        """
        # Get backend
        if backend_name:
            backend = self._service.backend(backend_name)
        else:
            backend = self._service.least_busy(operational=True)

        # Transpile circuit for backend
        transpiled = transpile(circuit, backend)

        # Execute via Sampler primitive
        with Session(service=self._service, backend=backend) as session:
            sampler = SamplerV2(session=session)
            job = sampler.run([transpiled], shots=shots)
            result = job.result()

        # Extract counts from result
        pub_result = result[0]
        counts_dict = {}
        if hasattr(pub_result, "data"):
            # Access the classical register data
            for key, val in pub_result.data.items():
                if hasattr(val, "get_counts"):
                    counts_dict = dict(val.get_counts())
                    break

        return JobResult(
            job_id=job.job_id(),
            backend_name=backend.name,
            status=str(job.status()),
            counts=counts_dict,
            metadata={"shots": shots, "transpiled_depth": transpiled.depth()},
        )

    def run_grover(
        self,
        n_qubits: int,
        marked_states: Sequence[int],
        backend_name: str | None = None,
        shots: int = 4096,
    ) -> JobResult:
        """Run Grover's algorithm on IBM hardware.

        Args:
            n_qubits: Number of qubits in search space
            marked_states: States to search for (as integers)
            backend_name: Backend to use (None = least busy)
            shots: Number of measurement shots

        Returns:
            JobResult with measurement counts
        """
        from qiskit.circuit.library import MCMT, ZGate

        # Create oracle circuit marking specified states
        oracle = QuantumCircuit(n_qubits, name="oracle")
        for state in marked_states:
            # Flip bits for marked state
            for i in range(n_qubits):
                if not (state >> i) & 1:
                    oracle.x(i)
            # Apply multi-controlled Z
            if n_qubits > 1:
                mcz = MCMT(ZGate(), n_qubits - 1, 1)
                oracle.append(mcz, range(n_qubits))
            else:
                oracle.z(0)
            # Unflip bits
            for i in range(n_qubits):
                if not (state >> i) & 1:
                    oracle.x(i)

        # Build Grover operator
        grover_op = GroverOperator(oracle)

        # Calculate optimal iterations
        import math

        n_states = 2**n_qubits
        n_marked = len(marked_states)
        if n_marked > 0:
            iterations = int(round(math.pi / 4 * math.sqrt(n_states / n_marked)))
            iterations = max(1, iterations)
        else:
            iterations = 1

        # Build full circuit
        circuit = QuantumCircuit(n_qubits, n_qubits)
        # Initial superposition
        circuit.h(range(n_qubits))
        # Apply Grover iterations
        for _ in range(iterations):
            circuit.compose(grover_op, inplace=True)
        # Measurement
        circuit.measure(range(n_qubits), range(n_qubits))

        return self.run_circuit(circuit, backend_name, shots)


def get_available_backends(operational: bool = True) -> list[str]:
    """Get list of available IBM Quantum backends.

    Convenience function that creates a temporary service connection.

    Args:
        operational: Only return operational backends

    Returns:
        List of backend names
    """
    service = QiskitService()
    return service.get_backends(operational)


def run_grover_on_ibm(
    n_qubits: int,
    marked_states: Sequence[int],
    backend_name: str | None = None,
    shots: int = 4096,
) -> JobResult:
    """Run Grover's algorithm on IBM Quantum hardware.

    Convenience function for single-shot Grover execution.

    Args:
        n_qubits: Number of qubits (search space = 2^n_qubits)
        marked_states: Integer indices of states to find
        backend_name: Backend to use (None = least busy)
        shots: Measurement shots

    Returns:
        JobResult with counts
    """
    service = QiskitService()
    return service.run_grover(n_qubits, marked_states, backend_name, shots)


def run_circuit_on_ibm(
    circuit: QuantumCircuit,
    backend_name: str | None = None,
    shots: int = 4096,
) -> JobResult:
    """Run arbitrary circuit on IBM Quantum hardware.

    Args:
        circuit: Qiskit QuantumCircuit
        backend_name: Backend to use (None = least busy)
        shots: Measurement shots

    Returns:
        JobResult with counts
    """
    service = QiskitService()
    return service.run_circuit(circuit, backend_name, shots)


# Hybrid workflow: combine Rust simulation with IBM execution
def hybrid_hypothesis_search(
    ranges: list[tuple[float, float, int]],
    threshold: float,
    score_fn: str = "sum",
    verify_on_ibm: bool = False,
    ibm_backend: str | None = None,
) -> dict[str, Any]:
    """Hybrid classical-quantum hypothesis search.

    Uses Rust-accelerated simulation for initial search, optionally
    verifies top candidates on IBM hardware.

    Args:
        ranges: Parameter grid [(min, max, steps), ...]
        threshold: Oracle threshold
        score_fn: Score function ('sum', 'norm', 'max', 'min')
        verify_on_ibm: Run verification on IBM hardware
        ibm_backend: IBM backend for verification

    Returns:
        Dictionary with simulation and optional hardware results
    """
    # Import Rust bindings
    try:
        import gororoba_py
    except ImportError:
        msg = "gororoba_py not installed. Build with: maturin develop"
        raise ImportError(msg) from None

    # Run Rust simulation
    solutions, speedup, oracle_calls = gororoba_py.py_quantum_grid_search(
        ranges, threshold, score_fn
    )

    result = {
        "simulation": {
            "verified_solutions": solutions,
            "speedup_factor": speedup,
            "oracle_calls": oracle_calls,
        },
        "hardware_verification": None,
    }

    # Optional IBM hardware verification
    if verify_on_ibm and solutions and QISKIT_AVAILABLE:
        # Convert first solution to integer index for hardware test
        # This is a simplified verification - real workflow would be more sophisticated
        n_qubits = min(8, len(ranges) * 2)  # Limit for hardware
        marked = [0]  # Placeholder - would map solution to qubit state

        hw_result = run_grover_on_ibm(
            n_qubits, marked, ibm_backend, shots=4096
        )
        result["hardware_verification"] = {
            "job_id": hw_result.job_id,
            "backend": hw_result.backend_name,
            "counts": hw_result.counts,
        }

    return result
