# Quantum computing module for gororoba
#
# This module provides:
# - IBM Qiskit runtime integration via qiskit-ibm-runtime
# - Rust-accelerated quantum algorithms via gororoba_py bindings
# - Hybrid classical-quantum workflows

from .ibm_bridge import (
    QiskitService,
    run_grover_on_ibm,
    run_circuit_on_ibm,
    get_available_backends,
)

__all__ = [
    "QiskitService",
    "run_grover_on_ibm",
    "run_circuit_on_ibm",
    "get_available_backends",
]
