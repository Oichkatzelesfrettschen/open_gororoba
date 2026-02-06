# Quantum computing module for gororoba
#
# This module provides:
# - Quantum algorithm benchmarks
# - Integration with IBM Qiskit runtime
# - Rust-accelerated quantum simulation via gororoba_py

from .benchmarks import grover_classical_comparison

__all__ = ["grover_classical_comparison"]
