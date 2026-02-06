# Quantum benchmarks module
#
# Provides performance comparisons between quantum and classical algorithms.

from .grover_classical_comparison import (
    BenchmarkSummary,
    SearchResult,
    classical_random_search,
    compare_search_methods,
    format_benchmark_table,
    quantum_grover_search,
    run_benchmark_suite,
)

__all__ = [
    "BenchmarkSummary",
    "SearchResult",
    "classical_random_search",
    "compare_search_methods",
    "format_benchmark_table",
    "quantum_grover_search",
    "run_benchmark_suite",
]
