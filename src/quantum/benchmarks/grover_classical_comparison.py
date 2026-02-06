# Quantum vs Classical Search Benchmark
#
# Compares Grover's algorithm (simulated) against classical search
# to verify the theoretical O(sqrt(N/M)) speedup.
#
# Reference: Grover, L. K. (1996). arXiv:quant-ph/9605043

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Try to import Rust bindings
try:
    import gororoba_py

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


@dataclass
class SearchResult:
    """Result of a search benchmark."""

    method: str
    n_states: int
    n_marked: int
    oracle_calls: int
    wall_time_ms: float
    found_solution: bool
    theoretical_calls: float


@dataclass
class BenchmarkSummary:
    """Summary comparing quantum and classical search."""

    n_states: int
    n_marked: int
    classical_calls: int
    quantum_calls: int
    speedup_factor: float
    theoretical_speedup: float
    speedup_ratio: float  # actual / theoretical


def classical_random_search(
    oracle: Callable[[int], bool],
    n_states: int,
    n_marked: int,
    seed: int = 42,
) -> SearchResult:
    """Classical random search until a marked state is found.

    Args:
        oracle: Function returning True for marked states
        n_states: Total number of states
        n_marked: Number of marked states (for reference)
        seed: Random seed for reproducibility

    Returns:
        SearchResult with performance metrics
    """
    rng = random.Random(seed)
    calls = 0
    found = False
    start = time.perf_counter()

    # Random search with replacement
    visited = set()
    while len(visited) < n_states:
        idx = rng.randint(0, n_states - 1)
        if idx not in visited:
            visited.add(idx)
            calls += 1
            if oracle(idx):
                found = True
                break

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Theoretical: E[calls] = N/M for random search
    theoretical = n_states / n_marked if n_marked > 0 else n_states

    return SearchResult(
        method="classical_random",
        n_states=n_states,
        n_marked=n_marked,
        oracle_calls=calls,
        wall_time_ms=elapsed_ms,
        found_solution=found,
        theoretical_calls=theoretical,
    )


def quantum_grover_search(
    marked_indices: list[int],
    n_states: int,
) -> SearchResult:
    """Quantum Grover search (simulated via Rust).

    Args:
        marked_indices: Indices of marked states
        n_states: Total number of states

    Returns:
        SearchResult with performance metrics
    """
    if not RUST_AVAILABLE:
        msg = "gororoba_py not available"
        raise ImportError(msg)

    n_qubits = int(math.ceil(math.log2(n_states)))
    n_marked = len(marked_indices)

    start = time.perf_counter()
    iterations, success_prob, candidates = gororoba_py.py_grover_search(
        n_qubits, marked_indices
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Check if any candidate is marked
    found = any(c in marked_indices for c in candidates)

    # Theoretical: O(sqrt(N/M)) iterations
    theoretical = (math.pi / 4) * math.sqrt(n_states / n_marked) if n_marked > 0 else 1

    return SearchResult(
        method="quantum_grover",
        n_states=n_states,
        n_marked=n_marked,
        oracle_calls=iterations,
        wall_time_ms=elapsed_ms,
        found_solution=found,
        theoretical_calls=theoretical,
    )


def compare_search_methods(
    n_states: int,
    n_marked: int,
    seed: int = 42,
) -> BenchmarkSummary:
    """Compare quantum and classical search on the same problem.

    Args:
        n_states: Total search space size (should be power of 2)
        n_marked: Number of marked states
        seed: Random seed

    Returns:
        BenchmarkSummary comparing the methods
    """
    # Ensure power of 2
    n_qubits = int(math.ceil(math.log2(n_states)))
    n_states = 2**n_qubits

    # Generate random marked states
    rng = random.Random(seed)
    marked_indices = rng.sample(range(n_states), n_marked)
    marked_set = set(marked_indices)

    def oracle(idx: int) -> bool:
        return idx in marked_set

    # Run classical search
    classical_result = classical_random_search(oracle, n_states, n_marked, seed)

    # Run quantum search
    if RUST_AVAILABLE:
        quantum_result = quantum_grover_search(marked_indices, n_states)
    else:
        # Fallback: estimate based on theoretical formula
        theoretical_iters = int(
            round((math.pi / 4) * math.sqrt(n_states / n_marked))
        )
        quantum_result = SearchResult(
            method="quantum_grover_theoretical",
            n_states=n_states,
            n_marked=n_marked,
            oracle_calls=theoretical_iters,
            wall_time_ms=0.0,
            found_solution=True,
            theoretical_calls=theoretical_iters,
        )

    # Compute speedup
    speedup = classical_result.oracle_calls / quantum_result.oracle_calls
    theoretical_speedup = math.sqrt(n_states / n_marked)
    speedup_ratio = speedup / theoretical_speedup

    return BenchmarkSummary(
        n_states=n_states,
        n_marked=n_marked,
        classical_calls=classical_result.oracle_calls,
        quantum_calls=quantum_result.oracle_calls,
        speedup_factor=speedup,
        theoretical_speedup=theoretical_speedup,
        speedup_ratio=speedup_ratio,
    )


def run_benchmark_suite(
    sizes: list[int] | None = None,
    marked_fractions: list[float] | None = None,
) -> list[BenchmarkSummary]:
    """Run benchmark suite across multiple problem sizes.

    Args:
        sizes: List of search space sizes (powers of 2)
        marked_fractions: Fractions of states to mark

    Returns:
        List of benchmark summaries
    """
    if sizes is None:
        sizes = [64, 256, 1024, 4096]
    if marked_fractions is None:
        marked_fractions = [0.01, 0.05, 0.1, 0.25]

    results = []
    for n_states in sizes:
        for frac in marked_fractions:
            n_marked = max(1, int(n_states * frac))
            summary = compare_search_methods(n_states, n_marked)
            results.append(summary)

    return results


def format_benchmark_table(results: list[BenchmarkSummary]) -> str:
    """Format benchmark results as a table.

    Args:
        results: List of benchmark summaries

    Returns:
        Formatted table string
    """
    lines = [
        "| N_states | N_marked | Classical | Quantum | Speedup | Theory | Ratio |",
        "|----------|----------|-----------|---------|---------|--------|-------|",
    ]

    for r in results:
        lines.append(
            f"| {r.n_states:8d} | {r.n_marked:8d} | {r.classical_calls:9d} | "
            f"{r.quantum_calls:7d} | {r.speedup_factor:7.2f} | "
            f"{r.theoretical_speedup:6.2f} | {r.speedup_ratio:5.2f} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    print("Quantum vs Classical Search Benchmark")
    print("=" * 60)
    print()

    if not RUST_AVAILABLE:
        print("WARNING: gororoba_py not available, using theoretical estimates")
        print()

    results = run_benchmark_suite()
    print(format_benchmark_table(results))
    print()
    print("Speedup ratio near 1.0 indicates algorithm matches theory.")
