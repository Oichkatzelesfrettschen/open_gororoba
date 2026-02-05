# Research Roadmap: From Toy Model to Theory

**Date:** January 26, 2026
**Phase:** 3. Formalization & Quantum Verification

This roadmap mixes verified mathematical replication work with speculative physics/quantum plans.
Treat anything not backed by `tests/` + first-party sources as **unverified**.

## Completed Milestones
*   [x] **Replication:** Verified 42 assessors / 7 box-kites (see `tests/test_de_marrais_boxkites.py`).
*   [x] **Sanity:** $F_4$ has 48 roots (standard math fact; not a repo-specific discovery).
*   [ ] **Symmetry:** Box-kite antipodal invariance via Qiskit simulation (not currently validated here).
*   [x] **Formalism:** LaTeX definitions drafted (`MATHEMATICAL_FORMALISM.tex`).

## Active Workstreams

### 1. Quantum "F4 Circuit" Design
*   **Goal:** Implement the full $F_4$ Weyl group operations on the 48-root quantum state.
*   **Next Step:** Translate the mathematical permutation group of $F_4$ into a sequence of CNOT/SWAP gates.

### 2. Holographic Entropy Simulation
*   **Goal:** Run the "Recursive Entropy PDE" on a hyperbolic tensor network.
*   **Next Step:** Map the `entropy_pde_fit.py` logic onto a `networkx` hyperbolic graph instead of a Euclidean grid.

### 3. Publication Prep
*   **Goal:** Assemble the whitepaper components.
*   **Next Step:** Generate PDF from `MATHEMATICAL_FORMALISM.tex` (requires local latex install or container).

## Future Directions (2026+)
*   **Hardware Run:** Execute the Box-Kite Symmetry circuit on IBM Eagle (127q) via the Cloud.
*   **Standard Model Embedding:** Attempt to embed the $SU(3) \times SU(2) \times U(1)$ generators into the $G_2$ subalgebra of the verified $F_4$ root system.
