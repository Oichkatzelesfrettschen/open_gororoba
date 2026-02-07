# Research Roadmap: From Toy Model to Theory

**Date:** January 26, 2026 (updated 2026-02-06)
**Phase:** Post-migration; all 15 Python modules ported to Rust.
**Consolidated into:** [`docs/ROADMAP.md`](ROADMAP.md) Section 6 (Research Workstreams).

This roadmap mixes verified mathematical replication work with speculative physics/quantum plans.
Treat anything not backed by Rust tests + first-party sources as **unverified**.

## Completed Milestones
*   [x] **Replication:** Verified 42 assessors / 7 box-kites (Rust: `crates/algebra_core/src/boxkites.rs`).
*   [x] **Sanity:** F4 has 48 roots (standard math fact; not a repo-specific discovery).
*   [ ] **Symmetry:** Box-kite antipodal invariance via Qiskit simulation (not currently validated here).
*   [x] **Formalism:** LaTeX definitions drafted (`MATHEMATICAL_FORMALISM.tex`).
*   [x] **Python-to-Rust migration:** All 15 Python modules ported; gororoba_kernels removed.
*   [x] **Zero-divisor motifs at dim=32:** 15 components found (8 heptacross + 7 mixed-degree).
*   [x] **Real cosmology fitting:** Pantheon+ 1578 SNe + DESI DR1 7 BAO bins.

## Active Workstreams

### 1. Quantum "F4 Circuit" Design
*   **Goal:** Implement the full F4 Weyl group operations on the 48-root quantum state.
*   **Next Step:** Translate the mathematical permutation group of F4 into a sequence of CNOT/SWAP gates.

### 2. Holographic Entropy Simulation
*   **Goal:** Run the "Recursive Entropy PDE" on a hyperbolic tensor network.
*   **Next Step:** Map entropy PDE logic onto a hyperbolic graph (Rust: `crates/quantum_core/`).

### 3. Publication Prep
*   **Goal:** Assemble the whitepaper components.
*   **Next Step:** Generate PDF from `MATHEMATICAL_FORMALISM.tex` (requires local latex install or container).

### 4. GR Module Expansion (Blackhole C++ Port)
*   **Goal:** Port verified GR computations from the Blackhole C++ codebase.
*   **Next Step:** Kerr Christoffel symbols, elliptic integrals, Novikov-Thorne disk.
*   **Tracker:** See pending tasks #28-#47 in the task list.

## Future Directions (2026+)
*   **Hardware Run:** Execute the Box-Kite Symmetry circuit on IBM Eagle (127q) via the Cloud.
*   **Standard Model Embedding:** Attempt to embed SU(3) x SU(2) x U(1) generators into the G2 subalgebra of the verified F4 root system.
*   **Cross-domain ultrametric analysis:** Extend C-437 multi-attribute Euclidean ultrametricity to new catalogs.
