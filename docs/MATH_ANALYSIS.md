# Unified Algebra-Lattice-Graph Monograph

## Abstract
This document reconstructs the research program found in the project artifacts: a family of high-dimensional Cayley-Dickson-style algebras coupled to an 8-dimensional integer lattice codebook.

## 1. The Codebook Ladder
We have identified a strict filtration of lattice subsets:
`Lambda_256 < Lambda_512 < Lambda_1024 < Lambda_2048 < {-1, 0, 1}^8`

### Predicate Cuts
*   **2048D**: Base universe (trinary, even sum, even weight) minus 139 forbidden prefixes.
*   **1024D**: 2048D intersected with `l_0 = -1`.
*   **512D**: 1024D minus 6 forbidden regions (trie cuts).
*   **256D**: 512D minus 6 forbidden regions.

This structure implies a deliberate **hierarchical encoding** rather than random sampling.

## 2. Scalar Shadow Action
The "2048D" algebra currently couples to the lattice via a **scalar shadow** `pi(b)` in `{-1, 0, 1}`.
*   **Addition**: `l_out = l + pi(b) * [1,1,1,1,1,1,1,1]` (Affine shift).
*   **Multiplication**: `l_out = pi(b) * l` (Linear scaling/sign flip).

## 3. Graph Projections (Motifs)
We have identified two distinct graph projections for the 64D/128D algebras:

1.  **Pathion Adjacency (Perfect Matching)**:
    *   Rule: `j = i XOR (Dim/16)`
    *   Spectrum: `{-1, +1}` (multiplicity N/2).
    *   Observed at 64D and 128D.

2.  **Zero-Divisor Adjacency (Parity Cliques)**:
    *   Rule: `i != j` and `i % 2 == j % 2`.
    *   Structure: `K_{N/2} U K_{N/2}`.
    *   Observed at 32D and 64D.

## 4. Implementation Status
*   **Typed Carrier**: Implemented `UniversalAlgebra` struct.
*   **Optimization**: Implemented non-allocating `cd_multiply_mut`.
*   **Logic**: Implemented predicate cuts in `algebra_core::analysis::codebook`.
*   **Graphs**: Implemented generators in `algebra_core::analysis::graph_projections`.

## 5. Next Steps (Research)
*   **Generate 32D/256D Matching Graphs**: Verify the `i XOR (Dim/16)` conjecture.
*   **Non-Scalar Action**: Experimentally determine `rho(b)` for non-scalar basis elements.
