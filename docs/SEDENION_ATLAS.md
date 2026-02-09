<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# THE SEDENION ATLAS: First-Party Math & Source Synthesis

**Subject:** Algebraic Meltdown & Exceptional Manifolds
**Date:** January 27, 2026
**Current Understanding:** Mixed (some math verified; many mappings pending)

## 1. The 42 Assessors (de Marrais, 2000)
**Primary Source:** Robert P. C. de Marrais, *The 42 Assessors and the Box-Kites they fly* (arXiv:math/0011260). (Mirror: https://lygeros.org/wp-content/uploads/perfections/sedenions.pdf)
*   **Discovery:** The sedenion ($16D$) zero-divisors are not chaotic; they are organized into **7 Box-Kites**.
*   **The Math:** Each Box-Kite is an octahedral structure of **Quartets of Assessors**.
*   **Symmetry (partially verified):**
    *   The group-order arithmetic `|PSL(2,7)| = 168` is standard and is checked in-repo (`tests/test_group_theory.py`).
    *   The *mapping* "PSL(2,7) governs box-kites" is **unverified** until we reproduce de Marrais' construction in code and tests.

**In-repo replication (validated):**
- `src/gemini_physics/de_marrais_boxkites.py` computes:
  - 42 primitive assessors (as cross-pairs that participate in diagonal zero-products),
  - 7 box-kites as 7 connected components, each an octahedron graph (6 vertices, 12 edges, degree 4).
- `tests/test_de_marrais_boxkites.py` verifies:
  - the 42/7 counts,
  - the 168 "quartets along 42 assessors" count via `42 * 4 = 168`,
  - the octahedron structure and sign-solutions per edge.

## 2. Zero-Divisor Geometry (Reggiani, 2024)
**Primary Source:** Silvio Reggiani, *The geometry of sedenion zero divisors* (arXiv:2411.18881, 2024-11-28).
*   **Definitions (paper):**
    *   $\mathcal{Z}(\mathbb{S})$: the submanifold of **normalized pairs** $(x,y) \in \mathbb{S}\times\mathbb{S}$
        whose product equals zero ($xy=0$).
    *   $\operatorname{ZD}(\mathbb{S})$: the submanifold of sedenions with **non-trivial annihilators**,
        normalized in the paper as having Euclidean norm $\sqrt{2}$ (see Introduction).
*   **Claim:** $\mathcal{Z}(\mathbb{S})$ is isometric to the exceptional Lie group **$G_2$** equipped with a
    naturally reductive left-invariant metric.
*   **Claim:** $\operatorname{ZD}(\mathbb{S})$ is isometric to the **Stiefel manifold $V_2(\mathbb{R}^7)$**
    with a specific $G_2$-invariant metric.
*   **Note:** For exact hypotheses and proofs, consult the paper directly: https://arxiv.org/abs/2411.18881

**In-repo alignment (validated subset):**
- `src/gemini_physics/sedenion_annihilator.py` computes left/right annihilator dimensions via the
  nullspace of the left/right multiplication operators.
- `tests/test_reggiani_alignment.py` validates that a known zero divisor has a non-trivial annihilator, and
  that a basis unit (e.g. `e1`) is not a zero divisor under this convention. The example zero divisor
  `(e1 + e10)` has norm $\sqrt{2}$, matching Reggiani's `ZD(S)` normalization.
- `docs/REGGIANI_REPLICATION.md` and `tests/test_reggiani_standard_zero_divisors.py` additionally replicate
  Reggiani's "84 standard zero divisors" and verify left/right annihilator nullity `(4,4)` for all of them.

## 3. Algebraic Property Attrition
| Dimension | Name | Property Lost | Property Retained |
| :--- | :--- | :--- | :--- |
| 1D | Real ($\mathbb{R}$) | - | Ordered, Commutative, Associative |
| 2D | Complex ($\mathbb{C}$) | Ordering | Commutative, Associative |
| 4D | Quaternion ($\mathbb{H}$) | Commutativity | Associative |
| 8D | Octonion ($\mathbb{O}$) | Associativity | Alternative, Division |
| 16D | Sedenion ($\mathbb{S}$) | **Alternativity & Division** | *(unverified here; add citations/tests)* |
| 32D+ | Pathions+ | *(varies)* | *(unverified here; add citations/tests)* |

## 4. The Silicon Isomorphism (Paper 2 Synthesis)
**Status:** Speculative analogy (not a validated mathematical isomorphism).
*   **x87 (Scalar):** The "Real" stage. Straightforward, but stack-heavy.
*   **SSE (128-bit):** The "Quaternion/Octonion" stage. Highly symmetrical, normed (unit vectors), fast.
*   **AVX (256-bit):** The "Sedenion" stage. Emergence of **Encoding Zero-Divisors** (VEX prefix, state transition penalties).
*   **AVX-512:** The "Pathion" stage. Total Meltdown. Frequency throttling, complex mask registers, fragmented subsets.

---
*Synthesized; partially verified. See `docs/MATH_VALIDATION_REPORT.md` for the validated subset.*
