//! # Foundations of Non-Associative Geometry
//!
//! This section elucidates the algebraic primitives of the project, focusing on
//! Cayley-Dickson (CD) doubling and the emergence of zero divisors.
//!
//! ## 1. Cayley-Dickson Doubling
//!
//! Let $A$ be an algebra with involution $x \mapsto \bar{x}$. The CD doubling $A \times A$
//! defines multiplication as:
//!
//! $$(a, b)(c, d) = (ac - \gamma \bar{d}b, da + b\bar{c})$$
//!
//! where $\gamma \in \{-1, 1\}$.
//!
//! ### Theorem: Structural Non-Commutativity
//! For all dimensions $d \ge 4$, CD algebras are non-commutative independent of $\gamma$.
//! This is verified computationally across 28 standard signatures (C-546).
//!
//! ## 2. Sedenion Zero Divisors (16D)
//!
//! At 16D, the norm composition law fails: $N(xy) \ne N(x)N(y)$.
//! This permits zero divisors: $xy = 0$ for $x, y \ne 0$.
//!
//! ### The 3:1 Universal Theorem
//! In any sedenion zero-divisor motif, the ratio of pure to mixed triangles
//! is constrained by the Anti-Diagonal Parity Theorem.
//!
//! Let $\eta(a,b) = \psi(\text{lo}_a, \text{hi}_b) \oplus \psi(\text{hi}_a, \text{lo}_b)$.
//! A triangle is pure if $\eta$ is constant across all edges.
//! The Klein-four invariant $F$ forces a 1:3 combinatorial ratio.
