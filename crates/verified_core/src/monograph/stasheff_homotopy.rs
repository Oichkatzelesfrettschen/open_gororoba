//! # Homotopical Algebra and the Stasheff Polytope
//!
//! This section elucidates the deep connection between the Cayley-Dickson
//! hierarchy and the Stasheff associahedra, modeling non-associativity
//! as a strongly homotopy associative ($A_\infty$) structure.
//!
//! ## 1. The Stasheff Hierarchy
//!
//! The Stasheff polytope $K_n$ (associahedron) encodes all possible ways to
//! parenthesize a product of $n$ elements.
//!
//! - **$K_3$ (Edge):** Connects $(ab)c$ and $a(bc)$. The boundary of associativity.
//! - **$K_4$ (Pentagon):** Encodes the 5 parenthesizations of 4 elements.
//!   The octonions satisfy the pentagon identity, making them the first
//!   "coherent" non-associative structure.
//! - **$K_5$ (3D Polytope):** Encodes 14 parenthesizations of 5 elements.
//!   The failure of coherence in sedenions ($F_4$ operator) corresponds to
//!   the boundary of $K_5$.
//!
//! ## 2. Cayley-Dickson as an $A_\infty$ Chain
//!
//! Each step in the CD construction introduces a non-vanishing map $m_n$
//! in the Stasheff hierarchy:
//!
//! | Algebra | Dimension | Property Lost | Stasheff Map | Polytope |
//! | :--- | :---: | :--- | :---: | :--- |
//! | Complex | 2 | Ordering | $m_1$ | $K_2$ (Point) |
//! | Quaternion | 4 | Commutativity | $m_2$ | $K_3$ (Edge) |
//! | Octonion | 8 | Associativity | $m_3$ | $K_4$ (Pentagon) |
//! | Sedenion | 16 | Alternativity | $m_4$ | $K_5$ (14-face) |
//!
//! ## 3. Physical Implication: String Theory and 10D
//!
//! The octonionic associator satisfies the pentagon identity ($K_4$). In
//! supersymmetric Yang-Mills theory, the "3-Psi's rule" required for
//! consistency is mathematically equivalent to this pentagon identity.
//!
//! This provides the algebraic foundation for why superstring theory
//! requires **10 dimensions** ($8 + 2$ target space / worldsheet).
//!
//! ## 4. The 16-inator ($F_4$): Sedenion Chaos
//!
//! In sedenions, the pentagon identity fails. This "16-inator" residual
//! is implemented in `stasheff.rs` as the `pentagon_residual`.
//!
//! The non-zero mean residual across 5-tuples of sedenion units signifies the
//! transition from coherent non-associativity (Octonions) to "Algebraic Chaos"
//! (Sedenions), marking the point where standard field theories must give way
//! to the "topological friction" model.
