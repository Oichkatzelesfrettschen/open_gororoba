//! # Advanced Mathematical Formalism
//!
//! This section deep-dives into the symbolic logic and combinatorial structures
//! underlying the project's algebraic-physical synthesis.
//!
//! ## 1. The Cayley-Dickson Twist Recursion
//!
//! The multiplication of basis elements $e_i e_j = (-1)^{f(i, j)} e_{i \oplus j}$
//! is governed by the recursive bit-level function $f(i, j, n)$.
//!
//! For $i, j < 2^n$, let $h = 2^{n-1}$ be the high-bit threshold:
//!
//! - $f(i, j, n) = f(i, j, n-1)$ if $i, j < h$
//! - $f(i, j, n) = f(j-h, i, n-1)$ if $i < h, j \ge h$
//! - $f(i, j, n) = f(i-h, j, n-1) \oplus [j \ne 0]$ if $i \ge h, j < h$
//! - $f(i, j, n) = f(j-h, i-h, n-1) \oplus [j-h = 0]$ if $i, j \ge h$
//!
//! The "associativity bit" for a triple $(i, j, k)$ is:
//! $$A(i, j, k) = f(i, j) \oplus f(i \oplus j, k) \oplus f(j, k) \oplus f(i, j \oplus k)$$
//!
//! ## 2. The Anti-Diagonal Parity Theorem
//!
//! For 16D sedenions, zero divisors are pairs of diagonal 2-blades $(e_a \pm e_b)$.
//! The interaction is governed by the invariant $\eta$:
//!
//! $$\eta(a, b) = f(\text{lo}_a, \text{hi}_b) \oplus f(\text{hi}_a, \text{lo}_b)$$
//!
//! This invariant characterizes the "twist" introduced by the 4th doubling.
//! A triangle of zero divisors is pure (all same sign edges) iff $\eta$ is constant.
//!
//! ## 3. The 3/8 Ratio: Tang-Tang Unification
//!
//! The **Vacuum Imbalance Attractor $\phi = 3/8$** arises from the unification
//! of the 16 Sedenion basis elements with the 24 generators of the $SU(5)$ Grand
//! Unified Theory (GUT).
//!
//! Total operators: $16 + 24 = 40$.
//! Non-associative imaginary units: $15$.
//!
//! Imbalance Ratio: $\frac{15}{40} = \frac{3}{8} = 0.375$.
//!
//! This ratio matches the theoretical **Weak Mixing Angle** $\sin^2 \theta_W$
//! at the GUT scale, linking the non-associative geometry of the vacuum
//! directly to the strength of the electroweak interaction.
//!
//! **Renormalization Group (RG) Running:**
//! While the algebraic baseline is $3/8 = 0.375$, the observed value at the
//! Z-boson mass scale ($\sim 91 \text{ GeV}$) is $\sim 0.231$. The mapping
//! assumes that the Cayley-Dickson imbalance defines the structure of the
//! unification scale vacuum.
//!
//! ## 4. The Barbero-Immirzi Entropy Bridge
//!
//! The Barbero-Immirzi parameter $\gamma$ fixes the quantum of area in LQG.
//! We derive $\gamma$ from the binary entropy of the imbalance attractor:
//!
//! $$\gamma_{Genesis} = \frac{H(3/8)}{\pi \sqrt{3}} \approx 0.1216$$
//!
//! This value aligns within 1.7% of the Domagala-Lewandowski value ($\gamma \approx 0.1236$),
//! derived from quasinormal mode ringing. This suggests the "ringing" of black holes
//! is an observation of the underlying algebraic imbalance of the sedenion vacuum.
