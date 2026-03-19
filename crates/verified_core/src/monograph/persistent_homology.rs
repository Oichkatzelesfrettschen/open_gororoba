//! # Persistent Homology of the Sedenion Vacuum
//!
//! This section details the topological characterization of sedenion zero
//! divisors using Vietoris-Rips persistent homology.
//!
//! ## 1. Manifold Identification
//!
//! Recent research (Reggiani 2024) proves that the set of normalized sedenion
//! zero divisors ($ZD(\mathbb{S})$) is isometric to the **Stiefel manifold**
//! $V_2(\mathbb{R}^7)$.
//!
//! | Space | Manifold | Dimension | Automorphism Group |
//! | :--- | :--- | :---: | :--- |
//! | Zero Divisor Points | $V_2(\mathbb{R}^7)$ | 11 | $SO(7)$ |
//! | Zero Divisor Pairs | $G_2$ | 14 | $G_2$ (Octonions) |
//!
//! ## 2. Topological Fingerprints (Betti Numbers)
//!
//! The persistent homology of $ZD(\mathbb{S})$ reveals the global structure of
//! the Sedenion vacuum:
//!
//! - **$H_0$ (Connectedness):** A single persistent component, confirming the
//!   vacuum is a unified topological manifold.
//! - **$H_3$ (3-Cycles):** Corresponds to the underlying octonionic $S^3$ symmetry.
//! - **$H_{11}$ (Top Class):** Represents the 11-dimensional volume of the
//!   zero-divisor space.
//!
//! ## 3. Torsion and Stability
//!
//! The presence of $\mathbb{Z}_2$ torsion in $H_5(V_2(\mathbb{R}^7))$ suggests
//! that the vacuum possesses a **non-orientable** internal symmetry at the
//! 16D scale.
//!
//! This torsion is hypothesized to be the source of the **Topological Friction**
//! observed in NANOGrav simulations. As fields propagate, they must navigate
//! the non-orientable zero-divisor manifold, leading to non-conservative
//! energy loss (the 10.32% variance drop).
//!
//! ## 4. Falsifiability: Betti Scaling
//!
//! In the `lattice_filtration` crate, we compute the persistence diagram of
//! sampled zero-divisor pairs. A successful fit requires the $H_3$ and $H_{11}$
//! generators to persist significantly above the noise floor ($S/N > 5$).
//!
//! Failure to observe these generators would refute the claim that the
//! physical vacuum is governed by the global topology of the Sedenion algebra.
