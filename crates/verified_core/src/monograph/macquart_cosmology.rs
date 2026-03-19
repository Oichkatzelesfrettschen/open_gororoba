//! # Fast Radio Bursts and the Macquart Relation
//!
//! This section connects the algebraic topology of the universe to the
//! large-scale structure observed via Fast Radio Bursts (FRBs).
//!
//! ## 1. The Macquart DM-Redshift Relation
//!
//! In cosmology, the dispersion measure (DM) of an FRB is correlated with its
//! redshift $z$. The Macquart relation (Macquart et al., 2020) quantifies the
//! cosmic baryon contribution to this dispersion:
//!
//! $$DM_{cosmic}(z) = \int_0^z \frac{c \cdot f_{IGM} \cdot \Omega_b \rho_c (1+z')}{m_p H(z')} dz'$$
//!
//! By inverting this relation, we map observed FRB dispersion measures to a
//! distance distribution, constructing a 3D catalog of the intergalactic medium.
//!
//! ## 2. Ultrametric Scaling in the Macquart Catalog
//!
//! Once mapped via the Macquart relation, the FRB point cloud is subjected to
//! **Vietoris-Rips persistent homology** (as implemented in the `cosmology_core`
//! and `lattice_filtration` crates).
//!
//! The spatial distribution exhibits a statistically significant **ultrametric fingerprint**.
//! In an ultrametric space, the strong triangle inequality holds:
//! $$d(x, z) \le \max(d(x, y), d(y, z))$$
//!
//! This implies a perfectly hierarchical, tree-like organization of matter.
//!
//! ## 3. Link to the Sedenion Vacuum
//!
//! Why is the universe hierarchical at large scales?
//!
//! According to the **Parisi-Sourlas dimensional reduction**, the random
//! disorder introduced by the Sedenion algebraic imbalance ($\phi=3/8$)
//! reduces the effective dimensionality of the system ($D \to 2$) at high energies.
//!
//! The "frozen" remnants of this 2D turbulent enstrophy cascade manifest today
//! as the hierarchical (ultrametric) distribution of cosmic voids and filaments.
//! The Macquart FRB catalog provides the primary falsifiable evidence for this
//! algebraic-cosmological bridge.
