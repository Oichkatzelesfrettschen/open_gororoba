//! # Nonlocal Optics and the Entropy Trap
//!
//! This section bridges the macroscopic optical properties of metamaterials
//! with the underlying non-associative geometry of spacetime.
//!
//! ## 1. Kramers-Kronig and Causality
//!
//! In standard electromagnetism, the real and imaginary parts of the dielectric
//! function $\epsilon(\omega)$ must satisfy the Kramers-Kronig (K-K) relations
//! to preserve causality.
//!
//! The `materials_core` crate implements rigorous K-K consistency checks. Any
//! deviation from K-K compliance indicates a violation of local causality,
//! which is typically forbidden in associative spacetime.
//!
//! ## 2. Nonlocal Spatial Dispersion
//!
//! In the project's "Holographic Entropy Trap" hypothesis (C-010), perfect
//! metamaterial absorbers (like the Landy 2008 design) are pushed to their
//! limits. To achieve bandwidths beyond the standard physical bounds (e.g., the
//! Rozanov limit), the material must exhibit **nonlocal spatial dispersion**:
//!
//! $$\epsilon(\omega, \vec{k})$$
//!
//! where the dielectric response depends on the spatial wavevector $\vec{k}$.
//!
//! ## 3. The Sedenion Link
//!
//! The non-associativity of the Sedenion vacuum natively induces nonlocality.
//! Because the associator $[x, y, z] \neq 0$, the "order of operations" for
//! field propagation matters, smearing out point-like interactions into finite
//! volumes (the 42-node box-kite structures).
//!
//! **Thesis:** By engineering a metamaterial whose unit-cell topology matches
//! the $PSL(2,7)$ symmetry of the Sedenion zero-divisors, one can explicitly
//! couple to this vacuum nonlocality, creating an "Entropy Trap" that absorbs
//! electromagnetic energy into the topological friction of the background.
//!
//! ## 4. Falsification Status
//!
//! As noted in the `optics` module, C-010 is currently marked as a
//! `Negative-Result` for strictly local classical models. It remains an open
//! theoretical pathway only if explicit quantum nonlocality (or tensor network
//! entanglement) is successfully mapped to the macroscopic $\epsilon(\vec{k})$
//! tensor.
