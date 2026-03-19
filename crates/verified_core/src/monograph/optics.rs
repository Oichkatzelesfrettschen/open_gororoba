//! # Perfect Metamaterial Absorbers and TMM
//!
//! This section details the implementation of the Landy 2008 Perfect Metamaterial Absorber (PMA).
//!
//! ## 1. Lorentz Effective-Medium Model
//!
//! We model effective permittivity $\epsilon_{eff}$ and permeability $\mu_{eff}$ using
//! the Lorentz oscillator model:
//!
//! $$\epsilon_{eff}(\omega) = \epsilon_{\infty} + \frac{\omega_{pe}^2}{\omega_{0e}^2 - \omega^2 - i\gamma_e\omega}$$
//!
//! Near-unity absorption is achieved when $Z = \sqrt{\mu/\epsilon} \approx Z_0$ at the resonance frequency.
//!
//! ## 2. Transfer Matrix Method (TMM)
//!
//! For a layer of thickness $d$, the transfer matrix $M$ is:
//!
//! $$M = \begin{pmatrix} \cos(nkd) & -\frac{iZ}{Z_0}\sin(nkd) \\ -i\frac{Z_0}{Z}\sin(nkd) & \cos(nkd) \end{pmatrix}$$
//!
//! where $n = \sqrt{\epsilon\mu}$ and $k = \omega/c$.
//!
//! ### Falsification Criteria
//! The "Holographic Entropy Trap" (C-010) hypothesis implies that specific unit-cell
//! couplings based on the sedenion ZD graph could improve bandwidth.
//! This has been marked as `Closed/Negative-Result` in the primary claims registry,
//! as local absorbers were insufficient without explicit non-local coupling.
