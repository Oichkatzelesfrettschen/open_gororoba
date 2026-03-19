//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/docs_root_narratives.toml -->
//!
//! # Physical Claim Boundaries
//!
//! This document records the repo's evidence-first boundary conditions for the
//! four physical claim families touched by the thesis bundle. The goal is to keep
//! the supported algebra, model code, literature anchors, and unsupported
//! interpretations separate.
//!
//! ## Direct physical antimatter claims
//!
//! Status:
//! - physical claim unsupported
//! - model claim supported only as algebraic or condensed-matter analog work
//!
//! What is supported:
//! - `crates/algebra_experimental/src/majorana_braiding.rs` computes braid,
//!   fusion, and complex-time friction invariants in a Cayley-Dickson model.
//! - The supported interpretation is structural isomorphism to Ising-anyon or
//!   Majorana-zero-mode rules, not a particle-antimatter simulation.
//!
//! Primary source anchors:
//! - Das Sarma, Freedman, Nayak (2015), `https://doi.org/10.1038/npjqi.2015.1`
//! - Ivanov (2001), `https://doi.org/10.1103/PhysRevLett.86.268`
//! - Kitaev (2001), `https://doi.org/10.1070/1063-7869/44/10S/S29`
//!
//! Boundary:
//! - A physical antimatter claim would require a Hamiltonian, Hilbert-space
//!   dynamics, decoherence model, and an experimentally relevant observable map.
//! - The current repo does not implement those layers. The theorem scope is
//!   algebra only.
//!
//! ## Metamaterial fabrication success
//!
//! Status:
//! - exact fabricated 42-assessor topology unsupported
//! - non-local design-stage recovery supported
//!
//! What is supported:
//! - `C-010` remains a closed local negative result.
//! - `materials_core::nonlocal_metamaterial` and the thesis bundle support a
//!   non-local recovery lane that projects the masked assessor topology through
//!   synthetic coupling surrogates and benchmarks those against connected local
//!   baselines.
//!
//! Primary source anchors:
//! - Wang et al. (2020), `https://doi.org/10.1038/s41467-020-15940-3`
//! - Yuan et al. (2017), `https://arxiv.org/abs/1710.01373`
//! - Dutt et al. (2022), `https://doi.org/10.1038/s41467-022-31140-7`
//! - Chen et al. (2023), `https://doi.org/10.1002/adma.202209988`
//!
//! Boundary:
//! - The repo supports literature-backed analog platforms and deterministic gate
//!   checks on surrogate calibrations.
//! - It does not yet support the stronger claim that an exact masked
//!   42-assessor topology has been fabricated and measured.
//!
//! ## Dark matter halo physics from the algebra
//!
//! Status:
//! - exact model-law recovery supported
//! - observational forecast lane supported
//! - dark matter microphysics claim unsupported
//!
//! What is supported:
//! - `crates/cosmology_core/src/harmonic_halos.rs` defines a harmonic modulation
//!   of standard NFW circular velocities.
//! - At `alpha_zd = 0`, the model recovers the NFW baseline exactly.
//! - The thesis bundle reports forecast-style detectability thresholds and keeps
//!   SKAO in the future-observability lane.
//!
//! Primary source anchors:
//! - Navarro, Frenk, White (1997), `https://doi.org/10.1086/304888`
//! - Li et al. (2020) SPARC fits, `https://doi.org/10.3847/1538-4365/ab700e`
//! - SKAO HI Galaxy Science, `https://www.skao.int/en/science-users/118/hi-galaxy-science`
//!
//! Boundary:
//! - The algebra-to-dark-matter-microphysics interpretation is not established.
//! - Current repo evidence supports exact baseline recovery, a falsifiable
//!   modulation law, and forecast tables, not a derived particle model.
//!
//! ## Gravastar bridge claims
//!
//! Status:
//! - physical bridge obstructed
//! - bridge-law model under assumptions supported
//!
//! What is supported:
//! - `crates/cosmology_core/src/homotopy_bridge.rs` defines an explicit linear
//!   bridge law `lambda = coupling * obstruction_norm`.
//! - The thesis bundle now reports that lane as a model-under-assumptions audit
//!   and exports `gravastar_bridge_model.csv`.
//!
//! Primary source anchors:
//! - Mazur and Mottola (2001), `https://arxiv.org/abs/gr-qc/0109035`
//! - Mazur and Mottola (2004), `https://doi.org/10.1073/pnas.0402717101`
//! - Visser and Wiltshire (2004), `https://arxiv.org/abs/gr-qc/0310107`
//! - Bowers and Liang (1974), `https://doi.org/10.1086/152638`
//!
//! Boundary:
//! - `C-011` remains obstructed because the repo does not derive a stress-energy
//!   map from the algebra into GR.
//! - The internal linear bridge law can still be studied and formalized as an
//!   explicit assumption, but that is not a derivation of physical gravastars
//!   from the algebra.
//!
//! ## Rocq scope
//!
//! The current Rocq formalization for this evidence lane is intentionally narrow:
//!
//! - `C1313_Thesis42Arithmetic.v` formalizes the discrete arithmetic scaffold.
//! - `C1363_HarmonicHaloExactRecovery.v` formalizes exact `alpha_zd = 0` recovery
//!   of the harmonic-halo model law.
//! - `C1364_HomotopyBridgeLaw.v` formalizes consequences of the linear homotopy
//!   bridge law under explicit assumptions.
//!
//! Theorems are not used to certify fabrication success, observational detection,
//! antimatter physics, or a derived gravastar equivalence.
//!
