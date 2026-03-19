//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/docs_root_narratives.toml -->
//!
//! # Thesis 42 Support
//!
//! This document records the evidence-first support posture for the "42 / 1764"
//! thesis lane implemented by `thesis-42-support`.
//!
//! ## Current disposition
//!
//! - Non-local metamaterials:
//! design-stage only. C-010 remains a closed local negative result, and the
//! repo only supports the non-local recovery lane built from the 42-assessor
//! masked topology and the `m3` lift.
//! The default thesis bundle now points at `nonlocal_cable_chen_2023` because
//! it is the strongest current C-010 passing surrogate in the candidate scan.
//! `yig_magnonic_kaman_2026` remains available as an opt-in measured-table
//! comparison row.
//! - Majorana braid friction:
//! algebraic, not physical. The sweep reports Cayley-Dickson associator
//! invariants and complex-time damping, but it does not claim a Hamiltonian
//! Majorana or antimatter simulation.
//! - Dark matter:
//! falsifiable observable lane. The supported bridge is harmonic-halo/NFW
//! observables with exact `alpha_zd = 0` recovery, not a derived `10^-42`
//! coupling constant.
//! - Gravastar stability:
//! physical bridge unsupported. Gravastar solvers remain in `cosmology_core`,
//! and the thesis bundle now reports the current linear bridge-law sweep only as
//! a model-under-assumptions lane. C-011 stays obstructed even when the internal
//! linear law has stable or causal windows.
//!
//! ## Artifacts
//!
//! Running
//!
//! `cargo run --release --bin thesis-42-support -- --output-dir data/evidence/thesis_42_support`
//!
//! writes:
//!
//! - `data/evidence/thesis_42_support/summary.toml`
//! - `data/evidence/thesis_42_support/nonlocal_topologies.csv`
//! - `data/evidence/thesis_42_support/majorana_friction_sweep.csv`
//! - `data/evidence/thesis_42_support/harmonic_halo_reference.csv`
//! - `data/evidence/thesis_42_support/gravastar_bridge_model.csv`
//!
//! The summary file carries the policy labels:
//!
//! - `design_stage_only`
//! - `algebraic_not_physical`
//! - `falsifiable_observable_lane`
//! - `gravastar_bridge_unsupported`
//!
//! The per-lane boundary blocks now also record:
//!
//! - `physical_claim_status`
//! - `model_claim_status`
//! - `theorem_scope_status`
//! - `assumption_surface`
//!
//! and a source appendix for each lane using the primary papers and science-case
//! links that anchor the current evidence posture.
//!
//! ## Formalization scope
//!
//! The Rocq layer now mirrors the exact kernels and explicit bridge laws only:
//!
//! - `proofs/verified/C1313_Thesis42Arithmetic.v`
//! - `proofs/verified/C1363_HarmonicHaloExactRecovery.v`
//! - `proofs/verified/C1364_HomotopyBridgeLaw.v`
//!
//! These proofs intentionally do not formalize fabrication success, antimatter
//! physics, dark matter microphysics, or a derived gravastar-to-GR equivalence.
//! They formalize only:
//!
//! - exact arithmetic identities used by the thesis bundle,
//! - exact `alpha_zd = 0` recovery of the harmonic-halo model law,
//! - explicit consequences of the linear homotopy bridge law under stated
//! positivity assumptions.
//!
//! ## Intended use
//!
//! Use this bundle when a thesis narrative needs a repo-consistent summary of the
//! 42-assessor metamaterial lane, the Majorana friction lane, the harmonic-halo
//! observable lane, and the assumption-labeled gravastar bridge audit without
//! silently promoting speculative claims to verified physics.
//!
