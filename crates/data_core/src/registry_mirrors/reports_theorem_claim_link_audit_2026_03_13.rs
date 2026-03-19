//! # Theorem Claim Link Audit
//!
//! Date: 2026-03-13
//!
//! Scope: `docs/THEOREMS.md`, `proofs/_RocqProject`, `registry/claims.toml`, and the canonical SQLite control plane at `registry/canonical/control_plane.sqlite3`.
//!
//! ## Summary
//!
//! - Indexed theorem rows: 144
//! - Theorem rows with linked claims after normalization/backfill: 128
//! - Remaining unlinked theorem rows: 16
//!
//! This audit closed the clear normalization gaps where theorem stems encode a claim number but do not exactly match the claim ID format:
//!
//! - `C1313_Thesis42Arithmetic` now links to `C-1313`
//! - `C1140_PathionQuantizedGap` now links to `C-1140`
//! - `C1140b_PathionGap_6_10` now links to `C-1140`
//! - `C1140c_PathionGap_11_15` now links to `C-1140`
//! - `C878_ImbalanceAttractor` now links to `C-878`
//! - `C932_OrthoplexThawing` now links to `C-932`
//! - `C910_Right_e0` .. `C910_Right_e7` now link to `C-910`
//!
//! The residual unlinked set appears to be a mix of:
//!
//! - proof artifacts without a corresponding claim row in `registry/claims.toml`
//! - helper lemmas or theorem families using nonstandard IDs that do not map cleanly to an existing `C-XXXX` claim
//!
//! ## Residual Unlinked Theorem Rows
//!
//! - `C1007_CDPropertyLoss`
//! - `C958_ZDGraphTopology`
//! - `C958b_ZDAdjacencyAnalytical`
//! - `C959_CHSHClassicalBound`
//! - `C993_CarlsonBranchFree`
//! - `C999_PathionEntropyBound`
//! - `C_ConjugateInvolution`
//! - `C_NormConjugate`
//! - `C_OctConjInvolution`
//! - `C_OverImbalancedSign`
//! - `C_QIBoundNegative`
//! - `C_QITauScaling`
//! - `C_SedConjInvolution`
//! - `C_TraceTracefreeVanishes`
//! - `C_WECImpliesNEC`
//! - `C_WarpEnergyNonpositive`
//!
//! ## Next Step
//!
//! Add a verifier that flags theorem rows with no linked claims when they look claim-shaped, while allowing explicitly justified proof artifacts that intentionally have no claim row.
//!
