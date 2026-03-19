//! # Heliosphere Sparse Policy Mainline Summary
//!
//! ## Summary
//!
//! The promoted mainline sparse-policy candidate for the `modern2020` reference cube was
//! `mission_quiet|full`.
//!
//! This candidate improved the supervised `modern2020` sparse-policy lane relative to the
//! current invariant-only comparator while staying under the hard `12 GiB` projected
//! `1024^3` sparse BF16 A-A budget across the evaluated split seeds.
//!
//! The promotion did **not** survive the full cross-cube gate, so no claim or insight
//! uplift was performed from this run.
//!
//! One technical correction matters here: an earlier draft of this result showed
//! `imap2026` and `inner1976` exploding toward `~70 GiB`. That was a code-path bug
//! caused by retraining the sparse policy on unlabeled target cubes. The corrected
//! mainline bin now trains on the labeled `modern2020` reference cube and transfer-applies
//! that fixed policy to the other cubes.
//!
//! ## modern2020 Improvement
//!
//! Reference cube:
//! - `reports/heliosphere_feature_cube_modern2020_2026-03-15.csv`
//!
//! Comparator policy:
//! - `mission_product_quiet|invariants_only`
//!
//! Promoted candidate:
//! - `mission_quiet|full`
//!
//! Three-seed aggregate comparison (`split_seed = 0, 1, 2`):
//!
//! | Metric | Comparator | Promoted |
//! | --- | ---: | ---: |
//! | Mean active fraction | `0.157291` | `0.140466` |
//! | Mean event-label recall | `0.488584` | `0.554033` |
//! | Mean event-label precision | `0.028797` | `0.037427` |
//! | Mean projected GiB | `11.007034` | `9.832629` |
//! | Max projected GiB | `11.427865` | `10.962023` |
//! | Mean median lead time hours | `6.0` | `10.166667` |
//!
//! So for `modern2020`, the promoted candidate was better on all of the quantities that
//! matter here:
//! - lower projected memory
//! - lower active fraction
//! - higher recall
//! - higher precision
//! - longer median lead time
//!
//! ## Why It Was Not Promoted Repo-Wide
//!
//! The cross-cube generalization gate still failed.
//!
//! On the unsupervised stress cubes:
//! - `imap2026`: the corrected transfer-applied `mission_quiet|full` path still failed at
//!   `33.939406 GiB` mean projected memory and `37.474756 GiB` max projected memory,
//!   versus the invariant comparator at `31.582510 GiB` mean and `34.645755 GiB` max
//! - `inner1976`: the corrected transfer-applied `mission_quiet|full` path failed at
//!   `26.687015 GiB` mean projected memory and `29.175301 GiB` max, versus the invariant
//!   comparator at `25.437876 GiB` mean and `28.391253 GiB` max
//!
//! Because of that:
//! - `promotion_survives_all_cubes = false`
//! - `registry_update_performed = false`
//!
//! ## Interpretation
//!
//! The useful result is narrower than a full promotion, but it is still real:
//!
//! - `mission_quiet|full` is a better sparse-policy candidate for the labeled
//!   `modern2020` lane than the current invariant-only comparator
//! - that gain is not yet stable enough to declare a general cross-cube mainline winner
//! - the remaining blocker is now a real transfer-budget problem, not a broken unlabeled
//!   retraining path
//!
//! This means the next iteration should treat `modern2020` as a confirmed local success
//! case and `imap2026` / `inner1976` as the blockers that still need a more robust policy
//! design.
//!
