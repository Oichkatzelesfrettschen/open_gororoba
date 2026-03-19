//! <!-- AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT. -->
//! <!-- Source of truth: registry/external_sources.toml -->
//! <!-- Canonical write path: registry/canonical/control_plane.sqlite3 -->
//! <!-- Source label: XS-017 -->
//! <!-- Regenerate with: cargo run -p gororoba_cli_data --bin provenance -- export-external-sources -->
//!
//! # Pioneer and Flyby Anomaly Primary Sources
//!
//! This index governs the external observational inputs used by the Pioneer and
//! Earth-flyby anomaly verification lane. These artifacts are benchmark inputs,
//! not proofs of any proposed explanation.
//!
//! ## 1. Pioneer 10/11 anomaly benchmark
//!
//! - Target:
//!   published anomalous sunward acceleration benchmark for Pioneer 10/11
//! - Primary source:
//!   Anderson et al. (2002), "Study of the anomalous acceleration of Pioneer 10
//!   and 11", Phys. Rev. D 65, 082004
//! - DOI:
//!   `10.1103/PhysRevD.65.082004`
//! - Bibliography:
//!   `BIB-0317`
//! - Governed artifact:
//!   `data/external/pioneer_anomaly/pioneer_anomaly_benchmark.csv`
//! - Current scope:
//!   batch-1 benchmark rows with explicit interval bounds and published combined
//!   acceleration reference values
//! - Important limitation:
//!   this is not a time-resolved Doppler residual series; it is a governed
//!   benchmark table for fit and falsification harnesses
//!
//! ## 2. Earth flyby anomaly event catalog
//!
//! - Target:
//!   observed asymptotic velocity jumps for the Anderson et al. Earth flybys
//! - Primary source:
//!   Anderson et al. (2008), "Anomalous Orbital-Energy Changes Observed during
//!   Spacecraft Flybys of Earth", Phys. Rev. Lett. 100, 091102
//! - DOI:
//!   `10.1103/PhysRevLett.100.091102`
//! - Bibliography:
//!   `BIB-0297`
//! - Governed artifact:
//!   `data/external/flyby_anomaly/anderson_2008_flyby_catalog.csv`
//! - Current scope:
//!   six canonical flyby events mirrored from the paper-backed values already used
//!   by `config::all_flybys()`
//! - Important limitation:
//!   this batch treats the external CSV as the source-of-truth observation table;
//!   internal Rust config values must match it exactly or the audit fails
//!
//! ## 3. Conventional heliosphere environment context
//!
//! - Governed providers:
//!   `OMNI`, `SOHO CELIAS`, `Voyager merged`, `Pioneer merged/encounter`
//! - Role:
//!   provide date-valid plasma and field context at anomaly epochs
//! - Important limitation:
//!   batch-1 uses these providers for environment sampling and benchmark context.
//!   It does not claim a full spacecraft thermal-recoil or engineering-force model.
//!
//! ## 4. Falsification posture
//!
//! - A governed benchmark artifact is not the same thing as a solved anomaly.
//! - Encounter subsets do not satisfy blocked annual provider contracts unless the
//!   annual lane is independently staged and verified.
//! - If the fit harness cannot match the observational benchmarks within declared
//!   tolerances, the current conjecture is recorded as refuted in its present
//!   operational form rather than stretched in prose.
//!
