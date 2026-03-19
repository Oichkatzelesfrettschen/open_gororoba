//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->
//!
//! # Requirements: Materials Datasets
//!
//! For reproducible, keyless materials-science property datasets, prefer sources with clear licensing.
//!
//! Recommended starting points (see `docs/BIBLIOGRAPHY.md` for provenance and citations):
//! - JARVIS-DFT (CC BY 4.0)
//! - OQMD (CC BY 4.0, API/OPTiMaDe)
//! - NOMAD (public archive, CC BY 4.0)
//!
//! This repo provides scripts to download and cache small, test-friendly subsets.
//!
//! ```ignore
//! make install
//! make artifacts-materials
//! ```ignore
//!
//! ## Provenance governance checklist
//!
//! ```ignore
//! cargo run -p gororoba_cli_data --bin fetch-datasets -- --all --skip-existing --output-dir data/external
//! cargo run -p gororoba_cli_data --bin hepdata-refresh -- --dirs alice_pbpb_raa,cms_oo_raa
//! cargo run -p gororoba_cli_data --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json
//! cargo run -p gororoba_cli_data --bin external-redownload-audit -- --execute true --out reports/external_redownload_audit_YYYY-MM-DD.toml --backend-order wget,curl,fetch
//! cargo run -p gororoba_cli_data --bin external-blocked-burndown -- --out reports/external_blocked_burndown_YYYY-MM-DD.toml
//! cargo run -p gororoba_cli_data --bin external-blocked-retry-ledger -- --seed-missing true --status seeded --phase governance_contract --note "Seed blocked_action_plan ledger rows"
//! cargo run -p gororoba_cli_data --bin data-origin-audit -- --fail-on-strict-unknown
//! cargo run -p gororoba_cli_data --bin data-governance-gate --
//! cargo run -p gororoba_cli_data --bin data-semantic-validate --
//! cargo run -p gororoba_cli_data --bin data-semantic-validate -- --fail-on-unverifiable true
//! ```ignore
//!
//! Use this lane after materials refreshes to verify source-of-origin, replayability, and schema validity.
//!
//! References:
//! - JARVIS paper: https://www.nature.com/articles/s41524-020-00440-1
//! - JARVIS-DFT data (Figshare): https://figshare.com/articles/dataset/JARVIS-DFT_Database/6815699
//!
