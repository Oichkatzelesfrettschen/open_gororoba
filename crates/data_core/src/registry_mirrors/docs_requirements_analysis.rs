//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->
//!
//! # Requirements: Analysis (Topology + Embeddings)
//!
//! These scripts use optional dependencies such as `networkx`, `ripser`, and friends.
//!
//! ```texttext
//! make install-analysis
//! ```texttext
//!
//! Notes:
//! - Common entrypoints that require these extras:
//!   - `src/topology_analysis.py` (ripser/persim)
//!   - `src/vis_box_kites.py` (scikit-learn)
//!   - `src/vis_advanced_projections.py` (scikit-learn, networkx)
//!   - `src/holo_tensor_net.py`, `src/vis_hyper_mera*.py` (networkx)
//! - Some scientific wheels may not exist for very new Python versions; if installs fail, use a container (or a Python 3.11/3.12 env) for this module.
//!
//! ## Data governance and provenance
//!
//! Analysis outputs are treated as reproducible build artifacts and must pass Rust-native governance checks:
//!
//! ```texttext
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
//! cargo run -p gororoba_cli_data --bin data-clean -- --scope reproducible --apply
//! ```texttext
//!
//! This sequence verifies origin contracts, replayability, semantic integrity, and clean rebuild behavior for analysis lanes.
//!
