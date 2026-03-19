//! # C070 NANOGrav Spectrum Match Sources
//!
//! ## Claim
//!
//! `C-070`: "CD associator power spectrum shape matches NANOGrav GW background."
//!
//! ## Official Sources
//!
//! 1. NANOGrav 15-year SMBHB background program page:
//! >  - https://nanograv.org/15yr/SMBHB
//! 2. NANOGrav 15-year free-spectrum paper extract tracked in-repo:
//! >  - `data/papers/documents_extracted/arxiv-2306-16213-agazie-et-al-2023-nanograv-15yr-gwb/paper.toml`
//! 3. KDE free-spectrum archive URL tracked by the repo:
//! >  - https://zenodo.org/api/records/10344086/files/NANOGrav15yr_KDE-FreeSpectra_v1.1.0.zip/content
//!
//! ## Local Reproducible Sources
//!
//! 1. Cached official surfaces and KDE archive:
//! >  - `data/external/nanograv_15yr/smbhb.html`
//! >  - `data/external/nanograv_15yr/record_10344086.json`
//! >  - `data/external/nanograv_15yr_kde.zip`
//! >  - `data/external/nanograv_15yr/kde_contents/ceffyl_data/README.md`
//! 2. Checked-in free-spectrum CSV:
//! >  - `data/external/nanograv_15yr_freespectrum.csv`
//! 3. Rust provider/extractor surface:
//! >  - `crates/data_core/src/catalogs/nanograv.rs`
//! 4. Surface audit and CSV repair lane:
//! >  - `crates/gororoba_cli_data/src/bin/pdg_nanograv_surface_audit.rs`
//! >  - `reports/pdg_nanograv_surface_audit.toml`
//! 5. Deterministic shape-baseline audit:
//! >  - `crates/gororoba_cli_data/src/bin/cd_pattern_baseline_audit.rs`
//! >  - `data/output/claims_falsification/cd_pattern_baseline_audit.toml`
//!
//! ## Scope Note
//!
//! The legacy shape-match lane is no longer accepted on rank correlation alone. The current standard is:
//!
//! - compare against the checked-in NANOGrav free-spectrum medians,
//! - derive the Cayley-Dickson curve exactly from local algebra code, and
//! - test whether the observed Frechet distance is exceptional relative to random monotone nulls and simple template families.
//!
//! If the CD curve is not more distinctive than those null baselines, the resemblance is treated as methodology-insufficient rather than evidence of a physical match.
//!
