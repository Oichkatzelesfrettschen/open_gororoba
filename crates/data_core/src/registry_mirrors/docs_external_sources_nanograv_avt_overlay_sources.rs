//! # NANOGrav AVT Overlay Sources
//!
//! ## Scope
//!
//! This source index supports the exploratory `nanograv-avt-filter` lane, which audits
//! whether a sightline-static Cayley-Dickson / AVT field can explain part of the
//! released NANOGrav timing-residual structure.
//!
//! This is not a reopening of `C-070`. `C-070` remains the closed
//! free-spectrum shape-match lane. The AVT overlay is a distinct residual-network /
//! residual-whitening prototype and must be judged by its own null controls.
//!
//! ## Primary Sources
//!
//! 1. NANOGrav official data page:
//! >  - https://nanograv.org/science/data
//! 2. NANOGrav 15-year timing release:
//! >  - https://zenodo.org/records/16051178
//! 3. NANOGrav 15-year observations and timing paper:
//! >  - https://doi.org/10.3847/2041-8213/acda9a
//! 4. NANOGrav 15-year GWB evidence paper:
//! >  - https://doi.org/10.3847/2041-8213/acdac6
//! 5. PINT fitter/model guidance:
//! >  - https://nanograv-pint.readthedocs.io/en/latest/examples/understanding_fitters.html
//! 6. de Marrais pathion/chingon terminology and higher-dimensional ZD context:
//! >  - https://arxiv.org/abs/math/0207003
//! 7. Reggiani sedenion zero-divisor geometry:
//! >  - https://arxiv.org/abs/2411.18881
//!
//! ## Local Reproducible Surfaces
//!
//! 1. Timing-release parser and release products:
//! >  - `crates/gororoba_cli_data/src/nanograv_timing.rs`
//! >  - `reports/nanograv_15yr_timing_inventory.toml`
//! >  - `reports/nanograv_15yr_propagation_audit.toml`
//! 2. AVT prototype lane:
//! >  - `crates/gororoba_cli_data/src/bin/nanograv_avt_filter.rs`
//! >  - `data/csv/nanograv_avt_whitening_sweep.csv`
//! >  - `reports/nanograv_avt_filter.toml`
//! 3. TensorAVT implementation:
//! >  - `crates/gororoba_algebra/src/gpu/tensor_avt/mod.rs`
//! >  - `crates/gororoba_algebra/src/gpu/tensor_avt/tests.rs`
//! 4. Lattice/codebook surfaces relevant to projection hypotheses:
//! >  - `crates/algebra_experimental/src/cd_external.rs`
//! >  - `registry/hypercomplex_taxonomy.toml`
//!
//! ## Scope Notes
//!
//! - The current AVT lane applies a single scalar field value per pulsar sightline.
//! That can shift per-pulsar means, but it cannot whiten centered intra-pulsar
//! scatter unless the model becomes time-dependent or pair-dependent.
//! - `C-458` verifies Lambda/codebook parity properties; it does not by itself
//! establish a physically correct 3D-sky to 1024D projection.
//! - `C-1113` is the live high-dimensional sampled-AVT claim in this checkout.
//! The off-registry `C-1645` identifier is not a current canonical claim ID here.
//! - `C-075` still needs recertification against live Rust artifacts before any
//! pathion-spectrum overlay claim should be strengthened.
//!
