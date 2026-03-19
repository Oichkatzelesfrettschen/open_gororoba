//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/docs_root_narratives.toml -->
//!
//! # E-027 Phase 2 Execution Plan: Grid Upscaling and Lambda Sweep
//!
//! ## Context
//!
//! **Phase 1 Results** (2026-02-11):
//! - Grid: 16^3 = 4,096 cells
//! - Lambda: 5.0 (single point)
//! - p-value: 0.605 (inconclusive)
//! - Channels detected: 1 (full z-slice, not real channels)
//! - **Decision**: DEFERRED to Phase 2 for grid upscaling
//!
//! **Phase 1.4.1 Infrastructure** (2026-02-11):
//! - Lambda sweep [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] executed at **64^3 grid**
//! - Binary: `percolation-experiment` with GPU support (`--use-gpu` flag)
//! - Results: `data/e027/lambda_sweep/gpu_sweep_v2.log`
//! - **Status**: Infrastructure validated, ready for Phase 2 upscaling
//!
//! ## Objectives
//!
//! 1. **Scale to 128^3 grid** (262,144 cells, 64x larger than Phase 1)
//! 2. **Execute full lambda sweep** [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] using existing binary
//! 3. **Determine if coupling exists** at higher resolution or if Thesis 1 requires pivot
//! 4. **If 128^3 inconclusive**: Sample 256^3 grid (16M cells) at 2-3 lambda values
//!
//! ## Phase 2 Week 3 Schedule (3 days)
//!
//! ### Day 1: 128^3 Lambda Sweep (GPU)
//!
//! **Duration**: 4-5 GPU hours
//! **Command**:
//! ```texttext
//! cd /home/eirikr/Github/open_gororoba
//!
//! # Create output directory
//! mkdir -p data/e027/lambda_sweep_phase2_128
//!
//! # Run sweep (8 lambda values)
//! for lambda in 0.1 0.5 1.0 2.0 5.0 10.0 20.0 50.0; do
//!   echo "========================================="
//!   echo "Lambda = $lambda"
//!   echo "========================================="
//!
//!   cargo run --release --bin percolation-experiment -- \
//!     --grid-size 128 \
//!     --lbm-steps 5000 \
//!     --nu-base 0.333 \
//!     --lambda $lambda \
//!     --n-permutations 1000 \
//!     --seed 42 \
//!     --output-dir data/e027/lambda_sweep_phase2_128 \
//!     --use-gpu \
//!     --verbose \
//!     2>&1 | tee data/e027/lambda_sweep_phase2_128/lambda_${lambda}.log
//! done
//! ```texttext
//!
//! **Expected Runtime**:
//! - Per run: 30-40 minutes (128^3 LBM + GPU percolation)
//! - Total: 4-5 hours
//!
//! **Success Criteria**:
//! - All 8 runs complete without OOM errors
//! - VRAM usage < 10GB (12GB available on RTX 4070 Ti)
//! - At least 3 channels detected per run (not single z-slice blob)
//!
//! ### Day 2: Analysis and Decision
//!
//! **Tasks**:
//! 1. Parse all 8 TOML output files from `data/e027/lambda_sweep_phase2_128/`
//! 2. Extract p-values, effect sizes, channel counts
//! 3. Create summary table:
//!    ```texttext
//!    Lambda | p-value | effect_size | n_channels | verdict
//!    -------+---------+-------------+------------+---------
//!    0.1    | ...     | ...         | ...        | ...
//!    0.5    | ...     | ...         | ...        | ...
//!    ...
//!    ```texttext
//!
//! **Decision Matrix**:
//! - **SCENARIO A**: At least 1 lambda has p < 0.05 AND effect_size > 10%
//!   -> **THESIS 1 VALIDATED** at 128^3, document in registry, mark C-657 "Verified"
//!
//! - **SCENARIO B**: All lambda have p > 0.05 BUT effect_size > 10% for some
//!   -> **Grid still insufficient**, escalate to 256^3 sample (Day 3)
//!
//! - **SCENARIO C**: All lambda have effect_size < 5%
//!   -> **Coupling too weak or absent**, PIVOT to alternative validation:
//!     - Option 1: Direct viscosity measurement via shear relaxation time
//!     - Option 2: Thesis 1 remains "Provisional" with STPT-006 as PRIMARY validator
//!
//! ### Day 3: Contingency Actions
//!
//! **If SCENARIO B** (escalate to 256^3):
//! ```texttext
//! # Sample 2 lambda values at 256^3
//! for lambda in 1.0 5.0; do
//!   cargo run --release --bin percolation-experiment -- \
//!     --grid-size 256 \
//!     --lbm-steps 10000 \
//!     --lambda $lambda \
//!     --use-gpu \
//!     --output-dir data/e027/lambda_sweep_phase2_256 \
//!     --verbose
//! done
//! ```texttext
//! **Runtime**: ~2 hours per run, 4 hours total
//! **VRAM**: ~11GB (close to 12GB limit, monitor carefully)
//!
//! **If SCENARIO C** (pivot):
//! 1. Document negative result in `reports/e027_negative_result_2026_02_16.toml`
//! 2. Design alternative experiment (Pivot A: direct viscosity measurement)
//! 3. Update registry:
//!    - C-657 remains "Provisional", note "E-027 exhaustive sweep yields no coupling signal"
//!    - STPT-006 remains PRIMARY validator
//!
//! ## GPU Resource Requirements
//!
//! | Grid Size | Cells    | VRAM (est) | LBM Steps | Runtime/run | Total (8 lambda) |
//! |-----------|----------|------------|-----------|-------------|-------------|
//! | 64^3      | 262K     | 2GB        | 2500      | 10 min      | 1.3 hours   |
//! | 128^3     | 2.1M     | 8GB        | 5000      | 35 min      | 4.7 hours   |
//! | 256^3     | 16.8M    | 11GB       | 10000     | 120 min     | 16 hours    |
//!
//! **Hardware**: RTX 4070 Ti (12GB VRAM, 7680 CUDA cores)
//! **Bottleneck**: 256^3 is close to VRAM limit, may need memory optimizations
//!
//! ## Percolation Algorithm Upgrade (Optional)
//!
//! **Current (Phase 1)**: BFS with threshold = mean + 1.5*std
//! **Issue**: Detects single massive blob instead of multiple channels
//!
//! **Phase 2 Upgrade** (if BFS still insufficient at 128^3):
//! 1. **Union-Find Spanning Cluster Detection**:
//!    - Identify all connected components above threshold
//!    - Count only components that span at least 50% of domain in one direction
//!    - Filter out tiny isolated pockets (< 100 cells)
//!
//! 2. **Hoshen-Kopelman Algorithm**:
//!    - Label all connected regions
//!    - Compute size distribution
//!    - Select spanning clusters only
//!
//! **Implementation**: Add to `sign_imbalance/src/percolation.rs`
//! **Effort**: 2-3 hours (if needed)
//!
//! ## Success Metrics
//!
//! **Phase 2 Complete** if ANY of:
//! 1. At least 1 lambda value yields p < 0.05 AND effect_size > 10% at 128^3 (Thesis 1 validated)
//! 2. Negative result documented with exhaustive sweep 64^3/128^3/256^3 (Thesis 1 refuted or pivoted)
//! 3. Alternative validation path selected (direct viscosity measurement)
//!
//! **Phase 2 Incomplete** if:
//! - GPU OOM errors prevent 128^3 runs (requires memory optimization)
//! - Percolation detection still unreliable (requires algorithm upgrade)
//! - Results ambiguous (effect_size 5-10%, p-values near threshold)
//!
//! ## Rollback Plan
//!
//! If 128^3 runs encounter issues:
//! 1. Diagnose failure (OOM vs algorithm vs data quality)
//! 2. If OOM: Implement memory chunking or downscale to 96^3
//! 3. If algorithm: Upgrade percolation detection first, re-run
//! 4. If data quality: Check APT frustration field generation
//!
//! ## References
//!
//! - Phase 1 E-027 results: `data/e027/e027_results.toml`
//! - 64^3 lambda sweep: `data/e027/lambda_sweep/gpu_sweep_v2.log`
//! - E-027 decision protocol: `reports/e027_decision_protocol.toml`
//! - Thesis 1 claims: registry/claims.toml (C-657, C-658, C-659)
//!
