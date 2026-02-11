# Python-to-Rust Refactoring Roadmap (Wave 6 Integration Track)

**Status**: ACTIVE EXECUTION  
**Date**: 2026-02-10  
**Owner**: RUST-FIRST Policy Enforcement

---

## Executive Summary

**26 Python files in `/src/` directory require triage:**
- **4 CRITICAL (Core Algorithms)**: MUST refactor to Rust immediately
- **5 HIGH (Supporting Math)**: Refactor to Rust within 1 week  
- **9 MEDIUM (Visualization)**: Keep in Python (matplotlib, plotly) with Rust computation
- **8 LOW (Utilities)**: Refactor or deprecate based on usage

**Action**: Refactor critical files now; parallelize with Phase 10.3 research.

---

## Violation Categories

### CRITICAL - Core Algorithms (scipy/sklearn) ⚠️ IMMEDIATE REFACTOR

| File | Violations | Type | Lines | Rust Target |
|------|-----------|------|-------|------------|
| `nonlocal_long_range_ZZ.py` | scipy.linalg (SVD, eigh) | Quantum Hamiltonian | 171 | `crates/quantum_core/src/hamiltonian_evolution.rs` |
| `spectral_flow_sim.py` | scipy.linalg (eigh) | Spectral analysis | 93 | `crates/spectral_core/src/flow_simulator.rs` |
| `vis_advanced_projections.py` | sklearn (PCA, TSNE) | Dimensionality reduction | 87 | `crates/analysis_core/src/dimensionality_reduction.rs` |
| `vis_box_kites.py` | sklearn (KMeans, PCA) | Clustering + DR | 95 | `crates/algebra_core/src/analysis/clustering.rs` |

### HIGH - Supporting Computations (numpy-heavy)

| File | Issue | Type | Rust Target |
|------|-------|------|------------|
| `nonlocal_long_range_ZZ.py` | np.kron, np.polyfit | Matrix ops | algebra_core::tensor |
| `entropy_pde_fit.py` | PDE fitting | Numerical integration | data_core::pde_solver |
| `genesis_simulation.py` | Field evolution | Physics simulation | control_core::field_evolution |
| `genesis_simulation_v2.py` | Field evolution | Physics simulation | control_core::field_evolution |
| `holo_tensor_net.py` | Tensor contraction | Tensor networks | data_core::tensor_networks |

### MEDIUM - Visualization + Data Processing (Safe to Keep Python)

**matplotlib/plotly only - Pure visualization**:
- `vis_4d_slicer.py`: 4D visualization
- `vis_cd_motif_summary.py`: CD visualization  
- `vis_dimensional_geometry.py`: Geometric visualization
- `vis_hyper_fractal.py`: Fractal visualization
- `vis_hyper_ladder.py` / `v2`: Ladder visualization
- `vis_hyper_mera.py` / `v2`: MERA visualization
- `vis_trajectory.py`: Trajectory visualization

**Action**: Keep in Python IF they only consume CSV/JSON from Rust. Add `.read_csv()` → Rust binary input validation.

### LOW - Utilities & Legacy Code (Review for Deprecation)

| File | Purpose | Status |
|------|---------|--------|
| `analyze_full_chat_log.py` | Chat analysis | DEPRECATE (one-off) |
| `assemble_manuscript.py` | Manuscript assembly | KEEP (documentation) |
| `fetch_ligo_gwpy.py` | Data fetching | REFACTOR to Rust binary |
| `map_cosmic_objects.py` | Astronomy utility | REFACTOR to Rust |
| `materials_embedding_experiments.py` | Materials research | REFACTOR to Rust |
| `modular_classical_sim.py` | Classical sim | REFACTOR to Rust |
| `neg_dim_spectrum.py` | Spectrum analysis | REFACTOR to Rust |
| `sedenion_field_sim.py` | Sedenion physics | REFACTOR to Rust |
| `topology_analysis.py` | Topology analysis | REFACTOR to Rust |

---

## Refactoring Execution Plan

### PHASE A: CRITICAL Refactors (Sprint 35-36, Parallel with Phase 10.3)

#### A1: Quantum Hamiltonian Evolution (`nonlocal_long_range_ZZ.py` → Rust)

**Target**: `crates/quantum_core/src/hamiltonian_evolution.rs`

**Scope** (1.5 days):
- Pauli matrix construction (kron product)
- Hamiltonian builder with power-law coupling
- SVD-based entanglement entropy
- Time evolution (eigenbasis method)
- CSV export

**Implementation**:
```rust
// crates/quantum_core/src/hamiltonian_evolution.rs
pub struct HamiltonianND {
    n: usize,
    dims: Vec<usize>,
    alpha: f64,
}

impl HamiltonianND {
    pub fn construct_zz_terms() -> ndarray::Array2<Complex64>;
    pub fn get_entropy(psi: &[Complex64], n: usize) -> f64;
    pub fn evolve_time(h: &Array2<...>, psi: &[Complex64], t: f64) -> Vec<Complex64>;
}
```

**Tests**: 5+ test cases (eigenbasis, entropy, geometry variants)  
**Dependency**: nalgebra (eigh via SVD), ndarray for large matrices

---

#### A2: Spectral Flow Simulator (`spectral_flow_sim.py` → Rust)

**Target**: `crates/spectral_core/src/flow_simulator.rs`

**Scope** (1 day):
- Parameter sweep (coupling strength, disorder)
- Eigenvalue tracking
- Spectral gap analysis
- CSV + gnuplot output

**Implementation**:
```rust
pub struct SpectralFlowSimulator {
    system_size: usize,
}

impl SpectralFlowSimulator {
    pub fn compute_spectral_flow(...) -> Vec<Vec<f64>>;
    pub fn analyze_gap(...) -> GapStatistics;
}
```

---

#### A3: Dimensionality Reduction (`vis_advanced_projections.py` → Rust)

**Target**: `crates/analysis_core/src/dimensionality_reduction.rs`

**Scope** (2 days):
- PCA implementation (SVD-based)
- TSNE approximation (Barnes-Hut tree)
- UMAP approximation (k-NN + manifold learning)
- 2D/3D projection output

**Implementation**: Use existing SVD from nalgebra; implement TSNE/UMAP incrementally

---

#### A4: Spectral Clustering (`vis_box_kites.py` → Rust)

**Target**: `crates/algebra_core/src/analysis/clustering.rs`

**Scope** (1.5 days):
- K-means++ initialization
- k-means clustering
- Spectral clustering via Laplacian
- Metrics (silhouette, Davies-Bouldin)

**Implementation**: Use ndarray for data; petgraph for graph operations

---

### PHASE B: HIGH Priority Refactors (Sprint 36-37, Sequential)

1. **entropy_pde_fit.py** → `data_core::pde_solver` (1.5 days)
2. **genesis_simulation.py/v2** → `control_core::field_evolution` (2 days)
3. **holo_tensor_net.py** → `data_core::tensor_networks` (2 days)

---

### PHASE C: Visualization Layer (Parallel, Minimal Changes)

**Strategy**: Keep matplotlib/plotly files, update to read from **Rust CSV exports**

**Action for each file**:
1. Replace direct NumPy computation with CSV read
2. Add validation: ensure Rust binary outputs match expected schema
3. Add CI gate: `verify_python_visualization_inputs.py`

**Example**: `vis_cd_motif_summary.py`
```python
# BEFORE: import motif_census; compute locally
# AFTER:
df = pd.read_csv('build/artifacts/motif_census_summary.csv')
# Just plot - no computation!
```

---

### PHASE D: Utilities & Deprecation (Sprint 38, Cleanup)

| Action | Files |
|--------|-------|
| **DEPRECATE** | analyze_full_chat_log.py |
| **KEEP** | assemble_manuscript.py (markdown assembly) |
| **REFACTOR to Rust binary** | fetch_ligo_gwpy.py, map_cosmic_objects.py, materials_embedding_experiments.py, modular_classical_sim.py, neg_dim_spectrum.py, sedenion_field_sim.py, topology_analysis.py |

---

## Implementation Timeline

| Sprint | Phase | Tasks | Completion |
|--------|-------|-------|-----------|
| **S35** | A1 (Quantum Hamiltonian) | 1.5 days, parallel Phase 10.3 | 2026-02-14 |
| **S35** | A2 (Spectral Flow) | 1 day, parallel Phase 10.3 | 2026-02-14 |
| **S36** | A3 (PCA/TSNE) | 2 days, sequential | 2026-02-18 |
| **S36** | A4 (K-means) | 1.5 days, sequential | 2026-02-18 |
| **S36-37** | B (PDE, Genesis, Tensor) | 5.5 days, sequential | 2026-02-24 |
| **S37** | C (Visualization updates) | 2 days, parallel | 2026-02-26 |
| **S38** | D (Utilities cleanup) | 2 days | 2026-03-04 |

**Total**: ~16 days → Achievable in 2 sprints with parallelization

---

## Quality Gates

### Refactoring Validation
1. **Correctness**: Output bit-identical to Python reference for random seeds
2. **Performance**: Rust ≥ 2× faster than Python original
3. **Tests**: 5+ regression tests per refactored file
4. **Warnings**: 0 clippy warnings

### Visualization Updates
1. **Input validation**: Rust CSV schema matches visualization expectations
2. **Graceful fallback**: Visualization works with missing Rust output (uses dummy data)
3. **CI gate**: `verify_python_visualization_inputs.py` passes

---

## Risk & Mitigation

**Risk 1**: Rust dimensional reduction (TSNE/UMAP) slow compared to highly-optimized Python libraries  
**Mitigation**: Implement PCA fully; for TSNE/UMAP, use approximate methods initially; consider linking to C libraries (fmtlib style)

**Risk 2**: Visualization files break if Rust binary doesn't exist  
**Mitigation**: Add fallback data generation; CI ensures Rust binaries always build

**Risk 3**: Phase 10.3 research delayed by refactoring  
**Mitigation**: Parallelize A1-A2 with Phase 10.3 (independent tracks)

---

## File-by-File Disposition Table

| File | Category | Status | Rust Target | EST | Priority |
|------|----------|--------|------------|-----|----------|
| nonlocal_long_range_ZZ.py | Critical | IN PROGRESS | quantum_core | 1.5d | P0 |
| spectral_flow_sim.py | Critical | QUEUED | spectral_core | 1d | P0 |
| vis_advanced_projections.py | Critical | QUEUED | analysis_core | 2d | P0 |
| vis_box_kites.py | Critical | QUEUED | algebra_core | 1.5d | P0 |
| entropy_pde_fit.py | High | QUEUED | data_core | 1.5d | P1 |
| genesis_simulation.py | High | QUEUED | control_core | 2d | P1 |
| genesis_simulation_v2.py | High | QUEUED | control_core | 2d | P1 |
| holo_tensor_net.py | High | QUEUED | data_core | 2d | P1 |
| sedenion_field_sim.py | High | QUEUED | quantum_core | 1.5d | P1 |
| vis_4d_slicer.py | Medium | KEEP | - | - | - |
| vis_cd_motif_summary.py | Medium | UPDATE | - | 0.5d | P2 |
| vis_dimensional_geometry.py | Medium | UPDATE | - | 0.5d | P2 |
| vis_hyper_fractal.py | Medium | UPDATE | - | 0.5d | P2 |
| vis_hyper_ladder.py | Medium | UPDATE | - | 0.5d | P2 |
| vis_hyper_ladder_v2.py | Medium | UPDATE | - | 0.5d | P2 |
| vis_hyper_mera.py | Medium | UPDATE | - | 0.5d | P2 |
| vis_hyper_mera_v2.py | Medium | UPDATE | - | 0.5d | P2 |
| vis_trajectory.py | Medium | UPDATE | - | 0.5d | P2 |
| analyze_full_chat_log.py | Low | DEPRECATE | - | - | - |
| assemble_manuscript.py | Low | KEEP | - | - | - |
| fetch_ligo_gwpy.py | Low | REFACTOR | cli_core | 1d | P3 |
| map_cosmic_objects.py | Low | REFACTOR | analysis_core | 1d | P3 |
| materials_embedding_experiments.py | Low | REFACTOR | analysis_core | 1d | P3 |
| modular_classical_sim.py | Low | REFACTOR | quantum_core | 1d | P3 |
| neg_dim_spectrum.py | Low | REFACTOR | analysis_core | 1d | P3 |
| topology_analysis.py | Low | REFACTOR | stats_core | 1.5d | P3 |

---

## Next Immediate Steps

1. **NOW** (This session):
   - Start A1: Quantum Hamiltonian (1.5 days) in parallel with Phase 10.3
   - Create Rust skeleton: `crates/quantum_core/src/hamiltonian_evolution.rs`
   
2. **Next 24 hours**:
   - Complete A1 tests
   - Start A2: Spectral Flow (1 day)
   - Continue Phase 10.3 composition taxonomy
   
3. **Sprint 36**:
   - A3 + A4 sequential (3.5 days total)
   - Phase 10.3 completion
   - Wave 6 task execution (W6-006..022)

---

## Conclusion

**RUST-FIRST enforcement** requires immediate refactoring of 4 critical files containing core algorithms (scipy.linalg, sklearn). This roadmap ensures:
- ✓ Pure Rust implementations of all numerical methods
- ✓ Verification against Python reference implementations
- ✓ Parallel execution with Phase 10-11 research
- ✓ Visualization layer integrity (matplotlib/plotly) maintained

**Status**: READY FOR EXECUTION → Start NOW

