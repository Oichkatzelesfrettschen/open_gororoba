# Claims: tensor-networks

Source: docs/claims/CLAIMS_DOMAIN_MAP.csv + docs/CLAIMS_EVIDENCE_MATRIX.md

Count: 3

- Hypothesis C-009 (**Not supported** (rejected (rejected for brickwork circuit (Phase 2A fit complete))), 2026-01-30): Tensor-network experiment exhibits entropy scaling `S ~ log(L) + L^{0.5}`.
  - Where stated: `docs/archive/RESEARCH_STATUS_FINAL.md`, `docs/external_sources/TENSOR_NETWORK_SOURCES.md`, `crates/quantum_core/src/tensor_networks.rs`, `src/scripts/measure/measure_tensor_network_entropy.py`, `src/scripts/measure/measure_tensor_network_entropy_scaling.py`, `src/scripts/measure/measure_tensor_network_entropy_decision.py`, `src/scripts/analysis/tensor_entropy_multi_system_fit.py`, `data/csv/tensor_network_entropy_metrics.csv`, `data/csv/tensor_network_entropy_scaling.csv`, `data/csv/tensor_network_entropy_decision.csv`, `data/csv/tensor_entropy_scaling_fit.csv`, `tests/test_tensor_network_entropy_tiny.py`, `tests/test_tensor_network_entropy_scaling.py`, `tests/test_tensor_network_entropy_decision.py`, `tests/test_tensor_entropy_multi_system_fit.py`, `docs/preregistered/TENSOR_ENTROPY_SCALING.md`
- Hypothesis C-052 (**Verified** (Simulation), 2026-01-31): MERA (Multi-scale Entanglement Renormalization) circuit produces logarithmic entropy scaling S ~ log(L).
  - Where stated: `src/scripts/analysis/verify_phase_3_tasks.py`
- Hypothesis C-406 (**Partially verified** (method), 2026-02-02): TSCP/box-kite sky mapping must be invariant under the relevant algebraic symmetries (or else explicitly enumerate tested embeddings) to avoid "picked the embedding that worked".
  - Where stated: `docs/convos/CONVOS_CLAIMS_INBOX.md`, `docs/external_sources/TSCP_METHOD_SOURCES.md`, `docs/preregistered/TSCP_SKY_ALIGNMENT.md`, `crates/algebra_core/src/boxkites.rs`, `tests/test_tscp_alignment_offline.py`, `tests/test_tscp_embedding_sweep.py`
