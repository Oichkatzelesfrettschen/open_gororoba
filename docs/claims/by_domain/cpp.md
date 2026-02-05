# Claims: cpp

Source: docs/claims/CLAIMS_DOMAIN_MAP.csv + docs/CLAIMS_EVIDENCE_MATRIX.md

Count: 2

- Hypothesis C-024 (**Verified** (scaffold; Phase 4), 2026-01-28): C++ acceleration kernels reproduce Python CD multiplication results exactly (within float64 tolerance).
  - Where stated: `cpp/`, `cpp/tests/test_cd_algebra.cpp`, `cpp/benchmarks/bench_cd_multiply.cpp`
- Hypothesis C-432 (**Partially verified** (Ray-trace only; analytic overlay pending), 2026-02-02): Kerr geodesic trajectories computed via the C++ Mino-time integrator can be compared against the analytic Bardeen (1973) shadow boundary; an overlay check is pending.
  - Where stated: `docs/external_sources/OPEN_CLAIMS_SOURCES.md`, `src/scripts/simulation/integrated_warp_geodesic.py`, `data/artifacts/images/kerr_trace_demo.png`
