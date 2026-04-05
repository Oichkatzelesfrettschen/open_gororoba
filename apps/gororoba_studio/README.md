<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# Requirements: Gororoba Studio (Rust Web App)

`gororoba-studio` is a Rust-native web app lane that serves interactive UI
assets and runs thesis pipelines through HTTP endpoints.

Run:

```ignore
make studio-run
```

Verify:

```ignore
make studio-check
```

Primary API endpoints:

- `GET /api/health`
- `GET /api/pipelines`
- `GET /api/history`
- `POST /api/run/{experiment_id}`
- `POST /api/run-suite`
- `POST /api/benchmark/{experiment_id}`
- `POST /api/reproducibility/{experiment_id}`

Dependencies:

- Rust toolchain matching workspace policy
- Cargo workspace dependencies such as `axum` and `tokio`
- Existing thesis pipeline crates such as `gororoba_engine`, `sign_imbalance`,
  `lbm_core`, `neural_homotopy`, and `lattice_filtration`

Notes:

- Keep this app in warnings-as-errors mode.
- Benchmark mode provides aggregate timing and metric stats for repeat runs.
- Reproducibility mode validates metric drift tolerance and gate consistency.
