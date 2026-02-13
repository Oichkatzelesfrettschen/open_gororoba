<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

# Requirements: Gororoba Studio (Rust Web App)

`gororoba-studio` is a Rust-native web app lane that serves interactive UI assets
and runs thesis pipelines through HTTP endpoints.

Run:

```bash
make studio-run
```

Verify:

```bash
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
- Cargo workspace dependencies (`axum`, `tokio`)
- Existing thesis pipeline crates (`gororoba_engine`, `vacuum_frustration`, `lbm_core`, `neural_homotopy`, `lattice_filtration`)

Notes:
- Designed for Linux/BSD/Windows/macOS host execution immediately.
- Android/iOS deployment should wrap this surface using a mobile shell/webview lane after API stabilization.
- Keep this app in warnings-as-errors mode (`cargo clippy ... -D warnings`).
- Benchmark mode provides aggregate timing/metric stats for repeat runs.
- Reproducibility mode validates metric drift tolerance and gate consistency.
