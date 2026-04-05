<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# Requirements: Heliosphere Data

The heliosphere lane is a mixed Rust-plus-data workflow centered on the
Rust-native external dataset fetchers in `gororoba_cli_data`.

Primary datasets in this lane:

- `Helios 1 Merged Hourly`
- `Helios 2 Merged Hourly`
- `Voyager 1 Merged Hourly (2016)`
- `Voyager 1 CRS Daily Flux (2016)`
- `NASA OMNI2 Solar Wind + IMF (2016)`

Recommended fetch/verify sequence:

```ignore
cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset "Helios 1 Merged Hourly" --skip-existing
cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset "Helios 2 Merged Hourly" --skip-existing
cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset "Voyager 1 Merged Hourly (2016)" --skip-existing
cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset "Voyager 1 CRS Daily Flux (2016)" --skip-existing
cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset "NASA OMNI2 Solar Wind + IMF (2016)" --skip-existing
cargo run -p gororoba_cli_data --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json
cargo run -p gororoba_cli_data --bin data-origin-audit -- --fail-on-strict-unknown
cargo run -p gororoba_cli_data --bin data-governance-gate --
```

Notes:

- Prefer `--skip-existing` for repeatable refreshes.
- Treat provenance and semantic validation as part of the install contract, not
  as an afterthought.
