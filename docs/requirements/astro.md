<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# Requirements: Astronomy Data Fetching

Astro scripts depend on `astroquery` and, for some workflows, `gwpy`.

```ignore
make install-astro
```

Rust-based unified fetcher for cosmology, astro, and geophysical pillars:

```ignore
cargo run -p gororoba_cli_data --bin fetch-datasets -- --list
cargo run -p gororoba_cli_data --bin fetch-datasets -- --all --skip-existing --output-dir data/external
```

If a dataset is missing, fetch it explicitly and record provenance in:

- `data/external/PROVENANCE.local.json`
- `docs/BIBLIOGRAPHY.md`
- `docs/external_sources/DATASET_MANIFEST.md`

## Provenance governance checklist

```ignore
cargo run -p gororoba_cli_data --bin fetch-datasets -- --all --skip-existing --output-dir data/external
cargo run -p gororoba_cli_data --bin hepdata-refresh -- --dirs alice_pbpb_raa,cms_oo_raa
cargo run -p gororoba_cli_data --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json
cargo run -p gororoba_cli_data --bin external-redownload-audit -- --execute true --out reports/external_redownload_audit_YYYY-MM-DD.toml --backend-order wget,curl,fetch
cargo run -p gororoba_cli_data --bin external-blocked-burndown -- --out reports/external_blocked_burndown_YYYY-MM-DD.toml
cargo run -p gororoba_cli_data --bin external-blocked-retry-ledger -- --seed-missing true --status seeded --phase governance_contract --note "Seed blocked_action_plan ledger rows"
cargo run -p gororoba_cli_data --bin data-origin-audit -- --fail-on-strict-unknown
cargo run -p gororoba_cli_data --bin data-governance-gate --
cargo run -p gororoba_cli_data --bin data-semantic-validate --
cargo run -p gororoba_cli_data --bin data-semantic-validate -- --fail-on-unverifiable true
```
