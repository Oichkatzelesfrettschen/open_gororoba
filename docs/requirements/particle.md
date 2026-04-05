<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# Requirements: Particle Data Workflows

Particle workflows are optional and use Python extras:

```ignore
make install-particle
```

This installs support for:

- `uproot`
- `awkward`
- `vector`

Use this module only for offline analyses against cached or open data snapshots.
Do not add network calls to tests. Record provenance for imported snapshots under
`data/external/`.

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
