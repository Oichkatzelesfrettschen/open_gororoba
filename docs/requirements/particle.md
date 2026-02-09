<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

# Requirements: Particle Data Workflows

Particle workflows are optional and use Python extras:

```bash
make install-particle
```

This installs:
- `uproot` for ROOT file reading in pure Python
- `awkward` for columnar event arrays
- `vector` for Lorentz-vector convenience types

Use this module only for offline analyses against cached/open data snapshots.
Do not add network calls to tests. Record provenance for any imported snapshots
under `data/external/` and update `docs/BIBLIOGRAPHY.md`.
