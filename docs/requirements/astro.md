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

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

# Requirements: Astronomy Data Fetching

Astro scripts depend on `astroquery` and (for some workflows) `gwpy`.

```bash
make install-astro
```

Rust-based unified fetcher for cosmology/astro/geophysical pillars:

```bash
cargo run --bin fetch-datasets -- --list
cargo run --bin fetch-datasets -- --all --skip-existing
```

If a dataset is missing, fetch it explicitly and record provenance in:
- `data/external/PROVENANCE.local.json` (`make provenance`)
- `docs/BIBLIOGRAPHY.md`
- `docs/external_sources/DATASET_MANIFEST.md`

Common entrypoints:
- `src/fetch_observatory_data.py` (MAST, SDSS)
- `src/fetch_pulsars_rigorous.py`, `src/map_cosmic_objects.py` (VizieR)
- `src/fetch_ligo_gwpy.py` (GWpy)
