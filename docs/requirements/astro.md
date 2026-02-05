# Requirements: Astronomy Data Fetching

Astro scripts depend on `astroquery` and (for some workflows) `gwpy`.

```bash
make install-astro
```

If a dataset is missing, fetch it explicitly and record the provenance in `docs/BIBLIOGRAPHY.md`.

Common entrypoints:
- `src/fetch_observatory_data.py` (MAST, SDSS)
- `src/fetch_pulsars_rigorous.py`, `src/map_cosmic_objects.py` (VizieR)
- `src/fetch_ligo_gwpy.py` (GWpy)
