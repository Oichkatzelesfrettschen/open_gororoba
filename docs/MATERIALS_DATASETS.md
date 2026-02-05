# Materials Datasets (Real Properties, Reproducible)

This repo now includes a reproducible pipeline for pulling a **small, redistributable subset** of
real materials properties from **JARVIS-DFT** (via Figshare).

## Fetch a subset

```bash
PYTHONWARNINGS=error ./venv/bin/python3 src/fetch_materials_jarvis_subset.py --n 200 --seed 0
```

Outputs:
- `data/csv/materials_jarvis_subset.csv`
- `data/external/jarvis_dft/PROVENANCE.json`

## Embedding experiments (4D -> 32D)

Run:
```bash
PYTHONWARNINGS=error ./venv/bin/python3 src/materials_embedding_experiments.py
```

Outputs:
- `data/csv/materials_embedding_benchmarks.csv`
- `data/artifacts/images/materials_pca_4d.png`
- `data/artifacts/images/materials_pca_8d.png`
- `data/artifacts/images/materials_pca_16d.png`
- `data/artifacts/images/materials_pca_32d.png`

## Columns (current)

- `jid`: JARVIS ID
- `formula`: derived chemical formula (when missing in source record)
- `elements`: comma-separated unique elements
- `nelements`: count of unique elements
- `formation_energy_peratom`: formation energy per atom (eV/atom)
- `optb88vdw_bandgap`: band gap (eV), where present
- `ehull`: energy above hull (eV/atom), where present
- `volume`: unit-cell volume (Angstrom^3), derived from lattice matrix when possible

## Provenance and licensing

- JARVIS paper: https://www.nature.com/articles/s41524-020-00440-1
- Figshare article used by the fetcher: https://figshare.com/articles/dataset/JARVIS-DFT_Database/6815699

Always keep `PROVENANCE.json` alongside derived datasets and avoid editing it manually.
