# C-070 NANOGrav Spectrum Match Sources

Official public surfaces used by the local audit:

- PDG 2024 review highlights:
  `https://pdg.lbl.gov/2024/reviews/rpp2024-rev-highlights.pdf`
- NANOGrav 15-year SMBHB background:
  `https://nanograv.org/15yr/SMBHB`
- NANOGrav 15-year KDE free-spectrum archive:
  `https://zenodo.org/api/records/10344086/files/NANOGrav15yr_KDE-FreeSpectra_v1.1.0.zip/content`

Current authoritative Rust surfaces in this repo:

- PMNS matrix reference surface:
  `crates/stats_core/src/lib.rs`
- NANOGrav parser/provider surface:
  `crates/data_core/src/catalogs/nanograv.rs`
- Deterministic particle numerology audit:
  `crates/gororoba_cli_data/src/bin/particle_numerology_audit.rs`

Local audit rule:

- The checked-in `data/external/nanograv_15yr_freespectrum.csv` must match the
  embedded Rust best-fit table exactly and contain only finite interval values.
- Legacy Python fetch/test paths are migration debt and are not treated as the
  primary verification surface.
