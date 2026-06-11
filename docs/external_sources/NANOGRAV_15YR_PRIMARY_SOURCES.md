# NANOGrav 15-Year Primary Sources

# Snapshot date: 2026-03-19

This index lists first-party or primary-reference surfaces that are directly relevant to the
current NANOGrav timing, propagation, and refit-preflight lanes in this repository.

## Official collaboration and dataset surfaces

1. NANOGrav data release page
   - https://nanograv.org/science/data

2. NANOGrav 15-year release announcement
   - https://nanograv.org/news/15yrDataSet

3. Zenodo dataset record, version 2.1.0, published July 17, 2025
   - https://zenodo.org/records/16051178
   - DOI: https://doi.org/10.5281/zenodo.16051178

## Timing and analysis references named by the dataset

4. The NANOGrav 15-year data set: observations and timing of 68 millisecond pulsars
   - DOI: https://doi.org/10.3847/2041-8213/acda9a
   - arXiv landing page referenced by the Zenodo record: https://arxiv.org/abs/2306.16217

5. The NANOGrav 15-year data set: evidence for a gravitational-wave background
   - DOI: https://doi.org/10.3847/2041-8213/acdac6

6. PINT repository
   - https://github.com/nanograv/PINT

7. PINT documentation and fitter/model examples
   - https://nanograv-pint.readthedocs.io/en/latest/
   - https://nanograv-pint.readthedocs.io/en/latest/examples/understanding_fitters.html

8. tempo2 repository named by the dataset record
   - https://bitbucket.org/psrsoft/tempo2

## Local repo surfaces that should be read together with the primary sources

1. Release parser
   - `crates/gororoba_cli_data/src/nanograv_timing.rs`

2. Typed timing-model parser
   - `crates/gororoba_cli_data/src/nanograv_timing_model.rs`

3. Timing inventory lane
   - `reports/nanograv_15yr_timing_inventory.toml`
   - `data/csv/nanograv_15yr_timing_inventory.csv`

4. Propagation audit lane
   - `reports/nanograv_15yr_propagation_audit.toml`
   - `data/csv/nanograv_15yr_propagation_pulsars.csv`
   - `data/csv/nanograv_15yr_pairwise_hd_audit.csv`

5. Timing-refit preflight lane
   - `reports/nanograv_15yr_timing_refit_preflight.toml`
   - `data/csv/nanograv_15yr_timing_refit_preflight.csv`

## Scope note

The primary sources above justify:

- what the official data release contains,
- how the timing release should be interpreted,
- which software surfaces are appropriate for timing-model work,
- and what the collaboration itself claims.

They do not justify any repo-local claim of a detected Cayley-Dickson, sedenion, AVT,
or "topological vacuum friction" signal. Those claims remain local hypotheses until they are
tested against a proper timing-refit and stronger statistical null models.
