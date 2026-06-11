# NANOGrav 15-Year Cayley-Dickson Synthesis

# Repo-grounded technical monograph and gap audit

# Snapshot date: 2026-03-19

**Status**: Active research synthesis, not a publication-ready detection paper.
**Scope**: This document reconstructs the current NANOGrav-related lane in `open_gororoba`
from the repository artifacts, the recent timing-refit tranche, and first-party release
surfaces. It is intentionally over-explicit about what is implemented, what is only
heuristic, and what is presently unsupported.

---

## 1. Executive statement

The repository now contains three scientifically useful and directly supported NANOGrav
surfaces:

1. A release inventory lane over the official 15-year timing dataset.
2. A propagation audit lane over released residual, DMX, and wideband-DM products.
3. A typed timing-model preflight lane over `.par` files for eventual refit work.

These surfaces support statements about:

- dataset presence and file completeness,
- pulsar-level metadata and timing-model structure,
- released residual and DM-related summary diagnostics,
- simple pairwise residual-vs-Hellings-Downs discrepancy summaries.

They do **not** yet support a claim that a sedenion, Cayley-Dickson, or AVT-derived field
has been detected in the NANOGrav 15-year data.

The strongest reason is structural:

- the current AVT lane applies a pulsar-static scalar correction,
- it does not refit TOAs,
- it does not regenerate residuals from a timing model,
- it does not enter the PTA likelihood used by the collaboration,
- and the synthetic null generator does not sample a correlated Hellings-Downs field in the
  mathematically intended way.

That means the repo already contains a meaningful bridge from dataset ingest to solver
planning, but it does **not** yet contain a falsification-grade test of the hypothesis
"Cayley-Dickson structure is required by the NANOGrav 15-year sky correlations."

---

## 2. First-principles decomposition of the current pipeline

### 2.1 Official data surface

The local release root used throughout the repo is:

`data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0`

The first-party Zenodo record for version `2.1.0`, published on **July 17, 2025**, states:

- the release contains narrowband and wideband TOAs,
- timing solutions for 68 millisecond pulsars,
- clock files,
- timing-model `.par` files,
- timing `.tim` files,
- noise-chain and correlation products,
- and, in v2.1.0, narrowband post-fit residual files.

This matches the repo's current loading strategy in
`crates/gororoba_cli_data/src/nanograv_timing.rs`, which recursively inventories:

- wideband `.tim` files for `pp_dm` and `pp_dme`,
- narrowband and wideband `.par` files,
- DMX `.out` files,
- released residual `.res` files.

### 2.2 Timing inventory lane

The timing inventory binary is the dataset-surface census. Its current report
`reports/nanograv_15yr_timing_inventory.toml` says:

- `pulsar_count = 68`,
- all 68 pulsars have narrowband-ready, wideband-ready, residual-ready, noise-ready,
  DMX-ready, and parallax-ready surfaces,
- only `23` pulsars have whitened full residuals available.

This is valuable because it separates "what the release contains" from "what the repo
still needs to derive." It is evidence of inventory completeness, not of any new physics.

### 2.3 Propagation audit lane

The propagation audit binary consumes released products and emits:

- `data/csv/nanograv_15yr_propagation_pulsars.csv`
- `data/csv/nanograv_15yr_pairwise_hd_audit.csv`
- `reports/nanograv_15yr_propagation_audit.toml`

Its report states:

- `pulsar_count = 68`
- `pair_count = 2278 = 68 * 67 / 2`
- `dmx_ready_count = 65`
- `wideband_dm_ready_count = 68`
- `avg_residual_ready_count = 68`

The code computes simple statistics from released series:

- per-pulsar DMX and wideband-DM scatter,
- per-pulsar residual RMS and residual-vs-uncertainty correlations,
- pairwise Pearson correlations of binned average residuals,
- comparison of those pairwise correlations to the Hellings-Downs kernel.

This is a useful diagnostic surface, but it is still a post-fit, release-level summary.
The report itself is candid and already states:

`status = "scoped_only_not_recomputed_in_repo"`

That is exactly the correct status.

One concrete pair is especially diagnostic:

- `B1937+21` versus `J1713+0747`
- `overlap_bins = 164`
- `separation_deg = 37.729374742367`
- `hellings_downs = 0.119748145966`
- `avg_residual_pearson = -0.631273141590`
- `avg_white_residual_pearson = 0.095854671539`

That raw-to-white change is large. It shows that preprocessing and whitening materially
change the pairwise residual surface, which is another reason the present AVT overlay must
not be read as a substitute for a true PTA-likelihood analysis.

### 2.4 Timing-refit preflight lane

The new timing-model tranche consists of:

- `crates/gororoba_cli_data/src/nanograv_timing_model.rs`
- `crates/gororoba_cli_data/src/bin/nanograv_timing_refit_preflight.rs`
- `data/csv/nanograv_15yr_timing_refit_preflight.csv`
- `reports/nanograv_15yr_timing_refit_preflight.toml`

This is the strongest current solver-facing addition in the repo.

It parses wideband and narrowband `.par` files into typed structures:

- binary-family labels (`BT`, `DD`, `DDK`, `ELL1`, `ELL1H`, or other),
- astrometric terms,
- spin terms,
- FD terms,
- JUMPs,
- DMJUMPs,
- noise terms,
- DMX windows,
- solution IDs distinct from canonical pulsar IDs.

Current report summary:

- `solution_count = 76`
- `unique_pulsar_count = 68`
- `multi_solution_pulsar_count = 6`
- phase-1 subset size `6`

Current wideband binary-family census:

- `BT = 1`
- `DD = 25`
- `DDK = 3`
- `ELL1 = 24`
- `ELL1H = 3`
- `isolated = 20`

Current phase-1 seed solutions:

- `J1713+0747_PINT_20230131.wb`
- `J1903+0327_PINT_20230131.wb`
- `J2214+3000_PINT_20230131.wb`
- `J0709+0458_PINT_20230131.wb`
- `J1312+0051_PINT_20230131.wb`
- `J2317+1439_PINT_20230131.wb`

This is a real advance because future refit work needs typed model structure, not just
directory traversal.

---

## 3. Mathematical reconstruction of the implemented AVT lane

### 3.1 Pulsar-static scalar field

The current AVT filter binary is:

`crates/gororoba_cli_data/src/bin/nanograv_avt_filter.rs`

Let pulsars be indexed by `i = 1, ..., N`, with `N = 68`.

For each pulsar:

- sky direction `s_i` is inferred from the timing metadata,
- `s_i` is projected into a chosen hypercomplex-lattice basis,
- a single scalar frustration score `f_i` is computed from AVT norms,
- the score is standardized to a z-score.

Thus the entire high-dimensional algebraic object enters the data analysis only through a
single scalar per pulsar:

`f_i in R`

That is already a major reduction in model expressivity.

### 3.2 Residual transform actually implemented

For a pulsar residual time series `r_i(t_k)`, the implemented transform is:

`r'_i(t_k; lambda) = r_i(t_k) - lambda f_i`

where `lambda` is scanned over a 1D grid.

This is a constant offset for pulsar `i`, independent of:

- epoch `t_k`,
- observing frequency,
- TOA uncertainty,
- binary state,
- DM state,
- pair geometry.

### 3.3 Why centered RMS cannot improve

Let the pulsar mean be:

`mu_i = (1 / n_i) sum_k r_i(t_k)`

Then the centered residual is:

`tilde{r}_i(t_k) = r_i(t_k) - mu_i`

After the constant shift:

`mu'_i = (1 / n_i) sum_k (r_i(t_k) - lambda f_i) = mu_i - lambda f_i`

Therefore

`tilde{r}'_i(t_k) = r'_i(t_k) - mu'_i`
`= (r_i(t_k) - lambda f_i) - (mu_i - lambda f_i)`
`= r_i(t_k) - mu_i`
`= tilde{r}_i(t_k)`

So the centered residual series is invariant under the transform.

This is not a numerical accident. It is an exact identity.

That is why the reports consistently show:

- `static_field_centered_invariant = true`
- `centered_drop_pct = 0`

The implemented model cannot whiten within-pulsar centered scatter because it is a constant
mean shift, not a temporal or chromatic correction.

### 3.4 Pairwise objective actually implemented

Let:

- `rho_ij` be the released pairwise Pearson correlation,
- `Gamma_ij` be the Hellings-Downs prediction for angular separation `zeta_ij`,
- `w_ij` be the overlap-bin weight.

The baseline discrepancy objective is:

`J_0 = sum_{i<j} w_ij (rho_ij - Gamma_ij)^2`

The AVT-adjusted objective is:

`J(lambda) = sum_{i<j} w_ij (rho_ij - lambda f_i f_j - Gamma_ij)^2`

This is a rank-1 correction in pair space, because the added term factorizes as
`f_i f_j`.

So the present "resonance" is not a timing-model refit and not a direct sky-correlation
recovery. It is the best one-parameter low-rank perturbation of an already-computed pairwise
correlation table.

That makes it a heuristic alignment test, not evidence of a physical field.

### 3.5 Reported values and what they mean

The main report `reports/nanograv_avt_filter.toml` records, for dimension 512:

- `cross_corr_drop_pct = 10.315610483005`
- `cv_cross_corr_drop_pct = 4.738506991497`
- `null_pvalue = 0.375`

Mathematically, this means:

- the one-parameter rank-1 correction reduces the chosen pairwise squared-error objective
  by about 10.3 percent on the full set,
- by about 4.74 percent under the binary's own cross-validation scheme,
- and its shift-based null is weak, because `p = 0.375` is not remotely near a discovery
  threshold.

This does **not** mean "physical anomaly confirmed."

### 3.6 Synthetic control lane and the key implementation gap

The synthetic generator is:

`crates/gororoba_cli_data/src/bin/nanograv_synthetic_gen.rs`

It builds an HD matrix `Gamma`, computes a Cholesky factor, then discards it:

- `let _chol = Cholesky::new(gamma)...`

The emitted synthetic pairwise correlations are then generated as:

`rho^syn_ij = clamp(Gamma_ij + epsilon_ij, -1, 1)`

with independent Gaussian noise `epsilon_ij`.

That is **not** a draw from a correlated Gaussian field with covariance `Gamma`.

The mathematically intended procedure would be:

1. Compute `Gamma = L L^T`.
2. Draw `z ~ N(0, I)`.
3. Set `x = L z`.
4. Derive time-domain or pair-domain observables from the correlated field `x`.

Because the current code never uses `L`, the synthetic control is only a noisy table around
the HD mean curve. It does not preserve the joint covariance structure of a PTA background.

This is a central gap.

---

## 4. Concrete files that currently matter most

### 4.1 Data loading and typed timing-model surfaces

- `crates/gororoba_cli_data/src/nanograv_timing.rs`
  - release traversal, residual loading, DMX parsing, wideband-DM parsing, sky-vector derivation

- `crates/gororoba_cli_data/src/nanograv_timing_model.rs`
  - typed `.par` parsing and solver-planning metadata extraction

- `crates/gororoba_cli_data/src/bin/nanograv_timing_refit_preflight.rs`
  - per-solution model census and phase-1 refit subset selection

### 4.2 Release-derived audit surfaces

- `crates/gororoba_cli_data/src/bin/nanograv_propagation_audit.rs`
  - first-pass released residual and pairwise HD discrepancy audit

- `reports/nanograv_15yr_timing_inventory.toml`
  - availability census

- `reports/nanograv_15yr_propagation_audit.toml`
  - released-product diagnostic summary

- `reports/nanograv_15yr_timing_refit_preflight.toml`
  - typed timing-model summary for solver planning

### 4.3 Hypothesis-generating rather than claim-closing surfaces

- `crates/gororoba_cli_data/src/bin/nanograv_avt_filter.rs`
  - one-parameter AVT mean-shift / rank-1 pair correction audit

- `crates/gororoba_cli_data/src/bin/nanograv_synthetic_gen.rs`
  - synthetic control generator with currently incomplete covariance sampling

- `crates/gororoba_cli_data/src/bin/nanograv_gauge_resonance.rs`
  - gauge-sector mapping heuristic

- `crates/gororoba_cli_data/src/bin/nanograv_entropy_audit.rs`
  - entropy/bound mapping heuristic

- `crates/gororoba_cli_data/src/bin/nanograv_vacuum_symmetry.rs`
  - symmetry-scanning heuristic over sampled AVT counts

### 4.4 Overclaim surfaces that should not be treated as settled evidence

- `reports/nanograv_falsification_report.toml`
  - currently states `status = "PHYSICAL_ANOMALY_CONFIRMED"`

- `crates/verified_core/src/monograph/nanograv_resonance.rs`
  - previously described the 512D result as confirmed physical significance

These should be read as hypothesis surfaces pending a proper refit and a stronger null model.

---

## 5. Claims that are currently supported

The following statements are supported by the repo as it exists now:

1. The official NANOGrav 15-year release surface is cached locally and can be traversed.
2. The repo can inventory 68 pulsars from the release.
3. The repo can summarize released DMX, wideband-DM, and residual products.
4. The repo can compute pairwise released-residual correlation summaries over all `2278`
   pulsar pairs.
5. The repo now parses `.par` files into typed timing-model structures and distinguishes
   `76` solution IDs from `68` canonical pulsar identities.
6. The repo can define a phase-1 timing-refit subset motivated by data richness and binary
   family coverage.
7. A one-parameter AVT-derived rank-1 perturbation can reduce the chosen pairwise discrepancy
   objective by roughly 8 to 15 percent depending on slice and dimension.
8. That same scalar-field construction cannot whiten centered intra-pulsar scatter, by exact
   algebraic identity.

These are honest and reproducible statements.

---

## 6. Claims that are not yet supported

The following statements are **not** currently justified by the repo evidence chain:

1. "The 512D resonance is a physical detection."
2. "The NANOGrav background requires a Cayley-Dickson or sedenion vacuum."
3. "The AVT mechanism explains the signal away from the Hellings-Downs curve."
4. "Topological vacuum friction has been observed."
5. "The synthetic null excludes a geometric artifact."
6. "The preferred dimension is 512 in a discovery-grade statistical sense."
7. "Gauge-sector, entropy-bound, or point-group symmetry overlays are physically identified
   in the PTA dataset."

Why these remain unsupported:

- no TOA-level refit,
- no full PTA likelihood,
- no proper correlated synthetic sky generator,
- weak null p-values in the current AVT report,
- many heuristic mappings from sky coordinates into hypercomplex basis labels,
- no blind out-of-sample confirmation on regenerated residuals.

---

## 7. Stability analysis of the AVT slice family

The repo contains several slice reports:

- `reports/nanograv_avt_filter.toml`
- `reports/nanograv_avt_disk_filter.toml`
- `reports/nanograv_avt_halo_filter.toml`
- `reports/nanograv_avt_early_filter.toml`
- `reports/nanograv_avt_middle_filter.toml`
- `reports/nanograv_avt_late_filter.toml`
- `reports/nanograv_avt_synthetic_filter.toml`

For `dim = 512`, the reported cross-correlation drop varies noticeably across slices:

| Slice     | cross_corr_drop_pct | cv_cross_corr_drop_pct | null_pvalue |
| --------- | ------------------: | ---------------------: | ----------: |
| full      |             10.3156 |                 4.7385 |       0.375 |
| disk      |             14.6648 |                15.5884 |       0.125 |
| halo      |              8.2685 |                 1.0007 |       0.000 |
| early     |             10.1343 |                11.0124 |       0.875 |
| middle    |             11.4430 |                 8.4270 |       0.500 |
| late      |             10.3156 |                 4.7385 |       0.625 |
| synthetic |              0.0102 |                 0.1344 |       0.375 |

Interpretation:

- the real-data objective can move by several percentage points under slicing,
- some slices improve cross-validated drop and some degrade it,
- the shift-based null is unstable and low-resolution because `null_shifts = 8`,
- the synthetic control is near zero, but the synthetic generator itself is incomplete.

So the slice family suggests "nonzero heuristic alignment exists," but not "the physical
signal has been isolated."

---

## 8. The actual missing engine

What is missing is not another post-fit audit. What is missing is a timing engine.

At minimum, the next solver tranche must implement:

1. TOA ingestion with clock and ephemeris handling.
2. Delay-model evaluation from timing parameters.
3. Design-matrix assembly for fit parameters.
4. Weighted or generalized least squares:

   `theta_hat = (M^T C^-1 M)^-1 M^T C^-1 y`

   where:

   - `y` is the timing residual vector,
   - `M` is the design matrix,
   - `C` is the noise covariance.

5. Regenerated post-fit residuals from the fitted model.
6. Pairwise sky-correlation diagnostics computed from regenerated residuals, not only from
   released collaboration products.

Only after that point can a Cayley-Dickson-inspired perturbation be tested against a
reconstructed PTA analysis chain.

---

## 9. Falsifiable next experiments

### 9.1 Experiment A: repair the synthetic null

Goal:

- replace the current pairwise-noise surrogate with a proper correlated draw from an HD
  covariance model.

Minimum acceptance criterion:

- the synthetic generator must actually use the Cholesky factor or an equivalent spectral
  factorization.

### 9.2 Experiment B: prove or kill the rank-1 correction on regenerated residuals

Goal:

- run the same `f_i f_j` correction against regenerated residual products from a local
  timing refit.

Acceptance criterion:

- compare improvement against block bootstrap, sky permutation, and band-split controls.

### 9.3 Experiment C: broaden the null family

The current `null_shifts = 8` cyclic-shift family is too small and too special.

Required nulls:

- pulsar-label permutation,
- frustration-score permutation,
- sky rotation or scrambling,
- time-block bootstrap,
- synthetic HD correlated skies,
- random rank-1 feature vectors with the same norm distribution as `f`.

### 9.4 Experiment D: test whether 512 is special after multiple-comparison control

If dimensions scanned are:

`D = {16, 32, 64, 128, 256, 512, 1024}`

then a peak at one dimension is a multiple-testing problem.

Required:

- a max-statistic null or Bonferroni/Sidak-corrected threshold,
- repeated seeds or lattice variants,
- stability under alternative sky-to-basis projections.

### 9.5 Experiment E: replace scalar frustration with time-dependent structure

Because the current transform is a constant mean shift, it cannot affect centered scatter.

Any future physically meaningful model must be at least one of:

- time-dependent,
- frequency-dependent,
- pair-dependent,
- or likelihood-integrated through the timing model.

Otherwise the same invariance theorem will block it.

---

## 10. Higher-order synthesis: what the repo is really building

The cleanest interpretation of the current project is not "we have found sedenions in PTA
data." The cleanest interpretation is:

1. A hypercomplex-geometry program is being developed as a feature-engineering language.
2. The NANOGrav release is serving as a high-value observational testbed.
3. The repo now has a credible ingest -> audit -> solver-planning bridge.
4. The remaining missing piece is the actual inverse problem solver.

That is scientifically respectable.

The project becomes much stronger if stated this way:

- Cayley-Dickson structure is a candidate basis family for engineered low-rank, graph-based,
  or symmetry-informed perturbations of PTA observables.
- The present code shows that such features can be projected onto real PTA metadata and can
  produce measurable changes in a heuristic pairwise objective.
- The present code does **not** yet demonstrate that these features are required by the data
  under the PTA likelihood.

This framing is both more rigorous and more powerful, because it turns the program into a
sequence of falsifiable estimation problems instead of a premature interpretation claim.

---

## 11. Repo-canonical recommendation

For this topic, the canonical long-form synthesis document should live at:

`docs/research/nanograv_15yr_cayley_dickson_synthesis_2026.md`

The canonical first-party source index should live at:

`docs/external_sources/NANOGRAV_15YR_PRIMARY_SOURCES.md`

If a later claims-to-evidence pass is desired, this document should then be linked into:

- `docs/claims/by_domain/gravitational_waves.md` or equivalent domain lane,
- `docs/tickets/` for the missing timing-engine and null-model tasks,
- and the requirements narrative once the refit engine exists.

---

## 12. Bottom line

The repo already supports a serious statement:

> The NANOGrav 15-year release has now been promoted from file presence checks to typed timing
> model parsing, released-product diagnostics, and a concrete refit-preflight plan.

The repo does not yet support this stronger statement:

> A sedenion or Cayley-Dickson vacuum has been detected in the NANOGrav 15-year dataset.

The path from the first statement to the second is clear:

1. finish the timing engine,
2. regenerate residuals,
3. repair the synthetic null,
4. rerun the AVT family inside a real refit loop,
5. and require the signal to survive stronger nulls and multiple-comparison control.

That is the mathematically honest next frontier.
