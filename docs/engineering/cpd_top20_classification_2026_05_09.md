# CPD top-20 cluster classification (2026-05-09)

This document classifies the 20 largest duplicate-line clusters reported by
`make cpd-audit CPD_TOP=20` (PMD CPD 7.x, `--minimum-tokens 42`). It is the
deliverable for Stage B / Roadmap PH-1.B and feeds PH-1.A
(bounded_nelder_mead optimizer hoist, the first concrete dedupe target).

## Run summary

- Total duplicate clusters at min-tokens=42: **6120**
- Total duplicated lines: **69450**
- Top-20 clusters reported below; the full report is on stdout when
  `make cpd-audit CPD_TOP=20` runs.

## Classification taxonomy

Per the in-repo roadmap (`plans/repo_debt_roadmap_2026_04_11.toml`), each
cluster is one of:

- **algorithmic**: hand-written logic that is genuinely duplicated. Real
  dedupe target. Extract to a shared helper.
- **generated**: produced by registry-emit or similar. Quarantine via
  `// @generated` (already in place after Phase A.3) and exclude from CPD.
- **fixture-or-test**: test fixtures, golden data, or test setup boilerplate.
  Keep duplicated unless the test signal degrades.
- **literal-table**: hard-coded numeric arrays, lookup tables, JSON
  fixtures. Keep as-is; deduping risks data integrity.
- **cli-boilerplate**: standard `fn main()`, `Cli::parse()`, output-dir
  scaffolding repeated across CLI binaries. Mostly unavoidable; can be
  modestly reduced via shared helpers but watch readability.

## Top 20 clusters

Each entry: `#<rank>` (lines / tokens) - shorthand of the cluster head -
classification + recommendation.

### #1 (136 lines / 875 tokens) -- "Probe: {} Embedding: {}D, lag={}min"

Classification: **cli-boilerplate**. Probe/embedding initialization repeated
across heliosphere fetcher binaries (Wind, Pioneer, PSP, Voyager).
Recommendation: extract a `print_run_header(probe, embedding_dim, lag_min)`
helper into a shared `data_fetch_common` module.

### #2 (132 lines / 610 tokens) -- "measured_thresholds"

Classification: **algorithmic**. Statistical-threshold computation chain
duplicated across magnetopause / bow-shock / sheath detectors.
Recommendation: extract `compute_threshold_envelope(samples, mode)`. Real
dedupe target; ~130 lines of hand-written logic.

### #3 (117 lines / 879 tokens) -- "for comp in components.iter()"

Classification: **algorithmic**. Connected-component iteration with shared
neighbor-stitching logic. Likely shared between zero-divisor graph census
binaries. Recommendation: extract `iter_components_with_neighbors(graph)`.

### #4 (112 lines / 1166 tokens) -- "themis_dir = cli.data_dir.join(...)"

Classification: **cli-boilerplate**. Sub-directory-creation boilerplate for
mission-specific data lanes (THEMIS, Wind, PSP, etc.). Recommendation:
shared `prepare_mission_dirs(root, mission)` helper. Lower priority than
algorithmic; the boilerplate is shallow.

### #5 (105 lines / 680 tokens) -- "detect_magnetopause_crossings_filtered"

Classification: **algorithmic**. Magnetopause-detection filter shared
across at least two heliosphere binaries. Recommendation: hoist into
`heliosphere_crossings::filter::detect_magnetopause`.

### #6 (99 lines / 771 tokens) -- "build_kinetic_operator(n, l, alpha, epsilon)"

Classification: **algorithmic**. Quantum kinetic-operator builder
duplicated between hydrogenic spectroscopy binaries. Recommendation:
hoist into `quantum_core::kinetic::build_operator`.

### #7 (93 lines / 497 tokens) -- "pub fn step(&mut self, cmd: vk::CommandBuffer, frame: u32)"

Classification: **algorithmic** (with framework constraint). Vulkan
dispatch boilerplate shared across `lbm_vulkan` compute pipelines.
Recommendation: extract a `VulkanStep::execute` trait method on a
shared step-builder; modest yield because Vulkan handle types vary.

### #8 (87 lines / 699 tokens) -- "channels * steps; window_rows = (steps - 1) * lag"

Classification: **algorithmic**. CD-tower embedding dim setup.
Recommendation: hoist `cd_embedding_dim_layout(channels, steps, lag)` into
`spectral_core` since it already owns `cd_embedding`.

### #9 (84 lines / 689 tokens) -- "best_angles" rotation search

Classification: **algorithmic**. Coarse-grained rotation search loop in
the CKM/PMNS selector scan binaries. Recommendation: hoist
`coarse_rotation_search(predicate, granularity)` into
`algebra_experimental::neutrino_sector::rotation_search`.

### #10 (84 lines / 480 tokens) -- "fn main() -> anyhow::Result<()> { let cli = Cli::parse() ..."

Classification: **cli-boilerplate**. Standard CLI entry point. Keep
duplicated; deduping reduces readability of binary main().

### #11 (82 lines / 524 tokens) -- "lattice_pos.iter().map(|&k| (k as f64) * (k as f64))"

Classification: **algorithmic**. Squared-distance computation on an
iterator of integer lattice positions. Recommendation: hoist
`lattice_distance_sq(coords)` into `gororoba_sparse_grid` or
`tensor_core`.

### #12 (80 lines / 592 tokens) -- "let dim = 512; let t0 = Instant::now(); let components"

Classification: **algorithmic**. Component-extraction benchmark
boilerplate. Recommendation: extract `bench_extract_components(dim,
input)` into a dedicated bench helper.

### #13 (79 lines / 764 tokens) -- "Window: {} to {} ({} days), probe: {}, dim={}"

Classification: **cli-boilerplate**. Run-window prologue print. Same
recommendation as #1: shared header helper.

### #14 (78 lines / 564 tokens) -- "Not enough data for embedding: need {} rows, have {}"

Classification: **algorithmic** (boundary check). Embedding-input
validation duplicated across CD-embedding binaries. Recommendation:
extract `validate_embedding_input(rows, needed)` into `spectral_core`.

### #15 (77 lines / 758 tokens) -- "Window: {} to {} ({} days), probe: {}"

Classification: **cli-boilerplate**. Same as #13 minus the `dim` field;
likely a slightly older variant. Recommendation: collapse with #13's
extraction.

### #16 (76 lines / 451 tokens) -- "rayleigh_csv: PathBuf"

Classification: **cli-boilerplate**. Cli struct field group. Probably
shared across the Rayleigh-spectrum binaries. Recommendation: extract
a shared `RayleighArgs` struct in `gororoba_cli`.

### #17 (73 lines / 470 tokens) -- "let path = psp_dir.join(&fname); let content = if path.exists()"

Classification: **cli-boilerplate** (file-IO). Conditional file-load
pattern. Recommendation: shared `load_or_fetch(dir, fname,
fetch_fn)` helper in `data_fetch_common`.

### #18 (72 lines / 467 tokens) -- "fs::write(path, text).with_context(|| format!(\"write {}\", ...))"

Classification: **cli-boilerplate**. Standard write-and-context helper.
Already trivially abstractable: `write_text(path, text)`. Low priority.

### #19 (72 lines / 366 tokens) -- "validate_scalar_trace_signal(channel, &values, nonzero_thr)"

Classification: **algorithmic**. Scalar-trace validation duplicated
across kerr/pathion/spin-tomography binaries. Recommendation: hoist
into `spin_tomography_core::validate`.

### #20 (70 lines / 436 tokens) -- "Compute spatial correlation between imbalance and viscosity"

Classification: **algorithmic**. Cross-domain spatial-correlation
computation. Recommendation: hoist `spatial_correlation(field_a,
field_b)` into `spectral_core`.

## Summary

- **algorithmic clusters (real dedupe targets)**: #2, #3, #5, #6, #7, #8,
  #9, #11, #12, #14, #19, #20 -- 12 of 20 (60%).
- **cli-boilerplate (lower priority)**: #1, #4, #10, #13, #15, #16, #17,
  #18 -- 8 of 20 (40%).
- **generated / literal-table / fixture**: 0 of 20. The Phase A.3
  `@generated` marker work removed those from the report (the marker
  is recognized by PMD's lexer as a hint to suppress).

## Recommended dedupe order (PH-1.A onwards)

1. **Bounded Nelder-Mead optimizer** (PH-1.A; not in this top-20 because
   the optimizer is small enough to fall under min-tokens, but it is
   cited in the in-repo roadmap as a known duplicate between cosmology
   bounce and observational fitting. Hoist first.)
2. Cluster #2 (`measured_thresholds`): largest algorithmic cluster.
3. Cluster #6 (`build_kinetic_operator`): tightly scoped to quantum_core.
4. Cluster #5 (`detect_magnetopause_crossings_filtered`): heliosphere lane.
5. Cluster #8 (`cd_embedding_dim_layout`): clean spectral_core ownership.
6. Cluster #11 (`lattice_distance_sq`): one-liner extraction.
7. Cluster #14 (`validate_embedding_input`): boundary check; small.
8. Cluster #19 (`validate_scalar_trace_signal`): spin-tomography lane.

The boilerplate clusters (#1, #10, #13, #15) deserve a shared
`run_header_print(probe, dim, lag, days)` helper but the readability
trade-off should be evaluated case by case.

## Re-running

```sh
make cpd-audit CPD_TOP=20
```

Output is currently to stdout; consider tee-ing to
`data/output/audit/<date>/cpd_top20.txt` for archival.

## See also

- `plans/repo_debt_roadmap_2026_04_11.toml` (PH-1, PH-1.A, PH-1.B).
- `crates/gororoba_cli_data/src/bin/cpd_report.rs` (the local renderer).
- The PMD CPD primary source: https://docs.pmd-code.org/latest/pmd_userdocs_cpd.html
