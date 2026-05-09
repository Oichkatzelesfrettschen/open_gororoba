# open_gororoba glossary

Authoritative definitions of acronyms and project-specific terms used in source
code, claims, experiments, and reports. Edit this file directly; it is not
generated.

## Core terms

### CD (Cayley-Dickson)

The doubling construction that builds R, C, H (quaternion), O (octonion),
S (sedenion), P (pathion), and higher dimensions. See `cd_kernel/`.

### Tower depth trichotomy

Three regimes governing the behavior of CD-based embeddings:

- 8-16D (octonion / sedenion): discrimination regime. Lowest noise floor.
- 32-64D (pathion / chingon): nonlinearity regime. Maximum sensitivity.
- 128D and above: channel-starved regime. Marginal returns; relies on rare
  high-rank zero divisors.

This trichotomy is a SYNTHETIC-data property; on real data (e.g., MaNGA) the
null is dimension-INDEPENDENT per claim C-1366.

### LBM (Lattice Boltzmann Method)

A computational fluid dynamics technique using discrete velocity sets on a
lattice. Used here for cosmological structure formation and dark-halo
falsification. Implementations: `lbm_core`, `lbm_3d_cuda`, `lbm_vulkan`.

### MRT (Multi-Relaxation-Time)

An LBM collision operator that relaxes different moments at different rates.
More stable at high Reynolds numbers than the single-relaxation-time BGK
operator. See `LbmSolver3DCuda::new_mrt()`.

### vf (void fraction)

`vf = void_count / total_cells` in the LBM grid (the fraction of cells
classified as topological voids by the associator-frustration threshold).
The complement `(1 - vf)` is the baryon filling fraction.

CDG-2 observational constraint: `(1 - vf) < 0.0006`. Source: Li et al. 2025
(arXiv:2506.15644), which derives a dark-matter fraction `>= 99.94%`.

Earlier prose described the CDG-2 constraint as `VF=0.0`; the precise statement
is `(1 - vf) < 0.0006`. The standalone label "VF" is ambiguous (does it refer
to the void or the baryon side?) -- always pair it with "void" or
"filling fraction" in new code and reports.

Literature precedent: Lei et al. 2023 defines VF as void fraction in LBM
porous-media flow; Liu et al. 2024 (arXiv:2409.02964) uses the same convention.

### CDG-2

The "Cosmic Dark Galaxy 2" observational target: a system whose baryon content
is below the detectability threshold of stellar surveys, used as a
falsification benchmark for dark-halo models in this repo.

### MaNGA

Mapping Nearby Galaxies at APO -- the Sloan Digital Sky Survey IV integral-field
spectroscopic survey. Used here for null-hypothesis validation of CD-based
algebraic models. See E-183, E-192, E-201, E-202; null is dimension-independent
per C-1366.

### Pantheon+

Type Ia supernova compilation (arXiv:2202.04077, ApJ 938, 110, 2022) used to
constrain dark-energy parameters. The DATA is retained as falsification
evidence; only the orthoplex diffusion MODEL (C-932) is FALSIFIED. C-441,
C-787, C-788 depend on Pantheon+ as valid evidence.

### DESI DR1 / DR2

Dark Energy Spectroscopic Instrument BAO measurements:

- DR1: arXiv:2404.03002 -- 7 BAO bins. Superseded for parameter inference.
- DR2: arXiv:2503.14738, PRD 112, 083515 (2025). 14M galaxies; provides
  3.1-4.2 sigma evidence for dynamical dark energy depending on SN compilation.
  Strengthens C-441 (bounce cosmology disfavored).

### Associator frustration

A measure of how non-associative a CD product is at a given input triple,
used as a heuristic threshold for classifying lattice cells as "topological
voids" in `generate-topological-voids` and `dark-halo-hunt`.

### Sign imbalance

The named replacement for "vacuum frustration" (one of the 8 banned terms in
the terminology-gate). See `sign_imbalance` crate.

## Stage A / Stage B / debt baseline

### Debt baseline (debt-baseline-v0)

Git tag on commit `970b4da3` (2026-04-30). Snapshots all measurable repo debt
at that point: 442 unsafe blocks (16.5% with SAFETY comments), 1326 claims
without formal_proof field, 209 broad proof gaps, 7 ignored RUSTSEC advisories,
215G cache.

### Stage A audit

Forensic data-collection pass producing 30+ artifacts under
`data/output/audit/2026-04-30/`. Inventory only; no source code changes.

### Stage B execution

Action plan derived from Stage A; tracked in
`~/.claude/plans/stage-b-debt-resolution.md`. Six phases: gate restoration,
memory corrections, registry corrections, code fixes, documentation, schema
extension.

## Registry layer

### Three-layer registry architecture (since 2026-03-19)

1. Source: `registry/canonical/control_plane.sqlite3` (canonical write target).
2. Build: `.cache/registry.sqlite3` (gitignored, derived).
3. Query: `gororoba-db` CLI with FTS5 search, crossrefs, audit.

### Compatibility export

The 36 `registry/*.toml` files. Each starts with the header
`# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.` Regenerated via
`cargo run -p gororoba_cli_data --bin provenance -- export-control-plane`.

### integrity-resolution

The binary that regenerates `registry/schema_signatures.toml` after any
upstream change. Required by the governance gate (the gate fails on
content_sha mismatch).
