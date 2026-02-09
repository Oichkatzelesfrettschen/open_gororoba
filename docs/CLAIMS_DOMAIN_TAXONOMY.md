<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Claims Domain Taxonomy

Goal: make the C-001..C-427 claim audit mechanically tractable by grouping claims into a small
set of stable domains (plus optional tags), without changing the canonical claim table in
`docs/CLAIMS_EVIDENCE_MATRIX.md`.

Rules:
- Domains are ASCII-only slugs (lowercase, hyphen-separated).
- A claim may list multiple domains; the first entry is the "primary" domain.
- Keep the domain list small and stable; prefer tags only when needed.

## Canonical domains

- `meta`: Methodology, falsifiability, statistics hygiene, repo policy.
- `algebra`: Cayley-Dickson algebras, sedenions, zero divisors, box-kites, motifs.
- `spectral`: Spectral geometry, spectral triples, inverse-spectral issues.
- `holography`: AdS/CFT, RT/HRT, modular Hamiltonians, entanglement wedges, holographic QEC.
- `open-systems`: Lindblad/GKSL, decoherence, pointer states, Quantum Darwinism.
- `tensor-networks`: Tensor-network models, entropy scaling, QEC toy codes.
- `cosmology`: FRW/LCDM comparisons, dark energy models, negative/fractional dimension models.
- `gravitational-waves`: GWTC catalogs, mass distributions, selection effects, null models.
- `stellar-cartography`: TSCP alignment, sky mapping, trial factors for sky searches.
- `materials`: Metamaterials, absorbers, TCMT, JARVIS/NOMAD-derived baselines.
- `engineering`: Optimization/implementation work, performance, systems-level demos.
- `datasets`: Fetchers, dataset snapshots, provenance, schema integrity.
- `visualization`: Plotting standards, dashboards, artifact generation/verification.
- `cpp`: C++ kernels, bindings, build/test/bench pipelines.
- `coq`: Rocq/Coq formalization and proof artifacts.
- `legacy`: Legacy artifacts and reproduction/characterization work.

## Mapping location

The mapping file is:
- `docs/claims/CLAIMS_DOMAIN_MAP.csv`

It is consumed by offline scripts/verifiers and can be used to generate per-domain indexes under
`docs/claims/by_domain/`.
