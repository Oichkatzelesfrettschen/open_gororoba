# Verification Ladder

Every result in open\_gororoba is classified on a four-tier verification ladder.
This prevents conflation of proven algebra with speculative physics.

## Tier 1: Algebraic (verified)

Deterministic tests against known results from the literature.

**Examples:**
- Cayley-Dickson multiplication tables (C-001 through C-006)
- Zero-divisor census: 42 assessors, 7 box-kites (C-050 through C-060)
- De Marrais strut table match (C-454)
- Motif scaling laws at dim=16..256 (C-126 through C-130)
- Emanation architecture L1-L18 (C-468 through C-475)

**Standard:** Unit tests using integer-exact arithmetic.  Every algebraic
claim has a corresponding `#[test]` function.

## Tier 2: Mathematical (partially verified)

Tests plus literature cross-check, with some gaps remaining.

**Examples:**
- Lattice codebook filtration (8 monograph theses A-H, C-458 through C-467)
- E8 root system connection (REFUTED, C-455)
- XOR balanced search extension (C-445 through C-448)

**Standard:** Cross-validation against external data files.  Null-model
rejection where applicable.

## Tier 3: Statistical (modeled)

Permutation-tested hypotheses with Benjamini-Hochberg FDR correction.

**Examples:**
- Ultrametric hierarchy fingerprint across 9 catalogs (I-011, I-013)
- GWTC-3 mass distribution multimodality (E-008)
- Temporal cascade analysis for FRB repeaters (C-438)

**Standard:** p-values from permutation null, BH-FDR correction at 0.05.
Results classified as Verified, Refuted, or Partial.

## Tier 4: Speculative (hypothetical)

Exploratory code with explicitly tracked falsification criteria.

**Examples:**
- Gravastar TOV parameter sweep (C-400 through C-410)
- Bounce cosmology vs Lambda-CDM (C-200 through C-210, Delta BIC = +7.37)
- Negative-dimension eigenvalue convergence (C-420 through C-425)

**Standard:** Claims tracked in `registry/claims.toml` with `where_stated`
and falsification criteria.  No claim of physical validity.

## Claims workflow

1. New physical hypothesis gets a C-nnn entry in `claims.toml`
2. Entry includes: statement, status, where\_stated, test function
3. Status progresses through: Pending -> Verified / Refuted / Partial
4. `registry-check` validates all entries for consistency
