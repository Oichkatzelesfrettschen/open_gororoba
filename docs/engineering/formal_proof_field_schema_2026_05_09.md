# `formal_proof` field schema (DEBT-CLAIMS-PROOF design)

This document defines the canonical taxonomy for the `formal_proof` column
on the `claims` table. The current schema is `formal_proof TEXT` with no
constraint on the value; 1297 of 1441 rows are the empty string and 144
carry a file path.

## Current state (2026-05-09)

```
SELECT formal_proof, COUNT(*) FROM claims GROUP BY formal_proof ORDER BY COUNT(*) DESC LIMIT 5;

(empty)                                          1297
proofs/verified/C1262_FlatBandFractionHalf.v        3
proofs/verified/C1137_MissingEdgeQuantizedGap.v     3
proofs/theories/C1544_MorKerSDecomp.v               2
proofs/verified/C957_VoudonZD.v                     1
... (139 more single-claim file paths)
```

## Proposed taxonomy

Each entry must match one of these patterns:

### `na_empirical[:rationale]`

The claim is empirical (data-driven) and is not formalizable in Rocq. The
optional `:rationale` is a short hint such as `pantheon_plus_data`,
`manga_observation`, `gpu_benchmark`. Claims that simply observe a number
(e.g., "delta-BIC = +705 against Pantheon+") fall here.

### `na_observational[:source]`

Specific case of `na_empirical` for astronomical/cosmological observations
where the claim quotes a published measurement. The source is a paper key
or arXiv id (e.g., `2503.14738`).

### `na_methodology[:tool]`

Methodology/instrumentation claim where the "proof" is the tool's
correctness, not a Rocq theorem. Example: "cargo deny detects yanked
crates" is methodological. Tool key: `cargo_deny`, `pmd_cpd`, `clippy`,
etc.

### `pending[:reason]`

A formal proof is INTENDED but not yet written. The reason is one of:

- `pending:proof_skeleton` -- there is a proof skeleton in `proofs/theories/`
  but the body is not complete.
- `pending:lib_gap` -- the proof requires Rocq stdlib facilities not yet
  available (e.g., a missing differential-geometry library).
- `pending:reviewed_pending` -- triaged as a real proof-debt target;
  awaiting a focused proof sprint.
- `pending:literature` -- the result is known in published mathematical
  literature but no Rocq port exists yet.

### `proofs/verified/<file>.v[#theorem]`

A complete proof exists in `proofs/verified/`. The fragment after `#` is
the specific theorem name when one .v file proves multiple claims (e.g.,
`C1262_FlatBandFractionHalf.v#flat_band_d64_le_half`).

### `proofs/theories/<file>.v[#theorem]`

The proof exists at the Module Type level only (signature + axioms; no
body for the bound theorem). This is a half-step between `pending` and
`verified`.

### `external:<citation>`

The proof exists in published literature and the project accepts it as a
referenced result without re-proving in Rocq. Citation is a paper key or
arXiv id (e.g., `external:hurwitz_1898`, `external:moreno_1997`).

## Disposition rules

1. **Newly-authored claims** SHOULD set `formal_proof` to one of the
   above categories at creation time. Empty string is no longer
   acceptable for new rows.
2. **Backfill** of the 1297 empty-string rows uses the heuristic below.
3. **Re-classification** is allowed; record changes via the
   `claim_revisions` audit table (the gororoba-db Claim mutator already
   carries this for status_note edits and would extend trivially to
   formal_proof).

## Backfill heuristic (machine-applied)

```
For each claim row with formal_proof = '':
    if claim.status in ('Refuted', 'Falsified', 'Closed_negative_result'):
        formal_proof = 'na_empirical'                  # falsifications need no proof
    elif claim.where_stated contains 'arXiv:' or 'doi.org/':
        formal_proof = 'external:<extracted-key>'
    elif a file matching proofs/verified/<claim_id>_*.v exists:
        formal_proof = 'proofs/verified/<file>.v'
    elif a file matching proofs/theories/<claim_id>_*.v exists:
        formal_proof = 'proofs/theories/<file>.v'
    elif claim.statement contains an LBM, GPU, or simulation reference:
        formal_proof = 'na_methodology:simulation'
    elif claim.statement contains 'observed', 'measured', 'detected':
        formal_proof = 'na_observational'
    else:
        formal_proof = 'pending:reviewed_pending'
```

The heuristic is intentionally conservative: when in doubt, mark
`pending`. Human review can promote `pending` rows to a more specific
category later.

## Bulk apply via gororoba-db

The Claim mutator added in commit ed33535e edits `status_note`. A
parallel `update-formal-proof` action would be a small extension:

```rust
ClaimMutationAction::UpdateFormalProof {
    #[arg(long)]
    id: String,
    #[arg(long)]
    formal_proof: String,
    #[arg(long)]
    actor: Option<String>,
    #[arg(long)]
    reason: Option<String>,
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    regen_toml: bool,
},
```

The provenance_store helper would mirror `claim_update_status_note`
verbatim modulo the column name; the `entity_update_status_note` generic
helper introduced in commit edfb84bc could be parameterized further to
take the column name, allowing reuse.

## Acceptance criteria for closing DEBT-CLAIMS-PROOF

1. The taxonomy above is wired into `claim_revisions.field_name` (rows
   with field_name='formal_proof' represent value transitions).
2. The bulk backfill heuristic runs once over the 1297 empty rows; the
   resulting distribution is committed as
   `data/output/audit/<date>/formal_proof_backfill_distribution.toml`.
3. The governance gate adds a check that no new claim row has
   `formal_proof = ''` (only `pending:*` and the named categories).
4. `make repo-audit-strict` reports `formal_proof_present` >= 1441
   (i.e., zero empty strings remaining).

## Open questions

- **Where does `external:` live?** The citation key needs a registry
  (registry/bibliography_normalized.toml already exists; the `external:`
  tag points at a key there). Commit a cross-reference rule.
- **Multi-claim proofs**: when one .v proves three claims (C1262 does
  this), do all three rows share the same `formal_proof` path or do we
  break them apart with `#theorem` fragments? Recommend the latter.

## See also

- `data/output/debt_baseline_2026_05_09.toml` (formal_proof_present count).
- `crates/provenance_store/src/lib.rs::claim_update_status_note` (the
  template for the parallel update-formal-proof method).
- `registry/identity_gap_policy.toml` (the parallel pattern for accepted
  ID gaps).
