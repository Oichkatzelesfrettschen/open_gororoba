# Experiment reproducibility manifest schema

This file defines the fields that every entry in `registry/experiments.toml`
should carry to make experiments reproducible. It is a definition document
only; no backfill of existing experiments is implied.

## Why this file exists

`registry/experiments.toml` currently has 251 entries. They have a
`run_command_sha256` field but lack:

- `code_commit_sha`: which git commit the experiment was run against.
- `input_hashes`: SHA256 of each input file at execution time.
- `output_hash_refs`: SHA256 of each output file produced by the run.
- A precise definition of what `reproducibility_class` means operationally.

Without these, "reproducing" an experiment is a soft claim. With them, anyone
can verify bit-for-bit that a re-run produces the same result.

## Field definitions (proposed extensions)

### `reproducibility_class`

A string token from the controlled vocabulary below.

| Token           | Meaning                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `bit_exact`     | A re-run on identical inputs at the same `code_commit_sha` produces byte-identical outputs.       |
| `numeric_close` | Outputs match within a published `numeric_tolerance` (relative or absolute, declared per output). |
| `statistical`   | Outputs match within a declared statistical envelope (chi-squared, KS-distance, etc.).            |
| `inferential`   | The qualitative conclusion (claim verdict) is preserved; raw outputs may differ.                  |
| `external_only` | Reproducibility depends on third-party data that may change; cite the snapshot date.              |

Default: `numeric_close` for floating-point pipelines, `bit_exact` for
deterministic integer/symbolic pipelines.

### `code_commit_sha`

40-char lowercase hex git SHA. The commit at which the experiment was last
executed and produced the recorded outputs. Should match a tag in the
`baseline_tags` array in `registry/project.toml` when the experiment is part
of a baseline.

### `input_hashes`

An array of `{path, sha256}` records, one per input file. `path` is relative
to the repo root. `sha256` is lowercase 64-char hex. Use `blake3` for new
fields if the entry is added after 2026-Q3 (faster, but document the change).

### `output_hash_refs`

An array of `{artifact_id, sha256}` records, one per output file. The
`artifact_id` references `registry/binaries.toml` or
`data/artifacts/ARTIFACTS_MANIFEST.csv`. `sha256` is the hash of the output
content at execution time.

### `numeric_tolerance` (only for `reproducibility_class = "numeric_close"`)

Per-output relative and absolute tolerances. Schema:

```toml
[[experiment.numeric_tolerance]]
output_id = "result.csv"
column = "void_fraction"
rel_tol = 1e-6
abs_tol = 1e-9
```

### `inferred_timestamp` vs `actual_timestamp`

- `actual_timestamp`: the wall-clock time the experiment was run, in
  ISO-8601 UTC. Required for new entries.
- `inferred_timestamp`: only set when the actual is unknown (e.g.,
  back-filled experiments). MUST be marked with the field name
  `inferred_timestamp` rather than `actual_timestamp`.

The governance gate should warn (not fail) on missing `actual_timestamp`
for experiments registered after 2026-04-30.

## Hashing tools

- `sha256sum <path>` for input/output hashes (POSIX standard).
- `cargo run -p gororoba_cli_data --bin manifest-hash <path>` is the
  preferred binary; it normalizes line endings and strips inferred
  timestamps before hashing, ensuring stability across git checkouts.

## Backwards compatibility

Existing experiments have no obligation to backfill these fields. The
schema extension is additive: missing fields simply mean "reproducibility
class undeclared". The governance gate must remain green during the
transition period.

When backfilling becomes a deliberate sprint:

1. Run experiments in logical batches (e.g., all experiments under a single
   `claim_refs` family).
2. Capture outputs with the binary above.
3. Land changes via the SQLite canonical write path (the TOML files are
   read-only compatibility exports as of 2026-03-23).

## Worked example

A fully-populated experiment entry would look like:

```toml
[[experiment]]
id = "E-201"
status = "completed"
status_token = "COMPLETED"
title = "MaNGA face-on red-noise-corrected null re-test"
claim_refs = ["C-1432", "C-1433"]

reproducibility_class = "numeric_close"
code_commit_sha = "8c3a4d7e0e8f2b1a6c5d9e3f7a2b8c4d1e6f9a3b"
actual_timestamp = "2026-03-18T14:32:00Z"
run_command_sha256 = "abc123..."

[[experiment.input_hashes]]
path = "data/external/manga/manga_rotcurves_all.csv"
sha256 = "deadbeef..."

[[experiment.output_hash_refs]]
artifact_id = "data/results/e201/face_on_chi2.csv"
sha256 = "feedface..."

[[experiment.numeric_tolerance]]
output_id = "face_on_chi2.csv"
column = "p_value"
rel_tol = 1e-3
abs_tol = 1e-6
```

## See also

- `registry/source_manifest.toml` -- declares which TOMLs are subject to
  compatibility round-trip verification.
- `data/artifacts/ARTIFACTS_MANIFEST.csv` -- canonical artifact index.
- Stage B plan task B-Doc4 (this file is the deliverable).
- The Turing Way reproducibility chapter for cross-disciplinary precedent.
