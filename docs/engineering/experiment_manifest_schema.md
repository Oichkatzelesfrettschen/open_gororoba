---
description: executable reproducibility manifest contract for registered experiments
last_verified: 2026-08-01
evidence_class: operational_schema
---

# Experiment reproducibility manifest schema

This file defines the fields for a standalone experiment run manifest. The
manifest is checked by `experiment-manifest verify` before its evidence is
described as replayable. Existing rows in `registry/experiments.toml` remain
legacy compatibility records until they are deliberately backfilled through
the SQLite canonical write path.

## Why this file exists

The compatibility export currently contains 227 experiment rows. Most legacy
rows have a `run_command_sha256` field but lack:

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

The executable verifier accepts only the five tokens in the table. Legacy
tokens such as `deterministic_replay`, `seeded_stochastic_replay`, and
`non_deterministic` remain valid in the compatibility export but do not pass
the new standalone manifest gate until they are classified.

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

An array of `{artifact_id, path, sha256}` records, one per output file.
`artifact_id` names an artifact in the canonical SQLite registry. `path` is
the exact repository-relative path registered for that artifact in
`artifact_paths` or `canonical_download_path`. `sha256` is the hash of the
retained output content at execution time. The verifier checks the identity to
path relation before hashing, so a path cannot be relabeled under another
artifact ID.

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

### Execution environment and randomness

Every standalone manifest also records:

| Field | Contract |
| --- | --- |
| `run_command` | Exact command string used for the run. |
| `run_command_sha256` | Lowercase SHA-256 of the UTF-8 bytes in `run_command`. |
| `toolchain` | Compiler, interpreter, or solver version string. |
| `features` | Explicit array of enabled features, including an empty array when none are enabled. |
| `hardware` | Host, accelerator, or simulator identity relevant to the result. |
| `randomness_mode` | `none`, `seeded`, or `external_entropy`. |
| `random_seed` | Required with `randomness_mode = "seeded"`. |
| `random_generator` | Required for seeded or external entropy runs. |
| `randomness_source` | Required for external entropy runs. |

`experiment-manifest verify` compares `code_commit_sha` with the checked-out
commit, resolves every input `path` below the repo root, checks every output
`artifact_id` and `path` against the read-only SQLite artifact registry, and
recomputes every SHA-256. It therefore requires output files to be retained and
registered before verification. `numeric_close` manifests also require at
least one `numeric_tolerance` row with nonnegative finite values.

## Hashing and verification tools

- `cargo run -p gororoba_cli_data --bin experiment-manifest -- hash <path>`
  prints a byte-preserving SHA-256 for one retained file.
- `cargo run -p gororoba_cli_data --bin experiment-manifest -- verify
  <manifest.toml>` checks the complete manifest contract.
- `make experiment-manifest-verify EXPERIMENT_MANIFEST=<manifest.toml>` runs
  the same gate through the repository Makefile.

The verifier does not normalize input or output bytes. Scientific evidence
uses the exact retained bytes, so line-ending conversion is a content change
and must produce a new hash.

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
run_command = "cargo run -p example --release -- --seed 42"
run_command_sha256 = "<sha256 of run_command>"
toolchain = "rustc 1.97.0"
features = ["default"]
hardware = "x86_64 host CPU"
randomness_mode = "seeded"
random_seed = 42
random_generator = "ChaCha8Rng"

[[experiment.input_hashes]]
path = "data/external/manga/manga_rotcurves_all.csv"
sha256 = "deadbeef..."

[[experiment.output_hash_refs]]
artifact_id = "ASOT-0984"
path = "data/csv/apt_dimensional_census_summary.csv"
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
- `docs/engineering/evidence_ledger_operating_contract_2026_08_01.md` -- P2
  ledger row and closure conditions.
- `crates/gororoba_cli_data/src/bin/experiment_manifest.rs` -- executable
  validator and unit tests.
