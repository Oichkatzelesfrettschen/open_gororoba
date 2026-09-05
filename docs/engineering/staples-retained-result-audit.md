---
description: Independent checksum, grouping, convergence, and percentile verification for retained Staples calibration and refit records.
last_verified: 2026-09-04
---

# Retained Staples result audit

The independent auditor reads retained experiment records and recomputes their
checksums, seeded file assignments, bootstrap multiplicities, training RMS,
convergence conditions, metric differences, global minima, and summary
percentiles. The auditor does not fit models or prepare raw scoring data.

Run from the repository root after the production runner exits successfully:

```bash
CARGO_TARGET_DIR="$PWD/.cache/retained-result-audit-target" \
  cargo run --locked --release \
  --manifest-path data/output/audit/staples-calibration-grouped-refits/retained-result-audit/Cargo.toml \
  -- "$PWD" --complete --source-ref 291ce5023d4ae8ca30c5ee4535bd9152520233ef \
  > data/output/audit/staples-calibration-grouped-refits/completion-audit.json
```

The `--complete` check requires all five CV records, all 100 bootstrap records,
750 converged fits, and the retained summary. Omitting `--complete` audits the
completed bootstrap subset and reports `complete = false`, while withholding
conditional intervals. Both modes require all five CV records.

The isolated Cargo manifest pins rand 0.10.1, rand_chacha 0.10.0, serde_json
1.0.149 with `float_roundtrip`, and sha2 0.11.0. Its lockfile pins transitive
dependencies. The float parser feature preserves the payload bytes used for
checksums after JSON reload. Rust 1.97.0 compiled the auditor with warnings
as errors. By default, the auditor compares the live checkout against every
source hash in `dataset.json`. The explicit `--source-ref` option instead reads
those files from the full Git commit ID using Git with replacement objects
disabled. The auditor still reads results and the sealed protocol from the
live checkout and verifies every recorded hash. Missing Git objects and source
hash mismatches fail the audit. Stderr identifies the selected source boundary;
stdout retains the numerical completion report format.

The production commit above precedes the preparation-error retention repair.
Use that commit to audit the retained production records. Running the repaired
runner produces a different source identity and requires a separate output
directory. A replay of the original numerical execution requires a checkout of
the production commit and the admitted input files described by `replay.toml`.
