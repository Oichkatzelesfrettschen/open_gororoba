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
  -- "$PWD" --complete \
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
as errors. The source checkout must match the source hashes in `dataset.json`.
