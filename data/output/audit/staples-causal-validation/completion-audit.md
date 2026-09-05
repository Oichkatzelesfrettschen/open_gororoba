# Retained causal campaign completion audit

`completion-audit.rs` independently verifies the completed THEMIS-A result
bundle. The validator checks the exact 129 root record names, each record's
payload hash and dataset identity, the protocol and compiled-source hashes,
all three widths and 19 control seeds, and every validation/final epoch.

The validator hashes and decodes all 813 support files. Each of the 22,263,526
records must preserve the common 1031-sample history, six-vector feature
window, strictly earlier feature timestamp, ordered decision timestamp, and
its file identity. Reconstructed labels must match the retained label digest
and positive count. Model training counts must equal the admitted 2007-2012
counts; holdout point counts must equal their own years. Every fit must
converge, use the fixed ridge, and retain the required coefficient dimensions.
All augmented models must share the baseline's six training means and scales.

All 20 tensor declarations must share the exact 1848-term support digest and
16-dimensional representation, with distinct coefficient digests. The audit
checks all summary point metrics, increments, and finite-ensemble ranks against
the corresponding point records. The declarations record support equality;
the runner's separate tensor construction checks establish coefficient/support
construction from the source implementation.

For the primary uncertainty calculation, the validator rebuilds all 12,000
panel increments from the retained AUC win matrices and the 2000 paired daily
multiplicity vectors. The validator recomputes the global minimum in each draw,
the interpolated 95 percent interval, and the threshold decision. RNG draws
remain retained inputs; the independent validator does not regenerate the
ChaCha stream. The validator does not refit features from raw vectors, establish
interday exchangeability, or validate external transfer.

The seven executable mutation probes reject missing and extra records,
contaminated training counts, failed convergence, incorrect width, and corrupt
payloads, and preserve an inconclusive verdict when an interval touches the
threshold. The probes mutate in-memory values and leave result files intact.

Run from the worktree root:

```text
clippy-driver @data/output/audit/staples-causal-validation/completion-audit-build.args --emit=metadata
rustc @data/output/audit/staples-causal-validation/completion-audit-build.args
.cache/staples-external-intake-build/completion-audit data/output/audit/staples-causal-validation/results > data/output/audit/staples-causal-validation/completion-audit-findings.json
```

The build arguments use pinned workspace dependency artifacts. On a fresh
cache, rebuild equivalent artifacts from the workspace lock and resolve their
paths. `serde_json` must enable `float_roundtrip`, matching the campaign:
parsing with the default feature set changes some floating-point values and
correctly fails the payload checksum comparison. Clippy's metadata emission
uses the configured output path, so the subsequent `rustc` invocation restores
the runnable executable. Both compilation checks treat warnings as errors.

`completion-audit-findings.json` records the passing counts and result.
`completion-audit-checksums.sha256` identifies the validator, arguments,
findings, and executable. The approximate interval remains positive and its
upper endpoint remains below 0.005. The separate algebra-specific comparison
retains the inconclusive status because its uncertainty intervals remain
unrun. The proposed bounded successor preserves both distinctions and the
blocked external intake.
