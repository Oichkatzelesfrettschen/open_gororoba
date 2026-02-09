<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/book_docs.toml -->

# Claims Evidence Matrix

The claims evidence matrix tracks 494 claims (C-001 through C-494) in
`registry/claims.toml`.  Each claim records a testable statement about the
codebase's mathematical or physical results.

## Claim structure

Each claim entry contains:

- **id**: C-nnn identifier
- **statement**: The testable assertion
- **status**: Canonical claim status token (Verified, Refuted, Open, Partial,
  Pending, Superseded, Established, Inconclusive, Closed/*, etc.)
- **where\_stated**: Source reference (paper, conversation, analysis)
- **test**: Rust test function that verifies the claim (when applicable)

## Status distribution

The bulk of claims fall into these categories:

- **Verified**: Deterministic test passes against known results
- **Refuted**: Test demonstrates the claim is false (e.g., C-455 E8 connection)
- **Open**: Not yet tested, or test is pending implementation
- **Partial**: Some aspects verified, others remain open

## Validation

The `registry-check` binary validates all claims for:

- Sequential ID gaps
- Valid status enum values
- Consistency with project.toml counts

```sh
cargo run --release --bin registry-check
```

## Auto-generated appendix

The `generate-latex` binary produces `docs/latex/claims_appendix.tex`, a
claims longtable from claims.toml.  This ensures the LaTeX documents
stay synchronized with the TOML source of truth.

## Markdown mirror

`docs/CLAIMS_EVIDENCE_MATRIX.md` is a TOML-generated markdown mirror.
Operational authoring is TOML-first (`registry/claims.toml` -> markdown export).
`migrate-claims` is retained for explicit legacy/bootstrap import flows only.
