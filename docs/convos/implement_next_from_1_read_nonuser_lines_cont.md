<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_convos.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_convos.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_convos.toml -->

# Implement-next backlog (from convos/1_read_nonuser_lines_cont.md)

Scope:
- This backlog is extracted from the audited transcript `convos/1_read_nonuser_lines_cont.md`.
- The transcript contains many unsourced, notebook-fragile narratives; implement only what is testable.

Pointers:
- Chunk audit: `docs/convos/audit_1_read_nonuser_lines_cont.md`
- Keyword scan: `docs/convos/keywords_1_read_nonuser_lines_cont.md`
- Concept index: `docs/CONVOS_CONCEPTS_STATUS_INDEX.md`

## Near-term (highest value)

- [ ] CX-017 (wheels): disambiguate "wheel algebra" usage and cite first-party sources for wheels-as-division-by-zero.
- [ ] CX-017 (wheels): implement a minimal wheel axioms checker and unit tests (keep scope separate from Cayley-Dickson ZDs).
- [ ] CX-019 (viz hygiene): eliminate warning-prone patterns in repo plotting code:
  - no divide-by-zero grids (mask or redesign),
  - no Matplotlib deprecations/warnings under `PYTHONWARNINGS=error`,
  - avoid unsupported MathText (no `\\text{...}` in Matplotlib labels).
- [ ] CX-019 (viz hygiene): replace `plt.show()` in artifact scripts with deterministic file outputs under `data/artifacts/`.
- [ ] CX-020 (atlas): decide whether to adopt the "plates/atlas" framing:
  - if yes, unify existing export scripts under one `make atlas` target and validate via `src/verification/verify_generated_artifacts.py`.

## Cleanup of false/ambiguous claims (docs-only, source-first)

- [ ] CX-005: ensure no repo-authored doc treats "negative dimension" as `x -> 1/x` inversion symmetry.
- [ ] CX-010: add a short, sourced note clarifying that "surreal time" rhetoric in the transcript is not surreal number theory.
- [ ] CX-007: correct and source any exceptional-algebra claims inherited from the transcript (e.g., avoid stating
      an unsupported "Cl(8) -> E6 -> E7 -> E8" embedding chain).

## Longer-term (optional)

- [ ] If we want "Plate 5/6" topics as real work:
  - Plate 5: define a concrete operator/tensor and compute a real spectrum with tests.
  - Plate 6: define a representation + projection map onto a Lie algebra basis and add tests (start with su(2)).

## Quality gate (required before expanding scope)

- [ ] `make ascii-check`
- [ ] `PYTHONWARNINGS=error make check`
