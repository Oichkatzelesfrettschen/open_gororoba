<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# TSCP Trial-Factor Ledger (2026-02-04)

Scope: method-level multiple-testing accounting for TSCP alignment (C-407).

## Tuned degrees of freedom (current proxy)

- Reference kite index: 7 choices
- Opposite-pair axis per kite: 3 choices (0,5), (1,3), (2,4)
- Embedding sweep size: N_embed = 21

## Familywise alpha

- alpha_family = 0.01

## Thresholds by scenario

| Scenario | N_trials | Bonferroni alpha_per | Sidak alpha_per | Notes |
|---|---:|---:|---:|---|
| Confirmatory: fixed embedding, per-kite scan | 7 | 1.429e-03 | 1.435e-03 | Fix embedding a priori (canonical projection), then test 7 kites. |
| Robustness: embedding sweep (42-node score) | 21 | 4.762e-04 | 4.785e-04 | Scan 21 embeddings (reference kite * opposite-pair axis), no per-kite restriction. |
| Exploratory: embedding sweep + per-kite scan | 147 | 6.803e-05 | 6.837e-05 | Scan 21 embeddings and 7 per-kite restrictions; report best result. |

## Source anchors (cached)

- Bonferroni/Sidak background: `data/external/papers/abdi_2007_bonferroni_sidak_corrections.pdf`
- Holm procedure: `data/external/papers/holm_1979_sequentially_rejective_multiple_test_procedure.pdf`

## Notes

- Bonferroni is conservative but robust under dependence.
- Sidak assumes independent trials; use as an informative bound only.
- This ledger must be updated whenever additional tuned degrees of freedom are introduced (catalog cuts, smoothing scales, coordinate frames, etc.).
