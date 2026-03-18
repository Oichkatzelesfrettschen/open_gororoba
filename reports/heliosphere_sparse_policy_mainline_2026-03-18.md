# Heliosphere Sparse Policy Mainline Summary

## Summary

The promoted mainline sparse-policy candidate for the `modern2020` reference cube was
`mission_quiet|full`.

This candidate improved the supervised `modern2020` sparse-policy lane relative to the
current invariant-only comparator while staying under the hard `12 GiB` projected
`1024^3` sparse BF16 A-A budget across the evaluated split seeds.

The promotion did **not** survive the full cross-cube gate, so no claim or insight
uplift was performed from this run.

## modern2020 Improvement

Reference cube:
- `reports/heliosphere_feature_cube_modern2020_2026-03-15.csv`

Comparator policy:
- `mission_product_quiet|invariants_only`

Promoted candidate:
- `mission_quiet|full`

Three-seed aggregate comparison (`split_seed = 0, 1, 2`):

| Metric | Comparator | Promoted |
| --- | ---: | ---: |
| Mean active fraction | `0.164452` | `0.144496` |
| Mean event-label recall | `0.504566` | `0.616438` |
| Mean event-label precision | `0.028391` | `0.039969` |
| Mean projected GiB | `11.511656` | `10.114729` |
| Max projected GiB | `11.565385` | `10.911765` |
| Mean median lead time hours | `5.333333` | `8.166667` |

So for `modern2020`, the promoted candidate was better on all of the quantities that
matter here:
- lower projected memory
- lower active fraction
- higher recall
- higher precision
- longer median lead time

## Why It Was Not Promoted Repo-Wide

The cross-cube generalization gate still failed.

On the unsupervised stress cubes:
- `imap2026`: `mission_quiet|full` regressed to `70.0 GiB` projected memory and failed
  the unsupervised memory check relative to the invariant comparator
- `inner1976`: `mission_quiet|full` also regressed to `70.0 GiB` projected memory and
  failed the same check

Because of that:
- `promotion_survives_all_cubes = false`
- `registry_update_performed = false`

## Interpretation

The useful result is narrower than a full promotion, but it is still real:

- `mission_quiet|full` is a better sparse-policy candidate for the labeled
  `modern2020` lane than the current invariant-only comparator
- that gain is not yet stable enough to declare a general cross-cube mainline winner

This means the next iteration should treat `modern2020` as a confirmed local success
case and `imap2026` / `inner1976` as the blockers that still need a more robust policy
design.
