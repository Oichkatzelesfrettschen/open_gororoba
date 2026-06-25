---
description: Audit for C-037, the Barbero-Immirzi, F4 Casimir, and Gauss-Bonnet near-coincidence
last_verified: 2026-06-25
status: active
---

# C037 Numerical Coincidence Audit

C-037 tested whether the Barbero-Immirzi parameter `gamma`, the F4 Casimir
ratio `epsilon`, and a Gauss-Bonnet coupling scale form a single structural
identity near one quarter. The registered result is refuted.

## Claim

The rejected claim was:

```text
gamma ~= epsilon ~= 4 * lambda_GB ~= 1/4
```

The F4 ratio is exact in the project normalization:

```text
epsilon = C2(26) / |Delta+(F4)| = 6 / 24 = 1/4
```

That exact algebraic identity does not force the other two quantities.

## Audit Result

The near-equality is a numerical coincidence, not a shared derivation.

- `epsilon` is an exact F4 root-system ratio and is verified by
  `proofs/verified/C035_CasimirQuarter.v`.
- `gamma` belongs to loop quantum gravity area-spectrum normalization and is
  not derived in this repository from F4.
- `lambda_GB` is a coupling in modified-gravity models and is not fixed by the
  F4 Casimir ratio.

## Falsifier

The conjecture would need an independent derivation that maps the same
normalization convention into all three quantities without fitting the value
after the fact. The registered proof surface `proofs/verified/C037_EpsilonNotGamma.v`
anchors the present negative result.

## Registry Links

- `registry/claims.toml` entry C-037
- `docs/EXCEPTIONAL_COSMOLOGY.md`
- `docs/external_sources/EXCEPTIONAL_COSMOLOGY_SOURCES.md`
