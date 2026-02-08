# Motif Scaling Laws

Exact enumeration of the zero-product graph's connected components across
5 Cayley-Dickson doublings reveals strict scaling laws (I-006).

## Scaling formulas

| Invariant | Formula | Verified dims |
|-----------|---------|---------------|
| Components | dim/2 - 1 | 16--256 |
| Nodes per component | dim/2 - 2 | 16--256 |
| Motif classes | dim/16 | 16--256 |
| K2 components | 3 + log2(dim) | 16--256 |
| K2 part count | dim/4 - 1 | 16--256 |

## Key observations

- **No octahedra beyond dim=16.**  The K(2,2,2) structure that characterizes
  sedenion box-kites vanishes at the first doubling.

- **Complete restructuring at each doubling.**  At dim=32, the structure
  reorganizes into 8 heptacross K(2,2,2,2,2,2,2) and 7 mixed-degree
  components.

- **The pathion cubic anomaly** (I-012): The dim=32 8/7 motif split requires
  a degree-3 GF(2) polynomial, NOT degree 4.  Degrees 1 and 2 fail.

## Computation

All motif census computations are exact (not sampled) and complete in under
2 seconds for dim=256 in release mode.  The census binary:

```sh
cargo run --release --bin motif-census -- --dims 16,32,64,128,256 --details
```

Implementation: `crates/algebra_core/src/boxkites.rs` (16 regression tests
covering dim=64,128,256).

## Claims

C-126 through C-130: Scaling law formulas.
C-445 through C-448: XOR filter properties.

## Experiment

[E-001: Cayley-Dickson Motif Census](../experiments/e001.md)
