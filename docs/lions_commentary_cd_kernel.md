# Lion's Commentary on the Cayley-Dickson Kernel

*After John Lions, "A Commentary on the UNIX Operating System" (1977)*

---

## Preface

This document is a pedagogical guide to the `cd_kernel::cayley_dickson` module
(5,681 lines of Rust across 14 source files). It follows Lion's principle:
**order by concept dependency, not by filename**. A reader proceeding linearly
from Section 1 through Section 5 will encounter each concept only after its
prerequisites have been explained.

The Cayley-Dickson kernel is the computational heart of the CD associator
diagnostic. It implements hypercomplex algebra multiplication at arbitrary
power-of-two dimensions, computes the trilinear associator norm as a sliding-
window observable, and optimizes the hot path through SIMD vectorization.

**Total**: 5,681 LOC. **Hot path**: `simd.rs` (906 LOC, 16% of total).
**Test surface**: `tests.rs` (1,473 LOC, 26% of total -- tests are specification).

---

## Section 1: Foundation (Algebraic Structure)

*"How hypercomplex algebras are built"*

### 1.1 `mod.rs` (61 LOC) -- API Surface

The module root re-exports the public API. Start here to understand what
contracts the kernel exposes. The key public functions are:

- `cd_multiply(a, b, dim)` -- allocating multiply
- `cd_multiply_f32_fused(a, b, out, dim, ws)` -- zero-allocation multiply (hot path)
- `associator_norm(a, b, c, dim)` -- the diagnostic
- `batch_sliding_associator_norms_f32(vectors, dim)` -- batch computation

### 1.2 `signs.rs` (248 LOC) -- The Primitive Truth

Every Cayley-Dickson algebra is completely determined by its sign table:

```
e_p * e_q = sign(p, q) * e_{p XOR q}
```

The product index `p XOR q` follows from the doubling construction. The sign
`+1` or `-1` is computed recursively through the tower levels. This recursion
mirrors the algebra itself: quaternion signs determine octonion signs, which
determine sedenion signs.

**Key function**: `cd_basis_mul_sign(dim, p, q) -> i32`

The sign table is NOT arbitrary. It must satisfy the constraints imposed by
the doubling formula. For dim <= 8, these constraints additionally enforce
the Moufang identities (alternative algebra). For dim >= 16, the constraints
are weaker and zero divisors become possible.

### 1.3 `signature.rs` (273 LOC) -- Parameterization

The standard Cayley-Dickson construction uses gamma = -1 at each doubling level:
`(a,b)* = (a*, -b)`. But the construction generalizes to any gamma in {-1, +1}.
Split signatures (gamma = +1) produce split algebras with different metric
properties.

**Key type**: `CdSignature` parameterizes the algebra.

### 1.4 `arith.rs` (260 LOC) -- Core Multiply

The Cayley-Dickson multiplication formula:

```
(a, b)(c, d) = (ac - conj(d)*b, d*a + b*conj(c))
```

This module provides both allocating (`cd_multiply -> Vec<f64>`) and
non-allocating (`cd_multiply_into(&mut out)`) variants. The non-allocating
path is critical: the associator calls multiply 4 times per triplet.

**Key insight**: Conjugation (`cd_conjugate`) negates all imaginary components.
This is O(dim) but memory-bound, not compute-bound.

---

## Section 2: Algebraic Properties (Non-Associativity Analysis)

*"How and why these algebras fail classical properties"*

### 2.1 `associator.rs` (404 LOC) -- The Core Diagnostic

The associator `[a,b,c] = (a*b)*c - a*(b*c)` measures non-associativity.
Its norm `||[a,b,c]||` is our primary physical observable.

For a sliding window of N embedded vectors, we compute:
```
norms[i] = ||[v[i], v[i+1], v[i+2]]||
```

This timeseries captures the degree of algebraic disorder in the multichannel
field data. Smooth fields -> small norms. Turbulent fields -> large norms.
Boundary crossings -> sharp peaks.

**Why 4 multiplies?** The two bracketed products `(a*b)*c` and `a*(b*c)` each
require one multiply for the inner product and one for the outer. No shortcut
exists because non-commutativity prevents algebraic simplification.

### 2.2 `zero_divisors.rs` (423 LOC) -- Algebraic Defects

Zero divisors are nonzero elements `a` with `a*b = 0` for some nonzero `b`.
They do not exist in division algebras (dim 1-8). At dim 16 (sedenions),
they appear for the first time. Their density increases with dimension.

**Connection to the 16D crossover**: Zero-divisor-bearing directions provide
genuinely non-associative information that generic multilinear forms cannot
access. This is the algebraic mechanism behind conjecture C-1628.

### 2.3 `predicates.rs` (417 LOC) -- Type Membership

Once non-associativity and zero divisors exist, element classification becomes
subtle. This module provides Moreno (1997) predicates: `is_alternative_element`,
`is_special_element`, `is_special_couple`, `is_special_triple`.

---

## Section 3: Specialized Dimensions

*"Explicit formulas for small, important algebras"*

### 3.1 `sedenion.rs` (121 LOC) -- 16D Explicit

The 35 defining triads of the sedenion algebra, enumerated explicitly. At this
dimension, explicit enumeration beats generic recursion. The triads are the
"primitive truth" of 16D -- every multiplication can be resolved by table lookup.

### 3.2 `trigintaduonion.rs` (178 LOC) -- 32D Explicit

At 32D, explicit triads become impractical. This module bridges between
table-driven and algorithmic approaches.

### 3.3 `cariow_factorization.rs` (278 LOC) -- Karatsuba for CD

Cariow (2012, 2013) showed that CD multiplication at each tower level can be
reduced from 4 recursive calls to 3, achieving O(N^1.58) instead of O(N^2).
The factorization works at the REAL-NUMBER level by analyzing the bilinear form.

For 32D-128D, this is faster than generic recursion but slower than hand-tuned
SIMD. It is a bridge algorithm for dimensions too large for explicit tables
but too small for parallel decomposition.

---

## Section 4: Implementation Optimizations (Performance Paths)

*"How to actually compute these algebras fast"*

### 4.1 `soa_cache.rs` (250 LOC) -- Memory Layout

Generic recursion produces fragmented memory access patterns. The SoA (Structure
of Arrays) transformation reorganizes data so that the vectorizer can see
opportunity. Instead of `[a0,a1,a2,a3, b0,b1,b2,b3]`, we store
`[a0,b0, a1,b1, a2,b2, a3,b3]`. This enables 4-wide SIMD loads without gather.

Translated from steinmarder's D3Q19 LBM SoA layout (2.85x speedup measured).

### 4.2 `simd.rs` (906 LOC) -- THE KERNEL

**This is where the algebra meets the hardware.**

Three tiers:
1. **Explicit base cases** (dim 2, 4, 8): zero recursion, hand-unrolled
2. **Fused recursive** (dim 16-128): workspace-based, zero allocation
3. **Parallel allocating** (dim 256+): rayon-parallel recursive halving

The hot function is `cd_multiply_f32_fused`. It is called 4 times per triplet,
~10K triplets per sliding window, potentially millions of times per analysis run.
Every nanosecond matters.

**"You are not expected to understand this"**: The f64x4 vectorization in the
quaternion base case reorders operations to minimize register pressure while
maintaining FMA(x*y + z) fusion. The layout is NOT the textbook Hamilton product.
It is algebraically equivalent but microarchitecturally optimized. See the
inline comments for the operation-by-operation correspondence.

### 4.3 `fast_associator.rs` (264 LOC) -- Batch Computation

Pre-allocated workspace for the sliding-window associator:
- `buf1, buf2, buf3`: 3 product buffers of size `dim`
- `mul_ws`: recursive multiply workspace

The 3-buffer overlap strategy eliminates one allocation per triplet:
1. `buf1 = a * b`
2. `buf2 = b * c`
3. `buf3 = buf1 * c`   (= (a*b)*c)
4. `buf1 = a * buf2`   (overwrite: a*b no longer needed; = a*(b*c))
5. `return ||buf3 - buf1||`

---

## Section 5: Measurement and Theory (Structural Properties)

*"What the algebra tells us about the data"*

### 5.1 `symmetry.rs` (125 LOC) -- Metrics

Gourlay friction, cross-generational metrics, and structure constants. These
observables quantify how much a signal "violates" algebraic structure, which
reveals anomalies. The metrics are value-level: they take computed products
and extract scalar invariants.

### 5.2 `tests.rs` (1,473 LOC) -- The Specification

**These tests are not afterthoughts. They are part of the specification.**

Key test families:
- `test_simd_matches_scalar`: validates all SIMD paths compute identically to scalar
- `test_associator_quaternion_zero`: verifies A=0 for dim <= 4
- `test_associator_octonion_nonzero`: verifies A > 0 for dim 8
- `test_zero_divisor_sedenion`: confirms ZDs exist at dim 16
- `test_fused_matches_allocating`: validates workspace reuse correctness

If any optimized path disagrees with the scalar reference, the optimization
is wrong -- not the test. This principle is inviolable.

---

## Cross-Reference Index

| Module | Calls | Called by |
|--------|-------|----------|
| `signs` | (none) | `arith`, `signature`, `sedenion` |
| `arith` | `signs` | `associator`, `simd`, `zero_divisors` |
| `associator` | `arith` | `fast_associator`, `symmetry`, binaries |
| `simd` | `arith` | `fast_associator`, `soa_cache` |
| `fast_associator` | `simd`, `associator` | all analysis binaries |
| `zero_divisors` | `arith`, `signs` | `predicates`, Moreno theorems |
| `predicates` | `arith`, `zero_divisors` | Moreno theorem verification |
| `tests` | (all modules) | (none -- terminal) |

---

## Acknowledgment

This commentary follows John Lions' principle that source code, properly
annotated, is the best documentation of a system. The original "Lion's
Commentary on UNIX 6th Edition" (1977) demonstrated that even a complete
operating system kernel can be made teachable through pedagogical ordering
and adjacent prose. We apply the same principle to a 5,681-line algebraic
kernel that, like UNIX, hides deep structure behind a deceptively simple API.
