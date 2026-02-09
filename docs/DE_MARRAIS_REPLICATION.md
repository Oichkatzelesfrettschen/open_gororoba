<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# de Marrais Replication (Computational)

**Primary source:** Robert P. C. de Marrais, *The 42 Assessors and the Box-Kites they fly* (2000),
arXiv:math/0011260.

Additional de Marrais first-party sources used for terminology/context are listed in
`docs/BIBLIOGRAPHY.md` (Complex Systems published versions of the Placeholder Substructures work, and
the Wolfram Science Conference 2004 slides).

This repo contains a **computational replication** of the central discrete structures described in the
paper, using the repository's fixed 16D Cayley-Dickson multiplication convention
(`gemini_physics.optimized_algebra.cd_multiply_jit`).

## What is replicated (and tested)

### 1) 42 primitive assessors

We model an "assessor" as an unordered pair of imaginary basis indices `(i,j)` (a 2-plane spanned by
pure imaginaries `e_i,e_j`) and its **two diagonals**:
- `e_i + e_j`
- `e_i - e_j`

We compute the **primitive** assessors by filtering the 56 "cross" pairs `(i,j)` with `iin{1..7}`,
`jin{8..15}` to those that participate in at least one diagonal zero-product with another cross pair.

Result: exactly **42** primitive assessors.

Validated by: `tests/test_de_marrais_boxkites.py`.

### 2) 168 primitive unit zero divisors

de Marrais' abstract states the "exact count of primitive unit zero-divisors" is **168**, arranged as
quartets along the 42 assessors. In this basis, each assessor yields 4 unit points:

`+/-(e_i + e_j)/sqrt2` and `+/-(e_i - e_j)/sqrt2`.

So `42 * 4 = 168`, and the repo verifies these are unit vectors.

Validated by: `tests/test_de_marrais_boxkites.py`.

### 3) 7 box-kites (octahedra)

We construct an undirected graph on the 42 assessors where an edge exists between two assessors if
there exists a sign choice for their diagonals such that the product is exactly zero:

`(e_i +/- e_j) * (e_k +/- e_l) = 0`.

Result:
- exactly **7** connected components,
- each component has **6** assessors,
- each component is the **octahedron graph** (6 vertices, 12 edges, degree 4 at every vertex).

Validated by: `tests/test_de_marrais_boxkites.py`.

### 4) Strut signatures (1..7) and trefoil/zigzag face counts

de Marrais' "Strut Table" associates each box-kite with a Roman numeral that is also its **strut
signature**: the unique octonion index `o in {1..7}` **missing** from the low indices of that box-kite's
six cross-assessors.

In addition, each octahedral box-kite has 8 triangular faces ("sails" + "vents"). The paper distinguishes
two lanyard sign-patterns for co-assessor trios:
- **trefoil**: two "+" edges and one "-" edge
- **triple zigzag**: three "-" edges

In this repo's convention (using diagonal-zero-product sign constraints per edge), each box-kite has:
- **6 trefoil** faces and **2 triple-zigzag** faces
- 3 trefoils + 1 zigzag among sails, and likewise among vents

Validated by: `tests/test_de_marrais_boxkites.py`.

### 5) Production Rule #1 (partial)

The paper's Production Rule #1 ("Three-Ring Circuits") defines a combinatorial way to construct a
third assessor `(E,F)` from two co-assessors `(A,B)` and `(C,D)` using XOR identities.

This repo implements the XOR construction as `gemini_physics.de_marrais_boxkites.production_rule_1`
and validates that, for every box-kite edge, it produces a third assessor that completes a valid co-assessor
trio (trefoil or triple-zigzag) under the repo's multiplication/sign conventions.

Validated by: `tests/test_de_marrais_boxkites.py`.

### 6) Production Rule #2 (partial)

The paper's Production Rule #2 ("Skew-Symmetric Twisting") describes a "twist" operation on two
co-assessors that yields two new assessors which are co-assessors with each other, but not with either
propagator.

This repo implements an operational version as `gemini_physics.de_marrais_boxkites.production_rule_2`
and validates that, for every box-kite edge `(a,b)`, the produced pair `(p,q)` satisfies:
- `(p,q)` have diagonal zero-products (are co-assessors),
- neither `p` nor `q` is a co-assessor with `a` or `b`.

Validated by: `tests/test_de_marrais_boxkites.py`.

### 7) Production Rule #3 (automorphemes / GoTo listings)

de Marrais introduces 7 "automorphemes" (his GoTo listings), each tied to one octonion O-trip (a Fano-plane
line) and an "exclude list" in the sedenion indices. The paper's "Behind the 8-Ball Theorem" states that
every such exclude list contains index 8 (and the three indices obtained by combining 8 with the O-trip),
so each O-trip pairs with exactly four remaining "O-copy" sedenions. This yields 12 assessors per O-trip.

The paper's Production Rule #3 implies a uniqueness property: for any cross-assessor `(o,S)` in the 42,
there is a **unique other** automorpheme (distinct O-trip) that also contains it.

This repo encodes these automorphemes directly from the 7 O-trips and the 8-ball exclude rule, and verifies:
- each automorpheme contains 12 assessors,
- their union is exactly the 42 primitive assessors,
- each primitive assessor lies in exactly 2 automorphemes,
- given one automorpheme for `(o,S)`, `production_rule_3` returns the unique other.

Validated by: `tests/test_de_marrais_automorphemes.py`.

## Where the replication lives

- Core implementation: `src/gemini_physics/de_marrais_boxkites.py`
- Tests: `tests/test_de_marrais_boxkites.py`
- CSV export: `src/export_de_marrais_boxkites.py` and `make artifacts-boxkites`
  - `data/csv/de_marrais_assessors.csv`
  - `data/csv/de_marrais_boxkites.csv`
  - `data/csv/de_marrais_boxkite_edges.csv`
  - `data/csv/de_marrais_strut_table.csv`

## What is *not* yet fully replicated

de Marrais also describes additional structure and "production rules" (trefoils / triple-zigzags /
Seinfeld production, strut signatures, etc.). Those require a careful, line-by-line encoding of the
paper's production rules and are **not** yet implemented here.
