# Brown 1972 Dissertation Roadmap (2026-03-27)

## Source packet

This roadmap is based directly on the Brown 1972 source packet cached off-repo:

- `~/Documents/Projects/CayleyDickson/tier1_core_cd_algebra/foundational_followups/brown_1972_zero_divisors_structure_cd.pdf`
- `~/Documents/Projects/CayleyDickson/tier1_core_cd_algebra/foundational_followups/brown_1972_extraction.md`

The goal is to keep the Brown lane source-driven rather than treating Chapter
III as the only meaningful next step.

## Full dissertation picture

Brown's dissertation has:

- 9 chapters
- 3 appendices
- bibliography and vita

Table-of-contents chapter layout:

1. Introduction
2. Review of Literature
3. Basic Concepts
4. The Cayley-Dickson Algebras
5. Exponent Properties
6. Basis Element Properties
7. Zero Divisors
8. Examples and Counter Examples
9. Conclusions

Appendices:

- A. Basis multiplication table for octonions
- B. Basis multiplication table for `A4`
- C. PL/1 program for character-string multiplication of Cayley-Dickson algebra elements

The table of contents exposes only two explicit internal lettered sections in
the main body:

- III.A Definitions
- III.B Basic consequences of the definitions

The later chapters appear in the dissertation as largely single-lane theorem
blocks rather than many nested subsections.

Chapters I and II are still part of the source-driven map, but they are not
currently treated as proof targets because the local dissertation text shows no
numbered definitions, theorems, lemmas, or corollaries in either chapter:

- Chapter I is an introduction and motivation chapter
- Chapter II is a literature-review and positioning chapter
- both matter for scope and provenance, not as immediate Rocq theorem surfaces

## OCR-derived numbered-result counts

The local PDF text extraction yields the following chapter-by-chapter counts of
numbered items.

| Chapter | Definitions | Theorems | Lemmas | Corollaries | Total |
|---------|-------------|----------|--------|-------------|-------|
| I | 0 | 0 | 0 | 0 | 0 |
| II | 0 | 0 | 0 | 0 | 0 |
| III | 14 | 3 | 5 | 2 | 24 |
| IV | 0 | 2 | 1 | 1 | 4 |
| V | 2 | 4 | 1 | 2 | 9 |
| VI | 7 | 7 | 6 | 3 | 23 |
| VII | 3 | 11 | 4 | 2 | 20 |
| VIII | 0 | 3 | 1 | 0 | 4 |

This gives an OCR-derived total of 84 numbered results. A rough whole-file scan
also finds 21 parenthesized equation references, but that number should be
treated as a lower bound until equation numbering is normalized more carefully.

Practical reading of the source:

- Chapter III is foundational, but not the whole Brown picture.
- Chapters VI and VII are the densest theorem carriers.
- Chapter V is already partially encoded in Rust and should not remain
  "crate-only" forever.
- Chapter IV is small but structurally important because it bridges from
  general CD construction to the later theorem lanes.
- Chapter VIII and Appendix C matter for concrete witness construction and
  historical computation, even if they are not the first proof targets.

## Current repo landing versus source

| Source slice | Current repo landing | Status |
|--------------|----------------------|--------|
| Chapter I | Brown lane header / roadmap only | contextual chapter, not a theorem surface |
| Chapter II | Brown lane header / roadmap only | literature chapter, not a theorem surface |
| Chapter III | `proofs/theories/Brown1972.v` + `crates/brown_1972/src/norm_symmetry.rs` | abstract Rocq surface plus standard-octonion and direct standard-sedenion witnesses for Brown 3.9 / 3.10 landed; broader generalized-norm surfacing still open |
| Chapter IV | `proofs/theories/Brown1972.v` plus local associator support | source-driven standard-tower witness surface for 4.2, 4.3, and 4.4 landed |
| Chapter V | `proofs/theories/Brown1972.v` + `crates/brown_1972/src/exponent_properties.rs` | Brown-numbered quaternion witness surface for 5.11 / 5.12 / 5.13 / 5.14 / 5.15 / 5.16 / 5.17 landed; broader generalized exponent surface still open |
| Chapter VI | `crates/brown_1972/src/basis_element_properties.rs` plus sign-table infrastructure | source present, Brown-numbered Rocq surface still open |
| Chapter VII | `ZD_Criterion.v`, `C1538_MorZDSymmetry.v`, `BrownAssessorEquivalence.v`, Rust support | partially formalized, but not yet chapter-complete in Brown numbering |
| Chapter VIII | no dedicated paper surface yet | open |
| Appendix C | `crates/brown_1972/src/pl1_emulator.rs` | Rust bridge only |

## Recommended source-driven implementation order

To avoid overfitting to Chapter III, Brown should now be mined in this order:

1. Keep Chapters I-II explicit but non-theorem
   - preserve their role as scope, motivation, and literature map
   - do not treat them as missing Rocq theorem lanes
2. Chapter V generalized exponent surface
   - extend the landed quaternion witness surface beyond the associative witness case
   - keep reusing `CDPowerAssociative.v` and related infrastructure
3. Chapter III broader generalized-norm surfacing beyond the landed standard-sedenion witnesses
   - extend the current abstract surface beyond the standard octonion witness
   - use the dissertation statements directly, not only the Rust checklist
4. Chapter VI basis-element surface
   - map Brown's basis identities onto the repo's sign-table and basis-index infrastructure
5. Chapter VII gap fill
   - continue from the already-landed zero-divisor lane to the remaining Brown-numbered theorems
6. Chapter VIII and Appendix C
   - extract explicit examples/counterexamples and connect the PL/1 program lane to the modern executable witness stack

## Why this reordering is better

This order follows the real dissertation shape more closely:

- IV and V build structural language that makes later Brown theorem naming cleaner.
- VI and VII are theorem-dense and highly relevant to the repo's existing
  computational infrastructure.
- III remains important, but it is only one foundational chapter inside a much
  broader Brown picture.

## Immediate next Brown proof tranche

The next Rocq tranche should therefore be:

1. keep the landed Chapter IV surface as the structural anchor
2. move next into the broader Brown Chapter V generalized exponent surface
3. then resume the broader Chapter III generalized-norm surfacing with the source text
   still open beside the proof file

## Remaining Brown steps

1. Chapters I-II
   - keep explicit in the paper map as context, not as theorem backlogs
2. Chapter IV
   - source-driven 4.2/4.3/4.4 witness surface is landed
   - next work there is only presentation cleanup if needed
3. Chapter V
   - broaden the new 5.11 / 5.12 / 5.13 / 5.14 / 5.15 / 5.16 / 5.17 quaternion witness surface toward Brown's fuller generalized exponent lane
4. Chapter III
   - extend from the landed octonion and standard-sedenion witnesses to the broader sourced generalized-norm lane
5. Chapter VI
   - mine basis-element identities theorem-by-theorem from the source packet
6. Chapter VII
   - continue from the already-landed zero-divisor surfaces to the remaining Brown-numbered theorems
7. Chapter VIII and Appendix C
   - add the example/counterexample surface and the PL/1 bridge
