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
| Chapter III | `proofs/theories/Brown1972ChapterIII.v` + `crates/brown_1972/src/norm_symmetry.rs` | Brown 3.1 standard-octonion trace surface, direct standard-sedenion Brown 3.1 trace witnesses, Brown 3.3 / Lemma 3.7 standard-octonion involution-quadratic surface, abstract Rocq 3.9 / 3.10 surface with standard-octonion and direct standard-sedenion witnesses, a shared weaker quadratic/conjugation core instantiated for both octonions and sedenions, a generic trace/quadratic wrapper above that core with both octonion and sedenion instances, a quaternion trace/quadratic extension instance, an extended packaged Chapter III surface, and a reusable sourced Chapter III interface across quaternion/octonion/sedenion are landed; the next abstraction step is to lift that lane farther beyond the current tower packaging |
| Chapter IV | `proofs/theories/Brown1972ChapterIV.v` plus local associator support | source-driven standard-tower witness surface for 4.2, 4.3, and 4.4 landed |
| Chapter V | `proofs/theories/Brown1972ChapterV.v` + `crates/brown_1972/src/exponent_properties.rs` | generic one-generated/trace-zero Rocq exponent surface landed, instantiated at quaternion and octonion level, and now extracted chapterwise from the paper aggregator |
| Chapter VI | `proofs/theories/Brown1972ChapterVI.v` plus sign-table infrastructure and `crates/brown_1972/src/basis_element_properties.rs` | standard-octonion Brown 6.10 / 6.11 / 6.12 / 6.13 / 6.14 / 6.15 basis-associator surface landed, source-faithful standard-octonion Brown 6.16 / 6.17 anticommutator surface landed, a direct standard-sedenion adjoined-element / polynomial surface for 6.1 / 6.2 / 6.3 / 6.4 / 6.5 / 6.6 / 6.7 / 6.8 landed, that `6.4-6.7` lane is now also packaged both as a broader adjoined/polynomial lift interface above the literal `mkSed` coordinates and as a packaged decomposition-level `lo/hi` surface, proof-faithful constructive 6.9 witnesses are landed, a packaged adjoined/conjugation decomposition surface is landed, and those Chapter VI pieces are now also bundled as both an extended packaged surface and a reusable adjoined-interface anchor; Brown's printed 6.9 pointwise iff wording does not survive unchanged in the repo's literal standard-pair coordinates, so the current Rocq surface records the constructive implications and the family form Brown's p.35 proof actually uses; next source-driven tranche is any farther non-standard-model Chapter VI lift |
| Chapter VII | `ZD_Criterion.v`, `C1538_MorZDSymmetry.v`, `BrownAssessorEquivalence.v`, Rust support | partially formalized, but not yet chapter-complete in Brown numbering |
| Chapter VIII | no dedicated paper surface yet | open |
| Appendix C | `crates/brown_1972/src/pl1_emulator.rs` | Rust bridge only |

## Recommended source-driven implementation order

To avoid overfitting to Chapter III, Brown should now be mined in this order:

1. Keep Chapters I-II explicit but non-theorem
   - preserve their role as scope, motivation, and literature map
   - do not treat them as missing Rocq theorem lanes
2. Chapter V generalized exponent surface
   - a generic one-generated/trace-zero exponent surface is now landed, instantiated at quaternion and octonion level
   - that Chapter V lane now lives in `proofs/theories/Brown1972ChapterV.v`, keeping the paper aggregator thinner
   - lift beyond the current octonion 5.1 / 5.2 / 5.3 / 5.5 / 5.8 / 5.11 / 5.12 / 5.13 / 5.14 and 5.15 / 5.16 / 5.17 witnesses into Brown's broader generalized exponent lane
   - keep reusing `CDPowerAssociative.v` and related infrastructure
3. Chapter III broader quadratic/conjugation surfacing beyond the landed Brown 3.1 / 3.3 / Lemma 3.7 standard-octonion source surface and standard-sedenion witnesses
   - the weaker packaged quadratic/conjugation core is now landed and instantiated for both octonions and sedenions
   - a generic trace/quadratic wrapper above that core is also landed, with both standard-octonion and standard-sedenion instances
   - the currently landed Chapter III pieces are now bundled in `proofs/theories/Brown1972ChapterIII.v`, with a new quaternion trace/quadratic extension surface above the earlier octonion/sedenion core
   - continue lifting away from full norm multiplicativity toward Brown's actual quadratic/conjugation hypotheses
   - use the dissertation statements directly, not only the Rust checklist
4. Chapter VI basis-element surface
   - standard-octonion 6.10 / 6.11 / 6.12 / 6.13 / 6.14 / 6.15 basis-associator witnesses are now landed
   - standard-octonion 6.16 / 6.17 anticommutator witnesses are now landed
   - standard-sedenion adjoined-element / polynomial 6.1 / 6.2 / 6.3 / 6.4 / 6.5 / 6.6 / 6.7 / 6.8 plus proof-faithful constructive 6.9 witnesses are now landed
   - the `6.4-6.7` adjoined/polynomial lane is now also packaged as a broader lift interface above the literal standard-sedenion coordinates
   - the same lane now has a packaged decomposition-level `lo/hi` surface above the explicit standard embeddings
   - the currently landed Chapter VI pieces are now bundled in `proofs/theories/Brown1972ChapterVI.v`, with a new packaged adjoined/conjugation decomposition surface above the earlier adjoined/polynomial lift
   - next fold is any farther non-standard-model Chapter VI lift
   - then lift beyond the current octonion witness layer
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
2. keep the landed generic Chapter V one-generated/trace-zero surface as a reusable exponent anchor
3. then split the fresh follow-on work:
   - Chapter VI follow-on: farther non-standard-model lifting on top of the current 6.1 / 6.2 / 6.3 / 6.4 / 6.5 / 6.6 / 6.7 / 6.8 / 6.9 / 6.10 / 6.11 / 6.12 / 6.13 / 6.14 / 6.15 / 6.16 / 6.17 landing and the new reusable adjoined-interface anchor
   - Chapter III follow-on: broader quadratic/conjugation surfacing on top of the landed 3.1 / 3.3 / Lemma 3.7 source layer and the new reusable sourced Chapter III interface

## Reusable anchor modules for the wider corpus

The split Brown files are no longer only Brown-local cleanup; they are now
intended anchor modules for later paper mining:

- `proofs/theories/Brown1972ChapterIII.v` is the current reusable Rocq anchor
  for trace/quadratic/conjugation statements that recur across the broader
  Cayley-Dickson literature.
- `proofs/theories/Brown1972ChapterVI.v` is the current reusable Rocq anchor
  for adjoined-element, decomposition, and trace-conditioned reassociation
  statements that later zero-divisor papers repeatedly touch.
- `proofs/theories/Brown1972ChapterV.v` is the current reusable Rocq anchor
  for one-generated exponent arguments and their quaternion/octonion witnesses.

## Completed 42-step tranche

1. Brown 6.15 was landed as a source-faithful sign-conditioned octonion basis reassociation theorem on top of Brown 6.14.
2. Brown 6.16 was landed as a source-faithful octonion anticommutator coordinate formula.
3. Brown 6.17 was landed as a source-faithful octonion anticommutator zero criterion with the paper's nonzero hypotheses kept explicit.
4. The Chapter VI octonion surface now extends through 6.17, while the Chapter III side now has a weaker shared octonion/sedenion quadratic/conjugation core rather than only the older stronger norm-oriented abstraction.

## Remaining Brown steps

1. Chapters I-II
   - keep explicit in the paper map as context, not as theorem backlogs
2. Chapter IV
   - source-driven 4.2/4.3/4.4 witness surface is landed
   - next work there is only presentation cleanup if needed
3. Chapter V
   - broaden the landed generic quaternion/octonion one-generated/trace-zero surface toward Brown's fuller generalized exponent lane
4. Chapter III
   - extend from the landed Brown 3.1 / 3.3 / Lemma 3.7 standard-octonion source surface, the weaker shared octonion/sedenion quadratic/conjugation core, and the octonion/sedenion 3.9 / 3.10 witnesses to the broader sourced quadratic/conjugation lane
5. Chapter VI
   - standard-octonion 6.10 basis anticommutation, 6.11 basis alternativity, 6.12 repeated-basis associator vanishing, 6.13 basis conjugation, 6.14 sign-controlled basis reassociation, 6.15 sign-conditioned basis reassociation, and 6.16 / 6.17 anticommutator surfaces are landed
   - direct standard-sedenion adjoined-element / polynomial 6.1 / 6.2 / 6.3 / 6.4 / 6.5 / 6.6 / 6.7 / 6.8 plus proof-faithful constructive 6.9 witnesses are now landed
   - the `6.4-6.7` lane now also has a broader adjoined/polynomial lift surface above literal standard coordinates
   - next target is any farther non-standard-model lift beyond the current octonion witness surface and current Chapter VI adjoined-element / polynomial surfaces
6. Chapter VII
   - continue from the already-landed zero-divisor surfaces to the remaining Brown-numbered theorems
7. Chapter VIII and Appendix C
   - add the example/counterexample surface and the PL/1 bridge
