# Cayley-Dickson Legacy Pre-1954 Roadmap (2026-03-27)

## Summary

This roadmap covers the older paper corpus already cached under
`~/Documents/Projects/CayleyDickson`, with proof-first prioritization.

Priority rule:

- source-complete papers with immediate theorem-lane payoff come first
- provenance-only cleanup stays behind theorem-lane work
- Brown 1972 remains the active next paper lane, but the older corpus is now
  ranked tightly enough that the post-Brown queue is decision-complete

## Ranked paper queue

| Rank | Paper family | Source status | Current repo landing | Next concrete tranche | Blocker |
|------|--------------|---------------|----------------------|-----------------------|---------|
| 1 | Hurwitz 1898 | exact source on disk | paired Rust + Rocq | extend from fixed-point / composition lane to fuller all-`n` converse and matrix classification surface | proof |
| 2 | Wedderburn 1914 | exact source on disk | Rocq paper surface | lift beyond the landed `n=2` cyclic/matric surfaces to the broader all-`n` primitive/matric theorem layer | proof |
| 3 | Dickson 1921 | exact source on disk | Rocq paper surface | extend the landed presentation-to-obstruction surfaces to fuller field-abstract obstruction semantics | proof |
| 4 | Dickson 1919 | exact source on disk | paired Rust + Rocq | strengthen the constructive tower/presentation handoff into the 1921 obstruction lane | proof |
| 5 | Dickson 1906 / 1912 / 1914 | exact / near-exact packets on disk | tracker + cache dossiers | split the early Dickson papers into explicit theorem lanes instead of leaving them as source cache only | proof + tracker |
| 6 | Zorn 1935 | source on disk | tracker only | create a paper lane for alternative / composition-adjacent structural results | proof |
| 7 | Jacobson 1939 | source on disk | tracker only | create a paper lane for pre-Schafer structural algebra results relevant to derivations and isotopes | proof |
| 8 | Albert 1942 | source on disk | tracker only | isolate the pre-Schafer structural results that feed later Brown / Schafer semantics | proof |
| 9 | Schafer 1945 | source on disk | paired Rust + Rocq | keep as completed predecessor lane; only refresh if Brown Chapter III/IV needs new references | low |
| 10 | Euler 1770 | source on disk | reference only | add a short provenance-backed chronology distillation row and, if useful, a sums-of-squares companion note | tracker |
| 11 | Degen 1818 | source on disk | reference only | add a short provenance-backed chronology distillation row and eight-square predecessor note | tracker |
| 12 | Hamilton / Graves / Cayley / Cockle 1835-1866 | source packet family on disk | reference only | distill into a single precursor-chain report plus a small chronology update, not a theorem lane | tracker |

## Immediate theorem-lane order after Brown Chapter III

1. Hurwitz 1898 converse / matrix-classification extension
2. Wedderburn 1914 all-`n` primitive/matric lift
3. Dickson 1921 fuller field-abstract obstruction semantics
4. Dickson 1906 / 1912 / 1914 split into explicit paper lanes
5. Zorn 1935, Jacobson 1939, Albert 1942 as distinct pre-Schafer structural lanes

## Completion notes for the cached precursor chain

These are older than Schafer and already cached, but they should stay
lightweight in-repo until the theorem-first papers above are further along:

- Euler 1770: keep as four-square ancestry, not a separate proof lane yet
- Degen 1818: keep as eight-square ancestry, not a separate proof lane yet
- Hamilton / Graves / Cayley / Cockle: maintain as an origin-chain chronology
  surface rather than fragmenting into many half-complete theorem stubs

## Brown-facing implications

The legacy audit changes the post-Schafer queue in one practical way:

- Brown 1972 Chapter III starts immediately
- after that, the highest-value older papers are no longer “open-ended legacy
  mining”; they are the ranked theorem lanes above
