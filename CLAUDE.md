# Claude Code project instructions

All agent and contributor guidance lives in `AGENTS.md`.
See that file for hard rules, build commands, project layout, and workflows.

---

## Research Methodology for This Codebase

open_gororoba is a research workbench, not a library.  It explores whether
algebraic structure in Cayley-Dickson algebras (real -> complex -> quaternion
-> octonion -> sedenion -> ...) can explain or predict phenomena in quantum
gravity, particle physics, and cosmology.  This means the codebase contains
a spectrum from **verified mathematics** to **speculative hypotheses**, and
every agent must respect that spectrum.

### The verification ladder

| Layer | Example | Standard of evidence |
|-------|---------|---------------------|
| **Algebraic (verified)** | Cayley-Dickson multiplication, zero-divisor census, de Marrais box-kite count | Unit tests against known results (Baez, Hurwitz, de Marrais) |
| **Mathematical (partially verified)** | Reggiani manifold dimension, spectral dimension flow, E8 root structure | Tests + literature cross-check; some gaps remain (e.g., PSL(2,7) geometric mapping) |
| **Physical (modeled)** | Gravastar stability, Tang mass ratios, holographic entropy scaling | Toy models with parameter fits; no empirical validation |
| **Speculative (hypothetical)** | Negative-dimension vacuum, sedenion-gravastar equivalence, associator-to-Standard-Model | Exploratory code only; claims tracked in docs/CLAIMS_EVIDENCE_MATRIX.md |

### When facing a hard problem in this repo

The algebraic and physical content here is dense.  Cutting corners produces
wrong answers that look plausible.  Follow this discipline:

**Research** -- Read the relevant Rust crate (under `crates/`), the Python
module (under `src/gemini_physics/`), the tests, and the claim entries in
`docs/CLAIMS_EVIDENCE_MATRIX.md`.  Trace the mathematics from definition to
computation to test assertion.  Identify what is proven, what is tested, and
what is merely conjectured.

**Comprehend** -- Build a model of the algebraic structure you are working
with.  Cayley-Dickson algebras lose properties at each doubling (ordering at
C, commutativity at H, associativity at O, norm composition at S, division
at 32D).  Zero-divisor geometry creates structure where division fails.
Understand which layer of the property-degradation ladder your task lives on.

**Scope** -- Separate the mathematical claim from the physical interpretation.
A zero-divisor census is algebra; mapping it to metamaterial layers is
speculation.  Keep these layers distinct in code, tests, and documentation.
If a change touches both layers, decompose it into algebraic and physical
sub-tasks with independent verification.

**Synthesize** -- Verify that your approach is consistent with the existing
verification ladder.  Algebraic code must pass deterministic tests.  Physical
models must declare their assumptions and parameter ranges.  Speculative
claims must be tracked in the claims matrix with explicit WHERE STATED and
WHAT WOULD REFUTE entries.

**Build out completely** -- Implement the full solution with tests, not a
sketch.  This codebase already has 150+ Python tests and 76+ Rust tests.
New code must meet the same standard: unit tests for algebraic properties,
null tests for statistical claims, convergence tests for numerical methods.
No placeholders, no deferred verification.

### What this means in practice

- Do not conflate "the algebra checks out" with "the physics is correct."
- Do not skip the claims workflow for new physical hypotheses.
- Do not add untested numerical code -- even toy models need convergence tests.
- Do not assume a result is known -- check `docs/BIBLIOGRAPHY.md` and the
  test suite before citing or depending on it.
- When genuinely stuck, trace the mathematics by hand before writing code.
  The Cayley-Dickson product is the foundation; everything else follows from it.

---

## References

- `AGENTS.md` -- full contributor guide (hard rules, build, layout, workflow)
- `docs/CLAIMS_EVIDENCE_MATRIX.md` -- master claims tracker (459 rows)
- `docs/MATH_VALIDATION_REPORT.md` -- algebraic verification status
- `docs/BIBLIOGRAPHY.md` -- external source citations
