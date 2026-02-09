<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Wheels (Division by Zero) - Source-first notes

Primary sources:
- Jesper Carlstrom (2001), "Wheels - On Division by Zero", Research Reports in Mathematics 2001:11,
  Stockholm University (published 2001-09-10). PDF: https://www2.math.su.se/reports/2001/11/2001-11.pdf
- Jesper Carlstrom (2004), "Wheels - on division by zero", Mathematical Structures in Computer Science
  14(1), 143-184. DOI: 10.1017/S0960129503004110.

This doc exists because the convos repeatedly mention a "wheel algebra" while mixing in graph
language and (sometimes) operad/category language. For this repo, the intended meaning is:
wheels as a division-by-zero algebraic structure (Carlstrom).

Hard rule:
- Do not treat the word "wheel" as a single concept. Disambiguate:
  - wheels (division-by-zero algebra: Carlstrom/Setzer),
  - wheel graphs (graph theory),
  - wheeled operads/PROPs (operad theory).

## 1. Definition (Carlstrom 2001, Definition 1.1)

Carlstrom defines a wheel as a structure <H, 0, 1, +, *, /> with:
- <H, 0, +> a commutative monoid, and
- <H, 1, *, /> a commutative monoid with involution / (the "reciprocal").

The axioms are given as equations (numbering follows the report):
1) <H, 0, +> is a commutative monoid
2) <H, 1, *, /> is a commutative monoid with involution /
3) (x + y)*z + 0*z = x*z + y*z
4) x/y + z + 0*y = (x + y*z)/y
5) 0*0 = 0
6) (x + 0*y)*z = x*z + 0*y
7) /(x + 0*y) = /x + 0*y
8) x + 0/0 = 0/0

Notes:
- In wheels, 0*x = 0 does NOT hold in general (this is the key tradeoff for total division).
- The report derives additional useful identities (e.g. /1 = 1 and x/x = 1 + 0*x/x).

## 2. Relation to Cayley-Dickson zero divisors (scope boundary)

Cayley-Dickson algebras (beyond 8D) are non-associative and (from 16D onward) have zero divisors:
there exist nonzero a,b with a*b = 0. This is an internal multiplicative pathology of that algebra.

Wheels are different objects:
- They are commutative and focus on making division total.
- They do not automatically remove or explain zero divisors in noncommutative/nonassociative algebras.

Therefore, any claim of the form:
"wheels explain/resolve Cayley-Dickson zero divisors"
is treated as a hypothesis and must be sourced and tested before it is accepted into the repo narrative.

## 3. Minimal implementable next step

To bring wheels into the repo without overclaiming:
- Implement a tiny "wheel axioms checker" that validates Carlstrom's axioms (3)-(8), plus the required
  commutative-monoid/involution laws.
- Add unit tests that confirm the axioms on at least one concrete wheel model.
- Only after that, propose any mapping to Cayley-Dickson language as an interpretation layer, clearly
  labeled as speculative unless a first-party source establishes the connection.
