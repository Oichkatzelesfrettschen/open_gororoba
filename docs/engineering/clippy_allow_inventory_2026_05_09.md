# `#[allow(clippy::*)]` inventory (2026-05-09, DEBT-TEST-2 Phase A)

This document inventories the 143 `#[allow(clippy::*)]` annotations across
the crates/ tree (anchored repo-audit count) and groups them by lint name
so contributors can decide which are defensible (keep with rationale) and
which should be fixed properly.

## Distribution by lint

| Count | Lint                                | Disposition (default)         |
|------:|-------------------------------------|-------------------------------|
|    69 | clippy::needless_range_loop          | usually defensible: index-arithmetic clarity in physics loops |
|    54 | clippy::too_many_arguments           | usually defensible: simulation parameter packs; consider struct refactor only when the call site is hot |
|     7 | clippy::type_complexity              | review: most are generics in mid-level abstractions; bundle a type alias |
|     6 | clippy::approx_constant              | KEEP: literal values like 0.318 that look like 1/pi but are measurement errors; comment is required |
|     4 | clippy::should_implement_trait       | KEEP: methods named `from`/`into`/`add`/`mul` with custom non-trait semantics |
|     3 | clippy::mem_replace_with_uninit      | review: unsafe mem swaps; verify alignment-safety hand-written |
|     3 | clippy::large_enum_variant           | review: physics result enums with one variant carrying a big array; box if hot |
|     2 | clippy::excessive_precision          | KEEP: scientific constants where the trailing digits matter |
|     1 | clippy::manual_abs_diff              | review: trivial fix to `.abs_diff()` if available |
|     1 | clippy::items_after_test_module      | KEEP: test module organization choice |
|     1 | clippy::cast_sign_loss               | KEEP: documented narrowing in known-positive math |
|     1 | (other / multi-line)                 | inspect manually |

Total: 152 entries (anchored grep). The repo-audit count of 143 differs by
9 due to multi-line `#[cfg_attr(..., allow(clippy::...))]` forms and other
attribute layouts not matched by the simple regex; the 9-entry gap is
extraction artifact, not lost data.

## Recommended action

The default disposition is to **leave most allows in place** but require
that each one carries a one-line rationale comment explaining why the
lint is suppressed at this site. Today:

- ~10 entries (mostly `approx_constant`, `cast_sign_loss`,
  `excessive_precision`) already have inline rationale comments.
- The other ~140 lack site-specific rationale.

The right work is therefore NOT a mass fix-the-lint sweep, but a
documentation sweep adding `// allow: <reason>` comments. Each takes
under a minute; a single sprint can cover the whole set.

## Quick-win category: clippy::needless_range_loop (69)

The lint suggests `.iter().enumerate()` instead of `for i in 0..n`. In
physics inner loops with computed multidimensional indices (i, j, k)
where the index is the actual data, the iterator form is harder to read.
The standard rationale is "index arithmetic semantically central". A
single-template comment + grep-fix is feasible but lower priority than
SAFETY annotation work.

## Quick-win category: clippy::too_many_arguments (54)

Large parameter lists (more than 7) are flagged. In LBM kernels and
spectral analysis, the parameter count reflects the math (e.g.,
solver(rho, u, v, w, omega, tau, force_x, force_y, force_z)). The
right answer is rarely a parameter struct -- it disrupts the inner-loop
inlining. Standard rationale: "parameter list mirrors the underlying
equation".

## Action items

1. Document which sites are KEEP vs review vs fix (this doc is the
   start; per-site dispositions belong in code comments).
2. Add a `// allow:` rationale comment to each entry that lacks one.
3. After the comment sweep, the `#[allow(clippy)]` count should not
   shrink dramatically -- the goal is rationale coverage, not lint
   reduction.
4. The 12 `review` entries (type_complexity, mem_replace_with_uninit,
   large_enum_variant, manual_abs_diff) deserve a focused look; some
   may be cleanly fixable.

## Acceptance criteria for closing DEBT-TEST-2

- Every `#[allow(clippy::*)]` in crates/ has either an inline comment
  on the same or preceding line or a justification in the enclosing
  function's docstring.
- The repo-audit `allow_clippy_attrs` count tracks the rationale-
  coverage metric, not just the raw allow count.
- New allow attrs require a rationale at code-review time (this is a
  process change, not a code change).

## See also

- `data/output/debt_baseline_2026_05_09.toml` (anchored counts).
- `crates/data_core/src/catalogs/desi_bao.rs:134` (canonical example
  of a documented `clippy::approx_constant` allow).
