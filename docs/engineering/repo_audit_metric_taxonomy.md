# repo-audit metric taxonomy

This document explains every metric reported by the `repo-audit` binary
(at `crates/gororoba_cli_data/src/bin/repo_audit.rs`), how it is
counted, and what counts as "real debt" versus "valid suppression".

The audit replaces ad-hoc grep heuristics that produced phantom counts
(the original 2026-04-30 baseline reported 142 TODO markers when only
3 existed in code; the rest were comment-stripped artifacts). Every
metric below is anchored to a definition that can be reproduced from
the binary source.

## Anchoring strategies

The audit uses three different anchoring strategies depending on what
the lint is trying to catch:

- **Comment-and-string-stripped**. Source is preprocessed via
  `strip_rust` to replace block comments, line comments, and string /
  char / byte literals with whitespace of the same length. The result
  preserves line numbers (so other tools can cross-reference) but
  removes all text that should not match an attribute pattern. Used
  for all attribute-style lints (`unsafe`, `ignore`, `allow(clippy::*)`,
  `allow(dead_code)`, `unimplemented!`, `todo!`, `unreachable!`).

- **Original-source, line-anchored**. Used for lints that are themselves
  comments. `safety_comments` matches `^\s*(?://|/\*)\s*SAFETY\s*:` and
  `todo_fixme_xxx_hack` matches `^\s*(?://|/\*)\s*(?:TODO|FIXME|XXX|HACK)\b`
  on the original (un-stripped) source.

- **Rocq line-anchored**. Used for `.v` files. Patterns are anchored
  to line start with optional leading whitespace to distinguish
  top-level `Axiom`/`Parameter` from indented (signature-level)
  occurrences.

## Metric reference

### File counts

- `rust_files`: total `.rs` files walked.
- `rocq_files`: total `.v` files walked.
- `other_files`: every other file under the walked roots.

### Rust attribute counts (post-strip)

- `unsafe_blocks`: number of `unsafe { ... }` blocks. Real debt unless
  paired with a SAFETY comment. Compare against `safety_comments` for
  coverage.
- `safety_comments`: lines matching `// SAFETY:` or `/* SAFETY:`. Each
  unsafe block should have one. Coverage = `safety_comments / unsafe_blocks`.
- `ignore_attrs`: number of `#[ignore]` test attributes. Each is either
  a real test debt (test was skipped without explanation) or a real
  reason (GPU only, network only). Compare against the test suite's
  `--ignored` opt-in lane to see how many get exercised.
- `allow_clippy_attrs`: total `#[allow(clippy::*)]` annotations. NOT
  the debt count -- many are legitimate suppressions for multi-cursor
  matrix loops, physics-constant precision, etc.
- `allow_clippy_unjustified`: subset of `allow_clippy_attrs` that lack
  any justification comment (no trailing `// ...` on the same line and
  no `// ...` line directly above the attribute). **This is the actual
  debt count**. Regression-gated by `repo-audit --strict-unjustified`.
- `allow_dead_code_attrs`: `#[allow(dead_code)]` count. Mostly
  acceptable in test scaffolding and feature-gated code; flag for
  case-by-case review.
- `todo_fixme_xxx_hack`: count of source-comment TODO/FIXME/XXX/HACK
  markers in original source (not stripped). Stage B reduced this from
  3 to 0 by reframing each as a TaskList item.
- `unimplemented_macros`: `unimplemented!()` macro calls. Each is a
  panic site; flag for refactor.
- `todo_macros`: `todo!()` macro calls. Same.
- `unreachable_macros`: `unreachable!()` macro calls. Often legitimate
  (exhaustive match arms); high count is informational not actionable.

### Rocq counts (line-anchored on `.v`)

- `rocq_admitted_strict`: `^\s*Admitted\s*\.` -- proofs explicitly
  conceded. Hard zero target.
- `rocq_admit_strict`: `^\s*admit\s*\.` -- partial admits. Hard zero
  target.
- `rocq_axiom_strict`: `^Axiom\b` (top-level, no indentation). These
  are in module bodies and should ideally be theorems. Stage B audit
  classified all 32 occurrences; most are necessary external-theorem
  axiomatizations.
- `rocq_axiom_indented`: `^\s+Axiom\b` (inside a module type signature).
  These declare interface obligations that an implementer must satisfy;
  much weaker than top-level axioms.
- `rocq_parameter_strict`: `^Parameter\b`. Like axiom_strict but for
  values rather than propositions.
- `rocq_parameter_indented`: `^\s+Parameter\b`. Like axiom_indented.

## Justified vs unjustified detection

The justified-vs-unjustified split for `allow_clippy_attrs` works on
the original source (un-stripped) so it can read the comments. An
attribute is "justified" if:

1. The attribute line has a trailing `// ...` comment AFTER the closing
   bracket, OR
2. The line directly above the attribute is itself a `// ...` comment.

A multi-line preceding doc comment block (`/// ...`) only counts if the
line directly above the attribute is the closing line of that block.
This intentionally undercounts justifications a little: the goal is to
push contributors to leave the rationale near the suppression, not
buried five lines up.

False negatives: an attribute placed on a closing brace line with no
intervening comment can still appear unjustified even when the
function's doc comment explains why; the fix is to add a one-line
comment immediately above the attribute.

False positives: an unrelated comment above an attribute (e.g., a
section header) can mark it as justified. This is acceptable because
the policy is "comment near the attribute"; specific content is not
enforced.

## SQLite revisions integration

When invoked with `--sqlite registry/canonical/control_plane.sqlite3`,
the audit also reads the canonical revisions audit trail and emits a
`[revisions]` block:

```toml
[revisions]
sqlite_path = "registry/canonical/control_plane.sqlite3"
claim_revisions = 1299
insight_revisions = 0
experiment_revisions = 35

[revisions.by_field]
formal_proof = 1297
status_note = 37

[[revisions.top_actors]]
actor = "eirikr"
revisions = 1334
```

This synthesizes the static debt count (what the repo currently is)
with the dynamic mutation flow (what changed to produce it). A reader
sees both surfaces in one report with a shared timestamp.

## Baseline comparison

`repo-audit --baseline-compare <prior.toml>` emits a `[baseline_delta]`
block listing per-class growth and shrinkage. With `--strict`, the
binary exits non-zero if any class grew, except for the
"safety-positive" classes (`safety_comments`) where growth is good.
`--strict-unjustified` (planned, see TaskList #94) gates only on
`allow_clippy_unjustified` so the safety-positive split applies.

## How the gate consumes this

`make repo-audit` runs a vanilla audit. `make repo-audit-strict` runs
with `--baseline-compare data/output/debt_baseline_2026_04_30.toml
--strict` which is the durable regression check. The strict gate is
not yet wired into the pre-push 4-gate chain pending a few sprints of
manual verification that the metric definitions are stable.

## Cross-references

- Binary source:
  `crates/gororoba_cli_data/src/bin/repo_audit.rs`
- Make targets: `make repo-audit`, `make repo-audit-strict`
- Baseline tag: `debt-baseline-v0` on commit `970b4da3`
- Baseline TOML: `data/output/debt_baseline_2026_04_30.toml`
- Stage B closure summary: `~/.claude/projects/.../memory/project_stage_b_complete.md`
- Architecture walkthrough: `docs/engineering/registry_canonical_architecture.md`
