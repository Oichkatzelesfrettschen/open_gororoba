# Living Verification Artifact -- Pipeline Architecture

## Overview

The LVA pipeline connects Rocq formal proofs to the LaTeX paper, ensuring
the PDF cannot desynchronize from the prover.

```
Rocq .v sources
    |
    v
rocq compile -> .vo (kernel-checked proof objects)
    |
    +-> Extraction -> OCaml functor -> golden file diff
    |                                     |
    |                                     v (manual translation)
    |                                 verified_core/ (Rust crate)
    |                                     |
    |                                     v
    |                                 cross_validate.rs (FFI Oracle)
    |
    +-> rocq doc --html (browsable documentation)
    +-> collect_metrics.sh -> compile_times.csv + summary.csv (PGFPlots)
    +-> dep_graph.sh + dot2tex -> dep_graph.tikz (paper figure)
    +-> catchfilebetweentags -> source-anchored code in LaTeX
```

## Quick Start

```sh
# Full verification pipeline
cd proofs && just all

# Paper-ready artifacts only (metrics + TikZ dep graph)
cd proofs && just paper-artifacts

# Build proofs + paper in one command (from repo root)
make lva-paper

# Browsable HTML documentation
cd proofs && just rocqdoc
```

## Pipeline Targets

| Target | Command | Output |
|--------|---------|--------|
| Kernel check | `just check` | 38 .vo files, zero Admitted |
| Extraction | `just extract` | OCaml golden file diff |
| Metrics | `just metrics` | `metrics/compile_times.csv`, `metrics/summary.csv` |
| Dep graph | `just depgraph` | `metrics/dep_graph.{dot,pdf,svg}` |
| TikZ graph | `just depgraph-tikz` | `metrics/dep_graph.tikz` |
| HTML docs | `just rocqdoc` | `html/index.html` |
| Paper artifacts | `just paper-artifacts` | metrics + TikZ |
| Full pipeline | `just all` | check + extract + metrics + tikz |

## Trust Boundary

The trust boundary is the single point where abstract proofs meet concrete
IEEE 754 arithmetic:

- **Rocq side**: `FLOAT_OPS` module type in `theories/FloatAxioms.v` --
  abstract field axioms (add_comm, mul_assoc, etc.)
- **Rust side**: `crates/verified_core/src/axioms.rs` -- maps each axiom
  to an `f64` primitive operation
- **Validation**: `crates/verified_core/tests/cross_validate.rs` -- sweeps
  36 rotation test cases at tolerance < 1e-12

The axioms hold exactly for real numbers and approximately for f64. The
cross-validation confirms that rounding errors are negligible for the
quaternion rotation use case.

## Source Anchoring

LaTeX code listings are pulled directly from source files using
`\inputminted` (from the `minted` package) with `firstline`/`lastline`
options. Tag comments in the source files mark the boundaries:

- Rocq: `(*<*tagname>*) ... (*</tagname>*)`
- Rust: `//<*tagname> ... //</tagname>`

NOTE: `catchfilebetweentags` was evaluated but its `%<*tag>` delimiter
format is incompatible with Rocq/Rust comment syntax. `\inputminted`
provides the same source-anchoring guarantee plus syntax highlighting.

Tags in use:

| File | Tag | Content |
|------|-----|---------|
| `theories/FloatAxioms.v` | `floatops` | FLOAT_OPS module type |
| `theories/FloatQuaternion.v` | `quatrotate` | quat_rotate definition |
| `verified/C876_QuaternionRotation.v` | `c876rotation` | rotation = matrix theorem |
| `verified/C871_CasimirExact.v` | `c871cubic` | cubic scaling theorem |
| `theories/CayleyDicksonAlgebra.v` | `normconj` | norm-conjugate identity |
| `verified_core/src/axioms.rs` | `trustboundary` | f64 axiom realization |
| `verified_core/src/quaternion.rs` | `quatrotatefn` | quat_rotate function |
| `verified_core/tests/cross_validate.rs` | `crossval` | sweep test |

## Opam Environment

- Switch: `rocq-9.1.0-isolated`
- Rocq: 9.1.1 (OCaml 5.4.0)
- No external Rocq dependencies (stdlib only)
- Backup: `opam switch export rocq-9.1.0-isolated-backup.export`

## Tool Versions

| Tool | Version | Purpose |
|------|---------|---------|
| rocq | 9.1.1 | Proof compiler |
| latexmk | system | LaTeX build |
| dot2tex | 2.11.3 | DOT -> TikZ |
| pygmentize | 2.19.2 | Syntax highlighting (minted) |
| dot | system | Graph rendering |

## Known Limitations

- **Alectryon**: BLOCKED. SerAPI not ported to Rocq 9.x. Using `rocq doc`
  as fallback for HTML documentation.
- **coq-dpdgraph**: SKIPPED. Forces rocq-stdlib 9.1.0 -> 9.0.0 downgrade,
  breaking `From Stdlib Require Import` namespace. Using file-level
  dep_graph.sh + dot2tex instead.
- **arXiv**: Does not support `--shell-escape` (required by minted). The
  arXiv version should use `listings` package instead.
- **verified_core scope**: Currently covers only quaternion rotation (1 of
  ~40 gr_core modules). Expansion to ADM, Casimir, and energy conditions
  is future work.
- **rocq-rust-extraction 0.2.0**: Evaluated and DEFERRED. Extracts Records
  as Rust enums (not structs), uses lifetime-heavy `&'a` style with
  `PhantomData`, and does not map Module Types to Rust traits. Our manual
  translation in verified_core is simpler and more idiomatic for numerical
  f64 code. Re-evaluate when the tool supports struct extraction and trait
  mapping from abstract module types.
- **No CI**: Proofs are not yet checked in GitHub Actions. Pipeline runs
  locally via `just all`.
