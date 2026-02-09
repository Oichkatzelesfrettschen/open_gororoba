<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/book_docs.toml -->

# Paper Collection

The project maintains a curated collection of 19 research papers relevant to
Cayley-Dickson algebra and its applications.  Papers are stored as PDFs under
`papers/pdf/` with structured TOML extractions under `papers/extracted/`.

## De Marrais papers (8)

| Year | Title | arXiv |
|------|-------|-------|
| 2000 | The 42 Assessors and the Box-Kites they Fly | math/0011260 |
| 2002 | Placeholder Substructures I | math/0207003 |
| 2004 | Placeholder Substructures III | math/0403113 |
| 2004 | Wolfram Science Conference slides | -- |
| 2006 | Presto! Digitization: Part I | math/0603281 |
| 2007 | Placeholder Substructures III (revised) | math/0703745 |
| 2007 | Catamaran Sails and Spindles | 0704.0026 |
| 2007 | SedenionsXOR: Things Barely Thought of Fly High | 0704.0112 |

## Moreno papers (2)

| Year | Title | arXiv |
|------|-------|-------|
| 2005 | Zero Divisors of 2^n-ions | math/0512517 |
| 2005 | The Zero Divisors of the CD Algebras over R | math/0512516 |

## Reggiani papers (3)

| Year | Title | arXiv |
|------|-------|-------|
| 2024 | Cayley-Dickson Algebras and Projective Planes | 2411.18881 |
| 2025 | Automorphisms of the Sedenions and Beyond | 2512.07210 |
| 2025 | Cayley-Dickson Algebras: Structure and Applications | 2512.13002 |

## Additional papers (6)

| Year | Authors | Title | arXiv |
|------|---------|-------|-------|
| 2002 | Baez | The Octonions | math/0105155 |
| 2009 | Cawagas et al. | Subalgebra Structure of the Trigintaduonions | 0907.2047 |
| 2023 | Tang & Tang | Sedenion Extension of the Standard Model | 2308.14768 |
| 2024 | Brown & Pumpluen | Flipped Polynomial Rings and CD Construction | 2403.03763 |
| 2025 | -- | The Structure of Cayley-Dickson Algebras | 2505.11747 |
| 2025 | -- | A Mnemonic for the Cayley-Dickson Tower | 2512.22134 |

## Extraction pipeline

Papers are extracted using the `docpipe` crate (pdfium-render backend).
The extraction produces structured TOML with title, authors, abstract,
sections, and definitions.

**Limitation:** Equation extraction from PDF OCR text is structurally unable
to find LaTeX-formatted equations because the OCR output contains plain text
without `$` or `\begin{equation}` delimiters.  All 19 paper TOML files have
`equations = []`.  Genuine equation extraction would require downloading
arXiv LaTeX source files (future work).

```sh
# Extract a specific paper
cargo run --release --bin extract-papers -- --only demarrais-2000-math0011260
```
