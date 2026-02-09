# Emanation Architecture

The de Marrais emanation construction builds 18 layers (L1-L18) from the
sedenion box-kite structure.  This was implemented and verified in Sprint 10
(I-016).

## Layers

The 18 layers encode progressively deeper structural relationships between
box-kite elements:

- **DMZ** (L1): Sign-concordance graph.  12 edges per box-kite, encoding
  which assessor pairs share sign patterns.

- **Sail-loops** (L2-L7): Fano-plane incidence patterns within each box-kite.
  The sail structure relates to the 7-point projective plane PG(2,2).

- **Oriented Trip Sync** (L8+): Universal across all 7 box-kites.  Every
  box-kite exhibits the same trip synchronization pattern, suggesting a
  deep structural invariant.

## Implementation

- Source: `crates/algebra_core/src/experimental/emanation.rs`
- Size: 4400 lines
- Tests: 113 dedicated tests
- Sprint: S10

## Claims

C-468: DMZ sign-concordance structure.
C-469: Sail-loop Fano incidence.
C-470: Oriented Trip Sync universality.
C-471 through C-475: Additional emanation layer properties.

## References

- de Marrais (2006), "Presto! Digitization: Part I" (arXiv: math/0603281)
- de Marrais (2007), "Catamaran Sails and Spindles" (arXiv: 0704.0026)
