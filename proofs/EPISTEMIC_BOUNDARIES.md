# Epistemic Boundaries

Known limitations of the Rocq formal proof suite (142 .v files, 107 verified claims).

## 1. Universal Octonion Alternativity

**Status**: Basis-verified (not universally proven).
**Claim refs**: C-910 (left alt, basis), C910_Right_e0..e7 (right alt, basis).
**Barrier**: Expanding `oct_mul (oct_mul a b) b` for arbitrary `a, b : CDOct`
produces degree-6 polynomial identities in 16 real variables. The `ring` and
`nra` tactics cannot close this. `nsatz` (Groebner bases) may work but requires
manual intermediate term guidance that has not been attempted.
**Coverage**:
- Left alternativity: verified for basis e1, e4 (C910_OctonionAlternative.v).
  Each theorem universally quantifies over one argument (`forall b`), fixing the
  other to a basis element.
- Right alternativity: verified for all 8 basis elements e0..e7
  (C910_Right_e0.v .. C910_Right_e7.v, one shard per basis element).
  Each theorem universally quantifies over one argument (`forall a`),
  fixing the other to a basis element.
**Reference**: Baez, "The Octonions" (2002), Section 2.1.
**Difficulty**: High. Likely requires custom Ltac automation or a reflection-based
decision procedure for multilinear CD identities.

## 2. Sedenion Non-Alternativity

**Status**: Proven via zero-divisor existence (C-908, C908_SedenionZeroDivisor.v).
**Rationale**: Zero divisors imply the algebra is not a division algebra, hence
not alternative (Artin's theorem: an alternative ring with no zero divisors is
a division ring). The contrapositive gives non-alternativity.

## 3. Octonion Non-Associativity

**Status**: Proven as counterexample (C-909, C909_OctonionNonAssociative.v).
**Limitation**: The proof shows a specific triple (e1, e2, e4) for which
associativity fails. It does not prove a universal quantifier statement like
"for all a b c, oct_mul is not associative" (which is false -- some triples
do associate, e.g. when one element is the identity).

## 4. Extraction Pipeline (Rocq -> Rust)

**Status**: Blocked.
**Barrier**: rocq-rust-extraction 0.2.0 maps Records to enums (not structs),
introduces lifetime annotations, and does not map Module Types to traits.
Numerical code extracted this way is not usable without heavy manual editing.
**Workaround**: OCaml extraction works via `make -C proofs extract-check`.

## 5. Alectryon Rendering

**Status**: Blocked.
**Barrier**: Alectryon 1.4 requires coq-serapi (sertop), which supports
Coq <= 8.20. Not ported to Rocq 9.x. No known timeline for a port.

## 6. Conjugation Involution Coverage

**Status**: Complete at all CD levels.
**Proof files**:
- dim 2 (complex): C_ConjugateInvolution.v (claim_complex_conj_involution)
- dim 4 (quaternion): C_ConjugateInvolution.v (claim_quat_conj_involution)
- dim 8 (octonion): C_OctConjInvolution.v (oct_conj_involution)
- dim 16 (sedenion): C_SedConjInvolution.v (sed_conj_involution)
All proofs are universally quantified and kernel-checked. The CD induction
pattern (conj on lo-half + negation on hi-half) generalizes mechanically
to dim 32, 64, etc., but we stop at 16 because the project scope does not
require pathions or higher.
