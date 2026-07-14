(* Verified Gallina -> C extraction of the r300 typed-carry admission gate.

   CertiRocq (rocq-certirocq 0.9.1+9.1) compiles sint_range_admit -- the signed
   production predicate faithful to r300_nir_ssa_cut.c lines 433-439: a carry
   interval [lo, hi] is admitted iff -131072 <= lo and hi <= 131072 (inclusive).
   sint_range_admit_exact (FP24Representable.v) proves every value an admitted
   interval carries is FP24-exact.  The uint form (uint_range_admit) and the
   inclusive value gate (fp24_value_admit) sit alongside it; the strict B <= 127
   IDCT-operand policy is a separate function and is not this gate.

   The emitted C uses the CertiCoq runtime; the trust boundary is the CertiRocq
   compiler pipeline (proved correct in CertiRocq.Compiler), not this file.
   R300_MP_FP24_EXACT_INT = 131072 = 2^17 in r300_numeric_domain.c.  ASCII only. *)
From CertiRocq.Plugin Require Import CertiRocq.
From OpenGororoba Require Import FP24Representable.

CertiRocq Compile sint_range_admit.
