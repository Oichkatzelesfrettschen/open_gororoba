(* Verified Gallina -> C extraction of the FP24 exact-integer admission gate.

   CertiRocq (rocq-certirocq 0.9.1+9.1) compiles fp24_value_admit -- the r300
   R2VB carry-value gate |n| < 2^17 whose every admitted value is FP24-exact by
   fp24_value_admit_exact (FP24Representable.v) -- to Clight/C.  The emitted C
   uses the CertiCoq runtime; the trust boundary is the CertiRocq compiler
   pipeline (proved correct in CertiRocq.Compiler), not this file.

   R300_MP_FP24_EXACT_INT = 131072 = 2^17 in r300_numeric_domain.c.  ASCII only. *)
From CertiRocq.Plugin Require Import CertiRocq.
From OpenGororoba Require Import FP24Representable.

CertiRocq Compile fp24_value_admit.
