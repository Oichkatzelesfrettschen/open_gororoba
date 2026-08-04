(* Verified Gallina -> C extraction of the R2VB DP4 transform kernel.

   CertiRocq (rocq-certirocq 0.9.1+9.1) compiles dp4 -- the 4-term integer dot
   product that is one lane of the producer's four-DP4 matrix apply -- and
   dp4_operand_admit, the operand-magnitude gate 0 <= B /\ 4 B^2 <= 131072.
   dp4_operand_admit_exact and mvp4_rows_exact (R2VBTransformDP4.v) prove that
   every lane of an admitted transform is FP24-exact in FLX(17); the tight
   admission boundary is B = 181 (dp4_admit_boundary).

   The emitted C uses the CertiCoq runtime; the trust boundary is the CertiRocq
   compiler pipeline (proved correct in CertiRocq.Compiler), not this file.
   ASCII only. *)
From CertiRocq.Plugin Require Import CertiRocq.
From OpenGororoba Require Import R2VBTransformDP4.

CertiRocq Compile dp4.
CertiRocq Compile dp4_operand_admit.

CertiRocq Generate Glue -file "r2vb_dp4_glue" [ Corelib.Numbers.BinNums.Z, Corelib.Numbers.BinNums.positive, bool ].
