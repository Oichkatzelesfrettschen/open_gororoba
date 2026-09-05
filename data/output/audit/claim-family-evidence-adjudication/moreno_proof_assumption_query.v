From OpenGororoba Require Import MorenoSedenionInstance.
Require Import C1538_MorZDSymmetry C1539_MorSkewSymm.

Module AuditedSkew := C1539_MorSkewSymm.MorSkewSymm SedMoreno.
Print Assumptions AuditedSkew.l_x_skew_symm.
Print Assumptions AuditedSkew.r_x_skew_symm.
Print Assumptions AuditedSkew.ker_lx_eq_ker_rx.

Module AuditedSymmetry (Alg : C1538_MorZDSymmetry.CDAlgInnerTrace).
  Module Symmetry := C1538_MorZDSymmetry.MorZDSymmetry Alg.
  Check Symmetry.zd_symmetry.
  Check Symmetry.zd_conj_vanishes.
  Print Assumptions Symmetry.zd_symmetry.
End AuditedSymmetry.
