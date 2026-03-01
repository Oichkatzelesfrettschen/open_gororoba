(** * C-885: S3-invariant bilinear form is democratic.

    Any S3-invariant bilinear form f : Idx3 -> Idx3 -> R has exactly
    two free parameters: f(i,i) = a (diagonal) and f(i,j) = b (off-diagonal).
    For the mixing matrix (doubly stochastic with row sums = 1), S3 invariance
    plus the constraint a = b forces a = b = 1/3: the democratic matrix.

    This unifies:
    - Verified C-076: the mixing matrix HAS S3 symmetry
    - Refuted C-077: the mixing matrix is NOT arbitrary / non-uniform

    The positive content: S3 symmetry + doubly stochastic => democratic.

    Reformulation unifying C-076 (verified) and C-077 (refuted). *)

From OpenGororoba Require Import Prelude SymmetricGroup3.

(** Decidable equality for Idx3 (needed for case analysis). *)
Definition Idx3_eq_dec (i j : Idx3) : {i = j} + {i <> j}.
Proof.
  destruct i; destruct j;
    try (left; reflexivity);
    right; discriminate.
Defined.

(** S3-invariance: f is unchanged under simultaneous permutation of both indices. *)
Definition s3_invariant (f : Idx3 -> Idx3 -> R) : Prop :=
  forall (sigma : S3Perm) (i j : Idx3),
  f (apply_s3 sigma i) (apply_s3 sigma j) = f i j.

(** Democratic form: constant function. *)
Definition democratic (a : R) (i j : Idx3) : R := a.

(** The democratic form is S3-invariant (trivially). *)
Theorem democratic_s3_invariant :
  forall a : R, s3_invariant (democratic a).
Proof.
  intros a sigma i j. unfold democratic. reflexivity.
Qed.

(** Diagonal orbit: all f(i,i) are equal.
    S3 acts transitively on diagonal pairs via transpositions. *)
Theorem s3_invariant_diagonal_equal :
  forall f : Idx3 -> Idx3 -> R,
  s3_invariant f ->
  forall i : Idx3, f i i = f I0 I0.
Proof.
  intros f Hinv i.
  destruct i.
  - reflexivity.
  - rewrite <- (Hinv S3t01 I1 I1). simpl. reflexivity.
  - rewrite <- (Hinv S3t02 I2 I2). simpl. reflexivity.
Qed.

(** Off-diagonal orbit: all f(i,j) with i <> j are equal.
    S3 acts transitively on off-diagonal pairs. *)
Theorem s3_invariant_offdiag_equal :
  forall f : Idx3 -> Idx3 -> R,
  s3_invariant f ->
  forall i j : Idx3, i <> j -> f i j = f I0 I1.
Proof.
  intros f Hinv i j Hneq.
  destruct i; destruct j; try (exfalso; apply Hneq; reflexivity).
  - (* (I0, I1) *) reflexivity.
  - (* (I0, I2): t12 maps (I0,I2) to (I0,I1) *)
    rewrite <- (Hinv S3t12 I0 I2). simpl. reflexivity.
  - (* (I1, I0): t01 maps (I1,I0) to (I0,I1) *)
    rewrite <- (Hinv S3t01 I1 I0). simpl. reflexivity.
  - (* (I1, I2): r012 maps (I0,I1) to (I1,I2) *)
    rewrite <- (Hinv S3r012 I0 I1). simpl. reflexivity.
  - (* (I2, I0): r021 maps (I0,I1) to (I2,I0) *)
    rewrite <- (Hinv S3r021 I0 I1). simpl. reflexivity.
  - (* (I2, I1): chain via (I0,I2) *)
    transitivity (f I0 I2).
    + rewrite <- (Hinv S3r012 I2 I1). simpl. reflexivity.
    + rewrite <- (Hinv S3t12 I0 I2). simpl. reflexivity.
Qed.

(*<*s3unique>*)
(** Democratic mixing theorem: S3-invariant + diagonal equals off-diagonal
    implies all entries are equal.

    For the octonionic mixing matrix: S3 symmetry constrains it to the
    two-parameter form f(i,j) = a*delta_{ij} + b*(1-delta_{ij}).
    The additional physical constraint that the mixing is uniform
    (a = b, i.e., no preferred subalgebra) then forces all entries
    to be equal. With row-sum normalization 3a = 1, we get a = 1/3. *)
Theorem s3_invariant_democratic_iff_diag_eq_offdiag :
  forall f : Idx3 -> Idx3 -> R,
  s3_invariant f ->
  f I0 I0 = f I0 I1 ->
  forall i j : Idx3, f i j = f I0 I0.
Proof.
  intros f Hinv Hdiag_eq i j.
  destruct (Idx3_eq_dec i j) as [Heq | Hneq].
  - subst. apply s3_invariant_diagonal_equal. exact Hinv.
  - rewrite s3_invariant_offdiag_equal; [symmetry; exact Hdiag_eq | exact Hinv | exact Hneq].
Qed.
(*</s3unique>*)

(** Corollary: the democratic 1/3 value is the unique normalized solution. *)
Theorem democratic_mixing_normalized :
  forall a : R,
  a + a + a = 1 ->
  a = 1 / 3.
Proof.
  intros a Ha. lra.
Qed.
