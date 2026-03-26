(** * CDSignSection: Fixed-dimension sign-table sections.

    Packages the recursive sign function into section-style views for the
    octonion, sedenion, and pathion levels.  This keeps the common fuel/dim
    parameters fixed and exposes cross-section agreement on the octonion block.

    Sources of truth:
    - M3IsAssociator.v for the canonical cd_sign_fuel recursion
    - FanoPlane.v for the 7 oriented octonion lines
    - Moreno/Baez corpus for the XOR + orientation convention *)

From Stdlib Require Import ZArith Bool List Arith Lia.
Import ListNotations.

From OpenGororoba Require Import M3IsAssociator CDSignHalfStep FanoPlane.

Open Scope Z_scope.

Definition all_pairs_below (n : nat) : list (nat * nat) :=
  List.flat_map (fun i => List.map (fun j => (i, j)) (List.seq 0 n)) (List.seq 0 n).

Definition all_imag_pairs_below (n : nat) : list (nat * nat) :=
  List.flat_map
    (fun i => List.map (fun j => (i, j)) (List.seq 1 (Nat.pred n)))
    (List.seq 1 (Nat.pred n)).

Definition check_anticomm_below (fuel dim n : nat) : bool :=
  List.forallb
    (fun ij =>
      let i := fst ij in
      let j := snd ij in
      if Nat.eqb i j
      then true
      else Z.eqb (cd_sign_fuel fuel dim i j) (Z.opp (cd_sign_fuel fuel dim j i)))
    (all_imag_pairs_below n).

Definition check_fano_line_signs (fuel dim : nat) : bool :=
  List.forallb
    (fun l =>
      match l with
      | [a; b; c] =>
          let sab := cd_sign_fuel fuel dim a b in
          let sbc := cd_sign_fuel fuel dim b c in
          let sca := cd_sign_fuel fuel dim c a in
          andb (Nat.eqb (Nat.lxor a b) c)
            (orb
              (andb (Z.eqb sab 1)
                    (andb (Z.eqb sbc 1) (Z.eqb sca 1)))
              (andb (Z.eqb sab (-1))
                    (andb (Z.eqb sbc (-1)) (Z.eqb sca (-1)))))
      | _ => false
      end)
    fano_lines.

Section Oct8Signs.
  Let fuel : nat := 4.
  Let dim : nat := 8.

  Definition oct8_sign (i j : nat) : Z := cd_sign_fuel fuel dim i j.

  Lemma oct8_sign_zero_left : forall q : nat,
    (q < dim)%nat ->
    oct8_sign 0 q = 1%Z.
  Proof.
    intros q Hq. unfold oct8_sign, fuel, dim.
    apply cd_sign_fuel_0_left; exact Hq.
  Qed.

  Lemma oct8_sign_zero_right : forall p : nat,
    (p < dim)%nat ->
    oct8_sign p 0 = 1%Z.
  Proof.
    intros p Hp. unfold oct8_sign, fuel, dim.
    apply cd_sign_fuel_0_right; exact Hp.
  Qed.

  Lemma oct8_sign_self_neg : forall p : nat,
    (0 < p)%nat ->
    (p < dim)%nat ->
    oct8_sign p p = (-1)%Z.
  Proof.
    intros p Hp Hq. unfold oct8_sign, fuel, dim.
    apply cd_sign_fuel_self_neg; assumption.
  Qed.

  Lemma oct8_sign_anticomm : forall p q : nat,
    (0 < p)%nat ->
    (p < dim)%nat ->
    (0 < q)%nat ->
    (q < dim)%nat ->
    p <> q ->
    oct8_sign p q = Z.opp (oct8_sign q p).
  Proof.
    intros p q Hp0 Hp Hq0 Hq Hneq. unfold oct8_sign, fuel, dim.
    apply cd_sign_fuel_anticomm; assumption.
  Qed.

  Theorem oct8_fano_line_sign_check :
    check_fano_line_signs fuel dim = true.
  Proof. vm_compute. reflexivity. Qed.
End Oct8Signs.

Section Sed16Signs.
  Let fuel : nat := 5.
  Let dim : nat := 16.

  Definition sed16_sign (i j : nat) : Z := cd_sign_fuel fuel dim i j.

  Theorem sed16_anticomm_check :
    check_anticomm_below fuel dim dim = true.
  Proof. vm_compute. reflexivity. Qed.
End Sed16Signs.

Section Path32Signs.
  Let fuel : nat := 6.
  Let dim : nat := 32.

  Definition path32_sign (i j : nat) : Z := cd_sign_fuel fuel dim i j.

  Theorem path32_anticomm_check :
    check_anticomm_below fuel dim dim = true.
  Proof. vm_compute. reflexivity. Qed.
End Path32Signs.

Theorem sign_agrees_across_sections : forall i j : nat,
  (i < 8)%nat ->
  (j < 8)%nat ->
  cd_sign_fuel 4 8 i j = cd_sign_fuel 5 16 i j /\
  cd_sign_fuel 5 16 i j = cd_sign_fuel 6 32 i j.
Proof.
  intros i j Hi Hj.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  destruct j as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  vm_compute; split; reflexivity.
Qed.
