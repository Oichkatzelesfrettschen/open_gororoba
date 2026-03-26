(** * FanoPlane: The Fano plane PG(2,2) -- 7 points, 7 lines.

    The Fano plane governs octonion multiplication sign rules.
    Its 7 lines correspond to the 7 imaginary octonion triples
    (i,j,k) where e_i * e_j = +/- e_k.

    STRUCTURE (Baez 2002, Section 2.1, p.7):
    The Fano plane is the PROJECTIVE PLANE over the 2-element field Z_2.
    Concretely: points are the 7 nonzero elements of Z_2^3 (the 3-dim vector
    space over GF(2)); lines are the 2-dim subspaces {a, b, a+b} for distinct
    nonzero a, b in Z_2^3 (where + is component-wise mod 2, i.e., XOR).

    Since each "subspace" triple {a, b, c} satisfies a+b+c = 0 in Z_2^3,
    this is exactly the XOR rule: a XOR b = c for each Fano line {a,b,c}.

    INDEXING: We index the 7 points as 1..7 (matching octonion basis e_1..e_7).
    The Fano lines below match Baez (2002) Table 1 and the Fano plane diagram
    on p.7 (orientation: cyclic order in each triple gives e_i*e_j = +e_k).

    Properties proved here (Baez 2002, Section 2.1):
    - 7 points (octonion basis indices 1..7): fano_line_count
    - Every line has exactly 3 points (Baez p.7): fano_line_sizes
    - Every point lies on exactly 3 lines (Baez p.7 diagram): fano_point_incidence
    - Any two distinct points lie on exactly 1 common line (Baez p.7:
      "Each pair of distinct points lies on a unique line"): fano_unique_line
    - Any two distinct lines meet in exactly 1 point (dual projective axiom,
      PG(2,2) self-duality): fano_unique_point
    - All Fano lines satisfy a XOR b = c (Baez p.7, Z_2^3 addition rule):
      fano_xor_rule
    - The 7 lines are pairwise distinct (NoDup): fano_lines_distinct

    Sources:
    - J.C. Baez (2002), "The Octonions", Bull. AMS 39, pp. 145-205.
      arXiv:math/0105155. Section 2.1 "The Fano plane", p.6-7.
    - Standard projective geometry: PG(2,2) has 7 points and 7 lines,
      each line has 3 points, each point is on 3 lines (the parameters
      of a (7,3,1)-design, equivalently, a Steiner triple system S(2,3,7)).

    Downstream use: G2OctonionAutomorphisms.v, OctonionStandardModel.v,
    BrownAssessorEquivalence.v all import this file. *)

From Stdlib Require Import List Arith Bool.
Import ListNotations.

(** The 7 Fano lines as triples of octonion basis indices (1-indexed).

    These match Baez (2002) Section 2.1, p.7 and Table 1 on p.6.
    Index correspondence: point k corresponds to octonion basis element e_k.

    Line {a,b,c} means: in the cyclic orientation shown in Baez's diagram,
      e_a * e_b = e_c,  e_b * e_a = -e_c  (and cyclic permutations).
    Equivalently (by the Z_2^3 structure): a XOR b = c.

    Cross-check: line [1;2;3] matches e_1*e_2=e_3 from Baez Table 1 (p.6).
    Cross-check: line [1;4;5] matches e_1*e_4=e_5 from Baez Table 1 (p.6). *)
Definition fano_lines : list (list nat) :=
  [[1; 2; 3];   (** e_1*e_2=e_3: 001 XOR 010 = 011 in Z_2^3 *)
   [1; 4; 5];   (** e_1*e_4=e_5: 001 XOR 100 = 101 in Z_2^3 *)
   [1; 6; 7];   (** e_1*e_6=e_7: 001 XOR 110 = 111 in Z_2^3 *)
   [2; 4; 6];   (** e_2*e_4=e_6: 010 XOR 100 = 110 in Z_2^3 *)
   [2; 5; 7];   (** e_2*e_5=e_7: 010 XOR 101 = 111 ... wait: 010 XOR 101 = 111 = 7; yes *)
   [3; 4; 7];   (** e_3*e_4=e_7: 011 XOR 100 = 111 in Z_2^3 *)
   [3; 5; 6]].  (** e_3*e_5=e_6: 011 XOR 101 = 110 in Z_2^3 *)

(** There are exactly 7 lines (Baez Section 2.1, p.7: "7 points and 7 lines"). *)
Theorem fano_line_count : length fano_lines = 7.
Proof. reflexivity. Qed.

(** Each line has exactly 3 points
    (Baez Section 2.1, p.7: "Each line contains three points"). *)
Theorem fano_line_sizes :
  List.map (@length _) fano_lines = [3; 3; 3; 3; 3; 3; 3].
Proof. reflexivity. Qed.

(** Check if a nat is a member of a list. *)
Fixpoint nat_mem (n : nat) (l : list nat) : bool :=
  match l with
  | [] => false
  | x :: rest => if Nat.eqb n x then true else nat_mem n rest
  end.

(** Count how many lines contain a given point. *)
Definition lines_through (p : nat) : nat :=
  length (List.filter (fun l => nat_mem p l) fano_lines).

(** Every point lies on exactly 3 lines.
    Derivable from Baez Section 2.1 p.7 diagram: each of the 7 points
    has 3 lines through it (the Fano plane has "3 lines through each point"
    by the (7,3,1)-design / Steiner triple system S(2,3,7) structure).
    Verified by exhaustive computation. *)
Theorem fano_point_incidence :
  List.map lines_through [1; 2; 3; 4; 5; 6; 7] = [3; 3; 3; 3; 3; 3; 3].
Proof. vm_compute. reflexivity. Qed.

(** ================================================================== *)
(** * Projective plane axioms.                                         *)
(** ================================================================== *)

(** Count how many Fano lines contain BOTH points p and q. *)
Definition lines_through_both (p q : nat) : nat :=
  length (List.filter (fun l => nat_mem p l && nat_mem q l) fano_lines).

(** Any two distinct points from {1..7} lie on exactly 1 common line.

    This is the first PROJECTIVE PLANE AXIOM (Baez Section 2.1, p.7):
    "Each pair of distinct points lies on a unique line."

    Equivalently: any two distinct points determine a unique Fano line.
    This holds because the Fano plane is PG(2,2), the smallest projective
    plane, which satisfies all projective plane axioms by construction.

    Proof: exhaustive check over all C(7,2) = 21 unordered pairs {p,q}
    with p < q. Each pair appears in exactly one of the 7 Fano lines.
    (Baez 2002, Section 2.1, p.7; arXiv:math/0105155.) *)
Theorem fano_unique_line :
  List.forallb (fun p =>
    List.forallb (fun q =>
      if Nat.ltb p q
      then Nat.eqb (lines_through_both p q) 1
      else true)
    [1; 2; 3; 4; 5; 6; 7])
  [1; 2; 3; 4; 5; 6; 7] = true.
Proof. vm_compute. reflexivity. Qed.

(** Any two distinct lines from {1..7} share exactly 1 common point.

    This is the DUAL PROJECTIVE PLANE AXIOM. Since PG(2,2) is self-dual
    (its point-line incidence structure is isomorphic to its dual), the
    same axiom "any two distinct lines meet in exactly one point" holds.
    Baez (2002) Section 2.1, p.7 establishes the projective plane structure;
    self-duality follows from PG(2,2) being a symmetric design. *)
Definition points_on_both (l1 l2 : list nat) : nat :=
  length (List.filter (fun p => nat_mem p l1 && nat_mem p l2) [1;2;3;4;5;6;7]).

Theorem fano_unique_point :
  List.forallb (fun i =>
    List.forallb (fun j =>
      if Nat.ltb i j
      then
        let l1 := List.nth i fano_lines [] in
        let l2 := List.nth j fano_lines [] in
        Nat.eqb (points_on_both l1 l2) 1
      else true)
    [0; 1; 2; 3; 4; 5; 6])
  [0; 1; 2; 3; 4; 5; 6] = true.
Proof. vm_compute. reflexivity. Qed.

(** ================================================================== *)
(** * XOR rule for Fano lines.                                        *)
(** ================================================================== *)

(** Every Fano line {a, b, c} (in increasing order) satisfies a XOR b = c.

    MATHEMATICAL BASIS (Baez 2002, Section 2.1, p.7):
    The Fano plane is PG(2,2) = the projective plane over GF(2) = Z_2.
    Points are nonzero elements of Z_2^3 (7 elements: 001,010,011,100,101,110,111).
    Lines are 2-dim subspaces: {a, b, a+b} in Z_2^3 where + = XOR.
    Since a+b+c = 0 in Z_2^3 iff a XOR b XOR c = 0 iff a XOR b = c,
    the XOR rule is exactly the Z_2^3 subspace condition.

    INDEXING: point k corresponds to the binary representation of k
    (1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111) = standard GF(2)^3.

    OCTONION CONNECTION: Baez p.7 states e_i*e_j = e_k (with sign from orientation)
    when {i,j,k} is a Fano line. The XOR rule determines WHICH triple is a line;
    the cyclic orientation in Baez's diagram determines the SIGN of the product.

    Verified by vm_compute over all 7 lines. *)
Theorem fano_xor_rule :
  List.forallb
    (fun l =>
      match l with
      | [a; b; c] => Nat.eqb (Nat.lxor a b) c
      | _ => false
      end)
    fano_lines = true.
Proof. vm_compute. reflexivity. Qed.

(** ================================================================== *)
(** * Distinctness of lines.                                          *)
(** ================================================================== *)

(** The 7 Fano lines are pairwise distinct.

    Each of the 7 lines is a different subset of {1..7}.
    This follows from the Fano plane being a well-defined combinatorial design:
    PG(2,2) has exactly 7 distinct lines (each corresponding to a distinct
    2-dim subspace of Z_2^3). Baez (2002) Section 2.1, p.7; standard result.

    Proof: structural induction on the list, discharging inequality goals
    by discrimination (the lists have syntactically different elements). *)
Theorem fano_lines_distinct :
  List.NoDup fano_lines.
Proof.
  repeat constructor; simpl; intuition (discriminate).
Qed.
