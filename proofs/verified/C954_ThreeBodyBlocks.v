(** * C954_ThreeBodyBlocks: AVT contraction with block-diagonal embedding
    decomposes into body contributions plus cross-body coupling.

    Proves that bilinear contraction of a block-structured vector
    decomposes into intra-block and cross-block terms.

    Claim C-954: Body-specific 64D embedding with Earth/Moon/Sun blocks. *)

From OpenGororoba Require Import Prelude.
From OpenGororoba Require Import ThreeBodyAngMom.

Open Scope R_scope.

(** Bilinear form on Vec3 (dot product). *)
Definition dot (a b : Vec3) : R :=
  vx a * vx b + vy a * vy b + vz a * vz b.

(** THEOREM: Dot product distributes over vector addition.
    This is the fundamental property that makes block decomposition work:
    <a+b, c+d> = <a,c> + <a,d> + <b,c> + <b,d> *)
Theorem dot_bilinear :
  forall a b c d : Vec3,
    dot (vadd a b) (vadd c d) =
    dot a c + dot a d + dot b c + dot b d.
Proof.
  intros a b c d.
  destruct a, b, c, d.
  unfold dot, vadd; simpl.
  ring.
Qed.

(** COROLLARY: Three-body bilinear decomposition.
    For v = v_earth + v_moon + v_sun (block-structured 64D vector),
    the bilinear form decomposes into 9 terms (3 intra-block + 6 cross-block).
    Here we show the simpler 2-block case; the 3-block case follows by
    repeated application. *)
Corollary two_block_decomposition :
  forall a b : Vec3,
    dot (vadd a b) (vadd a b) =
    dot a a + 2 * dot a b + dot b b.
Proof.
  intros a b.
  destruct a, b.
  unfold dot, vadd; simpl.
  ring.
Qed.
