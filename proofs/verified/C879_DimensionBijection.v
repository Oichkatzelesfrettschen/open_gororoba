(** * C-879: Cayley-Dickson dimension-nacelle bijection.

    Formal proof that the CD dimension -> nacelle count map is the
    identity on {1, 2, 4, 8, 16}, and that this map is injective.

    Mirrors: cd_dimension_nacelle_map() in adm_algebra_bridge.rs:189-198.
    Rust test: test_cd_dimension_map (line 287-295). *)

From OpenGororoba Require Import FiniteBijection.

(** CLAIM C-879: The map is the identity on all CD dimensions. *)
Theorem C879_nacelle_identity :
  forall d : CDDim, nacelle_count d = cd_dim_nat d.
Proof.
  exact nacelle_is_identity.
Qed.

(** Corollary: injectivity (no two algebras share a nacelle count). *)
Theorem C879_nacelle_injective :
  forall d1 d2 : CDDim, nacelle_count d1 = nacelle_count d2 -> d1 = d2.
Proof.
  exact nacelle_injective.
Qed.

(** Corollary: all nacelle counts are positive. *)
Theorem C879_nacelle_positive :
  forall d : CDDim, 0 < nacelle_count d.
Proof.
  intro d. rewrite nacelle_is_identity. apply cd_dim_positive.
Qed.
