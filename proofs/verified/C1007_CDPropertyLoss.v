(** * C-1007: Every CD doubling for n >= 2 loses at least one algebraic property.

    Claim C-1007: The Cayley-Dickson construction at dimensions 4, 8, 16
    produces algebras that each lose a property held by the previous level:
    - dim 4: commutativity lost (quat)
    - dim 8: associativity lost (octonion)
    - dim 16: division algebra lost (sedenion zero divisors)

    This is the formal backbone of the CD Ladder of Chaos experiment:
    the monotonic loss of algebraic structure with dimension directly
    maps to increasing dynamical drag via the Alternativity Violation Tensor.

    Mirrors: gr_core/src/cd_ladder_force.rs *)

From OpenGororoba Require Import CDPropertyTower.

(** C-1007: Three losses, three doublings. *)
Theorem C1007_cd_property_loss :
  commutativity_lost_at_4 /\
  associativity_lost_at_8 /\
  division_lost_at_16.
Proof. exact cd_property_tower. Qed.
