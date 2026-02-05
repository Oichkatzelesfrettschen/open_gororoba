From Stdlib Require Import String.

Open Scope string_scope.

Inductive Action : Type :=
| Send
| Receive
| Map
| Grant.

Parameter has_right : string -> string -> Action -> Prop.
Parameter reachable_delegation : string -> string -> Action -> Prop.
