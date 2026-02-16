# Standard ML (SML) Implementations of Lambda Calculus

This directory contains implementations of lambda calculus concepts in Standard ML (SML), a functional programming language with a strong static type system. SML is widely used in programming language research and education due to its clear semantics and powerful type inference.

## Untyped Lambda Calculus

An example implementation of an untyped lambda calculus is provided, demonstrating core concepts such as:

*   **Term Representation:** Defining a `datatype` for lambda calculus terms (variables, abstractions, applications).
*   **Substitution with Alpha-Conversion:** Implementing the substitution operation, which correctly handles variable capture through alpha-conversion by generating fresh variable names.
*   **Beta-Reduction:** Performing evaluation using a call-by-value strategy.

### Source Code

The implementation details can be found in `lambda.sml` within this directory.

### Usage

To use this implementation, save the `lambda.sml` file and load it into an SML interpreter (e.g., SML/NJ or Moscow ML). The file includes example terms and their evaluation.

```sml
(* Identity function: (λx.x) *)
val id = Abs ("x", Var "x");

(* Application of identity to y: (λx.x) y *)
val app_id_y = App (id, Var "y");

val _ = print ("Identity function applied to y: " ^ term_to_string app_id_y ^ "\n");
val _ = print ("Result: " ^ term_to_string (eval app_id_y) ^ "\n\n");
```

This implementation serves as a clear example of how to model and execute lambda calculus within a statically typed functional language, showcasing SML's features for defining algebraic data types and pattern matching.

