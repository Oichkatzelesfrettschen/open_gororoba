# Scala 3 Implementations of Lambda Calculus

This directory contains implementations of lambda calculus concepts in Scala 3 (formerly known as Dotty), a powerful multi-paradigm programming language that combines object-oriented and functional programming. Scala 3's advanced type system, including features like union types, intersection types, and dependent function types, makes it an interesting platform for exploring type theory and language design.

## Untyped Lambda Calculus

An example implementation of an untyped lambda calculus is provided, demonstrating core concepts such as:

*   **Abstract Syntax Tree (AST):** Defining the structure of lambda terms using Scala's `enum` (or `sealed trait` and `case class`) for `Var`, `Abs`, and `App`.
*   **Free Variables:** A method to compute the set of free variables, crucial for correct substitution.
*   **Substitution with Alpha-Conversion:** Implementing the substitution operation, ensuring proper handling of variable capture through alpha-conversion.
*   **Beta-Reduction:** Performing one-step and full normalization using a call-by-name evaluation strategy.

### Source Code

The implementation details can be found in `LambdaCalculus.scala` within this directory.

### Usage

To use this implementation, compile and run the `LambdaCalculus.scala` file. The file includes example lambda terms and their evaluation.

```scala
// Example: Identity function (λx.x)
val id = Abs("x", Var("x"))

// Example: Application of identity to 'y': (λx.x)y
val appId = App(id, Var("y"))

println(s"(λx.x)y normalizes to: ${appId.normalize}") // Expected: y
```

This implementation showcases how Scala 3's expressive syntax and functional programming features can be used to elegantly model and evaluate lambda calculus expressions.
