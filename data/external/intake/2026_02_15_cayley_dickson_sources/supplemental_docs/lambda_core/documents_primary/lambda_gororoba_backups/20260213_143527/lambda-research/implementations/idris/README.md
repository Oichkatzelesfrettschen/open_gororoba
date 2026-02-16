# Idris Implementations of Lambda Calculus

This directory contains implementations of lambda calculus concepts in Idris, a dependently typed functional programming language. Idris is particularly well-suited for such implementations due to its powerful type system, which allows for expressing and verifying properties of programs at compile-time.

## Untyped Lambda Calculus

An example implementation of an untyped lambda calculus is provided, demonstrating core concepts such as:

*   **Term Representation:** Defining a data type for lambda calculus terms (variables, abstractions, applications).
*   **Free Variables:** Calculating the free variables within a term.
*   **Substitution:** Implementing the substitution operation, crucial for beta-reduction, with proper handling of alpha-conversion to prevent variable capture.
*   **Beta-Reduction:** Performing one-step and full normalization using a call-by-name evaluation strategy.

### Source Code

The implementation details can be found in `Lambda.idr` within this directory.

### Usage

To use this implementation, you would typically load the `Lambda.idr` file into the Idris REPL (e.g., `idris2 Lambda.idr`) and interact with the defined terms and functions.

```idris
-- Example terms
-- Identity function: (\x.x)
idTerm : Term
idTerm = Lam "x" (Var "x")

-- Application of identity to y: ((\x.x) y)
appId : Term
appId = App idTerm (Var "y")

-- Normalize the application
testId : Term
testId = normalize appId -- Should be Var "y"
```

This implementation serves as a clear example of how to model and execute lambda calculus within a dependently typed setting, leveraging Idris's features for correctness and clarity.
