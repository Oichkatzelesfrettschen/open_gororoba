# Scheme/Racket Implementations of Lambda Calculus

This directory contains implementations of lambda calculus concepts in Scheme/Racket, a dialect of Lisp known for its strong support for functional programming and metaprogramming. Racket's expressive power makes it an excellent choice for implementing and experimenting with language semantics.

## Untyped Lambda Calculus

An example implementation of an untyped lambda calculus is provided, demonstrating core concepts such as:

*   **Term Representation:** Defining data structures (using `struct`) for lambda calculus terms (variables, abstractions, applications).
*   **Substitution with Alpha-Conversion:** Implementing the crucial substitution operation, which correctly handles variable capture through alpha-conversion by generating fresh variable names.
*   **Beta-Reduction:** Performing one-step and full normalization using a call-by-value evaluation strategy.

### Source Code

The implementation details can be found in `lambda-racket.rkt` within this directory.

### Usage

To use this implementation, save the `lambda-racket.rkt` file and run it with Racket. The file includes example terms and their evaluation.

```racket
#lang racket

;; Identity function: (λx.x)
(define id (lam 'x (var 'x)))

;; Application of identity to 'y': (λx.x)y  -> y
(define ex1 (app id (var 'y)))
(printf "Example 1: (λx.x)y -> ~a\n" (full-eval-lc ex1)) ; Expected: (var 'y)

;; Example with potential capture (alpha-conversion needed)
;; (λx. (λy. (x y))) y  -- if we substitute 'y' for 'x' in (λy. (x y)),
;;                        the inner 'y' would capture the outer 'y'.
;;                        So, (λy. (x y)) should become (λv1. (x v1)) first.
;; Then substitute 'y' for 'x' -> (λv1. (y v1))
(define inner-lam (lam 'y (app (var 'x) (var 'y))))
(define outer-lam (lam 'x inner-lam))
(define ex3 (app outer-lam (var 'y)))
(printf "Example 3: (λx. (λy. (x y))) y -> ~a\n" (full-eval-lc ex3)) ; Expected: (lam 'v1 (app (var 'y) (var 'v1)))
```

This implementation serves as a clear example of how to model and execute lambda calculus within a Lisp-like environment, highlighting the elegance of symbolic manipulation in Racket.
