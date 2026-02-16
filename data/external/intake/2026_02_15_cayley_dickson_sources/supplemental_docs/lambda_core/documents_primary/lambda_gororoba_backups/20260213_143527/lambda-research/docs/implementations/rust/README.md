# Rust Implementation Guide: tapl-rust

## Overview

**tapl-rust** is a comprehensive, production-ready implementation of lambda calculus following the structure and principles from Benjamin Pierce's *Types and Programming Languages* (TAPL). This implementation emphasizes type safety, performance, and academic rigor.

### Key Features

- **Zero compiler warnings** - Clean, idiomatic Rust code (482 LOC)
- **Modular workspace architecture** - 5 specialized crates for separation of concerns
- **Multiple evaluation strategies** - Call-by-name, call-by-value, weak head normal form
- **Type systems** - Simply Typed Lambda Calculus (STLC) and System F
- **Academic rigor** - Direct references to original papers and TAPL chapters
- **Comprehensive testing** - Unit tests, integration tests, property-based testing
- **Performance focused** - Optimized release builds with LTO

### Location

```
sources/rust-implementations/tapl-rust/
├── Cargo.toml              # Workspace configuration
├── lambda-core/            # Term representation and substitution
├── lambda-eval/            # Evaluation strategies
├── lambda-types/           # Type checking and inference
├── lambda-parser/          # Parsing and pretty-printing
└── lambda-examples/        # Working examples
```

## Quick Start

### Prerequisites

- Rust 1.70+ (2021 edition)
- Cargo (comes with Rust)

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### Build and Test

```bash
cd sources/rust-implementations/tapl-rust/

# Build all crates
cargo build

# Build optimized release
cargo build --release

# Run all tests
cargo test

# Run with verbose output
cargo test -- --nocapture

# Run specific crate tests
cargo test -p lambda-core
```

### Verify Code Quality

```bash
# Format check
cargo fmt --check

# Linting
cargo clippy -- -D warnings

# Full CI pipeline
cargo build --release && cargo test && cargo fmt --check && cargo clippy -- -D warnings
```

### Run Examples

```bash
# Run examples (when available)
cargo run --example church_encodings
cargo run --example y_combinator
cargo run --example type_checking
```

## High-Level Architecture

### Workspace Structure

```
tapl-rust/
├── lambda-core      [Foundation]
│   ├── Term enum (Var, Abs, App)
│   ├── Substitution algorithm (capture-avoiding)
│   ├── Free variable computation
│   └── Common combinators (I, K, S, Y, Ω)
│
├── lambda-eval      [Evaluation]
│   ├── Call-by-name evaluator
│   ├── Call-by-value evaluator
│   ├── Weak head normal form
│   └── Evaluation traces
│
├── lambda-types     [Type Systems]
│   ├── STLC type checker
│   ├── System F polymorphism
│   └── Type inference
│
├── lambda-parser    [I/O]
│   ├── Lambda expression parser
│   └── Pretty printer
│
└── lambda-examples  [Demonstrations]
    ├── Church encodings
    ├── Recursion examples
    └── Type system examples
```

### Dependency Graph

```
lambda-examples
    ├── lambda-parser
    ├── lambda-eval
    └── lambda-types
        └── lambda-core

lambda-parser
    └── lambda-core

lambda-eval
    └── lambda-core

lambda-types
    └── lambda-core
```

**Design principle**: Everything depends on `lambda-core`, but crates at the same layer are independent.

## Core Concepts

### Term Representation

```rust
pub enum Term {
    Var(String),                           // Variable reference
    Abs { param: String, body: Box<Term> }, // λx.body
    App { rator: Box<Term>, rand: Box<Term> }, // (f x)
}
```

Named variables with capture-avoiding substitution. Design tradeoff documented in [architecture.md](architecture.md).

### Evaluation Strategies

- **Call-by-name**: Substitute unevaluated arguments (lazy)
- **Call-by-value**: Evaluate arguments before substitution (strict)
- **Weak head normal form**: Reduce only at the top level (Haskell-style)

See [examples.md](examples.md) for performance comparisons.

### Type Systems

- **Simply Typed Lambda Calculus (STLC)**: Function types (T₁ → T₂)
- **System F**: Second-order polymorphism (∀α.T)

Type checking validates terms before evaluation. See [extending.md](extending.md) for adding new type systems.

## Academic Foundation

Every implementation references original papers:

```rust
//! References:
//! - Church, A. (1932). "A Set of Postulates for the Foundation of Logic"
//! - Pierce, B. (2002). "Types and Programming Languages" Chapter 5
```

Integration with bibliography system documented in [integration.md](integration.md).

## Documentation Map

| Document | Purpose | Audience |
|----------|---------|----------|
| **[architecture.md](architecture.md)** | Workspace design decisions | Developers wanting to understand structure |
| **[development.md](development.md)** | Build, test, debug workflows | Contributors |
| **[extending.md](extending.md)** | How to add new features | Researchers implementing papers |
| **[examples.md](examples.md)** | Working code demonstrations | Learners and users |
| **[testing.md](testing.md)** | Testing strategies | Quality-focused developers |
| **[performance.md](performance.md)** | Benchmarks and optimization | Performance engineers |
| **[integration.md](integration.md)** | Academic paper integration | Academic users |

## Development Workflow

```bash
# 1. Make changes
edit sources/rust-implementations/tapl-rust/lambda-core/src/lib.rs

# 2. Verify locally
cargo build && cargo test && cargo clippy

# 3. Format
cargo fmt

# 4. Commit (all checks pass)
git add . && git commit -m "feat: implement feature"
```

See [development.md](development.md) for detailed workflows including IDE setup.

## Common Tasks

### Adding a New Combinator

```rust
// In lambda-core/src/lib.rs, add to combinators module:
pub fn true_combinator() -> Term {
    Term::abs("x", Term::abs("y", Term::var("x")))  // K combinator
}
```

### Adding an Evaluation Strategy

See [extending.md](extending.md) for step-by-step guide with full examples.

### Running Specific Tests

```bash
# Test specific module
cargo test free_vars

# Test with pattern
cargo test "test_substitution*"

# Test specific crate
cargo test -p lambda-eval
```

## Performance Characteristics

- **Substitution**: O(n) where n is term size
- **Free variables**: O(n) with HashSet operations
- **Evaluation**: Depends on strategy and term structure

See [performance.md](performance.md) for detailed benchmarks.

## Troubleshooting

### Build Failures

```bash
# Clean and rebuild
cargo clean && cargo build

# Update dependencies
cargo update
```

### Test Failures

```bash
# Run with backtrace
RUST_BACKTRACE=1 cargo test

# Run single test with output
cargo test test_name -- --nocapture --exact
```

### Clippy Warnings

```bash
# See all lints
cargo clippy -- -W clippy::all

# Auto-fix some issues
cargo clippy --fix
```

## Contributing

1. Read [development.md](development.md) for environment setup
2. Read [extending.md](extending.md) for adding features
3. Ensure all tests pass: `cargo test`
4. Ensure code is formatted: `cargo fmt`
5. Ensure no clippy warnings: `cargo clippy -- -D warnings`
6. Update documentation if adding public APIs
7. Reference academic papers when implementing algorithms

## Resources

### External References

- **TAPL Book**: Pierce, B. (2002). *Types and Programming Languages*
- **Rust Book**: [doc.rust-lang.org/book](https://doc.rust-lang.org/book/)
- **Cargo Book**: [doc.rust-lang.org/cargo](https://doc.rust-lang.org/cargo/)

### Repository Documentation

- **Academic papers**: `papers-archive/` directory
- **Bibliography**: `**/bibliography.md` files in numbered categories
- **Implementation catalog**: `docs/implementations/`

### Related Implementations

- **Haskell**: `implementations/haskell/` (when available)
- **OCaml**: `implementations/ocaml/` (when available)
- **Agda/Coq sources**: `sources/agda-src/`, `sources/coq-src/`

## Next Steps

- **New to lambda calculus?** Start with [examples.md](examples.md)
- **Want to contribute?** Read [development.md](development.md) and [extending.md](extending.md)
- **Performance critical application?** See [performance.md](performance.md)
- **Academic research?** See [integration.md](integration.md)

---

**Questions or issues?** Check existing documentation or open an issue in the repository.
