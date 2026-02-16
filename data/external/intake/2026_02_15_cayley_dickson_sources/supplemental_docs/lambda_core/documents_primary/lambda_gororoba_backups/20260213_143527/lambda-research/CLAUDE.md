# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a comprehensive lambda calculus research repository focused on building production-ready implementations with world-class academic foundations. The repository contains:

**CORE STRENGTHS:**
- 31 numbered categories with 700+ citations (world-class academic organization)
- Internal Rust workspace with modular lambda calculus implementations
- Modern documentation infrastructure with MkDocs
- Automated validation and quality assurance systems
- Comprehensive cross-reference system and bibliography management

**CURRENT FOCUS:**
- Production-quality Rust implementations in `sources/rust-implementations/`
- Progressive development: Untyped λ-calculus → Simply Typed → System F → Dependent Types
- Academic rigor with practical implementation emphasis
- Quality over quantity: deep implementation rather than superficial coverage

## Repository Structure

### Primary Implementation Core
Internal Rust workspace at `sources/rust-implementations/tapl-rust/` with modular crates:
- **lambda-core**: Term representation, substitution, alpha-equivalence
- **lambda-eval**: Evaluation strategies (call-by-value, call-by-name, normal order)
- **lambda-types**: Type checking and inference (STLC, System F)
- **lambda-parser**: Parsing lambda expressions, pretty-printing
- **lambda-examples**: Working examples and demonstrations

### Academic Reference Structure (01-31)
The numbered directories provide comprehensive academic foundations:
- **01-XX-name/** - Academic organization: `papers/`, `implementations/`, `tutorials/`, `historical/`
- **Current State**: Excellent bibliographies (700+ citations), minimal in-repo implementations
- **Strategy**: Use as reference while building implementations in `sources/`

### Supporting Infrastructure
- **`papers-archive/`** - Academic paper collection with download automation
- **`implementations/`** - Multi-language research implementations (secondary to Rust core)
- **Root Documentation** - Cross-references, bibliographies, comprehensive indices
- **Build System** - Unified Makefile targeting actual implementations

### Key Files
- **`sources/rust-implementations/tapl-rust/`** - Primary Rust workspace (482 LOC)
- **`sources/rust-implementations/church-unsolvable-1936/`** - Experimental quantum extensions (929 LOC)
- **`Makefile`** - Master build system targeting internal implementations
- **`validate-repository.py`** - Quality assurance and validation
- **`mkdocs.yml`** - Documentation generation
- **`COMPREHENSIVE_INDEX.md`** - Academic reference index

## Build Systems and Development

### PRIMARY: Internal Rust Workspace

```bash
# Build internal Rust implementations
cd sources/rust-implementations/tapl-rust
cargo build --all                    # Build all crates
cargo test --all                     # Run all tests
cargo clippy --all -- -D warnings    # Zero warnings policy
cargo fmt --all --check              # Formatting compliance
cargo bench --all                    # Performance benchmarks

# Build experimental implementations
cd sources/rust-implementations/church-unsolvable-1936
cargo build --release
cargo test
```

### Essential Commands

#### Repository Health & Quality Assurance
```bash
# ALWAYS RUN THESE FIRST
./validate-repository.py                 # Full validation of structure and links
./standardize_bibliography.py           # Bibliography consistency check
make status                             # Repository health status
make verify                            # Academic integrity verification
```

#### Core Development Workflow
```bash
# Primary development cycle
cd sources/rust-implementations/tapl-rust
cargo build && cargo test && cargo clippy && cargo fmt

# Documentation development
source venv/bin/activate               # ALWAYS activate before docs work
mkdocs serve  # Live server (127.0.0.1:8000)
mkdocs build  # Production build

# Master build system (targets available implementations)
make build                             # Build available implementations + docs
make test                              # Run all available tests
make clean                             # Clean all build artifacts
make ci                               # Full CI pipeline
```

#### Supporting Language Targets (Secondary)
```bash
# Multi-language implementations (when they exist)
make build-scala                       # Build Scala implementations (if any)
make build-haskell                     # Build Haskell implementations (if any)
make build-idris                       # Build Idris implementations (if any)

# Academic source repositories (reference only)
cd sources/agda-src/ && make install   # Agda compiler source
cd sources/coq-src/ && make world      # Coq/Rocq theorem prover source
```

#### Paper Archive Management
```bash
cd papers-archive/
make download-high                  # Download high-priority papers
make search QUERY="linear logic"    # Search papers by topic
make verify                         # Verify URL accessibility
make status                         # Get repository status
```

## Code Architecture

### PRODUCTION-READY RESEARCH PLATFORM

This repository balances academic rigor with production-quality implementations:

1. **Implementation-First**: Working code drives theoretical understanding
2. **Academic Foundation**: 708+ papers provide comprehensive theoretical backing
3. **Quality Focus**: Deep implementation of core concepts vs shallow coverage of everything
4. **Progressive Complexity**: Build understanding through working implementations

### RUST-CENTERED ARCHITECTURE

**IMPLEMENTATION THEMES:**
- **Type-Safe Design**: Rust's type system enforces correctness
- **Performance Focus**: Zero-cost abstractions for lambda calculus operations
- **Memory Safety**: Deterministic performance with zero compiler warnings
- **Modular Design**: Workspace crates for separation of concerns
- **Academic Validation**: Algorithms reference original papers (TAPL, Pierce)

**IMPLEMENTATION PATTERNS:**
```rust
// Type-safe lambda calculus representations
pub enum Term {
    Var(String),
    Abs(String, Box<Term>),
    App(Box<Term>, Box<Term>),
}

// Context-aware type checking
pub fn typecheck(ctx: &Context, term: &Term) -> Result<Type, TypeError>

// Capture-avoiding substitution
pub fn substitute(term: Term, var: &str, replacement: Term) -> Term
```

### SUPPORTING MULTI-LANGUAGE ECOSYSTEM
- **Rust**: Primary production implementations (tapl-rust workspace)
- **Haskell**: Academic prototyping in `implementations/haskell/`
- **OCaml**: Formal verification links (Coq integration)
- **Scala**: JVM-based implementations in `implementations/scala/`
- **Scheme/SML/Idris**: Educational examples in `implementations/`
- **Academic Sources**: Agda/Coq compiler sources for formal proofs

## Development Workflow

### PRIORITY: Extending Core Implementations

**IMPLEMENTATION-FIRST APPROACH:**
1. **Start in Rust**: Implement in `sources/rust-implementations/tapl-rust/`
2. **Build Core Functionality**: Focus on working, tested, documented code
3. **Academic Integration**: Reference papers, add to bibliography after implementation
4. **Quality Gate**: `cargo test && cargo fmt && cargo clippy` before any commit

**NEW VARIANT DEVELOPMENT:**
```bash
# 1. Implement in Rust workspace
cd sources/rust-implementations/tapl-rust/
# Create new crate or extend existing ones
cargo new lambda-<feature>
# Implement, test, document with paper references

# 2. Update academic structure
mkdir 32-new-variant/
cd 32-new-variant/ && mkdir papers implementations tutorials historical
# Add bibliography.md with academic references

# 3. Cross-reference integration
./validate-repository.py    # Ensure quality
./standardize_bibliography.py    # Bibliography consistency
```

## Rust Implementation Strategy

This repository contains production-quality Rust implementations in `sources/rust-implementations/`:

### Primary Workspace: tapl-rust (482 LOC)

A modular workspace following Types and Programming Languages (TAPL) textbook structure:

**Crate Structure:**
- **lambda-core**: Core term representation, substitution, alpha-equivalence
- **lambda-eval**: Evaluation strategies (call-by-value, call-by-name, normal order)
- **lambda-types**: Type checking and inference (STLC, System F)
- **lambda-parser**: Parsing lambda expressions, pretty-printing
- **lambda-examples**: Working examples and demonstrations

**Build Instructions:**
```bash
cd sources/rust-implementations/tapl-rust
cargo build --release       # Build all crates
cargo test --all            # Run all tests (target: 80% coverage)
cargo clippy --all -- -D warnings  # Zero warnings required
cargo fmt --all             # Code formatting
cargo doc --all --open      # Generate and view documentation
```

**Development Workflow:**
1. Implement new calculus variant in appropriate crate
2. Add comprehensive tests with property-based testing where applicable
3. Document in code comments with paper references
4. Update `docs/implementations/rust/` with high-level documentation
5. Link to papers in relevant `XX-*/bibliography.md` files
6. Run full validation before commit

### Secondary Project: church-unsolvable-1936 (929 LOC)

Experimental implementation exploring Church's original 1936 paper with quantum extensions.

**Build Instructions:**
```bash
cd sources/rust-implementations/church-unsolvable-1936
cargo build --release
cargo test
cargo bench  # Performance benchmarks
```

## Implementation-Paper Integration

Every Rust implementation must:
1. **Reference Original Papers**: Include citations in code comments
2. **Document Implementation Status**: Add status to relevant `bibliography.md`
3. **Provide Working Examples**: Demonstrate paper's algorithms in `lambda-examples/`
4. **Validate Correctness**: Tests that validate against formal specifications

Example code structure:
```rust
//! Beta Reduction Implementation
//!
//! Based on Alonzo Church's lambda calculus (1936).
//! 
//! References:
//! - Church, A. (1936). "An unsolvable problem of elementary number theory"
//! - Pierce, B. (2002). "Types and Programming Languages", Chapter 5

pub fn beta_reduce(term: Term) -> Term {
    // Implementation with paper-referenced algorithm
}
```

### REALISTIC BIBLIOGRAPHY MANAGEMENT
- **Implementation-Driven**: Add papers as you implement concepts
- **Quality Over Quantity**: Focus on foundational papers, not everything ever written
- **Academic Standards**: Author, year, title, venue, DOI when available
- **Regular Validation**: `./validate-repository.py` catches broken links and inconsistencies

### IMPLEMENTATION STANDARDS (ENFORCED)
```rust
//! Module: System F Polymorphism
//!
//! Implementation of second-order polymorphic lambda calculus.
//!
//! References:
//! - Girard, J-Y. (1972). "Interpretation fonctionnelle et elimination des coupures"
//! - Reynolds, J. (1974). "Towards a theory of type structure"
//! - Pierce, B. (2002). "Types and Programming Languages" Chapter 23

use std::collections::HashMap;

/// System F term representation with type abstraction and application
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    // ... type-safe implementation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polymorphic_identity() {
        // Test cases based on TAPL examples
    }
}
```

## Quality Standards

### IMPLEMENTATION QUALITY (ENFORCED)
```bash
# REQUIRED before any commit
cd sources/rust-implementations/tapl-rust/
cargo test     # All tests pass
cargo fmt      # Code formatted
cargo clippy   # No linting warnings
cargo build --release  # Release build succeeds

# Repository validation
./validate-repository.py  # No errors allowed
./standardize_bibliography.py  # Bibliography consistency
```

### ACADEMIC RIGOR
- **Paper References**: Every algorithm must cite original paper
- **Implementation Validation**: Code validated against formal specifications
- **Bibliography Quality**: Verified URLs, consistent formatting, academic sources
- **Theoretical Accuracy**: Cross-reference system validates theoretical connections

### PERFORMANCE STANDARDS
```rust
// Example: Beta reduction performance requirement
#[cfg(test)]
mod performance_tests {
    #[test]
    fn test_beta_reduction_performance() {
        let large_term = generate_large_term(1000);
        let start = std::time::Instant::now();
        let result = beta_reduce(large_term);
        let duration = start.elapsed();
        assert!(duration < std::time::Duration::from_millis(100));
    }
}
```

### DOCUMENTATION STANDARDS
- **Comprehensive**: Every public function documented with examples
- **Mathematical Notation**: MathJax for complex formulas
- **Educational Links**: Connect theory to implementation
- **Working Examples**: All code examples must compile and run

## CURRENT DEVELOPMENT PRIORITIES

### PHASE 1: Core Implementation Completion (IMMEDIATE)
```bash
# Focus areas for next development cycle
cd sources/rust-implementations/tapl-rust/

# 1. Enhance existing implementations
# - Add comprehensive error messages
# - Improve performance of beta reduction
# - Add pretty-printing and debugging tools

# 2. Add missing core features
# - De Bruijn indices for efficient substitution
# - Parallel reduction strategies
# - Interactive evaluation REPL

# 3. Testing and validation
# - Property-based testing with QuickCheck
# - Performance benchmarks
# - Cross-implementation validation
```

### PHASE 2: Advanced Type Systems (NEXT)
- **Linear Types**: Resource-aware computation (extend existing foundation)
- **Dependent Types**: Complete the x2_dependent prototype
- **Effect Systems**: Computational effects and monadic encapsulation

### PHASE 3: Ecosystem Integration (FUTURE)
- **Multi-Language**: Haskell and OCaml implementations
- **Formal Verification**: Coq proofs for key properties
- **Educational Tools**: Interactive learning environment

## REQUIRED DEPENDENCIES

### ESSENTIAL (INSTALL FIRST)
```bash
# Rust toolchain (primary development)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable && rustup default stable

# Python documentation environment
python3 -m venv venv
source venv/bin/activate
pip install mkdocs mkdocs-material mkdocs-mermaid2-plugin

# Repository validation
python3 -m pip install requests beautifulsoup4 pyyaml
```

### SECONDARY (OPTIONAL)
- Haskell Platform (for academic implementations)
- OCaml/Dune (for Coq integration)
- LaTeX (for mathematical document generation)

## DAILY DEVELOPMENT WORKFLOW

### STANDARD DEVELOPMENT CYCLE
```bash
# 1. Repository health check
./validate-repository.py && ./standardize_bibliography.py

# 2. Core implementation work
cd sources/rust-implementations/tapl-rust/
cargo build && cargo test && cargo fmt && cargo clippy

# 3. Documentation updates
source venv/bin/activate
mkdocs serve  # Live preview

# 4. Quality assurance
make ci  # Full build and test cycle

# 5. Commit only after all checks pass
git add . && git commit -m "feat: implement <feature> with academic validation"
```

### RESEARCH WORKFLOW
```bash
# Academic paper integration
cd papers-archive/
make search QUERY="<topic>"  # Find relevant papers
# Read, understand, implement, then add to bibliography

# Cross-reference updates
./validate-repository.py  # Ensure theoretical connections are valid
```

## PROJECT VISION STATEMENT

**This repository transforms lambda calculus from academic theory into production-ready, performant implementations while maintaining the highest standards of academic rigor.**

**CORE VALUES:**
- **Quality over Quantity**: Deep implementation beats superficial coverage
- **Academic Rigor**: Every line of code references foundational papers
- **Performance**: Production-ready systems, not just academic toys
- **Progressive Complexity**: Build understanding through working implementations
- **Open Science**: Comprehensive bibliography and reproducible research

**SUCCESS METRICS:**
- Zero-warning Rust builds with comprehensive test coverage
- All implementations validated against academic specifications
- Documentation that teaches both theory and practice
- Used by both researchers and practitioners in industry

---

*This repository represents a new paradigm: academically rigorous lambda calculus implementations optimized for both research and production use, bridging the gap between theory and practice in programming language design.*
- treat all warnings as errors: build, scope out, sanity check, engineer, design, code, fix, test, implement and synthesize fully complete solutions.
