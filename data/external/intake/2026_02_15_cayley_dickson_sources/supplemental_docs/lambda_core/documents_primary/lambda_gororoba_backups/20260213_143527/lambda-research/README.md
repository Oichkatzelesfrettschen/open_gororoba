# Lambda Calculus Learning Journey

**Essential papers and guided learning paths through the mathematics of computation**

**[TARGET] Purpose**: Help students and researchers learn lambda calculus through carefully curated, accessible papers with clear explanations.

**[METRICS] What we provide**: 10 essential papers + guided learning paths + 200+ external bibliography links

**[OK] What works**: PDF viewers, mobile-friendly design, honest scope

## Quick Start

###  New to Lambda Calculus?
**[Start with Fundamentals →](https://oichkatzelesfrettschen.github.io/lambda-research/fundamentals/)**
- Church (1936) - Where it all began
- Girard (1989) - Proofs and Types (most readable)
- Essential concepts explained clearly

### 🧠 Want Advanced Topics?
**[Explore Advanced Concepts →](https://oichkatzelesfrettschen.github.io/lambda-research/advanced/)**
- Dependent types, linear logic, session types
- Modern type theory and category theory
- Research frontier topics

###  Prefer Learning by Coding?
**[Try Implementations →](https://oichkatzelesfrettschen.github.io/lambda-research/implementations/)**
- Build your own lambda calculus interpreter
- Working examples in multiple languages
- Hands-on tutorials and exercises

### [DOCS] Need Specific Papers?
**[Browse Research Library →](https://oichkatzelesfrettschen.github.io/lambda-research/research/)**
- 10 curated, accessible research papers
- Links to 200+ external papers
- Organized by topic and difficulty

## What's Actually Here

**Essential Papers (10)**: Carefully selected foundational papers with working PDF viewers
- Church (1936, 1941) - Original lambda calculus
- Girard (1987, 1989) - Linear logic and proofs-as-programs
- Martin-Löf (1984) - Type theory foundations
- Modern developments (HoTT, quantum calculi)

**Learning Structure**: Guided paths from beginner to advanced
- Prerequisite tracking
- Difficulty indicators
- Contextual explanations for each paper

**External Bibliography**: Links to 200+ papers organized by topic
- Verified accessibility
- Clear descriptions of what each paper contributes
- Honest about what we don't host locally

## Papers Archive

The **papers-archive/** directory contains a systematic collection of open access lambda calculus research papers, organized for research and educational purposes. This includes:

- **Historical papers** (pre-1980): Church's original lambda calculus, Curry-Feys combinatory logic, Scott's domain theory
- **Classical works** (1980-2000): Girard's System F and Linear Logic, Martin-Löf type theory, polymorphism foundations
- **Modern developments** (2000-2020): Homotopy type theory, practical dependent types, session types
- **Recent research** (2020+): Quantum lambda calculi, probabilistic types, AI applications

### Features

- **Automated Downloads**: Scripts to download open access papers from arXiv, author homepages, and institutional repositories
- **Metadata Management**: BibTeX bibliography, author index, topic classification, and citation tracking
- **Search Capabilities**: Full-text search across titles, authors, abstracts, and topic tags
- **Copyright Compliance**: Only downloads legally available papers (open access, public domain, author permission)

### Usage

```bash
cd papers-archive/

# Download high priority papers
make download-high

# Search for papers
make search QUERY="linear logic"
make search-author AUTHOR="Girard"

# Generate indices and cross-references
make generate-indices

# Verify URL accessibility
make verify

# Get status report
make status
```

See [papers-archive/README.md](/papers-archive/README.md) for detailed documentation.

## Contributing

This repository aims to be a comprehensive academic resource. Sources are organized by theoretical importance, practical impact, and educational value.