# Lambda Synthesis Experiments (USS)

**Extracted from:** [lambda-research](../lambda-research/)  
**Extraction Date:** 2025-12-23  
**Purpose:** Neuro-symbolic lambda term synthesis using GPU-accelerated ML

## Overview

This repository contains the **Unified Spandrel Synthesis (USS)** experimental system, a high-performance neuro-symbolic engine for lambda calculus term synthesis and analysis. USS was extracted from the main lambda-research repository to reduce repository size and maintain clear separation between core lambda calculus research and GPU-accelerated ML experiments.

## System Architecture

### Core Components

- **`src/`** - Python implementation (119MB)
  - `data/` - Data generation and processing pipelines
  - `experiments/` - Experimental configurations and runners
  - `kernels/` - Custom Triton/CUDA kernels for GPU acceleration
  - `models/` - Transformer-based synthesis models

- **`uss-venv/`** - Python virtual environment (9.5GB)
  - PyTorch 2.9+ with CUDA 12.x support
  - Triton 3.x for JIT-compiled GPU kernels
  - JAX, TensorFlow for multi-framework experimentation
  - Full scientific Python stack

### Documentation

- **`USS_REPORT.md`** - Experimental results and performance benchmarks
- **`EXPERIMENTAL_CONFIG.yml`** - Configuration for synthesis experiments
- **`requirements_experiments.txt`** - Python dependencies specification

## Hardware Requirements

USS is optimized for NVIDIA Ada Lovelace architecture (SM89):

- **GPU**: NVIDIA GeForce RTX 4070 Ti or better
- **VRAM**: 12GB minimum (16GB recommended)
- **CUDA**: Version 12.x
- **Storage**: ~10GB for virtual environment + datasets

## Performance Characteristics

From experimental validation (see USS_REPORT.md):

- **Dataset Scale**: 10M synthetic lambda terms
- **Generation Throughput**: 1.68M terms/sec
- **Model Architecture**: 12-layer Transformer (768-dim, 12-head)
- **Inference Throughput**: 2,551 samples/sec
- **Peak Batch Latency**: 200.67ms (batch size 512)

## Installation

### Option 1: Use Existing Virtual Environment

```bash
# Activate the pre-configured environment
source uss-venv/bin/activate

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option 2: Fresh Installation

```bash
# Create new virtual environment
python3 -m venv uss-venv
source uss-venv/bin/activate

# Install dependencies
pip install -r requirements_experiments.txt

# Verify CUDA support
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
```

## Usage

### Generate Synthetic Lambda Terms

```bash
source uss-venv/bin/activate
cd src/data/
python generate_terms.py --count 10000000 --output terms.parquet
```

### Train Synthesis Model

```bash
source uss-venv/bin/activate
cd src/experiments/
python train_synthesis.py --config ../../EXPERIMENTAL_CONFIG.yml
```

### Run Custom Triton Kernels

```bash
source uss-venv/bin/activate
cd src/kernels/
python tensor_contraction.py --benchmark
```

## Relationship to Lambda Research

USS is **orthogonal** to the core lambda calculus research in the main repository:

### Main Repository Focus
- Rust-based type-safe implementations (TAPL-style)
- Academic paper integration (700+ citations)
- Formal verification and theoretical foundations
- Production-ready lambda calculus systems

### USS Focus
- GPU-accelerated neuro-symbolic synthesis
- Large-scale term generation and analysis
- Transformer-based learning approaches
- Experimental ML architectures

## Why Extracted?

USS was extracted from lambda-research for several reasons:

1. **Size Reduction**: USS virtual environment is 9.5GB (77% of main repo)
2. **Clear Separation**: ML experiments vs core lambda calculus research
3. **Different Dependencies**: Heavy ML stack (PyTorch, CUDA) vs Rust toolchain
4. **Independent Evolution**: USS experiments evolve independently from core research
5. **Minimal Integration**: USS not referenced in main README/docs/build system

## Integration Points

While separated, USS can still integrate with main repository:

- **Type Validation**: Generated terms can be validated against Rust type checkers
- **Academic Grounding**: USS results can inform theoretical research
- **Benchmark Data**: USS-generated terms useful for performance testing
- **Hybrid Approaches**: Combine symbolic reasoning (Rust) with learned synthesis (USS)

## Future Directions

From USS_REPORT.md Section 6:

1. **Real Recursive Generators**: Move beyond synthetic terms to structurally complex expressions
2. **Formal Verification Integration**: Integrate Coq/Lean into training loss functions
3. **SM89 Optimization**: Leverage WGMMA and other Ada-specific instructions
4. **Cross-Framework Synthesis**: Validate terms across Rust, Haskell, OCaml implementations

## Repository Structure

```
lambda-synthesis-experiments/
├── README.md                      # This file
├── USS_REPORT.md                  # Experimental results
├── EXPERIMENTAL_CONFIG.yml        # Synthesis configuration
├── requirements_experiments.txt   # Python dependencies
├── src/                          # USS implementation (119MB)
│   ├── data/                     # Data generation
│   ├── experiments/              # Experiment runners
│   ├── kernels/                  # Custom GPU kernels
│   └── models/                   # Synthesis models
└── uss-venv/                     # Virtual environment (9.5GB)
    └── [Python packages]
```

## License

Same as main lambda-research repository (see LICENSE).

## References

- **Main Repository**: [lambda-research](../lambda-research/)
- **Extraction Rationale**: See lambda-research/experiments/README.md
- **Performance Report**: USS_REPORT.md
- **Configuration**: EXPERIMENTAL_CONFIG.yml

---

**Extracted:** 2025-12-23  
**Original Integration:** Minimal (not linked in main docs)  
**Decision:** Extract to separate repository for size reduction and clarity
