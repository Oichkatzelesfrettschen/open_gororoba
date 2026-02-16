# Experiments Directory

## USS Extraction Notice

**Status:** USS (Unified Spandrel Synthesis) has been **extracted to a separate repository**  
**Location:** `../lambda-synthesis-experiments/`  
**Extraction Date:** 2025-12-23  
**Reason:** Repository size reduction (12.4GB → <2GB)

## What Was USS?

USS was a GPU-accelerated neuro-symbolic lambda term synthesis system using:
- PyTorch 2.9+ with CUDA 12.x
- Custom Triton kernels for Tensor Contraction
- 12-layer Transformer models for term synthesis
- 10M+ synthetic lambda term generation

**Hardware Target:** NVIDIA RTX 4070 Ti (SM89, 12GB VRAM)  
**Performance:** 1.68M terms/sec generation, 2,551 samples/sec inference

## Why Extracted?

### Size Impact
- **USS Virtual Environment:** 9.5GB (77% of repository)
- **USS Source Code:** 119MB
- **Total USS Size:** 9.6GB
- **Remaining Repo Size:** <2GB (core lambda calculus research)

### Architectural Reasons
1. **Orthogonal Focus**: USS (ML experiments) vs Core (lambda calculus theory/implementation)
2. **Different Dependencies**: Heavy ML stack (PyTorch, CUDA) vs Rust toolchain
3. **Minimal Integration**: USS not referenced in README, MkDocs, or build system
4. **Independent Evolution**: USS experiments evolve independently from core research

### Consolidation Analysis
From CONSOLIDATION_ANALYSIS.md:
> USS feels "bolted on" rather than integrated into the lambda-research ecosystem.
> It operates in a parallel universe: different dependencies, different workflow,
> different documentation, minimal cross-referencing.

## Accessing USS

### Location
```bash
cd ../lambda-synthesis-experiments/
```

### Quick Start
```bash
# Navigate to USS repository
cd ../lambda-synthesis-experiments/

# Activate USS environment
source uss-venv/bin/activate

# Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run experiments
cd src/experiments/
python uss_pipeline.py
```

### Full Documentation
See `../lambda-synthesis-experiments/README.md` for:
- Installation instructions
- Hardware requirements
- Performance benchmarks
- Usage examples
- Integration guidance

## Integration Points

While separated, USS can still integrate with lambda-research:

### Type Validation
```bash
# Generate terms with USS
cd ../lambda-synthesis-experiments/
source uss-venv/bin/activate
python src/data/generator.py --output terms.json

# Validate with Rust type checker
cd ../lambda-research/
cargo run --bin type-checker -- --input ../lambda-synthesis-experiments/terms.json
```

### Benchmark Data
USS-generated terms can serve as test data for lambda-research implementations:
```bash
# Use USS terms for performance benchmarking
cd ../lambda-research/benchmarks/
./benchmark_runner.sh --input ../lambda-synthesis-experiments/src/data/shards/
```

### Academic Grounding
USS experimental results can inform theoretical research documented in lambda-research.

## Future Experiments

This directory (`experiments/`) is reserved for **lightweight** experiments that:
- Don't require heavy ML dependencies
- Integrate directly with Rust/Haskell implementations
- Are documented in main repository docs
- Don't add significant repository size

**Heavy ML experiments** (like USS) should remain in `../lambda-synthesis-experiments/`.

## Decision Rationale

### Option A: Extract to Separate Repository (CHOSEN) ✓
**Pros:**
- Immediate 9.6GB size reduction
- Clean separation of concerns
- Independent dependency management
- Clear architectural boundaries

**Cons:**
- Requires cross-repository coordination for integration
- USS changes don't appear in main repo history

### Option B: Make USS Optional (NOT CHOSEN)
**Pros:**
- Keep USS in main repository
- Simpler for users who want everything

**Cons:**
- Still requires 9.6GB storage even if optional
- Clutter in main repository
- Mixed dependency management

## References

- **USS Repository:** `../lambda-synthesis-experiments/`
- **USS Documentation:** `../lambda-synthesis-experiments/README.md`
- **Performance Report:** `../lambda-synthesis-experiments/USS_REPORT.md`
- **Extraction Commit:** lambda-research commit [TBD]
- **USS Initial Commit:** lambda-synthesis-experiments commit c624846

---

**Decision:** Extract USS to separate repository  
**Executed:** 2025-12-23  
**Impact:** Repository size reduced from 12.4GB to <2GB  
**USS Location:** `../lambda-synthesis-experiments/`
