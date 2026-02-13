#!/bin/bash
# Phase 1.4.1: Lambda Scaling Parameter Sweep
# Systematically explore lambda in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
# Grid: 64^3 (optimal resolution from Phase 1.4.2)
# Steps: 2500 (sufficient convergence)

set -euo pipefail

GRID_SIZE=64
LBM_STEPS=2500
NU_BASE=0.333
N_PERMUTATIONS=1000
SEED=42
OUTPUT_DIR="data/e027/lambda_sweep"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Lambda values to sweep
LAMBDAS=(0.1 0.5 1.0 2.0 5.0 10.0 20.0 50.0)

echo "========================================"
echo "Phase 1.4.1: Lambda Scaling Sweep"
echo "========================================"
echo "Grid: ${GRID_SIZE}^3"
echo "LBM steps: $LBM_STEPS"
echo "Lambda values: ${LAMBDAS[*]}"
echo "Total runs: ${#LAMBDAS[@]}"
echo ""

# Ensure percolation-experiment binary is built with GPU support
echo "Building percolation-experiment binary with GPU support..."
cargo build --release --features gpu --bin percolation-experiment -j$(nproc)
echo ""

# Run sweep
for lambda in "${LAMBDAS[@]}"; do
    echo "========================================"
    echo "Running: lambda = $lambda"
    echo "========================================"

    OUTPUT_FILE="${OUTPUT_DIR}/e027_lambda_${lambda}_grid${GRID_SIZE}.toml"

    # Run experiment with GPU acceleration
    cargo run --release --features gpu --bin percolation-experiment -- \
        --grid-size "$GRID_SIZE" \
        --lbm-steps "$LBM_STEPS" \
        --nu-base "$NU_BASE" \
        --lambda "$lambda" \
        --n-permutations "$N_PERMUTATIONS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        --use-gpu \
        --verbose

    # Move result to labeled file
    if [ -f "$OUTPUT_DIR/e027_results.toml" ]; then
        mv "$OUTPUT_DIR/e027_results.toml" "$OUTPUT_FILE"
        echo "Saved result to: $OUTPUT_FILE"
    else
        echo "WARNING: Expected output file not found"
    fi

    echo ""
done

echo "========================================"
echo "Lambda Sweep Complete"
echo "========================================"
echo "Results in: $OUTPUT_DIR"
echo ""

# Summary
echo "Generating summary..."
python3 - <<'EOF'
import os
import glob

output_dir = "data/e027/lambda_sweep"
results = []

for file in sorted(glob.glob(f"{output_dir}/e027_lambda_*.toml")):
    with open(file, 'r') as f:
        content = f.read()
        # Simple TOML parsing
        lambda_val = None
        p_value = None
        n_channels = None
        for line in content.split('\n'):
            if 'lambda = ' in line:
                lambda_val = line.split('=')[1].strip()
            if 'p_value = ' in line and p_value is None:  # First p_value is correlation
                p_value = line.split('=')[1].strip()
            if 'n_channels_detected = ' in line:
                n_channels = line.split('=')[1].strip()
        if lambda_val and p_value:
            results.append((lambda_val, p_value, n_channels))

if results:
    print("\nLambda Sweep Summary:")
    print("=" * 60)
    print(f"{'Lambda':<10} {'P-Value':<15} {'Channels':<10} {'Status'}")
    print("-" * 60)
    for lambda_val, p_value, n_channels in results:
        status = "PASS" if float(p_value) < 0.05 else "FAIL"
        print(f"{lambda_val:<10} {p_value:<15} {n_channels:<10} {status}")
    print("=" * 60)
else:
    print("No results found")
EOF

echo ""
echo "Next steps:"
echo "1. Review results to identify optimal lambda"
echo "2. Update Phase 1.4.3 grid sweep with optimal lambda"
echo "3. Proceed to Phase 1.5 analytical work"
