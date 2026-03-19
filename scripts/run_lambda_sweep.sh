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
echo ""
echo "Lambda Sweep Summary:"
echo "============================================================"
printf "%-10s %-15s %-10s %-s\n" "Lambda" "P-Value" "Channels" "Status"
echo "------------------------------------------------------------"

for file in $(ls "$OUTPUT_DIR"/e027_lambda_*.toml | sort -V); do
    l_val=$(grep "lambda =" "$file" | head -n 1 | awk -F'=' '{print $2}' | xargs)
    p_val=$(grep "p_value =" "$file" | head -n 1 | awk -F'=' '{print $2}' | xargs)
    n_chan=$(grep "n_channels_detected =" "$file" | head -n 1 | awk -F'=' '{print $2}' | xargs)
    
    if [[ -n "$l_val" && -n "$p_val" ]]; then
        status="FAIL"
        if (( $(echo "$p_val < 0.05" | bc -l) )); then
            status="PASS"
        fi
        printf "%-10s %-15s %-10s %-s\n" "$l_val" "$p_val" "$n_chan" "$status"
    fi
done
echo "============================================================"

echo ""
echo "Next steps:"
echo "1. Review results to identify optimal lambda"
echo "2. Update Phase 1.4.3 grid sweep with optimal lambda"
echo "3. Proceed to Phase 1.5 analytical work"
