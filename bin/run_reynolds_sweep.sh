#!/bin/bash
set -e
echo "Starting Reynolds Independence Sweep..."

# Size 8 (Fastest)
echo "Running Size 8^3..."
"${REPO_CARGO_TARGET_DIR:-.cache/gate-target}/release/warp-gpu-experiment" --experiment C --size 8 --steps 25000 > warp_exp_c_8.log 2>&1

# Size 16 (Intermediate)
echo "Running Size 16^3..."
"${REPO_CARGO_TARGET_DIR:-.cache/gate-target}/release/warp-gpu-experiment" --experiment C --size 16 --steps 25000 > warp_exp_c_16.log 2>&1

# Size 32 (Baseline)
echo "Running Size 32^3..."
"${REPO_CARGO_TARGET_DIR:-.cache/gate-target}/release/warp-gpu-experiment" --experiment C --size 32 --steps 10000 > warp_exp_c_32.log 2>&1

echo "Sweep Complete."
