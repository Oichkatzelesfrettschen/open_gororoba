#!/bin/bash
# Wrapper to run Qiskit scripts in the Docker container
# Mounts the current data directory to capture outputs

# Ensure data directories exist
mkdir -p data/artifacts/images
mkdir -p data/csv

# Run the container
# -v $(pwd)/data:/app/data: maps host data dir to container data dir (rw)
# -v $(pwd)/src:/app/src: maps host src dir (ro) to allow script updates without rebuild
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/src:/app/src" \
    qiskit-env "$@"
