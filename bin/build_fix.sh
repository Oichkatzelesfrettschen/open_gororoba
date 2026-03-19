#!/bin/bash
export RUST_BACKTRACE=1
cargo build -p gororoba_engine --features gpu > engine_build.log 2>&1
cargo build -p gororoba_cli --bin warp-ring-3d --features gpu >> engine_build.log 2>&1
