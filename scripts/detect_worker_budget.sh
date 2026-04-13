#!/bin/sh
# Detect a conservative worker budget (half of logical CPUs, minimum 1).
# Used by Makefile at parse time via $(shell ...).
# The authoritative Rust equivalent is `xtask worker-budget`
# (std::thread::available_parallelism / 2); this script exists only because
# cargo run at $(shell) parse time adds unacceptable startup overhead.
set -eu
threads="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
budget="$(expr "$threads" / 2)"
if [ "$budget" -lt 1 ]; then budget=1; fi
printf '%s\n' "$budget"
