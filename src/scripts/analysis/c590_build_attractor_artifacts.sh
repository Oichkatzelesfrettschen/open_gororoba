#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

out_dir="data/csv"
mkdir -p "$out_dir"

runtime_csv="$out_dir/c590_attractor_runtime_baseline.csv"
sweep_csv="$out_dir/c590_attractor_ratio_sweep.csv"

fast_mode="${C590_FAST_MODE:-0}"
if [[ "$fast_mode" == "1" ]]; then
  # CI-friendly fast profile: avoid high-cost 512/1024 unless explicitly re-enabled.
  runtime_dims="${C590_RUNTIME_BASELINE_DIMS:-128,256}"
  sweep_dims="${C590_SWEEP_DIMS:-16,32,64,128,256}"
  timeout_s="${C590_TIMEOUT_S:-90}"
else
  runtime_dims="${C590_RUNTIME_BASELINE_DIMS:-128,256}"
  sweep_dims="${C590_SWEEP_DIMS:-16,32,64,128,256,512}"
  timeout_s="${C590_TIMEOUT_S:-0}"
fi

if [[ "${C590_INCLUDE_DIM512:-0}" == "1" && "$sweep_dims" != *"512"* ]]; then
  sweep_dims="${sweep_dims},512"
fi

if [[ "${C590_INCLUDE_DIM1024:-0}" == "1" && "$sweep_dims" != *"1024"* ]]; then
  sweep_dims="${sweep_dims},1024"
fi

echo "C590 runtime dims: $runtime_dims"
echo "C590 sweep dims:   $sweep_dims"
echo "C590 timeout(s):   $timeout_s"

run_with_optional_timeout() {
  if [[ "$timeout_s" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    timeout "${timeout_s}s" "$@"
  else
    "$@"
  fi
}

# Baseline runtimes (default dims 128/256) across debug and release profiles.
run_with_optional_timeout env CARGO_TARGET_DIR=/tmp/codex_target_algebra \
  cargo run -q --bin c590_attractor_sweep -- \
  --dims "$runtime_dims" \
  --profile-tag debug \
  --output "$runtime_csv"

run_with_optional_timeout env CARGO_TARGET_DIR=/tmp/codex_target_algebra \
  cargo run -q --release --bin c590_attractor_sweep -- \
  --dims "$runtime_dims" \
  --profile-tag release \
  --output "$runtime_csv" \
  --append

# Full attractor sweep used by C-590 tracking.
run_with_optional_timeout env CARGO_TARGET_DIR=/tmp/codex_target_algebra \
  cargo run -q --release --bin c590_attractor_sweep -- \
  --dims "$sweep_dims" \
  --profile-tag release \
  --output "$sweep_csv"

echo "Wrote $runtime_csv"
echo "Wrote $sweep_csv"
