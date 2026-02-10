#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

out_dir="data/csv"
mkdir -p "$out_dir"

runtime_csv="$out_dir/c590_attractor_runtime_baseline.csv"
sweep_csv="$out_dir/c590_attractor_ratio_sweep.csv"

runtime_dims="${C590_RUNTIME_BASELINE_DIMS:-128,256}"
sweep_dims="${C590_SWEEP_DIMS:-16,32,64,128,256,512}"
if [[ "${C590_INCLUDE_DIM1024:-0}" == "1" && "$sweep_dims" != *"1024"* ]]; then
  sweep_dims="${sweep_dims},1024"
fi

timeout_s="${C590_TIMEOUT_S:-0}"
run_with_optional_timeout() {
  if [[ "$timeout_s" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    timeout "${timeout_s}s" "$@"
  else
    "$@"
  fi
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

debug_csv="$tmp_dir/debug.csv"
release_csv="$tmp_dir/release.csv"

# Baseline runtimes (default dims 128/256) across debug and release profiles.
run_with_optional_timeout env CARGO_TARGET_DIR=/tmp/codex_target_algebra \
  cargo run -q --bin c590_attractor_sweep -- \
  --dims "$runtime_dims" \
  --profile-tag debug \
  --output "$debug_csv"

run_with_optional_timeout env CARGO_TARGET_DIR=/tmp/codex_target_algebra \
  cargo run -q --release --bin c590_attractor_sweep -- \
  --dims "$runtime_dims" \
  --profile-tag release \
  --output "$release_csv"

{
  head -n 1 "$debug_csv"
  tail -n +2 "$debug_csv"
  tail -n +2 "$release_csv"
} > "$runtime_csv"

# Full attractor sweep used by C-590 tracking.
run_with_optional_timeout env CARGO_TARGET_DIR=/tmp/codex_target_algebra \
  cargo run -q --release --bin c590_attractor_sweep -- \
  --dims "$sweep_dims" \
  --profile-tag release \
  --output "$sweep_csv"

echo "Wrote $runtime_csv"
echo "Wrote $sweep_csv"
