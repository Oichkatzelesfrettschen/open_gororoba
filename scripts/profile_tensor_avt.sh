#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${PROFILE_TIMESTAMP:-$(date +%Y-%m-%d/%H%M%S)}"
OUT_ROOT="${TENSOR_AVT_PROFILE_ROOT:-$ROOT/.cache/profiles/tensor_avt/$STAMP}"
CARGO_HOME="${CARGO_HOME:-$ROOT/.cache/cargo-home}"
CPU_TARGET_DIR="${TENSOR_AVT_CPU_TARGET_DIR:-$ROOT/.cache/tensor_avt_profile/cpu-target}"
GPU_TARGET_DIR="${TENSOR_AVT_GPU_TARGET_DIR:-$ROOT/.cache/tensor_avt_profile/gpu-target}"
PATH_KIND="${TENSOR_AVT_PATH:-single}"
DIM="${TENSOR_AVT_DIM:-256}"
BATCH_SIZE="${TENSOR_AVT_BATCH_SIZE:-16}"
ITERS="${TENSOR_AVT_ITERS:-32}"
WARMUP="${TENSOR_AVT_WARMUP:-4}"
SEED="${TENSOR_AVT_SEED:-7}"
WITH_NCU="${TENSOR_AVT_WITH_NCU:-0}"
WITH_FLAMEGRAPH="${TENSOR_AVT_WITH_FLAMEGRAPH:-1}"
WITH_SAMPLY="${TENSOR_AVT_WITH_SAMPLY:-0}"
export CARGO_HOME

mkdir -p "$OUT_ROOT/cpu" "$OUT_ROOT/gpu" "$CPU_TARGET_DIR" "$GPU_TARGET_DIR"

ARGS=(
  --path "$PATH_KIND"
  --dim "$DIM"
  --batch-size "$BATCH_SIZE"
  --iters "$ITERS"
  --warmup "$WARMUP"
  --seed "$SEED"
)

metric_or_na() {
  local pattern="$1"
  local file="$2"
  grep -F "$pattern" "$file" | sed 's/^[[:space:]]*//' | sed "s/^$pattern//" | head -n1 || true
}

heaptrack_artifact() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f \( -name 'heaptrack*.gz' -o -name 'heaptrack*.zst' -o -name 'heaptrack*.gz.zst' \) | sort | head -n1
}

capture_nvidia() {
  local output="$1"
  nvidia-smi --query-gpu=name,driver_version,pstate,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 1 >"$output" 2>&1 &
  echo $!
}

run_backend() {
  local backend="$1"
  local target_dir="$2"
  local out_dir="$3"
  shift 3

  local cargo_args=("$@")
  local bin="$target_dir/release/examples/tensor_avt_profile"
  local result_txt="$out_dir/result.txt"
  local perf_txt="$out_dir/perf.txt"
  local perf_record_txt="$out_dir/perf-record.txt"
  local perf_report_txt="$out_dir/perf-report.txt"
  local time_txt="$out_dir/time.txt"
  local flamegraph_svg="$out_dir/flamegraph.svg"
  local samply_json="$out_dir/samply.json.gz"
  local heaptrack_prefix="$out_dir/heaptrack"
  local nvidia_log="$out_dir/nvidia-smi.csv"
  local ncu_prefix="$out_dir/ncu"
  local nvidia_pid=""

  env CARGO_HOME="$CARGO_HOME" CARGO_TARGET_DIR="$target_dir" \
    cargo build --profile bench -p algebra_core "${cargo_args[@]}" --example tensor_avt_profile

  if [[ "$backend" == "gpu" ]]; then
    nvidia_pid="$(capture_nvidia "$nvidia_log")"
  fi

  perf stat --all-user \
    -e task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses \
    "$bin" --backend "$backend" "${ARGS[@]}" >"$result_txt" 2>"$perf_txt"

  perf record --call-graph dwarf -F 999 -o "$out_dir/perf.data" \
    "$bin" --backend "$backend" "${ARGS[@]}" >"$out_dir/perf-record.stdout" 2>"$perf_record_txt"
  perf report --stdio -i "$out_dir/perf.data" >"$perf_report_txt"

  /usr/bin/time -v "$bin" --backend "$backend" "${ARGS[@]}" \
    >"$out_dir/time.stdout" 2>"$time_txt"

  heaptrack --record-only -o "$heaptrack_prefix" "$bin" --backend "$backend" "${ARGS[@]}" \
    >"$out_dir/heaptrack.stdout" 2>"$out_dir/heaptrack.stderr"

  if [[ "$WITH_FLAMEGRAPH" == "1" ]]; then
    env CARGO_HOME="$CARGO_HOME" CARGO_TARGET_DIR="$target_dir" \
      cargo flamegraph --profile bench -p algebra_core "${cargo_args[@]}" \
      --example tensor_avt_profile -o "$flamegraph_svg" -- \
      --backend "$backend" "${ARGS[@]}" >"$out_dir/flamegraph.stdout" 2>"$out_dir/flamegraph.stderr"
  fi

  if [[ "$WITH_SAMPLY" == "1" ]]; then
    if ! samply record --save-only -o "$samply_json" "$bin" --backend "$backend" "${ARGS[@]}" \
      >"$out_dir/samply.stdout" 2>"$out_dir/samply.stderr"; then
      echo "samply capture skipped; see $out_dir/samply.stderr" >>"$out_dir/notes.txt"
    fi
  fi

  if [[ "$backend" == "gpu" && "$WITH_NCU" == "1" ]] && command -v ncu >/dev/null 2>&1; then
    ncu --set full --target-processes all --force-overwrite --export "$ncu_prefix" \
      "$bin" --backend "$backend" "${ARGS[@]}" >"$out_dir/ncu.stdout" 2>"$out_dir/ncu.stderr"
  fi

  if [[ -n "$nvidia_pid" ]]; then
    kill "$nvidia_pid" >/dev/null 2>&1 || true
    wait "$nvidia_pid" 2>/dev/null || true
  fi
}

run_backend cpu "$CPU_TARGET_DIR" "$OUT_ROOT/cpu"
run_backend gpu "$GPU_TARGET_DIR" "$OUT_ROOT/gpu" --features gpu

SUMMARY="$OUT_ROOT/summary.md"
{
  echo "# TensorAVT Profile Summary"
  echo
  echo "- path: $PATH_KIND"
  echo "- dim: $DIM"
  echo "- batch_size: $BATCH_SIZE"
  echo "- iters: $ITERS"
  echo "- warmup: $WARMUP"
  echo "- seed: $SEED"
  echo
  for backend in cpu gpu; do
    result_txt="$OUT_ROOT/$backend/result.txt"
    perf_txt="$OUT_ROOT/$backend/perf.txt"
    perf_record_txt="$OUT_ROOT/$backend/perf-record.txt"
    perf_report_txt="$OUT_ROOT/$backend/perf-report.txt"
    time_txt="$OUT_ROOT/$backend/time.txt"
    echo "## $backend"
    echo
    echo "- seconds: $(metric_or_na 'seconds=' "$result_txt")"
    echo "- ns_per_iter: $(metric_or_na 'ns_per_iter=' "$result_txt")"
    echo "- checksum: $(metric_or_na 'checksum=' "$result_txt")"
    echo "- max_rss_kb: $(metric_or_na 'Maximum resident set size (kbytes):' "$time_txt")"
    echo "- task_clock: $(metric_or_na 'task-clock' "$perf_txt")"
    echo "- cycles: $(metric_or_na 'cycles' "$perf_txt")"
    echo "- instructions: $(metric_or_na 'instructions' "$perf_txt")"
    echo "- cache_references: $(metric_or_na 'cache-references' "$perf_txt")"
    echo "- cache_misses: $(metric_or_na 'cache-misses' "$perf_txt")"
    echo "- branches: $(metric_or_na 'branches' "$perf_txt")"
    echo "- branch_misses: $(metric_or_na 'branch-misses' "$perf_txt")"
    echo "- perf: $perf_txt"
    echo "- perf_record: $perf_record_txt"
    echo "- perf_report: $perf_report_txt"
    echo "- time: $time_txt"
    echo "- heaptrack: $(heaptrack_artifact "$OUT_ROOT/$backend")"
    if [[ -f "$OUT_ROOT/$backend/flamegraph.svg" ]]; then
      echo "- flamegraph: $OUT_ROOT/$backend/flamegraph.svg"
    fi
    if [[ -f "$OUT_ROOT/$backend/samply.json.gz" ]]; then
      echo "- samply: $OUT_ROOT/$backend/samply.json.gz"
    fi
    if [[ -f "$OUT_ROOT/$backend/nvidia-smi.csv" ]]; then
      echo "- nvidia_smi: $OUT_ROOT/$backend/nvidia-smi.csv"
    fi
    if compgen -G "$OUT_ROOT/$backend/ncu*" >/dev/null; then
      echo "- ncu: $OUT_ROOT/$backend/ncu"
    fi
    if [[ -f "$OUT_ROOT/$backend/notes.txt" ]]; then
      echo "- notes: $OUT_ROOT/$backend/notes.txt"
    fi
    echo
  done
} >"$SUMMARY"

echo "OK: tensor_avt profiles written to $OUT_ROOT"
echo "Summary: $SUMMARY"
