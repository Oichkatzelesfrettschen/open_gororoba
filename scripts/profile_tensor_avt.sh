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
DIMS_RAW="${TENSOR_AVT_DIMS:-$DIM}"
BATCH_SIZE="${TENSOR_AVT_BATCH_SIZE:-16}"
ITERS="${TENSOR_AVT_ITERS:-32}"
WARMUP="${TENSOR_AVT_WARMUP:-4}"
SEED="${TENSOR_AVT_SEED:-7}"
GPU_MODES_RAW="${TENSOR_AVT_GPU_MODES:-${TENSOR_AVT_GPU_MODE:-host,workspace,resident}}"
WITH_NCU="${TENSOR_AVT_WITH_NCU:-0}"
WITH_FLAMEGRAPH="${TENSOR_AVT_WITH_FLAMEGRAPH:-1}"
WITH_SAMPLY="${TENSOR_AVT_WITH_SAMPLY:-0}"
export CARGO_HOME

read -r -a DIMS <<<"${DIMS_RAW//,/ }"
if [[ "${#DIMS[@]}" -eq 0 ]]; then
  echo "TENSOR_AVT_DIMS resolved to an empty dimension list" >&2
  exit 2
fi
read -r -a GPU_MODES <<<"${GPU_MODES_RAW//,/ }"
if [[ "${#GPU_MODES[@]}" -eq 0 ]]; then
  echo "TENSOR_AVT_GPU_MODES resolved to an empty mode list" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT" "$CPU_TARGET_DIR" "$GPU_TARGET_DIR"

COMMON_ARGS=(
  --path "$PATH_KIND"
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
  local gpu_mode="$2"
  local target_dir="$3"
  local out_dir="$4"
  local dim="$5"
  shift 5

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
  local run_args=(
    --backend "$backend"
    --dim "$dim"
    "${COMMON_ARGS[@]}"
  )

  if [[ "$backend" == "gpu" ]]; then
    run_args+=(--gpu-mode "$gpu_mode")
  fi

  env CARGO_HOME="$CARGO_HOME" CARGO_TARGET_DIR="$target_dir" \
    cargo build --profile bench -p algebra_core "${cargo_args[@]}" --example tensor_avt_profile

  if [[ "$backend" == "gpu" ]]; then
    nvidia_pid="$(capture_nvidia "$nvidia_log")"
  fi

  perf stat --all-user \
    -e task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses \
    "$bin" "${run_args[@]}" >"$result_txt" 2>"$perf_txt"

  perf record --call-graph dwarf -F 999 -o "$out_dir/perf.data" \
    "$bin" "${run_args[@]}" >"$out_dir/perf-record.stdout" 2>"$perf_record_txt"
  perf report --stdio -i "$out_dir/perf.data" >"$perf_report_txt"

  /usr/bin/time -v "$bin" "${run_args[@]}" \
    >"$out_dir/time.stdout" 2>"$time_txt"

  heaptrack --record-only -o "$heaptrack_prefix" "$bin" "${run_args[@]}" \
    >"$out_dir/heaptrack.stdout" 2>"$out_dir/heaptrack.stderr"

  if [[ "$WITH_FLAMEGRAPH" == "1" ]]; then
    env CARGO_HOME="$CARGO_HOME" CARGO_TARGET_DIR="$target_dir" \
      cargo flamegraph --profile bench -p algebra_core "${cargo_args[@]}" \
      --example tensor_avt_profile -o "$flamegraph_svg" -- \
      "${run_args[@]}" >"$out_dir/flamegraph.stdout" 2>"$out_dir/flamegraph.stderr"
  fi

  if [[ "$WITH_SAMPLY" == "1" ]]; then
    if ! samply record --save-only -o "$samply_json" "$bin" "${run_args[@]}" \
      >"$out_dir/samply.stdout" 2>"$out_dir/samply.stderr"; then
      echo "samply capture skipped; see $out_dir/samply.stderr" >>"$out_dir/notes.txt"
    fi
  fi

  if [[ "$backend" == "gpu" && "$WITH_NCU" == "1" ]] && command -v ncu >/dev/null 2>&1; then
    ncu --set full --target-processes all --force-overwrite --export "$ncu_prefix" \
      "$bin" "${run_args[@]}" >"$out_dir/ncu.stdout" 2>"$out_dir/ncu.stderr"
  fi

  if [[ -n "$nvidia_pid" ]]; then
    kill "$nvidia_pid" >/dev/null 2>&1 || true
    wait "$nvidia_pid" 2>/dev/null || true
  fi
}

for dim in "${DIMS[@]}"; do
  dim_root="$OUT_ROOT/dim${dim}"
  mkdir -p "$dim_root/cpu"
  run_backend cpu host "$CPU_TARGET_DIR" "$dim_root/cpu" "$dim"
  for gpu_mode in "${GPU_MODES[@]}"; do
    mkdir -p "$dim_root/gpu-${gpu_mode}"
    run_backend gpu "$gpu_mode" "$GPU_TARGET_DIR" "$dim_root/gpu-${gpu_mode}" "$dim" --features gpu
  done
done

SUMMARY="$OUT_ROOT/summary.md"
{
  echo "# TensorAVT Profile Summary"
  echo
  echo "- path: $PATH_KIND"
  echo "- dims: ${DIMS[*]}"
  echo "- batch_size: $BATCH_SIZE"
  echo "- iters: $ITERS"
  echo "- warmup: $WARMUP"
  echo "- seed: $SEED"
  echo "- gpu_modes: ${GPU_MODES[*]}"
  echo
  for dim in "${DIMS[@]}"; do
    echo "## dim $dim"
    echo
    for backend_dir in "cpu" $(printf 'gpu-%s\n' "${GPU_MODES[@]}"); do
      backend_label="$backend_dir"
      result_txt="$OUT_ROOT/dim${dim}/$backend_dir/result.txt"
      perf_txt="$OUT_ROOT/dim${dim}/$backend_dir/perf.txt"
      perf_record_txt="$OUT_ROOT/dim${dim}/$backend_dir/perf-record.txt"
      perf_report_txt="$OUT_ROOT/dim${dim}/$backend_dir/perf-report.txt"
      time_txt="$OUT_ROOT/dim${dim}/$backend_dir/time.txt"
      echo "### $backend_label"
      echo
      echo "- backend: $(metric_or_na 'backend=' "$result_txt")"
      echo "- gpu_mode: $(metric_or_na 'gpu_mode=' "$result_txt")"
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
      echo "- heaptrack: $(heaptrack_artifact "$OUT_ROOT/dim${dim}/$backend_dir")"
      if [[ -f "$OUT_ROOT/dim${dim}/$backend_dir/flamegraph.svg" ]]; then
        echo "- flamegraph: $OUT_ROOT/dim${dim}/$backend_dir/flamegraph.svg"
      fi
      if [[ -f "$OUT_ROOT/dim${dim}/$backend_dir/samply.json.gz" ]]; then
        echo "- samply: $OUT_ROOT/dim${dim}/$backend_dir/samply.json.gz"
      fi
      if [[ -f "$OUT_ROOT/dim${dim}/$backend_dir/nvidia-smi.csv" ]]; then
        echo "- nvidia_smi: $OUT_ROOT/dim${dim}/$backend_dir/nvidia-smi.csv"
      fi
      if compgen -G "$OUT_ROOT/dim${dim}/$backend_dir/ncu*" >/dev/null; then
        echo "- ncu: $OUT_ROOT/dim${dim}/$backend_dir/ncu"
      fi
      if [[ -f "$OUT_ROOT/dim${dim}/$backend_dir/notes.txt" ]]; then
        echo "- notes: $OUT_ROOT/dim${dim}/$backend_dir/notes.txt"
      fi
      echo
    done
  done
} >"$SUMMARY"

echo "OK: tensor_avt profiles written to $OUT_ROOT"
echo "Summary: $SUMMARY"
