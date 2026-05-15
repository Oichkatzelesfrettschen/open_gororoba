#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_hash="$(printf "%s" "$repo_root" | sha256sum | cut -c1-16)"
tmp_root="${TMPDIR:-/tmp}/open_gororoba-cargo-build"
gate_root="${tmp_root}/gate/${repo_hash}"
ambient_root="${tmp_root}/ambient"
repo_budget_gib="${CARGO_CACHE_REPO_BUDGET_GIB:-150}"
tmp_budget_gib="${CARGO_CACHE_TMP_BUDGET_GIB:-16}"
status=0

show_path() {
    local path="$1"
    if [ -e "$path" ]; then
        du -sh "$path"
    else
        printf "0\t%s\n" "$path"
    fi
}

path_kib() {
    local path="$1"
    if [ -e "$path" ]; then
        du -sk "$path" | cut -f1
    else
        echo 0
    fi
}

format_gib() {
    local kib="$1"
    awk -v kib="$kib" 'BEGIN { printf "%.2f", kib / 1024 / 1024 }'
}

emit_budget_status() {
    local label="$1"
    local kib="$2"
    local budget_gib="$3"
    local budget_kib=$((budget_gib * 1024 * 1024))
    local gib
    gib="$(format_gib "$kib")"

    echo "${label}_total_gib=${gib}"
    echo "${label}_budget_gib=${budget_gib}"
    if [ "$kib" -gt "$budget_kib" ]; then
        echo "status.${label}=over-budget"
        status=1
    else
        echo "status.${label}=ok"
    fi
}

sum_kib() {
    local total=0
    local path
    for path in "$@"; do
        total=$((total + $(path_kib "$path")))
    done
    echo "$total"
}

echo "cargo-cache-layout"
echo "repo_root=$repo_root"
echo "repo_hash=$repo_hash"
echo "tmp_root=$tmp_root"
echo "ambient_root=$ambient_root"
echo "gate_root=$gate_root"
echo

show_path "$repo_root/.cache/cargo-default-target"
show_path "$repo_root/.cache/gate-target"
show_path "$repo_root/.cache/gate-cbuild"
show_path "$repo_root/.cache/phase7-target"
show_path "$repo_root/.cache/sparse-mainline-target"
show_path "$repo_root/.cache/cargo-home"
show_path "$repo_root/target"
show_path "$ambient_root"
show_path "$gate_root"
show_path "$HOME/.cargo"

echo

repo_total_kib="$(sum_kib \
    "$repo_root/.cache/cargo-default-target" \
    "$repo_root/.cache/gate-target" \
    "$repo_root/.cache/gate-cbuild" \
    "$repo_root/.cache/phase7-target" \
    "$repo_root/.cache/sparse-mainline-target" \
    "$repo_root/.cache/cargo-home" \
    "$repo_root/target")"
tmp_total_kib="$(sum_kib "$ambient_root" "$gate_root")"

emit_budget_status "repo_cache" "$repo_total_kib" "$repo_budget_gib"
emit_budget_status "tmp_cache" "$tmp_total_kib" "$tmp_budget_gib"

if [ "$status" -ne 0 ]; then
    echo "remediation=make cache-sweep-soft"
    exit "$status"
fi
