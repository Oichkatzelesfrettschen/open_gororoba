#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_hash="$(printf "%s" "$repo_root" | sha256sum | cut -c1-16)"
tmp_root="${TMPDIR:-/tmp}/open_gororoba-cargo-build"
gate_root="${tmp_root}/gate/${repo_hash}"
ambient_root="${tmp_root}/ambient"

show_path() {
    local path="$1"
    if [ -e "$path" ]; then
        du -sh "$path"
    else
        printf "0\t%s\n" "$path"
    fi
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
show_path "$repo_root/.cache/phase7-target"
show_path "$repo_root/.cache/sparse-mainline-target"
show_path "$repo_root/.cache/cargo-home"
show_path "$ambient_root"
show_path "$gate_root"
show_path "$HOME/.cargo"
