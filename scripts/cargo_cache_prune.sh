#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_hash="$(printf "%s" "$repo_root" | sha256sum | cut -c1-16)"
tmp_root="${TMPDIR:-/srv/fast/tmp}/open_gororoba-cargo-build"

paths=(
    "$repo_root/.cache/cargo-default-target"
    "$repo_root/.cache/gate-target"
    "$repo_root/.cache/phase7-target"
    "$repo_root/.cache/sparse-mainline-target"
    "$repo_root/target"
    "$tmp_root"
)

for path in "${paths[@]}"; do
    if [ -e "$path" ]; then
        echo "removing $path"
        rm -rf "$path"
    fi
done

echo "cargo cache prune complete"
