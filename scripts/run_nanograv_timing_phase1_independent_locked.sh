#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root"

exec cargo run -q -p gororoba_cli_data --bin nanograv-timing-phase1-independent "$@"
