#!/bin/bash
set -x
git commit -m "infra: update core configuration and documentation" .github/workflows/ci.yml AGENTS.md CLAUDE.md Cargo.lock Cargo.toml GEMINI.md Makefile README.md docs/engineering/user_local_bootstrap.txt scripts/bootstrap_user_local_xdg.sh
