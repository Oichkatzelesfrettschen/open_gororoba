#!/bin/bash
set -x
git commit -m "infra: update core configuration and documentation" .cargo/config.toml .github/workflows/ci.yml AGENTS.md CLAUDE.md Cargo.lock Cargo.toml GEMINI.md Makefile README.md
