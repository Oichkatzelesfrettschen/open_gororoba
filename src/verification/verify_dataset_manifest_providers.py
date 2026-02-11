#!/usr/bin/env python3
"""
Verify dataset providers listed in registry/external_sources.toml exist in
fetch_datasets.rs.

Policy:
- Every provider token in provider-manifest narrative rows (FooProvider) must
  exist in the Rust fetch registry (Box::new(...FooProvider)).
- One-way subset check is intentional: Rust may include extra providers that are
  not yet documented in the TOML manifest narrative.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


MANIFEST_REGISTRY_PATH = Path("registry/external_sources.toml")
FETCH_RS_PATH = Path("crates/gororoba_cli/src/bin/fetch_datasets.rs")
PROVIDER_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+Provider)")
RUST_PROVIDER_RE = re.compile(
    r"Box::new\(\s*[A-Za-z0-9_:]*?([A-Za-z0-9_]+Provider)\)"
)


def _parse_manifest_providers(external_sources_toml: dict) -> set[str]:
    providers: set[str] = set()
    for row in external_sources_toml.get("document", []):
        if str(row.get("authority_level", "")).strip() != "provider_manifest":
            continue
        body = str(row.get("body_markdown", ""))
        providers.update(PROVIDER_TOKEN_RE.findall(body))
    return providers


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_file = repo_root / MANIFEST_REGISTRY_PATH
    fetch_rs_file = repo_root / FETCH_RS_PATH

    if not manifest_file.exists():
        print(f"ERROR: missing {MANIFEST_REGISTRY_PATH}")
        return 1
    if not fetch_rs_file.exists():
        print(f"ERROR: missing {FETCH_RS_PATH}")
        return 1

    manifest_data = tomllib.loads(manifest_file.read_text(encoding="utf-8"))
    fetch_rs_text = fetch_rs_file.read_text(encoding="utf-8")

    manifest_providers = _parse_manifest_providers(manifest_data)
    rust_providers = set(RUST_PROVIDER_RE.findall(fetch_rs_text))

    if not manifest_providers:
        print("ERROR: no Provider tokens parsed from registry/external_sources.toml")
        return 1

    missing = sorted(manifest_providers - rust_providers)
    if missing:
        print("ERROR: registry external-source providers missing from Rust fetch registry:")
        for provider in missing:
            print(f"- {provider}")
        return 1

    print(
        "OK: dataset providers verified from TOML registry. "
        f"manifest={len(manifest_providers)} rust={len(rust_providers)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
