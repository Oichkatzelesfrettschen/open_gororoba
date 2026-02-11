#!/usr/bin/env python3
"""
Verify dataset providers listed in DATASET_MANIFEST.md exist in fetch_datasets.rs.

Policy:
- Every provider token in the manifest Provider column (FooProvider) must exist
  in the Rust fetch registry (Box::new(...FooProvider)).
- One-way subset check is intentional: Rust may include extra providers that are
  not yet documented in the manifest.
"""

from __future__ import annotations

import re
from pathlib import Path


MANIFEST_PATH = Path("docs/external_sources/DATASET_MANIFEST.md")
FETCH_RS_PATH = Path("crates/gororoba_cli/src/bin/fetch_datasets.rs")
PROVIDER_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+Provider)")
RUST_PROVIDER_RE = re.compile(
    r"Box::new\(\s*[A-Za-z0-9_:]*?([A-Za-z0-9_]+Provider)\)"
)


def _parse_manifest_providers(manifest_text: str) -> set[str]:
    providers: set[str] = set()
    for line in manifest_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if cells[0] == "Dataset":
            continue
        if cells[0].startswith("-") and set(cells[0]) <= {"-"}:
            continue
        match = PROVIDER_TOKEN_RE.search(cells[1])
        if match:
            providers.add(match.group(1))
    return providers


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_file = repo_root / MANIFEST_PATH
    fetch_rs_file = repo_root / FETCH_RS_PATH

    if not manifest_file.exists():
        print(f"ERROR: missing {MANIFEST_PATH}")
        return 1
    if not fetch_rs_file.exists():
        print(f"ERROR: missing {FETCH_RS_PATH}")
        return 1

    manifest_text = manifest_file.read_text(encoding="utf-8")
    fetch_rs_text = fetch_rs_file.read_text(encoding="utf-8")

    manifest_providers = _parse_manifest_providers(manifest_text)
    rust_providers = set(RUST_PROVIDER_RE.findall(fetch_rs_text))

    if not manifest_providers:
        print("ERROR: no Provider tokens parsed from dataset manifest")
        return 1

    missing = sorted(manifest_providers - rust_providers)
    if missing:
        print("ERROR: dataset manifest providers missing from Rust fetch registry:")
        for provider in missing:
            print(f"- {provider}")
        return 1

    print(
        "OK: dataset manifest providers verified. "
        f"manifest={len(manifest_providers)} rust={len(rust_providers)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
