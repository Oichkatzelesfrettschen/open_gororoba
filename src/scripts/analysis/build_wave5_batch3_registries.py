#!/usr/bin/env python3
"""Deprecated compatibility shim for the integrity-resolution registry build lane."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

DEPRECATION_GUIDANCE = (
    "DEPRECATED: src/scripts/analysis/build_wave5_batch3_registries.py is a legacy alias. "
    "Use src/scripts/analysis/build_registry_integrity_resolution.py."
)


def _load_main(path: Path):
    spec = importlib.util.spec_from_file_location("_canonical_build_registry_integrity_resolution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load canonical script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def main() -> int:
    print(DEPRECATION_GUIDANCE, file=sys.stderr)
    canonical = Path(__file__).with_name("build_registry_integrity_resolution.py")
    return int(_load_main(canonical)())


if __name__ == "__main__":
    raise SystemExit(main())
