#!/usr/bin/env python3
"""Deprecated shim for semantic-atoms registry verification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

DEPRECATION_GUIDANCE = (
    "DEPRECATED: src/verification/verify_wave5_batch1_registries.py is a compatibility shim; "
    "use src/verification/verify_registry_semantic_atoms.py."
)


def _load_main(path: Path):
    spec = importlib.util.spec_from_file_location("_canonical_verify_registry_semantic_atoms", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load canonical script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def main() -> int:
    print(DEPRECATION_GUIDANCE, file=sys.stderr)
    canonical = Path(__file__).with_name("verify_registry_semantic_atoms.py")
    return _load_main(canonical)()


if __name__ == "__main__":
    raise SystemExit(main())
