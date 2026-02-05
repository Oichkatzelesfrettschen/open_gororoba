#!/usr/bin/env python3
"""
Normalize legacy inline tokens inside docs/CLAIMS_EVIDENCE_MATRIX.md.

Why: the claims matrix is meant to be a canonical, machine-parseable table.
Some historical rows embedded metadata-like tags such as:

  \\|**CONFIRMED**\\|

These are not part of the canonical schema and make auditing noisier. This
script replaces those escaped-pipe bold tokens with plain ASCII annotations:

  [legacy: CONFIRMED]

This is an opt-in fixer. It is intentionally NOT run as part of `make smoke`.
"""

# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":[],"version":1}

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Replacement:
    old: str
    new: str


REPLACEMENTS: tuple[Replacement, ...] = (
    Replacement(old="\\|**CONFIRMED**\\|", new=" [legacy: CONFIRMED] "),
    Replacement(old="\\|**REJECTED**\\|", new=" [legacy: REJECTED] "),
    Replacement(old="\\|**SUGGESTIVE**\\|", new=" [legacy: SUGGESTIVE] "),
    Replacement(old="\\|**WEAK**\\|", new=" [legacy: WEAK] "),
    Replacement(old="\\|**Trivially monotonic**\\|", new=" [legacy: Trivially monotonic] "),
    # Some rows have the opening escaped pipe but use the table delimiter '|' as the closer.
    Replacement(old="\\|**CONFIRMED**", new=" [legacy: CONFIRMED] "),
    Replacement(old="\\|**REJECTED**", new=" [legacy: REJECTED] "),
    Replacement(old="\\|**SUGGESTIVE**", new=" [legacy: SUGGESTIVE] "),
    Replacement(old="\\|**WEAK**", new=" [legacy: WEAK] "),
    Replacement(old="\\|**Trivially monotonic**", new=" [legacy: Trivially monotonic] "),
)


def _apply(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    out = text
    for r in REPLACEMENTS:
        before = out
        out = out.replace(r.old, r.new)
        counts[r.old] = before.count(r.old)
    return out, counts


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    matrix_path = repo_root / "docs" / "CLAIMS_EVIDENCE_MATRIX.md"
    if not matrix_path.exists():
        raise SystemExit("ERROR: missing docs/CLAIMS_EVIDENCE_MATRIX.md")

    original = matrix_path.read_text(encoding="utf-8")
    updated, counts = _apply(original)

    total = sum(counts.values())
    if total == 0:
        print("OK: no legacy tokens found in claims matrix")
        return 0

    matrix_path.write_text(updated, encoding="utf-8")
    print(f"Wrote: {matrix_path}")
    for old, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if n:
            print(f"Replaced: {old!r} x {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
