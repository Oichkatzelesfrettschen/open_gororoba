#!/usr/bin/env python3
"""Verify that smoke-time verifiers do not write to reports.

reports/ is used for opt-in, human-facing outputs (inventories, backlogs, etc.).
The smoke gate must be verification-only. Verifiers under src/verification/
(excluding tools/) may *reference* reports/ (e.g., as a forbidden token in
Makefile parsing), but they must not create or write files under reports/.
"""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ver_root = repo_root / "src" / "verification"
    if not ver_root.exists():
        print("ERROR: Missing src/verification directory")
        return 2

    failures: list[str] = []
    for p in sorted(ver_root.glob("*.py")):
        rel = p.relative_to(repo_root).as_posix()
        if p.name == "verify_no_reports_writes.py":
            continue
        text = p.read_text(encoding="utf-8")

        # Heuristic: only flag files that both mention reports/ AND appear to write.
        #
        # This intentionally allows "reports/" in read-only contexts (like parsing
        # the Makefile), while still blocking common accidental writes.
        if "reports/" not in text:
            continue

        write_tokens = [
            "open(",
            "write(",
            "write_text(",
            "write_bytes(",
            ".mkdir(",
            ".touch(",
            "Path(\"reports/",
            "Path('reports/",
        ]
        if any(t in text for t in write_tokens):
            failures.append(f"{rel}: appears to write under reports/ (disallowed in smoke)")

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 2

    print("OK: verifiers do not write under reports/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
