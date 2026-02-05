#!/usr/bin/env python3
"""
Generate an audit report for non-Python toolchains referenced by the Makefile.

This is opt-in (writes under reports/). It is not part of smoke/check.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path


def _find_repo_root() -> Path:
    p = Path.cwd().resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _has(repo_root: Path, rel: str) -> str:
    return "yes" if (repo_root / rel).exists() else "no"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/audit_nonpython_toolchains.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root()
    today = _dt.date.today().isoformat()

    lines: list[str] = []
    lines.append(f"# Non-Python Toolchains Audit ({today})")
    lines.append("")
    lines.append("Goal: ensure non-Python targets have minimal reproducible setup guidance.")
    lines.append("")
    lines.append("## Documentation presence")
    lines.append("")
    lines.append(f"- docs/requirements/cpp.md: {_has(repo_root, 'docs/requirements/cpp.md')}")
    lines.append(f"- docs/requirements/coq.md: {_has(repo_root, 'docs/requirements/coq.md')}")
    lines.append(f"- docs/requirements/latex.md: {_has(repo_root, 'docs/requirements/latex.md')}")
    quantum_doc = "docs/requirements/quantum-docker.md"
    lines.append(f"- {quantum_doc}: {_has(repo_root, quantum_doc)}")
    lines.append("")
    lines.append("## Makefile targets (manual check)")
    lines.append("")
    lines.append("- C++: `cpp-deps`, `cpp-build`, `cpp-test`, `cpp-bench`, `cpp-clean`")
    lines.append("- Coq/Rocq: `coq` (depends on `coqc`)")
    lines.append("- LaTeX: `latex` (depends on `latexmk`)")
    lines.append(
        "- Docker/Qiskit: `docker-quantum-build`, `docker-quantum-run`, "
        "`docker-quantum-shell`"
    )
    lines.append("")
    lines.append("## Checklist")
    lines.append("")
    lines.append("- [ ] Each target above has a matching requirements doc entry.")
    lines.append("- [ ] Targets are opt-in (not part of `make smoke` or `make check`).")
    lines.append("- [ ] Version/pinning guidance exists where feasible.")
    lines.append("")

    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
