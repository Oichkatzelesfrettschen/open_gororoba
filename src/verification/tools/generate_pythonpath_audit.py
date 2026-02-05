#!/usr/bin/env python3
"""
Generate a pythonpath hygiene audit report.

This is opt-in (writes under reports/). It helps track drift between:
- src layout / packaging expectations
- pytest.ini pythonpath injections (especially scripts-as-modules)
"""

from __future__ import annotations

import argparse
import configparser
import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

_IMPORT_RE = re.compile(r"^\s*(from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)(\b|\.|\s)")


@dataclass(frozen=True)
class ImportUse:
    test_path: str
    module: str


def _read_pytest_pythonpath(repo_root: Path) -> list[str]:
    ini_path = repo_root / "pytest.ini"
    cp = configparser.ConfigParser()
    cp.read(ini_path, encoding="utf-8")
    if "pytest" not in cp:
        return []
    raw = cp["pytest"].get("pythonpath", fallback="")
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(s)
    return out


def _script_modules_for_dir(repo_root: Path, rel_dir: str) -> set[str]:
    d = (repo_root / rel_dir).resolve()
    if not d.exists():
        return set()
    out: set[str] = set()
    for p in d.glob("*.py"):
        if p.name.startswith("_"):
            continue
        out.add(p.stem)
    return out


def _scan_test_imports(repo_root: Path) -> list[ImportUse]:
    out: list[ImportUse] = []
    for p in sorted((repo_root / "tests").rglob("test_*.py")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            m = _IMPORT_RE.match(line)
            if not m:
                continue
            out.append(ImportUse(test_path=p.as_posix(), module=m.group(2)))
    return out


def _render(
    repo_root: Path,
    pythonpath_entries: list[str],
    script_dirs: list[str],
    uses: list[ImportUse],
    script_modules: set[str],
) -> str:
    today = _dt.date.today().isoformat()
    lines: list[str] = []
    lines.append(f"# Pythonpath Hygiene Audit ({today})")
    lines.append("")
    lines.append("Goal: keep imports reproducible and avoid scripts-as-modules drift.")
    lines.append("")
    lines.append("## Current pytest.ini pythonpath")
    lines.append("")
    for e in pythonpath_entries:
        lines.append(f"- `{e}`")
    lines.append("")
    lines.append("## Script directories injected into pythonpath")
    lines.append("")
    for d in script_dirs:
        lines.append(f"- `{d}`")
    lines.append("")

    suspicious = [u for u in uses if u.module in script_modules]
    lines.append("## Evidence: tests importing script modules")
    lines.append("")
    lines.append(f"- Script modules discovered (union): {len(script_modules)}")
    lines.append(f"- Import hits in tests: {len(suspicious)}")
    lines.append("")
    if suspicious:
        lines.append("| test | module |")
        lines.append("| --- | --- |")
        for u in sorted(suspicious, key=lambda x: (x.module, x.test_path))[:400]:
            rel = str(Path(u.test_path).resolve().relative_to(repo_root))
            lines.append(f"| `{rel}` | `{u.module}` |")
        lines.append("")
    else:
        lines.append("(No direct imports from script modules detected in tests.)")
        lines.append("")

    lines.append("## Recommendations (phase-gated)")
    lines.append("")
    lines.append(
        "- Prefer: install package (src layout) and import only from packages under"
    )
    lines.append("  `src/`.")
    lines.append("- Treat `src/scripts/**` as entrypoints, not importable libraries.")
    lines.append(
        "- If a script contains reusable logic, move that logic into a package module and keep"
    )
    lines.append("  the script as a thin wrapper.")
    lines.append("")
    lines.append("## Checklist")
    lines.append("")
    lines.append("- [ ] Remove `src/scripts/*` entries from pytest.ini pythonpath.")
    lines.append(
        "- [ ] Refactor any tests that import script modules to import package modules"
    )
    lines.append("  instead.")
    lines.append("- [ ] Ensure `pip install -e .` is sufficient for all test imports.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/audit_pythonpath.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    # Repo root via CWD heuristic.
    here = Path.cwd().resolve()
    p = here
    repo_root = None
    while p != p.parent:
        if (p / ".git").exists():
            repo_root = p
            break
        p = p.parent
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    pythonpath_entries = _read_pytest_pythonpath(repo_root)
    script_dirs = [e for e in pythonpath_entries if e.startswith("src/scripts/")]

    script_modules: set[str] = set()
    for d in script_dirs:
        script_modules |= _script_modules_for_dir(repo_root, d)

    uses = _scan_test_imports(repo_root)
    content = _render(
        repo_root=repo_root,
        pythonpath_entries=pythonpath_entries,
        script_dirs=script_dirs,
        uses=uses,
        script_modules=script_modules,
    )

    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
