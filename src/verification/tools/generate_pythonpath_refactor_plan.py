#!/usr/bin/env python3
"""
Generate a plan for refactoring pytest.ini pythonpath injections.

This repository currently allows tests to import many scripts as top-level modules
by injecting src/scripts/<category> directories into pytest.ini pythonpath.

Best practice for a src-layout project is to keep imports namespaced and avoid
scripts-as-modules. This report inventories the current coupling and proposes a
phase-gated migration plan.
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
class ScriptModule:
    module: str
    rel_path: str
    category: str


def _find_repo_root() -> Path:
    p = Path.cwd().resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _read_pytest_pythonpath(repo_root: Path) -> list[str]:
    ini_path = repo_root / "pytest.ini"
    cp = configparser.ConfigParser()
    cp.read(ini_path, encoding="utf-8")
    raw = cp.get("pytest", "pythonpath", fallback="")
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if s:
            out.append(s)
    return out


def _scan_test_imports(repo_root: Path) -> set[str]:
    mods: set[str] = set()
    for p in sorted((repo_root / "tests").rglob("test_*.py")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            m = _IMPORT_RE.match(line)
            if not m:
                continue
            mods.add(m.group(2))
    return mods


def _index_script_modules(repo_root: Path, categories: list[str]) -> dict[str, ScriptModule]:
    out: dict[str, ScriptModule] = {}
    for cat in categories:
        d = repo_root / "src" / "scripts" / cat
        if not d.exists():
            continue
        for p in sorted(d.glob("*.py")):
            if p.name.startswith("_"):
                continue
            out[p.stem] = ScriptModule(
                module=p.stem,
                rel_path=p.relative_to(repo_root).as_posix(),
                category=cat,
            )
    return out


def _render(
    repo_root: Path,
    pythonpath_entries: list[str],
    script_modules: dict[str, ScriptModule],
    imported_modules: set[str],
) -> str:
    today = _dt.date.today().isoformat()
    lines: list[str] = []
    lines.append(f"# Pythonpath Refactor Plan ({today})")
    lines.append("")
    lines.append("Goal: migrate away from scripts-as-top-level-modules without breaking tests.")
    lines.append("")

    script_path_entries = [e for e in pythonpath_entries if e.startswith("src/scripts/")]
    cats = sorted({Path(e).parts[-1] for e in script_path_entries})

    lines.append("## Current state (pytest.ini)")
    lines.append("")
    lines.append("- pythonpath entries:")
    for e in pythonpath_entries:
        lines.append(f"  - `{e}`")
    lines.append("")
    lines.append(f"- injected script categories: {', '.join(cats) if cats else '(none)'}")
    lines.append("")

    coupled = sorted([m for m in imported_modules if m in script_modules])
    lines.append("## Coupling inventory")
    lines.append("")
    lines.append(f"- script modules indexed: {len(script_modules)}")
    lines.append(f"- test-imported modules that resolve to scripts: {len(coupled)}")
    lines.append("")
    if coupled:
        lines.append("| module | current file | proposed import |")
        lines.append("| --- | --- | --- |")
        for m in coupled[:300]:
            sm = script_modules[m]
            proposed = f"scripts.{sm.category}.{sm.module}"
            lines.append(f"| `{m}` | `{sm.rel_path}` | `{proposed}` |")
        lines.append("")

    lines.append("## Phase-gated migration plan")
    lines.append("")
    lines.append("Phase A: make scripts importable as a package (no behavior change)")
    lines.append("- Add `__init__.py` under `src/scripts/` and each script category directory.")
    lines.append("- Keep pytest.ini pythonpath entries unchanged initially.")
    lines.append("")
    lines.append("Phase B: update tests to use namespaced imports")
    lines.append("- Replace imports like `from fetch_pdg_particle_data import ...` with")
    lines.append("  `from scripts.data.fetch_pdg_particle_data import ...`.")
    lines.append("- Update any script-to-script imports that break under module import context.")
    lines.append("")
    lines.append("Phase C: remove pytest.ini script pythonpath injections")
    lines.append("- Remove `src/scripts/*` entries from pytest.ini pythonpath.")
    lines.append("- Keep only `src` (src-layout) and rely on `pip install -e .` for packages.")
    lines.append("")
    lines.append("Acceptance criteria")
    lines.append(
        "- `PYTHONWARNINGS=error make check` passes with no pytest.ini script "
        "pythonpath injections."
    )
    lines.append("- No tests import scripts as top-level modules.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/pythonpath_refactor_plan.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root()
    pythonpath_entries = _read_pytest_pythonpath(repo_root)
    categories = sorted(
        {Path(e).parts[-1] for e in pythonpath_entries if e.startswith("src/scripts/")}
    )
    script_modules = _index_script_modules(repo_root, categories=categories)
    imported_modules = _scan_test_imports(repo_root)

    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _render(
            repo_root=repo_root,
            pythonpath_entries=pythonpath_entries,
            script_modules=script_modules,
            imported_modules=imported_modules,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
