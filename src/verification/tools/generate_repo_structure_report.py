#!/usr/bin/env python3
"""
Generate a lightweight, reproducible repo structure report.

This intentionally avoids external dependencies and produces ASCII-only markdown.
It is designed for "audit" visibility, not for CI gating.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_OMIT_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
}


@dataclass(frozen=True)
class DirStats:
    file_count: int
    dir_count: int
    total_bytes: int


def _iter_tree(root: Path, omit_dirs: set[str]) -> tuple[int, int, int]:
    file_count = 0
    dir_count = 0
    total_bytes = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames if d not in omit_dirs and not d.startswith(".tox")
        ]
        dir_count += len(dirnames)
        for fn in filenames:
            if fn.startswith("."):
                continue
            file_count += 1
            try:
                total_bytes += (dirpath_p / fn).stat().st_size
            except FileNotFoundError:
                # Race with generators is fine; this is an audit view.
                continue

    return file_count, dir_count, total_bytes


def _human_bytes(n: int) -> str:
    # Simple, stable formatting (no locale).
    for unit, denom in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]:
        if n >= denom:
            return f"{n / denom:.2f} {unit}"
    return f"{n} B"


def _top_level_stats(repo_root: Path, omit_dirs: set[str]) -> dict[str, DirStats]:
    out: dict[str, DirStats] = {}
    for child in sorted(repo_root.iterdir(), key=lambda p: p.name):
        if child.name in omit_dirs:
            continue
        if not child.is_dir():
            continue
        fc, dc, tb = _iter_tree(child, omit_dirs=omit_dirs)
        out[child.name] = DirStats(file_count=fc, dir_count=dc, total_bytes=tb)
    return out


def _has(p: Path) -> str:
    return "yes" if p.exists() else "no"


def _render_md(repo_root: Path, stats: dict[str, DirStats]) -> str:
    today = _dt.date.today().isoformat()
    artifacts_manifest = repo_root / "data" / "artifacts" / "ARTIFACTS_MANIFEST.csv"

    lines: list[str] = []
    lines.append(f"# Repo Structure Audit ({today})")
    lines.append("")
    lines.append(f"Scope: `{repo_root}`")
    lines.append("")
    lines.append("## Top-level directories")
    lines.append("")
    for name, s in stats.items():
        lines.append(
            f"- `{name}/`: {s.file_count} files, {s.dir_count} dirs, ~{_human_bytes(s.total_bytes)}"
        )

    lines.append("")
    lines.append("## Structure signals (quick checks)")
    lines.append("")
    lines.append(f"- `src/` exists: {_has(repo_root / 'src')}")
    lines.append(f"- `tests/` exists: {_has(repo_root / 'tests')}")
    lines.append(f"- `docs/` exists: {_has(repo_root / 'docs')}")
    lines.append(f"- `data/` exists: {_has(repo_root / 'data')}")
    lines.append(f"- `data/artifacts/` exists: {_has(repo_root / 'data' / 'artifacts')}")
    lines.append(f"- `data/artifacts/ARTIFACTS_MANIFEST.csv` exists: {_has(artifacts_manifest)}")
    lines.append(f"- `src/scripts/` exists: {_has(repo_root / 'src' / 'scripts')}")
    lines.append(
        f"- `src/verification/` exists: {_has(repo_root / 'src' / 'verification')}"
    )
    lines.append(
        f"- `src/gemini_physics/` exists: {_has(repo_root / 'src' / 'gemini_physics')}"
    )

    lines.append("")
    lines.append("## Notes: what is `gemini_physics`?")
    lines.append("")
    lines.append(
        "- `src/gemini_physics/` is the primary Python package for domain code (algebra, cosmology,"
    )
    lines.append(
        "  materials, etc.). It exists to keep reusable library code separate from runnable"
    )
    lines.append(
        "  scripts."
    )
    lines.append(
        "- `src/scripts/**` holds entrypoints and one-off pipelines; scripts import from"
    )
    lines.append("  `gemini_physics` (and other packages) rather than duplicating logic.")
    lines.append(
        "- Keeping domain code inside a package (instead of flat files under `src/`) improves"
    )
    lines.append(
        "  testability, import hygiene, and reproducibility (e.g., network gating and determinism"
    )
    lines.append("  policies can be enforced centrally).")

    lines.append("")
    lines.append("## Notes: current `src/` layout (high level)")
    lines.append("")
    lines.append("- `src/gemini_physics/`: primary package for research code.")
    lines.append("- `src/scripts/`: runnable scripts with contract headers")
    lines.append("  (inputs/outputs/network).")
    lines.append("- `src/verification/`: verifiers for repo contracts and reproducibility gates.")
    lines.append(
        "- Other packages under `src/` (for now) may represent legacy or domain-specific modules"
    )
    lines.append(
        "  that are candidates for consolidation into `gemini_physics/` once import paths and tests"
    )
    lines.append("  updated safely (phase-gated refactor).")

    lines.append("")
    lines.append("## Suggested repo type classification")
    lines.append("")
    lines.append(
        "- This is a reproducible research monorepo: code + tests + docs + datasets + generated"
    )
    lines.append(
        "  artifacts with strict provenance and offline-default behavior. Best practice is to keep"
    )
    lines.append("  explicit (artifact targets) and keep smoke/check verification-only.")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root path (defaults to git root via CWD heuristic).",
    )
    parser.add_argument(
        "--output",
        default="reports/repo_structure_audit.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    if args.repo_root is None:
        # Heuristic: assume script run from repo or subdir; walk up until .git found.
        here = Path.cwd().resolve()
        p = here
        while p != p.parent:
            if (p / ".git").exists():
                repo_root = p
                break
            p = p.parent
        else:
            repo_root = here
    else:
        repo_root = Path(args.repo_root).resolve()

    omit_dirs = set(_DEFAULT_OMIT_DIRS)
    stats = _top_level_stats(repo_root, omit_dirs=omit_dirs)
    content = _render_md(repo_root, stats=stats)

    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
