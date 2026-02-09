#!/usr/bin/env python3
"""
Normalize operational tracker and requirements narrative markdown into TOML overlays.

Inputs:
- docs/ROADMAP.md
- docs/TODO.md
- docs/NEXT_ACTIONS.md
- REQUIREMENTS.md
- docs/REQUIREMENTS.md
- docs/requirements/*.md

Outputs:
- registry/roadmap_narrative.toml
- registry/todo_narrative.toml
- registry/next_actions_narrative.toml
- registry/requirements_narrative.toml

These TOML overlays preserve long-form narrative markdown while keeping TOML as
source of truth for machine-readable state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

UPDATED_STAMP = "2026-02-09"


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _escape_toml(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _to_toml_multiline(content: str) -> str:
    out: list[str] = []
    for ch in content:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\n":
            out.append("\n")
        elif code < 32:
            out.append(f"\\u{code:04X}")
        else:
            out.append(ch)
    return '"""\n' + "".join(out) + '\n"""'


def _title_from_markdown(path: Path, body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _render_single_overlay(section: str, source_markdown: str, body: str) -> str:
    lines: list[str] = []
    lines.append(f"# {section.replace('_', ' ').title()} narrative overlay registry (TOML-first).")
    lines.append("")
    lines.append(f"[{section}]")
    lines.append("authoritative = true")
    lines.append(f"updated = {_escape_toml(UPDATED_STAMP)}")
    lines.append(f"source_markdown = {_escape_toml(source_markdown)}")
    lines.append(f"body_markdown = {_to_toml_multiline(body.strip())}")
    lines.append("")
    return "\n".join(lines)


def _render_requirements_overlay(documents: list[tuple[str, str, str]]) -> str:
    lines: list[str] = []
    lines.append("# Requirements narrative overlay registry (TOML-first).")
    lines.append("")
    lines.append("[requirements_narrative]")
    lines.append("authoritative = true")
    lines.append(f"updated = {_escape_toml(UPDATED_STAMP)}")
    lines.append('source_markdown_glob = "docs/requirements/*.md"')
    lines.append(f"document_count = {len(documents)}")
    lines.append("")

    for path, title, body in documents:
        lines.append("[[document]]")
        lines.append(f"path = {_escape_toml(path)}")
        lines.append(f"title = {_escape_toml(title)}")
        lines.append(f"body_markdown = {_to_toml_multiline(body.strip())}")
        lines.append("")

    return "\n".join(lines)


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    roadmap_src = repo_root / "docs/ROADMAP.md"
    todo_src = repo_root / "docs/TODO.md"
    next_actions_src = repo_root / "docs/NEXT_ACTIONS.md"

    roadmap_out = repo_root / "registry/roadmap_narrative.toml"
    todo_out = repo_root / "registry/todo_narrative.toml"
    next_actions_out = repo_root / "registry/next_actions_narrative.toml"
    requirements_out = repo_root / "registry/requirements_narrative.toml"

    roadmap_toml = _render_single_overlay(
        "roadmap_narrative", "docs/ROADMAP.md", _read_file(roadmap_src)
    )
    todo_toml = _render_single_overlay("todo_narrative", "docs/TODO.md", _read_file(todo_src))
    next_actions_toml = _render_single_overlay(
        "next_actions_narrative", "docs/NEXT_ACTIONS.md", _read_file(next_actions_src)
    )

    requirement_files: list[Path] = [
        repo_root / "REQUIREMENTS.md",
        repo_root / "docs/REQUIREMENTS.md",
        *sorted((repo_root / "docs/requirements").glob("*.md")),
    ]
    requirement_docs: list[tuple[str, str, str]] = []
    for file_path in requirement_files:
        body = _read_file(file_path)
        rel = file_path.relative_to(repo_root).as_posix()
        requirement_docs.append((rel, _title_from_markdown(file_path, body), body))

    requirements_toml = _render_requirements_overlay(requirement_docs)

    for out_path, content in [
        (roadmap_out, roadmap_toml),
        (todo_out, todo_toml),
        (next_actions_out, next_actions_toml),
        (requirements_out, requirements_toml),
    ]:
        _assert_ascii(content, str(out_path))
        out_path.write_text(content, encoding="utf-8")

    print(
        "Normalized operational narrative overlays: "
        "registry/roadmap_narrative.toml, "
        "registry/todo_narrative.toml, "
        "registry/next_actions_narrative.toml, "
        "registry/requirements_narrative.toml."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
