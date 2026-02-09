#!/usr/bin/env python3
"""
Normalize narrative markdown overlays into TOML registries.

Inputs:
- docs/INSIGHTS.md
- docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md

Outputs:
- registry/insights_narrative.toml
- registry/experiments_narrative.toml

These TOML files capture long-form narrative sections so markdown overlays can be
regenerated from TOML without losing detail.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

INSIGHT_HEADING_RE = re.compile(r"^##\s+(I-\d{3})\s*:\s*(.+?)\s*$")
EXPERIMENT_HEADING_RE = re.compile(r"^##\s+(E-\d{3})\s*:\s*(.+?)\s*$")
HEADER_PREFIXES = (
    "<!-- AUTO-GENERATED:",
    "<!-- Source of truth:",
)


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


def _parse_sections(
    source_path: Path, heading_re: re.Pattern[str]
) -> tuple[str, list[tuple[str, str, str]]]:
    raw_lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = _strip_generated_header(raw_lines)
    preamble: list[str] = []
    entries: list[tuple[str, str, list[str]]] = []

    current_id: str | None = None
    current_title: str | None = None
    current_body: list[str] = []

    seen_first = False
    for line in lines:
        m = heading_re.match(line)
        if m:
            if current_id is not None and current_title is not None:
                entries.append((current_id, current_title, current_body))
            current_id = m.group(1)
            current_title = m.group(2).strip()
            current_body = []
            seen_first = True
            continue

        if not seen_first:
            preamble.append(line)
        else:
            current_body.append(line)

    if current_id is not None and current_title is not None:
        entries.append((current_id, current_title, current_body))

    return "\n".join(preamble).strip(), [
        (entry_id, title, _clean_entry_body("\n".join(body)))
        for entry_id, title, body in entries
    ]


def _strip_generated_header(lines: list[str]) -> list[str]:
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if any(stripped.startswith(prefix) for prefix in HEADER_PREFIXES):
            i += 1
            continue
        break
    return lines[i:]


def _clean_entry_body(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and lines[0].strip() == "---":
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    out: list[str] = []
    for line in lines:
        if line.strip() == "---":
            continue
        out.append(line)

    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out).strip()


def _render_insights_toml(
    source_path: str, preamble: str, entries: list[tuple[str, str, str]]
) -> str:
    lines: list[str] = []
    lines.append("# Insights narrative overlay registry (TOML-first).")
    lines.append("# Captures long-form narrative previously maintained in docs/INSIGHTS.md.")
    lines.append("")
    lines.append("[insights_narrative]")
    lines.append("authoritative = true")
    lines.append("updated = \"2026-02-09\"")
    lines.append(f"source_markdown = {_escape_toml(source_path)}")
    lines.append(f"entry_count = {len(entries)}")
    lines.append(f"preamble_markdown = {_to_toml_multiline(preamble)}")
    lines.append("")

    for entry_id, title, body in entries:
        lines.append("[[entry]]")
        lines.append(f"id = {_escape_toml(entry_id)}")
        lines.append(f"title = {_escape_toml(title)}")
        lines.append(f"body_markdown = {_to_toml_multiline(body)}")
        lines.append("")

    return "\n".join(lines)


def _render_experiments_toml(
    source_path: str, preamble: str, entries: list[tuple[str, str, str]]
) -> str:
    lines: list[str] = []
    lines.append("# Experiments narrative overlay registry (TOML-first).")
    lines.append(
        "# Captures long-form narrative previously maintained in "
        "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md."
    )
    lines.append("")
    lines.append("[experiments_narrative]")
    lines.append("authoritative = true")
    lines.append("updated = \"2026-02-09\"")
    lines.append(f"source_markdown = {_escape_toml(source_path)}")
    lines.append(f"entry_count = {len(entries)}")
    lines.append(f"preamble_markdown = {_to_toml_multiline(preamble)}")
    lines.append("")

    for entry_id, title, body in entries:
        lines.append("[[entry]]")
        lines.append(f"id = {_escape_toml(entry_id)}")
        lines.append(f"title = {_escape_toml(title)}")
        lines.append(f"body_markdown = {_to_toml_multiline(body)}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    insights_src = repo_root / "docs/INSIGHTS.md"
    experiments_src = repo_root / "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md"
    insights_out = repo_root / "registry/insights_narrative.toml"
    experiments_out = repo_root / "registry/experiments_narrative.toml"

    insights_preamble, insight_entries = _parse_sections(insights_src, INSIGHT_HEADING_RE)
    experiments_preamble, experiment_entries = _parse_sections(
        experiments_src, EXPERIMENT_HEADING_RE
    )

    insights_toml = _render_insights_toml(
        "docs/INSIGHTS.md", insights_preamble, insight_entries
    )
    experiments_toml = _render_experiments_toml(
        "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md",
        experiments_preamble,
        experiment_entries,
    )

    _assert_ascii(insights_toml, str(insights_out))
    _assert_ascii(experiments_toml, str(experiments_out))

    insights_out.write_text(insights_toml, encoding="utf-8")
    experiments_out.write_text(experiments_toml, encoding="utf-8")

    print(
        "Normalized narrative overlays: "
        f"{insights_out.relative_to(repo_root)} ({len(insight_entries)} entries), "
        f"{experiments_out.relative_to(repo_root)} ({len(experiment_entries)} entries)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
