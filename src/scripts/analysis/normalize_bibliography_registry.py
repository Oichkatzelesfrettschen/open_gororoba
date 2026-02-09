#!/usr/bin/env python3
"""
Bootstrap a TOML bibliography registry from docs/BIBLIOGRAPHY.md.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

H2_RE = re.compile(r"^##\s+(.+?)\s*$")
H3_RE = re.compile(r"^###\s+(.+?)\s*$")
URL_RE = re.compile(r"\((https?://[^)\s]+)\)")
DOI_RE = re.compile(r"\bDOI:\s*([0-9A-Za-z./()_-]+)")


@dataclass(frozen=True)
class Entry:
    entry_id: str
    order_index: int
    group: str
    section: str
    citation_markdown: str
    notes: list[str]
    urls: list[str]
    dois: list[str]
    source_line: int


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


def _render_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_escape_toml(v) for v in values) + "]"


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_entry_start(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("*") and "**" in stripped


def _extract_urls_and_dois(text: str, notes: list[str]) -> tuple[list[str], list[str]]:
    corpus = " ".join([text, *notes])
    urls = sorted(dict.fromkeys(URL_RE.findall(corpus)))
    dois = sorted(dict.fromkeys(DOI_RE.findall(corpus)))
    return urls, dois


def parse_bibliography(path: Path) -> tuple[list[Entry], list[str], list[str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    group = ""
    section = ""
    groups: list[str] = []
    sections: list[str] = []
    entries: list[Entry] = []
    order_index = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        h2 = H2_RE.match(line)
        if h2:
            group = h2.group(1).strip()
            i += 1
            continue
        h3 = H3_RE.match(line)
        if h3:
            section = h3.group(1).strip()
            i += 1
            continue
        if not _is_entry_start(line):
            i += 1
            continue

        if not group:
            group = "Primary Research and Data Sources"
        if not section:
            section = "Unscoped"
        if group not in groups:
            groups.append(group)
        if section not in sections:
            sections.append(section)

        order_index += 1
        citation_chunks: list[str] = [line.lstrip().lstrip("*").strip()]
        notes: list[str] = []
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if H2_RE.match(nxt) or H3_RE.match(nxt) or _is_entry_start(nxt):
                break
            stripped = nxt.strip()
            if stripped.startswith("*"):
                note = stripped.lstrip("*").strip()
                if note:
                    notes.append(note)
            elif stripped:
                citation_chunks.append(stripped)
            j += 1

        citation = _normalize_space(" ".join(citation_chunks))
        urls, dois = _extract_urls_and_dois(citation, notes)
        entries.append(
            Entry(
                entry_id=f"BIB-{order_index:04d}",
                order_index=order_index,
                group=group,
                section=section,
                citation_markdown=citation,
                notes=notes,
                urls=urls,
                dois=dois,
                source_line=i + 1,
            )
        )
        i = j
    return entries, groups, sections


def render_registry(entries: list[Entry], groups: list[str], sections: list[str], now: str) -> str:
    lines: list[str] = []
    lines.append("# Bibliography registry (TOML-first).")
    lines.append("# Bootstrap source markdown is docs/BIBLIOGRAPHY.md.")
    lines.append("")
    lines.append("[bibliography]")
    lines.append("authoritative = true")
    lines.append(f"updated = {_escape_toml(now)}")
    lines.append('source_markdown = "docs/BIBLIOGRAPHY.md"')
    lines.append(f"group_count = {len(groups)}")
    lines.append(f"section_count = {len(sections)}")
    lines.append(f"entry_count = {len(entries)}")
    lines.append("")

    for idx, name in enumerate(groups, start=1):
        entry_count = sum(1 for entry in entries if entry.group == name)
        section_count = len({entry.section for entry in entries if entry.group == name})
        lines.append("[[group]]")
        lines.append(f"id = {_escape_toml(f'BGR-{idx:03d}')}")
        lines.append(f"name = {_escape_toml(name)}")
        lines.append(f"section_count = {section_count}")
        lines.append(f"entry_count = {entry_count}")
        lines.append("")

    for idx, name in enumerate(sections, start=1):
        entry_count = sum(1 for entry in entries if entry.section == name)
        group_names = sorted({entry.group for entry in entries if entry.section == name})
        lines.append("[[section]]")
        lines.append(f"id = {_escape_toml(f'BSEC-{idx:03d}')}")
        lines.append(f"name = {_escape_toml(name)}")
        lines.append(f"groups = {_render_list(group_names)}")
        lines.append(f"entry_count = {entry_count}")
        lines.append("")

    for entry in entries:
        lines.append("[[entry]]")
        lines.append(f"id = {_escape_toml(entry.entry_id)}")
        lines.append(f"order_index = {entry.order_index}")
        lines.append(f"group = {_escape_toml(entry.group)}")
        lines.append(f"section = {_escape_toml(entry.section)}")
        lines.append(f"citation_markdown = {_escape_toml(entry.citation_markdown)}")
        lines.append(f"notes = {_render_list(entry.notes)}")
        lines.append(f"urls = {_render_list(entry.urls)}")
        lines.append(f"dois = {_render_list(entry.dois)}")
        lines.append(f"source_line = {entry.source_line}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--source",
        default="docs/BIBLIOGRAPHY.md",
        help="Source markdown path.",
    )
    parser.add_argument(
        "--out",
        default="registry/bibliography.toml",
        help="Output bibliography registry path.",
    )
    parser.add_argument(
        "--bootstrap-from-markdown",
        action="store_true",
        help=(
            "Explicitly allow bootstrap from markdown. "
            "Operational updates should happen in registry/bibliography.toml."
        ),
    )
    args = parser.parse_args()

    if not args.bootstrap_from_markdown:
        print(
            "SKIP: normalize_bibliography_registry.py is bootstrap-only. "
            "Use --bootstrap-from-markdown for explicit re-bootstrap."
        )
        return 0

    repo_root = Path(args.repo_root).resolve()
    source = repo_root / args.source
    out = repo_root / args.out
    entries, groups, sections = parse_bibliography(source)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rendered = render_registry(entries, groups, sections, now)
    _assert_ascii(rendered, str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out} from {source}; entries={len(entries)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
