#!/usr/bin/env python3
"""
Export human-facing markdown mirrors from authoritative TOML registries.
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from io import StringIO
from pathlib import Path

CHECK_MODE = False
CHANGED_PATHS: list[str] = []


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    _assert_ascii(text, str(path))
    if CHECK_MODE:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if existing != text:
            CHANGED_PATHS.append(str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _header(title: str) -> list[str]:
    return [
        f"# {title}",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: TOML registry files under registry/ -->",
        "",
    ]


def _claim_sort_key(claim_id: str) -> int:
    if claim_id.startswith("C-") and claim_id[2:].isdigit():
        return int(claim_id[2:])
    return 999999


def _pipe_escape(text: str) -> str:
    return text.replace("|", "\\|")


def _render_csv(rows: list[list[str]]) -> str:
    buf = StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def _load_optional_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    return _load_toml(path)


def _ascii_sanitize(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    out: list[str] = []
    for ch in text:
        mapped = replacements.get(ch, ch)
        for item in mapped:
            code = ord(item)
            if item in {"\n", "\r", "\t"}:
                out.append(item)
            elif code < 32:
                out.append(" ")
            elif code <= 127:
                out.append(item)
            else:
                out.append(f"<U+{code:04X}>")
    return "".join(out)


def _narrative_map(data: dict) -> tuple[str, dict[str, str]]:
    section = data.get("insights_narrative") or data.get("experiments_narrative") or {}
    preamble = str(section.get("preamble_markdown", "")).strip()
    mapping: dict[str, str] = {}
    for row in data.get("entry", []):
        entry_id = str(row.get("id", "")).strip()
        body = str(row.get("body_markdown", "")).strip()
        if entry_id:
            mapping[entry_id] = body
    return preamble, mapping


def _single_overlay_body(data: dict, section_key: str) -> str:
    section = data.get(section_key, {})
    return str(section.get("body_markdown", "")).strip()


def _legacy_header(sources: list[str]) -> list[str]:
    return [
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        f"<!-- Source of truth: {', '.join(sources)} -->",
        "",
    ]


def _legacy_lines_from_body(body: str, fallback_title: str, fallback_lines: list[str]) -> list[str]:
    lines: list[str] = []
    if body:
        lines.extend(body.splitlines())
    else:
        lines.append(f"# {fallback_title}")
        lines.append("")
        lines.extend(fallback_lines)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def export_insights(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/insights.toml")
    insights = sorted(data.get("insight", []), key=lambda x: x.get("id", ""))
    lines = _header("Insights Registry Mirror")
    lines.append("Authoritative source: `registry/insights.toml`.")
    lines.append("")
    lines.append(f"Total insights: {len(insights)}")
    lines.append("")
    for row in insights:
        claims = ", ".join(row.get("claims", [])) if row.get("claims") else "(none)"
        lines.append(f"## {row.get('id', 'I-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Date: {row.get('date', '')}")
        lines.append(f"- Status: {row.get('status', '')}")
        lines.append(f"- Sprint: {row.get('sprint', '')}")
        lines.append(f"- Claims: {claims}")
        lines.append("")
        lines.append(row.get("summary", "").strip())
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_claims(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/claims.toml")
    claims = sorted(
        data.get("claim", []), key=lambda row: _claim_sort_key(str(row.get("id", "")))
    )
    lines = _header("Claims Registry Mirror")
    lines.append("Authoritative source: `registry/claims.toml`.")
    lines.append("")
    lines.append(f"Total claims: {len(claims)}")
    lines.append("")
    for row in claims:
        claim_id = str(row.get("id", "C-???"))
        statement = str(row.get("statement", "")).strip()
        status = str(row.get("status", "")).strip()
        last_verified = str(row.get("last_verified", "")).strip()
        where_stated = str(row.get("where_stated", "")).strip()
        verify_refute = str(row.get("what_would_verify_refute", "")).strip()
        lines.append(f"## {claim_id}")
        lines.append("")
        lines.append(f"- Status: `{status}`")
        lines.append(f"- Last verified: {last_verified}")
        lines.append(f"- Statement: {statement}")
        lines.append(f"- Where stated: {where_stated}")
        lines.append(f"- What would verify/refute it: {verify_refute}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_bibliography(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/bibliography.toml")
    meta = data.get("bibliography", {})
    groups = data.get("group", [])
    entries = data.get("entry", [])
    lines = _header("Bibliography Registry Mirror")
    lines.append("Authoritative source: `registry/bibliography.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown: `{meta.get('source_markdown', '')}`")
    lines.append(f"- Group count: {meta.get('group_count', len(groups))}")
    lines.append(f"- Entry count: {meta.get('entry_count', len(entries))}")
    lines.append("")

    for group in groups:
        group_name = str(group.get("name", "(group)"))
        lines.append(f"## {group_name}")
        lines.append("")
        group_entries = [row for row in entries if str(row.get("group", "")) == group_name]
        for row in sorted(group_entries, key=lambda item: int(item.get("order_index", 0))):
            lines.append(f"- {row.get('citation_markdown', '')}")
            for note in row.get("notes", []):
                lines.append(f"  - {note}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_bibliography_legacy(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/bibliography.toml")
    groups = data.get("group", [])
    entries = data.get("entry", [])
    lines: list[str] = [
        "# Unified Bibliography",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: registry/bibliography.toml -->",
        "",
        "This file is generated from `registry/bibliography.toml`.",
        "",
    ]
    for group in groups:
        group_name = str(group.get("name", "")).strip()
        lines.append(f"## {group_name}")
        lines.append("")
        section_names = []
        for row in entries:
            if str(row.get("group", "")) != group_name:
                continue
            section_name = str(row.get("section", "Unscoped")).strip()
            if section_name not in section_names:
                section_names.append(section_name)
        for section_name in section_names:
            lines.append(f"### {section_name}")
            section_entries = [
                row
                for row in entries
                if str(row.get("group", "")) == group_name
                and str(row.get("section", "Unscoped")).strip() == section_name
            ]
            for row in sorted(section_entries, key=lambda item: int(item.get("order_index", 0))):
                lines.append(f"*   {row.get('citation_markdown', '')}")
                for note in row.get("notes", []):
                    lines.append(f"    *   {note}")
            lines.append("")
    _write(out_path, "\n".join(lines))


def export_claims_matrix_legacy(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/claims.toml")
    claims = sorted(
        data.get("claim", []), key=lambda row: _claim_sort_key(str(row.get("id", "")))
    )
    lines: list[str] = [
        "# Claims / Evidence Matrix (Markdown Mirror)",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: registry/claims.toml -->",
        "",
        "This file is generated from `registry/claims.toml`.",
        "",
        "| ID | Claim | Where stated | Status | Last verified | What would verify/refute it |",
        "|---:|---|---|---|---|---|",
    ]
    for row in claims:
        claim_id = str(row.get("id", "C-???")).strip()
        statement = _pipe_escape(str(row.get("statement", "")).strip())
        where_stated = _pipe_escape(str(row.get("where_stated", "")).strip())
        status = _pipe_escape(str(row.get("status", "")).strip())
        last_verified = _pipe_escape(str(row.get("last_verified", "")).strip())
        verify_refute = _pipe_escape(str(row.get("what_would_verify_refute", "")).strip())
        lines.append(
            f"| {claim_id} | {statement} | {where_stated} | **{status}** | "
            f"{last_verified} | {verify_refute} |"
        )
    lines.append("")
    _write(out_path, "\n".join(lines))


def export_insights_legacy(repo_root: Path, out_path: Path) -> None:
    registry_data = _load_toml(repo_root / "registry/insights.toml")
    narrative_data = _load_optional_toml(repo_root / "registry/insights_narrative.toml")
    preamble, body_by_id = _narrative_map(narrative_data)
    insights = sorted(registry_data.get("insight", []), key=lambda x: x.get("id", ""))

    lines: list[str] = _legacy_header(
        ["registry/insights.toml", "registry/insights_narrative.toml"]
    )
    if preamble:
        lines.extend(preamble.splitlines())
    else:
        lines.extend(
            [
                "# Insights",
                "",
                "Source-of-truth policy:",
                "- Authoritative machine-readable registry: `registry/insights.toml`",
                "- Narrative overlay registry: `registry/insights_narrative.toml`",
                "- TOML-driven markdown mirror: `docs/generated/INSIGHTS_REGISTRY_MIRROR.md`",
                "- This file is generated from TOML sources.",
                "",
            ]
        )
    lines.append("")

    for row in insights:
        insight_id = str(row.get("id", "I-???"))
        title = str(row.get("title", "(untitled)"))
        lines.append(f"## {insight_id}: {title}")
        lines.append("")
        body = body_by_id.get(insight_id, "").strip()
        if body:
            lines.extend(body.splitlines())
        else:
            claims = ", ".join(row.get("claims", [])) if row.get("claims") else "(none)"
            lines.append(f"Date: {row.get('date', '')}")
            lines.append(f"Status: {row.get('status', '')}")
            lines.append(f"Claims: {claims}")
            lines.append("")
            lines.append(str(row.get("summary", "")).strip())
        lines.append("")
        lines.append("---")
        lines.append("")

    while lines and not lines[-1].strip():
        lines.pop()
    _write(out_path, "\n".join(lines) + "\n")


def export_experiments(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/experiments.toml")
    experiments = sorted(data.get("experiment", []), key=lambda x: x.get("id", ""))
    lines = _header("Experiments Registry Mirror")
    lines.append("Authoritative source: `registry/experiments.toml`.")
    lines.append("")
    lines.append(f"Total experiments: {len(experiments)}")
    lines.append("")
    for row in experiments:
        claims = ", ".join(row.get("claims", [])) if row.get("claims") else "(none)"
        outputs = ", ".join(row.get("output", [])) if row.get("output") else "(none)"
        lines.append(f"## {row.get('id', 'E-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Binary: `{row.get('binary', '')}`")
        lines.append(f"- Input: {row.get('input', '')}")
        lines.append(f"- Output: {outputs}")
        lines.append(f"- Deterministic: `{row.get('deterministic', False)}`")
        if "seed" in row:
            lines.append(f"- Seed: `{row.get('seed')}`")
        lines.append(f"- GPU: `{row.get('gpu', False)}`")
        lines.append(f"- Claims: {claims}")
        lines.append("")
        lines.append("Method:")
        lines.append(row.get("method", "").strip())
        lines.append("")
        lines.append("Run command:")
        lines.append("```bash")
        lines.append(row.get("run", "").strip())
        lines.append("```")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_experiments_legacy(repo_root: Path, out_path: Path) -> None:
    registry_data = _load_toml(repo_root / "registry/experiments.toml")
    narrative_data = _load_optional_toml(repo_root / "registry/experiments_narrative.toml")
    preamble, body_by_id = _narrative_map(narrative_data)
    experiments = sorted(registry_data.get("experiment", []), key=lambda x: x.get("id", ""))

    lines: list[str] = _legacy_header(
        ["registry/experiments.toml", "registry/experiments_narrative.toml"]
    )
    if preamble:
        lines.extend(preamble.splitlines())
    else:
        lines.extend(
            [
                "# Experiments Portfolio Shortlist",
                "",
                "Source-of-truth policy:",
                "- Authoritative machine-readable registry: `registry/experiments.toml`",
                "- Narrative overlay registry: `registry/experiments_narrative.toml`",
                "- TOML-driven markdown mirror: `docs/generated/EXPERIMENTS_REGISTRY_MIRROR.md`",
                "- This file is generated from TOML sources.",
                "",
            ]
        )
    lines.append("")

    for row in experiments:
        experiment_id = str(row.get("id", "E-???"))
        title = str(row.get("title", "(untitled)"))
        lines.append(f"## {experiment_id}: {title}")
        lines.append("")
        body = body_by_id.get(experiment_id, "").strip()
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append(f"Method: {row.get('method', '')}")
            lines.append(f"Input: {row.get('input', '')}")
            outputs = ", ".join(row.get("output", [])) if row.get("output") else "(none)"
            lines.append(f"Output: {outputs}")
            lines.append("Run:")
            lines.append("```bash")
            lines.append(str(row.get("run", "")).strip())
            lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    while lines and not lines[-1].strip():
        lines.pop()
    _write(out_path, "\n".join(lines) + "\n")


def export_roadmap(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/roadmap.toml")
    roadmap = data.get("roadmap", {})
    workstreams = data.get("workstream", [])
    lines = _header("Roadmap Registry Mirror")
    lines.append("Authoritative source: `registry/roadmap.toml`.")
    lines.append("")
    lines.append(f"- Consolidated date: {roadmap.get('consolidated_date', '')}")
    lines.append(f"- Source markdown: `{roadmap.get('source_markdown', '')}`")
    lines.append(f"- Status: `{roadmap.get('status', '')}`")
    lines.append("")
    lines.append("## Companion Docs")
    lines.append("")
    for item in roadmap.get("companion_docs", []):
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## Workstreams")
    lines.append("")
    for ws in workstreams:
        lines.append(f"### {ws.get('id', 'WS-???')}: {ws.get('name', '(unnamed)')}")
        lines.append("")
        lines.append(f"- Priority: `{ws.get('priority', '')}`")
        lines.append(f"- Status: `{ws.get('status', '')}`")
        lines.append(f"- Description: {ws.get('description', '')}")
        lines.append("- Primary outputs:")
        for out in ws.get("primary_outputs", []):
            lines.append(f"  - `{out}`")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_roadmap_legacy(repo_root: Path, out_path: Path) -> None:
    narrative = _load_optional_toml(repo_root / "registry/roadmap_narrative.toml")
    body = _single_overlay_body(narrative, "roadmap_narrative")
    fallback = [
        (
            "This file is generated from `registry/roadmap.toml` "
            "and `registry/roadmap_narrative.toml`."
        ),
        "",
        "See the structured mirror at `docs/generated/ROADMAP_REGISTRY_MIRROR.md`.",
        "",
    ]
    lines = _legacy_header(["registry/roadmap.toml", "registry/roadmap_narrative.toml"])
    lines.extend(_legacy_lines_from_body(body, "ROADMAP", fallback))
    _write(out_path, "\n".join(lines) + "\n")


def export_todo(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/todo.toml")
    todo = data.get("todo", {})
    sprint = data.get("sprint", {})
    tasks = data.get("task", [])
    lines = _header("TODO Registry Mirror")
    lines.append("Authoritative source: `registry/todo.toml`.")
    lines.append("")
    lines.append(f"- Updated: {todo.get('updated', '')}")
    lines.append(f"- Sprint: {sprint.get('id', '')} ({sprint.get('name', '')})")
    lines.append(f"- Sprint status: `{sprint.get('status', '')}`")
    lines.append("")
    lines.append("## Tasks")
    lines.append("")
    for task in tasks:
        lines.append(f"### {task.get('id', 'TASK-???')}: {task.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Status: `{task.get('status', '')}`")
        lines.append("- Evidence:")
        for ev in task.get("evidence", []):
            lines.append(f"  - `{ev}`")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_todo_legacy(repo_root: Path, out_path: Path) -> None:
    narrative = _load_optional_toml(repo_root / "registry/todo_narrative.toml")
    body = _single_overlay_body(narrative, "todo_narrative")
    fallback = [
        "This file is generated from `registry/todo.toml` and `registry/todo_narrative.toml`.",
        "",
        "See the structured mirror at `docs/generated/TODO_REGISTRY_MIRROR.md`.",
        "",
    ]
    lines = _legacy_header(["registry/todo.toml", "registry/todo_narrative.toml"])
    lines.extend(_legacy_lines_from_body(body, "TODO", fallback))
    _write(out_path, "\n".join(lines) + "\n")


def export_next_actions(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/next_actions.toml")
    meta = data.get("next_actions", {})
    actions = data.get("action", [])
    lines = _header("Next Actions Registry Mirror")
    lines.append("Authoritative source: `registry/next_actions.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Status: `{meta.get('status', '')}`")
    lines.append("")
    lines.append("## Priority Queue")
    lines.append("")
    for action in actions:
        lines.append(
            f"### {action.get('id', 'NA-???')} ({action.get('priority', 'P?')}): "
            f"{action.get('title', '(untitled)')}"
        )
        lines.append("")
        lines.append(f"- Status: `{action.get('status', '')}`")
        lines.append(f"- Description: {action.get('description', '')}")
        lines.append("- References:")
        for ref in action.get("references", []):
            lines.append(f"  - `{ref}`")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_next_actions_legacy(repo_root: Path, out_path: Path) -> None:
    narrative = _load_optional_toml(repo_root / "registry/next_actions_narrative.toml")
    body = _single_overlay_body(narrative, "next_actions_narrative")
    fallback = [
        (
            "This file is generated from `registry/next_actions.toml` "
            "and `registry/next_actions_narrative.toml`."
        ),
        "",
        "See the structured mirror at `docs/generated/NEXT_ACTIONS_REGISTRY_MIRROR.md`.",
        "",
    ]
    lines = _legacy_header(["registry/next_actions.toml", "registry/next_actions_narrative.toml"])
    lines.extend(_legacy_lines_from_body(body, "NEXT ACTIONS", fallback))
    _write(out_path, "\n".join(lines) + "\n")


def export_requirements(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/requirements.toml")
    req = data.get("requirements", {})
    modules = data.get("module", [])
    gaps = data.get("coverage_gap", [])
    lines = _header("Requirements Registry Mirror")
    lines.append("Authoritative source: `registry/requirements.toml`.")
    lines.append("")
    lines.append(f"- Updated: {req.get('updated', '')}")
    lines.append(f"- Python recommended: `{req.get('python_recommended', '')}`")
    lines.append(f"- Python allowed: `{req.get('python_allowed', '')}`")
    lines.append(f"- Primary markdown: `{req.get('primary_markdown', '')}`")
    lines.append("")
    lines.append("## Modules")
    lines.append("")
    for module in modules:
        lines.append(f"### {module.get('id', 'REQ-???')}: {module.get('name', '(unnamed)')}")
        lines.append("")
        lines.append(f"- Status: `{module.get('status', '')}`")
        lines.append(f"- Markdown: `{module.get('markdown', '')}`")
        targets = module.get("install_targets", [])
        if targets:
            lines.append("- Install targets:")
            for target in targets:
                lines.append(f"  - `{target}`")
        lines.append("")
    lines.append("## Coverage Gaps")
    lines.append("")
    for gap in gaps:
        lines.append(f"### {gap.get('id', 'REQ-GAP-???')}: {gap.get('area', '(area)')}")
        lines.append("")
        lines.append(f"- Status: `{gap.get('status', '')}`")
        lines.append(f"- Description: {gap.get('description', '')}")
        lines.append(f"- Proposed resolution: {gap.get('proposed_resolution', '')}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_knowledge_migration_plan(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/knowledge_migration_plan.toml")
    meta = data.get("migration", {})
    domains = data.get("domain", [])
    phases = data.get("phase", [])
    policies = data.get("policy", [])
    lines = _header("Knowledge Migration Plan Registry Mirror")
    lines.append("Authoritative source: `registry/knowledge_migration_plan.toml`.")
    lines.append("")
    lines.append(f"- Status: `{meta.get('status', '')}`")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Scope: {meta.get('scope', '')}")
    lines.append("")
    lines.append("## Domains")
    lines.append("")
    for domain in domains:
        lines.append(f"### {domain.get('id', 'KM-???')}: {domain.get('name', '(unnamed)')}")
        lines.append("")
        lines.append(f"- Strategy: `{domain.get('strategy', '')}`")
        lines.append(f"- Status: `{domain.get('status', '')}`")
        src = domain.get("source_markdown", [])
        if src:
            lines.append("- Source markdown:")
            for item in src:
                lines.append(f"  - `{item}`")
        auth = domain.get("authoritative_toml", [])
        if auth:
            lines.append("- Authoritative TOML:")
            for item in auth:
                lines.append(f"  - `{item}`")
        gen = domain.get("generated_mirror", [])
        if gen:
            lines.append("- Generated mirrors:")
            for item in gen:
                lines.append(f"  - `{item}`")
        notes = str(domain.get("notes", "")).strip()
        if notes:
            lines.append(f"- Notes: {notes}")
        lines.append("")
    lines.append("## Phases")
    lines.append("")
    for phase in phases:
        lines.append(f"### {phase.get('id', 'KMP-?')}: {phase.get('name', '(unnamed)')}")
        lines.append("")
        lines.append(f"- Status: `{phase.get('status', '')}`")
        deliverables = phase.get("deliverables", [])
        if deliverables:
            lines.append("- Deliverables:")
            for item in deliverables:
                lines.append(f"  - {item}")
        lines.append("")
    if policies:
        lines.append("## Policies")
        lines.append("")
        for policy in policies:
            lines.append(f"### {policy.get('id', 'KMPOL-?')}: {policy.get('name', '(unnamed)')}")
            lines.append("")
            lines.append(f"- Status: `{policy.get('status', '')}`")
            lines.append(f"- Statement: {policy.get('statement', '')}")
            enforcement = policy.get("enforcement", [])
            if enforcement:
                lines.append("- Enforcement:")
                for item in enforcement:
                    lines.append(f"  - {item}")
            lines.append("")
    _write(out_path, "\n".join(lines))


def export_markdown_governance(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/markdown_governance.toml")
    meta = data.get("markdown_governance", {})
    docs = data.get("document", [])
    lines = _header("Markdown Governance Registry Mirror")
    lines.append("Authoritative source: `registry/markdown_governance.toml`.")
    lines.append("")
    lines.append(f"- Generated at: {meta.get('generated_at', '')}")
    lines.append(f"- Document count: {meta.get('document_count', len(docs))}")
    lines.append(f"- TOML generated mirrors: {meta.get('toml_generated_mirror_count', 0)}")
    lines.append(f"- TOML manual sources: {meta.get('toml_manual_source_count', 0)}")
    lines.append(f"- Generated artifacts: {meta.get('generated_artifact_count', 0)}")
    lines.append(f"- Manual narratives: {meta.get('manual_narrative_count', 0)}")
    lines.append(f"- Immutable transcripts: {meta.get('immutable_transcript_count', 0)}")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    for row in sorted(docs, key=lambda item: item.get("path", "")):
        lines.append(f"### {row.get('id', 'MDG-????')}: `{row.get('path', '')}`")
        lines.append("")
        lines.append(f"- Kind: `{row.get('kind', '')}`")
        lines.append(f"- Mode: `{row.get('mode', '')}`")
        lines.append(f"- Header required: `{row.get('header_required', False)}`")
        refs = row.get("source_toml_refs", [])
        if refs:
            lines.append("- Source TOML refs:")
            for ref in refs:
                lines.append(f"  - `{ref}`")
        notes = str(row.get("notes", "")).strip()
        if notes:
            lines.append(f"- Notes: {notes}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_navigator(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/navigator.toml")
    meta = data.get("navigator", {})
    sections = data.get("section", [])
    lines = _header("Navigator Registry Mirror")
    lines.append("Authoritative source: `registry/navigator.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Epoch: {meta.get('epoch', '')}")
    lines.append(f"- Mission: {meta.get('mission', '')}")
    lines.append(f"- Section count: {len(sections)}")
    lines.append("")
    for section in sections:
        lines.append(f"## {section.get('id', 'NAV-?')}: {section.get('title', '(untitled)')}")
        lines.append("")
        summary = str(section.get("summary", "")).strip()
        if summary:
            lines.append(summary)
            lines.append("")
        for link in section.get("link", []):
            path = str(link.get("path", "")).strip()
            label = str(link.get("label", path)).strip()
            notes = str(link.get("notes", "")).strip()
            hypothesis = bool(link.get("hypothesis", False))
            lines.append(f"- `{label}` -> `{path}`")
            lines.append(f"  - Hypothesis narrative: `{hypothesis}`")
            if notes:
                lines.append(f"  - Notes: {notes}")
        lines.append("")
    disclaimer = data.get("navigator", {}).get("disclaimer", {})
    disclaimer_text = str(disclaimer.get("text", "")).strip()
    if disclaimer_text:
        lines.append("## Disclaimer")
        lines.append("")
        lines.append(disclaimer_text)
        claims_source = str(disclaimer.get("claims_source", "")).strip()
        legacy_claims_mirror = str(disclaimer.get("legacy_claims_mirror", "")).strip()
        if claims_source:
            lines.append(f"- Claims source: `{claims_source}`")
        if legacy_claims_mirror:
            lines.append(f"- Legacy claims mirror: `{legacy_claims_mirror}`")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_navigator_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/navigator.toml")
    meta = data.get("navigator", {})
    sections = data.get("section", [])
    lines = _legacy_header(["registry/navigator.toml"])
    title = str(meta.get("title", "Navigator")).strip()
    epoch = str(meta.get("epoch", "")).strip()
    mission = str(meta.get("mission", "")).strip()
    lines.append(f"# {title}")
    lines.append("")
    if epoch:
        lines.append(f"**Current Epoch:** {epoch}")
    if mission:
        lines.append(f"**Mission:** {mission}")
    if epoch or mission:
        lines.append("")
    lines.append(
        "**Important:** This file is generated from TOML registry state and is not authoritative."
    )
    lines.append("")
    for section in sections:
        lines.append(f"## {section.get('id', 'NAV-?')}: {section.get('title', '(untitled)')}")
        lines.append("")
        summary = str(section.get("summary", "")).strip()
        if summary:
            lines.append(summary)
            lines.append("")
        for link in section.get("link", []):
            path = str(link.get("path", "")).strip()
            label = str(link.get("label", path)).strip()
            notes = str(link.get("notes", "")).strip()
            lines.append(f"- **{label}:** `{path}`")
            if notes:
                lines.append(f"  - {notes}")
            if bool(link.get("hypothesis", False)):
                lines.append("  - Status: hypothesis narrative")
        lines.append("")
    disclaimer = data.get("navigator", {}).get("disclaimer", {})
    disclaimer_text = str(disclaimer.get("text", "")).strip()
    if disclaimer_text:
        lines.append("---")
        lines.append("")
        lines.append(disclaimer_text)
        claims_source = str(disclaimer.get("claims_source", "")).strip()
        if claims_source:
            lines.append(f"Claims source: `{claims_source}`")
    lines.append("")
    _write(repo_root / "NAVIGATOR.md", "\n".join(lines))


def export_entrypoint_docs(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/entrypoint_docs.toml")
    meta = data.get("entrypoint_docs", {})
    docs = data.get("document", [])
    lines = _header("Entrypoint Docs Registry Mirror")
    lines.append("Authoritative source: `registry/entrypoint_docs.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Document count: {meta.get('document_count', len(docs))}")
    lines.append("")
    for row in docs:
        lines.append(f"## `{row.get('path', '')}`")
        lines.append("")
        lines.append(f"- Title: {row.get('title', '')}")
        body = str(row.get("body_markdown", "")).strip()
        if body:
            lines.append(f"- Body lines: {len(body.splitlines())}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_entrypoint_docs_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/entrypoint_docs.toml")
    docs = data.get("document", [])
    for row in docs:
        path = str(row.get("path", "")).strip()
        if not path:
            continue
        body = str(row.get("body_markdown", "")).strip()
        title = str(row.get("title", "")).strip() or Path(path).stem
        lines = _legacy_header(["registry/entrypoint_docs.toml"])
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append(f"# {title}")
            lines.append("")
            lines.append("(No body_markdown captured in registry/entrypoint_docs.toml.)")
        lines.append("")
        _write(repo_root / path, "\n".join(lines))


def export_requirements_legacy(repo_root: Path) -> None:
    req_data = _load_toml(repo_root / "registry/requirements.toml")
    narrative = _load_optional_toml(repo_root / "registry/requirements_narrative.toml")
    narrative_rows = narrative.get("document", [])
    body_by_path = {
        str(row.get("path", "")).strip(): str(row.get("body_markdown", "")).strip()
        for row in narrative_rows
        if str(row.get("path", "")).strip()
    }
    title_by_path = {
        str(row.get("path", "")).strip(): str(row.get("title", "")).strip()
        for row in narrative_rows
        if str(row.get("path", "")).strip()
    }

    req_meta = req_data.get("requirements", {})
    primary_markdown = str(req_meta.get("primary_markdown", "docs/REQUIREMENTS.md"))
    target_paths = {"REQUIREMENTS.md", primary_markdown}
    for module in req_data.get("module", []):
        markdown_path = str(module.get("markdown", "")).strip()
        if markdown_path:
            target_paths.add(markdown_path)

    module_by_markdown = {
        str(module.get("markdown", "")).strip(): module for module in req_data.get("module", [])
    }

    sources = ["registry/requirements.toml", "registry/requirements_narrative.toml"]
    for rel_path in sorted(target_paths):
        body = body_by_path.get(rel_path, "")
        title = title_by_path.get(rel_path, Path(rel_path).stem.replace("-", " ").title())
        module = module_by_markdown.get(rel_path, {})
        fallback: list[str] = [
            f"This file is generated from `{sources[0]}` and `{sources[1]}`.",
            "",
            "See the structured mirror at `docs/generated/REQUIREMENTS_REGISTRY_MIRROR.md`.",
            "",
        ]
        if module:
            fallback.append(f"- Module ID: `{module.get('id', '')}`")
            fallback.append(f"- Module name: `{module.get('name', '')}`")
            fallback.append(f"- Status: `{module.get('status', '')}`")
            targets = module.get("install_targets", [])
            if targets:
                fallback.append("- Install targets:")
                for target in targets:
                    fallback.append(f"  - `{target}`")
                fallback.append("")

        lines = _legacy_header(sources)
        lines.extend(_legacy_lines_from_body(body, title, fallback))
        _write(repo_root / rel_path, "\n".join(lines) + "\n")


def export_claims_tasks(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/claims_tasks.toml")
    meta = data.get("claims_tasks", {})
    sections = data.get("section", [])
    tasks = data.get("task", [])
    lines = _header("Claims Tasks Registry Mirror")
    lines.append("Authoritative source: `registry/claims_tasks.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown: `{meta.get('source_markdown', '')}`")
    lines.append(f"- Task count: {meta.get('task_count', len(tasks))}")
    lines.append(f"- Section count: {meta.get('section_count', len(sections))}")
    lines.append(
        f"- Canonical status task count: {meta.get('canonical_status_task_count', '')}"
    )
    lines.append(
        f"- Noncanonical status task count: {meta.get('noncanonical_status_task_count', '')}"
    )
    lines.append("")
    lines.append("## Sections")
    lines.append("")
    for section in sections:
        section_id = section.get("id", "")
        section_name = section.get("name", "")
        section_count = section.get("task_count", 0)
        lines.append(f"- {section_id}: {section_name} ({section_count} tasks)")
    lines.append("")
    lines.append("## Tasks")
    lines.append("")
    for task in sorted(tasks, key=lambda row: int(row.get("order_index", 0))):
        task_id = task.get("id", "CTASK-???")
        claim_id = task.get("claim_id", "C-???")
        status_token = task.get("status_token", "UNKNOWN")
        lines.append(f"### {task_id} ({claim_id}, {status_token})")
        lines.append("")
        lines.append(f"- Section: {task.get('section', '')}")
        lines.append(f"- Source line: {task.get('source_line', '')}")
        lines.append(f"- Status raw: {task.get('status_raw', '')}")
        lines.append(f"- Canonical: `{task.get('status_canonical', False)}`")
        lines.append("")
        lines.append(task.get("task", "").strip())
        lines.append("")
        artifacts = task.get("output_artifacts", [])
        if artifacts:
            lines.append("Output artifacts:")
            for artifact in artifacts:
                lines.append(f"- `{artifact}`")
            lines.append("")
    _write(out_path, "\n".join(lines))


def export_claims_tasks_legacy(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/claims_tasks.toml")
    sections = data.get("section", [])
    tasks = data.get("task", [])

    by_section: dict[str, list[dict]] = {}
    for task in tasks:
        by_section.setdefault(str(task.get("section", "unscoped")), []).append(task)

    lines = [
        "# Claims -> Tasks Tracker (Generated Mirror)",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: registry/claims_tasks.toml -->",
        "",
        "This file is generated from `registry/claims_tasks.toml`.",
        "",
    ]

    for section in sorted(
        sections, key=lambda row: int(row.get("id", "CTS-999").split("-")[1])
    ):
        name = str(section.get("name", "unscoped"))
        section_tasks = sorted(
            by_section.get(name, []), key=lambda row: int(row.get("order_index", 0))
        )
        lines.append(f"## {name}")
        lines.append("")
        if not section_tasks:
            lines.append("No task rows currently in this section.")
            lines.append("")
            continue
        lines.append("| Claim ID | Task | Output artifact(s) | Status |")
        lines.append("|---|---|---|---|")
        for task in section_tasks:
            claim_id = str(task.get("claim_id", ""))
            task_text = _pipe_escape(str(task.get("task", "")).strip())
            artifacts = task.get("output_artifacts", [])
            artifact_cell = ", ".join(f"`{item}`" for item in artifacts) if artifacts else "(none)"
            status = str(task.get("status_token", ""))
            lines.append(f"| {claim_id} | {task_text} | {artifact_cell} | {status} |")
        lines.append("")

    _write(out_path, "\n".join(lines))


def export_claims_domains(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/claims_domains.toml")
    meta = data.get("claims_domains", {})
    domains = data.get("domain", [])
    claim_domains = data.get("claim_domain", [])
    lines = _header("Claims Domains Registry Mirror")
    lines.append("Authoritative source: `registry/claims_domains.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source CSV: `{meta.get('source_csv', '')}`")
    lines.append(f"- Source markdown glob: `{meta.get('source_markdown_glob', '')}`")
    lines.append(f"- Domain file count: {meta.get('domain_file_count', len(domains))}")
    lines.append(f"- Claim count: {meta.get('claim_count', len(claim_domains))}")
    lines.append("")
    lines.append("## Domain Files")
    lines.append("")
    for row in sorted(domains, key=lambda item: item.get("id", "")):
        lines.append(f"### {row.get('id', '')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Declared count: {row.get('declared_count', 0)}")
        lines.append(f"- CSV claim count: {row.get('csv_claim_count', 0)}")
        lines.append(f"- Markdown claim count: {row.get('markdown_claim_count', 0)}")
        lines.append(f"- Count match: `{row.get('count_match', False)}`")
        lines.append(f"- Mapping match: `{row.get('mapping_match', False)}`")
        lines.append("")
    lines.append("## Claim Crosswalk")
    lines.append("")
    for row in sorted(claim_domains, key=lambda item: item.get("claim_id", "")):
        claim_id = row.get("claim_id", "")
        domains_csv = row.get("domains_csv", [])
        domains_markdown = row.get("domains_markdown", [])
        domain_sets_match = row.get("domain_sets_match", False)
        lines.append(
            f"- {claim_id}: csv={domains_csv}, markdown={domains_markdown}, "
            f"match={domain_sets_match}"
        )
    lines.append("")
    _write(out_path, "\n".join(lines))


def export_claims_domains_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/claims_domains.toml")
    domains = sorted(data.get("domain", []), key=lambda row: row.get("id", ""))
    claim_domains = sorted(
        data.get("claim_domain", []), key=lambda row: row.get("claim_id", "")
    )
    entries = data.get("domain_entry", [])

    index_lines = [
        "# Claims by domain",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: registry/claims_domains.toml -->",
        "",
        "See also: docs/CLAIMS_DOMAIN_TAXONOMY.md",
        "",
    ]
    for row in domains:
        domain_id = str(row.get("id", ""))
        source_markdown = str(
            row.get("source_markdown", f"docs/claims/by_domain/{domain_id}.md")
        )
        count = int(row.get("markdown_claim_count", 0))
        index_lines.append(f"- `{domain_id}` ({count}): `{source_markdown}`")
    index_lines.append("")
    _write(repo_root / "docs/claims/INDEX.md", "\n".join(index_lines))

    csv_rows = [["claim_id", "domains"]]
    for row in claim_domains:
        domains_csv = row.get("domains_csv", [])
        csv_rows.append(
            [str(row.get("claim_id", "")), ";".join(str(item) for item in domains_csv)]
        )
    _write(repo_root / "docs/claims/CLAIMS_DOMAIN_MAP.csv", _render_csv(csv_rows))

    entries_by_domain: dict[str, list[dict]] = {}
    for entry in entries:
        entries_by_domain.setdefault(str(entry.get("domain", "")), []).append(entry)

    for row in domains:
        domain_id = str(row.get("id", ""))
        path = Path(
            str(row.get("source_markdown", f"docs/claims/by_domain/{domain_id}.md"))
        )
        domain_entries = sorted(
            entries_by_domain.get(domain_id, []),
            key=lambda item: _claim_sort_key(str(item.get("claim_id", ""))),
        )
        lines = [
            f"# Claims: {domain_id}",
            "",
            "<!-- AUTO-GENERATED: DO NOT EDIT -->",
            "<!-- Source of truth: registry/claims_domains.toml -->",
            "",
            f"Count: {len(domain_entries)}",
            "",
        ]
        for entry in domain_entries:
            claim_id = str(entry.get("claim_id", ""))
            status_text = str(entry.get("status_text", ""))
            status_date = str(entry.get("status_date", "")).strip()
            summary = str(entry.get("summary", "")).strip()
            where_stated = [str(item) for item in entry.get("where_stated", [])]
            status_blob = f"{status_text}, {status_date}" if status_date else status_text
            lines.append(f"- Hypothesis {claim_id} ({status_blob}): {summary}")
            if where_stated:
                where_joined = ", ".join(f"`{item}`" for item in where_stated)
                lines.append(f"  - Where stated: {where_joined}")
            lines.append("")

        _write(repo_root / path, "\n".join(lines))


def export_claim_tickets(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/claim_tickets.toml")
    meta = data.get("claim_tickets", {})
    tickets = data.get("ticket", [])
    lines = _header("Claim Tickets Registry Mirror")
    lines.append("Authoritative source: `registry/claim_tickets.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown glob: `{meta.get('source_markdown_glob', '')}`")
    lines.append(f"- Ticket count: {meta.get('ticket_count', len(tickets))}")
    lines.append("")
    lines.append("## Tickets")
    lines.append("")
    for row in sorted(tickets, key=lambda item: item.get("id", "")):
        lines.append(f"### {row.get('id', 'TICKET-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Kind: `{row.get('ticket_kind', '')}`")
        lines.append(f"- Owner: {row.get('owner', '')}")
        lines.append(f"- Created: {row.get('created', '')}")
        lines.append(f"- Status: `{row.get('status_token', '')}` ({row.get('status_raw', '')})")
        lines.append(
            f"- Claim range: {row.get('claim_range_start', 0)}..{row.get('claim_range_end', 0)}"
        )
        done = row.get("done_checkboxes", 0)
        open_checkboxes = row.get("open_checkboxes", 0)
        lines.append(f"- Checkbox progress: done={done}, open={open_checkboxes}")
        claims = row.get("claims_referenced", [])
        if claims:
            lines.append(f"- Claims referenced ({len(claims)}): {', '.join(claims)}")
        backlog = row.get("backlog_reports", [])
        if backlog:
            lines.append("- Backlog reports:")
            for item in backlog:
                lines.append(f"  - `{item}`")
        checks = row.get("acceptance_checks", [])
        if checks:
            lines.append("- Acceptance checks:")
            for check in checks:
                lines.append(f"  - `{check}`")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_claim_tickets_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/claim_tickets.toml")
    tickets = sorted(data.get("ticket", []), key=lambda item: item.get("source_markdown", ""))

    index_lines = [
        "# Claim Audit Tickets",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: registry/claim_tickets.toml -->",
        "",
        "This index and all files under `docs/tickets/*.md` are generated from TOML.",
        "",
    ]

    for row in tickets:
        rel_path = str(row.get("source_markdown", "")).strip()
        if not rel_path:
            continue
        path = repo_root / rel_path
        title = str(row.get("title", "(untitled)")).strip()
        owner = str(row.get("owner", "")).strip()
        created = str(row.get("created", "")).strip()
        status_raw = str(row.get("status_raw", "")).strip()
        status_token = str(row.get("status_token", "")).strip()
        ticket_kind = str(row.get("ticket_kind", "")).strip()
        claim_start = int(row.get("claim_range_start", 0))
        claim_end = int(row.get("claim_range_end", 0))
        goal_summary = str(row.get("goal_summary", "")).strip()
        done_checkboxes = int(row.get("done_checkboxes", 0))
        open_checkboxes = int(row.get("open_checkboxes", 0))
        claims = [str(item) for item in row.get("claims_referenced", [])]
        backlog = [str(item) for item in row.get("backlog_reports", [])]
        deliverables = [str(item) for item in row.get("deliverable_links", [])]
        checks = [str(item) for item in row.get("acceptance_checks", [])]

        lines: list[str] = []
        lines.append(f"# {title}")
        lines.append("")
        lines.append("<!-- AUTO-GENERATED: DO NOT EDIT -->")
        lines.append("<!-- Source of truth: registry/claim_tickets.toml -->")
        lines.append("")
        lines.append(f"Owner: {owner}")
        lines.append(f"Created: {created}")
        lines.append(f"Status: {status_raw}")
        lines.append("")
        lines.append("## Goal")
        lines.append("")
        lines.append(goal_summary if goal_summary else "(not specified)")
        lines.append("")
        lines.append("## Scope")
        lines.append("")
        lines.append(f"- Ticket ID: `{row.get('id', '')}`")
        lines.append(f"- Kind: `{ticket_kind}`")
        lines.append(f"- Status token: `{status_token}`")
        if claim_start > 0 and claim_end > 0:
            lines.append(f"- Claim range: C-{claim_start:03d}..C-{claim_end:03d}")
        else:
            lines.append("- Claim range: none (general ticket)")
        if claims:
            lines.append(f"- Claims referenced ({len(claims)}): {', '.join(claims)}")
        else:
            lines.append("- Claims referenced: none")
        lines.append("")
        lines.append("## Deliverables")
        lines.append("")
        if deliverables:
            for item in deliverables:
                lines.append(f"- `{item}`")
        else:
            lines.append("- (none recorded)")
        lines.append("")
        lines.append("## Acceptance checks")
        lines.append("")
        if checks:
            for item in checks:
                lines.append(f"- `{item}`")
        else:
            lines.append("- (none recorded)")
        lines.append("")
        lines.append("## Progress snapshot")
        lines.append("")
        lines.append(f"- Completed checkboxes: {done_checkboxes}")
        lines.append(f"- Open checkboxes: {open_checkboxes}")
        if backlog:
            lines.append("- Backlog reports:")
            for item in backlog:
                lines.append(f"  - `{item}`")
        lines.append("")

        _write(path, "\n".join(lines))
        index_lines.append(f"- `{row.get('id', '')}`: `{rel_path}`")

    index_lines.append("")
    _write(repo_root / "docs/tickets/INDEX.md", "\n".join(index_lines))


def export_external_sources(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/external_sources.toml")
    meta = data.get("external_sources", {})
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    lines = _header("External Sources Registry Mirror")
    lines.append("Authoritative source: `registry/external_sources.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown glob: `{meta.get('source_markdown_glob', '')}`")
    lines.append(f"- Document count: {meta.get('document_count', len(docs))}")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    for row in docs:
        lines.append(f"### {row.get('id', 'XS-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Slug: `{row.get('slug', '')}`")
        lines.append(f"- Status token: `{row.get('status_token', '')}`")
        lines.append(f"- Content kind: `{row.get('content_kind', '')}`")
        lines.append(f"- Authority level: `{row.get('authority_level', '')}`")
        lines.append(f"- Verification level: `{row.get('verification_level', '')}`")
        lines.append(f"- Has full transcript: `{row.get('has_full_transcript', False)}`")
        lines.append(f"- Line count: {row.get('line_count', 0)}")
        claims = row.get("claim_refs", [])
        if claims:
            lines.append(f"- Claim refs ({len(claims)}): {', '.join(claims)}")
        urls = row.get("url_refs", [])
        if urls:
            lines.append(f"- URL refs ({len(urls)}):")
            for url in urls[:10]:
                lines.append(f"  - `{url}`")
            if len(urls) > 10:
                lines.append(f"  - ... ({len(urls) - 10} more)")
        notes = str(row.get("notes", "")).strip()
        if notes:
            lines.append(f"- Notes: {notes}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_external_sources_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/external_sources.toml")
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    index_lines = [
        "# External Sources",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: registry/external_sources.toml -->",
        "",
        "This index and all files under `docs/external_sources/*.md` are generated from TOML.",
        "",
    ]

    for row in docs:
        rel_path = str(row.get("source_markdown", "")).strip()
        if not rel_path:
            continue
        path = repo_root / rel_path
        body = str(row.get("body_markdown", "")).strip("\n")
        lines: list[str] = [
            "<!-- AUTO-GENERATED: DO NOT EDIT -->",
            "<!-- Source of truth: registry/external_sources.toml -->",
            "",
        ]
        if body:
            lines.extend(body.splitlines())
        else:
            title = str(row.get("title", Path(rel_path).stem))
            lines.append(f"# {title}")
            lines.append("")
            lines.append("(No body_markdown captured in registry/external_sources.toml.)")
        lines.append("")
        _write(path, "\n".join(lines))
        index_lines.append(
            f"- `{row.get('id', 'XS-???')}` `{row.get('status_token', '')}`: `{rel_path}`"
        )

    index_lines.append("")
    _write(repo_root / "docs/external_sources/INDEX.md", "\n".join(index_lines))


def export_research_narratives(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/research_narratives.toml")
    meta = data.get("research_narratives", {})
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    lines = _header("Research Narratives Registry Mirror")
    lines.append("Authoritative source: `registry/research_narratives.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Document count: {meta.get('document_count', len(docs))}")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    for row in docs:
        lines.append(f"### {row.get('id', 'RN-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Domain: `{row.get('domain', '')}`")
        lines.append(f"- Status token: `{row.get('status_token', '')}`")
        lines.append(f"- Content kind: `{row.get('content_kind', '')}`")
        lines.append(f"- Verification level: `{row.get('verification_level', '')}`")
        lines.append(f"- Line count: {row.get('line_count', 0)}")
        claims = row.get("claim_refs", [])
        if claims:
            lines.append(f"- Claim refs ({len(claims)}): {', '.join(claims)}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_research_narratives_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/research_narratives.toml")
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    generated_index_paths = {"docs/theory/INDEX.md", "docs/engineering/INDEX.md"}
    theory_index = [
        "# Theory Narratives",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: registry/research_narratives.toml -->",
        "",
        "This index and all files under `docs/theory/*.md` are generated from TOML.",
        "",
    ]
    engineering_index = [
        "# Engineering Narratives",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: registry/research_narratives.toml -->",
        "",
        "This index and all files under `docs/engineering/*.md` are generated from TOML.",
        "",
    ]

    for row in docs:
        rel_path = str(row.get("source_markdown", "")).strip()
        if not rel_path:
            continue
        if rel_path in generated_index_paths:
            # These index files are generated at the end of this function from the
            # collected entries and should not be rewritten from body_markdown.
            continue
        path = repo_root / rel_path
        body = str(row.get("body_markdown", "")).strip("\n")
        lines: list[str] = [
            "<!-- AUTO-GENERATED: DO NOT EDIT -->",
            "<!-- Source of truth: registry/research_narratives.toml -->",
            "",
        ]
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append(f"# {row.get('title', Path(rel_path).stem)}")
            lines.append("")
            lines.append("(No body_markdown captured in registry/research_narratives.toml.)")
        lines.append("")
        _write(path, "\n".join(lines))

        entry_line = (
            f"- `{row.get('id', 'RN-???')}` `{row.get('status_token', '')}`: `{rel_path}`"
        )
        if rel_path.startswith("docs/theory/"):
            theory_index.append(entry_line)
        elif rel_path.startswith("docs/engineering/"):
            engineering_index.append(entry_line)

    theory_index.append("")
    engineering_index.append("")
    _write(repo_root / "docs/theory/INDEX.md", "\n".join(theory_index))
    _write(repo_root / "docs/engineering/INDEX.md", "\n".join(engineering_index))


def export_book_docs(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/book_docs.toml")
    meta = data.get("book_docs", {})
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    lines = _header("Book Docs Registry Mirror")
    lines.append("Authoritative source: `registry/book_docs.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown glob: `{meta.get('source_markdown_glob', '')}`")
    lines.append(f"- Document count: {meta.get('document_count', len(docs))}")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    for row in docs:
        lines.append(f"### {row.get('id', 'BOOK-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Section: `{row.get('section', '')}`")
        lines.append(f"- Slug: `{row.get('slug', '')}`")
        lines.append(f"- Line count: {row.get('line_count', 0)}")
        claims = row.get("claim_refs", [])
        if claims:
            lines.append(f"- Claim refs ({len(claims)}): {', '.join(claims)}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_book_docs_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/book_docs.toml")
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    for row in docs:
        rel_path = str(row.get("source_markdown", "")).strip()
        if not rel_path:
            continue
        path = repo_root / rel_path
        body = str(row.get("body_markdown", "")).strip("\n")
        lines: list[str] = [
            "<!-- AUTO-GENERATED: DO NOT EDIT -->",
            "<!-- Source of truth: registry/book_docs.toml -->",
            "",
        ]
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append(f"# {row.get('title', Path(rel_path).stem)}")
            lines.append("")
            lines.append("(No body_markdown captured in registry/book_docs.toml.)")
        lines.append("")
        _write(path, "\n".join(lines))


def export_docs_root_narratives(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/docs_root_narratives.toml")
    meta = data.get("docs_root_narratives", {})
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    lines = _header("Docs Root Narratives Registry Mirror")
    lines.append("Authoritative source: `registry/docs_root_narratives.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown glob: `{meta.get('source_markdown_glob', '')}`")
    lines.append(f"- Document count: {meta.get('document_count', len(docs))}")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    for row in docs:
        lines.append(f"### {row.get('id', 'DRN-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Slug: `{row.get('slug', '')}`")
        lines.append(f"- Status token: `{row.get('status_token', '')}`")
        lines.append(f"- Content kind: `{row.get('content_kind', '')}`")
        lines.append(f"- Line count: {row.get('line_count', 0)}")
        claims = row.get("claim_refs", [])
        if claims:
            lines.append(f"- Claim refs ({len(claims)}): {', '.join(claims)}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_docs_root_narratives_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/docs_root_narratives.toml")
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    for row in docs:
        rel_path = str(row.get("source_markdown", "")).strip()
        if not rel_path:
            continue
        path = repo_root / rel_path
        body = str(row.get("body_markdown", "")).strip("\n")
        lines: list[str] = [
            "<!-- AUTO-GENERATED: DO NOT EDIT -->",
            "<!-- Source of truth: registry/docs_root_narratives.toml -->",
            "",
        ]
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append(f"# {row.get('title', Path(rel_path).stem)}")
            lines.append("")
            lines.append("(No body_markdown captured in registry/docs_root_narratives.toml.)")
        lines.append("")
        _write(path, "\n".join(lines))


def export_data_artifact_narratives(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/data_artifact_narratives.toml")
    meta = data.get("data_artifact_narratives", {})
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    lines = _header("Data Artifact Narratives Registry Mirror")
    lines.append("Authoritative source: `registry/data_artifact_narratives.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown count: {meta.get('source_markdown_count', len(docs))}")
    lines.append(f"- Document count: {meta.get('document_count', len(docs))}")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    for row in docs:
        lines.append(f"### {row.get('id', 'ART-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Content kind: `{row.get('content_kind', '')}`")
        lines.append(f"- Line count: {row.get('line_count', 0)}")
        claims = row.get("claim_refs", [])
        if claims:
            lines.append(f"- Claim refs ({len(claims)}): {', '.join(claims)}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_data_artifact_narratives_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/data_artifact_narratives.toml")
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    for row in docs:
        rel_path = str(row.get("source_markdown", "")).strip()
        if not rel_path:
            continue
        path = repo_root / rel_path
        body = str(row.get("body_markdown", "")).strip("\n")
        lines: list[str] = [
            "<!-- AUTO-GENERATED: DO NOT EDIT -->",
            "<!-- Source of truth: registry/data_artifact_narratives.toml -->",
            "",
        ]
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append(f"# {row.get('title', Path(rel_path).stem)}")
            lines.append("")
            lines.append("(No body_markdown captured in registry/data_artifact_narratives.toml.)")
        lines.append("")
        _write(path, "\n".join(lines))


def export_monograph_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/monograph.toml")
    docs = sorted(data.get("document", []), key=lambda item: item.get("path", ""))
    for row in docs:
        rel_path = str(row.get("path", "")).strip()
        if not rel_path:
            continue
        path = repo_root / rel_path
        body = _ascii_sanitize(str(row.get("body_markdown", "")).strip("\n"))
        lines: list[str] = [
            "<!-- AUTO-GENERATED: DO NOT EDIT -->",
            "<!-- Source of truth: registry/monograph.toml -->",
            "",
        ]
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append(f"# {row.get('title', Path(rel_path).stem)}")
            lines.append("")
            lines.append("(No body_markdown captured in registry/monograph.toml.)")
        lines.append("")
        _write(path, "\n".join(lines))


def export_reports_narratives(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/reports_narratives.toml")
    meta = data.get("reports_narratives", {})
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    lines = _header("Reports Narratives Registry Mirror")
    lines.append("Authoritative source: `registry/reports_narratives.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown glob: `{meta.get('source_markdown_glob', '')}`")
    lines.append(f"- Document count: {meta.get('document_count', len(docs))}")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    for row in docs:
        lines.append(f"### {row.get('id', 'RPT-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Category: `{row.get('category', '')}`")
        lines.append(f"- Line count: {row.get('line_count', 0)}")
        claims = row.get("claim_refs", [])
        if claims:
            lines.append(f"- Claim refs ({len(claims)}): {', '.join(claims)}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_reports_narratives_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/reports_narratives.toml")
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    for row in docs:
        rel_path = str(row.get("source_markdown", "")).strip()
        if not rel_path:
            continue
        path = repo_root / rel_path
        body = str(row.get("body_markdown", "")).strip("\n")
        lines: list[str] = [
            "<!-- AUTO-GENERATED: DO NOT EDIT -->",
            "<!-- Source of truth: registry/reports_narratives.toml -->",
            "",
        ]
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append(f"# {row.get('title', Path(rel_path).stem)}")
            lines.append("")
            lines.append("(No body_markdown captured in registry/reports_narratives.toml.)")
        lines.append("")
        _write(path, "\n".join(lines))


def export_docs_convos(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/docs_convos.toml")
    meta = data.get("docs_convos", {})
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    lines = _header("Docs Convos Registry Mirror")
    lines.append("Authoritative source: `registry/docs_convos.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown glob: `{meta.get('source_markdown_glob', '')}`")
    lines.append(f"- Document count: {meta.get('document_count', len(docs))}")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    for row in docs:
        lines.append(f"### {row.get('id', 'CVX-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Content kind: `{row.get('content_kind', '')}`")
        lines.append(f"- Line count: {row.get('line_count', 0)}")
        claims = row.get("claim_refs", [])
        if claims:
            lines.append(f"- Claim refs ({len(claims)}): {', '.join(claims)}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_docs_convos_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/docs_convos.toml")
    docs = sorted(data.get("document", []), key=lambda item: item.get("source_markdown", ""))
    for row in docs:
        rel_path = str(row.get("source_markdown", "")).strip()
        if not rel_path:
            continue
        path = repo_root / rel_path
        body = str(row.get("body_markdown", "")).strip("\n")
        lines: list[str] = [
            "<!-- AUTO-GENERATED: DO NOT EDIT -->",
            "<!-- Source of truth: registry/docs_convos.toml -->",
            "",
        ]
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append(f"# {row.get('title', Path(rel_path).stem)}")
            lines.append("")
            lines.append("(No body_markdown captured in registry/docs_convos.toml.)")
        lines.append("")
        _write(path, "\n".join(lines))


def main() -> int:
    global CHECK_MODE

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/generated",
        help="Output markdown mirror directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: fail if any mirror would change.",
    )
    parser.add_argument(
        "--legacy-claims-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also regenerate legacy claims-support markdown/csv mirrors from TOML.",
    )
    args = parser.parse_args()

    CHECK_MODE = bool(args.check)

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    if not CHECK_MODE:
        out_dir.mkdir(parents=True, exist_ok=True)

    export_insights(repo_root, out_dir / "INSIGHTS_REGISTRY_MIRROR.md")
    export_claims(repo_root, out_dir / "CLAIMS_REGISTRY_MIRROR.md")
    export_bibliography(repo_root, out_dir / "BIBLIOGRAPHY_REGISTRY_MIRROR.md")
    export_experiments(repo_root, out_dir / "EXPERIMENTS_REGISTRY_MIRROR.md")
    export_roadmap(repo_root, out_dir / "ROADMAP_REGISTRY_MIRROR.md")
    export_todo(repo_root, out_dir / "TODO_REGISTRY_MIRROR.md")
    export_next_actions(repo_root, out_dir / "NEXT_ACTIONS_REGISTRY_MIRROR.md")
    export_requirements(repo_root, out_dir / "REQUIREMENTS_REGISTRY_MIRROR.md")
    export_knowledge_migration_plan(
        repo_root, out_dir / "KNOWLEDGE_MIGRATION_PLAN_REGISTRY_MIRROR.md"
    )
    export_navigator(repo_root, out_dir / "NAVIGATOR_REGISTRY_MIRROR.md")
    export_entrypoint_docs(repo_root, out_dir / "ENTRYPOINT_DOCS_REGISTRY_MIRROR.md")
    export_markdown_governance(repo_root, out_dir / "MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md")
    export_claims_tasks(repo_root, out_dir / "CLAIMS_TASKS_REGISTRY_MIRROR.md")
    export_claims_domains(repo_root, out_dir / "CLAIMS_DOMAINS_REGISTRY_MIRROR.md")
    export_claim_tickets(repo_root, out_dir / "CLAIM_TICKETS_REGISTRY_MIRROR.md")
    export_external_sources(repo_root, out_dir / "EXTERNAL_SOURCES_REGISTRY_MIRROR.md")
    export_research_narratives(repo_root, out_dir / "RESEARCH_NARRATIVES_REGISTRY_MIRROR.md")
    export_book_docs(repo_root, out_dir / "BOOK_DOCS_REGISTRY_MIRROR.md")
    export_docs_root_narratives(repo_root, out_dir / "DOCS_ROOT_NARRATIVES_REGISTRY_MIRROR.md")
    export_data_artifact_narratives(
        repo_root, out_dir / "DATA_ARTIFACT_NARRATIVES_REGISTRY_MIRROR.md"
    )
    export_reports_narratives(repo_root, out_dir / "REPORTS_NARRATIVES_REGISTRY_MIRROR.md")
    export_docs_convos(repo_root, out_dir / "DOCS_CONVOS_REGISTRY_MIRROR.md")
    export_insights_legacy(repo_root, repo_root / "docs/INSIGHTS.md")
    export_experiments_legacy(
        repo_root, repo_root / "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md"
    )
    export_roadmap_legacy(repo_root, repo_root / "docs/ROADMAP.md")
    export_todo_legacy(repo_root, repo_root / "docs/TODO.md")
    export_next_actions_legacy(repo_root, repo_root / "docs/NEXT_ACTIONS.md")
    export_requirements_legacy(repo_root)
    export_navigator_legacy(repo_root)
    export_entrypoint_docs_legacy(repo_root)
    export_bibliography_legacy(repo_root, repo_root / "docs/BIBLIOGRAPHY.md")

    if args.legacy_claims_sync:
        export_claims_matrix_legacy(repo_root, repo_root / "docs/CLAIMS_EVIDENCE_MATRIX.md")
        export_claims_tasks_legacy(repo_root, repo_root / "docs/CLAIMS_TASKS.md")
        export_claims_domains_legacy(repo_root)
        export_claim_tickets_legacy(repo_root)
        export_external_sources_legacy(repo_root)
        export_research_narratives_legacy(repo_root)
        export_book_docs_legacy(repo_root)
        export_docs_root_narratives_legacy(repo_root)
        export_data_artifact_narratives_legacy(repo_root)
        export_monograph_legacy(repo_root)
        export_reports_narratives_legacy(repo_root)
        export_docs_convos_legacy(repo_root)

    if CHECK_MODE:
        if CHANGED_PATHS:
            print("ERROR: TOML-driven mirrors are stale. Regenerate with make registry.")
            for path in sorted(set(CHANGED_PATHS)):
                print(path)
            return 1
        print("OK: TOML-driven mirrors are fresh.")
        return 0

    print(f"Wrote TOML-driven markdown mirrors to {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
