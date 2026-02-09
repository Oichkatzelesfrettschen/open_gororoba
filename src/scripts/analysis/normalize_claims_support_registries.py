#!/usr/bin/env python3
"""
Normalize markdown claim-support sources into TOML-first registries.

Inputs:
- docs/CLAIMS_TASKS.md
- docs/claims/CLAIMS_DOMAIN_MAP.csv
- docs/claims/by_domain/*.md
- docs/tickets/*.md

Outputs:
- registry/claims_tasks.toml
- registry/claims_domains.toml
- registry/claim_tickets.toml
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CLAIM_ID_RE = re.compile(r"\bC-\d{3}\b")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
HEADING_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$")
TASK_TABLE_HEADER_RE = re.compile(
    r"^\|\s*Claim ID\s*\|\s*Task\s*\|\s*Output artifact\(s\)\s*\|\s*Status\s*\|\s*$"
)
HYPOTHESIS_LINE_RE = re.compile(r"^-\s+Hypothesis\s+(C-\d{3})\s+\((.*?)\):\s+(.*)$")
DATE_AT_END_RE = re.compile(r",\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$")
TICKET_RANGE_RE = re.compile(r"^C(\d{3})_C(\d{3})_claims_audit\.md$")
META_LINE_RE = re.compile(r"^(Owner|Created|Status):\s*(.+?)\s*$")

CANONICAL_TASK_STATUS = (
    "TODO",
    "IN_PROGRESS",
    "PARTIAL",
    "DONE",
    "REFUTED",
    "DEFERRED",
    "BLOCKED",
)
TASK_STATUS_MAP = {
    "TODO": "TODO",
    "IN PROGRESS": "IN_PROGRESS",
    "PARTIAL": "PARTIAL",
    "DONE": "DONE",
    "REFUTED": "REFUTED",
    "DEFERRED": "DEFERRED",
    "BLOCKED": "BLOCKED",
}


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    claim_id: str
    section: str
    section_index: int
    table_index: int
    order_index: int
    task_text: str
    output_artifacts: list[str]
    output_raw: str
    status_raw: str
    status_token: str
    status_canonical: bool
    source_line: int


@dataclass(frozen=True)
class DomainEntry:
    domain: str
    claim_id: str
    status_text: str
    status_token: str
    status_date: str
    summary: str
    where_stated: list[str]
    source_markdown: str
    source_line: int


@dataclass(frozen=True)
class DomainRecord:
    domain: str
    source_markdown: str
    declared_count: int
    csv_claim_ids: list[str]
    markdown_claim_ids: list[str]

    @property
    def csv_claim_count(self) -> int:
        return len(self.csv_claim_ids)

    @property
    def markdown_claim_count(self) -> int:
        return len(self.markdown_claim_ids)

    @property
    def count_match(self) -> bool:
        return self.declared_count == self.markdown_claim_count

    @property
    def mapping_match(self) -> bool:
        return self.csv_claim_ids == self.markdown_claim_ids


@dataclass(frozen=True)
class TicketRecord:
    ticket_id: str
    title: str
    source_markdown: str
    ticket_kind: str
    owner: str
    created: str
    status_raw: str
    status_token: str
    claim_range_start: int
    claim_range_end: int
    claims_referenced: list[str]
    backlog_reports: list[str]
    deliverable_links: list[str]
    acceptance_checks: list[str]
    goal_summary: str
    done_checkboxes: int
    open_checkboxes: int


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


def _render_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(_escape_toml(item) for item in items) + "]"


def _claim_sort_key(claim_id: str) -> tuple[int, str]:
    m = CLAIM_ID_RE.search(claim_id)
    if not m:
        return (999999, claim_id)
    return (int(m.group(0).split("-")[1]), claim_id)


def _normalize_task_status(raw: str) -> tuple[str, bool]:
    cleaned = re.sub(r"\s+", " ", raw.strip().upper())
    token = TASK_STATUS_MAP.get(cleaned, cleaned.replace(" ", "_"))
    return token, token in CANONICAL_TASK_STATUS


def _normalize_domain_status(raw: str) -> str:
    text = raw.strip().lower()
    if "partially verified" in text:
        return "PARTIALLY_VERIFIED"
    if "not supported" in text:
        return "NOT_SUPPORTED"
    if "speculative" in text:
        return "SPECULATIVE"
    if "refuted" in text:
        return "REFUTED"
    if "verified" in text:
        return "VERIFIED"
    if "modeled" in text:
        return "MODELED"
    if "closed" in text:
        return "CLOSED"
    if "blocked" in text:
        return "BLOCKED"
    if "deferred" in text:
        return "DEFERRED"
    return "UNKNOWN"


def _extract_backtick_paths(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in BACKTICK_RE.findall(text):
        item = raw.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _is_path_like(token: str) -> bool:
    if not token or " " in token:
        return False
    if token.startswith("http://") or token.startswith("https://"):
        return True
    if "/" in token:
        return True
    return token.endswith(
        (
            ".md",
            ".toml",
            ".py",
            ".rs",
            ".csv",
            ".json",
            ".txt",
            ".pdf",
            ".yaml",
            ".yml",
            ".h5",
        )
    )


def _extract_output_artifacts(text: str) -> list[str]:
    links = _extract_backtick_paths(text)
    if re.search(r"\bCLAIMS_EVIDENCE_MATRIX\.md\b", text) and (
        "docs/CLAIMS_EVIDENCE_MATRIX.md" not in links
    ):
        links.append("docs/CLAIMS_EVIDENCE_MATRIX.md")
    return links


def _parse_task_table_row(row: str) -> tuple[str, str, str, str] | None:
    stripped = row.strip()
    if not stripped.startswith("|"):
        return None
    inner = stripped.strip("|")
    parts = [part.strip() for part in inner.split("|")]
    if len(parts) < 4:
        return None
    claim_id = parts[0]
    if not re.fullmatch(r"C-\d{3}", claim_id):
        return None
    status_raw = parts[-1]
    output_raw = parts[-2]
    task_text = "|".join(parts[1:-2]).strip()
    return claim_id, task_text, output_raw, status_raw


def parse_claims_tasks(source_path: Path) -> tuple[list[TaskRecord], list[str], int]:
    lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tasks: list[TaskRecord] = []
    section_names: list[str] = []
    section_index_by_name: dict[str, int] = {}
    current_section = "unscoped"
    table_index = 0
    order_index = 0
    i = 0
    while i < len(lines):
        heading_match = HEADING_RE.match(lines[i])
        if heading_match:
            current_section = heading_match.group(2).strip()
            if current_section not in section_index_by_name:
                section_index_by_name[current_section] = len(section_names) + 1
                section_names.append(current_section)
            i += 1
            continue
        if TASK_TABLE_HEADER_RE.match(lines[i]):
            table_index += 1
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                if re.match(r"^\|\s*[-:]+\s*\|\s*[-:]+\s*\|\s*[-:]+\s*\|\s*[-:]+\s*\|$", row):
                    i += 1
                    continue
                parsed = _parse_task_table_row(row)
                if parsed is None:
                    i += 1
                    continue
                order_index += 1
                claim_id, task_text, output_raw, status_raw = parsed
                status_token, canonical = _normalize_task_status(status_raw)
                section = current_section
                if section not in section_index_by_name:
                    section_index_by_name[section] = len(section_names) + 1
                    section_names.append(section)
                tasks.append(
                    TaskRecord(
                        task_id=f"CTASK-{order_index:04d}",
                        claim_id=claim_id,
                        section=section,
                        section_index=section_index_by_name[section],
                        table_index=table_index,
                        order_index=order_index,
                        task_text=task_text,
                        output_artifacts=_extract_output_artifacts(output_raw),
                        output_raw=output_raw,
                        status_raw=status_raw,
                        status_token=status_token,
                        status_canonical=canonical,
                        source_line=i + 1,
                    )
                )
                i += 1
            continue
        i += 1
    return tasks, section_names, table_index


def parse_claims_domain_map(csv_path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claim_id = str(row.get("claim_id", "")).strip()
            domains_raw = str(row.get("domains", "")).strip()
            if not claim_id:
                continue
            domains = [part.strip() for part in domains_raw.split(";") if part.strip()]
            out[claim_id] = sorted(dict.fromkeys(domains))
    return out


def parse_by_domain_markdown(
    markdown_path: Path, source_markdown: str
) -> tuple[int, list[DomainEntry]]:
    domain = markdown_path.stem
    lines = markdown_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    declared_count = 0
    entries: list[DomainEntry] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Count:"):
            try:
                declared_count = int(line.split(":", 1)[1].strip())
            except ValueError:
                declared_count = 0
        m = HYPOTHESIS_LINE_RE.match(line.strip())
        if m:
            claim_id = m.group(1).strip()
            status_blob = m.group(2).strip()
            summary = m.group(3).strip()
            status_date = ""
            date_match = DATE_AT_END_RE.search(status_blob)
            if date_match:
                status_date = date_match.group(1)
                status_text = status_blob[: date_match.start()].strip().rstrip(",")
            else:
                status_text = status_blob
            status_text = status_text.replace("**", "").strip()
            where_lines: list[str] = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                stripped = nxt.strip()
                if stripped.startswith("- Hypothesis "):
                    break
                if stripped.startswith("## "):
                    break
                if stripped.startswith("- Where stated:") or stripped.startswith("Where stated:"):
                    where_lines.append(stripped)
                    j += 1
                    continue
                if where_lines and (nxt.startswith("  ") or nxt.startswith("\t")):
                    where_lines.append(stripped)
                    j += 1
                    continue
                if where_lines:
                    break
                j += 1
            where_text = " ".join(where_lines)
            where_stated = _extract_backtick_paths(where_text)
            entries.append(
                DomainEntry(
                    domain=domain,
                    claim_id=claim_id,
                    status_text=status_text,
                    status_token=_normalize_domain_status(status_text),
                    status_date=status_date,
                    summary=summary,
                    where_stated=where_stated,
                    source_markdown=source_markdown,
                    source_line=i + 1,
                )
            )
            i = max(i + 1, j)
            continue
        i += 1
    return declared_count, entries


def parse_domains(
    csv_path: Path, by_domain_dir: Path, repo_root: Path
) -> tuple[list[DomainRecord], list[DomainEntry], dict[str, list[str]]]:
    csv_map = parse_claims_domain_map(csv_path)
    domain_records: list[DomainRecord] = []
    all_entries: list[DomainEntry] = []
    for path in sorted(by_domain_dir.glob("*.md")):
        rel_markdown = path.relative_to(repo_root).as_posix()
        declared_count, entries = parse_by_domain_markdown(path, rel_markdown)
        all_entries.extend(entries)
        domain = path.stem
        csv_claim_ids = sorted(
            [claim_id for claim_id, domains in csv_map.items() if domain in domains],
            key=_claim_sort_key,
        )
        markdown_claim_ids = sorted(
            sorted({entry.claim_id for entry in entries}, key=_claim_sort_key),
            key=_claim_sort_key,
        )
        domain_records.append(
            DomainRecord(
                domain=domain,
                source_markdown=rel_markdown,
                declared_count=declared_count,
                csv_claim_ids=csv_claim_ids,
                markdown_claim_ids=markdown_claim_ids,
            )
        )
    return domain_records, all_entries, csv_map


def _extract_ticket_section(lines: list[str], heading: str) -> list[str]:
    start = -1
    for i, line in enumerate(lines):
        if line.strip().lower() == heading.lower():
            start = i + 1
            break
    if start < 0:
        return []
    out: list[str] = []
    for i in range(start, len(lines)):
        line = lines[i]
        if i > start and line.startswith("## "):
            break
        out.append(line)
    return out


def _extract_ticket_meta(lines: list[str]) -> dict[str, str]:
    meta = {"Owner": "", "Created": "", "Status": ""}
    for line in lines[:40]:
        m = META_LINE_RE.match(line.strip())
        if not m:
            continue
        meta[m.group(1)] = m.group(2).strip()
    return meta


def _normalize_ticket_status(raw: str) -> str:
    text = raw.strip().upper()
    if "IN PROGRESS" in text:
        return "IN_PROGRESS"
    if "DONE" in text:
        return "DONE"
    if "TODO" in text:
        return "TODO"
    if "BLOCKED" in text:
        return "BLOCKED"
    if "DEFERRED" in text:
        return "DEFERRED"
    return text.replace(" ", "_")


def parse_ticket(path: Path, repo_root: Path) -> TicketRecord:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    title = path.stem
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    meta = _extract_ticket_meta(lines)
    stem = path.name
    range_match = TICKET_RANGE_RE.match(stem)
    if range_match:
        ticket_kind = "CLAIMS_AUDIT_BATCH"
        claim_range_start = int(range_match.group(1))
        claim_range_end = int(range_match.group(2))
        ticket_id = f"TICKET-C{claim_range_start:03d}-C{claim_range_end:03d}"
    else:
        ticket_kind = "GENERAL"
        claim_range_start = 0
        claim_range_end = 0
        normalized = re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-").upper()
        ticket_id = f"TICKET-{normalized}"

    claims_referenced = sorted(set(CLAIM_ID_RE.findall(text)), key=_claim_sort_key)
    backlog_reports = sorted(set(re.findall(r"reports/[A-Za-z0-9_./-]+\.md", text)))

    deliverables_block = "\n".join(_extract_ticket_section(lines, "## Deliverables"))
    deliverable_links = [
        item for item in _extract_backtick_paths(deliverables_block) if _is_path_like(item)
    ]
    if not deliverable_links:
        deliverable_links = [item for item in _extract_backtick_paths(text) if _is_path_like(item)]

    acceptance_block_lines = _extract_ticket_section(lines, "## Acceptance checks")
    acceptance_checks: list[str] = []
    for line in acceptance_block_lines:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        matches = BACKTICK_RE.findall(stripped)
        if matches:
            acceptance_checks.extend(item.strip() for item in matches if item.strip())
        else:
            acceptance_checks.append(stripped[2:].strip())
    acceptance_checks = sorted(dict.fromkeys(acceptance_checks))

    goal_lines = _extract_ticket_section(lines, "## Goal")
    goal_summary = " ".join(line.strip() for line in goal_lines if line.strip())

    done_checkboxes = len(re.findall(r"(?m)^\s*-\s+\[[xX]\]\s+", text))
    open_checkboxes = len(re.findall(r"(?m)^\s*-\s+\[\s\]\s+", text))

    return TicketRecord(
        ticket_id=ticket_id,
        title=title,
        source_markdown=path.relative_to(repo_root).as_posix(),
        ticket_kind=ticket_kind,
        owner=meta.get("Owner", ""),
        created=meta.get("Created", ""),
        status_raw=meta.get("Status", ""),
        status_token=_normalize_ticket_status(meta.get("Status", "")),
        claim_range_start=claim_range_start,
        claim_range_end=claim_range_end,
        claims_referenced=claims_referenced,
        backlog_reports=backlog_reports,
        deliverable_links=deliverable_links,
        acceptance_checks=acceptance_checks,
        goal_summary=goal_summary,
        done_checkboxes=done_checkboxes,
        open_checkboxes=open_checkboxes,
    )


def parse_tickets(tickets_dir: Path) -> list[TicketRecord]:
    out: list[TicketRecord] = []
    repo_root = tickets_dir.parent.parent
    for path in sorted(tickets_dir.glob("*.md")):
        out.append(parse_ticket(path, repo_root))
    return out


def render_claims_tasks_toml(
    tasks: list[TaskRecord], sections: list[str], table_count: int, now: str
) -> str:
    canonical_count = sum(1 for task in tasks if task.status_canonical)
    lines: list[str] = []
    lines.append("# Claims task registry (TOML-first).")
    lines.append("# Source markdown remains human-facing; this file is authoritative.")
    lines.append("")
    lines.append("[claims_tasks]")
    lines.append("authoritative = true")
    lines.append(f"updated = {_escape_toml(now)}")
    lines.append('source_markdown = "docs/CLAIMS_TASKS.md"')
    lines.append(f"table_count = {table_count}")
    lines.append(f"task_count = {len(tasks)}")
    lines.append(f"section_count = {len(sections)}")
    lines.append(f"canonical_status_task_count = {canonical_count}")
    lines.append(f"noncanonical_status_task_count = {len(tasks) - canonical_count}")
    lines.append(f"canonical_status_tokens = {_render_list(list(CANONICAL_TASK_STATUS))}")
    lines.append("")

    for idx, section in enumerate(sections, start=1):
        section_task_count = sum(1 for task in tasks if task.section == section)
        lines.append("[[section]]")
        lines.append(f"id = {_escape_toml(f'CTS-{idx:03d}')}")
        lines.append(f"name = {_escape_toml(section)}")
        lines.append(f"task_count = {section_task_count}")
        lines.append("")

    for task in tasks:
        lines.append("[[task]]")
        lines.append(f"id = {_escape_toml(task.task_id)}")
        lines.append(f"claim_id = {_escape_toml(task.claim_id)}")
        lines.append(f"section = {_escape_toml(task.section)}")
        lines.append(f"section_index = {task.section_index}")
        lines.append(f"table_index = {task.table_index}")
        lines.append(f"order_index = {task.order_index}")
        lines.append(f"status_token = {_escape_toml(task.status_token)}")
        lines.append(f"status_raw = {_escape_toml(task.status_raw)}")
        lines.append(f"status_canonical = {'true' if task.status_canonical else 'false'}")
        lines.append(f"task = {_escape_toml(task.task_text)}")
        lines.append(f"output_artifacts = {_render_list(task.output_artifacts)}")
        lines.append(f"output_artifacts_raw = {_escape_toml(task.output_raw)}")
        lines.append(f"source_line = {task.source_line}")
        lines.append("")
    return "\n".join(lines)


def render_claims_domains_toml(
    domain_records: list[DomainRecord],
    entries: list[DomainEntry],
    csv_map: dict[str, list[str]],
    now: str,
) -> str:
    markdown_claim_domains: dict[str, set[str]] = {}
    for entry in entries:
        markdown_claim_domains.setdefault(entry.claim_id, set()).add(entry.domain)
    all_claim_ids = sorted(
        set(csv_map.keys()) | set(markdown_claim_domains.keys()),
        key=_claim_sort_key,
    )

    lines: list[str] = []
    lines.append("# Claims domain registry (TOML-first).")
    lines.append("# Reconciles CLAIMS_DOMAIN_MAP.csv and generated by-domain markdown files.")
    lines.append("")
    lines.append("[claims_domains]")
    lines.append("authoritative = true")
    lines.append(f"updated = {_escape_toml(now)}")
    lines.append('source_csv = "docs/claims/CLAIMS_DOMAIN_MAP.csv"')
    lines.append('source_markdown_glob = "docs/claims/by_domain/*.md"')
    lines.append(f"domain_file_count = {len(domain_records)}")
    lines.append(f"claim_count = {len(all_claim_ids)}")
    lines.append(f"entry_count = {len(entries)}")
    lines.append("")

    for record in sorted(domain_records, key=lambda item: item.domain):
        lines.append("[[domain]]")
        lines.append(f"id = {_escape_toml(record.domain)}")
        lines.append(f"source_markdown = {_escape_toml(record.source_markdown)}")
        lines.append(f"declared_count = {record.declared_count}")
        lines.append(f"csv_claim_count = {record.csv_claim_count}")
        lines.append(f"markdown_claim_count = {record.markdown_claim_count}")
        lines.append(f"count_match = {'true' if record.count_match else 'false'}")
        lines.append(f"mapping_match = {'true' if record.mapping_match else 'false'}")
        lines.append(f"csv_claim_ids = {_render_list(record.csv_claim_ids)}")
        lines.append(f"markdown_claim_ids = {_render_list(record.markdown_claim_ids)}")
        lines.append("")

    for claim_id in all_claim_ids:
        csv_domains = sorted(csv_map.get(claim_id, []))
        markdown_domains = sorted(markdown_claim_domains.get(claim_id, set()))
        lines.append("[[claim_domain]]")
        lines.append(f"claim_id = {_escape_toml(claim_id)}")
        lines.append(f"domains_csv = {_render_list(csv_domains)}")
        lines.append(f"domains_markdown = {_render_list(markdown_domains)}")
        lines.append(
            f"domain_sets_match = {'true' if csv_domains == markdown_domains else 'false'}"
        )
        lines.append("")

    sorted_entries = sorted(
        entries,
        key=lambda item: (item.domain, _claim_sort_key(item.claim_id), item.source_line),
    )
    for entry in sorted_entries:
        lines.append("[[domain_entry]]")
        lines.append(f"domain = {_escape_toml(entry.domain)}")
        lines.append(f"claim_id = {_escape_toml(entry.claim_id)}")
        lines.append(f"status_token = {_escape_toml(entry.status_token)}")
        lines.append(f"status_text = {_escape_toml(entry.status_text)}")
        lines.append(f"status_date = {_escape_toml(entry.status_date)}")
        lines.append(f"summary = {_escape_toml(entry.summary)}")
        lines.append(f"where_stated = {_render_list(entry.where_stated)}")
        lines.append(f"source_markdown = {_escape_toml(entry.source_markdown)}")
        lines.append(f"source_line = {entry.source_line}")
        lines.append("")
    return "\n".join(lines)


def render_claim_tickets_toml(tickets: list[TicketRecord], now: str) -> str:
    lines: list[str] = []
    lines.append("# Claim-ticket registry (TOML-first).")
    lines.append("# Tracks claims audit batches and related planning tickets.")
    lines.append("")
    lines.append("[claim_tickets]")
    lines.append("authoritative = true")
    lines.append(f"updated = {_escape_toml(now)}")
    lines.append('source_markdown_glob = "docs/tickets/*.md"')
    lines.append(f"ticket_count = {len(tickets)}")
    lines.append("")
    for ticket in tickets:
        lines.append("[[ticket]]")
        lines.append(f"id = {_escape_toml(ticket.ticket_id)}")
        lines.append(f"title = {_escape_toml(ticket.title)}")
        lines.append(f"source_markdown = {_escape_toml(ticket.source_markdown)}")
        lines.append(f"ticket_kind = {_escape_toml(ticket.ticket_kind)}")
        lines.append(f"owner = {_escape_toml(ticket.owner)}")
        lines.append(f"created = {_escape_toml(ticket.created)}")
        lines.append(f"status_raw = {_escape_toml(ticket.status_raw)}")
        lines.append(f"status_token = {_escape_toml(ticket.status_token)}")
        lines.append(f"claim_range_start = {ticket.claim_range_start}")
        lines.append(f"claim_range_end = {ticket.claim_range_end}")
        lines.append(f"claims_referenced = {_render_list(ticket.claims_referenced)}")
        lines.append(f"backlog_reports = {_render_list(ticket.backlog_reports)}")
        lines.append(f"deliverable_links = {_render_list(ticket.deliverable_links)}")
        lines.append(f"acceptance_checks = {_render_list(ticket.acceptance_checks)}")
        lines.append(f"goal_summary = {_escape_toml(ticket.goal_summary)}")
        lines.append(f"done_checkboxes = {ticket.done_checkboxes}")
        lines.append(f"open_checkboxes = {ticket.open_checkboxes}")
        lines.append("")
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    _assert_ascii(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--claims-tasks-out",
        default="registry/claims_tasks.toml",
        help="Output path for claims tasks registry.",
    )
    parser.add_argument(
        "--claims-domains-out",
        default="registry/claims_domains.toml",
        help="Output path for claims domains registry.",
    )
    parser.add_argument(
        "--claim-tickets-out",
        default="registry/claim_tickets.toml",
        help="Output path for claim tickets registry.",
    )
    parser.add_argument(
        "--bootstrap-from-markdown",
        action="store_true",
        help=(
            "Explicitly allow markdown->TOML bootstrap for claim-support registries. "
            "Without this flag, the command exits to protect TOML-first authoring mode."
        ),
    )
    args = parser.parse_args()

    if not args.bootstrap_from_markdown:
        print(
            "SKIP: normalize_claims_support_registries.py is bootstrap-only. "
            "Use --bootstrap-from-markdown for one-time or explicit re-bootstrap operations."
        )
        return 0

    repo_root = Path(args.repo_root).resolve()
    claims_tasks_md = repo_root / "docs/CLAIMS_TASKS.md"
    claims_domain_map_csv = repo_root / "docs/claims/CLAIMS_DOMAIN_MAP.csv"
    claims_by_domain_dir = repo_root / "docs/claims/by_domain"
    tickets_dir = repo_root / "docs/tickets"

    tasks, sections, table_count = parse_claims_tasks(claims_tasks_md)
    domain_records, entries, csv_map = parse_domains(
        claims_domain_map_csv, claims_by_domain_dir, repo_root
    )
    tickets = parse_tickets(tickets_dir)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    claims_tasks_text = render_claims_tasks_toml(tasks, sections, table_count, now)
    claims_domains_text = render_claims_domains_toml(domain_records, entries, csv_map, now)
    claim_tickets_text = render_claim_tickets_toml(tickets, now)

    claims_tasks_out = repo_root / args.claims_tasks_out
    claims_domains_out = repo_root / args.claims_domains_out
    claim_tickets_out = repo_root / args.claim_tickets_out

    _write(claims_tasks_out, claims_tasks_text)
    _write(claims_domains_out, claims_domains_text)
    _write(claim_tickets_out, claim_tickets_text)

    print(
        "Wrote claims support registries: "
        f"{claims_tasks_out}, {claims_domains_out}, {claim_tickets_out}. "
        "Tasks="
        f"{len(tasks)}, Domains={len(domain_records)}, "
        f"DomainEntries={len(entries)}, Tickets={len(tickets)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
