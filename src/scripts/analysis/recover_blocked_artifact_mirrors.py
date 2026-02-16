#!/usr/bin/env python3
"""
Attempt automated recovery for blocked artifacts from artifact_source_of_truth.

Outputs:
- data/external/intake/global_link_audit_2026_02_15/link_audit_results_blocked_recovery.tsv
- reports/blocked_artifact_retry_plan_2026_02_15.toml
- reports/blocked_artifact_recovery_attempts_2026_02_15.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


URL_RE = re.compile(r"^https?://", re.IGNORECASE)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
ARXIV_PDF_RE = re.compile(r"arxiv\.org/pdf/(?P<id>\d{4}\.\d{4,5})(?:v(?P<v>\d+))?(?:\.pdf)?", re.IGNORECASE)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)


@dataclass
class ProbeRow:
    artifact_id: str
    artifact_key: str
    title: str
    origin: str
    url: str
    http_code: str
    content_type: str
    size_download: str
    is_pdf: bool
    status: str
    note: str


def _ascii_sanitize(text: str) -> str:
    out: list[str] = []
    for ch in str(text):
        code = ord(ch)
        if code >= 128:
            out.append(" ")
            continue
        if code < 32 and ch not in {"\n", "\r", "\t"}:
            out.append(" ")
            continue
        if code == 127:
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def _escape_toml(text: str) -> str:
    text = _ascii_sanitize(text)
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


def _normalize_url(url: str) -> str:
    value = str(url).strip()
    if not value:
        return ""
    value = value.strip("`")
    value = value.lstrip("(<[{\"'")
    value = value.rstrip(">)]}\"'")
    value = value.rstrip("`")
    value = value.rstrip(".,;:")
    return value.strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = _normalize_url(value)
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _status_from(code: str, is_pdf: bool) -> str:
    if code.startswith("2") and is_pdf:
        return "pdf_ok"
    if code.startswith("2"):
        return "ok_nonpdf"
    if code:
        return f"http_{code}"
    return "unknown"


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _extract_doi(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for match in DOI_RE.finditer(value):
            doi = match.group(0).strip().rstrip(".,;)")
            if doi.lower().startswith("https://doi.org/"):
                doi = doi.split("doi.org/", 1)[1]
            out.append(doi)
        parsed = urlparse(value)
        if parsed.netloc.lower() in {"doi.org", "dx.doi.org"}:
            doi = parsed.path.lstrip("/").strip().rstrip(".,;)")
            if doi:
                out.append(doi)
    return _dedupe(out)


def _fetch_json(url: str, timeout: int) -> dict | None:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
        return json.loads(payload)
    except Exception:
        return None


def _extract_openalex_urls(doi: str, timeout: int) -> list[tuple[str, str]]:
    encoded = quote(f"https://doi.org/{doi}", safe="")
    url = f"https://api.openalex.org/works/{encoded}"
    data = _fetch_json(url, timeout)
    if not isinstance(data, dict):
        return []

    candidates: list[tuple[str, str]] = []
    best = data.get("best_oa_location")
    if isinstance(best, dict):
        pdf_url = _normalize_url(best.get("pdf_url", ""))
        landing = _normalize_url(best.get("landing_page_url", ""))
        if pdf_url:
            candidates.append(("openalex_best_pdf", pdf_url))
        if landing:
            candidates.append(("openalex_best_landing", landing))

    primary = data.get("primary_location")
    if isinstance(primary, dict):
        pdf_url = _normalize_url(primary.get("pdf_url", ""))
        landing = _normalize_url(primary.get("landing_page_url", ""))
        if pdf_url:
            candidates.append(("openalex_primary_pdf", pdf_url))
        if landing:
            candidates.append(("openalex_primary_landing", landing))

    locations = data.get("locations", [])
    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict):
                continue
            pdf_url = _normalize_url(location.get("pdf_url", ""))
            landing = _normalize_url(location.get("landing_page_url", ""))
            if pdf_url:
                candidates.append(("openalex_location_pdf", pdf_url))
            if landing:
                candidates.append(("openalex_location_landing", landing))

    dedup: list[tuple[str, str]] = []
    seen: set[str] = set()
    for origin, candidate in candidates:
        if not URL_RE.match(candidate):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        dedup.append((origin, candidate))
    return dedup


def _extract_crossref_urls(doi: str, timeout: int) -> list[tuple[str, str]]:
    encoded = quote(doi, safe="")
    url = f"https://api.crossref.org/works/{encoded}"
    data = _fetch_json(url, timeout)
    if not isinstance(data, dict):
        return []
    message = data.get("message")
    if not isinstance(message, dict):
        return []

    candidates: list[tuple[str, str]] = []
    resource = message.get("resource")
    if isinstance(resource, dict):
        primary = _normalize_url(resource.get("primary", {}).get("URL", "")) if isinstance(resource.get("primary"), dict) else ""
        if primary:
            candidates.append(("crossref_resource_primary", primary))

    links = message.get("link", [])
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            link_url = _normalize_url(link.get("URL", ""))
            if link_url:
                candidates.append(("crossref_link", link_url))

    dedup: list[tuple[str, str]] = []
    seen: set[str] = set()
    for origin, candidate in candidates:
        if not URL_RE.match(candidate):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        dedup.append((origin, candidate))
    return dedup


def _smart_url_variants(url: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    normalized = _normalize_url(url)
    if not URL_RE.match(normalized):
        return variants

    parsed = urlparse(normalized)

    # Drop query to avoid transient tokens.
    if parsed.query:
        stripped = normalized.split("?", 1)[0]
        variants.append(("strip_query", stripped))

    # Upgrade to https when possible.
    if normalized.startswith("http://"):
        variants.append(("force_https", "https://" + normalized[len("http://"):]))

    # Known MSP malformed path variant fix.
    if "pjm-v117-n1-pjm-v117-n1-p10-s.pdf" in normalized:
        variants.append((
            "msp_path_fix",
            normalized.replace("pjm-v117-n1-pjm-v117-n1-p10-s.pdf", "pjm-v117-n1-p10-s.pdf"),
        ))

    # ArXiv withdrawn/unversioned fallback variants.
    match = ARXIV_PDF_RE.search(normalized)
    if match:
        aid = match.group("id")
        variants.extend(
            [
                ("arxiv_v1", f"https://arxiv.org/pdf/{aid}v1"),
                ("arxiv_v1_pdf", f"https://arxiv.org/pdf/{aid}v1.pdf"),
                ("arxiv_v2", f"https://arxiv.org/pdf/{aid}v2"),
                ("arxiv_v2_pdf", f"https://arxiv.org/pdf/{aid}v2.pdf"),
                ("export_arxiv", f"https://export.arxiv.org/pdf/{aid}"),
                ("export_arxiv_pdf", f"https://export.arxiv.org/pdf/{aid}.pdf"),
                ("arxiv_ftp", f"https://arxiv.org/ftp/arxiv/papers/{aid[:4]}/{aid}.pdf"),
                ("archive_org", f"https://archive.org/download/arxiv-{aid}/{aid}.pdf"),
            ]
        )

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for origin, candidate in variants:
        value = _normalize_url(candidate)
        if not URL_RE.match(value):
            continue
        if value in seen or value == normalized:
            continue
        seen.add(value)
        out.append((origin, value))
    return out


def _probe_url(url: str, timeout_s: int) -> tuple[str, str, str, bool, str]:
    cmd = [
        "curl",
        "-L",
        "--connect-timeout",
        "8",
        "--max-time",
        str(timeout_s),
        "--compressed",
        "-A",
        USER_AGENT,
        "-o",
        "/tmp/recover_blocked_probe.bin",
        "-w",
        "%{http_code}\t%{content_type}\t%{size_download}",
        url,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    raw = proc.stdout.decode("utf-8", errors="replace").strip()
    parts = raw.split("\t")
    http_code = parts[0].strip() if len(parts) > 0 else ""
    content_type = parts[1].strip() if len(parts) > 1 else ""
    size_download = parts[2].strip() if len(parts) > 2 else ""
    is_pdf = False
    note = ""
    try:
        with Path("/tmp/recover_blocked_probe.bin").open("rb") as handle:
            is_pdf = handle.read(5) == b"%PDF-"
    except OSError:
        note = "body_unreadable"

    if proc.returncode != 0 and not http_code:
        note = f"curl_exit_{proc.returncode}"

    status = _status_from(http_code, is_pdf)
    return http_code or "000", content_type, size_download or "0", is_pdf, status if not note else note


def _render_retry_plan(artifacts: list[dict], generated_at: str) -> str:
    lines: list[str] = []
    lines.append("# Retry plan for blocked artifact recovery")
    lines.append("")
    lines.append("[retry_plan]")
    lines.append('id = "BLOCKED-RETRY-2026-02-15"')
    lines.append(f"generated_at = {_escape_toml(generated_at)}")
    lines.append(f"artifact_count = {len(artifacts)}")
    lines.append("")

    for item in artifacts:
        lines.append("[[artifact]]")
        lines.append(f"id = {_escape_toml(item['id'])}")
        lines.append(f"key = {_escape_toml(item['key'])}")
        lines.append(f"title = {_escape_toml(item['title'])}")
        lines.append(f"tier = {_escape_toml(item['tier'])}")
        lines.append(f"recommended_url = {_escape_toml(item['recommended_url'])}")
        lines.append(f"recommended_status = {_escape_toml(item['recommended_status'])}")
        lines.append(f"blocked_urls = {_render_list(item['blocked_urls'])}")
        lines.append(f"new_working_urls = {_render_list(item['new_working_urls'])}")
        lines.append(f"manual_candidates = {_render_list(item['manual_candidates'])}")
        lines.append("")

    return "\n".join(lines)


def _render_recovered_registry(entries: list[dict], generated_at: str) -> str:
    lines: list[str] = []
    lines.append("# Recovered mirror candidates generated from blocked artifact recovery.")
    lines.append("")
    lines.append("[recovered_mirrors]")
    lines.append('id = "RECOVERED-MIRRORS-2026-02-15"')
    lines.append(f"generated_at = {_escape_toml(generated_at)}")
    lines.append("authoritative = false")
    lines.append(
        'policy = "Supplemental mirrors discovered by automated recovery; master registry remains authoritative after rebuild+verify."'
    )
    lines.append("")

    for item in entries:
        if not item.get("recovered_urls"):
            continue
        lines.append("[[recovered_mirror]]")
        lines.append(f"id = {_escape_toml(item['id'])}")
        lines.append(f"artifact_key = {_escape_toml(item['key'])}")
        lines.append(f"title = {_escape_toml(item['title'])}")
        # Keep blocked URLs first so URL-key artifacts can merge on identity.
        lines.append(f"blocked_urls = {_render_list(item['blocked_urls'])}")
        lines.append(f"recovered_urls = {_render_list(item['recovered_urls'])}")
        lines.append(f"recommended_url = {_escape_toml(item['recommended_url'])}")
        lines.append(f"recommended_status = {_escape_toml(item['recommended_status'])}")
        lines.append(f"tier = {_escape_toml(item['tier'])}")
        doi_values = [d for d in _extract_doi(item["blocked_urls"] + item["recovered_urls"]) if d]
        lines.append(f"doi_list = {_render_list(doi_values)}")
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
        default="registry/artifact_source_of_truth.toml",
        help="Source registry path.",
    )
    parser.add_argument(
        "--out-link-audit",
        default="data/external/intake/global_link_audit_2026_02_15/link_audit_results_blocked_recovery.tsv",
        help="Output link-audit-compatible TSV.",
    )
    parser.add_argument(
        "--out-attempts",
        default="reports/blocked_artifact_recovery_attempts_2026_02_15.tsv",
        help="Output detailed attempts TSV.",
    )
    parser.add_argument(
        "--out-plan",
        default="reports/blocked_artifact_retry_plan_2026_02_15.toml",
        help="Output retry plan TOML.",
    )
    parser.add_argument(
        "--out-compact",
        default="reports/blocked_artifacts_compact_2026_02_15.csv",
        help="Output compact blocked list CSV.",
    )
    parser.add_argument(
        "--out-recovered-registry",
        default="registry/recovered_mirrors.toml",
        help="Output recovered mirror supplemental registry.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20,
        help="Probe timeout per URL.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source_path = repo_root / args.source
    if not source_path.exists():
        raise SystemExit(f"ERROR: missing source registry: {source_path}")

    out_link_audit = repo_root / args.out_link_audit
    out_attempts = repo_root / args.out_attempts
    out_plan = repo_root / args.out_plan
    out_compact = repo_root / args.out_compact
    out_recovered_registry = repo_root / args.out_recovered_registry

    data = _load_toml(source_path)
    artifacts = data.get("artifact", [])
    blocked = [a for a in artifacts if str(a.get("status", "")).strip() == "blocked"]

    rows: list[ProbeRow] = []
    summary_rows: list[dict] = []

    for artifact in blocked:
        artifact_id = str(artifact.get("id", "")).strip()
        key = str(artifact.get("key", "")).strip()
        title = str(artifact.get("title", "")).strip()

        blocked_urls = [str(v).strip() for v in artifact.get("nonworking_mirrors", []) if str(v).strip()]
        all_links = [str(v).strip() for v in artifact.get("all_links", []) if str(v).strip()]
        doi_list = [str(v).strip() for v in artifact.get("doi_list", []) if str(v).strip()]
        doi_list = _dedupe(doi_list + _extract_doi(all_links) + _extract_doi(blocked_urls))

        candidates: list[tuple[str, str]] = []
        for url in _dedupe(blocked_urls + all_links):
            if URL_RE.match(url):
                candidates.append(("existing", url))
                candidates.extend(_smart_url_variants(url))

        for doi in doi_list:
            candidates.append(("doi_resolver", f"https://doi.org/{doi}"))
            candidates.extend(_extract_openalex_urls(doi, args.timeout_seconds))
            candidates.extend(_extract_crossref_urls(doi, args.timeout_seconds))

        # Deduplicate candidate URLs preserving first origin.
        seen: set[str] = set()
        deduped_candidates: list[tuple[str, str]] = []
        for origin, url in candidates:
            norm = _normalize_url(url)
            if not URL_RE.match(norm):
                continue
            if norm in seen:
                continue
            seen.add(norm)
            deduped_candidates.append((origin, norm))

        best_status = ""
        best_url = ""
        new_working: list[str] = []
        manual_candidates: list[str] = []

        for origin, candidate in deduped_candidates:
            http_code, content_type, size_download, is_pdf, status_or_note = _probe_url(
                candidate,
                args.timeout_seconds,
            )
            status = status_or_note
            note = ""
            if status.startswith("curl_exit_") or status == "body_unreadable":
                note = status
                status = "unknown"

            row = ProbeRow(
                artifact_id=artifact_id,
                artifact_key=key,
                title=title,
                origin=origin,
                url=candidate,
                http_code=http_code,
                content_type=content_type,
                size_download=size_download,
                is_pdf=is_pdf,
                status=status,
                note=note,
            )
            rows.append(row)

            if status in {"pdf_ok", "ok_nonpdf"}:
                if candidate not in blocked_urls:
                    new_working.append(candidate)
                if not best_url:
                    best_url = candidate
                    best_status = status
                elif best_status != "pdf_ok" and status == "pdf_ok":
                    best_url = candidate
                    best_status = status
            elif status in {"http_401", "http_403", "http_429"}:
                manual_candidates.append(candidate)

        if best_status == "pdf_ok":
            tier = "high_recoverable"
        elif best_status == "ok_nonpdf":
            tier = "medium_recoverable"
        elif manual_candidates:
            tier = "manual_intervention"
        else:
            tier = "hard_blocked"

        summary_rows.append(
            {
                "id": artifact_id,
                "key": key,
                "title": title,
                "tier": tier,
                "recommended_url": best_url,
                "recommended_status": best_status,
                "blocked_urls": _dedupe(blocked_urls),
                "new_working_urls": _dedupe(new_working),
                "manual_candidates": _dedupe(manual_candidates),
            }
        )

    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    out_link_audit.parent.mkdir(parents=True, exist_ok=True)
    with out_link_audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "url",
                "http_code",
                "content_type",
                "size_download",
                "is_pdf",
                "status",
                "note",
                "checked_at",
            ]
        )
        seen_rows: set[str] = set()
        for row in rows:
            key = row.url
            if key in seen_rows:
                continue
            seen_rows.add(key)
            writer.writerow(
                [
                    row.url,
                    row.http_code,
                    row.content_type,
                    row.size_download,
                    "1" if row.is_pdf else "0",
                    row.status,
                    row.note,
                    checked_at,
                ]
            )

    out_attempts.parent.mkdir(parents=True, exist_ok=True)
    with out_attempts.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "artifact_id",
                "artifact_key",
                "title",
                "origin",
                "url",
                "http_code",
                "is_pdf",
                "status",
                "note",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.artifact_id,
                    row.artifact_key,
                    row.title,
                    row.origin,
                    row.url,
                    row.http_code,
                    "1" if row.is_pdf else "0",
                    row.status,
                    row.note,
                ]
            )

    summary_rows.sort(key=lambda r: (r["tier"], r["key"]))
    retry_plan_text = _render_retry_plan(summary_rows, checked_at)

    merged_recovered: dict[str, dict] = {}
    if out_recovered_registry.exists():
        try:
            previous = _load_toml(out_recovered_registry)
            for entry in previous.get("recovered_mirror", []):
                key = str(entry.get("artifact_key", "")).strip()
                if not key:
                    continue
                merged_recovered[key] = {
                    "id": str(entry.get("id", "")).strip(),
                    "key": key,
                    "title": str(entry.get("title", "")).strip(),
                    "blocked_urls": _dedupe(
                        [str(v).strip() for v in entry.get("blocked_urls", []) if str(v).strip()]
                    ),
                    "recovered_urls": _dedupe(
                        [str(v).strip() for v in entry.get("recovered_urls", []) if str(v).strip()]
                    ),
                    "recommended_url": str(entry.get("recommended_url", "")).strip(),
                    "recommended_status": str(entry.get("recommended_status", "")).strip(),
                    "tier": str(entry.get("tier", "")).strip() or "manual_intervention",
                }
        except Exception:
            pass

    for item in summary_rows:
        if not item["new_working_urls"]:
            continue
        key = item["key"]
        existing = merged_recovered.get(key)
        if existing is None:
            merged_recovered[key] = {
                "id": item["id"],
                "key": key,
                "title": item["title"],
                "blocked_urls": _dedupe(item["blocked_urls"]),
                "recovered_urls": _dedupe(item["new_working_urls"]),
                "recommended_url": item["recommended_url"],
                "recommended_status": item["recommended_status"],
                "tier": item["tier"],
            }
            continue

        existing["blocked_urls"] = _dedupe(existing["blocked_urls"] + item["blocked_urls"])
        existing["recovered_urls"] = _dedupe(existing["recovered_urls"] + item["new_working_urls"])
        if item["recommended_url"]:
            existing["recommended_url"] = item["recommended_url"]
            existing["recommended_status"] = item["recommended_status"]
            existing["tier"] = item["tier"]

    merged_entries = sorted(merged_recovered.values(), key=lambda e: e["key"])
    recovered_registry_text = _render_recovered_registry(merged_entries, checked_at)
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out_plan.write_text(retry_plan_text, encoding="utf-8")
    out_recovered_registry.parent.mkdir(parents=True, exist_ok=True)
    out_recovered_registry.write_text(recovered_registry_text, encoding="utf-8")

    out_compact.parent.mkdir(parents=True, exist_ok=True)
    with out_compact.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact_id", "key", "tier", "recommended_url", "blocked_urls"])
        for item in summary_rows:
            writer.writerow(
                [
                    item["id"],
                    item["key"],
                    item["tier"],
                    item["recommended_url"],
                    "; ".join(item["blocked_urls"]),
                ]
            )

    high = sum(1 for r in summary_rows if r["tier"] == "high_recoverable")
    med = sum(1 for r in summary_rows if r["tier"] == "medium_recoverable")
    manual = sum(1 for r in summary_rows if r["tier"] == "manual_intervention")
    hard = sum(1 for r in summary_rows if r["tier"] == "hard_blocked")

    print(
        "Blocked recovery complete: "
        f"blocked={len(blocked)} high={high} medium={med} manual={manual} hard={hard}"
    )
    print(f"Wrote link audit TSV: {out_link_audit.relative_to(repo_root).as_posix()}")
    print(f"Wrote attempts TSV: {out_attempts.relative_to(repo_root).as_posix()}")
    print(f"Wrote retry plan: {out_plan.relative_to(repo_root).as_posix()}")
    print(f"Wrote compact blocked CSV: {out_compact.relative_to(repo_root).as_posix()}")
    print(
        "Wrote recovered mirror registry: "
        f"{out_recovered_registry.relative_to(repo_root).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
