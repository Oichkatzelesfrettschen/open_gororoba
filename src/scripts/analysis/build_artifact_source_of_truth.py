#!/usr/bin/env python3
"""
Build a single source-of-truth registry for cited/downloaded artifacts.

Inputs:
- registry/bibliography.toml
- registry/external_sources.toml
- registry/cayley_dickson_canonical_sources.toml (optional but preferred)
- data/external/intake/**/fetch_results*_normalized.tsv
- data/external/intake/**/mirror_retry_results*.tsv
- data/external/intake/**/pdf_success_added.tsv
- reports/cayley_dickson_source_recovery_2026_02_15.toml (optional)

Outputs:
- registry/artifact_source_of_truth.toml
- reports/artifact_source_of_truth_reconciliation_2026_02_15.toml
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


URL_RE = re.compile(r"^https?://", re.IGNORECASE)
URL_INLINE_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
BIB_ENTRY_RE = re.compile(
    r"@(?P<etype>[A-Za-z]+)\s*\{\s*(?P<key>[^,]+)\s*,(?P<body>.*?)\n\}\s*",
    re.DOTALL,
)

TITLE_KEYS = (
    "title",
    "paper_title",
    "name",
    "citation",
    "citation_markdown",
    "reference",
)
CITATION_KEYS = (
    "citation",
    "citation_markdown",
    "reference",
    "summary",
)
ID_KEYS = (
    "id",
    "key",
    "slug",
    "paper_id",
    "artifact_id",
)

# Keep extraction focused on bibliographic/source artifacts rather than every web URL in the repo.
REFERENCE_HOST_HINTS = {
    "arxiv.org",
    "export.arxiv.org",
    "scispace.com",
    "doi.org",
    "core.ac.uk",
    "sciencedirect.com",
    "linkinghub.elsevier.com",
    "msp.org",
    "projecteuclid.org",
    "mathnet.ru",
    "researchgate.net",
    "tandfonline.com",
    "mdpi.com",
    "link.springer.com",
    "springer.com",
    "degruyter.com",
    "cambridge.org",
    "iopscience.iop.org",
    "numdam.org",
    "aimspress.com",
    "dergipark.org.tr",
    "dr.lib.iastate.edu",
    "repository.essex.ac.uk",
    "openreview.net",
    "archive.org",
    "web.archive.org",
    "osf.io",
    "gutenberg.org",
    "jvoight.github.io",
    "kconrad.math.uconn.edu",
    "wstein.org",
    "journals.aps.org",
    "harvest.aps.org",
    "royalsocietypublishing.org",
    "zenodo.org",
    "isidore.co",
    "cms.math.ca",
    "bibliotekanauki.pl",
    "pldml.icm.edu.pl",
    "sciendo.com",
    "journals.sagepub.com",
    "pubmed.ncbi.nlm.nih.gov",
    "raw.githubusercontent.com",
}


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: non-ASCII output in {context}: {sample!r}")


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


def _slug(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or "unknown"


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


def _find_urls(text: str) -> list[str]:
    return [_normalize_url(m.group(0)) for m in URL_INLINE_RE.finditer(str(text))]


def _extract_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value.strip())
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                out.append(item.strip())
    return [s for s in out if s]


def _extract_urls(value: Any) -> list[str]:
    urls: list[str] = []
    for text in _extract_strings(value):
        normalized = _normalize_url(text)
        if URL_RE.match(normalized):
            urls.append(normalized)
        else:
            urls.extend(_find_urls(text))
    return _dedupe([u for u in urls if URL_RE.match(u)])


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_doi(doi: str) -> str:
    value = str(doi).strip()
    value = re.sub(r"^https?://doi\.org/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".,;)")
    value = value.lstrip("(")
    return value


def _extract_dois(value: Any) -> list[str]:
    out: list[str] = []
    for text in _extract_strings(value):
        cleaned = _normalize_doi(text)
        if DOI_RE.fullmatch(cleaned):
            out.append(cleaned)
            continue
        for match in DOI_RE.finditer(text):
            out.append(_normalize_doi(match.group(0)))
    return _dedupe(out)


def _looks_like_reference_url(url: str) -> bool:
    if not URL_RE.match(url):
        return False
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    for hint in REFERENCE_HOST_HINTS:
        if host == hint or host.endswith("." + hint):
            return True
    if host.endswith(".arxiv.org") or host.endswith(".scispace.com"):
        return True
    path_lower = (parsed.path or "").lower()
    if path_lower.endswith(".pdf"):
        return True
    if "/pdf" in path_lower:
        return True
    return False


def _extract_local_paths(value: Any, repo_root: Path) -> list[str]:
    out: list[str] = []
    for text in _extract_strings(value):
        candidate = text.strip()
        if not candidate:
            continue
        # Keep registry ASCII-safe; non-ASCII filenames cannot round-trip safely
        # through the ASCII-only output contract.
        if any(ord(ch) >= 128 for ch in candidate):
            continue
        if URL_RE.match(candidate):
            continue
        path = Path(candidate)
        if path.is_absolute():
            if path.exists():
                try:
                    out.append(path.relative_to(repo_root).as_posix())
                except ValueError:
                    out.append(path.as_posix())
            continue
        path_obj = repo_root / path
        if path_obj.exists():
            out.append(path.as_posix())
    return _dedupe(out)


def _doi_to_url(doi: str) -> str:
    return f"https://doi.org/{doi}"


def _doi_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host in {"doi.org", "dx.doi.org"}:
        return _normalize_doi(parsed.path.lstrip("/"))
    return ""


def _extract_dois_from_urls(urls: list[str]) -> list[str]:
    out: list[str] = []
    for url in urls:
        doi = _doi_from_url(url)
        if doi:
            out.append(doi)
    return _dedupe(out)


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append({str(k): str(v or "").strip() for k, v in row.items()})
    return rows


def _is_working_status(status: str) -> bool:
    return status in {"pdf_ok", "ok_nonpdf", "working_pdf", "working"}


def _derive_status(row: dict[str, str]) -> str:
    status = row.get("status", "").strip() or row.get("result", "").strip()
    if status:
        return status

    http_code = row.get("http_code", "").strip()
    is_pdf_raw = row.get("is_pdf", "").strip().lower()
    is_pdf = is_pdf_raw in {"yes", "true", "1"}
    if http_code.startswith("2") and is_pdf:
        return "pdf_ok"
    if http_code.startswith("2"):
        return "ok_nonpdf"
    if http_code:
        return f"http_{http_code}"
    return "unknown"


def _derive_pdf_flag(row: dict[str, str]) -> bool:
    raw = row.get("is_pdf", "").strip().lower()
    if raw in {"yes", "true", "1"}:
        return True
    if raw in {"no", "false", "0"}:
        return False
    # Fallback for mirror_retry rows.
    magic = row.get("magic", "")
    return magic.startswith("%PDF-")


@dataclass
class LinkObservation:
    url: str
    status: str
    source_table: str
    is_pdf: bool
    http_code: str


@dataclass
class CandidateRecord:
    source_kind: str
    source_ref: str
    title: str
    citation: str
    dois: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    local_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class UnifiedArtifact:
    key: str
    title: str = ""
    citation: str = ""
    source_kinds: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    doi_list: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    local_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    working_mirrors: list[str] = field(default_factory=list)
    working_pdf_mirrors: list[str] = field(default_factory=list)
    nonworking_mirrors: list[str] = field(default_factory=list)
    unverified_mirrors: list[str] = field(default_factory=list)
    downloaded_paths: list[str] = field(default_factory=list)
    canonical_functional_url: str = ""
    canonical_download_path: str = ""
    status: str = ""
    minimum_requirement_met: bool = False
    manual_intervention_required: bool = False
    manual_intervention_reason: str = ""


def collect_link_observations(repo_root: Path) -> tuple[dict[str, list[LinkObservation]], list[str]]:
    obs: dict[str, list[LinkObservation]] = {}
    table_paths: list[Path] = []
    intake_root = repo_root / "data" / "external" / "intake"
    if intake_root.exists():
        table_paths.extend(intake_root.glob("**/fetch_results*_normalized.tsv"))
        table_paths.extend(intake_root.glob("**/mirror_retry_results*.tsv"))
        table_paths.extend(intake_root.glob("**/link_audit_results*.tsv"))
    unique_paths = sorted({p.resolve() for p in table_paths})

    source_tables: list[str] = []
    for path in unique_paths:
        rel = path.relative_to(repo_root).as_posix()
        source_tables.append(rel)
        rows = _read_tsv(path)
        for row in rows:
            url = _normalize_url(row.get("url", ""))
            if not URL_RE.match(url):
                continue
            status = _derive_status(row)
            http_code = row.get("http_code", "").strip()
            is_pdf = _derive_pdf_flag(row)
            record = LinkObservation(
                url=url,
                status=status,
                source_table=rel,
                is_pdf=is_pdf,
                http_code=http_code,
            )
            obs.setdefault(url, []).append(record)

            # Track effective URL as equivalent observation when present.
            effective = _normalize_url(row.get("url_effective", ""))
            if URL_RE.match(effective) and effective != url:
                eff_record = LinkObservation(
                    url=effective,
                    status=status,
                    source_table=rel,
                    is_pdf=is_pdf,
                    http_code=http_code,
                )
                obs.setdefault(effective, []).append(eff_record)

    return obs, source_tables


def collect_download_map(repo_root: Path) -> dict[str, list[str]]:
    url_to_paths: dict[str, list[str]] = {}

    # Map from pdf_success_added tables.
    intake_root = repo_root / "data" / "external" / "intake"
    if intake_root.exists():
        for table in sorted(intake_root.glob("**/pdf_success_added.tsv")):
            rows = _read_tsv(table)
            pdf_dir = table.parent / "pdf_success"
            for row in rows:
                source_url = row.get("source_url", "").strip()
                source_url = _normalize_url(source_url)
                name = row.get("canonical_pdf_name", "").strip()
                if not (URL_RE.match(source_url) and name):
                    continue
                candidate = pdf_dir / name
                if candidate.exists():
                    rel = candidate.relative_to(repo_root).as_posix()
                    url_to_paths.setdefault(source_url, []).append(rel)

    # Map from Cayley-Dickson canonical source registry.
    cdcs_path = repo_root / "registry" / "cayley_dickson_canonical_sources.toml"
    if cdcs_path.exists():
        data = _load_toml(cdcs_path)
        for paper in data.get("paper", []):
            path = str(paper.get("canonical_pdf_path", "")).strip()
            url = _normalize_url(str(paper.get("canonical_functional_url", "")))
            if not path:
                continue
            candidate = repo_root / path
            if not candidate.exists():
                continue
            rel = candidate.relative_to(repo_root).as_posix()
            if URL_RE.match(url):
                url_to_paths.setdefault(url, []).append(rel)
            for mirror in paper.get("working_pdf_mirrors", []):
                mirror_url = str(mirror).strip()
                mirror_url = _normalize_url(mirror_url)
                if URL_RE.match(mirror_url):
                    url_to_paths.setdefault(mirror_url, []).append(rel)

    # Map from Brown recovery report.
    brown_report = repo_root / "reports" / "cayley_dickson_source_recovery_2026_02_15.toml"
    if brown_report.exists():
        data = _load_toml(brown_report)
        brown = data.get("brown_1972", {})
        path = str(brown.get("canonical_pdf_path", "")).strip()
        url = _normalize_url(str(brown.get("core_download_url", "")))
        if path and url:
            candidate = repo_root / path
            if candidate.exists() and URL_RE.match(url):
                rel = candidate.relative_to(repo_root).as_posix()
                url_to_paths.setdefault(url, []).append(rel)

    for url, paths in list(url_to_paths.items()):
        url_to_paths[url] = _dedupe(paths)

    return url_to_paths


def discover_candidate_source_files(repo_root: Path) -> list[Path]:
    suffixes = {".toml", ".bib", ".bibtex", ".md", ".txt", ".rst"}
    text_suffixes = {".md", ".txt", ".rst"}
    text_keywords = (
        "source",
        "bibli",
        "reconcil",
        "artifact",
        "intake",
        "cayley",
        "sedenion",
        "octonion",
        "quaternion",
        "mirror",
        "provenance",
    )
    allowed_prefixes = (
        "registry/",
        "reports/",
        "docs/",
        "papers/",
        "data/papers/",
    )
    excluded_prefixes = (
        ".git/",
        "target/",
        "data/external/intake/",
        "data/external/raw/",
        "data/external/cache/",
    )
    excluded_exact = {
        "registry/artifact_source_of_truth.toml",
        "reports/artifact_source_of_truth_reconciliation_2026_02_15.toml",
        "reports/artifact_blocked_links_2026_02_15.tsv",
        "reports/artifact_missing_minimum_2026_02_15.tsv",
    }

    paths: set[Path] = set()
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        rel = path.relative_to(repo_root).as_posix()
        if rel != "refs.bib" and not rel.startswith(allowed_prefixes):
            continue
        if rel in excluded_exact:
            continue
        if rel.startswith(excluded_prefixes):
            continue
        if path.suffix.lower() in text_suffixes:
            lowered = rel.lower()
            if not any(token in lowered for token in text_keywords):
                continue
        paths.add(path.resolve())
    return sorted(paths)


def _pick_first_str(node: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_candidates_from_toml_node(
    repo_root: Path,
    source_rel: str,
    node: Any,
    breadcrumbs: list[str],
) -> list[CandidateRecord]:
    out: list[CandidateRecord] = []
    if isinstance(node, list):
        for idx, item in enumerate(node):
            out.extend(
                _extract_candidates_from_toml_node(
                    repo_root=repo_root,
                    source_rel=source_rel,
                    node=item,
                    breadcrumbs=breadcrumbs + [str(idx)],
                )
            )
        return out

    if not isinstance(node, dict):
        return out

    title = _pick_first_str(node, TITLE_KEYS)
    citation = _pick_first_str(node, CITATION_KEYS) or title
    ref_hint = _pick_first_str(node, ID_KEYS)
    source_ref = f"{source_rel}::{'/'.join(breadcrumbs)}"
    if ref_hint:
        source_ref += f"::{ref_hint}"

    urls: list[str] = []
    dois: list[str] = []
    local_paths: list[str] = []
    notes: list[str] = []

    for key, value in node.items():
        key_l = str(key).lower()
        if "url" in key_l or "link" in key_l or "mirror" in key_l or "href" in key_l:
            urls.extend(_extract_urls(value))
        elif "doi" in key_l:
            dois.extend(_extract_dois(value))
        elif (
            "path" in key_l
            or key_l.endswith("_file")
            or key_l.endswith("_files")
            or key_l == "files"
        ):
            local_paths.extend(_extract_local_paths(value, repo_root))
        elif key_l in {"status", "note", "notes", "reason", "manual_intervention_reason"}:
            notes.extend(_extract_strings(value))

    filtered_urls = [url for url in _dedupe(urls) if _looks_like_reference_url(url)]
    dois = _dedupe(dois)
    local_paths = _dedupe(local_paths)
    notes = _dedupe(notes)

    if filtered_urls or dois or local_paths:
        if not title:
            if citation:
                title = citation
            elif filtered_urls:
                title = filtered_urls[0]
            elif dois:
                title = dois[0]
            else:
                title = source_ref
        out.append(
            CandidateRecord(
                source_kind="toml_source",
                source_ref=source_ref,
                title=title,
                citation=citation or title,
                dois=dois,
                links=filtered_urls + [_doi_to_url(d) for d in dois],
                local_paths=local_paths,
                notes=notes,
            )
        )

    for key, value in node.items():
        if isinstance(value, (dict, list)):
            out.extend(
                _extract_candidates_from_toml_node(
                    repo_root=repo_root,
                    source_rel=source_rel,
                    node=value,
                    breadcrumbs=breadcrumbs + [str(key)],
                )
            )
    return out


def _extract_bib_field(body: str, field: str) -> str:
    pattern_brace = re.compile(
        rf"{re.escape(field)}\s*=\s*\{{(?P<value>.*?)\}}",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern_brace.search(body)
    if m:
        return m.group("value").strip()
    pattern_quote = re.compile(
        rf'{re.escape(field)}\s*=\s*"(?P<value>.*?)"',
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern_quote.search(body)
    if m:
        return m.group("value").strip()
    return ""


def extract_candidates_from_bib_file(repo_root: Path, path: Path) -> list[CandidateRecord]:
    rel = path.relative_to(repo_root).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[CandidateRecord] = []
    for match in BIB_ENTRY_RE.finditer(text):
        etype = match.group("etype").strip()
        key = match.group("key").strip()
        body = match.group("body")
        title = _extract_bib_field(body, "title")
        citation = f"@{etype}{{{key}}}"
        urls = _extract_urls(body)
        dois = _extract_dois(_extract_bib_field(body, "doi"))
        if not dois:
            dois = _extract_dois(body)

        if not any(_looks_like_reference_url(url) for url in urls) and not dois:
            continue

        urls = [url for url in _dedupe(urls) if _looks_like_reference_url(url)]
        for doi in dois:
            urls.append(_doi_to_url(doi))
        urls = _dedupe(urls)

        out.append(
            CandidateRecord(
                source_kind="bibtex_entry",
                source_ref=f"{rel}::{key}",
                title=title or key,
                citation=citation,
                dois=_dedupe(dois),
                links=urls,
            )
        )
    return out


def _clean_line_title(line: str) -> str:
    title = line.strip()
    title = re.sub(r"https?://[^\s<>()\"']+", "", title, flags=re.IGNORECASE)
    title = title.strip(" -*|`[]()")
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def extract_candidates_from_text_file(repo_root: Path, path: Path) -> list[CandidateRecord]:
    rel = path.relative_to(repo_root).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[CandidateRecord] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        urls = [u for u in _find_urls(raw_line) if _looks_like_reference_url(u)]
        dois = _extract_dois(raw_line)
        if not urls and not dois:
            continue

        source_ref = f"{rel}:{line_no}"
        title = _clean_line_title(raw_line)
        if not title:
            title = urls[0] if urls else dois[0]
        citation = title
        links = list(urls)
        links.extend(_doi_to_url(doi) for doi in dois)
        out.append(
            CandidateRecord(
                source_kind="text_reference",
                source_ref=source_ref,
                title=title,
                citation=citation,
                dois=_dedupe(dois),
                links=_dedupe(links),
            )
        )
    return out


def extract_candidates_from_source_file(repo_root: Path, path: Path) -> list[CandidateRecord]:
    rel = path.relative_to(repo_root).as_posix()
    if path.suffix.lower() in {".bib", ".bibtex"}:
        return extract_candidates_from_bib_file(repo_root, path)

    if path.suffix.lower() == ".toml":
        try:
            data = _load_toml(path)
        except Exception:
            return []
        return _extract_candidates_from_toml_node(
            repo_root=repo_root,
            source_rel=rel,
            node=data,
            breadcrumbs=[],
        )

    if path.suffix.lower() in {".md", ".txt", ".rst"}:
        return extract_candidates_from_text_file(repo_root, path)

    return []


def build_candidates(repo_root: Path) -> tuple[list[CandidateRecord], list[str]]:
    candidates: list[CandidateRecord] = []

    bibliography_path = repo_root / "registry" / "bibliography.toml"
    if bibliography_path.exists():
        bib = _load_toml(bibliography_path)
        for entry in bib.get("entry", []):
            entry_id = str(entry.get("id", "")).strip()
            citation = str(entry.get("citation_markdown", "")).strip()
            title = citation
            links = [
                _normalize_url(str(u))
                for u in entry.get("urls", [])
                if _normalize_url(str(u))
            ]
            dois = [_normalize_doi(d) for d in entry.get("dois", []) if _normalize_doi(d)]
            for doi in dois:
                links.append(_doi_to_url(doi))
            note_list = [str(n).strip() for n in entry.get("notes", []) if str(n).strip()]
            candidates.append(
                CandidateRecord(
                    source_kind="bibliography_entry",
                    source_ref=entry_id or "BIB-UNKNOWN",
                    title=title,
                    citation=citation,
                    dois=_dedupe(dois),
                    links=_dedupe(links),
                    notes=note_list,
                )
            )

    external_sources_path = repo_root / "registry" / "external_sources.toml"
    if external_sources_path.exists():
        ext = _load_toml(external_sources_path)
        for doc in ext.get("document", []):
            doc_id = str(doc.get("id", "")).strip()
            title = str(doc.get("title", "")).strip()
            links = [
                _normalize_url(str(u))
                for u in doc.get("url_refs", [])
                if _normalize_url(str(u))
            ]
            path_refs = [str(p).strip() for p in doc.get("path_refs", []) if str(p).strip()]
            existing_paths: list[str] = []
            for path_ref in path_refs:
                path_obj = repo_root / path_ref
                if path_obj.exists():
                    existing_paths.append(path_ref)
            notes = [str(doc.get("notes", "")).strip()] if str(doc.get("notes", "")).strip() else []
            candidates.append(
                CandidateRecord(
                    source_kind="external_source_document",
                    source_ref=doc_id or "XS-UNKNOWN",
                    title=title,
                    citation=title,
                    links=_dedupe(links),
                    local_paths=_dedupe(existing_paths),
                    notes=notes,
                )
            )

    cdcs_path = repo_root / "registry" / "cayley_dickson_canonical_sources.toml"
    if cdcs_path.exists():
        cdcs = _load_toml(cdcs_path)
        for paper in cdcs.get("paper", []):
            key = str(paper.get("key", "")).strip() or "CDCS-UNKNOWN"
            title = str(paper.get("title", "")).strip()
            doi = _normalize_doi(str(paper.get("doi", "")).strip())
            links = []
            links.extend(
                [
                    _normalize_url(str(u))
                    for u in paper.get("working_mirrors", [])
                    if _normalize_url(str(u))
                ]
            )
            links.extend(
                [
                    _normalize_url(str(u))
                    for u in paper.get("working_pdf_mirrors", [])
                    if _normalize_url(str(u))
                ]
            )
            links.extend(
                [
                    _normalize_url(str(u))
                    for u in paper.get("nonworking_mirrors", [])
                    if _normalize_url(str(u))
                ]
            )
            links.extend(
                [
                    _normalize_url(str(u))
                    for u in paper.get("manual_intervention_urls", [])
                    if _normalize_url(str(u))
                ]
            )
            canonical_url = _normalize_url(str(paper.get("canonical_functional_url", "")))
            if canonical_url:
                links.append(canonical_url)
            if doi:
                links.append(_doi_to_url(doi))
            local_paths = []
            canonical_pdf_path = str(paper.get("canonical_pdf_path", "")).strip()
            if canonical_pdf_path and (repo_root / canonical_pdf_path).exists():
                local_paths.append(canonical_pdf_path)
            note_parts = []
            status = str(paper.get("status", "")).strip()
            if status:
                note_parts.append(f"status={status}")
            reason = str(paper.get("manual_intervention_reason", "")).strip()
            if reason:
                note_parts.append(reason)
            candidates.append(
                CandidateRecord(
                    source_kind="canonical_cayley_dickson",
                    source_ref=key,
                    title=title,
                    citation=title,
                    dois=[doi] if doi else [],
                    links=_dedupe(links),
                    local_paths=_dedupe(local_paths),
                    notes=_dedupe(note_parts),
                )
            )

    discovered_files = discover_candidate_source_files(repo_root)
    # Expand beyond hand-curated files using scoped repo-wide source-file discovery.
    for source_file in discovered_files:
        candidates.extend(extract_candidates_from_source_file(repo_root, source_file))

    # Normalize DOI identity by extracting DOI values embedded in doi.org links.
    for cand in candidates:
        link_dois = _extract_dois_from_urls(cand.links)
        if link_dois:
            cand.dois = _dedupe(cand.dois + link_dois)
            cand.links = _dedupe(cand.links + [_doi_to_url(d) for d in link_dois])

    source_files = [p.relative_to(repo_root).as_posix() for p in discovered_files]
    return candidates, source_files


def _identity_key(candidate: CandidateRecord) -> str:
    if candidate.dois:
        return f"doi:{candidate.dois[0].lower()}"
    if candidate.links:
        return f"url:{candidate.links[0].lower()}"
    if candidate.title:
        return f"title:{_slug(candidate.title)}"
    return f"source:{candidate.source_kind}:{_slug(candidate.source_ref)}"


def unify_candidates(candidates: list[CandidateRecord]) -> list[UnifiedArtifact]:
    merged: dict[str, UnifiedArtifact] = {}

    for cand in candidates:
        key = _identity_key(cand)
        item = merged.get(key)
        if item is None:
            item = UnifiedArtifact(key=key)
            merged[key] = item
        if cand.title and not item.title:
            item.title = cand.title
        if cand.citation and not item.citation:
            item.citation = cand.citation
        item.source_kinds.extend([cand.source_kind])
        item.source_refs.extend([cand.source_ref])
        item.doi_list.extend(cand.dois)
        item.links.extend(cand.links)
        item.local_paths.extend(cand.local_paths)
        item.notes.extend(cand.notes)

    out = list(merged.values())
    for item in out:
        item.source_kinds = _dedupe(item.source_kinds)
        item.source_refs = _dedupe(item.source_refs)
        item.doi_list = _dedupe(item.doi_list)
        item.links = _dedupe(item.links)
        item.local_paths = _dedupe(item.local_paths)
        item.notes = _dedupe(item.notes)
    return sorted(out, key=lambda x: x.key)


def classify_artifacts(
    artifacts: list[UnifiedArtifact],
    observations: dict[str, list[LinkObservation]],
    download_map: dict[str, list[str]],
) -> None:
    for art in artifacts:
        working: list[str] = []
        working_pdf: list[str] = []
        nonworking: list[str] = []
        unverified: list[str] = []
        downloaded: list[str] = list(art.local_paths)

        for url in art.links:
            obs_list = observations.get(url, [])
            statuses = [o.status for o in obs_list]
            has_pdf_ok = any(s == "pdf_ok" for s in statuses)
            has_ok = any(s == "ok_nonpdf" for s in statuses)
            has_nonworking = any(
                (s.startswith("http_") and s not in {"http_200", "http_201", "http_202", "http_203", "http_204"})
                or s in {"failed"}
                for s in statuses
            )

            if has_pdf_ok:
                working.append(url)
                working_pdf.append(url)
            elif has_ok:
                working.append(url)
            elif has_nonworking:
                nonworking.append(url)
            elif obs_list:
                # Unknown, but observed.
                unverified.append(url)
            else:
                unverified.append(url)

            for path in download_map.get(url, []):
                downloaded.append(path)

        art.working_mirrors = _dedupe(working)
        art.working_pdf_mirrors = _dedupe(working_pdf)
        art.nonworking_mirrors = _dedupe(nonworking)
        art.unverified_mirrors = _dedupe(unverified)
        art.downloaded_paths = _dedupe(downloaded)

        art.minimum_requirement_met = bool(art.working_mirrors or art.downloaded_paths)
        art.manual_intervention_required = bool(art.links and not art.minimum_requirement_met)

        if art.downloaded_paths:
            art.status = "downloaded"
        elif art.working_mirrors:
            art.status = "downloadable"
        elif not art.links:
            art.status = "citation_only_no_link"
        elif art.nonworking_mirrors and not art.working_mirrors:
            art.status = "blocked"
        else:
            art.status = "unverified"

        if art.working_pdf_mirrors:
            art.canonical_functional_url = art.working_pdf_mirrors[0]
        elif art.working_mirrors:
            art.canonical_functional_url = art.working_mirrors[0]
        elif art.links:
            art.canonical_functional_url = art.links[0]
        else:
            art.canonical_functional_url = ""

        art.canonical_download_path = art.downloaded_paths[0] if art.downloaded_paths else ""
        if art.manual_intervention_required:
            art.manual_intervention_reason = (
                "No working mirror observed from current fetch/retry ledgers; manual link intervention required."
            )
        else:
            art.manual_intervention_reason = ""


def render_artifact_registry(
    artifacts: list[UnifiedArtifact],
    source_tables: list[str],
    source_files: list[str],
    now: str,
) -> str:
    total = len(artifacts)
    downloaded = sum(1 for a in artifacts if a.status == "downloaded")
    downloadable = sum(1 for a in artifacts if a.status == "downloadable")
    blocked = sum(1 for a in artifacts if a.status == "blocked")
    citation_only = sum(1 for a in artifacts if a.status == "citation_only_no_link")
    unverified = sum(1 for a in artifacts if a.status == "unverified")
    missing_min = sum(1 for a in artifacts if not a.minimum_requirement_met)
    manual = sum(1 for a in artifacts if a.manual_intervention_required)

    lines: list[str] = []
    lines.append("# Single source-of-truth registry for cited artifacts and mirror status.")
    lines.append("")
    lines.append("[artifact_source_of_truth]")
    lines.append('id = "ASOT-2026-02-15"')
    lines.append(f"updated = {_escape_toml(now)}")
    lines.append("authoritative = true")
    lines.append(
        'policy = "Keep one working mirror minimum per artifact; retain working mirrors and non-working mirrors for manual intervention."'
    )
    lines.append(
        'minimum_requirement = "1 paper/document/artifact => >= 1 working mirror or downloaded local artifact."'
    )
    lines.append(f"source_table_count = {len(source_tables)}")
    lines.append(f"source_tables = {_render_list(source_tables)}")
    lines.append(f"source_file_count = {len(source_files)}")
    lines.append(f"source_files = {_render_list(source_files)}")
    lines.append(f"artifact_count = {total}")
    lines.append(f"downloaded_count = {downloaded}")
    lines.append(f"downloadable_count = {downloadable}")
    lines.append(f"blocked_count = {blocked}")
    lines.append(f"citation_only_no_link_count = {citation_only}")
    lines.append(f"unverified_count = {unverified}")
    lines.append(f"missing_minimum_requirement_count = {missing_min}")
    lines.append(f"manual_intervention_required_count = {manual}")
    lines.append("")

    missing_keys = [a.key for a in artifacts if not a.minimum_requirement_met]
    lines.append("[coverage]")
    lines.append(f"artifacts_without_working_mirror = {_render_list(missing_keys)}")
    lines.append(f"artifacts_without_working_mirror_count = {len(missing_keys)}")
    lines.append("")

    for idx, art in enumerate(artifacts, start=1):
        lines.append("[[artifact]]")
        lines.append(f"id = {_escape_toml(f'ASOT-{idx:04d}')}")
        lines.append(f"key = {_escape_toml(art.key)}")
        lines.append(f"title = {_escape_toml(art.title)}")
        lines.append(f"citation = {_escape_toml(art.citation)}")
        lines.append(f"source_kinds = {_render_list(art.source_kinds)}")
        lines.append(f"source_refs = {_render_list(art.source_refs)}")
        lines.append(f"doi_list = {_render_list(art.doi_list)}")
        lines.append(f"canonical_functional_url = {_escape_toml(art.canonical_functional_url)}")
        lines.append(f"canonical_download_path = {_escape_toml(art.canonical_download_path)}")
        lines.append(f"status = {_escape_toml(art.status)}")
        lines.append(f"minimum_requirement_met = {str(art.minimum_requirement_met).lower()}")
        lines.append(f"manual_intervention_required = {str(art.manual_intervention_required).lower()}")
        lines.append(f"manual_intervention_reason = {_escape_toml(art.manual_intervention_reason)}")
        lines.append(f"working_mirror_count = {len(art.working_mirrors)}")
        lines.append(f"working_pdf_mirror_count = {len(art.working_pdf_mirrors)}")
        lines.append(f"nonworking_mirror_count = {len(art.nonworking_mirrors)}")
        lines.append(f"unverified_mirror_count = {len(art.unverified_mirrors)}")
        lines.append(f"downloaded_path_count = {len(art.downloaded_paths)}")
        lines.append(f"all_links = {_render_list(art.links)}")
        lines.append(f"working_mirrors = {_render_list(art.working_mirrors)}")
        lines.append(f"working_pdf_mirrors = {_render_list(art.working_pdf_mirrors)}")
        lines.append(f"nonworking_mirrors = {_render_list(art.nonworking_mirrors)}")
        lines.append(f"unverified_mirrors = {_render_list(art.unverified_mirrors)}")
        lines.append(f"downloaded_paths = {_render_list(art.downloaded_paths)}")
        lines.append(f"notes = {_render_list(art.notes)}")
        lines.append("")

    return "\n".join(lines)


def render_reconciliation_report(artifacts: list[UnifiedArtifact], now: str) -> str:
    total = len(artifacts)
    downloaded = sum(1 for a in artifacts if a.status == "downloaded")
    downloadable = sum(1 for a in artifacts if a.status == "downloadable")
    blocked = sum(1 for a in artifacts if a.status == "blocked")
    citation_only = sum(1 for a in artifacts if a.status == "citation_only_no_link")
    unverified = sum(1 for a in artifacts if a.status == "unverified")
    missing = [a for a in artifacts if not a.minimum_requirement_met]

    lines: list[str] = []
    lines.append("# Reconciliation summary for artifact_source_of_truth.toml")
    lines.append("")
    lines.append("[report]")
    lines.append('id = "ASOT-RECON-2026-02-15"')
    lines.append(f"updated = {_escape_toml(now)}")
    lines.append("authoritative = true")
    lines.append(f"artifact_count = {total}")
    lines.append(f"downloaded_count = {downloaded}")
    lines.append(f"downloadable_count = {downloadable}")
    lines.append(f"blocked_count = {blocked}")
    lines.append(f"citation_only_no_link_count = {citation_only}")
    lines.append(f"unverified_count = {unverified}")
    lines.append(f"missing_minimum_requirement_count = {len(missing)}")
    lines.append("")

    for art in missing:
        lines.append("[[missing_minimum_requirement]]")
        lines.append(f"key = {_escape_toml(art.key)}")
        lines.append(f"title = {_escape_toml(art.title)}")
        lines.append(f"status = {_escape_toml(art.status)}")
        lines.append(f"source_refs = {_render_list(art.source_refs)}")
        lines.append(f"all_links = {_render_list(art.links)}")
        lines.append(f"nonworking_mirrors = {_render_list(art.nonworking_mirrors)}")
        lines.append(f"unverified_mirrors = {_render_list(art.unverified_mirrors)}")
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
        "--out-registry",
        default="registry/artifact_source_of_truth.toml",
        help="Output registry path.",
    )
    parser.add_argument(
        "--out-report",
        default="reports/artifact_source_of_truth_reconciliation_2026_02_15.toml",
        help="Output reconciliation report path.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_registry = repo_root / args.out_registry
    out_report = repo_root / args.out_report
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    observations, source_tables = collect_link_observations(repo_root)
    download_map = collect_download_map(repo_root)
    candidates, source_files = build_candidates(repo_root)
    artifacts = unify_candidates(candidates)
    classify_artifacts(artifacts, observations, download_map)

    registry_text = render_artifact_registry(artifacts, source_tables, source_files, now)
    report_text = render_reconciliation_report(artifacts, now)
    _assert_ascii(registry_text, str(out_registry))
    _assert_ascii(report_text, str(out_report))

    out_registry.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_registry.write_text(registry_text, encoding="utf-8")
    out_report.write_text(report_text, encoding="utf-8")

    print(
        "Wrote artifact source-of-truth registry: "
        f"{out_registry.relative_to(repo_root).as_posix()} "
        f"artifacts={len(artifacts)}"
    )
    print(
        "Wrote reconciliation report: "
        f"{out_report.relative_to(repo_root).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
