#!/usr/bin/env python3
"""
Build Wave 5 Batch 2 strict TOML registries:
- W5-011: registry/knowledge/derivation_steps.toml
- W5-012: registry/bibliography_normalized.toml
- W5-013: registry/provenance_sources.toml
- W5-014: registry/narrative_paragraph_atoms.toml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from collections import defaultdict
from pathlib import Path


CLAIM_ID_RE = re.compile(r"\bC-\d{3}\b")
EQUATION_ID_RE = re.compile(r"\bEQA2?-\d{4,5}\b")
SYMBOL_ID_RE = re.compile(r"\bSYM-\d{4}\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
URL_RE = re.compile(r"https?://[^\s)]+")
ARXIV_RE = re.compile(r"\barXiv[:\s]*([A-Za-z\-]+\/\d{7}|\d{4}\.\d{4,5}(?:v\d+)?)\b", re.I)
YEAR_RE = re.compile(r"\((\d{4})\)")
SHA_RE = re.compile(r"\b([a-fA-F0-9]{64})\b")
DATE_RE = re.compile(r"(?:Date retrieved|retrieved)\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", re.I)
BACKTICK_RE = re.compile(r"`([^`]+)`")


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _q(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _render_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_q(v) for v in values) + "]"


def _ascii_clean(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch in {"\n", "\r", "\t"}:
            out.append(ch)
        elif code < 32:
            out.append(" ")
        elif code <= 127:
            out.append(ch)
        else:
            out.append(f"\\u{code:04X}")
    return "".join(out)


def _collapse(text: str) -> str:
    return " ".join(_ascii_clean(text).split())


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: non-ASCII output in {context}: {sample!r}")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _normalize_claim_refs(values: list[str], claim_id: str) -> list[str]:
    refs = {item for item in values if CLAIM_ID_RE.fullmatch(item)}
    if CLAIM_ID_RE.fullmatch(claim_id):
        refs.add(claim_id)
    return sorted(refs)


def _classify_step(text: str) -> str:
    lower = text.lower()
    if "h0" in lower or "h1" in lower or "assumption" in lower:
        return "assumption"
    if "decision rule" in lower or "decision:" in lower:
        return "decision_rule"
    if "required evidence" in lower or "evidence" in lower:
        return "evidence_requirement"
    if "counterexample" in lower or "exhibit" in lower:
        return "witness_construction"
    if "status" in lower or "verified" in lower or "refuted" in lower or "rejected" in lower:
        return "status_update"
    if "therefore" in lower or "thus" in lower or "hence" in lower:
        return "conclusion"
    return "derivation_step"


def _split_derivation_chunks(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        text = _ascii_clean(str(raw)).replace("\r", "")
        for chunk in text.split("||"):
            line = _collapse(chunk).strip()
            line = line.lstrip("-").strip()
            if line:
                out.append(line)
    deduped: list[str] = []
    prev = ""
    for item in out:
        if item == prev:
            continue
        deduped.append(item)
        prev = item
    return deduped


def build_derivation_steps(proof_rows: list[dict]) -> tuple[list[dict], dict]:
    steps: list[dict] = []
    seq = 0
    max_per_skeleton = 0
    linked_claim_steps = 0
    skeleton_with_steps = 0
    for row in sorted(proof_rows, key=lambda item: str(item.get("id", ""))):
        skeleton_id = _collapse(str(row.get("id", "")))
        if not skeleton_id:
            continue
        claim_id = _collapse(str(row.get("claim_id", "")))
        claim_refs = [str(v) for v in row.get("claim_refs", []) if isinstance(v, str)]
        claim_refs = _normalize_claim_refs(claim_refs, claim_id)
        parts = _split_derivation_chunks([str(v) for v in row.get("derivation_steps", [])])
        if not parts:
            fallback: list[str] = []
            fallback.extend([_collapse(str(v)) for v in row.get("assumptions", []) if str(v).strip()])
            fallback.extend([_collapse(str(v)) for v in row.get("obligations", []) if str(v).strip()])
            decision = _collapse(str(row.get("decision_rule", "")))
            if decision:
                fallback.append(decision)
            conclusion = _collapse(str(row.get("conclusion", "")))
            if conclusion:
                fallback.append(conclusion)
            parts = [item for item in fallback if item]
        if not parts:
            continue
        skeleton_with_steps += 1
        max_per_skeleton = max(max_per_skeleton, len(parts))
        prev_step_id = ""
        for idx, text in enumerate(parts, start=1):
            seq += 1
            step_id = f"DS-{seq:06d}"
            equation_refs = sorted(set(EQUATION_ID_RE.findall(text)))
            symbol_refs = sorted(set(SYMBOL_ID_RE.findall(text)))
            number_refs = sorted(set(NUMBER_RE.findall(text)))
            token_pool = [
                token
                for token in IDENTIFIER_RE.findall(text)
                if token.lower() not in STOPWORDS and len(token) >= 2
            ]
            key_tokens = sorted(dict.fromkeys(token_pool))[:20]
            if claim_refs:
                linked_claim_steps += 1
            steps.append(
                {
                    "id": step_id,
                    "skeleton_id": skeleton_id,
                    "skeleton_kind": _collapse(str(row.get("skeleton_kind", ""))),
                    "source_path": _collapse(str(row.get("source_path", ""))),
                    "source_uid": _collapse(str(row.get("source_uid", ""))),
                    "claim_id": claim_id,
                    "claim_refs": claim_refs,
                    "step_index": idx,
                    "step_kind": _classify_step(text),
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "equation_refs": equation_refs,
                    "symbol_refs": symbol_refs,
                    "numeric_constants": number_refs,
                    "key_tokens": key_tokens,
                    "depends_on_step_ids": [prev_step_id] if prev_step_id else [],
                    "line_start": int(row.get("line_start", 0) or 0),
                    "line_end": int(row.get("line_end", 0) or 0),
                }
            )
            prev_step_id = step_id
    meta = {
        "step_count": len(steps),
        "skeleton_count": len(proof_rows),
        "skeleton_with_steps_count": skeleton_with_steps,
        "claim_linked_step_count": linked_claim_steps,
        "max_steps_per_skeleton": max_per_skeleton,
    }
    return steps, meta


def _extract_authors(citation: str) -> list[str]:
    citation = citation.strip()
    author_blob = ""
    bold_match = re.match(r"^\*\*(.+?)\.\*\*", citation)
    if bold_match:
        author_blob = bold_match.group(1)
    else:
        prefix = citation.split("(", 1)[0]
        author_blob = prefix.strip(" .")
    author_blob = author_blob.replace("&", " and ")
    parts = [p.strip(" .") for p in re.split(r"\band\b|,|;", author_blob) if p.strip(" .")]
    cleaned = [_collapse(p) for p in parts if _collapse(p)]
    return cleaned


def _extract_title(citation: str) -> tuple[str, str]:
    italic = re.search(r"\)\.\s*\*(.+?)\*\.", citation)
    if italic:
        return _collapse(italic.group(1)), citation[italic.end() :]
    quoted = re.search(r'\)\.\s*"([^"]+)"\.', citation)
    if quoted:
        return _collapse(quoted.group(1)), citation[quoted.end() :]
    fallback = citation
    after_year = re.search(r"\)\.\s*(.+)", citation)
    if after_year:
        fallback = after_year.group(1)
    first_stop = fallback.find("[")
    if first_stop >= 0:
        fallback = fallback[:first_stop]
    return _collapse(fallback.strip(" .")), ""


def _extract_venue(tail: str) -> str:
    if not tail:
        return ""
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", "", tail)
    text = URL_RE.sub("", text)
    text = text.strip(" .;:")
    return _collapse(text)


def _document_type(citation: str, arxiv_id: str, doi_list: list[str], urls: list[str]) -> str:
    lower = citation.lower()
    if arxiv_id:
        return "arxiv_preprint"
    if doi_list:
        return "doi_reference"
    if any(url.endswith(".csv") or "zenodo" in url for url in urls):
        return "dataset_reference"
    if "collaboration" in lower:
        return "collaboration_report"
    if "journal" in lower or "rev." in lower or "phys." in lower:
        return "journal_article"
    if "proceedings" in lower or "conference" in lower:
        return "conference_paper"
    return "general_reference"


def _relevance_score(citation: str, notes: list[str]) -> int:
    corpus = (citation + " " + " ".join(notes)).lower()
    score = 0
    for token in ("source", "dataset", "evidence", "verify", "verification", "reproducible", "hash"):
        if token in corpus:
            score += 1
    return min(score, 7)


def build_bibliography_normalized(entries: list[dict]) -> tuple[list[dict], dict]:
    out: list[dict] = []
    total_warnings = 0
    for idx, row in enumerate(sorted(entries, key=lambda item: int(item.get("order_index", 0))), start=1):
        citation = _collapse(str(row.get("citation_markdown", "")))
        notes = [_collapse(str(item)) for item in row.get("notes", []) if _collapse(str(item))]
        urls = sorted(set(str(item) for item in row.get("urls", []) if str(item).strip()))
        doi_list = sorted(set(str(item) for item in row.get("dois", []) if str(item).strip()))
        authors = _extract_authors(citation)
        year_match = YEAR_RE.search(citation)
        publication_year = int(year_match.group(1)) if year_match else 0
        title, tail = _extract_title(citation)
        venue = _extract_venue(tail)
        arxiv_match = ARXIV_RE.search(citation)
        arxiv_id = arxiv_match.group(1) if arxiv_match else ""
        warnings: list[str] = []
        if not authors:
            warnings.append("missing_authors")
        if publication_year == 0:
            warnings.append("missing_year")
        if not title:
            warnings.append("missing_title")
        if not venue:
            warnings.append("missing_venue")
        total_warnings += len(warnings)
        corpus = citation + " " + " ".join(notes)
        out.append(
            {
                "id": f"BIBN-{idx:04d}",
                "source_entry_id": _collapse(str(row.get("id", ""))),
                "order_index": int(row.get("order_index", idx)),
                "group": _collapse(str(row.get("group", ""))),
                "section": _collapse(str(row.get("section", ""))),
                "authors": authors,
                "author_count": len(authors),
                "publication_year": publication_year,
                "title": title,
                "venue": venue,
                "arxiv_id": _collapse(arxiv_id),
                "doi_list": doi_list,
                "url_list": urls,
                "document_type": _document_type(citation, arxiv_id, doi_list, urls),
                "evidence_relevance_score": _relevance_score(citation, notes),
                "claim_refs": sorted(set(CLAIM_ID_RE.findall(corpus))),
                "parse_warnings": warnings,
                "raw_citation": citation,
                "raw_notes": notes,
                "source_line": int(row.get("source_line", 0) or 0),
            }
        )
    meta = {
        "entry_count": len(out),
        "parse_warning_count": total_warnings,
    }
    return out, meta


def _extract_local_paths(body: str) -> list[str]:
    out: set[str] = set()
    for token in BACKTICK_RE.findall(body):
        candidate = _collapse(token).strip(" .;:,()[]{}")
        if not candidate:
            continue
        if "/" not in candidate and "." not in candidate:
            continue
        if candidate.startswith("http://") or candidate.startswith("https://"):
            continue
        out.add(candidate)
    return sorted(out)


def build_provenance_sources(documents: list[dict]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    seq = 0
    url_count = 0
    path_count = 0
    hash_count = 0
    doc_ids = set()
    for doc in sorted(documents, key=lambda item: str(item.get("id", ""))):
        doc_id = _collapse(str(doc.get("id", "")))
        if not doc_id:
            continue
        doc_ids.add(doc_id)
        rows_before = len(rows)
        source_markdown = _collapse(str(doc.get("source_markdown", "")))
        body = _ascii_clean(str(doc.get("body_markdown", "")))
        doc_claim_refs = [str(v) for v in doc.get("claim_refs", []) if isinstance(v, str)]
        claim_refs = _normalize_claim_refs(doc_claim_refs, "")
        urls = set(str(v) for v in doc.get("url_refs", []) if str(v).strip())
        urls.update(URL_RE.findall(body))
        paths = set(str(v) for v in doc.get("path_refs", []) if str(v).strip())
        paths.update(_extract_local_paths(body))
        hashes = sorted({item.lower() for item in SHA_RE.findall(body)})
        dates = sorted({item for item in DATE_RE.findall(body) if item})
        retrieved_date = dates[0] if dates else ""
        authority_level = _collapse(str(doc.get("authority_level", "")))
        verification_level = _collapse(str(doc.get("verification_level", "")))
        content_kind = _collapse(str(doc.get("content_kind", "")))
        notes = _collapse(str(doc.get("notes", "")))
        sorted_urls = sorted(urls)
        sorted_paths = sorted(paths)
        path_hash_map = {path: hashes[idx] for idx, path in enumerate(sorted_paths) if idx < len(hashes)}

        for url in sorted_urls:
            seq += 1
            url_count += 1
            rows.append(
                {
                    "id": f"PSR-{seq:05d}",
                    "document_id": doc_id,
                    "source_markdown": source_markdown,
                    "source_kind": "url",
                    "source_ref": _collapse(url),
                    "sha256": "",
                    "retrieved_date": retrieved_date,
                    "claim_refs": claim_refs,
                    "authority_level": authority_level,
                    "verification_level": verification_level,
                    "content_kind": content_kind,
                    "notes": notes,
                }
            )

        for path in sorted_paths:
            seq += 1
            path_count += 1
            rows.append(
                {
                    "id": f"PSR-{seq:05d}",
                    "document_id": doc_id,
                    "source_markdown": source_markdown,
                    "source_kind": "path",
                    "source_ref": _collapse(path),
                    "sha256": path_hash_map.get(path, ""),
                    "retrieved_date": retrieved_date,
                    "claim_refs": claim_refs,
                    "authority_level": authority_level,
                    "verification_level": verification_level,
                    "content_kind": content_kind,
                    "notes": notes,
                }
            )

        for digest in hashes:
            seq += 1
            hash_count += 1
            rows.append(
                {
                    "id": f"PSR-{seq:05d}",
                    "document_id": doc_id,
                    "source_markdown": source_markdown,
                    "source_kind": "sha256",
                    "source_ref": digest,
                    "sha256": digest,
                    "retrieved_date": retrieved_date,
                    "claim_refs": claim_refs,
                    "authority_level": authority_level,
                    "verification_level": verification_level,
                    "content_kind": content_kind,
                    "notes": notes,
                }
            )
        if len(rows) == rows_before:
            seq += 1
            path_count += 1
            rows.append(
                {
                    "id": f"PSR-{seq:05d}",
                    "document_id": doc_id,
                    "source_markdown": source_markdown,
                    "source_kind": "path",
                    "source_ref": source_markdown,
                    "sha256": "",
                    "retrieved_date": retrieved_date,
                    "claim_refs": claim_refs,
                    "authority_level": authority_level,
                    "verification_level": verification_level,
                    "content_kind": content_kind,
                    "notes": notes or "Fallback provenance anchor for document with no extracted urls/hashes.",
                }
            )
    meta = {
        "document_count": len(doc_ids),
        "record_count": len(rows),
        "url_record_count": url_count,
        "path_record_count": path_count,
        "hash_record_count": hash_count,
    }
    return rows, meta


def _paragraph_kind(text: str) -> str:
    stripped = text.strip()
    lines = [line.rstrip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return "empty"
    if lines[0].startswith("```"):
        return "code_block"
    if lines[0].startswith("#"):
        return "heading"
    if lines[0].startswith("<!--"):
        return "comment_block"
    if len(lines) >= 2 and all("|" in line for line in lines[:2]):
        return "table_block"
    if all(
        line.startswith("- ")
        or line.startswith("* ")
        or bool(re.match(r"^\d+\.\s+", line))
        for line in lines
    ):
        return "list_block"
    return "prose"


def _split_blocks(text: str) -> list[tuple[int, int, str]]:
    lines = _ascii_clean(text).splitlines()
    filtered: list[str] = []
    for line in lines:
        if line.startswith("<!-- AUTO-GENERATED: DO NOT EDIT -->"):
            continue
        if line.startswith("<!-- Source of truth:"):
            continue
        filtered.append(line)
    blocks: list[tuple[int, int, str]] = []
    buf: list[str] = []
    start_line = 1
    in_code = False
    for idx, line in enumerate(filtered, start=1):
        stripped = line.strip()
        fence = stripped.startswith("```")
        if fence:
            if not buf:
                start_line = idx
            buf.append(line)
            in_code = not in_code
            continue
        if in_code:
            if not buf:
                start_line = idx
            buf.append(line)
            continue
        if stripped:
            if not buf:
                start_line = idx
            buf.append(line)
            continue
        if buf:
            block_text = "\n".join(buf).rstrip()
            blocks.append((start_line, idx - 1, block_text))
            buf = []
    if buf:
        block_text = "\n".join(buf).rstrip()
        blocks.append((start_line, len(filtered), block_text))
    return blocks


def build_narrative_paragraph_atoms(
    docs_root_rows: list[dict],
    research_rows: list[dict],
    external_rows: list[dict],
    artifact_rows: list[dict],
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    seq = 0
    doc_counts: defaultdict[str, int] = defaultdict(int)
    para_counts: defaultdict[str, int] = defaultdict(int)
    sources = [
        ("registry/docs_root_narratives.toml", docs_root_rows),
        ("registry/research_narratives.toml", research_rows),
        ("registry/external_sources.toml", external_rows),
        ("registry/data_artifact_narratives.toml", artifact_rows),
    ]
    for source_registry, docs in sources:
        for doc in sorted(docs, key=lambda item: str(item.get("id", ""))):
            body = str(doc.get("body_markdown", ""))
            if not body.strip():
                continue
            doc_id = _collapse(str(doc.get("id", "")))
            source_markdown = _collapse(str(doc.get("source_markdown", "")))
            claim_refs_doc = [str(v) for v in doc.get("claim_refs", []) if isinstance(v, str)]
            claim_refs_doc = _normalize_claim_refs(claim_refs_doc, "")
            blocks = _split_blocks(body)
            if not blocks:
                continue
            doc_counts[source_registry] += 1
            for paragraph_index, (line_start, line_end, block) in enumerate(blocks, start=1):
                text = _collapse(block)
                if not text:
                    continue
                seq += 1
                claim_refs = sorted(set(claim_refs_doc + CLAIM_ID_RE.findall(text)))
                equation_refs = sorted(set(EQUATION_ID_RE.findall(text)))
                symbol_refs = sorted(set(SYMBOL_ID_RE.findall(text)))
                numeric_constants = sorted(set(NUMBER_RE.findall(text)))
                key_tokens = [
                    token
                    for token in IDENTIFIER_RE.findall(text)
                    if token.lower() not in STOPWORDS and len(token) >= 2
                ]
                para_counts[source_registry] += 1
                rows.append(
                    {
                        "id": f"NPA-{seq:06d}",
                        "source_registry": source_registry,
                        "document_id": doc_id,
                        "source_markdown": source_markdown,
                        "paragraph_index": paragraph_index,
                        "line_start": line_start,
                        "line_end": line_end,
                        "paragraph_kind": _paragraph_kind(block),
                        "text": text,
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "claim_refs": claim_refs,
                        "equation_refs": equation_refs,
                        "symbol_refs": symbol_refs,
                        "numeric_constants": numeric_constants,
                        "key_tokens": sorted(dict.fromkeys(key_tokens))[:24],
                    }
                )
    meta = {
        "paragraph_count": len(rows),
        "document_count": sum(doc_counts.values()),
        "docs_root_document_count": doc_counts["registry/docs_root_narratives.toml"],
        "research_document_count": doc_counts["registry/research_narratives.toml"],
        "external_sources_document_count": doc_counts["registry/external_sources.toml"],
        "artifact_document_count": doc_counts["registry/data_artifact_narratives.toml"],
        "docs_root_paragraph_count": para_counts["registry/docs_root_narratives.toml"],
        "research_paragraph_count": para_counts["registry/research_narratives.toml"],
        "external_sources_paragraph_count": para_counts["registry/external_sources.toml"],
        "artifact_paragraph_count": para_counts["registry/data_artifact_narratives.toml"],
    }
    return rows, meta


def _render_derivation_steps(rows: list[dict], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Derivation step registry (Wave 5 strict schema).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch2_registries.py.")
    lines.append("")
    lines.append("[knowledge_derivation_steps]")
    lines.append('updated = "2026-02-10"')
    lines.append("authoritative = true")
    lines.append('source_registry = "registry/knowledge/proof_skeletons.toml"')
    lines.append(f"step_count = {meta['step_count']}")
    lines.append(f"skeleton_count = {meta['skeleton_count']}")
    lines.append(f"skeleton_with_steps_count = {meta['skeleton_with_steps_count']}")
    lines.append(f"claim_linked_step_count = {meta['claim_linked_step_count']}")
    lines.append(f"max_steps_per_skeleton = {meta['max_steps_per_skeleton']}")
    lines.append("")
    for row in rows:
        lines.append("[[step]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"skeleton_id = {_q(row['skeleton_id'])}")
        lines.append(f"skeleton_kind = {_q(row['skeleton_kind'])}")
        lines.append(f"source_path = {_q(row['source_path'])}")
        lines.append(f"source_uid = {_q(row['source_uid'])}")
        lines.append(f"claim_id = {_q(row['claim_id'])}")
        lines.append(f"claim_refs = {_render_list(row['claim_refs'])}")
        lines.append(f"step_index = {row['step_index']}")
        lines.append(f"step_kind = {_q(row['step_kind'])}")
        lines.append(f"text = {_q(row['text'])}")
        lines.append(f"text_sha256 = {_q(row['text_sha256'])}")
        lines.append(f"equation_refs = {_render_list(row['equation_refs'])}")
        lines.append(f"symbol_refs = {_render_list(row['symbol_refs'])}")
        lines.append(f"numeric_constants = {_render_list(row['numeric_constants'])}")
        lines.append(f"key_tokens = {_render_list(row['key_tokens'])}")
        lines.append(f"depends_on_step_ids = {_render_list(row['depends_on_step_ids'])}")
        lines.append(f"line_start = {row['line_start']}")
        lines.append(f"line_end = {row['line_end']}")
        lines.append("")
    return "\n".join(lines)


def _render_bibliography_normalized(rows: list[dict], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Normalized bibliography registry (Wave 5 strict schema).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch2_registries.py.")
    lines.append("")
    lines.append("[bibliography_normalized]")
    lines.append('updated = "2026-02-10"')
    lines.append("authoritative = true")
    lines.append('source_registry = "registry/bibliography.toml"')
    lines.append(f"entry_count = {meta['entry_count']}")
    lines.append(f"parse_warning_count = {meta['parse_warning_count']}")
    lines.append("")
    for row in rows:
        lines.append("[[entry]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"source_entry_id = {_q(row['source_entry_id'])}")
        lines.append(f"order_index = {row['order_index']}")
        lines.append(f"group = {_q(row['group'])}")
        lines.append(f"section = {_q(row['section'])}")
        lines.append(f"authors = {_render_list(row['authors'])}")
        lines.append(f"author_count = {row['author_count']}")
        lines.append(f"publication_year = {row['publication_year']}")
        lines.append(f"title = {_q(row['title'])}")
        lines.append(f"venue = {_q(row['venue'])}")
        lines.append(f"arxiv_id = {_q(row['arxiv_id'])}")
        lines.append(f"doi_list = {_render_list(row['doi_list'])}")
        lines.append(f"url_list = {_render_list(row['url_list'])}")
        lines.append(f"document_type = {_q(row['document_type'])}")
        lines.append(f"evidence_relevance_score = {row['evidence_relevance_score']}")
        lines.append(f"claim_refs = {_render_list(row['claim_refs'])}")
        lines.append(f"parse_warnings = {_render_list(row['parse_warnings'])}")
        lines.append(f"raw_citation = {_q(row['raw_citation'])}")
        lines.append(f"raw_notes = {_render_list(row['raw_notes'])}")
        lines.append(f"source_line = {row['source_line']}")
        lines.append("")
    return "\n".join(lines)


def _render_provenance_sources(rows: list[dict], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Provenance source registry (Wave 5 strict schema).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch2_registries.py.")
    lines.append("")
    lines.append("[provenance_sources]")
    lines.append('updated = "2026-02-10"')
    lines.append("authoritative = true")
    lines.append('source_registry = "registry/external_sources.toml"')
    lines.append(f"document_count = {meta['document_count']}")
    lines.append(f"record_count = {meta['record_count']}")
    lines.append(f"url_record_count = {meta['url_record_count']}")
    lines.append(f"path_record_count = {meta['path_record_count']}")
    lines.append(f"hash_record_count = {meta['hash_record_count']}")
    lines.append("")
    for row in rows:
        lines.append("[[record]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"document_id = {_q(row['document_id'])}")
        lines.append(f"source_markdown = {_q(row['source_markdown'])}")
        lines.append(f"source_kind = {_q(row['source_kind'])}")
        lines.append(f"source_ref = {_q(row['source_ref'])}")
        lines.append(f"sha256 = {_q(row['sha256'])}")
        lines.append(f"retrieved_date = {_q(row['retrieved_date'])}")
        lines.append(f"claim_refs = {_render_list(row['claim_refs'])}")
        lines.append(f"authority_level = {_q(row['authority_level'])}")
        lines.append(f"verification_level = {_q(row['verification_level'])}")
        lines.append(f"content_kind = {_q(row['content_kind'])}")
        lines.append(f"notes = {_q(row['notes'])}")
        lines.append("")
    return "\n".join(lines)


def _render_narrative_paragraph_atoms(rows: list[dict], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Narrative paragraph atom registry (Wave 5 strict schema).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch2_registries.py.")
    lines.append("")
    lines.append("[narrative_paragraph_atoms]")
    lines.append('updated = "2026-02-10"')
    lines.append("authoritative = true")
    lines.append('source_registries = ["registry/docs_root_narratives.toml", "registry/research_narratives.toml", "registry/external_sources.toml", "registry/data_artifact_narratives.toml"]')
    lines.append(f"document_count = {meta['document_count']}")
    lines.append(f"paragraph_count = {meta['paragraph_count']}")
    lines.append(f"docs_root_document_count = {meta['docs_root_document_count']}")
    lines.append(f"research_document_count = {meta['research_document_count']}")
    lines.append(f"external_sources_document_count = {meta['external_sources_document_count']}")
    lines.append(f"artifact_document_count = {meta['artifact_document_count']}")
    lines.append(f"docs_root_paragraph_count = {meta['docs_root_paragraph_count']}")
    lines.append(f"research_paragraph_count = {meta['research_paragraph_count']}")
    lines.append(f"external_sources_paragraph_count = {meta['external_sources_paragraph_count']}")
    lines.append(f"artifact_paragraph_count = {meta['artifact_paragraph_count']}")
    lines.append("")
    for row in rows:
        lines.append("[[paragraph]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"source_registry = {_q(row['source_registry'])}")
        lines.append(f"document_id = {_q(row['document_id'])}")
        lines.append(f"source_markdown = {_q(row['source_markdown'])}")
        lines.append(f"paragraph_index = {row['paragraph_index']}")
        lines.append(f"line_start = {row['line_start']}")
        lines.append(f"line_end = {row['line_end']}")
        lines.append(f"paragraph_kind = {_q(row['paragraph_kind'])}")
        lines.append(f"text = {_q(row['text'])}")
        lines.append(f"text_sha256 = {_q(row['text_sha256'])}")
        lines.append(f"claim_refs = {_render_list(row['claim_refs'])}")
        lines.append(f"equation_refs = {_render_list(row['equation_refs'])}")
        lines.append(f"symbol_refs = {_render_list(row['symbol_refs'])}")
        lines.append(f"numeric_constants = {_render_list(row['numeric_constants'])}")
        lines.append(f"key_tokens = {_render_list(row['key_tokens'])}")
        lines.append("")
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    _assert_ascii(content, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--derivation-out",
        default="registry/knowledge/derivation_steps.toml",
        help="Derivation step registry output path.",
    )
    parser.add_argument(
        "--bibliography-out",
        default="registry/bibliography_normalized.toml",
        help="Bibliography normalized registry output path.",
    )
    parser.add_argument(
        "--provenance-out",
        default="registry/provenance_sources.toml",
        help="Provenance source registry output path.",
    )
    parser.add_argument(
        "--paragraph-out",
        default="registry/narrative_paragraph_atoms.toml",
        help="Narrative paragraph atom registry output path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    proof_rows = _load(root / "registry/knowledge/proof_skeletons.toml").get("skeleton", [])
    bibliography_entries = _load(root / "registry/bibliography.toml").get("entry", [])
    external_docs = _load(root / "registry/external_sources.toml").get("document", [])
    docs_root_rows = _load(root / "registry/docs_root_narratives.toml").get("document", [])
    research_rows = _load(root / "registry/research_narratives.toml").get("document", [])
    artifact_rows = _load(root / "registry/data_artifact_narratives.toml").get("document", [])

    derivation_rows, derivation_meta = build_derivation_steps(proof_rows)
    bibliography_rows, bibliography_meta = build_bibliography_normalized(bibliography_entries)
    provenance_rows, provenance_meta = build_provenance_sources(external_docs)
    paragraph_rows, paragraph_meta = build_narrative_paragraph_atoms(
        docs_root_rows, research_rows, external_docs, artifact_rows
    )

    derivation_text = _render_derivation_steps(derivation_rows, derivation_meta)
    bibliography_text = _render_bibliography_normalized(bibliography_rows, bibliography_meta)
    provenance_text = _render_provenance_sources(provenance_rows, provenance_meta)
    paragraph_text = _render_narrative_paragraph_atoms(paragraph_rows, paragraph_meta)

    _write(root / args.derivation_out, derivation_text)
    _write(root / args.bibliography_out, bibliography_text)
    _write(root / args.provenance_out, provenance_text)
    _write(root / args.paragraph_out, paragraph_text)

    print(
        "Wrote Wave5 Batch2 registries: "
        f"derivations={len(derivation_rows)} "
        f"bibliography={len(bibliography_rows)} "
        f"provenance={len(provenance_rows)} "
        f"paragraphs={len(paragraph_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
