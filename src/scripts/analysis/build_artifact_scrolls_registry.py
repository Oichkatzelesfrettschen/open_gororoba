#!/usr/bin/env python3
"""
Build TOML-native artifact scroll registries from high-information data/artifacts corpora.

Inputs:
- registry/data_artifact_narratives.toml
- registry/knowledge/equation_atoms.toml
- registry/knowledge/proof_atoms.toml
- registry/claims.toml

Outputs:
- registry/artifact_scrolls.toml (index and canonical owner for artifact markdown)
- registry/knowledge/artifacts/ART-*.toml (structured per-document scrolls)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CLAIM_RE = re.compile(r"\bC-\d{3}\b")
URL_RE = re.compile(r"https?://[^\s)>\"']+")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
ARXIV_RE = re.compile(r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\b", flags=re.I)
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")

STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "into",
    "their",
    "there",
    "where",
    "when",
    "then",
    "than",
    "have",
    "has",
    "been",
    "being",
    "were",
    "will",
    "would",
    "should",
    "could",
    "about",
    "between",
    "because",
    "while",
    "using",
    "used",
    "also",
    "only",
    "over",
    "under",
    "after",
    "before",
    "these",
    "those",
    "which",
    "such",
    "very",
    "more",
    "most",
    "some",
    "many",
    "much",
    "into",
    "across",
    "without",
    "within",
    "through",
    "between",
    "report",
    "final",
    "results",
    "result",
    "section",
}


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


def _collapse_ws(text: str) -> str:
    return " ".join(_ascii_sanitize(text).split())


def _esc(value: str) -> str:
    return json.dumps(_ascii_sanitize(value), ensure_ascii=True)


def _render_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(_esc(item) for item in items) + "]"


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _fingerprint(text: str) -> str:
    normalized = _collapse_ws(text).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _split_sections(text: str) -> list[dict[str, object]]:
    lines = _ascii_sanitize(text).splitlines()
    sections: list[dict[str, object]] = []
    current_title = "(root)"
    current_level = 0
    current_start = 1
    current_lines: list[str] = []

    for idx, raw in enumerate(lines, start=1):
        match = HEADING_RE.match(raw)
        if match:
            sections.append(
                {
                    "title": current_title,
                    "level": current_level,
                    "line_start": current_start,
                    "line_end": max(current_start, idx - 1),
                    "lines": current_lines,
                }
            )
            current_title = _collapse_ws(match.group(2))
            current_level = len(match.group(1))
            current_start = idx
            current_lines = []
            continue
        current_lines.append(raw)

    sections.append(
        {
            "title": current_title,
            "level": current_level,
            "line_start": current_start,
            "line_end": max(current_start, len(lines)),
            "lines": current_lines,
        }
    )

    trimmed: list[dict[str, object]] = []
    for section in sections:
        body = "\n".join(section["lines"]).strip()
        if section["title"] == "(root)" and not body:
            continue
        trimmed.append(section)
    if not trimmed:
        trimmed.append(
            {
                "title": "(root)",
                "level": 0,
                "line_start": 1,
                "line_end": max(1, len(lines)),
                "lines": lines,
            }
        )
    return trimmed


def _extract_key_terms(text: str, limit: int = 24) -> list[str]:
    counter: Counter[str] = Counter()
    for token in IDENTIFIER_RE.findall(_ascii_sanitize(text)):
        lowered = token.lower()
        if lowered in STOPWORDS:
            continue
        if lowered.startswith("http"):
            continue
        if len(lowered) < 4:
            continue
        counter[lowered] += 1
    return [item for item, _count in counter.most_common(limit)]


def _extract_source_refs(lines: list[str]) -> list[dict[str, object]]:
    refs: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()

    for line_no, raw in enumerate(lines, start=1):
        line = _ascii_sanitize(raw).strip()
        if not line:
            continue

        for url in URL_RE.findall(line):
            key = ("url", url, line_no)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                {
                    "kind": "url",
                    "value": url,
                    "line": line_no,
                    "excerpt": _collapse_ws(line)[:220],
                }
            )

        for doi in DOI_RE.findall(line):
            key = ("doi", doi, line_no)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                {
                    "kind": "doi",
                    "value": doi,
                    "line": line_no,
                    "excerpt": _collapse_ws(line)[:220],
                }
            )

        for arxiv in ARXIV_RE.findall(line):
            normalized = arxiv.replace("ARXIV:", "arXiv:").replace("ArXiv:", "arXiv:")
            key = ("arxiv", normalized, line_no)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                {
                    "kind": "arxiv",
                    "value": normalized,
                    "line": line_no,
                    "excerpt": _collapse_ws(line)[:220],
                }
            )

        maybe_citation = (
            line.startswith(("-", "*", "["))
            and re.search(r"(19|20)\d{2}", line) is not None
            and len(line) >= 24
        )
        if maybe_citation:
            citation_value = _collapse_ws(line)
            key = ("citation_line", citation_value, line_no)
            if key not in seen:
                seen.add(key)
                refs.append(
                    {
                        "kind": "citation_line",
                        "value": citation_value[:240],
                        "line": line_no,
                        "excerpt": citation_value[:240],
                    }
                )

    refs.sort(key=lambda item: (int(item["line"]), str(item["kind"]), str(item["value"])))
    return refs


def _write(path: Path, text: str) -> None:
    _assert_ascii(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_scroll_toml(
    doc: dict[str, object],
    claim_status: dict[str, str],
    equation_rows: list[dict[str, object]],
    proof_rows: list[dict[str, object]],
) -> tuple[str, dict[str, int], str]:
    source_path = str(doc.get("source_markdown", "")).strip()
    body = _ascii_sanitize(str(doc.get("body_markdown", "")).rstrip("\n"))
    sections = _split_sections(body)
    key_terms = _extract_key_terms(body)
    source_refs = _extract_source_refs(body.splitlines())
    claim_refs = sorted(set(str(item) for item in doc.get("claim_refs", [])) | set(CLAIM_RE.findall(body)))
    unknown_claim_refs = [item for item in claim_refs if item not in claim_status]
    document_fingerprint = _fingerprint(body)

    section_rows: list[dict[str, object]] = []
    for idx, section in enumerate(sections, start=1):
        section_text = _ascii_sanitize("\n".join(section["lines"]).rstrip("\n"))
        section_claim_refs = sorted(set(CLAIM_RE.findall(section_text)))
        section_equations = [
            row
            for row in equation_rows
            if int(section["line_start"]) <= int(row.get("source_line", -1)) <= int(section["line_end"])
        ]
        section_proofs = [
            row
            for row in proof_rows
            if not (
                int(row.get("line_end", -1)) < int(section["line_start"])
                or int(row.get("line_start", -1)) > int(section["line_end"])
            )
        ]
        if (
            not section_text.strip()
            and not section_claim_refs
            and not section_equations
            and not section_proofs
        ):
            continue
        paragraph_count = max(
            1,
            len([part for part in re.split(r"\n\s*\n", section_text) if part.strip()]),
        )
        section_rows.append(
            {
                "id": f"{doc.get('id', 'ART-XXX')}-SEC-{idx:03d}",
                "title": _collapse_ws(str(section["title"])),
                "level": int(section["level"]),
                "line_start": int(section["line_start"]),
                "line_end": int(section["line_end"]),
                "paragraph_count": paragraph_count,
                "char_count": len(section_text),
                "fingerprint": _fingerprint(section_text),
                "summary": _collapse_ws(section_text)[:220],
                "claim_refs": section_claim_refs,
                "equation_ref_ids": [str(item.get("id", "")) for item in section_equations if item.get("id")],
                "proof_ref_ids": [str(item.get("id", "")) for item in section_proofs if item.get("id")],
                "body_text": section_text,
            }
        )

    lines: list[str] = []
    lines.append("# Structured artifact scroll (TOML-first, canonical).")
    lines.append("# Generated by src/scripts/analysis/build_artifact_scrolls_registry.py")
    lines.append("")
    lines.append("[scroll]")
    lines.append(f"id = {_esc(str(doc.get('id', '')))}")
    lines.append(f"source_uid = {_esc(str(doc.get('id', '')))}")
    lines.append('source_registry = "registry/data_artifact_narratives.toml"')
    lines.append(f"source_markdown = {_esc(source_path)}")
    lines.append(f"title = {_esc(str(doc.get('title', '')))}")
    lines.append(f"content_kind = {_esc(str(doc.get('content_kind', 'artifact_narrative')))}")
    lines.append('canonical_registry = "registry/artifact_scrolls.toml"')
    lines.append("authoritative = true")
    lines.append('updated = "deterministic"')
    lines.append(f"line_count = {int(doc.get('line_count', 0))}")
    lines.append(f"section_count = {len(section_rows)}")
    lines.append(f"claim_ref_count = {len(claim_refs)}")
    lines.append(f"equation_ref_count = {len(equation_rows)}")
    lines.append(f"proof_ref_count = {len(proof_rows)}")
    lines.append(f"source_ref_count = {len(source_refs)}")
    lines.append(f"unknown_claim_ref_count = {len(unknown_claim_refs)}")
    lines.append(f"dedup_fingerprint = {_esc(document_fingerprint)}")
    lines.append(f"key_terms = {_render_list(key_terms)}")
    lines.append(f"claim_refs = {_render_list(claim_refs)}")
    lines.append(f"unknown_claim_refs = {_render_list(unknown_claim_refs)}")
    lines.append("")

    for claim_id in claim_refs:
        lines.append("[[claim_ref]]")
        lines.append(f"id = {_esc(claim_id)}")
        lines.append(f"status_token = {_esc(claim_status.get(claim_id, 'UNKNOWN'))}")
        lines.append("source = \"registry/claims.toml\"")
        lines.append("")

    for row in equation_rows:
        lines.append("[[equation_ref]]")
        lines.append(f"id = {_esc(str(row.get('id', '')))}")
        lines.append(f"source_line = {int(row.get('source_line', 0))}")
        lines.append(f"equation_kind = {_esc(str(row.get('equation_kind', '')))}")
        lines.append(f"relation_operator = {_esc(str(row.get('relation_operator', '')))}")
        lines.append(f"domain_hint = {_esc(str(row.get('domain_hint', '')))}")
        lines.append(f"expression = {_esc(str(row.get('expression', '')))}")
        lines.append(f"symbol_names = {_render_list([str(item) for item in row.get('symbol_names', [])])}")
        lines.append(f"numeric_constants = {_render_list([str(item) for item in row.get('numeric_constants', [])])}")
        lines.append(f"claim_refs = {_render_list([str(item) for item in row.get('claim_refs', [])])}")
        lines.append("")

    for row in proof_rows:
        lines.append("[[proof_ref]]")
        lines.append(f"id = {_esc(str(row.get('id', '')))}")
        lines.append(f"proof_kind = {_esc(str(row.get('proof_kind', '')))}")
        lines.append(f"section_title = {_esc(str(row.get('section_title', '')))}")
        lines.append(f"line_start = {int(row.get('line_start', 0))}")
        lines.append(f"line_end = {int(row.get('line_end', 0))}")
        lines.append(f"step_count = {int(row.get('step_count', 0))}")
        lines.append(f"supports_claim = {'true' if bool(row.get('supports_claim', False)) else 'false'}")
        lines.append(f"claim_refs = {_render_list([str(item) for item in row.get('claim_refs', [])])}")
        lines.append(f"excerpt = {_esc(str(row.get('excerpt', '')))}")
        lines.append("")

    for idx, row in enumerate(source_refs, start=1):
        lines.append("[[source_ref]]")
        lines.append(f"id = {_esc(f'SRC-{idx:03d}')}")
        lines.append(f"kind = {_esc(str(row.get('kind', '')))}")
        lines.append(f"value = {_esc(str(row.get('value', '')))}")
        lines.append(f"line = {int(row.get('line', 0))}")
        lines.append(f"excerpt = {_esc(str(row.get('excerpt', '')))}")
        lines.append("")

    for row in section_rows:
        lines.append("[[section]]")
        lines.append(f"id = {_esc(str(row['id']))}")
        lines.append(f"title = {_esc(str(row['title']))}")
        lines.append(f"level = {int(row['level'])}")
        lines.append(f"line_start = {int(row['line_start'])}")
        lines.append(f"line_end = {int(row['line_end'])}")
        lines.append(f"paragraph_count = {int(row['paragraph_count'])}")
        lines.append(f"char_count = {int(row['char_count'])}")
        lines.append(f"fingerprint = {_esc(str(row['fingerprint']))}")
        lines.append(f"summary = {_esc(str(row['summary']))}")
        lines.append(f"claim_refs = {_render_list([str(item) for item in row['claim_refs']])}")
        lines.append(
            f"equation_ref_ids = {_render_list([str(item) for item in row['equation_ref_ids']])}"
        )
        lines.append(f"proof_ref_ids = {_render_list([str(item) for item in row['proof_ref_ids']])}")
        lines.append(f"body_text = {_esc(str(row['body_text']))}")
        lines.append("")

    counts = {
        "section_count": len(section_rows),
        "claim_ref_count": len(claim_refs),
        "equation_ref_count": len(equation_rows),
        "proof_ref_count": len(proof_rows),
        "source_ref_count": len(source_refs),
    }
    return "\n".join(lines), counts, document_fingerprint


def _render_index(rows: list[dict[str, object]]) -> str:
    total_sections = sum(int(row["section_count"]) for row in rows)
    total_claims = sum(int(row["claim_ref_count"]) for row in rows)
    total_equations = sum(int(row["equation_ref_count"]) for row in rows)
    total_proofs = sum(int(row["proof_ref_count"]) for row in rows)
    total_sources = sum(int(row["source_ref_count"]) for row in rows)

    lines: list[str] = []
    lines.append("# Canonical index for structured artifact scrolls.")
    lines.append("# Generated by src/scripts/analysis/build_artifact_scrolls_registry.py")
    lines.append("")
    lines.append("[artifact_scrolls]")
    lines.append('updated = "deterministic"')
    lines.append("authoritative = true")
    lines.append('source_registry = "registry/data_artifact_narratives.toml"')
    lines.append(f"scroll_count = {len(rows)}")
    lines.append(f"total_section_count = {total_sections}")
    lines.append(f"total_claim_ref_count = {total_claims}")
    lines.append(f"total_equation_ref_count = {total_equations}")
    lines.append(f"total_proof_ref_count = {total_proofs}")
    lines.append(f"total_source_ref_count = {total_sources}")
    lines.append("")

    for row in rows:
        lines.append("[[scroll]]")
        lines.append(f"id = {_esc(str(row['id']))}")
        lines.append(f"source_uid = {_esc(str(row['id']))}")
        lines.append(f"source_markdown = {_esc(str(row['source_markdown']))}")
        lines.append(f"title = {_esc(str(row['title']))}")
        lines.append(f"content_kind = {_esc(str(row['content_kind']))}")
        lines.append(f"scroll_path = {_esc(str(row['scroll_path']))}")
        lines.append(f"canonical = true")
        lines.append(f"section_count = {int(row['section_count'])}")
        lines.append(f"claim_ref_count = {int(row['claim_ref_count'])}")
        lines.append(f"equation_ref_count = {int(row['equation_ref_count'])}")
        lines.append(f"proof_ref_count = {int(row['proof_ref_count'])}")
        lines.append(f"source_ref_count = {int(row['source_ref_count'])}")
        lines.append(f"dedup_fingerprint = {_esc(str(row['dedup_fingerprint']))}")
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
        "--source-registry",
        default="registry/data_artifact_narratives.toml",
        help="Input narratives registry path.",
    )
    parser.add_argument(
        "--equation-registry",
        default="registry/knowledge/equation_atoms.toml",
        help="Input equation atoms registry path.",
    )
    parser.add_argument(
        "--proof-registry",
        default="registry/knowledge/proof_atoms.toml",
        help="Input proof atoms registry path.",
    )
    parser.add_argument(
        "--claims-registry",
        default="registry/claims.toml",
        help="Claims registry path for status reconciliation.",
    )
    parser.add_argument(
        "--index-out",
        default="registry/artifact_scrolls.toml",
        help="Output artifact scroll index TOML path.",
    )
    parser.add_argument(
        "--scroll-dir",
        default="registry/knowledge/artifacts",
        help="Output directory for per-document artifact scroll TOML files.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source_data = tomllib.loads((repo_root / args.source_registry).read_text(encoding="utf-8"))
    equation_data = tomllib.loads((repo_root / args.equation_registry).read_text(encoding="utf-8"))
    proof_data = tomllib.loads((repo_root / args.proof_registry).read_text(encoding="utf-8"))
    claims_data = tomllib.loads((repo_root / args.claims_registry).read_text(encoding="utf-8"))

    claim_status = {
        str(row.get("id", "")).strip(): _collapse_ws(str(row.get("status", ""))).upper().replace(" ", "_")
        for row in claims_data.get("claim", [])
        if str(row.get("id", "")).strip()
    }

    equations_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in equation_data.get("atom", []):
        source_path = str(row.get("source_path", "")).strip()
        if source_path:
            equations_by_source[source_path].append(row)
    for value in equations_by_source.values():
        value.sort(key=lambda item: int(item.get("source_line", 0)))

    proofs_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in proof_data.get("atom", []):
        source_path = str(row.get("source_path", "")).strip()
        if source_path:
            proofs_by_source[source_path].append(row)
    for value in proofs_by_source.values():
        value.sort(key=lambda item: int(item.get("line_start", 0)))

    docs = sorted(
        source_data.get("document", []),
        key=lambda item: str(item.get("source_markdown", "")),
    )

    scroll_dir = repo_root / args.scroll_dir
    scroll_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    for doc in docs:
        source_markdown = str(doc.get("source_markdown", "")).strip()
        scroll_id = str(doc.get("id", "")).strip()
        if not source_markdown or not scroll_id:
            continue
        scroll_rel = f"{args.scroll_dir.rstrip('/')}/{scroll_id}.toml"
        scroll_path = repo_root / scroll_rel
        scroll_text, counts, doc_fingerprint = _render_scroll_toml(
            doc=doc,
            claim_status=claim_status,
            equation_rows=equations_by_source.get(source_markdown, []),
            proof_rows=proofs_by_source.get(source_markdown, []),
        )
        _write(scroll_path, scroll_text)
        index_rows.append(
            {
                "id": scroll_id,
                "source_markdown": source_markdown,
                "title": str(doc.get("title", "")),
                "content_kind": str(doc.get("content_kind", "")),
                "scroll_path": scroll_rel,
                "section_count": counts["section_count"],
                "claim_ref_count": counts["claim_ref_count"],
                "equation_ref_count": counts["equation_ref_count"],
                "proof_ref_count": counts["proof_ref_count"],
                "source_ref_count": counts["source_ref_count"],
                "dedup_fingerprint": doc_fingerprint,
            }
        )

    index_text = _render_index(index_rows)
    _write(repo_root / args.index_out, index_text)

    print(
        "Wrote artifact scroll registries: "
        f"scrolls={len(index_rows)} index={args.index_out} dir={args.scroll_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
