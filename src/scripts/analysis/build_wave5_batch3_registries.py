#!/usr/bin/env python3
"""
Build Wave 5 Batch 3 strict TOML registries:
- W5-015: registry/conflict_markers.toml
- W5-015: registry/lacunae.toml
- W5-019: registry/schema_signatures.toml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any


CLAIM_ID_RE = re.compile(r"\bC-\d{3}\b")
I_ID_RE = re.compile(r"\bI-\d{3}\b")
E_ID_RE = re.compile(r"\bE-\d{3}\b")
XS_ID_RE = re.compile(r"\bXS-\d{3}\b")
PC_ID_RE = re.compile(r"\bPC-\d{4}\b")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b")


POSITIVE_TERMS = (
    "verified",
    "established",
    "confirmed",
    "holds",
    "consistent",
    "valid",
    "supported",
    "reproduced",
)

NEGATIVE_TERMS = (
    "refuted",
    "invalid",
    "fails",
    "contradiction",
    "inconsistent",
    "obstruction",
    "inconclusive",
    "insufficient",
)

UNRESOLVED_STATUS_TOKENS = {
    "PARTIAL",
    "INCONCLUSIVE",
    "THEORETICAL",
    "CLOSED_SOURCE_INSUFFICIENT",
    "CLOSED_METHODOLOGY_INSUFFICIENT",
}

VERIFIED_STATUS_TOKENS = {
    "VERIFIED",
    "ESTABLISHED",
}

REFUTED_STATUS_TOKENS = {
    "REFUTED",
    "CLOSED_REFUTED",
    "CLOSED_NEGATIVE_RESULT",
}

STOPWORDS = {
    "about",
    "across",
    "after",
    "against",
    "all",
    "also",
    "analysis",
    "being",
    "between",
    "both",
    "claim",
    "data",
    "does",
    "each",
    "from",
    "have",
    "into",
    "its",
    "model",
    "more",
    "must",
    "needs",
    "only",
    "other",
    "over",
    "results",
    "should",
    "some",
    "than",
    "their",
    "there",
    "these",
    "this",
    "through",
    "under",
    "using",
    "when",
    "where",
    "which",
    "while",
    "with",
    "without",
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


def _write(path: Path, content: str) -> None:
    _assert_ascii(content, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def _status_token(status: str) -> str:
    token = _collapse(status).upper()
    token = token.replace("/", "_").replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^A-Z0-9_]", "", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "UNSPECIFIED"


def _extract_claim_refs(text: str) -> list[str]:
    return sorted(set(CLAIM_ID_RE.findall(text)))


def _extract_id_refs(text: str) -> dict[str, list[str]]:
    return {
        "claim_refs": sorted(set(CLAIM_ID_RE.findall(text))),
        "insight_refs": sorted(set(I_ID_RE.findall(text))),
        "experiment_refs": sorted(set(E_ID_RE.findall(text))),
        "source_refs": sorted(set(XS_ID_RE.findall(text))),
        "dataset_refs": sorted(set(PC_ID_RE.findall(text))),
    }


def _token_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in WORD_RE.findall(_ascii_clean(text))
        if token.lower() not in STOPWORDS and len(token) >= 4
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0


def _split_sections(text: str) -> list[dict[str, Any]]:
    lines = _ascii_clean(text).splitlines()
    sections: list[dict[str, Any]] = []
    current_title = "(root)"
    current_level = 0
    current_start = 1
    current_lines: list[str] = []
    for idx, line in enumerate(lines, start=1):
        hm = HEADING_RE.match(line)
        if hm:
            sections.append(
                {
                    "title": current_title,
                    "level": current_level,
                    "line_start": current_start,
                    "line_end": max(current_start, idx - 1),
                    "body": "\n".join(current_lines),
                }
            )
            current_title = _collapse(hm.group(2))
            current_level = len(hm.group(1))
            current_start = idx
            current_lines = []
            continue
        current_lines.append(line)
    sections.append(
        {
            "title": current_title,
            "level": current_level,
            "line_start": current_start,
            "line_end": max(current_start, len(lines)),
            "body": "\n".join(current_lines),
        }
    )
    return sections


def build_conflict_markers(
    claims_rows: list[dict[str, Any]],
    docs_root_rows: list[dict[str, Any]],
    research_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    markers: list[dict[str, Any]] = []
    seq = 0

    # Claim-level status/text tensions
    claim_by_id: dict[str, dict[str, Any]] = {}
    statement_tokens: dict[str, set[str]] = {}
    for row in claims_rows:
        cid = _collapse(str(row.get("id", "")))
        if not cid:
            continue
        status_token = _status_token(str(row.get("status", "")))
        statement = _collapse(str(row.get("statement", "")))
        claim_by_id[cid] = {
            "id": cid,
            "statement": statement,
            "status_token": status_token,
        }
        statement_tokens[cid] = _token_set(statement)
        lower = statement.lower()
        has_positive = any(term in lower for term in POSITIVE_TERMS)
        has_negative = any(term in lower for term in NEGATIVE_TERMS)
        if status_token in REFUTED_STATUS_TOKENS and has_positive:
            seq += 1
            markers.append(
                {
                    "id": f"CM-{seq:04d}",
                    "marker_kind": "claim_status_statement_tension",
                    "severity": "high",
                    "status": "open",
                    "claim_refs": [cid],
                    "source_registry": "registry/claims.toml",
                    "source_document": cid,
                    "section_label": "statement",
                    "line_start": 0,
                    "line_end": 0,
                    "positive_evidence": [statement],
                    "negative_evidence": [status_token],
                    "jaccard_overlap": 0.0,
                    "notes": "Refuted/negative status conflicts with positive language in statement.",
                }
            )
        if status_token in VERIFIED_STATUS_TOKENS and has_negative:
            seq += 1
            markers.append(
                {
                    "id": f"CM-{seq:04d}",
                    "marker_kind": "claim_status_statement_tension",
                    "severity": "high",
                    "status": "open",
                    "claim_refs": [cid],
                    "source_registry": "registry/claims.toml",
                    "source_document": cid,
                    "section_label": "statement",
                    "line_start": 0,
                    "line_end": 0,
                    "positive_evidence": [status_token],
                    "negative_evidence": [statement],
                    "jaccard_overlap": 0.0,
                    "notes": "Verified status conflicts with negative language in statement.",
                }
            )

    # Claim-to-claim semantic contradictions
    claim_ids = sorted(claim_by_id.keys())
    for idx, a_id in enumerate(claim_ids):
        a = claim_by_id[a_id]
        a_status = a["status_token"]
        if a_status not in VERIFIED_STATUS_TOKENS.union(REFUTED_STATUS_TOKENS):
            continue
        for b_id in claim_ids[idx + 1 :]:
            b = claim_by_id[b_id]
            b_status = b["status_token"]
            if b_status not in VERIFIED_STATUS_TOKENS.union(REFUTED_STATUS_TOKENS):
                continue
            opposite = (a_status in VERIFIED_STATUS_TOKENS and b_status in REFUTED_STATUS_TOKENS) or (
                a_status in REFUTED_STATUS_TOKENS and b_status in VERIFIED_STATUS_TOKENS
            )
            if not opposite:
                continue
            score = _jaccard(statement_tokens[a_id], statement_tokens[b_id])
            if score < 0.70:
                continue
            seq += 1
            markers.append(
                {
                    "id": f"CM-{seq:04d}",
                    "marker_kind": "claim_semantic_status_conflict",
                    "severity": "medium",
                    "status": "open",
                    "claim_refs": sorted([a_id, b_id]),
                    "source_registry": "registry/claims.toml",
                    "source_document": f"{a_id}|{b_id}",
                    "section_label": "cross_claim_statement",
                    "line_start": 0,
                    "line_end": 0,
                    "positive_evidence": [f"{a_id}: {a['statement']}", f"{b_id}: {b['statement']}"],
                    "negative_evidence": [f"{a_id}: {a_status}", f"{b_id}: {b_status}"],
                    "jaccard_overlap": round(score, 6),
                    "notes": "Semantically similar claims have opposite truth-status categories.",
                }
            )

    # Section-level contradictions from narrative corpora
    corpora = [
        ("registry/docs_root_narratives.toml", docs_root_rows),
        ("registry/research_narratives.toml", research_rows),
        ("registry/external_sources.toml", external_rows),
        ("registry/data_artifact_narratives.toml", artifact_rows),
    ]
    for source_registry, rows in corpora:
        for row in rows:
            body = str(row.get("body_markdown", ""))
            if not body.strip():
                continue
            source_markdown = _collapse(str(row.get("source_markdown", "")))
            source_uid = _collapse(str(row.get("id", "")))
            for section in _split_sections(body):
                text = _collapse(str(section["body"]))
                if not text:
                    continue
                lower = text.lower()
                pos_hits = [term for term in POSITIVE_TERMS if term in lower]
                neg_hits = [term for term in NEGATIVE_TERMS if term in lower]
                if not pos_hits or not neg_hits:
                    continue
                seq += 1
                claim_refs = _extract_claim_refs(text)
                markers.append(
                    {
                        "id": f"CM-{seq:04d}",
                        "marker_kind": "section_polarity_conflict",
                        "severity": "medium",
                        "status": "open",
                        "claim_refs": claim_refs,
                        "source_registry": source_registry,
                        "source_document": source_markdown or source_uid,
                        "section_label": _collapse(str(section["title"])),
                        "line_start": int(section["line_start"]),
                        "line_end": int(section["line_end"]),
                        "positive_evidence": sorted(set(pos_hits)),
                        "negative_evidence": sorted(set(neg_hits)),
                        "jaccard_overlap": 0.0,
                        "notes": "Section contains both positive and negative validation language.",
                    }
                )

    markers.sort(key=lambda item: item["id"])
    severity_counts = Counter(item["severity"] for item in markers)
    kind_counts = Counter(item["marker_kind"] for item in markers)
    meta = {
        "marker_count": len(markers),
        "high_severity_count": severity_counts.get("high", 0),
        "medium_severity_count": severity_counts.get("medium", 0),
        "low_severity_count": severity_counts.get("low", 0),
        "kind_count": len(kind_counts),
    }
    return markers, meta


def build_lacunae(
    claims_rows: list[dict[str, Any]],
    conflict_markers: list[dict[str, Any]],
    insights_rows: list[dict[str, Any]],
    experiments_rows: list[dict[str, Any]],
    legacy_lacunae: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Preserve existing manual lacunae first
    for row in legacy_lacunae:
        lid = _collapse(str(row.get("id", "")))
        if not lid or lid in seen_ids:
            continue
        seen_ids.add(lid)
        rows.append(
            {
                "id": lid,
                "origin": "legacy_manual",
                "area": _collapse(str(row.get("area", ""))) or "general",
                "title": _collapse(str(row.get("title", ""))),
                "description": _collapse(str(row.get("description", ""))),
                "priority": _collapse(str(row.get("priority", ""))) or "medium",
                "status": _collapse(str(row.get("status", ""))) or "open",
                "claim_refs": _extract_claim_refs(str(row.get("description", ""))),
                "source_refs": [],
                "related_marker_ids": [],
            }
        )

    # Unresolved claims
    for row in claims_rows:
        cid = _collapse(str(row.get("id", "")))
        status = _status_token(str(row.get("status", "")))
        if status not in UNRESOLVED_STATUS_TOKENS:
            continue
        lid = f"L-{cid[2:]}" if cid.startswith("C-") else f"L-AUTO-{len(rows)+1:04d}"
        if lid in seen_ids:
            continue
        seen_ids.add(lid)
        statement = _collapse(str(row.get("statement", "")))
        rows.append(
            {
                "id": lid,
                "origin": "claims_status_scan",
                "area": "claims",
                "title": f"Unresolved claim status for {cid}",
                "description": f"{cid} remains unresolved with status token {status}: {statement}",
                "priority": "high" if status in {"INCONCLUSIVE", "PARTIAL"} else "medium",
                "status": "open",
                "claim_refs": [cid],
                "source_refs": ["registry/claims.toml"],
                "related_marker_ids": [],
            }
        )

    # Conflict markers -> lacunae entries
    for marker in conflict_markers:
        mid = _collapse(str(marker.get("id", "")))
        if not mid:
            continue
        lid = f"L-CM-{mid.split('-')[-1]}"
        if lid in seen_ids:
            continue
        seen_ids.add(lid)
        claim_refs = [str(v) for v in marker.get("claim_refs", []) if str(v).strip()]
        rows.append(
            {
                "id": lid,
                "origin": "conflict_marker_scan",
                "area": "consistency",
                "title": f"Resolve conflict marker {mid}",
                "description": _collapse(str(marker.get("notes", ""))) or "Unresolved contradiction marker.",
                "priority": "high" if str(marker.get("severity", "")) == "high" else "medium",
                "status": "open",
                "claim_refs": sorted(set(claim_refs)),
                "source_refs": [str(marker.get("source_registry", "")), str(marker.get("source_document", ""))],
                "related_marker_ids": [mid],
            }
        )

    # Insight/experiment claim refs not present in claims registry (defensive)
    claim_ids = {str(row.get("id", "")) for row in claims_rows}
    dangling = 0
    for source_name, source_rows in (
        ("insights", insights_rows),
        ("experiments", experiments_rows),
    ):
        for row in source_rows:
            sid = _collapse(str(row.get("id", "")))
            refs = [str(v) for v in row.get("claims", []) if str(v).strip()]
            missing = sorted({ref for ref in refs if ref not in claim_ids})
            if not missing:
                continue
            dangling += 1
            lid = f"L-DANGLING-{source_name[:1].upper()}-{dangling:03d}"
            if lid in seen_ids:
                continue
            seen_ids.add(lid)
            rows.append(
                {
                    "id": lid,
                    "origin": "crossref_scan",
                    "area": source_name,
                    "title": f"Dangling claim references in {source_name} entry {sid}",
                    "description": f"Entry {sid} references unknown claims: {', '.join(missing)}",
                    "priority": "high",
                    "status": "open",
                    "claim_refs": [],
                    "source_refs": [f"registry/{source_name}.toml"],
                    "related_marker_ids": [],
                }
            )

    rows.sort(key=lambda item: item["id"])
    priority_counts = Counter(item["priority"] for item in rows)
    origin_counts = Counter(item["origin"] for item in rows)
    meta = {
        "lacuna_count": len(rows),
        "open_count": sum(1 for row in rows if row.get("status") == "open"),
        "high_priority_count": priority_counts.get("high", 0),
        "medium_priority_count": priority_counts.get("medium", 0),
        "low_priority_count": priority_counts.get("low", 0),
        "origin_kind_count": len(origin_counts),
    }
    return rows, meta


def _shape_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "table",
            "keys": sorted(value.keys()),
        }
    if isinstance(value, list):
        if not value:
            return {"type": "array", "row_count": 0, "entry_kind": "empty"}
        if all(isinstance(item, dict) for item in value):
            key_sets = [set(item.keys()) for item in value]
            intersection = sorted(set.intersection(*key_sets)) if key_sets else []
            union = sorted(set.union(*key_sets)) if key_sets else []
            return {
                "type": "array",
                "row_count": len(value),
                "entry_kind": "table",
                "required_keys": intersection,
                "union_keys": union,
            }
        elem_types = sorted({type(item).__name__ for item in value})
        return {
            "type": "array",
            "row_count": len(value),
            "entry_kind": "scalar_or_mixed",
            "entry_types": elem_types,
        }
    return {"type": type(value).__name__}


def build_schema_signatures(root: Path, registry_paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    seq = 0
    for rel in registry_paths:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        data = tomllib.loads(text)
        top_level = sorted(data.keys())
        shapes = {key: _shape_summary(data[key]) for key in top_level}
        schema_payload = {
            "path": rel,
            "top_level_keys": top_level,
            "shapes": shapes,
        }
        normalized = json.dumps(schema_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        schema_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seq += 1
        rows.append(
            {
                "id": f"SIG-{seq:04d}",
                "path": rel,
                "top_level_keys": top_level,
                "schema_version": "v1",
                "schema_sha256": schema_sha256,
                "content_sha256": content_sha256,
                "array_table_count": sum(
                    1
                    for key in top_level
                    if isinstance(data[key], list) and all(isinstance(item, dict) for item in data[key])
                ),
                "table_count": sum(1 for key in top_level if isinstance(data[key], dict)),
                "shape_json": normalized,
            }
        )
    rows.sort(key=lambda item: item["path"])
    for idx, row in enumerate(rows, start=1):
        row["id"] = f"SIG-{idx:04d}"
    meta = {
        "signature_count": len(rows),
        "version": 1,
    }
    return rows, meta


def _render_conflict_markers(rows: list[dict[str, Any]], meta: dict[str, int]) -> str:
    lines: list[str] = []
    lines.append("# Conflict marker registry (Wave 5 strict schema).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch3_registries.py.")
    lines.append("")
    lines.append("[conflict_markers]")
    lines.append('updated = "2026-02-10"')
    lines.append("authoritative = true")
    lines.append(f"marker_count = {meta['marker_count']}")
    lines.append(f"high_severity_count = {meta['high_severity_count']}")
    lines.append(f"medium_severity_count = {meta['medium_severity_count']}")
    lines.append(f"low_severity_count = {meta['low_severity_count']}")
    lines.append(f"kind_count = {meta['kind_count']}")
    lines.append("")
    for row in rows:
        lines.append("[[marker]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"marker_kind = {_q(row['marker_kind'])}")
        lines.append(f"severity = {_q(row['severity'])}")
        lines.append(f"status = {_q(row['status'])}")
        lines.append(f"claim_refs = {_render_list(row['claim_refs'])}")
        lines.append(f"source_registry = {_q(row['source_registry'])}")
        lines.append(f"source_document = {_q(row['source_document'])}")
        lines.append(f"section_label = {_q(row['section_label'])}")
        lines.append(f"line_start = {row['line_start']}")
        lines.append(f"line_end = {row['line_end']}")
        lines.append(f"positive_evidence = {_render_list(row['positive_evidence'])}")
        lines.append(f"negative_evidence = {_render_list(row['negative_evidence'])}")
        lines.append(f"jaccard_overlap = {row['jaccard_overlap']:.6f}")
        lines.append(f"notes = {_q(row['notes'])}")
        lines.append("")
    return "\n".join(lines)


def _render_lacunae(rows: list[dict[str, Any]], meta: dict[str, int]) -> str:
    lines: list[str] = []
    lines.append("# Lacunae registry (Wave 5 strict schema).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch3_registries.py.")
    lines.append("")
    lines.append("[lacunae]")
    lines.append('updated = "2026-02-10"')
    lines.append("authoritative = true")
    lines.append('status = "active"')
    lines.append(f"lacuna_count = {meta['lacuna_count']}")
    lines.append(f"open_count = {meta['open_count']}")
    lines.append(f"high_priority_count = {meta['high_priority_count']}")
    lines.append(f"medium_priority_count = {meta['medium_priority_count']}")
    lines.append(f"low_priority_count = {meta['low_priority_count']}")
    lines.append(f"origin_kind_count = {meta['origin_kind_count']}")
    lines.append("")
    for row in rows:
        lines.append("[[lacuna]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"origin = {_q(row['origin'])}")
        lines.append(f"area = {_q(row['area'])}")
        lines.append(f"title = {_q(row['title'])}")
        lines.append(f"description = {_q(row['description'])}")
        lines.append(f"priority = {_q(row['priority'])}")
        lines.append(f"status = {_q(row['status'])}")
        lines.append(f"claim_refs = {_render_list(row['claim_refs'])}")
        lines.append(f"source_refs = {_render_list(row['source_refs'])}")
        lines.append(f"related_marker_ids = {_render_list(row['related_marker_ids'])}")
        lines.append("")
    return "\n".join(lines)


def _render_schema_signatures(rows: list[dict[str, Any]], meta: dict[str, int], registry_paths: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Registry schema signatures (Wave 5 strict schema).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch3_registries.py.")
    lines.append("")
    lines.append("[schema_signatures]")
    lines.append('updated = "2026-02-10"')
    lines.append("authoritative = true")
    lines.append(f"version = {meta['version']}")
    lines.append(f"signature_count = {meta['signature_count']}")
    lines.append(f"registry_paths = {_render_list(sorted(registry_paths))}")
    lines.append("")
    for row in rows:
        lines.append("[[signature]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"path = {_q(row['path'])}")
        lines.append(f"top_level_keys = {_render_list(row['top_level_keys'])}")
        lines.append(f"schema_version = {_q(row['schema_version'])}")
        lines.append(f"schema_sha256 = {_q(row['schema_sha256'])}")
        lines.append(f"content_sha256 = {_q(row['content_sha256'])}")
        lines.append(f"array_table_count = {row['array_table_count']}")
        lines.append(f"table_count = {row['table_count']}")
        lines.append(f"shape_json = {_q(row['shape_json'])}")
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
        "--conflict-out",
        default="registry/conflict_markers.toml",
        help="Conflict marker registry output path.",
    )
    parser.add_argument(
        "--lacunae-out",
        default="registry/lacunae.toml",
        help="Lacunae registry output path.",
    )
    parser.add_argument(
        "--schema-out",
        default="registry/schema_signatures.toml",
        help="Schema signature registry output path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    claims_rows = _load(root / "registry/claims.toml").get("claim", [])
    insights_rows = _load(root / "registry/insights.toml").get("insight", [])
    experiments_rows = _load(root / "registry/experiments.toml").get("experiment", [])
    docs_root_rows = _load(root / "registry/docs_root_narratives.toml").get("document", [])
    research_rows = _load(root / "registry/research_narratives.toml").get("document", [])
    external_rows = _load(root / "registry/external_sources.toml").get("document", [])
    artifact_rows = _load(root / "registry/data_artifact_narratives.toml").get("document", [])

    legacy_lacunae = []
    legacy_path = root / "registry/lacunae.toml"
    if legacy_path.exists():
        legacy_lacunae = _load(legacy_path).get("lacuna", [])

    conflict_rows, conflict_meta = build_conflict_markers(
        claims_rows=claims_rows,
        docs_root_rows=docs_root_rows,
        research_rows=research_rows,
        external_rows=external_rows,
        artifact_rows=artifact_rows,
    )
    lacunae_rows, lacunae_meta = build_lacunae(
        claims_rows=claims_rows,
        conflict_markers=conflict_rows,
        insights_rows=insights_rows,
        experiments_rows=experiments_rows,
        legacy_lacunae=legacy_lacunae,
    )

    registry_paths = [
        "registry/artifact_experiment_links.toml",
        "registry/claims.toml",
        "registry/insights.toml",
        "registry/experiments.toml",
        "registry/external_sources.toml",
        "registry/project_csv_canonical_datasets.toml",
        "registry/project_csv_canonical.toml",
        "registry/claims_atoms.toml",
        "registry/claims_evidence_edges.toml",
        "registry/knowledge/equation_atoms_v2.toml",
        "registry/knowledge/equation_symbol_table.toml",
        "registry/knowledge/proof_skeletons.toml",
        "registry/knowledge/derivation_steps.toml",
        "registry/bibliography.toml",
        "registry/bibliography_normalized.toml",
        "registry/provenance_sources.toml",
        "registry/narrative_paragraph_atoms.toml",
        "registry/conflict_markers.toml",
        "registry/lacunae.toml",
        "registry/registry_events.toml",
        "registry/third_party_markdown_cache.toml",
        "registry/third_party_source_verification.toml",
    ]
    conflict_text = _render_conflict_markers(conflict_rows, conflict_meta)
    lacunae_text = _render_lacunae(lacunae_rows, lacunae_meta)
    _write(root / args.conflict_out, conflict_text)
    _write(root / args.lacunae_out, lacunae_text)

    # Build signatures after writing conflict/lacuna registries so checksums reflect
    # the newly materialized canonical state.
    signature_rows, signature_meta = build_schema_signatures(root=root, registry_paths=registry_paths)
    schema_text = _render_schema_signatures(signature_rows, signature_meta, registry_paths)
    _write(root / args.schema_out, schema_text)

    print(
        "Wrote Wave5 Batch3 registries: "
        f"conflict_markers={len(conflict_rows)} "
        f"lacunae={len(lacunae_rows)} "
        f"schema_signatures={len(signature_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
