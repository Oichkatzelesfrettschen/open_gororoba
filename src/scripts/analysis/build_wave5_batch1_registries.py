#!/usr/bin/env python3
"""
Build Wave 5 Batch 1 strict TOML registries:
- W5-007: registry/claims_atoms.toml
- W5-008: registry/claims_evidence_edges.toml
- W5-009: registry/knowledge/equation_atoms_v2.toml
- W5-009: registry/knowledge/equation_symbol_table.toml
- W5-010: registry/knowledge/proof_skeletons.toml
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections import defaultdict
from pathlib import Path


CLAIM_ID_RE = re.compile(r"\bC-\d{3}\b")
EVIDENCE_ID_RE = re.compile(r"\b(?:C|I|E)-\d{3}\b")
BACKTICK_RE = re.compile(r"`([^`]+)`")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
INLINE_MATH_RE = re.compile(r"\$([^$\n]{2,260})\$")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
LATEX_CMD_RE = re.compile(r"\\([A-Za-z]+)")


STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "over",
    "under",
    "after",
    "before",
    "where",
    "when",
    "while",
    "line",
    "note",
    "data",
    "source",
    "truth",
    "auto",
    "generated",
    "edit",
    "not",
    "registry",
    "toml",
    "markdown",
}


PROOF_KINDS = {
    "proof",
    "theorem",
    "lemma",
    "corollary",
    "proposition",
    "axiom",
    "derivation",
    "hypothesis_block",
    "argument_section",
}


def _q(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _render_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_q(item) for item in values) + "]"


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
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _status_token(status: str) -> str:
    token = _collapse(status).upper()
    token = token.replace("/", "_").replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^A-Z0-9_]", "", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "UNSPECIFIED"


def _hypothesis_class(statement: str, status: str) -> str:
    text = f"{statement} {status}".lower()
    if "falsifiable thesis" in text or "falsifiable" in text:
        return "falsifiable_thesis"
    if "toy" in text:
        return "toy_model"
    if "speculative" in text:
        return "speculative_claim"
    if "refuted" in text:
        return "refuted_claim"
    if "verified" in text:
        return "verified_claim"
    if "closed" in text:
        return "closed_claim"
    return "research_claim"


def _extract_backtick_refs(text: str) -> list[str]:
    refs: set[str] = set()
    for match in BACKTICK_RE.finditer(text):
        payload = match.group(1)
        for part in re.split(r"[,\n]", payload):
            candidate = _collapse(part).strip(" .;:()[]{}")
            if not candidate:
                continue
            if candidate.startswith("C-") and len(candidate) == 5:
                continue
            if "/" in candidate or "." in candidate:
                refs.add(candidate)
    return sorted(refs)


def _extract_claim_links(text: str) -> list[str]:
    return sorted(set(EVIDENCE_ID_RE.findall(text)))


def _split_sections(body: str) -> list[tuple[str, int, int, int, list[str]]]:
    lines = _ascii_clean(body).splitlines()
    sections: list[tuple[str, int, int, int, list[str]]] = []
    title = "(root)"
    level = 0
    start = 1
    buf: list[str] = []
    for idx, raw in enumerate(lines, start=1):
        hm = HEADING_RE.match(raw)
        if hm:
            sections.append((title, level, start, max(start, idx - 1), buf))
            title = _collapse(hm.group(2))
            level = len(hm.group(1))
            start = idx
            buf = []
            continue
        buf.append(raw)
    sections.append((title, level, start, max(start, len(lines)), buf))
    return sections


def _load_claims(claims_path: Path) -> list[dict]:
    raw = tomllib.loads(claims_path.read_text(encoding="utf-8"))
    claims = raw.get("claim", [])
    if not claims:
        raise SystemExit(f"ERROR: no claim rows found in {claims_path}")
    out: list[dict] = []
    for row in claims:
        claim_id = _collapse(str(row.get("id", "")))
        statement = _collapse(str(row.get("statement", "")))
        where_stated = str(row.get("where_stated", ""))
        verify_rule = str(row.get("what_would_verify_refute", ""))
        status_detail = _collapse(str(row.get("status", "")))
        where_refs = _extract_backtick_refs(where_stated)
        verify_refs = _extract_backtick_refs(verify_rule)
        cross_refs = _extract_claim_links(where_stated + " " + verify_rule)
        out.append(
            {
                "claim_id": claim_id,
                "statement": statement,
                "status_detail": status_detail,
                "status_token": _status_token(status_detail),
                "last_verified": _collapse(str(row.get("last_verified", ""))),
                "where_stated": _collapse(where_stated),
                "verification_rule": _collapse(verify_rule),
                "hypothesis_class": _hypothesis_class(statement, status_detail),
                "where_stated_refs": where_refs,
                "verification_refs": verify_refs,
                "cross_refs": cross_refs,
            }
        )
    out.sort(key=lambda item: item["claim_id"])
    return out


def _edge_target_kind(ref: str) -> str:
    if re.fullmatch(r"C-\d{3}", ref):
        return "claim_ref"
    if re.fullmatch(r"I-\d{3}", ref):
        return "insight_ref"
    if re.fullmatch(r"E-\d{3}", ref):
        return "experiment_ref"
    if ref.startswith("registry/"):
        return "registry_path"
    if ref.endswith(".rs"):
        return "rust_source_path"
    if ref.endswith(".py"):
        return "python_source_path"
    if ref.endswith(".toml"):
        return "toml_registry_path"
    if ref.endswith(".csv"):
        return "csv_artifact_path"
    if ref.endswith(".md"):
        return "markdown_path"
    return "generic_reference"


def _build_claim_edges(claim_atoms: list[dict]) -> list[dict]:
    edges: dict[tuple[str, str, str], dict] = {}
    for atom in claim_atoms:
        claim_id = atom["claim_id"]
        for ref in atom["where_stated_refs"]:
            key = (claim_id, "where_stated", ref)
            edges[key] = {
                "claim_id": claim_id,
                "edge_role": "where_stated",
                "target_ref": ref,
                "target_kind": _edge_target_kind(ref),
            }
        for ref in atom["verification_refs"]:
            key = (claim_id, "verification_rule", ref)
            edges[key] = {
                "claim_id": claim_id,
                "edge_role": "verification_rule",
                "target_ref": ref,
                "target_kind": _edge_target_kind(ref),
            }
        for ref in atom["cross_refs"]:
            key = (claim_id, "cross_reference", ref)
            edges[key] = {
                "claim_id": claim_id,
                "edge_role": "cross_reference",
                "target_ref": ref,
                "target_kind": _edge_target_kind(ref),
            }
    out = list(edges.values())
    out.sort(key=lambda item: (item["claim_id"], item["edge_role"], item["target_ref"]))
    return out


def _load_narrative_sources(repo_root: Path) -> list[dict]:
    sources: list[dict] = []
    for registry_path, source_group in (
        ("registry/docs_root_narratives.toml", "docs_root_narrative"),
        ("registry/research_narratives.toml", "research_narrative"),
        ("registry/data_artifact_narratives.toml", "data_artifact_narrative"),
    ):
        raw = tomllib.loads((repo_root / registry_path).read_text(encoding="utf-8"))
        for row in raw.get("document", []):
            source_uid = _collapse(str(row.get("id", "")))
            if not source_uid:
                continue
            body = str(row.get("body_markdown", ""))
            if not body.strip():
                continue
            sources.append(
                {
                    "source_uid": source_uid,
                    "source_group": source_group,
                    "source_registry": registry_path,
                    "source_path": _collapse(str(row.get("source_markdown", ""))),
                    "title": _collapse(str(row.get("title", source_uid))),
                    "body": body,
                }
            )
    sources.sort(key=lambda item: (item["source_group"], item["source_uid"]))
    return sources


def _parse_relation(expr: str) -> tuple[str, str, str]:
    for token in ("<=", ">=", "!=", "->", "="):
        if token in expr:
            lhs, rhs = expr.split(token, 1)
            lhs_c = _collapse(lhs)
            rhs_c = _collapse(rhs)
            if lhs_c:
                return token, lhs_c, rhs_c
    return "implicit", _collapse(expr), ""


def _infer_equation_kind(expr: str) -> str:
    lowered = expr.lower()
    if "->" in expr:
        return "mapping_relation"
    if any(token in lowered for token in ("d/d", "partial", "nabla", "delta", "laplacian")):
        return "differential_relation"
    if any(token in expr for token in ("<=", ">=", "!=", "<", ">")):
        return "constraint_relation"
    return "algebraic_relation"


def _extract_symbols(expr: str) -> tuple[list[str], list[str]]:
    symbols: set[str] = set()
    for cmd in LATEX_CMD_RE.findall(expr):
        token = _collapse(cmd)
        if token:
            symbols.add(token)
    for ident in IDENTIFIER_RE.findall(expr):
        token = _collapse(ident)
        if not token:
            continue
        if token.lower() in STOPWORDS:
            continue
        symbols.add(token)
    numbers = sorted(set(NUMBER_RE.findall(expr)))
    return sorted(symbols), numbers


def _quality_flags(expr: str, relation: str, symbol_names: list[str]) -> list[str]:
    flags: list[str] = []
    lowered = expr.lower()
    if "<!--" in lowered or "auto-generated" in lowered or "source of truth" in lowered:
        flags.append("header_noise")
    word_count = len(_collapse(expr).split())
    symbol_density = sum(expr.count(ch) for ch in ("=", "+", "-", "*", "/", "^", "_", "(", ")"))
    if word_count > 24 and symbol_density < 2:
        flags.append("text_heavy_fragment")
    if not symbol_names:
        flags.append("no_symbol_extracted")
    if relation == "implicit":
        flags.append("implicit_relation")
    return flags


def _extract_equation_atoms_v2(sources: list[dict]) -> list[dict]:
    atoms: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    for source in sources:
        section_title = "(root)"
        in_code_fence = False
        lines = _ascii_clean(source["body"]).splitlines()
        for line_no, raw in enumerate(lines, start=1):
            hm = HEADING_RE.match(raw)
            if hm:
                section_title = _collapse(hm.group(2))
                continue
            stripped = raw.strip()
            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                continue
            if in_code_fence:
                continue
            if (
                "auto-generated" in stripped.lower()
                or "source of truth" in stripped.lower()
                or stripped.startswith("<!--")
            ):
                continue

            candidates: list[tuple[str, str]] = []
            for inline in INLINE_MATH_RE.findall(raw):
                expr = _collapse(inline)
                if len(expr) >= 2:
                    candidates.append(("inline_math", expr))

            line_candidate = stripped
            if line_candidate.startswith("- "):
                line_candidate = line_candidate[2:].strip()
            if line_candidate and not line_candidate.startswith(("#", "|", "*", "<!--")):
                if any(op in line_candidate for op in ("=", "->", "<=", ">=", "!=")):
                    if re.search(r"[A-Za-z_\\]", line_candidate):
                        if len(line_candidate) <= 240:
                            candidates.append(("equation_like_line", _collapse(line_candidate)))

            for extraction_kind, expr in candidates:
                key = (source["source_uid"], line_no, expr)
                if key in seen:
                    continue
                seen.add(key)
                relation_operator, lhs, rhs = _parse_relation(expr)
                symbol_names, numeric_constants = _extract_symbols(expr)
                if not symbol_names:
                    if numeric_constants:
                        symbol_names = ["NUMERIC_LITERAL"]
                    else:
                        symbol_names = ["IMPLICIT_SYMBOL"]
                flags = _quality_flags(expr, relation_operator, symbol_names)
                if "header_noise" in flags:
                    continue
                if "text_heavy_fragment" in flags and extraction_kind == "equation_like_line":
                    continue
                confidence = "high"
                if relation_operator == "implicit":
                    confidence = "medium"
                if "no_symbol_extracted" in flags:
                    confidence = "low"
                claim_refs = sorted(set(CLAIM_ID_RE.findall(expr)))
                atoms.append(
                    {
                        "source_uid": source["source_uid"],
                        "source_group": source["source_group"],
                        "source_registry": source["source_registry"],
                        "source_path": source["source_path"],
                        "section_title": section_title,
                        "source_line": line_no,
                        "expression": expr,
                        "normalized_expression": _collapse(expr),
                        "relation_operator": relation_operator,
                        "lhs_expression": lhs,
                        "rhs_expression": rhs,
                        "equation_kind": _infer_equation_kind(expr),
                        "extraction_kind": extraction_kind,
                        "symbol_names": symbol_names,
                        "numeric_constants": numeric_constants,
                        "quality_flags": sorted(flags),
                        "extraction_confidence": confidence,
                        "claim_refs": claim_refs,
                    }
                )
    atoms.sort(
        key=lambda item: (
            item["source_uid"],
            int(item["source_line"]),
            item["expression"],
        )
    )
    return atoms


def _symbol_category(symbol: str) -> str:
    lower = symbol.lower()
    if lower in {
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "nabla",
        "partial",
        "delta",
        "sum",
        "prod",
        "int",
        "lim",
    }:
        return "operator_or_function"
    if symbol.startswith("mathbb"):
        return "set_marker"
    if len(symbol) == 1 and symbol.isalpha():
        return "scalar_variable"
    if "_" in symbol:
        return "indexed_symbol"
    if symbol.isupper():
        return "constant_or_group"
    return "identifier"


def _build_symbol_table(equation_atoms: list[dict]) -> tuple[list[dict], dict[str, str]]:
    usage: dict[str, dict] = {}
    for atom in equation_atoms:
        eq_id = atom["id"]
        source_uid = atom["source_uid"]
        for symbol in atom["symbol_names"]:
            state = usage.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "usage_count": 0,
                    "source_uids": set(),
                    "equation_ids": [],
                },
            )
            state["usage_count"] += 1
            state["source_uids"].add(source_uid)
            if len(state["equation_ids"]) < 12:
                state["equation_ids"].append(eq_id)

    symbol_rows: list[dict] = []
    for symbol, state in usage.items():
        symbol_rows.append(
            {
                "symbol": symbol,
                "normalized_symbol": symbol.lower(),
                "category": _symbol_category(symbol),
                "usage_count": state["usage_count"],
                "source_uids": sorted(state["source_uids"]),
                "example_equation_ids": list(state["equation_ids"]),
            }
        )

    symbol_rows.sort(key=lambda row: (-int(row["usage_count"]), row["symbol"].lower()))
    symbol_ref_map: dict[str, str] = {}
    for idx, row in enumerate(symbol_rows, start=1):
        symbol_id = f"SYM-{idx:04d}"
        row["id"] = symbol_id
        symbol_ref_map[row["symbol"]] = symbol_id
    return symbol_rows, symbol_ref_map


def _load_proof_atoms(proof_atoms_path: Path) -> list[dict]:
    if not proof_atoms_path.exists():
        return []
    raw = tomllib.loads(proof_atoms_path.read_text(encoding="utf-8"))
    return list(raw.get("atom", []))


def _build_proof_skeletons(claim_atoms: list[dict], proof_atoms: list[dict]) -> list[dict]:
    skeletons: list[dict] = []
    for atom in claim_atoms:
        assumptions = [value for value in [atom.get("h0", ""), atom.get("h1", "")] if value]
        rule = atom.get("decision_rule", "") or atom.get("verification_rule", "")
        if not assumptions and not rule:
            assumptions = [atom["statement"]]
        derivation_steps: list[str] = []
        for segment in re.split(r"[|.;]", atom.get("verification_rule", "")):
            piece = _collapse(segment)
            if piece:
                derivation_steps.append(piece)
            if len(derivation_steps) >= 8:
                break
        skeletons.append(
            {
                "skeleton_kind": "claim_decision_rule",
                "theorem_label": atom["claim_id"],
                "claim_id": atom["claim_id"],
                "assumptions": assumptions[:8],
                "obligations": [f"Establish evidence-backed status for {atom['claim_id']}."],
                "derivation_steps": derivation_steps[:12],
                "decision_rule": _collapse(rule),
                "conclusion": _collapse(f"{atom['status_token']}: {atom['status_detail']}"),
                "source_uid": atom["claim_id"],
                "source_registry": "registry/claims.toml",
                "source_path": "registry/claims.toml",
                "line_start": 0,
                "line_end": 0,
                "claim_refs": sorted(set([atom["claim_id"]] + atom["cross_refs"])),
                "evidence_refs": sorted(
                    set(atom["where_stated_refs"] + atom["verification_refs"] + atom["cross_refs"])
                ),
            }
        )

    for row in proof_atoms:
        kind = _collapse(str(row.get("proof_kind", "argument_section"))).lower()
        if kind not in PROOF_KINDS:
            kind = "argument_section"
        claim_refs = sorted(set(str(item) for item in row.get("claim_refs", [])))
        assumptions = [_collapse(str(item)) for item in row.get("assumption_lines", []) if _collapse(str(item))]
        derivation_steps = [
            _collapse(str(item))
            for item in row.get("inference_markers", [])
            if _collapse(str(item))
        ]
        if not derivation_steps:
            excerpt = _collapse(str(row.get("excerpt", "")))
            if excerpt:
                derivation_steps = [excerpt]
        obligations = [
            _collapse(str(item))
            for item in row.get("decision_lines", [])
            if _collapse(str(item))
        ]
        skeletons.append(
            {
                "skeleton_kind": kind,
                "theorem_label": _collapse(str(row.get("section_title", ""))),
                "claim_id": claim_refs[0] if claim_refs else "",
                "assumptions": assumptions[:10],
                "obligations": obligations[:10],
                "derivation_steps": derivation_steps[:12],
                "decision_rule": _collapse(str(row.get("decision_rule_text", ""))),
                "conclusion": _collapse(str(row.get("conclusion_text", ""))),
                "source_uid": _collapse(str(row.get("source_uid", ""))),
                "source_registry": _collapse(str(row.get("source_registry", ""))),
                "source_path": _collapse(str(row.get("source_path", ""))),
                "line_start": int(row.get("line_start", 0)),
                "line_end": int(row.get("line_end", 0)),
                "claim_refs": claim_refs,
                "evidence_refs": sorted(
                    set(
                        [_collapse(str(row.get("source_path", "")))]
                        + [_collapse(str(row.get("source_registry", "")))]
                        + claim_refs
                    )
                ),
            }
        )

    dedup: dict[tuple[str, str, str, str], dict] = {}
    for row in skeletons:
        key = (
            row["skeleton_kind"],
            row["theorem_label"],
            row["source_uid"],
            row["claim_id"],
        )
        dedup[key] = row
    out = list(dedup.values())
    out.sort(key=lambda item: (item["claim_id"], item["source_uid"], item["theorem_label"]))
    return out


def _render_claim_atoms(rows: list[dict]) -> str:
    lines = [
        "# Claim atoms registry (Wave 5 strict schema).",
        "# Generated by src/scripts/analysis/build_wave5_batch1_registries.py",
        "",
        "[claims_atoms]",
        'updated = "deterministic"',
        "authoritative = true",
        f"atom_count = {len(rows)}",
        f"unique_claim_count = {len({row['claim_id'] for row in rows})}",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                "[[atom]]",
                f"id = {_q(f'CLA-{idx:04d}')}",
                f"claim_id = {_q(row['claim_id'])}",
                f"statement = {_q(row['statement'])}",
                f"status_token = {_q(row['status_token'])}",
                f"status_detail = {_q(row['status_detail'])}",
                f"hypothesis_class = {_q(row['hypothesis_class'])}",
                f"last_verified = {_q(row['last_verified'])}",
                f"where_stated = {_q(row['where_stated'])}",
                f"verification_rule = {_q(row['verification_rule'])}",
                f"where_stated_refs = {_render_list(row['where_stated_refs'])}",
                f"verification_refs = {_render_list(row['verification_refs'])}",
                f"cross_refs = {_render_list(row['cross_refs'])}",
                "",
            ]
        )
    text = "\n".join(lines)
    _assert_ascii(text, "claims_atoms")
    return text


def _render_claim_edges(rows: list[dict]) -> str:
    lines = [
        "# Claim evidence edge registry (Wave 5 strict schema).",
        "# Generated by src/scripts/analysis/build_wave5_batch1_registries.py",
        "",
        "[claims_evidence_edges]",
        'updated = "deterministic"',
        "authoritative = true",
        f"edge_count = {len(rows)}",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                "[[edge]]",
                f"id = {_q(f'CED-{idx:05d}')}",
                f"claim_id = {_q(row['claim_id'])}",
                f"edge_role = {_q(row['edge_role'])}",
                f"target_ref = {_q(row['target_ref'])}",
                f"target_kind = {_q(row['target_kind'])}",
                "",
            ]
        )
    text = "\n".join(lines)
    _assert_ascii(text, "claims_evidence_edges")
    return text


def _render_equation_atoms_v2(rows: list[dict]) -> str:
    lines = [
        "# Equation atoms v2 registry (Wave 5 strict schema).",
        "# Generated by src/scripts/analysis/build_wave5_batch1_registries.py",
        "",
        "[knowledge_equation_atoms_v2]",
        'updated = "deterministic"',
        "authoritative = true",
        f"atom_count = {len(rows)}",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                "[[atom]]",
                f"id = {_q(f'EQA2-{idx:05d}')}",
                f"expression = {_q(row['expression'])}",
                f"normalized_expression = {_q(row['normalized_expression'])}",
                f"relation_operator = {_q(row['relation_operator'])}",
                f"lhs_expression = {_q(row['lhs_expression'])}",
                f"rhs_expression = {_q(row['rhs_expression'])}",
                f"equation_kind = {_q(row['equation_kind'])}",
                f"extraction_kind = {_q(row['extraction_kind'])}",
                f"extraction_confidence = {_q(row['extraction_confidence'])}",
                f"quality_flags = {_render_list(row['quality_flags'])}",
                f"symbol_names = {_render_list(row['symbol_names'])}",
                f"numeric_constants = {_render_list(row['numeric_constants'])}",
                f"symbol_refs = {_render_list(row['symbol_refs'])}",
                f"claim_refs = {_render_list(row['claim_refs'])}",
                f"source_uid = {_q(row['source_uid'])}",
                f"source_group = {_q(row['source_group'])}",
                f"source_registry = {_q(row['source_registry'])}",
                f"source_path = {_q(row['source_path'])}",
                f"section_title = {_q(row['section_title'])}",
                f"source_line = {int(row['source_line'])}",
                "",
            ]
        )
    text = "\n".join(lines)
    _assert_ascii(text, "equation_atoms_v2")
    return text


def _render_symbol_table(rows: list[dict]) -> str:
    lines = [
        "# Equation symbol table (Wave 5 strict schema).",
        "# Generated by src/scripts/analysis/build_wave5_batch1_registries.py",
        "",
        "[equation_symbol_table]",
        'updated = "deterministic"',
        "authoritative = true",
        f"symbol_count = {len(rows)}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                "[[symbol]]",
                f"id = {_q(row['id'])}",
                f"symbol = {_q(row['symbol'])}",
                f"normalized_symbol = {_q(row['normalized_symbol'])}",
                f"category = {_q(row['category'])}",
                f"usage_count = {int(row['usage_count'])}",
                f"source_uids = {_render_list(row['source_uids'])}",
                f"example_equation_ids = {_render_list(row['example_equation_ids'])}",
                "",
            ]
        )
    text = "\n".join(lines)
    _assert_ascii(text, "equation_symbol_table")
    return text


def _render_proof_skeletons(rows: list[dict]) -> str:
    lines = [
        "# Proof skeleton registry (Wave 5 strict schema).",
        "# Generated by src/scripts/analysis/build_wave5_batch1_registries.py",
        "",
        "[knowledge_proof_skeletons]",
        'updated = "deterministic"',
        "authoritative = true",
        f"skeleton_count = {len(rows)}",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                "[[skeleton]]",
                f"id = {_q(f'PRS-{idx:05d}')}",
                f"skeleton_kind = {_q(row['skeleton_kind'])}",
                f"theorem_label = {_q(row['theorem_label'])}",
                f"claim_id = {_q(row['claim_id'])}",
                f"assumptions = {_render_list(row['assumptions'])}",
                f"obligations = {_render_list(row['obligations'])}",
                f"derivation_steps = {_render_list(row['derivation_steps'])}",
                f"decision_rule = {_q(row['decision_rule'])}",
                f"conclusion = {_q(row['conclusion'])}",
                f"source_uid = {_q(row['source_uid'])}",
                f"source_registry = {_q(row['source_registry'])}",
                f"source_path = {_q(row['source_path'])}",
                f"line_start = {int(row['line_start'])}",
                f"line_end = {int(row['line_end'])}",
                f"claim_refs = {_render_list(row['claim_refs'])}",
                f"evidence_refs = {_render_list(row['evidence_refs'])}",
                "",
            ]
        )
    text = "\n".join(lines)
    _assert_ascii(text, "proof_skeletons")
    return text


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
        "--claims-path",
        default="registry/claims.toml",
        help="Canonical claims registry path.",
    )
    parser.add_argument(
        "--proof-atoms-path",
        default="registry/knowledge/proof_atoms.toml",
        help="Existing proof atoms path for skeleton enrichment.",
    )
    parser.add_argument(
        "--claims-atoms-out",
        default="registry/claims_atoms.toml",
        help="Output claims atoms path.",
    )
    parser.add_argument(
        "--claims-edges-out",
        default="registry/claims_evidence_edges.toml",
        help="Output claim evidence edges path.",
    )
    parser.add_argument(
        "--equation-atoms-out",
        default="registry/knowledge/equation_atoms_v2.toml",
        help="Output equation atoms v2 path.",
    )
    parser.add_argument(
        "--symbol-table-out",
        default="registry/knowledge/equation_symbol_table.toml",
        help="Output equation symbol table path.",
    )
    parser.add_argument(
        "--proof-skeletons-out",
        default="registry/knowledge/proof_skeletons.toml",
        help="Output proof skeleton registry path.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    claims = _load_claims(repo_root / args.claims_path)
    claim_edges = _build_claim_edges(claims)

    sources = _load_narrative_sources(repo_root)
    equation_atoms = _extract_equation_atoms_v2(sources)
    for idx, atom in enumerate(equation_atoms, start=1):
        atom["id"] = f"EQA2-{idx:05d}"
    symbol_rows, symbol_ref_map = _build_symbol_table(equation_atoms)
    for atom in equation_atoms:
        atom["symbol_refs"] = [symbol_ref_map[name] for name in atom["symbol_names"] if name in symbol_ref_map]

    proof_atoms = _load_proof_atoms(repo_root / args.proof_atoms_path)
    proof_skeletons = _build_proof_skeletons(claims, proof_atoms)

    claims_text = _render_claim_atoms(claims)
    edges_text = _render_claim_edges(claim_edges)
    eq_text = _render_equation_atoms_v2(equation_atoms)
    symbol_text = _render_symbol_table(symbol_rows)
    proof_text = _render_proof_skeletons(proof_skeletons)

    _write(repo_root / args.claims_atoms_out, claims_text)
    _write(repo_root / args.claims_edges_out, edges_text)
    _write(repo_root / args.equation_atoms_out, eq_text)
    _write(repo_root / args.symbol_table_out, symbol_text)
    _write(repo_root / args.proof_skeletons_out, proof_text)

    print(
        "Wrote Wave5 Batch1 registries: "
        f"claims_atoms={len(claims)} "
        f"claim_edges={len(claim_edges)} "
        f"equation_atoms_v2={len(equation_atoms)} "
        f"equation_symbols={len(symbol_rows)} "
        f"proof_skeletons={len(proof_skeletons)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
