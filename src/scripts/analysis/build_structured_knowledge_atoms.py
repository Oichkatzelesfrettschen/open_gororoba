#!/usr/bin/env python3
"""
Build stricter structured TOML registries for high-information knowledge corpora.

Selected corpora for this wave:
- registry/knowledge/docs/DOC-0023.toml (claims-heavy notes corpus)
- registry/research_narratives.toml (proof/derivation-heavy corpus)
- registry/data_artifact_narratives.toml (equation-heavy corpus)

Outputs:
- registry/knowledge/claim_atoms.toml
- registry/knowledge/equation_atoms.toml
- registry/knowledge/proof_atoms.toml
- registry/knowledge/structured_corpora.toml
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


CLAIM_ID_RE = re.compile(r"\bC-\d{3}\b")
CLAIM_HEADING_RE = re.compile(r"^###\s*(C-\d{3})\s*:\s*(.+?)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
INLINE_MATH_RE = re.compile(r"\$([^$\n]{3,240})\$")
BOLD_TOKEN_RE = re.compile(r"\*\*([^*]+)\*\*")

PROOF_KEYWORDS = (
    "proof",
    "theorem",
    "lemma",
    "corollary",
    "proposition",
    "axiom",
    "derivation",
    "hypothesis",
)


@dataclass(frozen=True)
class SourceDoc:
    source_uid: str
    source_group: str
    source_registry: str
    source_path: str
    title: str
    line_count: int
    claim_refs: list[str]
    body: str


@dataclass(frozen=True)
class Section:
    title: str
    level: int
    line_start: int
    line_end: int
    body_lines: list[str]


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
    return json.dumps(value, ensure_ascii=True)


def _render_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(_esc(item) for item in items) + "]"


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _split_sections(body: str) -> list[Section]:
    lines = _ascii_sanitize(body).splitlines()
    sections: list[Section] = []
    current_title = "(root)"
    current_level = 0
    current_start = 1
    current_body: list[str] = []

    for idx, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if match:
            sections.append(
                Section(
                    title=current_title,
                    level=current_level,
                    line_start=current_start,
                    line_end=max(current_start, idx - 1),
                    body_lines=current_body,
                )
            )
            current_title = _collapse_ws(match.group(2))
            current_level = len(match.group(1))
            current_start = idx
            current_body = []
            continue
        current_body.append(line)

    sections.append(
        Section(
            title=current_title,
            level=current_level,
            line_start=current_start,
            line_end=max(current_start, len(lines)),
            body_lines=current_body,
        )
    )
    return sections


def _load_sources(repo_root: Path) -> list[SourceDoc]:
    sources: list[SourceDoc] = []

    doc_0023_path = repo_root / "registry/knowledge/docs/DOC-0023.toml"
    doc_0023 = tomllib.loads(doc_0023_path.read_text(encoding="utf-8"))
    payload = doc_0023.get("document", {})
    sources.append(
        SourceDoc(
            source_uid="DOC-0023",
            source_group="doc_claim_matrix",
            source_registry="registry/knowledge/docs/DOC-0023.toml",
            source_path=str(payload.get("source_path", "")),
            title=_collapse_ws(str(payload.get("title", "DOC-0023"))),
            line_count=int(payload.get("source_line_count", 0)),
            claim_refs=sorted(set(CLAIM_ID_RE.findall(str(payload.get("content_markdown", ""))))),
            body=str(payload.get("content_markdown", "")),
        )
    )

    for registry_path, source_group in (
        ("registry/research_narratives.toml", "research_narrative"),
        ("registry/data_artifact_narratives.toml", "data_artifact_narrative"),
    ):
        raw = tomllib.loads((repo_root / registry_path).read_text(encoding="utf-8"))
        for row in raw.get("document", []):
            source_uid = str(row.get("id", "")).strip()
            source_path = str(row.get("source_markdown", "")).strip()
            if not source_uid or not source_path:
                continue
            body = str(row.get("body_markdown", ""))
            sources.append(
                SourceDoc(
                    source_uid=source_uid,
                    source_group=source_group,
                    source_registry=registry_path,
                    source_path=source_path,
                    title=_collapse_ws(str(row.get("title", source_uid))),
                    line_count=int(row.get("line_count", 0)),
                    claim_refs=sorted(set(str(item) for item in row.get("claim_refs", []))),
                    body=body,
                )
            )

    sources.sort(key=lambda item: (item.source_group, item.source_uid))
    return sources


def _parse_claim_rows(doc: SourceDoc) -> list[dict[str, object]]:
    lines = _ascii_sanitize(doc.body).splitlines()
    atoms: list[dict[str, object]] = []
    claim_row_by_id: dict[str, dict[str, object]] = {}

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 6:
            continue
        claim_id = cells[0]
        if not re.fullmatch(r"C-\d{3}", claim_id):
            continue
        status_raw = cells[3]
        status_token_match = BOLD_TOKEN_RE.search(status_raw)
        status_token = (
            _collapse_ws(status_token_match.group(1)).upper().replace(" ", "_")
            if status_token_match
            else "UNSPECIFIED"
        )
        atom = {
            "claim_id": claim_id,
            "statement": _collapse_ws(cells[1]),
            "where_stated": _collapse_ws(cells[2]),
            "status_token": status_token,
            "status_detail": _collapse_ws(status_raw),
            "last_verified": _collapse_ws(cells[4]),
            "verification_rule": _collapse_ws(" | ".join(cells[5:])),
            "h0": "",
            "h1": "",
            "decision_rule": "",
            "source_uid": doc.source_uid,
            "source_group": doc.source_group,
            "source_registry": doc.source_registry,
            "source_path": doc.source_path,
            "source_line": line_no,
        }
        claim_row_by_id[claim_id] = atom
        atoms.append(atom)

    sections = _split_sections(doc.body)
    for section in sections:
        heading = f"{section.title}".strip()
        heading_match = CLAIM_HEADING_RE.match(f"### {heading}") if not heading.startswith("C-") else None
        if heading_match is None:
            direct = re.match(r"^(C-\d{3})\s*:\s*(.+)$", heading)
            if not direct:
                continue
            claim_id = direct.group(1)
        else:
            claim_id = heading_match.group(1)
        if claim_id not in claim_row_by_id:
            continue

        h0 = ""
        h1 = ""
        decision = ""
        for raw in section.body_lines:
            line = _collapse_ws(raw)
            if not line:
                continue
            lowered = line.lower()
            if "**h0**" in lowered or lowered.startswith("h0:"):
                h0 = line
            elif "**h1**" in lowered or lowered.startswith("h1:"):
                h1 = line
            elif "decision rule" in lowered:
                decision = line
        atom = claim_row_by_id[claim_id]
        atom["h0"] = h0
        atom["h1"] = h1
        atom["decision_rule"] = decision

    atoms.sort(key=lambda item: (str(item["claim_id"]), int(item["source_line"])))
    return atoms


def _classify_equation(expr: str) -> str:
    if "->" in expr:
        return "mapping_relation"
    if any(token in expr for token in ("d/d", "partial", "nabla", "Delta")):
        return "differential_relation"
    if any(token in expr for token in ("<=", ">=", "!=", "<", ">")):
        return "inequality_or_constraint"
    return "algebraic_relation"


def _extract_equations_from_source(doc: SourceDoc) -> list[dict[str, object]]:
    lines = _ascii_sanitize(doc.body).splitlines()
    atoms: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    section_title = "(root)"
    in_code_fence = False

    for line_no, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        heading = HEADING_RE.match(raw)
        if heading:
            section_title = _collapse_ws(heading.group(2))
            continue

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue

        for token in INLINE_MATH_RE.findall(raw):
            expr = _collapse_ws(token)
            if len(expr) < 3:
                continue
            key = (doc.source_uid, line_no, expr)
            if key in seen:
                continue
            seen.add(key)
            atoms.append(
                {
                    "source_uid": doc.source_uid,
                    "source_group": doc.source_group,
                    "source_registry": doc.source_registry,
                    "source_path": doc.source_path,
                    "section_title": section_title,
                    "source_line": line_no,
                    "expression": expr,
                    "extraction_kind": "inline_math",
                    "equation_kind": _classify_equation(expr),
                    "claim_refs": sorted(set(CLAIM_ID_RE.findall(expr))),
                }
            )

        candidate = stripped
        if candidate.startswith("- "):
            candidate = candidate[2:].strip()
        if not candidate:
            continue
        if candidate.startswith(("#", "|", "*")):
            continue
        if in_code_fence:
            continue
        if not any(op in candidate for op in ("=", "->", "<=", ">=", "!=")):
            continue
        if not re.search(r"[A-Za-z_]", candidate):
            continue
        symbol_count = sum(candidate.count(ch) for ch in "=+-*/^_()<>")
        if symbol_count < 2:
            continue
        if len(candidate) > 220:
            continue
        expr = _collapse_ws(candidate)
        key = (doc.source_uid, line_no, expr)
        if key in seen:
            continue
        seen.add(key)
        atoms.append(
            {
                "source_uid": doc.source_uid,
                "source_group": doc.source_group,
                "source_registry": doc.source_registry,
                "source_path": doc.source_path,
                "section_title": section_title,
                "source_line": line_no,
                "expression": expr,
                "extraction_kind": "equation_like_line",
                "equation_kind": _classify_equation(expr),
                "claim_refs": sorted(set(CLAIM_ID_RE.findall(expr))),
            }
        )

    atoms.sort(key=lambda item: (str(item["source_uid"]), int(item["source_line"]), str(item["expression"])))
    return atoms


def _classify_proof_section(title: str, body: str) -> str:
    lowered_title = title.lower()
    if "theorem" in lowered_title:
        return "theorem"
    if "lemma" in lowered_title:
        return "lemma"
    if "corollary" in lowered_title:
        return "corollary"
    if "axiom" in lowered_title:
        return "axiom"
    if "proof" in lowered_title:
        return "proof"
    if "derivation" in lowered_title:
        return "derivation"
    if "hypothesis" in lowered_title or "**h0**" in body.lower() or "**h1**" in body.lower():
        return "hypothesis_block"
    return "argument_section"


def _extract_proofs_from_source(doc: SourceDoc) -> list[dict[str, object]]:
    atoms: list[dict[str, object]] = []
    sections = _split_sections(doc.body)
    for section in sections:
        section_text = "\n".join(section.body_lines)
        section_text_ascii = _ascii_sanitize(section_text)
        lowered_title = section.title.lower()
        lowered_body = section_text_ascii.lower()

        is_proof_candidate = any(token in lowered_title for token in PROOF_KEYWORDS) or any(
            marker in lowered_body for marker in ("**h0**", "**h1**", "decision rule", "therefore", "hence")
        )
        if not is_proof_candidate:
            continue

        non_empty_lines = [_collapse_ws(line) for line in section.body_lines if _collapse_ws(line)]
        assumption_lines = [
            line
            for line in non_empty_lines
            if any(token in line.lower() for token in ("**h0**", "**h1**", "assume", "given", "hypothesis"))
        ]
        decision_lines = [line for line in non_empty_lines if "decision rule" in line.lower()]
        conclusion_lines = [
            line
            for line in non_empty_lines
            if any(token in line.lower() for token in ("status", "therefore", "hence", "rejected", "verified", "refuted"))
        ]
        excerpt = " || ".join(non_empty_lines[:10])

        atoms.append(
            {
                "source_uid": doc.source_uid,
                "source_group": doc.source_group,
                "source_registry": doc.source_registry,
                "source_path": doc.source_path,
                "section_title": section.title,
                "section_level": section.level,
                "line_start": section.line_start,
                "line_end": section.line_end,
                "proof_kind": _classify_proof_section(section.title, section_text_ascii),
                "assumption_text": _collapse_ws(" | ".join(assumption_lines)),
                "decision_rule_text": _collapse_ws(" | ".join(decision_lines)),
                "conclusion_text": _collapse_ws(" | ".join(conclusion_lines)),
                "excerpt": _collapse_ws(excerpt),
                "claim_refs": sorted(set(CLAIM_ID_RE.findall(section_text_ascii))),
            }
        )

    atoms.sort(key=lambda item: (str(item["source_uid"]), int(item["line_start"]), str(item["section_title"])))
    return atoms


def _render_claim_atoms(claim_atoms: list[dict[str, object]]) -> str:
    unique_claim_ids = sorted(set(str(atom["claim_id"]) for atom in claim_atoms))
    lines: list[str] = []
    lines.append("# Structured claim atoms extracted from selected high-information corpora.")
    lines.append("# Generated by src/scripts/analysis/build_structured_knowledge_atoms.py")
    lines.append("")
    lines.append("[knowledge_claim_atoms]")
    lines.append('updated = "deterministic"')
    lines.append("authoritative = true")
    lines.append(f"atom_count = {len(claim_atoms)}")
    lines.append(f"unique_claim_id_count = {len(unique_claim_ids)}")
    lines.append("")

    for idx, atom in enumerate(claim_atoms, start=1):
        lines.append("[[atom]]")
        lines.append(f"id = {_esc(f'CLA-{idx:04d}')}")
        lines.append(f"claim_id = {_esc(str(atom['claim_id']))}")
        lines.append(f"statement = {_esc(str(atom['statement']))}")
        lines.append(f"status_token = {_esc(str(atom['status_token']))}")
        lines.append(f"status_detail = {_esc(str(atom['status_detail']))}")
        lines.append(f"last_verified = {_esc(str(atom['last_verified']))}")
        lines.append(f"where_stated = {_esc(str(atom['where_stated']))}")
        lines.append(f"verification_rule = {_esc(str(atom['verification_rule']))}")
        lines.append(f"h0 = {_esc(str(atom['h0']))}")
        lines.append(f"h1 = {_esc(str(atom['h1']))}")
        lines.append(f"decision_rule = {_esc(str(atom['decision_rule']))}")
        lines.append(f"source_uid = {_esc(str(atom['source_uid']))}")
        lines.append(f"source_group = {_esc(str(atom['source_group']))}")
        lines.append(f"source_registry = {_esc(str(atom['source_registry']))}")
        lines.append(f"source_path = {_esc(str(atom['source_path']))}")
        lines.append(f"source_line = {int(atom['source_line'])}")
        lines.append("")
    return "\n".join(lines)


def _render_equation_atoms(equation_atoms: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Structured equation atoms extracted from selected high-information corpora.")
    lines.append("# Generated by src/scripts/analysis/build_structured_knowledge_atoms.py")
    lines.append("")
    lines.append("[knowledge_equation_atoms]")
    lines.append('updated = "deterministic"')
    lines.append("authoritative = true")
    lines.append(f"atom_count = {len(equation_atoms)}")
    lines.append("")

    for idx, atom in enumerate(equation_atoms, start=1):
        lines.append("[[atom]]")
        lines.append(f"id = {_esc(f'EQA-{idx:04d}')}")
        lines.append(f"expression = {_esc(str(atom['expression']))}")
        lines.append(f"equation_kind = {_esc(str(atom['equation_kind']))}")
        lines.append(f"extraction_kind = {_esc(str(atom['extraction_kind']))}")
        lines.append(f"section_title = {_esc(str(atom['section_title']))}")
        lines.append(f"source_uid = {_esc(str(atom['source_uid']))}")
        lines.append(f"source_group = {_esc(str(atom['source_group']))}")
        lines.append(f"source_registry = {_esc(str(atom['source_registry']))}")
        lines.append(f"source_path = {_esc(str(atom['source_path']))}")
        lines.append(f"source_line = {int(atom['source_line'])}")
        lines.append(f"claim_refs = {_render_list(list(atom['claim_refs']))}")
        lines.append("")
    return "\n".join(lines)


def _render_proof_atoms(proof_atoms: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Structured proof/derivation atoms extracted from selected high-information corpora.")
    lines.append("# Generated by src/scripts/analysis/build_structured_knowledge_atoms.py")
    lines.append("")
    lines.append("[knowledge_proof_atoms]")
    lines.append('updated = "deterministic"')
    lines.append("authoritative = true")
    lines.append(f"atom_count = {len(proof_atoms)}")
    lines.append("")

    for idx, atom in enumerate(proof_atoms, start=1):
        lines.append("[[atom]]")
        lines.append(f"id = {_esc(f'PRF-{idx:04d}')}")
        lines.append(f"proof_kind = {_esc(str(atom['proof_kind']))}")
        lines.append(f"section_title = {_esc(str(atom['section_title']))}")
        lines.append(f"section_level = {int(atom['section_level'])}")
        lines.append(f"line_start = {int(atom['line_start'])}")
        lines.append(f"line_end = {int(atom['line_end'])}")
        lines.append(f"assumption_text = {_esc(str(atom['assumption_text']))}")
        lines.append(f"decision_rule_text = {_esc(str(atom['decision_rule_text']))}")
        lines.append(f"conclusion_text = {_esc(str(atom['conclusion_text']))}")
        lines.append(f"excerpt = {_esc(str(atom['excerpt']))}")
        lines.append(f"claim_refs = {_render_list(list(atom['claim_refs']))}")
        lines.append(f"source_uid = {_esc(str(atom['source_uid']))}")
        lines.append(f"source_group = {_esc(str(atom['source_group']))}")
        lines.append(f"source_registry = {_esc(str(atom['source_registry']))}")
        lines.append(f"source_path = {_esc(str(atom['source_path']))}")
        lines.append("")
    return "\n".join(lines)


def _render_structured_corpora(
    sources: list[SourceDoc],
    claim_atoms: list[dict[str, object]],
    equation_atoms: list[dict[str, object]],
    proof_atoms: list[dict[str, object]],
) -> str:
    claim_counts: dict[str, int] = {}
    eq_counts: dict[str, int] = {}
    proof_counts: dict[str, int] = {}
    for atom in claim_atoms:
        claim_counts[str(atom["source_uid"])] = claim_counts.get(str(atom["source_uid"]), 0) + 1
    for atom in equation_atoms:
        eq_counts[str(atom["source_uid"])] = eq_counts.get(str(atom["source_uid"]), 0) + 1
    for atom in proof_atoms:
        proof_counts[str(atom["source_uid"])] = proof_counts.get(str(atom["source_uid"]), 0) + 1

    lines: list[str] = []
    lines.append("# Structured corpus coverage and narrative compaction plan.")
    lines.append("# Generated by src/scripts/analysis/build_structured_knowledge_atoms.py")
    lines.append("")
    lines.append("[structured_corpora]")
    lines.append('updated = "deterministic"')
    lines.append("authoritative = true")
    lines.append(f"source_count = {len(sources)}")
    lines.append(f"claim_atom_count = {len(claim_atoms)}")
    lines.append(f"equation_atom_count = {len(equation_atoms)}")
    lines.append(f"proof_atom_count = {len(proof_atoms)}")
    lines.append("")

    for idx, source in enumerate(sources, start=1):
        target_summary_max_lines = max(8, min(48, max(1, source.line_count // 6)))
        lines.append("[[source]]")
        lines.append(f"id = {_esc(f'SCP-{idx:04d}')}")
        lines.append(f"source_uid = {_esc(source.source_uid)}")
        lines.append(f"source_group = {_esc(source.source_group)}")
        lines.append(f"source_registry = {_esc(source.source_registry)}")
        lines.append(f"source_path = {_esc(source.source_path)}")
        lines.append(f"title = {_esc(source.title)}")
        lines.append(f"line_count = {source.line_count}")
        lines.append(f"claim_atom_count = {claim_counts.get(source.source_uid, 0)}")
        lines.append(f"equation_atom_count = {eq_counts.get(source.source_uid, 0)}")
        lines.append(f"proof_atom_count = {proof_counts.get(source.source_uid, 0)}")
        lines.append("narrative_compaction_recommended = true")
        lines.append("reduction_stage = \"structured_atoms_extracted\"")
        lines.append(f"target_summary_max_lines = {target_summary_max_lines}")
        lines.append(
            "next_step = \"replace long body_markdown with structured summary overlays driven by claim/equation/proof atoms\""
        )
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
        "--claim-out",
        default="registry/knowledge/claim_atoms.toml",
        help="Output path for claim atoms TOML.",
    )
    parser.add_argument(
        "--equation-out",
        default="registry/knowledge/equation_atoms.toml",
        help="Output path for equation atoms TOML.",
    )
    parser.add_argument(
        "--proof-out",
        default="registry/knowledge/proof_atoms.toml",
        help="Output path for proof atoms TOML.",
    )
    parser.add_argument(
        "--summary-out",
        default="registry/knowledge/structured_corpora.toml",
        help="Output path for corpus summary TOML.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sources = _load_sources(repo_root)
    doc_0023 = next(item for item in sources if item.source_uid == "DOC-0023")
    claim_atoms = _parse_claim_rows(doc_0023)

    equation_atoms: list[dict[str, object]] = []
    proof_atoms: list[dict[str, object]] = []
    for source in sources:
        equation_atoms.extend(_extract_equations_from_source(source))
        proof_atoms.extend(_extract_proofs_from_source(source))

    claim_text = _render_claim_atoms(claim_atoms)
    equation_text = _render_equation_atoms(equation_atoms)
    proof_text = _render_proof_atoms(proof_atoms)
    summary_text = _render_structured_corpora(sources, claim_atoms, equation_atoms, proof_atoms)

    _write(repo_root / args.claim_out, claim_text)
    _write(repo_root / args.equation_out, equation_text)
    _write(repo_root / args.proof_out, proof_text)
    _write(repo_root / args.summary_out, summary_text)

    print(
        "Wrote structured knowledge atoms: "
        f"claims={len(claim_atoms)} equations={len(equation_atoms)} proofs={len(proof_atoms)} "
        f"sources={len(sources)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
