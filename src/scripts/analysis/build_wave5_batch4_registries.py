#!/usr/bin/env python3
"""
Build Wave 5 Batch 4 strict TOML registries:
- W5-016: registry/experiments.toml (normalized)
- W5-016: registry/experiment_lineage.toml
- W5-021: registry/roadmap.toml (schema-hardened)
- W5-021: registry/todo.toml (schema-hardened)
- W5-021: registry/next_actions.toml (schema-hardened)
- W5-022: registry/requirements.toml (schema-hardened)
- W5-022: registry/module_requirements.toml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any


ID_REF_RE = re.compile(r"\b(?:WS-[A-Z0-9-]+|T-\d{3}|NA-\d{3}|C-\d{3}|I-\d{3}|E-\d{3}|REQ-[A-Z0-9-]+)\b")
PATH_RE = re.compile(r"(?:data|registry|docs|crates|src|tests)/[A-Za-z0-9_./{}:+-]+")
DATASET_ID_RE = re.compile(r"\b(?:PC|PG|EX|AR|CU)-\d{4}\b")
DEP_SPEC_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(.*)$")


ROADMAP_STATUS_ALLOWLIST = ["planned", "active", "in_progress", "done", "paused", "blocked"]
TODO_STATUS_ALLOWLIST = ["open", "in_progress", "done", "blocked", "deferred"]
ACTION_STATUS_ALLOWLIST = ["todo", "in_progress", "done", "blocked", "deferred"]
PLANNING_PRIORITY_ALLOWLIST = ["high", "medium", "low"]
REQUIREMENT_STATUS_ALLOWLIST = ["active", "deprecated", "planned", "blocked"]
RUNTIME_STACK_ALLOWLIST = [
    "mixed",
    "rust",
    "python",
    "docker_python",
    "coq",
    "latex",
    "cpp",
]


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
    token = _collapse(status).upper().replace("/", "_").replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^A-Z0-9_]", "", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "UNSPECIFIED"


def _extract_id_refs(text: str) -> list[str]:
    return sorted(set(ID_REF_RE.findall(_ascii_clean(text))))


def _extract_paths(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in PATH_RE.findall(_ascii_clean(text)):
        item = _collapse(match).rstrip(".,;:)")
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_dependency_spec(spec: str) -> tuple[str, str]:
    clean = _collapse(spec)
    m = DEP_SPEC_RE.match(clean)
    if not m:
        return clean, ""
    return _collapse(m.group(1)), _collapse(m.group(2))


def _load_dataset_path_index(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in (
        "registry/project_csv_canonical_datasets.toml",
        "registry/project_csv_generated_artifacts.toml",
        "registry/project_csv_generated_datasets.toml",
        "registry/external_csv_datasets.toml",
        "registry/archive_csv_datasets.toml",
        "registry/curated_csv_datasets.toml",
    ):
        path = root / rel
        if not path.exists():
            continue
        raw = _load(path)
        for value in raw.values():
            if not isinstance(value, list):
                continue
            for row in value:
                if not isinstance(row, dict):
                    continue
                dataset_id = _collapse(str(row.get("id", "")))
                if not DATASET_ID_RE.fullmatch(dataset_id):
                    continue
                for key in ("source_csv", "canonical_toml"):
                    item = _collapse(str(row.get(key, "")))
                    if item:
                        out[item] = dataset_id
    return out


def _build_experiment_rows(
    experiments: list[dict[str, Any]],
    binaries: dict[str, dict[str, Any]],
    dataset_path_index: dict[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(sorted(experiments, key=lambda item: str(item.get("id", ""))), start=1):
        eid = _collapse(str(row.get("id", "")))
        binary = _collapse(str(row.get("binary", "")))
        method = _collapse(str(row.get("method", "")))
        input_text = _collapse(str(row.get("input", "")))
        outputs = [str(v) for v in row.get("output", []) if str(v).strip()]
        output_text = " ".join(outputs)
        claims = sorted(set(_collapse(str(v)) for v in row.get("claims", []) if _collapse(str(v))))
        seed_value = row.get("seed", None)
        seed = int(seed_value) if isinstance(seed_value, int) else None
        deterministic = bool(row.get("deterministic", False))
        gpu = bool(row.get("gpu", False))
        status = _collapse(str(row.get("status", "active"))).lower()
        if status not in {"active", "deprecated", "planned", "blocked"}:
            status = "active"
        run_cmd = _collapse(str(row.get("run", "")))
        lineage_id = f"XL-{idx:03d}"
        input_path_refs = _extract_paths(input_text)
        output_path_refs = _extract_paths(output_text)
        dataset_refs = sorted(
            set(
                [
                    *(dataset_path_index.get(path, "") for path in input_path_refs),
                    *(dataset_path_index.get(path, "") for path in output_path_refs),
                    *DATASET_ID_RE.findall(input_text + " " + output_text),
                ]
            )
            - {""}
        )
        reproducibility_class = (
            "deterministic_replay"
            if deterministic
            else ("seeded_stochastic_replay" if seed is not None else "non_deterministic")
        )
        binary_row = binaries.get(binary, {})
        binary_registered = binary in binaries
        binary_experiment_declared = _collapse(str(binary_row.get("experiment", "")))
        out.append(
            {
                "id": eid,
                "title": _collapse(str(row.get("title", ""))),
                "binary": binary,
                "binary_registered": binary_registered,
                "binary_experiment_declared": binary_experiment_declared,
                "method": method,
                "input": input_text,
                "output": [_collapse(str(v)) for v in outputs],
                "run": run_cmd,
                "run_command_sha256": hashlib.sha256(run_cmd.encode("utf-8")).hexdigest(),
                "claims": claims,
                "claim_refs": claims,
                "deterministic": deterministic,
                "seed": seed,
                "gpu": gpu,
                "status": status,
                "status_token": _status_token(status),
                "lineage_id": lineage_id,
                "input_path_refs": input_path_refs,
                "output_path_refs": output_path_refs,
                "dataset_refs": dataset_refs,
                "reproducibility_class": reproducibility_class,
            }
        )
    return out


def _build_experiment_lineage(
    experiment_rows: list[dict[str, Any]],
    claim_ids: set[str],
    dataset_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lineages: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    edge_seq = 0
    for row in experiment_rows:
        lineage = {
            "id": row["lineage_id"],
            "experiment_id": row["id"],
            "binary": row["binary"],
            "deterministic": row["deterministic"],
            "seed": row["seed"],
            "gpu": row["gpu"],
            "run_command": row["run"],
            "run_command_sha256": row["run_command_sha256"],
            "claim_refs": row["claim_refs"],
            "input_path_refs": row["input_path_refs"],
            "output_path_refs": row["output_path_refs"],
            "dataset_refs": row["dataset_refs"],
            "replay_steps": [
                "Confirm required input paths are available.",
                row["run"],
                "Verify expected outputs or stdout artifacts are produced.",
            ],
            "acceptance_criteria": [
                "Claim references resolve in registry/claims.toml.",
                "Binary is registered in registry/binaries.toml.",
                "Reproducibility class is explicitly declared.",
            ],
        }
        lineages.append(lineage)

        edge_seq += 1
        edges.append(
            {
                "id": f"XLE-{edge_seq:05d}",
                "lineage_id": lineage["id"],
                "from_id": row["id"],
                "to_ref": row["binary"],
                "to_kind": "binary",
                "edge_kind": "implemented_by_binary",
                "verified": bool(row["binary_registered"]),
            }
        )
        for cid in row["claim_refs"]:
            edge_seq += 1
            edges.append(
                {
                    "id": f"XLE-{edge_seq:05d}",
                    "lineage_id": lineage["id"],
                    "from_id": row["id"],
                    "to_ref": cid,
                    "to_kind": "claim",
                    "edge_kind": "supports_claim",
                    "verified": cid in claim_ids,
                }
            )
        for path in row["input_path_refs"]:
            edge_seq += 1
            edges.append(
                {
                    "id": f"XLE-{edge_seq:05d}",
                    "lineage_id": lineage["id"],
                    "from_id": row["id"],
                    "to_ref": path,
                    "to_kind": "path",
                    "edge_kind": "consumes_path",
                    "verified": True,
                }
            )
        for path in row["output_path_refs"]:
            edge_seq += 1
            edges.append(
                {
                    "id": f"XLE-{edge_seq:05d}",
                    "lineage_id": lineage["id"],
                    "from_id": row["id"],
                    "to_ref": path,
                    "to_kind": "path",
                    "edge_kind": "produces_path",
                    "verified": True,
                }
            )
        for did in row["dataset_refs"]:
            edge_seq += 1
            edges.append(
                {
                    "id": f"XLE-{edge_seq:05d}",
                    "lineage_id": lineage["id"],
                    "from_id": row["id"],
                    "to_ref": did,
                    "to_kind": "dataset",
                    "edge_kind": "touches_dataset",
                    "verified": did in dataset_ids,
                }
            )
    return lineages, edges


def _build_hardened_planning_rows(
    rows: list[dict[str, Any]],
    row_kind: str,
    id_key: str,
    status_allowlist: list[str],
    priority_allowlist: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    known_ids = {str(row.get(id_key, "")) for row in rows}
    for row in rows:
        rid = _collapse(str(row.get(id_key, "")))
        status = _collapse(str(row.get("status", ""))).lower()
        if status not in status_allowlist:
            status = status_allowlist[0]
        priority = _collapse(str(row.get("priority", "medium"))).lower()
        if priority not in priority_allowlist:
            priority = "medium"
        text_blob = " ".join(
            [
                _collapse(str(row.get("description", ""))),
                _collapse(str(row.get("title", ""))),
                " ".join(_collapse(str(v)) for v in row.get("claims", [])),
                _collapse(str(row.get("insight", ""))),
            ]
        )
        refs = [ref for ref in _extract_id_refs(text_blob) if ref != rid]
        deps = sorted(set(ref for ref in refs if ref in known_ids or ref.startswith(("C-", "I-", "E-", "REQ-"))))
        evidence_refs = sorted(set(_extract_paths(text_blob)))
        if "primary_outputs" in row and isinstance(row.get("primary_outputs"), list):
            evidence_refs = sorted(set(evidence_refs + [_collapse(str(v)) for v in row["primary_outputs"] if _collapse(str(v))]))
        acceptance = [
            f"{row_kind} status is constrained to declared enum values.",
            f"{row_kind} dependencies are explicit and machine-parseable.",
        ]
        if "claims" in row and row.get("claims"):
            acceptance.append("Claim references remain resolvable in registry/claims.toml.")
        if evidence_refs:
            acceptance.append("Evidence references point to maintained canonical paths.")

        hardened = dict(row)
        hardened["status"] = status
        hardened["status_token"] = _status_token(status)
        if "priority" in hardened:
            hardened["priority"] = priority
        hardened["dependencies"] = deps
        hardened["acceptance_criteria"] = acceptance
        hardened["evidence_refs"] = evidence_refs
        out.append(hardened)
    out.sort(key=lambda item: str(item.get(id_key, "")))
    return out


def _infer_runtime_stack(module_name: str) -> str:
    name = module_name.lower()
    mapping = {
        "core": "mixed",
        "algebra": "rust",
        "analysis": "python",
        "astro": "python",
        "materials": "mixed",
        "particle": "python",
        "quantum_docker": "docker_python",
        "coq": "coq",
        "latex": "latex",
        "cpp": "cpp",
    }
    return mapping.get(name, "mixed")


def _module_dependency_defaults(module_id: str) -> list[str]:
    if module_id == "REQ-CORE":
        return []
    return ["REQ-CORE"]


def _build_requirements_hardened(rows: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modules: list[dict[str, Any]] = []
    known_module_ids = {str(row.get("id", "")) for row in rows}
    for row in sorted(rows, key=lambda item: str(item.get("id", ""))):
        mid = _collapse(str(row.get("id", "")))
        status = _collapse(str(row.get("status", "active"))).lower()
        if status not in REQUIREMENT_STATUS_ALLOWLIST:
            status = "active"
        install_targets = [_collapse(str(v)) for v in row.get("install_targets", []) if _collapse(str(v))]
        verify_targets = sorted(set(install_targets))
        runtime_stack = _infer_runtime_stack(_collapse(str(row.get("name", ""))))
        requires_modules = [dep for dep in _module_dependency_defaults(mid) if dep in known_module_ids]
        acceptance = [
            "Runtime stack classification is explicit.",
            "Install and verify commands are reproducible.",
            "Module dependencies are fully declared in TOML.",
        ]
        module_row = dict(row)
        module_row["status"] = status
        module_row["status_token"] = _status_token(status)
        module_row["runtime_stack"] = runtime_stack
        module_row["requires_modules"] = requires_modules
        module_row["install_targets"] = install_targets
        module_row["verify_targets"] = verify_targets
        module_row["acceptance_criteria"] = acceptance
        modules.append(module_row)

    coverage_gaps: list[dict[str, Any]] = []
    for row in sorted(gaps, key=lambda item: str(item.get("id", ""))):
        status = _collapse(str(row.get("status", "open"))).lower()
        if status not in {"open", "in_progress", "done"}:
            status = "open"
        gap = dict(row)
        gap["status"] = status
        gap["status_token"] = _status_token(status)
        gap["related_module_ids"] = sorted(
            set(
                ref
                for ref in _extract_id_refs(
                    _collapse(str(row.get("description", ""))) + " " + _collapse(str(row.get("proposed_resolution", "")))
                )
                if ref.startswith("REQ-")
            )
        )
        coverage_gaps.append(gap)
    return modules, coverage_gaps


def _build_module_requirements(
    requirements_modules: list[dict[str, Any]],
    pyproject: dict[str, Any],
    cargo_toml: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    modules: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []

    module_ids = {str(row.get("id", "")) for row in requirements_modules}
    for row in requirements_modules:
        mid = _collapse(str(row.get("id", "")))
        install_targets = [_collapse(str(v)) for v in row.get("install_targets", []) if _collapse(str(v))]
        verify_targets = [_collapse(str(v)) for v in row.get("verify_targets", []) if _collapse(str(v))]
        command_refs: list[str] = []
        for cmd in install_targets:
            cid = f"CMD-{len(commands)+1:04d}"
            commands.append(
                {
                    "id": cid,
                    "module_id": mid,
                    "kind": "install",
                    "command": cmd,
                }
            )
            command_refs.append(cid)
        for cmd in verify_targets:
            cid = f"CMD-{len(commands)+1:04d}"
            commands.append(
                {
                    "id": cid,
                    "module_id": mid,
                    "kind": "verify",
                    "command": cmd,
                }
            )
            command_refs.append(cid)
        modules.append(
            {
                "id": mid,
                "name": _collapse(str(row.get("name", ""))),
                "runtime_stack": _collapse(str(row.get("runtime_stack", ""))),
                "status": _collapse(str(row.get("status", ""))),
                "status_token": _collapse(str(row.get("status_token", ""))),
                "source_markdown": _collapse(str(row.get("markdown", ""))),
                "requires_modules": [dep for dep in row.get("requires_modules", []) if dep in module_ids],
                "command_refs": command_refs,
                "package_refs": [],
            }
        )

    module_index = {row["id"]: row for row in modules}

    # Python packages (base dependencies + extras)
    project = pyproject.get("project", {})
    base_deps = project.get("dependencies", [])
    optional = project.get("optional-dependencies", {})
    extra_module_map = {
        "analysis": "REQ-ANALYSIS",
        "astro": "REQ-ASTRO",
        "particle": "REQ-PARTICLE",
        "quantum": "REQ-QUANTUM",
        "dev": "REQ-CORE",
    }

    def add_package(module_id: str, manager: str, spec: str, source: str, group: str, optional_flag: bool) -> None:
        if module_id not in module_index:
            module_id = "REQ-CORE"
        name, constraint = _parse_dependency_spec(spec)
        pid = f"PKG-{len(packages)+1:05d}"
        packages.append(
            {
                "id": pid,
                "module_id": module_id,
                "manager": manager,
                "name": name,
                "constraint": constraint,
                "spec": _collapse(spec),
                "group": group,
                "optional": optional_flag,
                "source": source,
            }
        )
        module_index[module_id]["package_refs"].append(pid)

    for spec in base_deps:
        add_package("REQ-CORE", "pip", str(spec), "pyproject.toml", "base", False)
    for extra, deps in optional.items():
        module_id = extra_module_map.get(str(extra), "REQ-CORE")
        for spec in deps:
            add_package(module_id, "pip", str(spec), "pyproject.toml", f"extra:{extra}", True)

    # Rust workspace dependencies
    workspace = cargo_toml.get("workspace", {})
    ws_deps = workspace.get("dependencies", {})
    quantum_rust = {"quantum", "qua_ten_net", "cudarc", "numpy", "pyo3"}
    for dep_name, dep_spec in ws_deps.items():
        if dep_name in quantum_rust:
            module_id = "REQ-QUANTUM"
        else:
            module_id = "REQ-CORE"
        if isinstance(dep_spec, str):
            spec = f"{dep_name} {dep_spec}"
        else:
            spec = f"{dep_name} {json.dumps(dep_spec, ensure_ascii=True, sort_keys=True)}"
        add_package(module_id, "cargo", spec, "Cargo.toml", "workspace.dependencies", False)

    for module in modules:
        module["package_refs"] = sorted(module["package_refs"])
        module["command_refs"] = sorted(module["command_refs"])

    modules.sort(key=lambda item: item["id"])
    packages.sort(key=lambda item: item["id"])
    commands.sort(key=lambda item: item["id"])
    return modules, packages, commands


def _render_experiments(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Experiments registry -- strict TOML schema (Wave 5 batch 4).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.")
    lines.append("")
    lines.append("[experiments]")
    lines.append('updated = "2026-02-10"')
    lines.append("authoritative = true")
    lines.append(f"experiment_count = {len(rows)}")
    lines.append(f"deterministic_count = {sum(1 for row in rows if row['deterministic'])}")
    lines.append(f"gpu_count = {sum(1 for row in rows if row['gpu'])}")
    lines.append(f"seeded_count = {sum(1 for row in rows if row['seed'] is not None)}")
    lines.append('status_allowlist = ["active", "deprecated", "planned", "blocked"]')
    lines.append("")
    for row in rows:
        lines.append("[[experiment]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"title = {_q(row['title'])}")
        lines.append(f"binary = {_q(row['binary'])}")
        lines.append(f"binary_registered = {str(row['binary_registered']).lower()}")
        lines.append(f"binary_experiment_declared = {_q(row['binary_experiment_declared'])}")
        lines.append(f"method = {_q(row['method'])}")
        lines.append(f"input = {_q(row['input'])}")
        lines.append(f"output = {_render_list(row['output'])}")
        lines.append(f"run = {_q(row['run'])}")
        lines.append(f"run_command_sha256 = {_q(row['run_command_sha256'])}")
        lines.append(f"claims = {_render_list(row['claims'])}")
        lines.append(f"claim_refs = {_render_list(row['claim_refs'])}")
        lines.append(f"deterministic = {str(row['deterministic']).lower()}")
        if row["seed"] is not None:
            lines.append(f"seed = {row['seed']}")
        lines.append(f"gpu = {str(row['gpu']).lower()}")
        lines.append(f"status = {_q(row['status'])}")
        lines.append(f"status_token = {_q(row['status_token'])}")
        lines.append(f"lineage_id = {_q(row['lineage_id'])}")
        lines.append(f"input_path_refs = {_render_list(row['input_path_refs'])}")
        lines.append(f"output_path_refs = {_render_list(row['output_path_refs'])}")
        lines.append(f"dataset_refs = {_render_list(row['dataset_refs'])}")
        lines.append(f"reproducibility_class = {_q(row['reproducibility_class'])}")
        lines.append("")
    return "\n".join(lines)


def _render_experiment_lineage(lineages: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Experiment lineage registry -- strict TOML schema (Wave 5 batch 4).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.")
    lines.append("")
    lines.append("[experiment_lineage]")
    lines.append('updated = "2026-02-10"')
    lines.append("authoritative = true")
    lines.append('source_registry = "registry/experiments.toml"')
    lines.append(f"lineage_count = {len(lineages)}")
    lines.append(f"edge_count = {len(edges)}")
    lines.append("")
    for row in lineages:
        lines.append("[[lineage]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"experiment_id = {_q(row['experiment_id'])}")
        lines.append(f"binary = {_q(row['binary'])}")
        lines.append(f"deterministic = {str(row['deterministic']).lower()}")
        if row["seed"] is not None:
            lines.append(f"seed = {row['seed']}")
        lines.append(f"gpu = {str(row['gpu']).lower()}")
        lines.append(f"run_command = {_q(row['run_command'])}")
        lines.append(f"run_command_sha256 = {_q(row['run_command_sha256'])}")
        lines.append(f"claim_refs = {_render_list(row['claim_refs'])}")
        lines.append(f"input_path_refs = {_render_list(row['input_path_refs'])}")
        lines.append(f"output_path_refs = {_render_list(row['output_path_refs'])}")
        lines.append(f"dataset_refs = {_render_list(row['dataset_refs'])}")
        lines.append(f"replay_steps = {_render_list(row['replay_steps'])}")
        lines.append(f"acceptance_criteria = {_render_list(row['acceptance_criteria'])}")
        lines.append("")
    for row in edges:
        lines.append("[[edge]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"lineage_id = {_q(row['lineage_id'])}")
        lines.append(f"from_id = {_q(row['from_id'])}")
        lines.append(f"to_ref = {_q(row['to_ref'])}")
        lines.append(f"to_kind = {_q(row['to_kind'])}")
        lines.append(f"edge_kind = {_q(row['edge_kind'])}")
        lines.append(f"verified = {str(row['verified']).lower()}")
        lines.append("")
    return "\n".join(lines)


def _render_roadmap(raw_meta: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Operational roadmap registry (TOML-first, schema-hardened).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.")
    lines.append("")
    lines.append("[roadmap]")
    lines.append(f"source_markdown = {_q(_collapse(str(raw_meta.get('source_markdown', 'docs/ROADMAP.md'))))}")
    lines.append('consolidated_date = "2026-02-10"')
    supersedes = [_collapse(str(v)) for v in raw_meta.get("supersedes", []) if _collapse(str(v))]
    companion_docs = [_collapse(str(v)) for v in raw_meta.get("companion_docs", []) if _collapse(str(v))]
    lines.append(f"supersedes = {_render_list(supersedes)}")
    lines.append(f"companion_docs = {_render_list(companion_docs)}")
    lines.append("status = \"active\"")
    lines.append("status_token = \"ACTIVE\"")
    lines.append("authoritative = true")
    lines.append(f"workstream_count = {len(rows)}")
    lines.append(f"status_allowlist = {_render_list(ROADMAP_STATUS_ALLOWLIST)}")
    lines.append(f"priority_allowlist = {_render_list(PLANNING_PRIORITY_ALLOWLIST)}")
    lines.append("")
    lines.append("[roadmap.schema]")
    lines.append("required_fields = [\"id\", \"name\", \"priority\", \"status\", \"status_token\", \"description\", \"dependencies\", \"acceptance_criteria\"]")
    lines.append("dependency_id_pattern = \"WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*\"")
    lines.append("")
    if "sections" in raw_meta and isinstance(raw_meta["sections"], dict):
        lines.append("[roadmap.sections]")
        for key, value in raw_meta["sections"].items():
            lines.append(f"{key} = {_q(_collapse(str(value)))}")
        lines.append("")
    for row in rows:
        lines.append("[[workstream]]")
        for key in (
            "id",
            "name",
            "priority",
            "status",
            "status_token",
            "description",
        ):
            lines.append(f"{key} = {_q(_collapse(str(row.get(key, ''))))}")
        if "sprint" in row and _collapse(str(row.get("sprint", ""))):
            lines.append(f"sprint = {_q(_collapse(str(row.get('sprint', ''))))}")
        if row.get("primary_outputs"):
            lines.append(f"primary_outputs = {_render_list([_collapse(str(v)) for v in row['primary_outputs'] if _collapse(str(v))])}")
        if row.get("claims"):
            lines.append(f"claims = {_render_list([_collapse(str(v)) for v in row['claims'] if _collapse(str(v))])}")
        if _collapse(str(row.get("insight", ""))):
            lines.append(f"insight = {_q(_collapse(str(row.get('insight', ''))))}")
        if row.get("lacunae"):
            lines.append(f"lacunae = {_render_list([_collapse(str(v)) for v in row['lacunae'] if _collapse(str(v))])}")
        lines.append(f"dependencies = {_render_list([_collapse(str(v)) for v in row.get('dependencies', []) if _collapse(str(v))])}")
        lines.append(f"acceptance_criteria = {_render_list([_collapse(str(v)) for v in row.get('acceptance_criteria', []) if _collapse(str(v))])}")
        lines.append(f"evidence_refs = {_render_list([_collapse(str(v)) for v in row.get('evidence_refs', []) if _collapse(str(v))])}")
        lines.append("")
    return "\n".join(lines)


def _render_todo(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# To-Do Registry (TOML-first, schema-hardened).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.")
    lines.append("")
    lines.append("[todo]")
    lines.append('updated = "2026-02-10"')
    lines.append('status = "active"')
    lines.append("status_token = \"ACTIVE\"")
    lines.append(f"item_count = {len(rows)}")
    lines.append(f"status_allowlist = {_render_list(TODO_STATUS_ALLOWLIST)}")
    lines.append(f"priority_allowlist = {_render_list(PLANNING_PRIORITY_ALLOWLIST)}")
    lines.append("")
    lines.append("[todo.schema]")
    lines.append("required_fields = [\"id\", \"area\", \"title\", \"description\", \"priority\", \"status\", \"status_token\", \"dependencies\", \"acceptance_criteria\"]")
    lines.append("dependency_id_pattern = \"WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*\"")
    lines.append("")
    for row in rows:
        lines.append("[[item]]")
        for key in ("id", "area", "title", "description", "priority", "status", "status_token"):
            lines.append(f"{key} = {_q(_collapse(str(row.get(key, ''))))}")
        lines.append(f"dependencies = {_render_list([_collapse(str(v)) for v in row.get('dependencies', []) if _collapse(str(v))])}")
        lines.append(f"acceptance_criteria = {_render_list([_collapse(str(v)) for v in row.get('acceptance_criteria', []) if _collapse(str(v))])}")
        lines.append(f"evidence_refs = {_render_list([_collapse(str(v)) for v in row.get('evidence_refs', []) if _collapse(str(v))])}")
        lines.append("")
    return "\n".join(lines)


def _render_next_actions(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Next Actions Registry (TOML-first, schema-hardened).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.")
    lines.append("")
    lines.append("[meta]")
    lines.append('updated = "2026-02-10"')
    lines.append('status = "active"')
    lines.append('status_token = "ACTIVE"')
    lines.append(f"action_count = {len(rows)}")
    lines.append(f"status_allowlist = {_render_list(ACTION_STATUS_ALLOWLIST)}")
    lines.append(f"priority_allowlist = {_render_list(PLANNING_PRIORITY_ALLOWLIST)}")
    lines.append("")
    lines.append("[next_actions.schema]")
    lines.append("required_fields = [\"id\", \"area\", \"title\", \"description\", \"priority\", \"status\", \"status_token\", \"dependencies\", \"acceptance_criteria\"]")
    lines.append("dependency_id_pattern = \"WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*\"")
    lines.append("")
    for row in rows:
        lines.append("[[action]]")
        for key in ("id", "area", "title", "description", "priority", "status", "status_token"):
            lines.append(f"{key} = {_q(_collapse(str(row.get(key, ''))))}")
        lines.append(f"dependencies = {_render_list([_collapse(str(v)) for v in row.get('dependencies', []) if _collapse(str(v))])}")
        lines.append(f"acceptance_criteria = {_render_list([_collapse(str(v)) for v in row.get('acceptance_criteria', []) if _collapse(str(v))])}")
        lines.append(f"evidence_refs = {_render_list([_collapse(str(v)) for v in row.get('evidence_refs', []) if _collapse(str(v))])}")
        lines.append("")
    return "\n".join(lines)


def _render_requirements(modules: list[dict[str, Any]], gaps: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Requirements registry (TOML-first, schema-hardened).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.")
    lines.append("")
    lines.append("[requirements]")
    lines.append("authoritative = true")
    lines.append('status = "active"')
    lines.append('status_token = "ACTIVE"')
    lines.append('updated = "2026-02-10"')
    lines.append(f"python_recommended = {_q(_collapse(str(meta.get('python_recommended', '3.11-3.12'))))}")
    lines.append(f"python_allowed = {_q(_collapse(str(meta.get('python_allowed', '3.13+ (with optional extras caveats)'))))}")
    lines.append(f"primary_markdown = {_q(_collapse(str(meta.get('primary_markdown', 'docs/REQUIREMENTS.md'))))}")
    lines.append(f"module_count = {len(modules)}")
    lines.append(f"coverage_gap_count = {len(gaps)}")
    lines.append(f"status_allowlist = {_render_list(REQUIREMENT_STATUS_ALLOWLIST)}")
    lines.append(f"runtime_stack_allowlist = {_render_list(RUNTIME_STACK_ALLOWLIST)}")
    lines.append("")
    lines.append("[requirements.schema]")
    lines.append("required_module_fields = [\"id\", \"name\", \"status\", \"status_token\", \"runtime_stack\", \"requires_modules\", \"install_targets\", \"verify_targets\", \"acceptance_criteria\"]")
    lines.append("required_gap_fields = [\"id\", \"area\", \"status\", \"status_token\", \"description\", \"proposed_resolution\", \"related_module_ids\"]")
    lines.append("")
    for row in modules:
        lines.append("[[module]]")
        lines.append(f"id = {_q(_collapse(str(row.get('id', ''))))}")
        lines.append(f"name = {_q(_collapse(str(row.get('name', ''))))}")
        lines.append(f"markdown = {_q(_collapse(str(row.get('markdown', ''))))}")
        lines.append(f"status = {_q(_collapse(str(row.get('status', ''))))}")
        lines.append(f"status_token = {_q(_collapse(str(row.get('status_token', ''))))}")
        lines.append(f"runtime_stack = {_q(_collapse(str(row.get('runtime_stack', ''))))}")
        lines.append(f"requires_modules = {_render_list([_collapse(str(v)) for v in row.get('requires_modules', []) if _collapse(str(v))])}")
        lines.append(f"install_targets = {_render_list([_collapse(str(v)) for v in row.get('install_targets', []) if _collapse(str(v))])}")
        lines.append(f"verify_targets = {_render_list([_collapse(str(v)) for v in row.get('verify_targets', []) if _collapse(str(v))])}")
        lines.append(f"acceptance_criteria = {_render_list([_collapse(str(v)) for v in row.get('acceptance_criteria', []) if _collapse(str(v))])}")
        lines.append("")
    for row in gaps:
        lines.append("[[coverage_gap]]")
        lines.append(f"id = {_q(_collapse(str(row.get('id', ''))))}")
        lines.append(f"area = {_q(_collapse(str(row.get('area', ''))))}")
        lines.append(f"status = {_q(_collapse(str(row.get('status', ''))))}")
        lines.append(f"status_token = {_q(_collapse(str(row.get('status_token', ''))))}")
        lines.append(f"description = {_q(_collapse(str(row.get('description', ''))))}")
        lines.append(f"proposed_resolution = {_q(_collapse(str(row.get('proposed_resolution', ''))))}")
        lines.append(f"related_module_ids = {_render_list([_collapse(str(v)) for v in row.get('related_module_ids', []) if _collapse(str(v))])}")
        lines.append("")
    return "\n".join(lines)


def _render_module_requirements(modules: list[dict[str, Any]], packages: list[dict[str, Any]], commands: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Module requirements decomposition registry (Wave 5 strict schema).")
    lines.append("# Generated by src/scripts/analysis/build_wave5_batch4_registries.py.")
    lines.append("")
    lines.append("[module_requirements]")
    lines.append('updated = "2026-02-10"')
    lines.append("authoritative = true")
    lines.append('source_registries = ["registry/requirements.toml", "pyproject.toml", "Cargo.toml"]')
    lines.append(f"module_count = {len(modules)}")
    lines.append(f"package_count = {len(packages)}")
    lines.append(f"command_count = {len(commands)}")
    lines.append(f"python_package_count = {sum(1 for pkg in packages if pkg['manager'] == 'pip')}")
    lines.append(f"rust_package_count = {sum(1 for pkg in packages if pkg['manager'] == 'cargo')}")
    lines.append("")
    for row in modules:
        lines.append("[[module]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"name = {_q(row['name'])}")
        lines.append(f"runtime_stack = {_q(row['runtime_stack'])}")
        lines.append(f"status = {_q(row['status'])}")
        lines.append(f"status_token = {_q(row['status_token'])}")
        lines.append(f"source_markdown = {_q(row['source_markdown'])}")
        lines.append(f"requires_modules = {_render_list(row['requires_modules'])}")
        lines.append(f"command_refs = {_render_list(row['command_refs'])}")
        lines.append(f"package_refs = {_render_list(row['package_refs'])}")
        lines.append("")
    for row in commands:
        lines.append("[[command]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"module_id = {_q(row['module_id'])}")
        lines.append(f"kind = {_q(row['kind'])}")
        lines.append(f"command = {_q(row['command'])}")
        lines.append("")
    for row in packages:
        lines.append("[[package]]")
        lines.append(f"id = {_q(row['id'])}")
        lines.append(f"module_id = {_q(row['module_id'])}")
        lines.append(f"manager = {_q(row['manager'])}")
        lines.append(f"name = {_q(row['name'])}")
        lines.append(f"constraint = {_q(row['constraint'])}")
        lines.append(f"spec = {_q(row['spec'])}")
        lines.append(f"group = {_q(row['group'])}")
        lines.append(f"optional = {str(row['optional']).lower()}")
        lines.append(f"source = {_q(row['source'])}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()

    claims = _load(root / "registry/claims.toml").get("claim", [])
    claim_ids = {str(row.get("id", "")) for row in claims}
    binaries_rows = _load(root / "registry/binaries.toml").get("binary", [])
    binaries = {str(row.get("name", "")): row for row in binaries_rows}
    dataset_path_index = _load_dataset_path_index(root)
    dataset_ids = set(dataset_path_index.values())
    experiments_input = _load(root / "registry/experiments.toml").get("experiment", [])

    experiment_rows = _build_experiment_rows(experiments_input, binaries, dataset_path_index)
    lineage_rows, lineage_edges = _build_experiment_lineage(experiment_rows, claim_ids, dataset_ids)

    roadmap_raw = _load(root / "registry/roadmap.toml")
    roadmap_meta = roadmap_raw.get("roadmap", {})
    roadmap_meta["sections"] = roadmap_raw.get("roadmap", {}).get("sections", {}) if isinstance(roadmap_raw.get("roadmap", {}), dict) else {}
    if not roadmap_meta.get("sections") and "sections" in roadmap_raw:
        roadmap_meta["sections"] = roadmap_raw.get("sections", {})
    roadmap_rows = _build_hardened_planning_rows(
        rows=roadmap_raw.get("workstream", []),
        row_kind="workstream",
        id_key="id",
        status_allowlist=ROADMAP_STATUS_ALLOWLIST,
        priority_allowlist=PLANNING_PRIORITY_ALLOWLIST,
    )

    todo_raw = _load(root / "registry/todo.toml")
    todo_rows = _build_hardened_planning_rows(
        rows=todo_raw.get("item", []),
        row_kind="todo_item",
        id_key="id",
        status_allowlist=TODO_STATUS_ALLOWLIST,
        priority_allowlist=PLANNING_PRIORITY_ALLOWLIST,
    )

    actions_raw = _load(root / "registry/next_actions.toml")
    action_rows = _build_hardened_planning_rows(
        rows=actions_raw.get("action", []),
        row_kind="next_action",
        id_key="id",
        status_allowlist=ACTION_STATUS_ALLOWLIST,
        priority_allowlist=PLANNING_PRIORITY_ALLOWLIST,
    )

    requirements_raw = _load(root / "registry/requirements.toml")
    req_meta = requirements_raw.get("requirements", {})
    req_modules, req_gaps = _build_requirements_hardened(
        rows=requirements_raw.get("module", []),
        gaps=requirements_raw.get("coverage_gap", []),
    )
    pyproject = _load(root / "pyproject.toml")
    cargo_toml = _load(root / "Cargo.toml")
    module_rows, package_rows, command_rows = _build_module_requirements(
        requirements_modules=req_modules,
        pyproject=pyproject,
        cargo_toml=cargo_toml,
    )

    experiments_text = _render_experiments(experiment_rows)
    lineage_text = _render_experiment_lineage(lineage_rows, lineage_edges)
    roadmap_text = _render_roadmap(roadmap_meta, roadmap_rows)
    todo_text = _render_todo(todo_rows)
    actions_text = _render_next_actions(action_rows)
    requirements_text = _render_requirements(req_modules, req_gaps, req_meta)
    module_requirements_text = _render_module_requirements(module_rows, package_rows, command_rows)

    _write(root / "registry/experiments.toml", experiments_text)
    _write(root / "registry/experiment_lineage.toml", lineage_text)
    _write(root / "registry/roadmap.toml", roadmap_text)
    _write(root / "registry/todo.toml", todo_text)
    _write(root / "registry/next_actions.toml", actions_text)
    _write(root / "registry/requirements.toml", requirements_text)
    _write(root / "registry/module_requirements.toml", module_requirements_text)

    print(
        "Wrote Wave5 Batch4 registries: "
        f"experiments={len(experiment_rows)} "
        f"lineages={len(lineage_rows)} "
        f"lineage_edges={len(lineage_edges)} "
        f"roadmap={len(roadmap_rows)} "
        f"todo={len(todo_rows)} "
        f"next_actions={len(action_rows)} "
        f"req_modules={len(req_modules)} "
        f"module_packages={len(package_rows)} "
        f"module_commands={len(command_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
