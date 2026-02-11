#!/usr/bin/env python3
"""
Verify Pantheon/PhysicsForge migration mapping completeness.

This verifier enforces consistency between:
- registry/pantheon_physicsforge_migration_matrix.toml
- registry/pantheon_physicsforge_migration_todo.toml
"""

from __future__ import annotations

from pathlib import Path
import tomllib


MATRIX_PATH = "registry/pantheon_physicsforge_migration_matrix.toml"
TODO_PATH = "registry/pantheon_physicsforge_migration_todo.toml"


STATUS_TO_COMPLETION = {
    "done": "completed",
    "in_progress": "in_progress",
    "blocked": "blocked",
    "todo": "pending",
}


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _check_todo_metadata(todo_meta: dict, rows: list[dict], failures: list[str]) -> None:
    if int(todo_meta.get("task_count", -1)) != len(rows):
        failures.append(
            "migration_todo.task_count metadata mismatch "
            f"({todo_meta.get('task_count')} != {len(rows)})"
        )

    counts = {
        "done": sum(1 for row in rows if str(row.get("status", "")).strip() == "done"),
        "in_progress": sum(
            1 for row in rows if str(row.get("status", "")).strip() == "in_progress"
        ),
        "blocked": sum(1 for row in rows if str(row.get("status", "")).strip() == "blocked"),
        "todo": sum(1 for row in rows if str(row.get("status", "")).strip() == "todo"),
    }

    for key in ("done", "in_progress", "blocked", "todo"):
        meta_key = f"{key}_count"
        if int(todo_meta.get(meta_key, -1)) != counts[key]:
            failures.append(
                f"migration_todo.{meta_key} metadata mismatch "
                f"({todo_meta.get(meta_key)} != {counts[key]})"
            )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    matrix_path = repo_root / MATRIX_PATH
    todo_path = repo_root / TODO_PATH

    matrix = _load(matrix_path)
    todo = _load(todo_path)

    matrix_meta = matrix.get("migration_matrix", {})
    task_completion = matrix.get("task_completion", {})
    boundary_rules = matrix.get("boundary_rule", [])
    module_rows = matrix.get("module_mapping", [])

    todo_meta = todo.get("migration_todo", {})
    todo_rows = todo.get("task", [])

    failures: list[str] = []

    todo_by_number: dict[int, dict] = {}
    for row in todo_rows:
        number = int(row.get("number", -1))
        if number in todo_by_number:
            failures.append(f"duplicate migration todo number: {number}")
            continue
        todo_by_number[number] = row

    _check_todo_metadata(todo_meta, todo_rows, failures)

    allowed_actions = {str(row.get("category", "")).strip() for row in boundary_rules}
    if not allowed_actions:
        failures.append("no boundary_rule categories found in migration matrix")

    seen_mapping_ids: set[str] = set()
    for row in module_rows:
        mapping_id = str(row.get("id", "")).strip()
        source_path = str(row.get("source_path", "")).strip()
        action = str(row.get("action", "")).strip()
        status = str(row.get("status", "")).strip()
        target_crate = str(row.get("target_crate", "")).strip()
        target_module = str(row.get("target_module", "")).strip()

        if not mapping_id:
            failures.append("module_mapping row missing id")
            continue
        if mapping_id in seen_mapping_ids:
            failures.append(f"duplicate module_mapping id: {mapping_id}")
        seen_mapping_ids.add(mapping_id)

        if not source_path:
            failures.append(f"module_mapping[{mapping_id}] missing source_path")
        if not action:
            failures.append(f"module_mapping[{mapping_id}] missing action")
        elif action not in allowed_actions:
            failures.append(
                f"module_mapping[{mapping_id}] action not in boundary rules: {action}"
            )
        if status != "mapped":
            failures.append(f"module_mapping[{mapping_id}] status must be 'mapped', got {status!r}")

        if action in {"port", "rewrite"}:
            if target_crate in {"", "none"}:
                failures.append(
                    f"module_mapping[{mapping_id}] action={action} requires target_crate"
                )
            if target_module in {"", "none"}:
                failures.append(
                    f"module_mapping[{mapping_id}] action={action} requires target_module"
                )

    if not module_rows:
        failures.append("migration matrix has zero module_mapping rows")

    scope_tasks = [str(x).strip() for x in matrix_meta.get("scope_tasks", [])]
    if not scope_tasks:
        failures.append("migration_matrix.scope_tasks is empty")

    for task_num_str in scope_tasks:
        if not task_num_str.isdigit():
            failures.append(f"migration_matrix.scope_tasks contains non-numeric task: {task_num_str!r}")
            continue
        task_num = int(task_num_str)
        todo_row = todo_by_number.get(task_num)
        if todo_row is None:
            failures.append(f"scope task missing in todo registry: {task_num}")
            continue

        completion_key = f"task_{task_num}"
        completion_state = str(task_completion.get(completion_key, "")).strip()
        if not completion_state:
            failures.append(f"task_completion missing key: {completion_key}")
            continue

        todo_status = str(todo_row.get("status", "")).strip()
        expected_completion = STATUS_TO_COMPLETION.get(todo_status)
        if expected_completion is None:
            failures.append(f"todo task {task_num} has unknown status token: {todo_status!r}")
            continue

        if completion_state != expected_completion:
            failures.append(
                f"task_completion[{completion_key}]={completion_state!r} does not match "
                f"todo status {todo_status!r}"
            )

    # Discovery/mapping tranche must point at the canonical matrix artifact.
    for task_num in range(10, 19):
        row = todo_by_number.get(task_num)
        if row is None:
            failures.append(f"missing phase-2 mapping task row: {task_num}")
            continue
        evidence_refs = [str(x) for x in row.get("evidence_refs", [])]
        if MATRIX_PATH not in evidence_refs:
            failures.append(
                f"todo task {task_num} evidence_refs must include {MATRIX_PATH}"
            )

    if failures:
        print("ERROR: Pantheon/PhysicsForge mapping completeness verification failed.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "OK: migration matrix and todo registries are consistent "
        f"(module_mappings={len(module_rows)}, scope_tasks={len(scope_tasks)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
