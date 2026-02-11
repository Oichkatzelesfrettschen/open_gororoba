#!/usr/bin/env python3
"""
Verify overflow tracker policy for Pantheon/PhysicsForge migration.

Policy:
- At most 5 active overflow tasks.
- Active tasks require owner, ETA, and rationale.
- Overflow IDs must follow OF-<phase>-<nnn>.
"""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


TRACKER_PATH = "registry/pantheon_physicsforge_overflow_tracker.toml"
OVERFLOW_ID_RE = re.compile(r"^OF-(\d{1,2})-(\d{3})$")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    tracker = _load(repo_root / TRACKER_PATH)

    meta = tracker.get("overflow_tracker", {})
    rows = tracker.get("overflow_task", [])

    max_active = int(meta.get("max_active_tasks", 5))
    active_statuses = {str(x).strip() for x in meta.get("active_statuses", ["open", "in_progress", "blocked"])}

    failures: list[str] = []

    seen_ids: set[str] = set()
    active_count = 0

    for row in rows:
        overflow_id = str(row.get("id", "")).strip()
        phase = int(row.get("phase", -1))
        status = str(row.get("status", "")).strip()
        owner = str(row.get("owner", "")).strip()
        eta = str(row.get("eta", "")).strip()
        rationale = str(row.get("rationale", "")).strip()
        deferral_rationale = str(row.get("deferral_rationale", "")).strip()

        if not overflow_id:
            failures.append("overflow_task row missing id")
            continue
        if overflow_id in seen_ids:
            failures.append(f"duplicate overflow id: {overflow_id}")
            continue
        seen_ids.add(overflow_id)

        match = OVERFLOW_ID_RE.fullmatch(overflow_id)
        if not match:
            failures.append(f"invalid overflow id format: {overflow_id}")
        else:
            id_phase = int(match.group(1))
            if id_phase != phase:
                failures.append(
                    f"overflow_task[{overflow_id}] phase mismatch "
                    f"(id encodes {id_phase}, row has {phase})"
                )

        if phase < 1 or phase > 10:
            failures.append(f"overflow_task[{overflow_id}] invalid phase: {phase}")

        if status in active_statuses:
            active_count += 1
            if not owner:
                failures.append(f"overflow_task[{overflow_id}] active status requires owner")
            if not eta:
                failures.append(f"overflow_task[{overflow_id}] active status requires eta")
            if not rationale:
                failures.append(f"overflow_task[{overflow_id}] active status requires rationale")

        if status == "deferred" and not deferral_rationale:
            failures.append(f"overflow_task[{overflow_id}] deferred status requires deferral_rationale")

    if active_count > max_active:
        failures.append(
            f"active overflow task limit exceeded ({active_count} > {max_active})"
        )

    if int(meta.get("active_count", -1)) != active_count:
        failures.append(
            "overflow_tracker.active_count metadata mismatch "
            f"({meta.get('active_count')} != {active_count})"
        )

    if int(meta.get("total_count", -1)) != len(rows):
        failures.append(
            "overflow_tracker.total_count metadata mismatch "
            f"({meta.get('total_count')} != {len(rows)})"
        )

    if failures:
        print("ERROR: Pantheon/PhysicsForge overflow tracker verification failed.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "OK: overflow tracker verified "
        f"(total={len(rows)}, active={active_count}, max_active={max_active})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
