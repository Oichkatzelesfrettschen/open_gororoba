#!/usr/bin/env python3
"""
Verify that all roadmap status and priority fields use enum-constrained values.

Policy: Each workstream must have status in allowed list and priority in allowed list.
Allowed status: ["planned", "active", "in_progress", "done", "paused", "blocked"]
Allowed priority: ["high", "medium", "low"]
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    roadmap_path = repo_root / "registry" / "roadmap.toml"

    try:
        roadmap_data = tomllib.loads(roadmap_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Failed to parse roadmap.toml: {e}")
        return 1

    # Extract enum constraints from schema
    schema = roadmap_data.get("roadmap", {}).get("schema", {})
    status_enum = roadmap_data.get("roadmap", {}).get("status_allowlist", [])
    priority_enum = roadmap_data.get("roadmap", {}).get("priority_allowlist", [])

    if not status_enum:
        print("ERROR: status_allowlist not found in roadmap metadata")
        return 1
    if not priority_enum:
        print("ERROR: priority_allowlist not found in roadmap metadata")
        return 1

    # Check all workstreams
    workstreams = roadmap_data.get("workstream", [])
    failures = []

    for ws in workstreams:
        ws_id = ws.get("id", "UNKNOWN")
        status = ws.get("status", "")
        priority = ws.get("priority", "")

        if status and status not in status_enum:
            failures.append(
                f"{ws_id}: status='{status}' not in enum {status_enum}"
            )

        if priority and priority not in priority_enum:
            failures.append(
                f"{ws_id}: priority='{priority}' not in enum {priority_enum}"
            )

    # Report results
    if failures:
        print("ERROR: Roadmap enum constraint violations:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        f"OK: Roadmap enum constraints verified ({len(workstreams)} workstreams). "
        f"status_allowlist={status_enum}, priority_allowlist={priority_enum}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
