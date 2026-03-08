#!/usr/bin/env python3
"""
Verify that markdown inventory remains TOML-first.

Policy:
- Tracked markdown is disallowed (except third-party/cache markdown).
- Ignored/untracked markdown may classify as toml_destination_exists_manual_markdown
  while migration/decommission is in progress, but must have explicit TOML destination.
- No unbacked manual markdown is allowed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ALLOWED_GOVERNANCE_MODES = {
    "toml_generated_mirror",
    "toml_manual_source",
    "immutable_transcript",
    "manual_narrative",
    "generated_artifact",
    "third_party_markdown",
}


def _load_governance(repo_root: Path) -> tuple[set[str], set[str], set[str]]:
    gov = tomllib.loads(
        (repo_root / "registry/markdown_governance.toml").read_text(encoding="utf-8")
    )
    policy = gov.get("policy", {})
    allowed = {
        str(item).strip() for item in policy.get("safe_classifications", []) if str(item).strip()
    }
    tracked_allowed_modes = {
        str(item).strip() for item in policy.get("tracked_allowed_modes", []) if str(item).strip()
    }
    tracked_allowed_paths = {
        str(item).strip().replace("\\", "/")
        for item in policy.get("tracked_allowed_paths", [])
        if str(item).strip()
    }
    for row in gov.get("document", []):
        path = str(row.get("path", "")).strip().replace("\\", "/")
        mode = str(row.get("mode", "")).strip()
        if path and mode in tracked_allowed_modes:
            tracked_allowed_paths.add(path)
    return allowed, tracked_allowed_modes, tracked_allowed_paths


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    inv_path = repo_root / "registry/markdown_inventory.toml"
    gov_path = repo_root / "registry/markdown_governance.toml"
    data = tomllib.loads(inv_path.read_text(encoding="utf-8"))
    governance = tomllib.loads(gov_path.read_text(encoding="utf-8"))
    allowed, _, tracked_allowed_paths = _load_governance(repo_root)
    governance_by_path = {
        str(row.get("path", "")).strip().replace("\\", "/"): row
        for row in governance.get("document", [])
    }

    failures: list[str] = []
    summary = data.get("markdown_inventory", {})
    tracked_count = int(summary.get("tracked_count", 0))
    disallowed_tracked_count = 0

    for row in data.get("document", []):
        path = str(row.get("path", "")).strip()
        git_status = str(row.get("git_status", "")).strip()
        classification = str(row.get("classification", "")).strip()
        destination = str(row.get("toml_destination", "")).strip()
        generated_declared = bool(row.get("generated_declared", False))
        tracked_allowed = path in tracked_allowed_paths
        governance_mode = str(governance_by_path.get(path, {}).get("mode", "")).strip()
        if git_status in {"untracked", "filesystem_only"}:
            # Untracked/filesystem markdown is acceptable during decommission
            # as long as classification and destination constraints pass.
            pass
        if (
            git_status == "tracked"
            and classification != "third_party_markdown"
            and not tracked_allowed
        ):
            disallowed_tracked_count += 1
            failures.append(f"{path}: tracked markdown is disallowed in strict TOML-only mode")
        if classification not in allowed:
            failures.append(f"{path}: disallowed classification={classification}")
        if classification == "generated_artifact":
            if not path.startswith("build/docs/generated/"):
                failures.append(
                    f"{path}: generated_artifact allowed only under build/docs/generated/"
                )
            if git_status == "tracked" and classification != "third_party_markdown":
                failures.append(f"{path}: generated_artifact must not be tracked")
            continue
        if classification == "toml_destination_exists_manual_markdown":
            if git_status == "tracked" and not tracked_allowed:
                failures.append(
                    f"{path}: manual markdown with TOML destination must not be tracked"
                )
            if governance_mode == "toml_generated_mirror":
                failures.append(
                    f"{path}: governance expects generated mirror "
                    "but inventory marks manual markdown"
                )
            if not destination:
                failures.append(f"{path}: missing toml_destination for manual markdown")
            elif not (repo_root / destination).is_file():
                failures.append(f"{path}: missing toml_destination file {destination}")
            continue
        if classification == "toml_published_markdown":
            if governance_mode and governance_mode not in ALLOWED_GOVERNANCE_MODES:
                failures.append(f"{path}: governance mode {governance_mode} is not publishable")
            if governance_mode and governance_mode != "toml_generated_mirror":
                failures.append(
                    f"{path}: governance mode {governance_mode} "
                    "conflicts with published classification"
                )
            if not generated_declared and not path.startswith("build/docs/generated/"):
                failures.append(
                    f"{path}: toml_published_markdown without explicit generated marker header"
                )
            if not destination:
                failures.append(f"{path}: toml_published_markdown without toml_destination")
            elif not (repo_root / destination).is_file():
                failures.append(f"{path}: missing toml_destination file {destination}")

    if tracked_count != 0 and disallowed_tracked_count == 0:
        # Allow a small explicit tracked markdown surface for entrypoints.
        pass
    elif tracked_count == 0:
        pass
    else:
        failures.append(
            "disallowed tracked markdown count="
            f"{disallowed_tracked_count} (tracked_count={tracked_count})"
        )

    if failures:
        print("ERROR: markdown inventory violates strict TOML-only policy.")
        for item in failures:
            print(f"- {item}")
        return 1

    print("OK: markdown inventory is TOML-first with governance-backed tracked exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
