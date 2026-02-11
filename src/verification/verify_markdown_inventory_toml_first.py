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

from pathlib import Path
import tomllib


ALLOWED = {
    "toml_published_markdown",
    "toml_destination_exists_manual_markdown",
    "third_party_markdown",
    "generated_artifact",
}

ALLOWED_TRACKED_MARKDOWN = {
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "README.md",
}


def _is_allowed_tracked_markdown(path: str) -> bool:
    norm = path.strip().replace("\\", "/")
    if norm in ALLOWED_TRACKED_MARKDOWN:
        return True
    return norm.endswith("/README.md")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    inv_path = repo_root / "registry/markdown_inventory.toml"
    data = tomllib.loads(inv_path.read_text(encoding="utf-8"))

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
        tracked_allowed = _is_allowed_tracked_markdown(path)
        if git_status in {"untracked", "filesystem_only"}:
            # Untracked/filesystem markdown is acceptable during decommission
            # as long as classification and destination constraints pass.
            pass
        if git_status == "tracked" and classification != "third_party_markdown" and not tracked_allowed:
            disallowed_tracked_count += 1
            failures.append(
                f"{path}: tracked markdown is disallowed in strict TOML-only mode"
            )
        if classification not in ALLOWED:
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
            if not destination:
                failures.append(f"{path}: missing toml_destination for manual markdown")
            elif not (repo_root / destination).is_file():
                failures.append(f"{path}: missing toml_destination file {destination}")
            continue
        if classification == "toml_published_markdown":
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
            f"disallowed tracked markdown count={disallowed_tracked_count} (tracked_count={tracked_count})"
        )

    if failures:
        print("ERROR: markdown inventory violates strict TOML-only policy.")
        for item in failures:
            print(f"- {item}")
        return 1

    print("OK: markdown inventory is TOML-first with explicit tracked entrypoint exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
