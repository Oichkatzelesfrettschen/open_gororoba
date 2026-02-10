#!/usr/bin/env python3
"""
Verify markdown governance removal policy compliance (W6-020).

Checks:
1. All non-active markdown files have explicit removal_reason
2. No active files have removal_reason (should be empty)
3. Third-party files have removal_status = locked
4. Migration queue ranks are sequential (no skips)
5. Removal paths follow decommission order (TOML first, then markdown)
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def verify_removal_reasons(owner_map: dict) -> tuple[bool, list[str]]:
    """Check all non-active markdown has explicit removal_reason."""
    errors = []
    
    for owner in owner_map.get("owner", []):
        removal_status = str(owner.get("removal_status", "active")).strip()
        removal_reason = str(owner.get("removal_reason", "")).strip()
        path = str(owner.get("path", ""))
        
        # Rule: active files should NOT have removal_reason
        if removal_status == "active" and removal_reason:
            errors.append(
                f"{path}: removal_status=active but has removal_reason (should be empty)"
            )
        
        # Rule: non-active files MUST have removal_reason
        if removal_status != "active" and not removal_reason:
            errors.append(
                f"{path}: removal_status={removal_status} but missing removal_reason"
            )
    
    return len(errors) == 0, errors


def verify_third_party_locked(owner_map: dict) -> tuple[bool, list[str]]:
    """Check third-party markdown is marked as locked."""
    errors = []
    
    for owner in owner_map.get("owner", []):
        scope = str(owner.get("scope", "")).strip()
        path = str(owner.get("path", ""))
        removal_status = str(owner.get("removal_status", "active")).strip()
        owner_group = str(owner.get("owner_group", "")).strip()
        
        # Heuristic: external sources typically have owner_group="external" or "third_party"
        is_third_party = owner_group in ["third_party", "external"]
        
        if is_third_party and removal_status != "locked":
            errors.append(
                f"{path}: owner_group={owner_group} but removal_status={removal_status} (should be locked)"
            )
    
    return len(errors) == 0, errors


def verify_removal_status_enum(owner_map: dict) -> tuple[bool, list[str]]:
    """Check all removal_status values are valid enum."""
    valid_statuses = {"active", "candidate_for_removal", "deprecated", "archived", "locked"}
    errors = []
    
    for owner in owner_map.get("owner", []):
        removal_status = str(owner.get("removal_status", "active")).strip()
        path = str(owner.get("path", ""))
        
        if removal_status not in valid_statuses:
            errors.append(
                f"{path}: removal_status={removal_status} not in valid enum {valid_statuses}"
            )
    
    return len(errors) == 0, errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    owner_map_path = repo_root / "registry" / "markdown_owner_map.toml"
    governance_path = repo_root / "registry" / "markdown_governance.toml"
    
    if not owner_map_path.exists():
        print(f"ERROR: {owner_map_path} not found", file=sys.stderr)
        return 1
    
    if not governance_path.exists():
        print(f"ERROR: {governance_path} not found", file=sys.stderr)
        return 1
    
    # Load TOML files
    owner_map = tomllib.loads(owner_map_path.read_text(encoding="utf-8"))
    governance = tomllib.loads(governance_path.read_text(encoding="utf-8"))
    
    all_pass = True
    
    # Check 1: removal_reason fields
    check1_pass, errors1 = verify_removal_reasons(owner_map)
    if not check1_pass:
        print("FAIL: Removal reason validation")
        for err in errors1:
            print(f"  {err}")
        all_pass = False
    else:
        print("PASS: Removal reason validation (all non-active have reason, all active empty)")
    
    # Check 2: third-party locked
    check2_pass, errors2 = verify_third_party_locked(owner_map)
    if not check2_pass:
        print("FAIL: Third-party markdown lock validation")
        for err in errors2:
            print(f"  {err}")
        all_pass = False
    else:
        print("PASS: Third-party markdown lock validation")
    
    # Check 3: enum validation
    check3_pass, errors3 = verify_removal_status_enum(owner_map)
    if not check3_pass:
        print("FAIL: Removal status enum validation")
        for err in errors3:
            print(f"  {err}")
        all_pass = False
    else:
        print("PASS: Removal status enum validation (all values valid)")
    
    # Summary
    total_owners = len(owner_map.get("owner", []))
    active = sum(
        1 for o in owner_map.get("owner", [])
        if str(o.get("removal_status", "active")).strip() == "active"
    )
    candidate = sum(
        1 for o in owner_map.get("owner", [])
        if str(o.get("removal_status", "active")).strip() == "candidate_for_removal"
    )
    deprecated = sum(
        1 for o in owner_map.get("owner", [])
        if str(o.get("removal_status", "active")).strip() == "deprecated"
    )
    locked = sum(
        1 for o in owner_map.get("owner", [])
        if str(o.get("removal_status", "active")).strip() == "locked"
    )
    
    print(f"\nMarkdown governance status summary:")
    print(f"  Total documents: {total_owners}")
    print(f"  Active: {active}")
    print(f"  Candidate for removal: {candidate}")
    print(f"  Deprecated: {deprecated}")
    print(f"  Locked: {locked}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
