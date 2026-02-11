#!/usr/bin/env python3

"""
Registry Event Tracking Verification (Wave 7, W7-001)
Validates monotonic event ordering, record ID resolution, and event consistency
Date: 2026-02-10
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime, timezone


def load_toml(path: str) -> dict:
    """Load TOML registry file with error handling."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(path, 'rb') as f:
        return tomllib.load(f)


def extract_claim_ids(registry: dict) -> Set[str]:
    """Extract all claim IDs from claims.toml."""
    claims = set()
    if 'claim' in registry:
        for entry in registry['claim']:
            if 'id' in entry:
                claims.add(entry['id'])
    return claims


def extract_insight_ids(registry: dict) -> Set[str]:
    """Extract all insight IDs from insights.toml."""
    insights = set()
    if 'insight' in registry:
        for entry in registry['insight']:
            if 'id' in entry:
                insights.add(entry['id'])
    return insights


def extract_experiment_ids(registry: dict) -> Set[str]:
    """Extract all experiment IDs from experiments.toml."""
    experiments = set()
    if 'experiment' in registry:
        for entry in registry['experiment']:
            if 'id' in entry:
                experiments.add(entry['id'])
    return experiments


def extract_scroll_ids(registry: dict) -> Set[str]:
    """Extract all scroll IDs from artifact_scrolls.toml."""
    scrolls = set()
    if 'scroll' in registry:
        for entry in registry['scroll']:
            if 'id' in entry:
                scrolls.add(entry['id'])
    return scrolls


def validate_event_monotonic_sequence(events: List[dict]) -> Tuple[bool, List[str]]:
    """Verify sequence_number is monotonically increasing."""
    errors = []
    prev_seq = -1
    for idx, event in enumerate(events):
        seq = event.get('sequence_number', -1)
        if seq <= prev_seq:
            errors.append(f"Event {idx} sequence_number {seq} not > previous {prev_seq}")
        prev_seq = seq
    return len(errors) == 0, errors


def validate_event_monotonic_timestamps(events: List[dict]) -> Tuple[bool, List[str]]:
    """Verify timestamps are monotonically increasing."""
    errors = []
    prev_ts = datetime.min.replace(tzinfo=timezone.utc)
    for idx, event in enumerate(events):
        ts_str = event.get('timestamp', '')
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            if ts <= prev_ts:
                errors.append(f"Event {idx} timestamp {ts_str} not > previous {prev_ts}")
            prev_ts = ts
        except ValueError:
            errors.append(f"Event {idx} timestamp {ts_str} invalid format")
    return len(errors) == 0, errors


def validate_event_record_ids(
    events: List[dict],
    valid_claims: Set[str],
    valid_insights: Set[str],
    valid_experiments: Set[str],
    valid_scrolls: Set[str],
) -> Tuple[bool, List[str]]:
    """Verify all record IDs exist in their target registries."""
    errors = []
    special_prefixes = ['W', 'EVT', 'FTR']  # Special record IDs (features, framework)

    for idx, event in enumerate(events):
        record_id = event.get('record_id', '')
        registry_file = event.get('registry_file', '')

        # Check if record_id is special
        if any(record_id.startswith(p) for p in special_prefixes):
            continue

        # Resolve by registry file target
        if 'claims' in registry_file.lower():
            if record_id not in valid_claims:
                errors.append(
                    f"Event {idx} record_id {record_id} not found in claims.toml"
                )
        elif 'insights' in registry_file.lower():
            if record_id not in valid_insights:
                errors.append(
                    f"Event {idx} record_id {record_id} not found in insights.toml"
                )
        elif 'experiments' in registry_file.lower():
            if record_id not in valid_experiments:
                errors.append(
                    f"Event {idx} record_id {record_id} not found in experiments.toml"
                )
        elif 'artifact_scrolls' in registry_file.lower():
            if record_id not in valid_scrolls:
                errors.append(
                    f"Event {idx} record_id {record_id} not found in artifact_scrolls.toml"
                )

    return len(errors) == 0, errors


def validate_event_change_types(events: List[dict]) -> Tuple[bool, List[str]]:
    """Verify change_type enumeration."""
    errors = []
    valid_types = {'add', 'update', 'refute', 'deprecate'}

    for idx, event in enumerate(events):
        change_type = event.get('change_type', '')
        if change_type not in valid_types:
            errors.append(
                f"Event {idx} change_type '{change_type}' not in {valid_types}"
            )

    return len(errors) == 0, errors


def validate_event_authors(events: List[dict]) -> Tuple[bool, List[str]]:
    """Verify author field is non-empty."""
    errors = []

    for idx, event in enumerate(events):
        author = event.get('author', '').strip()
        if not author:
            errors.append(f"Event {idx} author field is empty")

    return len(errors) == 0, errors


def validate_event_commit_shas(events: List[dict]) -> Tuple[bool, List[str]]:
    """Verify commit_sha format (40-char hex or 'uncommitted')."""
    errors = []

    for idx, event in enumerate(events):
        sha = event.get('commit_sha', '')
        if sha != 'uncommitted' and not re.match(r'^[a-f0-9]{40}$', sha):
            errors.append(f"Event {idx} commit_sha '{sha}' invalid format")

    return len(errors) == 0, errors


def validate_event_duplicate_ids(events: List[dict]) -> Tuple[bool, List[str]]:
    """Verify no duplicate event IDs."""
    errors = []
    seen_ids = {}

    for idx, event in enumerate(events):
        event_id = event.get('id', '')
        if event_id in seen_ids:
            errors.append(
                f"Event {idx} ID '{event_id}' duplicates earlier event at index {seen_ids[event_id]}"
            )
        else:
            seen_ids[event_id] = idx

    return len(errors) == 0, errors


def validate_event_refute_restrictions(events: List[dict]) -> Tuple[bool, List[str]]:
    """Verify refute events only apply to claims (C-NNN)."""
    errors = []

    for idx, event in enumerate(events):
        if event.get('change_type') == 'refute':
            record_id = event.get('record_id', '')
            if not record_id.startswith('C-'):
                errors.append(
                    f"Event {idx} refute change_type applied to non-claim record {record_id}"
                )

    return len(errors) == 0, errors


def validate_registry_events(registry_dir: str = "registry") -> Tuple[int, List[str]]:
    """
    Main verification function: validate all events in registry_events.toml.

    Returns:
        Tuple of (error_count, error_messages)
    """
    errors = []
    registry_path = Path(registry_dir)

    print("Loading canonical registries...")
    claims_reg = load_toml(str(registry_path / "claims.toml"))
    insights_reg = load_toml(str(registry_path / "insights.toml"))
    experiments_reg = load_toml(str(registry_path / "experiments.toml"))
    artifact_scrolls_reg = load_toml(str(registry_path / "artifact_scrolls.toml"))
    events_reg = load_toml(str(registry_path / "registry_events.toml"))

    # Extract valid IDs
    valid_claims = extract_claim_ids(claims_reg)
    valid_insights = extract_insight_ids(insights_reg)
    valid_experiments = extract_experiment_ids(experiments_reg)
    valid_scrolls = extract_scroll_ids(artifact_scrolls_reg)

    print(f"  Claims: {len(valid_claims)} IDs")
    print(f"  Insights: {len(valid_insights)} IDs")
    print(f"  Experiments: {len(valid_experiments)} IDs")
    print(f"  Scrolls: {len(valid_scrolls)} IDs")

    # Extract events
    events = events_reg.get('event', [])
    print(f"\nValidating {len(events)} events...")

    # Run all validation gates
    print("\n[Gate 1] Monotonic sequence_number")
    ok, errs = validate_event_monotonic_sequence(events)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  [FAIL] {err}")
    else:
        print("  [PASS] PASS")

    print("[Gate 2] Monotonic timestamps")
    ok, errs = validate_event_monotonic_timestamps(events)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  [FAIL] {err}")
    else:
        print("  [PASS] PASS")

    print("[Gate 3] Record ID resolution")
    ok, errs = validate_event_record_ids(
        events,
        valid_claims,
        valid_insights,
        valid_experiments,
        valid_scrolls,
    )
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  [FAIL] {err}")
    else:
        print("  [PASS] PASS")

    print("[Gate 4] Change type enumeration")
    ok, errs = validate_event_change_types(events)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  [FAIL] {err}")
    else:
        print("  [PASS] PASS")

    print("[Gate 5] Author non-empty")
    ok, errs = validate_event_authors(events)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  [FAIL] {err}")
    else:
        print("  [PASS] PASS")

    print("[Gate 6] Commit SHA format")
    ok, errs = validate_event_commit_shas(events)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  [FAIL] {err}")
    else:
        print("  [PASS] PASS")

    print("[Gate 7] Duplicate event IDs")
    ok, errs = validate_event_duplicate_ids(events)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  [FAIL] {err}")
    else:
        print("  [PASS] PASS")

    print("[Gate 8] Refute type restrictions")
    ok, errs = validate_event_refute_restrictions(events)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  [FAIL] {err}")
    else:
        print("  [PASS] PASS")

    if errors:
        print(f"\n[FAIL] Found {len(errors)} validation errors")
    else:
        print("\n[PASS] All registry events valid")

    return len(errors), errors


if __name__ == "__main__":
    error_count, error_messages = validate_registry_events()

    if error_messages:
        print("\nDetailed errors:")
        for msg in error_messages:
            print(f"  - {msg}")

    sys.exit(1 if error_count > 0 else 0)
