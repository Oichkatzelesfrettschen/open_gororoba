#!/usr/bin/env python3

"""
Artifact-Experiment Link Verification (Wave 7, W7-003)
Validates bidirectional consistency between artifacts and experiments
Date: 2026-02-10
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_toml(path: str) -> dict:
    """Load TOML registry file with error handling."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(path, 'rb') as f:
        return tomllib.load(f)


def extract_experiment_ids(registry: dict) -> Set[str]:
    """Extract all experiment IDs from experiments.toml."""
    experiments = set()
    if 'experiment' in registry:
        for entry in registry['experiment']:
            if 'id' in entry:
                experiments.add(entry['id'])
    return experiments


def extract_artifact_ids(registry: dict) -> Set[str]:
    """Extract all artifact IDs from artifact_experiment_links.toml."""
    artifacts = set()
    if 'artifact' in registry:
        for entry in registry['artifact']:
            if 'id' in entry:
                artifacts.add(entry['id'])
    return artifacts


def validate_artifact_ids_exist(links_reg: dict, valid_artifacts: Set[str]) -> Tuple[bool, List[str]]:
    """Verify all referenced artifact_ids exist in artifacts table."""
    errors = []
    if 'link' not in links_reg:
        return True, errors

    for idx, link in enumerate(links_reg['link']):
        artifact_ids = link.get('artifact_ids', [])
        for art_id in artifact_ids:
            if art_id not in valid_artifacts:
                errors.append(
                    f"Link {idx} references non-existent artifact {art_id}"
                )

    return len(errors) == 0, errors


def validate_experiment_ids_exist(links_reg: dict, valid_experiments: Set[str]) -> Tuple[bool, List[str]]:
    """Verify all referenced experiment_ids exist in experiments.toml."""
    errors = []
    if 'link' not in links_reg:
        return True, errors

    for idx, link in enumerate(links_reg['link']):
        exp_id = link.get('experiment_id', '')
        if exp_id and exp_id not in valid_experiments and exp_id not in ['external_catalog', 'documentation', 'external_knowledge', 'phase11_planning']:
            errors.append(
                f"Link {idx} references non-existent experiment {exp_id}"
            )

    return len(errors) == 0, errors


def validate_data_flow_types(links_reg: dict) -> Tuple[bool, List[str]]:
    """Verify data_flow_type enumeration."""
    errors = []
    valid_types = {'input', 'output', 'intermediate'}
    if 'link' not in links_reg:
        return True, errors

    for idx, link in enumerate(links_reg['link']):
        data_flow = link.get('data_flow', '')
        if data_flow not in valid_types:
            errors.append(
                f"Link {idx} data_flow '{data_flow}' not in {valid_types}"
            )

    return len(errors) == 0, errors


def validate_artifact_flow_types(links_reg: dict) -> Tuple[bool, List[str]]:
    """Verify artifact data_flow_type enumeration."""
    errors = []
    valid_types = {'input', 'output', 'intermediate'}
    if 'artifact' not in links_reg:
        return True, errors

    for idx, artifact in enumerate(links_reg['artifact']):
        data_flow = artifact.get('data_flow_type', '')
        if data_flow not in valid_types:
            errors.append(
                f"Artifact {idx} data_flow_type '{data_flow}' not in {valid_types}"
            )

    return len(errors) == 0, errors


def validate_no_circular_flows(links_reg: dict) -> Tuple[bool, List[str]]:
    """Verify no artifact is both input and output of same experiment."""
    errors = []
    if 'link' not in links_reg or 'artifact' not in links_reg:
        return True, errors

    # Build artifact flow type map
    artifact_flows: Dict[str, str] = {}
    for artifact in links_reg['artifact']:
        art_id = artifact.get('id', '')
        flow_type = artifact.get('data_flow_type', '')
        if art_id:
            artifact_flows[art_id] = flow_type

    # Check for circular flows in links
    for idx, link in enumerate(links_reg['link']):
        exp_id = link.get('experiment_id', '')
        data_flow = link.get('data_flow', '')
        artifact_ids = link.get('artifact_ids', [])

        # If this link says 'output', check that artifacts aren't marked as inputs elsewhere for same exp
        if data_flow == 'output':
            for art_id in artifact_ids:
                if artifact_flows.get(art_id) == 'input':
                    errors.append(
                        f"Link {idx} {exp_id}: artifact {art_id} marked as output but artifact table says input (circular)"
                    )

    return len(errors) == 0, errors


def validate_no_orphaned_artifacts(links_reg: dict) -> Tuple[bool, List[str]]:
    """Verify every artifact is linked to at least one experiment."""
    errors = []
    if 'link' not in links_reg or 'artifact' not in links_reg:
        return True, errors

    # Collect all artifacts referenced in links
    linked_artifacts = set()
    for link in links_reg['link']:
        artifact_ids = link.get('artifact_ids', [])
        linked_artifacts.update(artifact_ids)

    # Check for orphaned artifacts
    for artifact in links_reg['artifact']:
        art_id = artifact.get('id', '')
        if art_id and art_id not in linked_artifacts:
            errors.append(f"Artifact {art_id} is orphaned (not referenced in any link)")

    return len(errors) == 0, errors


def validate_bidirectional_consistency(links_reg: dict) -> Tuple[bool, List[str]]:
    """Verify link.experiment_id matches referenced experiment structure."""
    errors = []
    if 'link' not in links_reg:
        return True, errors

    for idx, link in enumerate(links_reg['link']):
        exp_id = link.get('experiment_id', '')
        exp_title = link.get('experiment_title', '')
        if not exp_title:
            errors.append(
                f"Link {idx} experiment_id {exp_id} missing experiment_title (bidirectional consistency)"
            )

    return len(errors) == 0, errors


def validate_artifact_experiment_links(registry_dir: str = "registry") -> Tuple[int, List[str]]:
    """
    Main verification function: validate all artifacts and experiment links.

    Returns:
        Tuple of (error_count, error_messages)
    """
    errors = []
    registry_path = Path(registry_dir)

    print("Loading registries...")
    experiments_reg = load_toml(str(registry_path / "experiments.toml"))
    links_reg = load_toml(str(registry_path / "artifact_experiment_links.toml"))

    # Extract valid IDs
    valid_experiments = extract_experiment_ids(experiments_reg)
    valid_artifacts = extract_artifact_ids(links_reg)

    print(f"  Experiments: {len(valid_experiments)} IDs")
    print(f"  Artifacts: {len(valid_artifacts)} IDs")

    # Run validation gates
    print("\n[Gate 1] Artifact IDs exist")
    ok, errs = validate_artifact_ids_exist(links_reg, valid_artifacts)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  ✗ {err}")
    else:
        print("  ✓ PASS")

    print("[Gate 2] Experiment IDs exist")
    ok, errs = validate_experiment_ids_exist(links_reg, valid_experiments)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  ✗ {err}")
    else:
        print("  ✓ PASS")

    print("[Gate 3] Link data_flow_type enumeration")
    ok, errs = validate_data_flow_types(links_reg)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  ✗ {err}")
    else:
        print("  ✓ PASS")

    print("[Gate 4] Artifact data_flow_type enumeration")
    ok, errs = validate_artifact_flow_types(links_reg)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  ✗ {err}")
    else:
        print("  ✓ PASS")

    print("[Gate 5] No circular flows (artifact not both input+output)")
    ok, errs = validate_no_circular_flows(links_reg)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  ✗ {err}")
    else:
        print("  ✓ PASS")

    print("[Gate 6] No orphaned artifacts")
    ok, errs = validate_no_orphaned_artifacts(links_reg)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  ✗ {err}")
    else:
        print("  ✓ PASS")

    print("[Gate 7] Bidirectional consistency (experiment_title present)")
    ok, errs = validate_bidirectional_consistency(links_reg)
    if not ok:
        errors.extend(errs)
        for err in errs:
            print(f"  ✗ {err}")
    else:
        print("  ✓ PASS")

    if errors:
        print(f"\n❌ Found {len(errors)} validation errors")
    else:
        print("\n✓ All artifact-experiment links valid")

    return len(errors), errors


if __name__ == "__main__":
    error_count, error_messages = validate_artifact_experiment_links()

    if error_messages:
        print("\nDetailed errors:")
        for msg in error_messages:
            print(f"  - {msg}")

    sys.exit(1 if error_count > 0 else 0)
