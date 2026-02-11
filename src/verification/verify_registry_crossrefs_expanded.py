#!/usr/bin/env python3
"""
Expanded Cross-Registry Reference Verification (Wave 6, W6-016)
Comprehensive dangling reference detection across ALL canonical registries
Date: 2026-02-10
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_registry(path: str) -> dict:
    """Load TOML registry file with error handling."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(path, 'rb') as f:
        return tomllib.load(f)


def extract_claim_refs(registry: dict) -> Set[str]:
    """Extract all claim IDs defined in claims.toml."""
    claims = set()
    if 'claim' in registry:
        for entry in registry['claim']:
            if 'id' in entry:
                claims.add(entry['id'])
    return claims


def extract_insight_refs(registry: dict) -> Set[str]:
    """Extract all insight IDs defined in insights.toml."""
    insights = set()
    if 'insight' in registry:
        for entry in registry['insight']:
            if 'id' in entry:
                insights.add(entry['id'])
    return insights


def extract_experiment_refs(registry: dict) -> Set[str]:
    """Extract all experiment IDs defined in experiments.toml."""
    experiments = set()
    if 'experiment' in registry:
        for entry in registry['experiment']:
            if 'id' in entry:
                experiments.add(entry['id'])
    return experiments


def scan_registry_for_references(registry: dict, registry_name: str) -> Dict[str, List[str]]:
    """Scan registry for external references (claims, insights, experiments)."""
    references = {
        'claims': [],
        'insights': [],
        'experiments': []
    }

    def scan_dict(d: dict):
        for key, value in d.items():
            if isinstance(value, str):
                # Check for claim refs (C-XXXX)
                if 'claim' in key.lower() or 'C-' in value:
                    if 'C-' in value:
                        # Extract all C-XXXX references
                        import re
                        for match in re.finditer(r'C-\d+', value):
                            references['claims'].append((registry_name, match.group()))

                # Check for insight refs (I-XXXX)
                if 'insight' in key.lower() or 'I-' in value:
                    if 'I-' in value:
                        import re
                        for match in re.finditer(r'I-\d+', value):
                            references['insights'].append((registry_name, match.group()))

                # Check for experiment refs (E-XXXX)
                if 'experiment' in key.lower() or 'E-' in value:
                    if 'E-' in value and 'E-' not in key:
                        import re
                        for match in re.finditer(r'E-\d+', value):
                            references['experiments'].append((registry_name, match.group()))

            elif isinstance(value, dict):
                scan_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        scan_dict(item)

    scan_dict(registry)
    return references


def verify_cross_registry_references(registry_dir: str = "registry") -> Tuple[int, List[str]]:
    """
    Main verification function: check all cross-registry references for dangling links.

    Args:
        registry_dir: Directory containing TOML registries

    Returns:
        Tuple of (error_count, error_messages)
    """
    errors = []
    registry_path = Path(registry_dir)

    # Load all canonical registries
    print("Loading canonical registries...")
    claims_reg = load_registry(str(registry_path / "claims.toml"))
    insights_reg = load_registry(str(registry_path / "insights.toml"))
    experiments_reg = load_registry(str(registry_path / "experiments.toml"))

    # Extract all valid IDs
    valid_claims = extract_claim_refs(claims_reg)
    valid_insights = extract_insight_refs(insights_reg)
    valid_experiments = extract_experiment_refs(experiments_reg)

    print(f"  Claims: {len(valid_claims)} IDs")
    print(f"  Insights: {len(valid_insights)} IDs")
    print(f"  Experiments: {len(valid_experiments)} IDs")

    # Scan all registries for external references
    registries_to_scan = [
        ("claims.toml", claims_reg),
        ("insights.toml", insights_reg),
        ("experiments.toml", experiments_reg),
        ("third_party_markdown_cache.toml", load_registry(str(registry_path / "third_party_markdown_cache.toml"))),
        ("project_csv_canonical.toml", load_registry(str(registry_path / "project_csv_canonical.toml"))),
        ("schema_signatures.toml", load_registry(str(registry_path / "schema_signatures.toml"))),
    ]

    # Additional registries if they exist
    optional_registries = [
        "experiment_lineage_edges.toml",
        "knowledge/equation_atoms_v3.toml",
        "visual_payloads.toml",
    ]

    for reg_file in optional_registries:
        full_path = registry_path / reg_file
        if full_path.exists():
            registries_to_scan.append((reg_file, load_registry(str(full_path))))

    print(f"\nScanning {len(registries_to_scan)} registries for cross-references...")

    # Verify references
    for reg_name, registry in registries_to_scan:
        refs = scan_registry_for_references(registry, reg_name)

        # Check claim references
        for source_reg, claim_id in refs['claims']:
            if claim_id not in valid_claims:
                msg = f"DANGLING: {reg_name} references {claim_id} (not found in claims.toml)"
                errors.append(msg)
                print(f"  ✗ {msg}")

        # Check insight references
        for source_reg, insight_id in refs['insights']:
            if insight_id not in valid_insights:
                msg = f"DANGLING: {reg_name} references {insight_id} (not found in insights.toml)"
                errors.append(msg)
                print(f"  ✗ {msg}")

        # Check experiment references
        for source_reg, exp_id in refs['experiments']:
            if exp_id not in valid_experiments:
                msg = f"DANGLING: {reg_name} references {exp_id} (not found in experiments.toml)"
                errors.append(msg)
                print(f"  ✗ {msg}")

    # Additional checks for W6-016 enhancements
    print("\nPerforming W6-016 expansion checks...")

    # Check visual_payloads references
    if Path(registry_path / "visual_payloads.toml").exists():
        visual_reg = load_registry(str(registry_path / "visual_payloads.toml"))
        if 'visual' in visual_reg:
            for visual in visual_reg['visual']:
                if 'related_claims' in visual:
                    for claim_id in visual['related_claims']:
                        if claim_id not in valid_claims:
                            msg = f"DANGLING: visual_payloads {visual.get('id', '?')} references {claim_id}"
                            errors.append(msg)

    # Check experiment_lineage_edges references
    if Path(registry_path / "experiment_lineage_edges.toml").exists():
        edge_reg = load_registry(str(registry_path / "experiment_lineage_edges.toml"))
        if 'edge' in edge_reg:
            for edge in edge_reg['edge']:
                src_exp = edge.get('source_experiment')
                tgt_exp = edge.get('target_experiment')
                if src_exp and src_exp not in valid_experiments:
                    msg = f"DANGLING: lineage edge references {src_exp} (source)"
                    errors.append(msg)
                if tgt_exp and tgt_exp not in valid_experiments:
                    msg = f"DANGLING: lineage edge references {tgt_exp} (target)"
                    errors.append(msg)

    # Check equation_atoms_v3 references
    if Path(registry_path / "knowledge" / "equation_atoms_v3.toml").exists():
        eq_reg = load_registry(str(registry_path / "knowledge" / "equation_atoms_v3.toml"))
        if 'atom' in eq_reg:
            for atom in eq_reg['atom']:
                if 'supports_claim' in atom:
                    claim_id = atom['supports_claim']
                    if claim_id not in valid_claims:
                        msg = f"DANGLING: equation {atom.get('id')} supports {claim_id}"
                        errors.append(msg)

    if errors:
        print(f"\n❌ Found {len(errors)} dangling references")
    else:
        print("\n✓ All cross-registry references valid")

    return len(errors), errors


if __name__ == "__main__":
    error_count, error_messages = verify_cross_registry_references()

    if error_messages:
        print("\nDetailed errors:")
        for msg in error_messages:
            print(f"  - {msg}")

    sys.exit(1 if error_count > 0 else 0)
