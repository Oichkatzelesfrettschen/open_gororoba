#!/usr/bin/env python3
"""CI routing: map changed files to affected Rust crates.

Builds the workspace dependency graph from crates/*/Cargo.toml, maps changed
files to their owning crate, then computes the transitive downstream closure
(dependents) to determine the minimal set of crates that need clippy + test.

Usage
-----
  # GitHub Actions -- writes GITHUB_OUTPUT variables
  python3 scripts/ci_affected_crates.py

  # Local pre-push -- prints cargo scope flags to stdout
  python3 scripts/ci_affected_crates.py --local

  # Dry-run against specific commits
  python3 scripts/ci_affected_crates.py --local --base HEAD~3

Exit codes
----------
  0  Always (routing is advisory; errors fall back to --workspace).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATES_DIR = REPO_ROOT / "crates"

# Files whose change forces a full workspace rebuild.
WORKSPACE_TRIGGERS = {
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
}

# Prefixes that never affect Rust compilation.
RUST_IRRELEVANT_PREFIXES = (
    "registry/",
    "docs/",
    "papers/",
    "proofs/",
    "src/gemini_physics/",
    "src/ghost_stats/",
    "src/verification/",
    "bin/",
    "tests/",       # Python tests
    "data/",
    ".github/",
    ".githooks/",
    "scripts/",
    "plans/",
    ".horusec/",
)

# Prefixes that indicate governance/registry work.
GOVERNANCE_PREFIXES = (
    "registry/",
    "bin/",
    "src/verification/",
    "tests/",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--local",
        action="store_true",
        help="Print cargo scope flags to stdout (for pre-push hook).",
    )
    p.add_argument(
        "--base",
        default=None,
        help="Base ref for diff (default: auto-detect from push or origin/main).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostic info to stderr.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args: str) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def detect_base_ref(explicit_base: str | None) -> str:
    """Pick the best base ref for diffing."""
    if explicit_base:
        return explicit_base

    # In GitHub Actions, compare against the event base.
    gh_base = os.environ.get("GITHUB_BASE_REF")
    if gh_base:
        return f"origin/{gh_base}"

    gh_before = os.environ.get("GITHUB_EVENT_BEFORE")
    if gh_before and gh_before != "0" * 40:
        return gh_before

    # Local: diff against origin/main (or origin/master).
    for candidate in ("origin/main", "origin/master"):
        if git("rev-parse", "--verify", candidate):
            return candidate

    # Last resort: diff against HEAD~1.
    return "HEAD~1"


def changed_files(base: str) -> list[str]:
    """Return list of changed file paths relative to repo root."""
    raw = git("diff", "--name-only", f"{base}...HEAD")
    if not raw:
        # Maybe not a range -- try two-dot diff.
        raw = git("diff", "--name-only", base, "HEAD")
    if not raw:
        # Fallback: check unstaged + staged changes (local pre-push).
        raw = git("diff", "--name-only", "HEAD")
    return [f for f in raw.splitlines() if f]


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------

def _parse_workspace_path_deps() -> dict[str, str]:
    """Read root Cargo.toml [workspace.dependencies] for path-based entries.

    Returns a mapping from dependency name to crate directory name.
    For example, ``cd_kernel = { path = "crates/cd_kernel" }`` yields
    ``{"cd_kernel": "cd_kernel"}``.
    """
    root_cargo = REPO_ROOT / "Cargo.toml"
    if not root_cargo.is_file():
        return {}

    ws_path_re = re.compile(
        r'^\s*(\w[\w-]*)\s*=\s*\{[^}]*path\s*=\s*"crates/([\w-]+)"',
        re.MULTILINE,
    )
    text = root_cargo.read_text()
    return {m.group(1): m.group(2) for m in ws_path_re.finditer(text)}


def build_dependency_graph() -> tuple[set[str], dict[str, set[str]]]:
    """Parse crates/*/Cargo.toml to build forward dependency edges.

    Resolves both inline ``path = "../<dir>"`` deps and ``{ workspace = true }``
    deps whose path is declared in root Cargo.toml [workspace.dependencies].

    Returns:
        (all_crates, deps) where deps[crate] = set of crates it depends on.
    """
    all_crates: set[str] = set()
    deps: dict[str, set[str]] = defaultdict(set)

    if not CRATES_DIR.is_dir():
        return all_crates, deps

    # Collect all crate names first.
    for cargo_toml in sorted(CRATES_DIR.glob("*/Cargo.toml")):
        crate_name = cargo_toml.parent.name
        all_crates.add(crate_name)

    # Build workspace dep_name -> crate_dir map from root Cargo.toml.
    ws_path_map = _parse_workspace_path_deps()

    # Parse dependencies.
    # Pass 1: inline path deps (e.g. ``some_crate = { path = "../some_crate" }``).
    path_dep_re = re.compile(
        r'^\s*(\w[\w-]*)\s*=\s*\{[^}]*path\s*=\s*"\.\./([\w-]+)"',
        re.MULTILINE,
    )
    # Pass 2: workspace deps (e.g. ``some_crate = { workspace = true }``).
    ws_dep_re = re.compile(
        r'^\s*(\w[\w-]*)\s*=\s*\{[^}]*workspace\s*=\s*true',
        re.MULTILINE,
    )

    for cargo_toml in sorted(CRATES_DIR.glob("*/Cargo.toml")):
        crate_name = cargo_toml.parent.name
        text = cargo_toml.read_text()

        # Pass 1: inline path deps.
        for match in path_dep_re.finditer(text):
            dep_dir = match.group(2)
            if dep_dir in all_crates:
                deps[crate_name].add(dep_dir)

        # Pass 2: workspace = true deps resolved via root Cargo.toml paths.
        for match in ws_dep_re.finditer(text):
            dep_name = match.group(1)
            dep_dir = ws_path_map.get(dep_name)
            if dep_dir and dep_dir in all_crates and dep_dir != crate_name:
                deps[crate_name].add(dep_dir)

    return all_crates, deps


def invert_graph(deps: dict[str, set[str]]) -> dict[str, set[str]]:
    """Build reverse graph: for each crate, which crates depend on it."""
    rdeps: dict[str, set[str]] = defaultdict(set)
    for crate, upstream_set in deps.items():
        for upstream in upstream_set:
            rdeps[upstream].add(crate)
    return rdeps


def transitive_closure(
    seeds: set[str],
    rdeps: dict[str, set[str]],
) -> set[str]:
    """Compute transitive downstream dependents from seed crates."""
    visited: set[str] = set()
    stack = list(seeds)
    while stack:
        crate = stack.pop()
        if crate in visited:
            continue
        visited.add(crate)
        for dependent in rdeps.get(crate, set()):
            if dependent not in visited:
                stack.append(dependent)
    return visited


# ---------------------------------------------------------------------------
# File-to-crate mapping
# ---------------------------------------------------------------------------

def classify_changes(
    files: list[str],
    all_crates: set[str],
) -> tuple[set[str], bool, bool]:
    """Classify changed files into affected crates and flags.

    Returns:
        (affected_crates, force_workspace, has_governance_changes)
    """
    affected: set[str] = set()
    force_workspace = False
    has_governance = False
    has_rust_relevant = False

    for f in files:
        # Root workspace files force full rebuild.
        if f in WORKSPACE_TRIGGERS:
            force_workspace = True
            has_rust_relevant = True
            continue

        # Check governance relevance.
        if any(f.startswith(pfx) for pfx in GOVERNANCE_PREFIXES):
            has_governance = True

        # Check if this file is Rust-irrelevant.
        if any(f.startswith(pfx) for pfx in RUST_IRRELEVANT_PREFIXES):
            continue

        # Map crates/<name>/... to that crate.
        if f.startswith("crates/"):
            parts = f.split("/", 2)
            if len(parts) >= 2 and parts[1] in all_crates:
                affected.add(parts[1])
                has_rust_relevant = True
            else:
                # Unknown crate directory -- be conservative.
                force_workspace = True
                has_rust_relevant = True
            continue

        # Any other file at root level that might affect Rust builds.
        # (e.g., build.rs at root, .cargo/ files not in WORKSPACE_TRIGGERS)
        if f.endswith(".rs") or f.endswith(".toml"):
            force_workspace = True
            has_rust_relevant = True

    # If no Rust-relevant files changed, signal to skip Rust entirely.
    run_rust = has_rust_relevant or force_workspace or bool(affected)

    return affected, force_workspace, has_governance


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_cargo_scope(crates: set[str]) -> str:
    """Format crate set as cargo -p flags."""
    if not crates:
        return ""
    return " ".join(f"-p {c}" for c in sorted(crates))


def emit_local(
    scope: str,
    run_rust: bool,
    run_governance: bool,
    verbose: bool,
) -> None:
    """Print scope for local pre-push consumption."""
    if verbose:
        print(f"[ci-routing] run_rust={run_rust}", file=sys.stderr)
        print(f"[ci-routing] run_governance={run_governance}", file=sys.stderr)
        print(f"[ci-routing] scope={scope or '(skip)'}", file=sys.stderr)

    # For the pre-push hook: print just the scope flags.
    # Empty string means skip Rust. "--workspace" means full rebuild.
    print(scope)


def emit_github(
    scope: str,
    run_rust: bool,
    run_governance: bool,
    verbose: bool,
) -> None:
    """Write GitHub Actions outputs."""
    output_file = os.environ.get("GITHUB_OUTPUT", "")

    if verbose:
        print(f"[ci-routing] run_rust={run_rust}", file=sys.stderr)
        print(f"[ci-routing] run_governance={run_governance}", file=sys.stderr)
        print(f"[ci-routing] rust_scope={scope or '(skip)'}", file=sys.stderr)

    if output_file:
        with open(output_file, "a") as f:
            f.write(f"rust_scope={scope}\n")
            f.write(f"run_rust={'true' if run_rust else 'false'}\n")
            f.write(f"run_governance={'true' if run_governance else 'false'}\n")
    else:
        # Fallback: use deprecated set-output syntax or just print.
        print(f"::set-output name=rust_scope::{scope}")
        print(f"::set-output name=run_rust::{'true' if run_rust else 'false'}")
        print(f"::set-output name=run_governance::{'true' if run_governance else 'false'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    try:
        base = detect_base_ref(args.base)
        if args.verbose:
            print(f"[ci-routing] base_ref={base}", file=sys.stderr)

        files = changed_files(base)
        if args.verbose:
            print(
                f"[ci-routing] {len(files)} changed files",
                file=sys.stderr,
            )

        all_crates, deps = build_dependency_graph()
        rdeps = invert_graph(deps)

        affected, force_workspace, has_governance = classify_changes(
            files, all_crates,
        )

        if force_workspace:
            scope = "--workspace"
            run_rust = True
        elif affected:
            # Compute transitive closure of dependents.
            full_set = transitive_closure(affected, rdeps)
            scope = format_cargo_scope(full_set)
            run_rust = True
        else:
            scope = ""
            run_rust = False

        if args.local:
            emit_local(scope, run_rust, has_governance, args.verbose)
        else:
            emit_github(scope, run_rust, has_governance, args.verbose)

    except Exception as e:
        # Conservative fallback: any error -> full workspace build.
        print(f"[ci-routing] ERROR: {e} -- falling back to --workspace",
              file=sys.stderr)
        scope = "--workspace"
        if args.local:
            print(scope)
        else:
            emit_github(scope, True, True, args.verbose)


if __name__ == "__main__":
    main()
