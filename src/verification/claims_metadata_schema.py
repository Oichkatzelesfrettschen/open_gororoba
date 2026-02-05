"""
Canonical metadata schema for claim-audit trackers.

Single source of truth for:
- docs/CLAIMS_EVIDENCE_MATRIX.md status tokens
- docs/CLAIMS_TASKS.md task status tokens

Rationale: keep verification scripts consistent and avoid silent drift.
"""

from __future__ import annotations

CANONICAL_CLAIMS_STATUS_TOKENS: tuple[str, ...] = (
    "Verified",
    "Partially verified",
    "Unverified",
    "Speculative",
    "Modeled",
    "Literature",
    "Theoretical",
    "Not supported",
    "Refuted",
    "Clarified",
    "Established",
)

CANONICAL_TASK_STATUS_TOKENS: tuple[str, ...] = (
    "TODO",
    "IN PROGRESS",
    "PARTIAL",
    "DONE",
    "REFUTED",
    "DEFERRED",
    "BLOCKED",
)

