#!/usr/bin/env python3
"""
Seed sqlite memoization tables for Pantheon/PhysicsForge migration bookkeeping.

Canonical sources:
- registry/pantheon_physicsforge_migration_findings.toml
- registry/pantheon_physicsforge_overflow_tracker.toml
"""

from __future__ import annotations

import argparse
import sqlite3
import tomllib
from pathlib import Path


FINDINGS_PATH = "registry/pantheon_physicsforge_migration_findings.toml"
OVERFLOW_PATH = "registry/pantheon_physicsforge_overflow_tracker.toml"


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _join_refs(values: list[str]) -> str:
    return " | ".join(str(v).strip() for v in values if str(v).strip())


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS migration_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS migration_findings (
            finding_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            phase INTEGER NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL,
            summary TEXT NOT NULL,
            owner TEXT NOT NULL,
            evidence_refs TEXT NOT NULL,
            updated TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS unresolved_risks (
            risk_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            phase INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            status TEXT NOT NULL,
            summary TEXT NOT NULL,
            mitigation TEXT NOT NULL,
            owner TEXT NOT NULL,
            eta TEXT NOT NULL,
            evidence_refs TEXT NOT NULL,
            updated TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS overflow_tasks (
            overflow_id TEXT PRIMARY KEY,
            source_task_id TEXT NOT NULL,
            phase INTEGER NOT NULL,
            status TEXT NOT NULL,
            owner TEXT NOT NULL,
            eta TEXT NOT NULL,
            rationale TEXT NOT NULL,
            deferral_rationale TEXT NOT NULL,
            evidence_refs TEXT NOT NULL,
            updated TEXT NOT NULL
        );
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        default="build/pantheon_physicsforge_migration.db",
        help="Path to sqlite database output.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    findings_raw = _load(repo_root / FINDINGS_PATH)
    overflow_raw = _load(repo_root / OVERFLOW_PATH)

    findings_meta = findings_raw.get("migration_findings", {})
    findings = findings_raw.get("finding", [])
    risks = findings_raw.get("risk", [])

    overflow_meta = overflow_raw.get("overflow_tracker", {})
    overflow_rows = overflow_raw.get("overflow_task", [])

    max_active = int(overflow_meta.get("max_active_tasks", 5))
    active_statuses = {
        str(v).strip() for v in overflow_meta.get("active_statuses", ["open", "in_progress", "blocked"])
    }
    active_rows = [row for row in overflow_rows if str(row.get("status", "")).strip() in active_statuses]
    if len(active_rows) > max_active:
        raise SystemExit(
            "ERROR: overflow tracker violates max active policy before sqlite seed: "
            f"active={len(active_rows)} max_active={max_active}"
        )

    db_path = (repo_root / args.db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)

        conn.execute(
            "INSERT INTO migration_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("source_findings_toml", FINDINGS_PATH),
        )
        conn.execute(
            "INSERT INTO migration_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("source_overflow_toml", OVERFLOW_PATH),
        )
        conn.execute(
            "INSERT INTO migration_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("max_active_overflow", str(max_active)),
        )

        for row in findings:
            conn.execute(
                """
                INSERT INTO migration_findings(
                    finding_id, task_id, phase, severity, status, summary, owner, evidence_refs, updated
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(finding_id) DO UPDATE SET
                    task_id=excluded.task_id,
                    phase=excluded.phase,
                    severity=excluded.severity,
                    status=excluded.status,
                    summary=excluded.summary,
                    owner=excluded.owner,
                    evidence_refs=excluded.evidence_refs,
                    updated=excluded.updated
                """,
                (
                    str(row.get("id", "")).strip(),
                    str(row.get("task_id", "")).strip(),
                    int(row.get("phase", 0)),
                    str(row.get("severity", "")).strip(),
                    str(row.get("status", "")).strip(),
                    str(row.get("summary", "")).strip(),
                    str(row.get("owner", "")).strip(),
                    _join_refs([str(v) for v in row.get("evidence_refs", [])]),
                    str(findings_meta.get("updated", "")).strip(),
                ),
            )

        for row in risks:
            conn.execute(
                """
                INSERT INTO unresolved_risks(
                    risk_id, task_id, phase, risk_level, status, summary, mitigation, owner, eta, evidence_refs, updated
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(risk_id) DO UPDATE SET
                    task_id=excluded.task_id,
                    phase=excluded.phase,
                    risk_level=excluded.risk_level,
                    status=excluded.status,
                    summary=excluded.summary,
                    mitigation=excluded.mitigation,
                    owner=excluded.owner,
                    eta=excluded.eta,
                    evidence_refs=excluded.evidence_refs,
                    updated=excluded.updated
                """,
                (
                    str(row.get("id", "")).strip(),
                    str(row.get("task_id", "")).strip(),
                    int(row.get("phase", 0)),
                    str(row.get("risk_level", "")).strip(),
                    str(row.get("status", "")).strip(),
                    str(row.get("summary", "")).strip(),
                    str(row.get("mitigation", "")).strip(),
                    str(row.get("owner", "")).strip(),
                    str(row.get("eta", "")).strip(),
                    _join_refs([str(v) for v in row.get("evidence_refs", [])]),
                    str(findings_meta.get("updated", "")).strip(),
                ),
            )

        for row in overflow_rows:
            conn.execute(
                """
                INSERT INTO overflow_tasks(
                    overflow_id, source_task_id, phase, status, owner, eta, rationale, deferral_rationale, evidence_refs, updated
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(overflow_id) DO UPDATE SET
                    source_task_id=excluded.source_task_id,
                    phase=excluded.phase,
                    status=excluded.status,
                    owner=excluded.owner,
                    eta=excluded.eta,
                    rationale=excluded.rationale,
                    deferral_rationale=excluded.deferral_rationale,
                    evidence_refs=excluded.evidence_refs,
                    updated=excluded.updated
                """,
                (
                    str(row.get("id", "")).strip(),
                    str(row.get("source_task_id", "")).strip(),
                    int(row.get("phase", 0)),
                    str(row.get("status", "")).strip(),
                    str(row.get("owner", "")).strip(),
                    str(row.get("eta", "")).strip(),
                    str(row.get("rationale", "")).strip(),
                    str(row.get("deferral_rationale", "")).strip(),
                    _join_refs([str(v) for v in row.get("evidence_refs", [])]),
                    str(overflow_meta.get("updated", "")).strip(),
                ),
            )

        finding_count = conn.execute("SELECT COUNT(*) FROM migration_findings").fetchone()[0]
        risk_count = conn.execute("SELECT COUNT(*) FROM unresolved_risks").fetchone()[0]
        overflow_count = conn.execute("SELECT COUNT(*) FROM overflow_tasks").fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    print(
        "OK: sqlite migration memoization seeded "
        f"(db={db_path}, findings={finding_count}, risks={risk_count}, overflow_tasks={overflow_count})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
