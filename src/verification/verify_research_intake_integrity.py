#!/usr/bin/env python3
"""
Verify resolved research-intake artifacts.

Checks:
- Resolved index matches registry entry IDs with no duplicates.
- Status invariants:
  - ok rows require existing output file + non-empty sha256 + matching size/hash.
  - failed rows require explicit error.
- resolved_from values reference existing known index files.
- provenance_resolved.toml matches resolved index counts and per-ID rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import tomllib


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry_ids(path: Path) -> list[str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    ids: list[str] = []
    for row in data.get("entry", []):
        entry_id = str(row.get("id", "")).strip()
        if entry_id:
            ids.append(entry_id)
    return ids


def _load_index_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "id",
            "topic",
            "resource_class",
            "status",
            "method",
            "size_bytes",
            "sha256",
            "output_path",
            "error",
            "resolved_from",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"ERROR: resolved index missing required columns: {sorted(missing)}"
            )
        rows: list[dict[str, str]] = []
        for row in reader:
            normalized = {str(k): str(v or "").strip() for k, v in row.items()}
            if normalized.get("id", ""):
                rows.append(normalized)
        return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--registry",
        default="registry/research_intake_2026_02_14.toml",
        help="Intake registry TOML path.",
    )
    parser.add_argument(
        "--intake-root",
        default="data/external/intake/2026_02_14_hypercomplex_news",
        help="Resolved intake root path.",
    )
    parser.add_argument(
        "--resolved-index",
        default="index_resolved.tsv",
        help="Resolved index filename within intake root.",
    )
    parser.add_argument(
        "--resolved-provenance",
        default="provenance_resolved.toml",
        help="Resolved provenance filename within intake root.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    registry_path = root / args.registry
    intake_root = root / args.intake_root
    resolved_index_path = intake_root / args.resolved_index
    resolved_provenance_path = intake_root / args.resolved_provenance

    failures: list[str] = []
    for required in [registry_path, intake_root, resolved_index_path, resolved_provenance_path]:
        if not required.exists():
            failures.append(f"missing required path: {required}")
    if failures:
        for item in failures:
            print(f"ERROR: {item}")
        return 1

    registry_ids = _load_registry_ids(registry_path)
    rows = _load_index_rows(resolved_index_path)
    row_ids = [row["id"] for row in rows]
    row_id_set = set(row_ids)
    registry_id_set = set(registry_ids)

    if len(row_ids) != len(row_id_set):
        failures.append("duplicate ids found in resolved index")

    missing_from_index = sorted(registry_id_set.difference(row_id_set))
    extra_in_index = sorted(row_id_set.difference(registry_id_set))
    if missing_from_index:
        failures.append(f"registry ids missing from resolved index: {missing_from_index}")
    if extra_in_index:
        failures.append(f"resolved index contains unknown ids: {extra_in_index}")

    known_index_files = {
        path.relative_to(root).as_posix()
        for path in intake_root.glob("index*.tsv")
        if path.is_file()
    }
    hex_chars = set("0123456789abcdef")

    success_count = 0
    failure_count = 0
    for row in rows:
        entry_id = row["id"]
        status = row["status"]
        method = row["method"]
        sha = row["sha256"]
        output_rel = row["output_path"]
        error = row["error"]
        resolved_from = row["resolved_from"]
        size_raw = row["size_bytes"]

        if resolved_from not in known_index_files:
            failures.append(
                f"{entry_id}: resolved_from does not point to known index file: {resolved_from}"
            )

        try:
            size_val = int(size_raw)
        except ValueError:
            failures.append(f"{entry_id}: size_bytes is not an integer: {size_raw!r}")
            size_val = 0

        if status == "ok":
            success_count += 1
            if not method:
                failures.append(f"{entry_id}: ok row has empty method")
            if not output_rel:
                failures.append(f"{entry_id}: ok row has empty output_path")
            if not sha:
                failures.append(f"{entry_id}: ok row has empty sha256")
            elif len(sha) != 64 or any(ch not in hex_chars for ch in sha.lower()):
                failures.append(f"{entry_id}: invalid sha256 format: {sha!r}")

            if output_rel:
                output_path = root / output_rel
                if not output_path.exists():
                    failures.append(f"{entry_id}: output file missing: {output_rel}")
                else:
                    actual_size = output_path.stat().st_size
                    if actual_size != size_val:
                        failures.append(
                            f"{entry_id}: size mismatch index={size_val} actual={actual_size}"
                        )
                    actual_sha = _sha256(output_path)
                    if sha and actual_sha != sha:
                        failures.append(f"{entry_id}: sha256 mismatch index={sha} actual={actual_sha}")
            if size_val <= 0:
                failures.append(f"{entry_id}: ok row must have size_bytes > 0")
            if error:
                failures.append(f"{entry_id}: ok row should have empty error field")
        elif status == "failed":
            failure_count += 1
            if not error:
                failures.append(f"{entry_id}: failed row missing explicit error trace")
        else:
            failures.append(f"{entry_id}: invalid status {status!r}")

    provenance = tomllib.loads(resolved_provenance_path.read_text(encoding="utf-8"))
    batch = provenance.get("batch", {})
    artifacts = provenance.get("artifact", [])

    prov_entry_count = int(batch.get("entry_count", -1))
    prov_success_count = int(batch.get("success_count", -1))
    prov_failure_count = int(batch.get("failure_count", -1))
    if prov_entry_count != len(rows):
        failures.append(
            f"provenance entry_count={prov_entry_count} does not match resolved rows={len(rows)}"
        )
    if prov_success_count != success_count:
        failures.append(
            f"provenance success_count={prov_success_count} does not match index success_count={success_count}"
        )
    if prov_failure_count != failure_count:
        failures.append(
            f"provenance failure_count={prov_failure_count} does not match index failure_count={failure_count}"
        )

    selected_retry = [str(item).strip() for item in batch.get("selected_retry_indices", [])]
    selected_retry_set = {item for item in selected_retry if item}
    for row in rows:
        source = row["resolved_from"]
        if source.endswith("/index.tsv"):
            continue
        if source not in selected_retry_set:
            failures.append(
                f"{row['id']}: resolved_from retry file is missing from selected_retry_indices"
            )

    artifacts_by_id = {str(item.get("id", "")).strip(): item for item in artifacts}
    if len(artifacts) != len(artifacts_by_id):
        failures.append("provenance_resolved contains duplicate artifact ids")
    for row in rows:
        entry_id = row["id"]
        art = artifacts_by_id.get(entry_id)
        if art is None:
            failures.append(f"{entry_id}: missing artifact row in provenance_resolved.toml")
            continue
        for field in ("status", "method", "output_path", "sha256", "resolved_from"):
            index_val = row[field]
            prov_val = str(art.get(field, "")).strip()
            if index_val != prov_val:
                failures.append(
                    f"{entry_id}: mismatch {field} index={index_val!r} provenance={prov_val!r}"
                )
        prov_size = int(art.get("size_bytes", -1))
        if prov_size != int(row["size_bytes"]):
            failures.append(
                f"{entry_id}: mismatch size_bytes index={row['size_bytes']} provenance={prov_size}"
            )

    if failures:
        print("ERROR: research intake integrity verification failed.")
        for item in failures:
            print(f"- {item}")
        return 1

    print(
        "OK: research intake integrity verified. "
        f"entries={len(rows)} success={success_count} failure={failure_count} "
        f"selected_retries={len(selected_retry_set)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
