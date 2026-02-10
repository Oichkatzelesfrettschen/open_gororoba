#!/usr/bin/env python3
"""
One-time full markdown payload migration to TOML:
- registry/markdown_payloads.toml
- registry/markdown_payload_chunks.toml

Includes tracked, untracked, ignored, archived, third-party, and cache markdown files.
Payload bytes are preserved as base64 chunks for lossless reconstruction.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path


def _q(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _render_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_q(item) for item in values) + "]"


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _git_paths(repo_root: Path, args: list[str]) -> set[str]:
    cmd = ["git", *args, "--", "*.md"]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(line.strip() for line in proc.stdout.splitlines() if line.strip())


def _discover_markdown_files(repo_root: Path) -> list[str]:
    paths: list[str] = []
    for path in repo_root.rglob("*.md"):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith(".git/"):
            continue
        paths.append(rel)
    paths.sort()
    return paths


def _origin_class(path: str, generated: bool, third_party: bool) -> str:
    lowered = path.lower()
    if (
        third_party
        or "/venv/" in lowered
        or "/site-packages/" in lowered
        or lowered.startswith(".pytest_cache/")
        or "/.pytest_cache/" in lowered
    ):
        return "third_party_cache"
    if generated:
        return "project_generated"
    return "project_manual"


def _chunk_string(payload: str, chunk_size: int) -> list[str]:
    return [payload[idx : idx + chunk_size] for idx in range(0, len(payload), chunk_size)] or [""]


def _write(path: Path, text: str) -> None:
    _assert_ascii(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--inventory-path",
        default="registry/markdown_inventory.toml",
        help="Markdown inventory registry.",
    )
    parser.add_argument(
        "--owner-map-path",
        default="registry/markdown_owner_map.toml",
        help="Markdown owner map registry.",
    )
    parser.add_argument(
        "--payload-out",
        default="registry/markdown_payloads.toml",
        help="Output markdown payload metadata path.",
    )
    parser.add_argument(
        "--chunks-out",
        default="registry/markdown_payload_chunks.toml",
        help="Output markdown payload chunks path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Base64 chunk size.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    inventory_raw = tomllib.loads((repo_root / args.inventory_path).read_text(encoding="utf-8"))
    owner_raw = tomllib.loads((repo_root / args.owner_map_path).read_text(encoding="utf-8"))

    inventory_by_path = {
        str(row.get("path", "")): row for row in inventory_raw.get("document", []) if row.get("path")
    }
    owner_by_path = {
        str(row.get("path", "")): row for row in owner_raw.get("owner", []) if row.get("path")
    }

    tracked = _git_paths(repo_root, ["ls-files"])
    untracked = _git_paths(repo_root, ["ls-files", "--others", "--exclude-standard"])
    ignored = _git_paths(
        repo_root,
        ["ls-files", "--others", "--ignored", "--exclude-standard"],
    )

    markdown_paths = _discover_markdown_files(repo_root)
    documents: list[dict] = []
    chunks: list[dict] = []
    status_counts = {"tracked": 0, "untracked": 0, "ignored": 0, "filesystem_only": 0}
    origin_counts = {"project_manual": 0, "project_generated": 0, "third_party_cache": 0}

    for idx, rel_path in enumerate(markdown_paths, start=1):
        abs_path = repo_root / rel_path
        raw = abs_path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        size_bytes = len(raw)
        line_count = raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0)
        payload_b64 = base64.b64encode(raw).decode("ascii")
        part_chunks = _chunk_string(payload_b64, args.chunk_size)
        doc_id = f"MPY-{idx:05d}"

        inv_row = inventory_by_path.get(rel_path, {})
        owner_row = owner_by_path.get(rel_path, {})
        generated = bool(inv_row.get("generated", False))
        third_party = bool(inv_row.get("third_party", False))
        origin_class = _origin_class(rel_path, generated=generated, third_party=third_party)

        if rel_path in tracked:
            git_status = "tracked"
        elif rel_path in untracked:
            git_status = "untracked"
        elif rel_path in ignored:
            git_status = "ignored"
        else:
            git_status = "filesystem_only"
        status_counts[git_status] = status_counts.get(git_status, 0) + 1
        origin_counts[origin_class] = origin_counts.get(origin_class, 0) + 1

        chunk_ids: list[str] = []
        for part_idx, payload_part in enumerate(part_chunks, start=1):
            chunk_id = f"{doc_id}-C{part_idx:04d}"
            chunk_ids.append(chunk_id)
            chunks.append(
                {
                    "id": chunk_id,
                    "document_id": doc_id,
                    "chunk_index": part_idx,
                    "payload_b64": payload_part,
                }
            )

        documents.append(
            {
                "id": doc_id,
                "path": rel_path,
                "git_status": git_status,
                "origin_class": origin_class,
                "generated": generated,
                "third_party": third_party,
                "canonical_toml_owner": str(owner_row.get("canonical_toml", "")),
                "size_bytes": size_bytes,
                "line_count": line_count,
                "content_sha256": sha256,
                "content_encoding": "base64_chunks",
                "chunk_count": len(chunk_ids),
                "chunk_ids": chunk_ids,
            }
        )

    payload_lines: list[str] = [
        "# Full markdown payload registry (lossless one-time migration to TOML).",
        "# Generated by src/scripts/analysis/build_markdown_payload_registries.py",
        "",
        "[markdown_payloads]",
        'updated = "deterministic"',
        "authoritative = true",
        f"document_count = {len(documents)}",
        f"tracked_count = {status_counts.get('tracked', 0)}",
        f"untracked_count = {status_counts.get('untracked', 0)}",
        f"ignored_count = {status_counts.get('ignored', 0)}",
        f"filesystem_only_count = {status_counts.get('filesystem_only', 0)}",
        f"project_manual_count = {origin_counts.get('project_manual', 0)}",
        f"project_generated_count = {origin_counts.get('project_generated', 0)}",
        f"third_party_cache_count = {origin_counts.get('third_party_cache', 0)}",
        "",
    ]
    for row in documents:
        payload_lines.extend(
            [
                "[[document]]",
                f"id = {_q(row['id'])}",
                f"path = {_q(row['path'])}",
                f"git_status = {_q(row['git_status'])}",
                f"origin_class = {_q(row['origin_class'])}",
                f"generated = {'true' if row['generated'] else 'false'}",
                f"third_party = {'true' if row['third_party'] else 'false'}",
                f"canonical_toml_owner = {_q(row['canonical_toml_owner'])}",
                f"size_bytes = {int(row['size_bytes'])}",
                f"line_count = {int(row['line_count'])}",
                f"content_sha256 = {_q(row['content_sha256'])}",
                f"content_encoding = {_q(row['content_encoding'])}",
                f"chunk_count = {int(row['chunk_count'])}",
                f"chunk_ids = {_render_list(row['chunk_ids'])}",
                "",
            ]
        )

    chunk_lines: list[str] = [
        "# Markdown payload chunks (lossless base64 content shards).",
        "# Generated by src/scripts/analysis/build_markdown_payload_registries.py",
        "",
        "[markdown_payload_chunks]",
        'updated = "deterministic"',
        "authoritative = true",
        f"chunk_count = {len(chunks)}",
        f"document_count = {len(documents)}",
        "",
    ]
    chunks.sort(key=lambda item: (item["document_id"], int(item["chunk_index"])))
    for row in chunks:
        chunk_lines.extend(
            [
                "[[chunk]]",
                f"id = {_q(row['id'])}",
                f"document_id = {_q(row['document_id'])}",
                f"chunk_index = {int(row['chunk_index'])}",
                f"payload_b64 = {_q(row['payload_b64'])}",
                "",
            ]
        )

    payload_text = "\n".join(payload_lines)
    chunk_text = "\n".join(chunk_lines)
    _write(repo_root / args.payload_out, payload_text)
    _write(repo_root / args.chunks_out, chunk_text)

    print(
        "Wrote markdown payload registries: "
        f"documents={len(documents)} chunks={len(chunks)} "
        f"third_party_cache={origin_counts.get('third_party_cache', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
