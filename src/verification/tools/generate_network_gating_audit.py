#!/usr/bin/env python3
"""
Generate an audit report for network gating hygiene.

This is opt-in (writes under reports/). CI/smoke use verifiers instead.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

_NETWORK_IMPORT_RE = re.compile(
    r"(^|\n)\s*(import|from)\s+(requests|urllib|urllib3|httpx|astroquery)\b",
    flags=re.MULTILINE,
)
_NETWORK_CMD_RE = re.compile(r"\b(curl|wget|aria2c)\b")
_GATING_TOKEN_RE = re.compile(r"\b(require_network|allow_network)\s*\(")
_ENV_GATE_RE = re.compile(r"\bGEMINI_ALLOW_NETWORK\b")


@dataclass(frozen=True)
class Finding:
    path: str
    kind: str  # "package" | "script"
    has_network: bool
    has_gate_token: bool
    has_env_gate: bool
    script_contract_network: str


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _parse_script_contract_network(text: str) -> str:
    # Minimal parse: look for "network:" lines after "# SCRIPT_CONTRACT:".
    lines = text.splitlines()[:200]
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "# SCRIPT_CONTRACT:":
            start = i + 1
            break
    if start is None:
        return "missing"

    for line in lines[start : start + 80]:
        if not line.startswith("#"):
            break
        m = re.match(r"^#\s*network\s*:\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip()
    return "missing"


def _scan_file(p: Path, kind: str) -> Finding:
    text = _read_text(p)
    has_network = bool(_NETWORK_IMPORT_RE.search(text) or _NETWORK_CMD_RE.search(text))
    has_gate_token = bool(_GATING_TOKEN_RE.search(text))
    has_env_gate = bool(_ENV_GATE_RE.search(text))
    contract_net = _parse_script_contract_network(text) if kind == "script" else "n/a"
    return Finding(
        path=p.as_posix(),
        kind=kind,
        has_network=has_network,
        has_gate_token=has_gate_token,
        has_env_gate=has_env_gate,
        script_contract_network=contract_net,
    )


def _render(findings: list[Finding], repo_root: Path) -> str:
    today = _dt.date.today().isoformat()
    lines: list[str] = []
    lines.append(f"# Network Gating Audit ({today})")
    lines.append("")
    lines.append("Goal: identify network-capable codepaths and confirm opt-in gating exists.")
    lines.append("")
    total = len(findings)
    net = sum(1 for f in findings if f.has_network)
    ungated = sum(1 for f in findings if f.has_network and not (f.has_gate_token or f.has_env_gate))
    lines.append(f"- Files scanned: {total}")
    lines.append(f"- Network-indicator hits: {net}")
    lines.append(f"- Network-indicator hits without obvious gating: {ungated}")
    lines.append("")
    lines.append("## Findings (network-indicator hits)")
    lines.append("")
    lines.append("| path | kind | contract.network | gate_token | env_gate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for f in sorted([x for x in findings if x.has_network], key=lambda x: (x.kind, x.path)):
        rel = str(Path(f.path).resolve().relative_to(repo_root))
        gate_token = str(f.has_gate_token).lower()
        env_gate = str(f.has_env_gate).lower()
        lines.append(
            f"| `{rel}` | {f.kind} | {f.script_contract_network} | "
            f"{gate_token} | {env_gate} |"
        )
    lines.append("")
    lines.append("## Checklist")
    lines.append("")
    lines.append("- [ ] Every script with network clients declares `# network: gated`.")
    lines.append(
        "- [ ] Every network script calls `require_network()` (directly or via a gated helper)."
    )
    lines.append(
        "- [ ] Every package module that can fetch remote data uses `gemini_physics.network`"
    )
    lines.append("  gating.")
    lines.append("- [ ] Make targets that use the network are gated behind")
    lines.append("  `GEMINI_ALLOW_NETWORK=1`.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/audit_network_gating.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    # Prefer locating repo root via CWD (works when invoked via Makefile).
    here = Path.cwd().resolve()
    p = here
    repo_root = None
    while p != p.parent:
        if (p / ".git").exists():
            repo_root = p
            break
        p = p.parent
    if repo_root is None:
        # Fallback: file layout repo_root/src/verification/tools/<this file>
        repo_root = Path(__file__).resolve().parents[3]

    src_root = repo_root / "src"

    findings: list[Finding] = []
    for p in sorted((src_root / "gemini_physics").rglob("*.py")):
        findings.append(_scan_file(p, kind="package"))
    for p in sorted(src_root.glob("*.py")):
        findings.append(_scan_file(p, kind="script"))
    for p in sorted((src_root / "scripts").rglob("*.py")):
        findings.append(_scan_file(p, kind="script"))

    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(findings, repo_root=repo_root), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
