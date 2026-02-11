#!/usr/bin/env python3
"""
Verify MCP server parity and runtime smoke health.

Checks:
1. Cross-client config parity against registry/mcp_server_matrix.toml.
2. Startup probes for each expected enabled server.
3. Disabled servers remain absent across all clients.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

HELP_TIMEOUT_SEC = 8
STARTUP_TIMEOUT_SEC = 3
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_servers(path: Path, payload: dict[str, Any]) -> tuple[set[str], dict[str, dict[str, Any]]]:
    if path.suffix.lower() == ".toml":
        table = payload.get("mcp_servers", {})
    else:
        table = payload.get("mcpServers", {})
    if not isinstance(table, dict):
        return set(), {}

    names = set()
    specs: dict[str, dict[str, Any]] = {}
    for name, raw in table.items():
        if not isinstance(name, str):
            continue
        names.add(name)
        if isinstance(raw, dict):
            specs[name] = raw
    return names, specs


def _expand_env_value(value: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return os.environ.get(key, match.group(0))

    return ENV_PATTERN.sub(_repl, value)


def _build_env(raw_env: Any) -> dict[str, str]:
    merged = dict(os.environ)
    if not isinstance(raw_env, dict):
        return merged
    for key, value in raw_env.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            merged[key] = _expand_env_value(value)
        else:
            merged[key] = str(value)
    return merged


def _extract_command(spec: dict[str, Any]) -> tuple[str | None, list[str], str | None, dict[str, str]]:
    command = spec.get("command")
    if not isinstance(command, str) or not command.strip():
        return None, [], None, dict(os.environ)

    raw_args = spec.get("args", [])
    args: list[str] = []
    if isinstance(raw_args, list):
        for entry in raw_args:
            args.append(str(entry))
    cwd = spec.get("cwd")
    if not isinstance(cwd, str):
        cwd = None
    env = _build_env(spec.get("env", {}))
    return command, args, cwd, env


def _probe_help(command: str, args: list[str], cwd: str | None, env: dict[str, str]) -> tuple[bool, str]:
    cmd = [command, *args, "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=HELP_TIMEOUT_SEC,
            check=False,
        )
    except FileNotFoundError:
        return False, "missing_binary"
    except subprocess.TimeoutExpired:
        return False, "help_timeout"
    except OSError as exc:
        return False, f"os_error:{exc}"

    combined = (proc.stdout + "\n" + proc.stderr).lower()
    if proc.returncode == 0:
        return True, f"help_ok_rc_{proc.returncode}"
    if any(token in combined for token in ("usage", "--help", "options", "mcp", "stdio")):
        return True, f"help_text_rc_{proc.returncode}"
    return False, f"help_failed_rc_{proc.returncode}"


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=1)


def _probe_startup(command: str, args: list[str], cwd: str | None, env: dict[str, str]) -> tuple[bool, str]:
    cmd = [command, *args]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return False, "missing_binary"
    except OSError as exc:
        return False, f"os_error:{exc}"

    try:
        stdout, stderr = proc.communicate(timeout=STARTUP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        _terminate_process(proc)
        return True, "startup_alive"

    message = (stderr or stdout).strip().splitlines()
    first_line = message[0] if message else ""
    lower_line = first_line.lower()
    if proc.returncode == 0:
        return True, "startup_exit_0"
    if "connectionclosed" in lower_line and "initialized request" in lower_line:
        return True, "startup_handshake_close"
    if first_line:
        return False, f"startup_failed_rc_{proc.returncode}:{first_line[:160]}"
    return False, f"startup_failed_rc_{proc.returncode}"


def _probe_server(name: str, spec: dict[str, Any]) -> tuple[bool, str]:
    command, args, cwd, env = _extract_command(spec)
    if command is None:
        return False, f"{name}:missing_command"

    ok, reason = _probe_help(command, args, cwd, env)
    if ok:
        return True, reason
    if reason in {"missing_binary"} or reason.startswith("os_error:"):
        return False, reason
    return _probe_startup(command, args, cwd, env)


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP parity and smoke verifier")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path.",
    )
    parser.add_argument(
        "--matrix-path",
        default="registry/mcp_server_matrix.toml",
        help="Path to MCP matrix TOML, relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matrix_path = repo_root / args.matrix_path
    if not matrix_path.is_file():
        print(f"ERROR: missing MCP matrix file: {matrix_path}")
        return 1

    matrix = _load_toml(matrix_path)
    config_paths = matrix.get("client_config_paths", {})
    if not isinstance(config_paths, dict):
        print("ERROR: [client_config_paths] missing in registry/mcp_server_matrix.toml")
        return 1
    policy = matrix.get("policy", {})

    expected_enabled = set(matrix.get("all_detected_servers", {}).get("names", []))
    expected_disabled = set(matrix.get("disabled_servers_2026_02_11", {}).get("names", []))
    if not expected_enabled:
        print("ERROR: expected enabled server set is empty")
        return 1
    expected_enabled_count = policy.get("expected_enabled_count") if isinstance(policy, dict) else None
    expected_disabled_count = policy.get("expected_disabled_count") if isinstance(policy, dict) else None
    if isinstance(expected_enabled_count, int) and expected_enabled_count != len(expected_enabled):
        print(
            f"ERROR: regression in matrix enabled count: expected {expected_enabled_count}, "
            f"found {len(expected_enabled)}"
        )
        return 1
    if isinstance(expected_disabled_count, int) and expected_disabled_count != len(expected_disabled):
        print(
            f"ERROR: regression in matrix disabled count: expected {expected_disabled_count}, "
            f"found {len(expected_disabled)}"
        )
        return 1

    print(f"INFO: expected enabled MCP servers: {len(expected_enabled)}")
    print(f"INFO: expected disabled MCP servers: {len(expected_disabled)}")

    failures: list[str] = []
    canonical_specs: dict[str, dict[str, Any]] = {}

    for client_name in sorted(config_paths.keys()):
        raw_path = config_paths[client_name]
        if not isinstance(raw_path, str) or not raw_path:
            failures.append(f"{client_name}: invalid config path")
            continue

        path = Path(raw_path).expanduser()
        if not path.is_file():
            failures.append(f"{client_name}: missing config file {path}")
            continue

        try:
            payload = _load_toml(path) if path.suffix.lower() == ".toml" else _load_json(path)
        except Exception as exc:
            failures.append(f"{client_name}: parse error {path}: {exc}")
            continue

        enabled_set, specs = _extract_servers(path, payload)
        if client_name == "codex_primary":
            canonical_specs = specs

        missing = sorted(expected_enabled - enabled_set)
        extra = sorted(enabled_set - expected_enabled)
        disabled_present = sorted(enabled_set & expected_disabled)

        print(f"INFO: {client_name}: {len(enabled_set)} servers configured")
        if missing:
            failures.append(f"{client_name}: missing expected servers: {', '.join(missing)}")
        if extra:
            failures.append(f"{client_name}: unexpected servers enabled: {', '.join(extra)}")
        if disabled_present:
            failures.append(f"{client_name}: disabled servers present: {', '.join(disabled_present)}")

    if not canonical_specs:
        failures.append("codex_primary: could not load canonical MCP command specs")

    for name in sorted(expected_enabled):
        spec = canonical_specs.get(name)
        if not isinstance(spec, dict):
            failures.append(f"probe:{name}: missing command spec in codex_primary config")
            continue

        ok, reason = _probe_server(name, spec)
        status = "PASS" if ok else "FAIL"
        print(f"MCP_SMOKE {status} {name} ({reason})")
        if not ok:
            failures.append(f"probe:{name}: {reason}")

    if failures:
        print("ERROR: MCP smoke verification failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print(f"OK: MCP smoke passed for {len(expected_enabled)} servers with cross-client parity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
