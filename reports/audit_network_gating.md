<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# Network Gating Audit (2026-02-06)

Goal: identify network-capable codepaths and confirm opt-in gating exists.

- Files scanned: 87
- Network-indicator hits: 5
- Network-indicator hits without obvious gating: 5

## Findings (network-indicator hits)

| path | kind | contract.network | gate_token | env_gate |
| --- | --- | --- | --- | --- |
| `src/gemini_physics/materials_jarvis.py` | package | n/a | false | false |
| `src/fetch_external_data.py` | script | missing | false | false |
| `src/fetch_observatory_data.py` | script | missing | false | false |
| `src/fetch_pulsars_rigorous.py` | script | missing | false | false |
| `src/map_cosmic_objects.py` | script | missing | false | false |

## Checklist

- [ ] Every script with network clients declares `# network: gated`.
- [ ] Every network script calls `require_network()` (directly or via a gated helper).
- [ ] Every package module that can fetch remote data uses `gemini_physics.network`
  gating.
- [ ] Make targets that use the network are gated behind
  `GEMINI_ALLOW_NETWORK=1`.
