<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/research_narratives.toml -->

# Fetching External Sources (wget-first, offline reproducibility)

Access date: 2026-02-02

This repo treats narrative text as hypotheses unless backed by:
- primary sources (cached under `data/external/`), and/or
- offline checks (tests/verifiers/scripts).

Network access is therefore opt-in and must never be required by `make test` or `make check`.

## Principles

- Prefer `wget` (user preference) for direct PDF/CSV/JSON fetches.
- If a source is blocked (403/WAF), cache a trace artifact and document the block.
- Never keep empty placeholder files under `data/external/` (smoke gates reject 0-byte files).
- After adding/removing external cached files:
  - run `make provenance` (updates `data/external/PROVENANCE.local.json`),
  - and (for PDFs) run `pdftotext` to create a `.txt` sidecar.

## Common patterns

### 1) Plain fetch (open access)

```bash
wget -O data/external/papers/<stable_name>.pdf 'https://arxiv.org/pdf/<id>.pdf'
pdftotext -layout data/external/papers/<stable_name>.pdf data/external/papers/<stable_name>.txt
PYTHONWARNINGS=error make provenance
```

Then index the source in `docs/external_sources/<TOPIC>_SOURCES.md`.

### 2) Tor fetch (when normal path is blocked)

Use `torsocks` with `wget` directly:

```bash
torsocks wget -O data/external/papers/<stable_name>.pdf 'https://example.com/file.pdf'
pdftotext -layout data/external/papers/<stable_name>.pdf data/external/papers/<stable_name>.txt
PYTHONWARNINGS=error make provenance
```

### 3) Blocked sources (paywall/WAF)

If a source cannot be fetched automatically (e.g., publisher auth, Cloudflare/Akamai):

1) Cache a short trace artifact under `data/external/traces/` describing:
   - attempted URL(s)
   - timestamp
   - error code / block page notes
2) Keep the claim in the appropriate status (e.g., `Literature`, `Speculative`) until the
   primary source is cached and/or independently reproduced.
3) If you have access as a human, manually download the PDF and place it at the documented
   target path, then run `pdftotext` and `make provenance`.

## Tor service troubleshooting (port 9050)

If running `tor` manually fails with "Address already in use", it usually means the system
Tor service is already running and bound to 127.0.0.1:9050.

Use:

```bash
systemctl status tor
```

For repo fetches, prefer `torsocks wget` against the existing Tor service.
