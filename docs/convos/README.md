<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_convos.toml -->

# Convos audit workspace

Policy:
- Files under `convos/` are treated as immutable inputs (they may contain Unicode).
- All synthesis produced under `docs/` should be ASCII-only.
- Every claim extracted from `convos/` is treated as a hypothesis that must be tied to:
  - a first-party source (paper/book/manual), and
  - a repo test and/or a reproducible artifact.

Entry points:
- Main index: `docs/CONVOS_CONCEPTS_STATUS_INDEX.md`
- Primary chunk audit log (file 1): `docs/convos/audit_1_read_nonuser_lines_cont.md`

Chunking convention (default):
- Chunk size: 800 lines
- Chunk ids: C1-0001, C1-0002, ...
- Reference format: `convos/1_read_nonuser_lines_cont.md:L<line>`

Audit entry fields (recommended):
- Chunk id
- Line range
- Extracted concepts (map to CX-xxx)
- Extracted claims (write as hypotheses)
- Immediate next actions (tests, docs, code, sources)
- Source needs (first-party citations to add)
