# plans/

Working directory for repo-internal roadmaps and execution plans.

## Lifecycle

1. **Active**. A roadmap file lives directly in `plans/` while it is the source
   of truth for ongoing work. At most one roadmap is the canonical "current"
   plan at any time; older roadmaps that are still consulted during sprint
   close (e.g., to settle priority drift) may also live at the top level
   temporarily.
2. **Archived**. When a plan is superseded by a newer one, it moves to
   `plans/archive/` with no other change. Move via `git mv` to preserve
   history; do not edit the archived file.
3. **Deleted**. Plans that were created in error or are fully redundant with
   another archived plan can be removed; the commit message must explain
   why deletion is preferable to archival.

## Naming convention

- `<topic>_<YYYY_MM_DD>.toml` -- TOML roadmaps used by older sprints.
- `<topic>.md` -- newer plans written via the plan-mode workflow.
  The canonical mirror of the user-side plan file may live in
  `~/.claude/plans/<topic>.md`; either location is valid as long as one
  of them is referenced from `registry/project.toml` or a commit message.

## Current state (2026-04-30)

- Active: `repo_debt_roadmap_2026_04_11.toml` -- the legacy debt roadmap
  whose execution is being subsumed and elaborated by the
  `elucidate-and-build-out-nested-hollerith` plan (mirror in
  `~/.claude/plans/`).
- Archived (10 files): see `plans/archive/`. All predate
  `repo_debt_roadmap_2026_04_11.toml` and were retired in T-102 of the
  Phase 0 baseline pass.

## Sprint-close ritual

Per the elucidate-and-build-out plan (T-138, Phase 3), a future
`make sprint-close` target should:

- archive the current `plans/*.toml` and `plans/*.md` files,
- increment the sprint counter in `registry/project.toml`,
- validate phase-completion gates,
- print a summary report.

Until that target is wired, archive moves are performed manually.
