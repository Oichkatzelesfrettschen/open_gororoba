---
description: Claude-specific overlay for open_gororoba; canonical policy lives in AGENTS.md
last_verified: 2026-08-13
---

# open_gororoba -- Claude operating guide

@AGENTS.md

The import above resolves to `./AGENTS.md`, the sibling of this file,
which carries the canonical operating guide: encoding and terminology
gates, warnings-as-errors, build environment, the SQLite-canonical
registry mutation workflow, research epistemics, GPU helper
foundation, comment hygiene, and commit-trailer policy. Read that file
directly if the import did not resolve; this one adds only what is
specific to Claude tooling and is not self-sufficient.

`AGENTS.md` is authoritative. When a Claude tool, skill, or agent
default conflicts with it, follow `AGENTS.md` and surface the conflict
to the user.

## Also read

| Source                                                                  | Carries                                                    |
| ----------------------------------------------------------------------- | ---------------------------------------------------------- |
| `~/.claude/CLAUDE.md`                                                   | Global user policy, which loads `~/AGENTS.md`              |
| `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/MEMORY.md` | Per-project memory index: claim counts, pitfalls, baselines |
| `data/output/debt_baseline_2026_04_30.toml`                             | Baseline debt numbers                                       |
| `plans/repo_debt_taxonomy_roadmap_2026_06_04.toml`                      | Active 22-class debt taxonomy                               |
| `docs/REQUIREMENTS.md`                                                  | Toolchain prerequisites                                     |

## Claude tool discipline

- **TaskCreate / TaskUpdate**: granular tracking is MANDATORY for
  multi-step work. Mark each subtask `in_progress` before starting and
  `completed` immediately after. One `in_progress` at a time.
- **AskUserQuestion**: ask EARLY when ambiguity would change the work.
  Do NOT begin implementation until the task list reflects the
  clarified scope.
- **Agent + Explore**: prefer the Explore agent for broad codebase
  searches over running 5+ greps directly. Use a specialised agent
  when its description matches the task.
- **Plan mode**: `ExitPlanMode` approves a plan; `AskUserQuestion`
  clarifies requirements within one. Do not substitute one for the
  other.
- **Registry writes**: never hand-edit a file whose first line is the
  AUTO-GENERATED header. Route through `gororoba-db` per the
  "Registry: SQLite-canonical" workflow in `AGENTS.md`.
- **`agents-render`**: MUST NOT be run. The renderer emits a 19-line
  stub and overwrites the hand-maintained `AGENTS.md`.

## Commit trailers for Claude

The full policy is in `AGENTS.md` under "Commit messages and PR
descriptions". The Claude-specific form:

```text
Assisted-by: Claude (Opus 5 1M context)
```

- USE `Assisted-by:`. Mesa reserves `Co-authored-by:` for human
  co-authors.
- DO NOT use the legacy
  `Co-Authored-By: Claude ... <noreply@anthropic.com>` trailer.
  Commits through 2026-05-17 carry it as a historical artifact; do not
  force-push to scrub them.
- Name the model actually used, not the string in this example.
- Commit bodies cover motivation, change, and evidence as prose in one
  to five sentences, cite primary sources by name, and list
  verification commands explicitly.

## Memory hygiene

- `MEMORY.md` is an index: one line per memory file, under ~150 chars.
  Topic files in the same directory hold the body.
- Save only what is non-obvious from the codebase, git history, or
  docs. Code patterns, file paths, and recent commits are derivable.
- Feedback memories record the reason behind the rule, so the edge
  cases stay judgeable. A bare rule rots faster than rule plus reason
  plus when-to-apply.
- Project memories convert relative dates to absolute on save.
- Before recommending from memory, verify the named files, functions,
  and flags still exist. A memory is a snapshot; renames happen.
