# Branch Lifecycle (Phase 0 T-103)

## Context

This repository uses short-lived feature branches per the user's git
workflow policy. A class of automatically generated agent branches --
`worktree-agent-<8hex>` -- accumulates over time when the `Agent` tool
is invoked with `isolation: "worktree"` but the spawned agent's run
ends without merging or deleting the branch ref.

At the start of the elucidate-and-build-out-nested-hollerith plan
(2026-04-30, baseline `970b4da3`) two such branches existed:

| Branch                      | SHA       |
|-----------------------------|-----------|
| `worktree-agent-ac02ad93`   | `b0fb87d6` |
| `worktree-agent-afe06576`   | `b0fb87d6` |

Both pointed to `b0fb87d6`, an ancestor of the then-current main.
Zero unique commits relative to main; no worktree checked out at
either ref.

## Decision

`worktree-agent-*` branches whose tip is reachable from `main`
(i.e., zero unique commits) and whose worktree is no longer present
in `git worktree list` are SAFE TO DELETE via `git branch -d`. The
underlying commits are preserved through `main`'s history; deletion
removes only the branch ref.

Branches with unique commits require a per-branch decision:
merge / rebase+merge / cherry-pick / discard-after-review. Discard is
only acceptable after `git log -p <branch> ^main` confirms no
salvageable work.

## Cleanup procedure

```bash
# 1. Verify no unique commits relative to main:
git log --oneline <branch> ^main
# Empty output means the branch is fully reachable from main.

# 2. Verify no worktree checked out at the branch:
git worktree list

# 3. Safe-delete (errors if not merged):
git branch -d <branch>
```

If step 1 shows unique commits, do not proceed with `-d`; pick one
of the resolution paths above and document the rationale in the
commit message.

## Phase 0 T-103 outcome

Per Phase 0 of the plan, both `worktree-agent-ac02ad93` and
`worktree-agent-afe06576` were deleted via `git branch -d`. The
commit `b0fb87d6` itself remains in main's history. No work was lost.

This ADR codifies the procedure so future sprint-close passes
(plan T-138) can apply it without rederiving the safety analysis.

## Related

- Plan: `plans/elucidate-and-build-out-nested-hollerith.md` (mirror in
  `~/.claude/plans/`)
- User policy: CLAUDE.md NO SHORTCUTS section -- destructive git ops
  require explicit authorization; this ADR documents that
  authorization for the `git branch -d` family applied to this branch
  class only.
