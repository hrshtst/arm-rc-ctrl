---
name: next-task
description: Start a docs/TASKS.md ledger task the way docs/PLAN.md section 15 requires (pick the given or next unblocked task, mark it IN PROGRESS, failing test first, focused tests then the gate, DONE with evidence in the same commit). Use when the user says "next task", "start M4-002", or asks what to work on next.
---

# Ledger task loop

Arguments: `$ARGUMENTS` is an optional task ID (for example `M4-002`). Work from
the repository root with absolute paths; never `cd`.

## 1. Pick the task

- Read `docs/TASKS.md`: the "Current focus" block (Next task, Current milestone,
  Active blockers) and the milestone tables. Columns: ID, Status, Task, Depends
  on, Acceptance/evidence. Statuses are `TODO`, `IN PROGRESS`, `BLOCKED`, `DONE`.
- With an ID given, use it. Otherwise take the "Next task" pointer, or the first
  `TODO` task whose "Depends on" entries are all `DONE`.
- Stop and ask instead of starting when the task is `BLOCKED`, when it needs an
  owner decision or protocol lock that is not recorded, when it involves a
  confirmatory run, hardware, a push, or a PR, or when another task is already
  `IN PROGRESS` and unrelated (keep that set small).

## 2. Frame it

Restate in a few lines: the acceptance/evidence text, the plan section it comes
from (docs/PLAN.md), the files it will touch, and what evidence the ledger row
will cite (tests, artifacts, gate result). Edit the row's status to
`IN PROGRESS` now; do not commit a status-only change.

## 3. Implement

1. Add or update a failing test or specification first.
2. Implement the smallest coherent behavior that passes it. Do not mix
   formatting, refactoring, and behavior changes unless inseparable.
3. Focused tests: `uv run pytest <path>::<test>`.
4. Full gate before claiming completion: run the `gate` skill
   (`uv run --locked nox`, then `uv run --locked nox -s pre_commit`), and
   report exit codes.
5. Align documentation with the tested behavior once it stabilizes.

Respect the CLAUDE.md gotchas: versioned artifacts are never edited in place,
generated reports are regenerated rather than hand-edited, and configs are
recorded only through `provenance.portable_config`.

## 4. Close

- Set the row to `DONE` and write the evidence into its Acceptance/evidence
  cell (commands, artifacts, gate result, CI run when one exists). Update the
  "Current focus" block (Next task, Latest completed work, Last updated).
- One commit for the task (or a small cohesive group) that contains the
  implementation, tests, docs, and the ledger update together. Conventional
  Commit subject, task IDs in the body (`Tasks: M4-002`), no session links.
  Write the message to a scratch file and use `git commit -F <file>`; then
  check `git log -1 --stat`.
- Do not push. Report what was done, the evidence, and the next unblocked task.
