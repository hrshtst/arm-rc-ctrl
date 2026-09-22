<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Documentation migration, 2026-09-22 (DOC-009)

The owner approved reorganizing the large roadmap and ledger and updating
`AGENTS.md`. This migration also registers the previously approved
[manual-data ESN search](../../experiments/task_1a_manual_esn_search/plan.md)
without starting its implementation or execution.

## Source identities

The pre-move files are available from Git commit `55b1f3d5f74167ac21d843e3f3346c5dd5cc346d`:

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| `docs/PLAN.md` | 54638 | `5e57e646c2ea2c127f67aff9198432b2563cd4f08bb6fc3a66a296538d9fcb4b` |
| `docs/TASKS.md` | 321597 | `95577dd33242992c2c6f642dbd9db24b643519a02a703f57dd4578eff1117955` |

Their detailed contents were moved, not discarded:

- All 18 numbered roadmap sections retain their original wording and section
  numbers under [shared design references](../../design/README.md) or the
  [historical phase roadmap](roadmap-2026-09-22.md). Relative links were rebased.
- All 181 original task IDs, statuses, dependencies and row text are retained
  in exactly one canonical row each, apart from adjusted link destinations.
  Closed milestone rows are in their archive; later gated epics and REP-001
  are in the backlog; open upstream rows remain in the active queue.
- `M3R-017` stays BLOCKED, `REP-001` stays TODO, and `M4-001` stays IN PROGRESS
  for protocol planning only. Their permissions and exit criteria did not change.
- The previous current-focus narrative and milestone-gate table are preserved
  in a [dated snapshot](status-2026-09-22.md), explicitly historical because
  several statements in that accumulated narrative supersede earlier ones.
- The old definition of done and milestone checklist moved to
  [workflow](../../design/workflow.md#definition-of-done-for-every-implementation-task).
  “This file” now means a task's canonical ledger location rather than the
  entry point; no acceptance requirement was removed.
- Legacy `PLAN.md` and `TASKS.md` section fragments remain valid navigation
  anchors. Digest-bound configs, code, recipes, experiment evidence and their
  embedded old section citations were not edited.

## Current structure and authority

[PLAN.md](../../PLAN.md) is the short roadmap and numbered section router.
[TASKS.md](../../TASKS.md) is the queue and task-location entry point.
[The task index](../README.md) locates a prefix or exact old ID. Each task has
one canonical status row; a milestone link is not a duplicate status record.

The approved search is registered as M3MS-001 through M3MS-GATE, all TODO.
DOC-009 records this documentation migration. No existing task was marked
complete merely because its row moved. The migration does not authorize task
1-b implementation, hardware operation or an unplanned scientific run.

`AGENTS.md` and `CLAUDE.md` now direct agents to read the two entry points first,
then only the current task's plan and relevant reference/evidence sections.
Roadmap/queue limits are 8 KiB / 15 KiB. Detailed review histories belong beside
the experiment or in a completed milestone archive, not in the current-focus
paragraph. Reopening moves a canonical row back to the queue; it does not
create a second status record.

## Size change

The two entry points shrank from 376,235 bytes to 17,528 bytes
(95.3% smaller). Their detailed contents remain available through
scoped links; this reduction does not count moving text as deleting history.

## Verification

The migration check compares every original task row and status to its unique
new location, ignoring only Markdown link destinations, and compares every
original numbered roadmap section to its new file. It resolves moved relative
links and fragments, checks inbound links to the two entry points and verifies
that all legacy subsection anchors still exist. It also checks the entry-point
size limits and the exact new task IDs. Pre-commit checks cover every changed
or added document. No runtime code, configuration or scientific payload was
changed; no simulation or numerical audit is needed for this documentation move.
