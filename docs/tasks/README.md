<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Task locations

Start with [the queue](../TASKS.md). Do not read this entire archive tree for a
current task. Each ID has one canonical row; active rows move on completion
rather than being copied. File names/group headings are navigation, not a
second status database.

| IDs | Canonical location |
| --- | --- |
| M3MS-001 through M3MS-005 | [M3MS archive](archive/M3MS.md) |
| M3MS-006 onward | [Active queue](../TASKS.md#manual-data-esn-search--m3ms) |
| UP-001 through UP-005 | [Open upstream queue](../TASKS.md#cross-cutting-upstream-work) |
| DOC-* | [DOC archive](archive/DOC.md) |
| M0-* | [M0 archive](archive/M0.md) |
| M1-* | [M1 archive](archive/M1.md) |
| M2-* | [M2 archive](archive/M2.md) |
| M3-* | [M3 archive](archive/M3.md) |
| TOOL-* | [TOOL archive](archive/TOOL.md) |
| M3R-* | [M3R archive](archive/M3R.md) |
| M3REP-* | [M3REP archive](archive/M3REP.md) |
| M3MAN-* | [M3MAN archive](archive/M3MAN.md) |
| UP-006 through UP-010 | [Completed upstream archive](archive/UP.md) |
| M4-* | [M4 backlog](backlog/M4.md) |
| M5-* | [M5 backlog](backlog/M5.md) |
| M6-* | [M6 backlog](backlog/M6.md) |
| M7-* | [M7 backlog](backlog/M7.md) |
| REP-001 | [Deferred replication](backlog/replication.md) |

M3R-017 remains BLOCKED in a closed protocol; M4-001 remains IN PROGRESS for
planning with an unresolved owner gate. Neither has been silently completed,
reopened or authorized by this move.

For an exact task ID, search the row rather than full histories:

```bash
rg -n '^\| M3MAN-012 \|' docs/TASKS.md docs/tasks
```

Historical [status snapshot](archive/status-2026-09-22.md),
[phase roadmap](archive/roadmap-2026-09-22.md), and
[migration account](archive/migration-2026-09-22.md) preserve context.
