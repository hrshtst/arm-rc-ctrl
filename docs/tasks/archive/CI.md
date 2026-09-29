<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# CI ledger archive

These are retained ledger records, not the active queue. IDs, statuses and
evidence are preserved; relative links were adjusted for the move.
Consult [the queue](../../TASKS.md) and [task index](../README.md) for
current priorities. Historical statements describe their recorded time.

## Cross-cutting quality

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| CI-001 | `DONE` | Restore the 90 % branch-coverage gate without the external evidence store | UP-010 | Found 2026-09-29: with no store (as on CI) the gate reached 88.63 % (last green CI `f808afa`: 90.63 %), because the report and reproduction modules added since are covered only by store-backed tests. Store-independent tests with synthetic inputs lift the store-less total to at least 90 % without new coverage omissions. Evidence 2026-09-29: six `tests/unit/*_offline.py` files (89 test functions, 92 collected cases) fake only the store reads and exercise the real rendering, binding and refusal paths; with the store unavailable the P-core-pinned gate passed at 91.69 % (3077 passed, 570 skipped, 1 xfailed) and `nox -s pre_commit` passed. `manual_expert_report` 39→99 %, `manual_search_report` 69→96 %, `manual_search_report_plots` 41→95 %, `recovery_report` 55→99 %, `trajectory_plots` 62→98 %, `reproduce_1a` 60→100 %, `reproduce_repetition` 66→97 %. No `src/` or coverage configuration change. Open observation for the owner: `manual_search_report.case_inputs` raises a bare `KeyError` for a missing illustrated row |
