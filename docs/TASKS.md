<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Implementation task queue

**Updated:** 2026-09-22. **Roadmap:** [PLAN.md](PLAN.md).

This is the authoritative entry point for work selection. Each task has exactly
one canonical status/acceptance row: here for active work, in the linked backlog
for later epics, or in its milestone archive for completed/deferred history.
[Find older IDs](tasks/README.md); search an exact ID instead of reading every archive.

## Status rules

`TODO`: defined, not started. `IN PROGRESS`: actively worked, keep this set small.
`BLOCKED`: record the blocker and required decision/evidence. `DONE`: acceptance
criteria and evidence complete. A negative scientific result can close a task.

## Current focus

- **Next:** M3MS-001, register the approved nominal-only ESN-search protocol.
- **Approved scope:** [M3MS plan](experiments/task_1a_manual_esn_search/plan.md), 100 M10 trials, nominal only, fixed v4/E filters and seed 896, searched ESN parameters/warm-up. Freeze three configurations before comparing S/M10/R10/C10/Replay. Preserve the declared shared resource ceiling.
- **Still gated:** task 1-b/M4-001 awaits D1–D8; its existing IN PROGRESS status is protocol drafting, not implementation permission. M3R-017 remains historically BLOCKED. Later hardware/online epics remain gated.
- **Latest closure:** DOC-009 reorganized navigation without changing prior task statuses or evidence; [record](tasks/archive/DOC.md). M3MAN-GATE accepted the [report](experiments/task_1a_manual_demonstration/report/report.md); the new search is a separate follow-up.

## Manual-data ESN search — M3MS

All rows below implement the [owner-approved plan](experiments/task_1a_manual_esn_search/plan.md).
They are registration only: no search or simulation has started. Longer review
rounds/results belong beside that plan, linked from the acceptance cell.

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M3MS-001 | `TODO` | Freeze the nominal-only search and final-comparison protocols | M3MAN-GATE | Versioned config binds approved ranges, five warm-ups including zero, seed 896, fixed trackers/filters, nominal success objective, three distinct scored selections with trial-number ties, all budgets and final scope. Perturbation access is excluded from selection. |
| M3MS-002 | `TODO` | Adapt manual training/evaluation to sampled configurations | M3MS-001 | Tested weighted S/M10/R10/C10 recipes, zero warm-up and fixed-policy invariance; no changes to closed manifest identities or protocols. |
| M3MS-003 | `TODO` | Implement resumable Optuna search and scope/budget guards | M3MS-002 | Parent and real workers enforce nominal-only evaluation; trials/resumes/failures and cumulative time/storage are accounted. Store evidence/provenance and retain partial work. No perturbed score can enter optimization or tie-breaking. |
| M3MS-004 | `TODO` | Measure a bounded timing pilot within the 100-trial cap | M3MS-003 | Explicit preflight and observed counts agree; pilot/anchor candidates consume the approved cap and can resume as part of the search. Check actual storage/elapsed cost before continuing. |
| M3MS-005 | `TODO` | Complete the approved search and freeze three configurations | M3MS-004 | At most 100 trials and 200 nominal RC runs; preserve failures and enforce shared resource limits. Freeze the three distinct scored configurations using only nominal success and trial order before perturbation evaluation. Report any shortfall without enlarging the cap. |
| M3MS-006 | `TODO` | Run the five-arm comparison on frozen configurations | M3MS-005 | Up to 93 learned models and 30 replay banks: at most 15,990 comparison runs before verified reuse, 16,190 including search. Same 65 cases/two trackers; full paired evidence, no post-evaluation reselection; shared 10 h/20 GiB ceiling. |
| M3MS-007 | `TODO` | Derive and audit machine-readable comparison evidence | M3MS-006 | Verify sources, selection, payloads, completeness and derived metrics/contrasts; retain failure records and reproducible figure inputs. Account actual replay/nominal reuse and explain withheld-from-search versus historically known perturbations. |
| M3MS-008 | `TODO` | Interpret and report the tuned comparison — reporting assistant | M3MS-007 | Human-readable paired S/M10/R10/C10/replay findings, failure examples, limitations and reproducible plots. Explain conditioning on M10-selected parameters; no claim of separate singleton tuning or fresh-scenario/multiple-seed validation. |
| M3MS-GATE | `TODO` | Review and close the follow-up — owner | M3MS-008 | Owner accepts positive/negative/inconclusive findings and records any next decision. No further study, deployment or hardware action is implied. |

## Future and deferred work

| Work | Canonical rows / gate |
| --- | --- |
| M4: broader planar tasks | [M4 backlog](tasks/backlog/M4.md); task 1-b protocol lock first |
| M5: C++ / 7-DOF simulation | [M5 backlog](tasks/backlog/M5.md) |
| M6: physical offline experiments | [M6 backlog](tasks/backlog/M6.md); separate safety admission |
| M7: online adaptation | [M7 backlog](tasks/backlog/M7.md); definition-only epic |
| REP-001: older cropped manual replication | [Deferred replication](tasks/backlog/replication.md); remains TODO |
| M3R-017: historical confirmatory authorization | [M3R archive](tasks/archive/M3R.md); remains BLOCKED |

## Cross-cutting upstream work

Use these IDs when a need is discovered before its owning milestone. Do not start
an upstream PR merely to make project-local code cleaner.

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| UP-001 | `TODO` | Track `rclib` serialization/import need | M2-006 | Local recipe unblocks Python; issue/PR scope is evidence-backed before M5 |
| UP-002 | `TODO` | Track `rclib` real-time allocation or latency findings | M5-006 | Reproducer and benchmark show generic library impact before PR work |
| UP-003 | `TODO` | Track missing generic `skelarm` extension seam | M2-009 | Existing controller/task/log registries are tested first; PR includes generic tests |
| UP-004 | `TODO` | Track missing generic `rtctrl` bridge/telemetry/safety seam | M5-005 | Existing `Arm`/`Controller` APIs are tested first; PR cannot weaken safety |
| UP-005 | `TODO` | Fix in-process seed reproducibility of `rclib` `RandomSparseReservoir` (power-iteration start vector uses `Eigen::Random()`/`std::rand`, never re-seeded; `cpp_core/src/reservoirs/RandomSparseReservoir.cpp:39` at pin `a015aca`) | M0-012 | Found 2026-08-29: fresh processes reproduce exactly, repeated construction in one process drifts; documented by strict-xfail `tests/integration/test_smoke_experiment.py::test_in_process_repeat_is_reproducible`. Upstream branch/PR with a regression test, then advance the pin and drop the xfail |

## Completed milestones and task lookup

- [DOC](tasks/archive/DOC.md): Documentation completed.
- [M0](tasks/archive/M0.md): M0 — Repository foundation.
- [M1](tasks/archive/M1.md): M1 — Demonstration pipeline and frozen baselines.
- [M2](tasks/archive/M2.md): M2 — Offline ESN task 1-a vertical slice.
- [M3](tasks/archive/M3.md): M3 — Tuning, robustness, and task 1-a reproduction.
- [TOOL](tasks/archive/TOOL.md): Cross-cutting result visualization.
- [M3R](tasks/archive/M3R.md): M3R — Task 1-a state-conditioned recovery.
- [M3REP](tasks/archive/M3REP.md): M3REP — Task 1-a repeated-demonstration control.
- [M3MAN](tasks/archive/M3MAN.md): M3MAN — Task 1-a manual demonstrations.
- [UP completed work](tasks/archive/UP.md).
- [Pre-migration status snapshot](tasks/archive/status-2026-09-22.md) and [historical phase gates](tasks/archive/roadmap-2026-09-22.md).

## Definition of done for every implementation task

Use the [shared definition of done](design/workflow.md#definition-of-done-for-every-implementation-task) and keep the task's canonical row, tests, documentation and evidence aligned in one commit.

## Milestone review checklist

Use the [review checklist](design/workflow.md#milestone-review-checklist). Favorable performance is not required; complete evidence and explicit owner gate decisions are.

<a id="milestone-gates"></a>
<a id="documentation-completed"></a>
<a id="m0--repository-foundation"></a>
<a id="m1--demonstration-pipeline-and-frozen-baselines"></a>
<a id="m11-data-contracts-and-preprocessing"></a>
<a id="m12-metrics-and-run-records"></a>
<a id="m13-direct-replay-baselines"></a>
<a id="m2--offline-esn-task-1-a-vertical-slice"></a>
<a id="m21-learning-contracts"></a>
<a id="m22-skelarm-closed-loop-integration"></a>
<a id="m3--tuning-robustness-and-task-1-a-reproduction"></a>
<a id="cross-cutting-result-visualization"></a>
<a id="m3r--task-1-a-state-conditioned-recovery"></a>
<a id="deferred-human-demonstration-replication"></a>
<a id="m3rep--task-1-a-repeated-demonstration-control"></a>
<a id="m3man--task-1-a-manual-demonstrations"></a>
<a id="m4--later-planar-experiments-gated-epics"></a>
<a id="m5--c-and-7-dof-simulation-gated-epic"></a>
<a id="m6--physical-crane-x7-offline-verification-gated-epic"></a>
<a id="m7--online-adaptation-definition-epic-only"></a>

Legacy section links land here; use the [canonical task index](tasks/README.md) to open that milestone.
