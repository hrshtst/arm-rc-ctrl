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

- **Next:** M3MS-004, the bounded timing pilot inside the 100-trial cap. M3MS-001 to M3MS-003 froze the protocol (`configs/studies/manual_esn_search_v1.toml`, digest `029739812525`), adapted training to sampled configurations and built the resumable search with its scope and budget guards; no search of the real bank has run.
- **Approved scope:** [M3MS plan](experiments/task_1a_manual_esn_search/plan.md), 100 M10 trials, nominal only, fixed v4/E filters and seed 896, searched ESN parameters/warm-up. Freeze three configurations before comparing S/M10/R10/C10/Replay, then evaluate them sequentially in selection order. Preserve the declared shared resource ceiling; a ceiling-stopped third configuration is reported as incomplete rather than dropped or traded for a larger allowance.
- **Still gated:** task 1-b/M4-001 awaits D1–D8; its existing IN PROGRESS status is protocol drafting, not implementation permission. M3R-017 remains historically BLOCKED. Later hardware/online epics remain gated.
- **Latest closure:** DOC-009 reorganized navigation without changing prior task statuses or evidence; [record](tasks/archive/DOC.md). M3MAN-GATE accepted the [report](experiments/task_1a_manual_demonstration/report/report.md); the new search is a separate follow-up.

## Manual-data ESN search — M3MS

All rows below implement the [owner-approved plan](experiments/task_1a_manual_esn_search/plan.md).
They are registration only: no search or simulation has started. Longer review
rounds/results belong beside that plan, linked from the acceptance cell.

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M3MS-001 | `DONE` | Freeze the nominal-only search and final-comparison protocols | M3MAN-GATE | Versioned config binds approved ranges, five warm-ups including zero, seed 896, fixed trackers/filters, nominal success objective, three distinct scored selections with trial-number ties, all budgets and final scope. Perturbation access is excluded from selection. Done 2026-09-22, re-closed 2026-09-23 after two owner review rounds (six P2 findings, each fixed with a reproducing test written first): [review rounds](experiments/task_1a_manual_esn_search/review_rounds.md). `configs/studies/manual_esn_search_v1.toml` (digest `029739812525`, sampler seed 20270301 in a namespace checked disjoint from every seed under `configs/`) is loaded and validated by `arm_rc_ctrl.experiments.manual_search`, whose single enforcement point refuses any protocol outside the approved scope while leaving the caps free to tighten: [implementation notes](experiments/task_1a_manual_esn_search/implementation.md). Tests first: `tests/unit/test_manual_search_protocol.py` (58) and the lock `tests/regression/test_manual_search_config.py` (5), which reads the approved ranges, fixed conditions and caps out of the plan's own tables so the protocol and the plan cannot drift. No search, fit or simulation has run. Gate green at 3,325 passed and 90.84 % coverage, pinned to the P-cores. |
| M3MS-002 | `DONE` | Adapt manual training/evaluation to sampled configurations | M3MS-001 | Tested weighted S/M10/R10/C10 recipes, zero warm-up and fixed-policy invariance; no changes to closed manifest identities or protocols. Done 2026-09-23, re-closed the same day after an owner review round (two P2 findings: a truncated reservoir size and an unchecked direct construction): [review rounds](experiments/task_1a_manual_esn_search/review_rounds.md). `arm_rc_ctrl.experiments.manual_sampled` expresses a sampled configuration as the `StudyConfiguration` the frozen study already uses, so the existing recipes and evidence paths take it unchanged and there is no second training path: [implementation notes](experiments/task_1a_manual_esn_search/implementation.md). Tests first (`tests/unit/test_manual_sampled.py`, 52 cases): every approved warm-up including zero reaches each arm's training spec; the four arms keep the closed experiment's weighting, multiplicities and count-scaled ridge at the sampled `alpha_0`; all four recipes build over the fixture bank; a point outside the approved space is refused wherever it enters; and the closed study's labels, manifest bytes and identities are untouched. Gate green at 3,370 passed and 90.80 % coverage, pinned to the P-cores. |
| M3MS-003 | `DONE` | Implement resumable Optuna search and scope/budget guards | M3MS-002 | Parent and real workers enforce nominal-only evaluation; trials/resumes/failures and cumulative time/storage are accounted. Store evidence/provenance and retain partial work. No perturbed score can enter optimization or tie-breaking. Numerical work runs in process-isolated workers that inherit the whole canonical affinity set with one thread each, never in per-worker CPU subsets; trials run serially to begin with, and reproducibility is tested through real subprocesses rather than in-process threads. Done 2026-09-23 after two owner review rounds (ten findings, six of them P1), each fixed with a reproduction: [review rounds](experiments/task_1a_manual_esn_search/review_rounds.md). `arm_rc_ctrl.experiments.manual_search_run` is the parent and the worker. A trial is reserved -- with the fit and evidence identities its work will be stored under -- before its worker process starts; a reconciliation pass repairs both write boundaries before anything is scheduled; pending work is recovered under its original number; every attempt's cost is charged; retained work is discovered from the reservation's identities even without completed evidence; a non-positive allowance stops scheduling and no worker starts without a real deadline; only a numerical refusal is a failed candidate, an infrastructure error stays recoverable; and a report is scored only after the parent rebuilds the expected entry, requires the reserved identities and loads every run through the reader a resume uses. Sampled configurations resolve through `ManualFitInputs`: [implementation notes](experiments/task_1a_manual_esn_search/implementation.md). Tests: `tests/unit/test_manual_search_run.py` (38), including the scope refusal at the real worker entry point, the recovery and reconciliation paths, and a sampled fit reproducing bitwise in a fresh interpreter. No search of the real bank has run. Gate green at 3,415 passed and 90.69 % coverage, pinned to the P-cores. |
| M3MS-004 | `TODO` | Measure a bounded timing pilot within the 100-trial cap | M3MS-003 | Explicit preflight and observed counts agree; pilot/anchor candidates consume the approved cap and can resume as part of the search. Measure fitting and verification overhead as well as simulation, since those are what the estimates omit, and check actual storage/elapsed cost before continuing. |
| M3MS-005 | `TODO` | Complete the approved search and freeze three configurations | M3MS-004 | At most 100 trials and 200 nominal RC runs; preserve failures and enforce shared resource limits. Freeze the three distinct scored configurations using only nominal success and trial order before perturbation evaluation. Report any shortfall without enlarging the cap. |
| M3MS-006 | `TODO` | Run the five-arm comparison on frozen configurations | M3MS-005 | Up to 93 learned models and 30 replay banks: at most 15,990 comparison runs before verified reuse, 16,190 including search. Same 65 cases/two trackers; full paired evidence, no post-evaluation reselection; shared 10 h/20 GiB ceiling. Complete one configuration's whole comparison before starting the next, in selection order; a ceiling-stopped third is retained as partial evidence and reported as two complete comparisons plus an incomplete third, which is not the approved three-configuration study. |
| M3MS-007 | `TODO` | Derive and audit machine-readable comparison evidence | M3MS-006 | Verify sources, selection, payloads, completeness and derived metrics/contrasts; retain failure records and reproducible figure inputs. Account actual replay/nominal reuse and explain withheld-from-search versus historically known perturbations. |
| M3MS-008 | `TODO` | Interpret and report the tuned comparison — reporting assistant | M3MS-007 | Human-readable paired S/M10/R10/C10/replay findings, failure examples, limitations and reproducible plots. Explain conditioning on M10-selected parameters; no claim of separate singleton tuning or fresh-scenario/multiple-seed validation. Call the frozen configurations the highest nominal scores, never "the three best", and keep nominal feasibility separate from robustness and trajectory quality. |
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
