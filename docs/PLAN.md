<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Reservoir-computing robot-arm roadmap

**Updated:** 2026-09-22. **Queue:** [TASKS.md](TASKS.md).

Read this roadmap and the queue first. Then read only the relevant experiment
plan and specification sections. Details: [design/](design/README.md);
completed evidence and future epics are indexed in [tasks/](tasks/README.md).
Legacy section numbers/anchors below remain navigation points for code and
frozen-config references; those files were not rewritten during this move.

**Current approved follow-up:** [manual-data ESN search](experiments/task_1a_manual_esn_search/plan.md).
Optimize M10 on nominal scenarios only for 100 Optuna trials, searching ESN
parameters and warm-up with fixed tracker/filter settings and seed. Freeze the
three highest-scoring configurations before the five-arm perturbation comparison.
No implementation/search has started. M3MAN remains closed. The separate task 1-b draft still awaits its own decisions D1–D8.

## 1. Objective

Study RC target generators from planar simulation through separately qualified CRANE-X7 hardware and online adaptation. Negative findings can close a study.

[Section 1](design/architecture.md#1-objective).

## 2. Research questions and hypotheses

Separate data coverage, augmentation, regularization and robustness effects. Freeze hypotheses and selection rules before execution.

<a id="21-primary-questions"></a>
<a id="22-stage-hypotheses"></a>
[Section 2](design/architecture.md#2-research-questions-and-hypotheses).

## 3. Project boundaries and dependencies

The project owns protocols and adapters; `rclib` reservoirs/readouts, `skelarm` planar dynamics, and `rtctrl` the CRANE-X7 bridge/safety. Preserve GPL-3.0-only and dependency/artifact terms.

<a id="31-licensing"></a>
[Section 3](design/architecture.md#3-project-boundaries-and-dependencies).

## 4. System architecture

A target generator produces desired joint motion; a separate tracker commands the backend. Learning never bypasses backend safety boundaries.

[Section 4](design/architecture.md#4-system-architecture).

## 5. Initial ESN control formulation

See shared ESN signal, training, priming and derivative contracts. Each study uses its own frozen recipes; historical settings are not universal defaults.

<a id="51-signals"></a>
<a id="52-reservoir-and-readout"></a>
<a id="53-completed-m3-teacher-forcing-priming-and-dwell"></a>
<a id="54-approved-task-1-a-recovery-extension"></a>
<a id="approved-repeated-demonstration-follow-up"></a>
<a id="approved-manual-demonstration-follow-up"></a>
<a id="55-desired-derivatives-and-low-level-tracking"></a>
[Section 5](design/controller.md#5-initial-esn-control-formulation).

## 6. Fair baseline protocol

Compare methods under matched task, tracker and disturbance conditions; report tuning effort and data dependence. Replay and model-based baselines remain explicit.

[Section 6](design/controller.md#6-fair-baseline-protocol).

## 7. Data contracts

Git holds portable pointers and approved fixtures/figures. Payloads and study state use external storage; verify hashes and never silently fall back to the checkout.

<a id="71-storage-location-and-portability"></a>
<a id="72-artifact-records-and-raw-demonstrations"></a>
<a id="73-canonical-processed-dataset"></a>
<a id="74-run-record"></a>
<a id="75-result-inspection-and-visualization"></a>
[Section 7](design/data.md#7-data-contracts).

## 8. Public software interfaces

Typed generator, robot-state and desired-state contracts separate policy from tracking and the backend. Preserve validation, reset and abort behavior.

[Section 8](design/architecture.md#8-public-software-interfaces).

## 9. Metrics and evaluation

Use the experiment's frozen success, robustness and effort definitions. Retain failures, missing results and denominators; endpoint error alone is not success.

<a id="91-task-1-a-metrics"></a>
<a id="92-task-1-a-recovery-metrics"></a>
<a id="93-robustness-protocol"></a>
<a id="94-later-task-metrics"></a>
[Section 9](design/evaluation.md#9-metrics-and-evaluation).

## 10. Hyperparameter tuning

Freeze search space, objective, data access and budget before tuning. Keep evaluation information out of parameter selection. Record sampler/reservoir seeds separately.

[Section 10](design/evaluation.md#10-hyperparameter-tuning).

## 11. Experiment and data management

Record revisions, environment, resolved configuration, sources, seeds and raw metrics; regenerate tables and figures from verified evidence.

[Section 11](design/data.md#11-experiment-and-data-management).

## 12. Proposed repository layout

Use `src/` for logic, `scripts/` for entry points, `configs/` for protocols, `data/` for pointers, and `docs/experiments/` for plans/evidence. Design and task indexes are linked above.

[Section 12](design/architecture.md#12-proposed-repository-layout).

## 13. Phased implementation and gates

| Stage | Status / next gate | Details |
| --- | --- | --- |
| M0–M3 | Completed foundation, data, ESN and robustness | [Ledger index](tasks/README.md) |
| M3R | Closed with a negative recovery result; historical confirmatory task remains blocked | [Archive](tasks/archive/M3R.md) |
| M3REP | Closed repetition/regularization study | [Plan](experiments/task_1a_repeated_demonstration/plan.md) |
| M3MAN | Closed and report accepted on 2026-09-22 | [Report](experiments/task_1a_manual_demonstration/report/report.md) |
| M3MS | Approved manual-data ESN search; next implementation queue | [Plan](experiments/task_1a_manual_esn_search/plan.md) |
| M4 | Task 1-b protocol planning; owner lock required before implementation | [Backlog](tasks/backlog/M4.md) |
| M5 | C++ / 7-DOF simulation, gated | [Backlog](tasks/backlog/M5.md) |
| M6 | Supervised physical trials, separately safety-qualified | [Backlog](tasks/backlog/M6.md) |
| M7 | Online adaptation definition only, separately reviewed | [Backlog](tasks/backlog/M7.md) |

<a id="phase-0--foundation"></a>
<a id="phase-1--data-and-direct-replay-baselines"></a>
<a id="phase-2--task-1-a-rc-vertical-slice"></a>
<a id="phase-3--tuning-and-robustness"></a>
<a id="phase-3r--task-1-a-state-conditioned-recovery"></a>
<a id="phase-3rep--task-1-a-repeated-demonstration-control"></a>
<a id="phase-3man--task-1-a-manual-demonstrations"></a>
<a id="phase-4--broader-planar-tasks"></a>
<a id="phase-5--c-and-7-dof-simulation"></a>
<a id="phase-6--physical-crane-x7-offline-learning"></a>
<a id="phase-7--online-learning"></a>
[Section 13](tasks/archive/roadmap-2026-09-22.md#13-phased-implementation-and-gates).

## 14. Testing strategy

Develop test-first, use deterministic fixtures and run checks appropriate to the change. Controller/safety changes require simulation and failure-path tests before hardware.

<a id="unit-tests"></a>
<a id="property-and-regression-tests"></a>
<a id="integration-tests"></a>
<a id="manual-and-hardware-tests"></a>
[Section 14](design/workflow.md#14-testing-strategy).

## 15. Development and review workflow

Choose an authorized unblocked task, update its canonical row, implement/test, and record evidence in the same commit. Generic dependency fixes belong upstream on dedicated branches; pin changes are separate.

[Section 15](design/workflow.md#15-development-and-review-workflow).

## 16. Reproducibility requirements

Reproduction binds exact code, submodules, config, data hashes, seeds and execution environment. Fail clearly on missing or mismatched inputs rather than choosing the latest artifact.

[Section 16](design/workflow.md#16-reproducibility-requirements).

## 17. Safety principles

Never bypass limits, watchdogs, command validation or aborts. Hardware needs its own approved procedure, operator and independent power cutoff; online learning cannot alter the safety envelope.

[Section 17](design/workflow.md#17-safety-principles).

## 18. Assumptions and deferred decisions

Later planar, C++, 7-DOF, physical and online-learning work remains gated. An approved simulation study does not authorize hardware or an unspecified follow-up.

[Section 18](design/workflow.md#18-assumptions-and-deferred-decisions).
