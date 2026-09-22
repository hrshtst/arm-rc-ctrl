<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Workflow reference

[Roadmap](../PLAN.md) · [Work queue](../TASKS.md)

Detailed sections moved from the roadmap on 2026-09-22. Original section
numbers and historical implementation notes are retained so source-code
and configuration references remain interpretable. Current priorities and
gate status are in the roadmap/queue; experiment-specific decisions live
in each linked experiment plan. Read only the sections relevant to the task.

## 14. Testing strategy

Development follows TDD. Tests are added before or with the behavior they cover.

### Unit tests

- dataset schemas, units, shapes, interval detection, and invalid inputs;
- smoothing/resampling and derivative estimation on analytic signals;
- normalization and inverse transformation;
- ESN input/target alignment, reset, washout, and episode isolation;
- metric definitions, angle handling, and failure penalties;
- config defaults, rejection of unknown keys, and path resolution;
- provenance collection and clean/dirty-worktree policy.

### Property and regression tests

- resampling preserves constant/linear signals within tolerance;
- metrics are zero for identical signals and nonnegative otherwise;
- fixed seeds reproduce the same reservoir recipe and predictions within the
  declared platform tolerance;
- a tiny committed fixture protects sample alignment and Python/C++ parity;
- no test relies on a GUI, network service, or robot by default.

### Integration tests

- raw `skelarm` fixture to processed dataset;
- processed demonstration to direct-replay baseline run;
- teacher-forced ESN fit to closed-loop `skelarm` run;
- MLflow run contains mandatory provenance and artifacts;
- a small Optuna study resumes and selects a valid trial;
- DVC reproduction rebuilds expected outputs;
- later, C++ controller through `rtctrl::arm::SimArm` and the emulator.

### Manual and hardware tests

Manual reproduction scripts generate key tables and plots. Hardware tests are
never CI jobs. They require a human operator, staged duration/limits, an
independent power cutoff, and explicit recording of deviations from the approved
procedure.

## 15. Development and review workflow

1. Select the next unblocked task from [TASKS.md](../TASKS.md) and mark it
   `IN PROGRESS` before implementation.
2. Add or update a failing test/specification.
3. Implement the smallest coherent behavior that passes it.
4. Run focused tests, then the repository quality gate.
5. Update human documentation after behavior stabilizes. If documentation and
   tested implementation disagree, immediately align documentation to the tested
   behavior or fix the implementation and tests when the behavior is wrong.
6. Update task status and evidence in the same commit.
7. Commit one task or a small cohesive group. Include task IDs in the commit
   message/body. Do not combine formatting, refactoring, and behavior changes
   unless inseparable.
8. At each phase gate, request review with the exact commands, configs, artifacts,
   known limitations, and unresolved research questions.

For upstream work:

Create a dedicated branch in the owning upstream repository before making
implementation changes, so those changes can be submitted as a PR later.
Record the branch name and base revision with the task evidence; document
dependencies when stacking branches for separate PRs.

1. Reproduce the missing generic capability in the owning library.
2. Create a dedicated branch in that library.
3. Add library-level tests and documentation.
4. Open a focused PR that discloses AI assistance when required by that project.
5. Keep this project compatible with the pinned revision until the PR is ready.
6. Advance the submodule in a separate integration commit and rerun this
   project's full relevant test suite.

## 16. Reproducibility requirements

A key result is reproducible only when another human can obtain it from:

- the project Git commit and clean/dirty state;
- exact `rclib`, `skelarm`, and `rtctrl` commits;
- `uv.lock`, compiler/CMake information, and platform metadata;
- resolved experiment config;
- logical artifact URIs, artifact-record revisions, payload SHA-256 digests, and
  DVC hashes where applicable;
- all random seeds and study/trial identifiers;
- one documented command or reproduction script;
- raw metrics in machine-readable form, not only a plot.

The reproduction script must fail clearly when storage configuration, required
payloads, data records, submodules, or versions are missing or mismatched. It
must not fall back to the repository, silently download mutable data, accept a
checksum mismatch, or select the latest model.

## 17. Safety principles

- Learning code never bypasses backend position, velocity, effort, current, or
  watchdog limits.
- Validate output shape, finiteness, timestamp freshness, and bounds before every
  command.
- On invalid ESN output, stale state, missed deadline, or internal exception,
  invoke the backend's documented safe abort/deactivation path; do not continue
  with newly generated commands.
- Simulation and wire/emulator tests precede hardware for every controller or
  safety-relevant change.
- `rtctrl` remains the authority for physical activation, watchdogs, command
  windows, and motor communication.
- Hardware operation is supervised and retains an independent actuator-power
  cutoff. Software deactivation is not treated as an emergency stop.
- Online learning never controls the safety envelope and can be frozen or
  bypassed without disabling the low-level safety controller.

## 18. Assumptions and deferred decisions

- Python 3.12+, `uv`, NumPy `float64`, TOML, pytest, Ruff, and a strict type
  checker form the initial Python stack.
- C++17, CMake, and Catch2 align with the current C++ dependencies.
- Large data and experiment state live below a per-machine external storage root,
  defaulting to `/external/arm-rc-ctrl`; Git stores portable records and DVC
  metafiles only. No cloud account is required.
- The initial task uses a horizontal, gravity-free `skelarm` model and controls
  arm joints only; the CRANE-X7 gripper is excluded until a task requires it.
- Original project code and documentation are GPL-3.0-only. Third-party and
  data/artifact terms remain separately applicable and must be inventoried.
- Exact online-learning tasks, weight bounds, rollback policy, and hardware
  admission criteria remain deferred until offline results exist. Before Phase 7
  starts, replace that epic with a separately reviewed, decision-complete plan.

## Definition of done for every implementation task

- The task status and evidence are updated in its canonical ledger location, indexed by `docs/TASKS.md`.
- Relevant tests were added first or alongside the behavior and pass.
- Focused checks and the repository quality gate pass.
- Public behavior is typed, validated, and documented.
- No warnings, ignored failures, unexplained numerical tolerances, or silent
  fallback paths were introduced.
- Generated scientific results include complete provenance and raw metrics.
- Documentation matches tested implementation behavior.
- The commit is reviewable and does not include unrelated refactoring or output.
- Safety-relevant changes include failure-path tests and never bypass library
  safety boundaries.

## Milestone review checklist

- [ ] All required tasks and the milestone gate are `DONE`.
- [ ] Clean recursive checkout and setup were tested.
- [ ] Exact commands and expected outputs are documented.
- [ ] Unit, integration, regression, lint, type, and build checks pass.
- [ ] Configs, seeds, code, submodules, data, and environment are pinned.
- [ ] Baselines use matched conditions and tuning effort is reported.
- [ ] Failures and excluded runs remain visible in machine-readable results.
- [ ] Plots can be regenerated from stored raw metrics.
- [ ] Human documentation agrees with code and tests.
- [ ] Known limitations and next research decisions are explicit.
- [ ] Hardware work, if any, has separate safety approval and operator records.

## Keeping documentation small

Read `docs/PLAN.md` and `docs/TASKS.md` first, then only the selected task's
plan, linked specification sections and relevant evidence. Do not load all
archives into routine context. Keep the roadmap below 8 KiB and the queue
below 15 KiB. Keep an active row to a few sentences; link longer review
rounds and results beside the experiment. On completion move its full row
to the appropriate milestone archive, preserving ID/status/dependencies and
evidence, and replace it in the queue with a short closure link. Future gated
epics live in `docs/tasks/backlog/`; each task has one canonical row. If a
closed task is reopened, move its row back to the queue and leave a link in
the archive rather than creating two status records.
