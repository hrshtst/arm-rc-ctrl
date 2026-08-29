# Implementation Task Ledger

**Last updated:** 2026-08-29

**Plan:** [PLAN.md](PLAN.md)

This file is the source of truth for implementation status. Update it whenever a
task starts, finishes, becomes blocked, or changes scope. Implementation work
must reference one or more stable task IDs.

## Status rules

| Status | Meaning |
| --- | --- |
| `TODO` | Defined but not started |
| `IN PROGRESS` | Actively being implemented; keep this set small |
| `BLOCKED` | Cannot proceed; record the blocker and required decision/evidence |
| `DONE` | Acceptance criterion passed and evidence is recorded |

A task is `DONE` only when its tests, documentation, and reproducibility evidence
are complete. Do not mark research software complete merely because it runs once.

## Current focus

- **Next task:** `M0-005` — add the CMake skeleton under `cpp/` with Catch2.
- **Current milestone:** M0 — repository foundation.
- **Active blockers:** None.
- **Latest completed planning work:** `DOC-001` and `DOC-002`.

## Milestone gates

| Milestone | Gate | Status |
| --- | --- | --- |
| M0 | Clean recursive checkout installs and passes documented Python/C++ quality commands | `IN PROGRESS` |
| M1 | Demonstration preprocessing and both direct-replay baselines are reproducible | `TODO` |
| M2 | Task 1-a ESN completes a provenance-complete nominal closed-loop run | `TODO` |
| M3 | Frozen task 1-a model and confirmatory robustness report reproduce with one command | `TODO` |
| M4 | Planar tasks 1-b through 3 and 4-DOF scaling have frozen protocols and reports | `TODO` |
| M5 | Python/C++ parity and `rtctrl` 7-DOF simulation timing/safety acceptance pass | `TODO` |
| M6 | Offline hardware trial receives safety approval and completes with full telemetry | `TODO` |
| M7 | Online-learning task and safety plan are separately approved | `TODO` |

## Documentation completed

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| DOC-001 | `DONE` | Define the research and implementation roadmap | — | `docs/PLAN.md` specifies architecture, equations, protocols, gates, safety, and reproducibility |
| DOC-002 | `DONE` | Initialize the implementation task ledger | DOC-001 | `docs/TASKS.md` contains stable IDs, dependencies, statuses, and acceptance criteria |

## M0 — Repository foundation

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M0-001 | `DONE` | Add `pyproject.toml`, `src/arm_rc_ctrl`, Python 3.12 requirement, and `uv` dependency groups | DOC-002 | `uv sync` installs a clean environment from `pyproject.toml`/`uv.lock` (Python 3.12 via `.python-version`); `tests/unit/test_package.py` imports the package and checks the installed version |
| M0-002 | `DONE` | Add pinned submodules at `third_party/rclib` (recursive), `third_party/skelarm`, and `third_party/rtctrl` (top-level until M5) | M0-001 | HTTPS submodules pinned at rclib `a015aca`, skelarm `ebb2611`, rtctrl `c601076` (table in `THIRD_PARTY_NOTICES.md`); bootstrap is `git submodule update --init third_party/skelarm third_party/rtctrl && git submodule update --init --recursive third_party/rclib` (rtctrl nested submodules deferred to M5); `tests/unit/test_submodule_pins.py` checks URLs, recorded gitlinks, checked-out HEADs, the notice table, and rclib nested submodules; fresh-clone bootstrap verified (see commit) |
| M0-003 | `DONE` | Wire local/pinned Python builds of `rclib` and `skelarm` into the `uv` environment | M0-002 | `[tool.uv.sources]` builds non-editable `rclib`/`skelarm` from the submodules (`uv sync`); `src/arm_rc_ctrl/dependencies.py` reports recorded/checked-out submodule revisions, installed versions, and stale installs; `tests/unit/test_dependency_wiring.py` imports both libraries headless, checks versions/sources against the pins, and records revisions |
| M0-004 | `DONE` | Add Ruff, strict type checking, pytest, coverage, and pre-commit configuration | M0-001 | Ruff (upstream policy), basedpyright `strict`, pytest (`filterwarnings=error`, strict markers/config), branch coverage, pre-commit, and `noxfile.py` (`lint`, `type_check`, `tests`, `pre_commit`) configured; `tests/unit/test_quality_tools.py` proves each tool reports the planted problems in `tests/fixtures/quality/`; `uv run nox` and `uv run nox -s pre_commit` pass on the clean tree |
| M0-005 | `TODO` | Add CMake skeleton under `cpp/` with Catch2 and an empty library/application vertical slice | M0-002 | Configure, build, and CTest smoke test pass without a robot |
| M0-006 | `TODO` | Add CI for Python lint/type/test and C++ configure/build/test | M0-003, M0-004, M0-005 | CI starts from recursive checkout and all jobs pass |
| M0-007 | `DONE` | Add `.gitignore` and external data/artifact conventions | M0-001 | `.gitignore` excludes payload formats, external-layout directories, DVC cache/local config, `storage.toml`, builds, and tool caches while `data/records/**/*.toml`, DVC metafiles, `configs/*.example.toml`, and `tests/fixtures/` stay trackable; `data/README.md` states the conventions; `tests/unit/test_gitignore_conventions.py` verifies 43 representative paths with `git check-ignore` |
| M0-008 | `TODO` | Add README setup, quality, and recursive-checkout commands | M0-003, M0-005 | A clean-room walkthrough uses only documented commands |
| M0-009 | `DONE` | Add GPL-3.0-only licensing and initial third-party notice inventory | — | Root `LICENSE`, plan policy, SPDX guidance, and `THIRD_PARTY_NOTICES.md` are committed |
| M0-010 | `TODO` | Add typed TOML loader with strict unknown-key rejection and relative path resolution | M0-001, M0-004 | Unit tests cover valid, missing, unknown, and type-invalid fields plus nested config paths |
| M0-011 | `TODO` | Define provenance record and collection utilities | M0-002, M0-010 | Tests capture project/submodule commits, dirty flag, lock/config hashes, logical artifact URIs/digests, platform, and seeds |
| M0-012 | `TODO` | Add headless deterministic smoke experiment independent of GUI/hardware | M0-003, M0-010, M0-011 | Two same-seed executions produce equal canonical outputs within declared tolerance |
| M0-013 | `TODO` | Add owner-approved citation and publication metadata | M0-001 | Citation file and publication metadata identify authorship, preferred citation, and release policy |
| M0-014 | `TODO` | Implement machine-local external storage-root resolution | M0-010 | Tests enforce environment/XDG/default precedence, `armrc://` resolution, access checks, and no repository fallback |
| M0-GATE | `TODO` | Review and close the M0 gate | M0-006, M0-007, M0-008, M0-009, M0-012, M0-013, M0-014 | Reviewer reproduces install and all quality commands from a clean recursive checkout |

## M1 — Demonstration pipeline and frozen baselines

### M1.1 Data contracts and preprocessing

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M1-001 | `TODO` | Define versioned raw-demonstration artifact records | M0-010, M0-014 | Round-trip tests cover immutable ID, logical URI, SHA-256/size, format/schema, license/access, scenario, intervals, revisions, and notes |
| M1-002 | `TODO` | Define canonical `samples.npz` arrays and processed artifact records | M1-001 | Schema tests enforce source IDs, logical URI, digest, names, shapes, float64 dtype, units, and phases |
| M1-003 | `TODO` | Implement raw `skelarm` log loader through the storage resolver | M0-003, M1-001 | Known fixture loads; missing, inaccessible, mismatched, corrupt, or unexpected data fails without source modification |
| M1-004 | `TODO` | Implement dataset validation | M1-002 | Tests reject NaN/Inf, time errors, shape errors, missing phases, invalid task codes, and limit violations |
| M1-005 | `TODO` | Implement configurable smoothing using zero-phase processing for offline demonstrations | M1-003, M1-004 | Analytic noisy-signal tests verify attenuation and absence of measurable phase shift |
| M1-006 | `TODO` | Implement resampling to the configured control period | M1-005 | Constant/linear analytic signals and endpoint inclusion pass within numerical tolerance |
| M1-007 | `TODO` | Implement offline derivative generation for `dq`, `ddq`, `dtip`, and `ddtip` | M1-006 | Polynomial/sinusoidal fixtures meet declared interior/boundary error tolerances |
| M1-008 | `TODO` | Implement or validate explicit prime/move/dwell interval annotation | M1-003 | Missing/overlapping/reversed intervals fail; all samples receive exactly one phase |
| M1-009 | `TODO` | Implement training-only normalization statistics | M1-004 | Tests prevent evaluation leakage and define near-zero scale handling |
| M1-010 | `TODO` | Add transactional preprocessing CLI as a thin wrapper around tested functions | M1-006, M1-007, M1-008, M1-009 | Command writes/validates external payload atomically, then creates its record; overwrite and repository fallback are rejected |
| M1-011 | `TODO` | Initialize DVC with external per-machine cache and remote | M0-007, M0-014, M1-010 | Ignored local config maps cache/remote below storage root; Git contains only portable DVC metadata and artifact records |
| M1-012 | `TODO` | Create or select the canonical 2-DOF robot/task 1-a scenario | M0-003 | Versioned TOML fixes robot parameters, target, `dt`, limits, duration, prime, and dwell intervals |
| M1-013 | `TODO` | Record/import one external task 1-a demonstration and commit its immutable record | M1-010, M1-012 | Payload is absent from Git, record resolves and validates, and visual/manual review confirms intended reach/dwell |
| M1-014 | `TODO` | Add preprocessing integration/regression fixture | M1-013 | Clean reproduction through a configured external store yields expected records, arrays, and digests |

### M1.2 Metrics and run records

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M1-015 | `TODO` | Define typed termination reasons and success/failure records | M0-010 | Unit tests cover normal end, invalid state/output, limits, divergence, timeout, and backend failure |
| M1-016 | `TODO` | Implement joint RMSE with per-joint continuous-angle policy | M1-002 | Hand-calculated fixtures test aggregate, per-joint, wrapping, and shape rejection |
| M1-017 | `TODO` | Implement final-dwell endpoint and stationarity metrics | M1-002 | Fixtures verify mean/RMS/max/p95 error, in-region fraction, dwell duration, and velocity metrics |
| M1-018 | `TODO` | Implement effort, peak torque, and saturation metrics | M1-002 | Irregular-time analytic fixtures verify integration and saturation fraction |
| M1-019 | `TODO` | Define provenance-complete external run-record schema and Git pointer record | M0-011, M0-014, M1-015 | Round-trip retains state, references, torque, disturbances, termination, config, provenance, logical URI, and digest |
| M1-020 | `TODO` | Add metric report generation from run records | M1-016, M1-017, M1-018, M1-019 | JSON/CSV summaries match pure metric functions and contain no hidden recomputation |

### M1.3 Direct-replay baselines

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M1-021 | `TODO` | Build a sampled joint reference from the canonical demonstration | M1-014 | Samples at grid/between-grid/boundaries match the processed data policy |
| M1-022 | `TODO` | Implement headless direct-replay PD experiment through `skelarm` | M1-019, M1-021 | Run terminates normally, logs required channels, and respects configured limits |
| M1-023 | `TODO` | Implement headless direct-replay computed-torque experiment | M1-019, M1-021 | Same acceptance as PD and dynamics feedforward is visible in telemetry |
| M1-024 | `TODO` | Define an equal-budget baseline-gain tuning protocol | M1-022, M1-023 | Versioned config fixes search spaces, objective, sampler seed, budget, limits, and development scenarios |
| M1-025 | `TODO` | Tune, review, and freeze PD gains | M1-024 | Selected config, study summary, metrics, and provenance are committed/logged |
| M1-026 | `TODO` | Tune, review, and freeze computed-torque gains | M1-024 | Selected config, study summary, metrics, and provenance are committed/logged |
| M1-027 | `TODO` | Add baseline determinism and paired-run regression tests | M1-025, M1-026 | Fixed fixtures reproduce state/metric results within declared tolerances |
| M1-028 | `TODO` | Calibrate safe/nontrivial posture and force perturbation levels using frozen baselines | M1-025, M1-026 | Pilot report justifies levels; confirmatory config and seed list are then locked |
| M1-GATE | `TODO` | Review and close the M1 gate | M1-011, M1-014, M1-020, M1-027, M1-028 | Reviewer resolves external records and reproduces preprocessing and both frozen baselines without payloads in Git |

## M2 — Offline ESN task 1-a vertical slice

### M2.1 Learning contracts

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M2-001 | `TODO` | Define `RobotState`, `DesiredJointState`, and `TargetGenerator` contracts | M0-010 | Type/unit/shape/finiteness invariants and reset-before-step behavior have tests |
| M2-002 | `TODO` | Build teacher-forcing input/target pairs `[q_k,dq_k] -> q_(k+1)` | M1-009, M2-001 | Tiny fixture proves exact one-step alignment and excludes washout rows from loss |
| M2-003 | `TODO` | Implement per-episode reservoir reset and priming | M2-002 | Tests prove episodes do not leak state and runtime priming matches training semantics |
| M2-004 | `TODO` | Implement ESN factory from typed config using `rclib` | M0-003, M2-002 | Fixed hyperparameters/seeds construct a model with expected input/output dimensions |
| M2-005 | `TODO` | Implement offline ridge training and validation prediction | M2-003, M2-004 | Synthetic learnable sequence improves over a constant predictor and is deterministic |
| M2-006 | `TODO` | Define deterministic model-recipe schema and reconstruction | M1-011, M2-004 | Recipe round-trip reconstructs/refits predictions within declared tolerance; no pickle is used |
| M2-007 | `TODO` | Implement `RcTargetGenerator.reset/step` using actual feedback | M2-005, M2-006 | Tests prove feedback, not previous prediction, forms the next input and invalid output is rejected |
| M2-008 | `TODO` | Implement causal desired derivative estimator | M2-001 | Analytic tests cover first sample, reset, filtering, irregular/nonpositive/excessive `dt`, and telemetry |

### M2.2 `skelarm` closed-loop integration

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M2-009 | `TODO` | Implement target-generator plus PD `skelarm.Controller` adapter | M1-025, M2-007, M2-008 | Adapter registers without patching `skelarm`, returns finite torque, and exposes all internal channels |
| M2-010 | `TODO` | Implement explicit initial hold/priming state machine | M2-003, M2-009 | Controller holds initial posture during priming and begins generation at the configured boundary without a command jump |
| M2-011 | `TODO` | Implement task 1-a training CLI | M2-006, M1-014 | Command validates inputs and emits recipe, training metrics, and provenance |
| M2-012 | `TODO` | Implement nominal RC+PD experiment runner | M2-009, M2-010, M2-011, M1-019 | End-to-end run records measured/generated states, derivatives, torque, ESN state summary, and termination |
| M2-013 | `TODO` | Add safety validation around generated commands | M2-012 | NaN, shape error, bound violation, stale time, and model exception cause structured safe termination |
| M2-014 | `TODO` | Add task 1-a nominal metric/report integration | M1-020, M2-012 | Paired RC+PD/replay+PD report uses identical interval and metric definitions |
| M2-015 | `TODO` | Add end-to-end deterministic regression fixture | M2-013, M2-014 | Raw fixture through training and closed-loop evaluation reproduces within tolerance |
| M2-016 | `TODO` | Implement target-generator plus computed-torque adapter | M1-026, M2-007, M2-008 | Adapter uses the same generated reference/derivative contract and respects frozen gains |
| M2-017 | `TODO` | Add paired RC+computed-torque/replay+computed-torque evaluation | M2-016, M2-014 | Report separates generator effects from tracker effects and logs identical metrics/provenance |
| M2-GATE | `TODO` | Review and close the M2 gate | M2-015, M2-017 | Nominal runs complete safely with full provenance; scientific performance is reported without a superiority requirement |

## M3 — Tuning, robustness, and task 1-a reproduction

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M3-001 | `TODO` | Integrate an external-root MLflow backend and mandatory run logger | M0-011, M0-014, M1-019 | Integration test verifies external storage plus config, revisions, hashes, seeds, metrics, recipe, and plots |
| M3-002 | `TODO` | Integrate external-root Optuna SQLite studies with seeded sampler/pruner | M0-014, M2-012 | Small study creates, resumes, prunes, and selects a trial deterministically where supported |
| M3-003 | `TODO` | Define versioned ESN search space and development-only scenarios | M1-028, M3-002 | Config includes all planned parameters, bounds, study seed, budget, and no confirmatory seeds |
| M3-004 | `TODO` | Implement feasibility checks and documented trial penalty | M2-013, M3-003 | Tests cover divergence, state/torque limits, early termination, dwell failure, and feasible objective |
| M3-005 | `TODO` | Implement parent study and child trial MLflow logging | M3-001, M3-002, M3-004 | Every trial is traceable and objective components are stored individually |
| M3-006 | `TODO` | Run ESN study and freeze the selected model recipe | M3-003, M3-005 | Selection report records budget, failures, chosen trial, recipe, and development metrics |
| M3-007 | `TODO` | Implement deterministic initial-posture perturbation generator | M1-028 | Fixtures prove seed, bounds, grid/random mode, and exact paired scenarios |
| M3-008 | `TODO` | Implement deterministic finite-duration endpoint force disturbances | M1-028 | Tests verify timing, direction, magnitude, replay, and run-record channels |
| M3-009 | `TODO` | Implement paired robustness suite across methods | M3-006, M3-007, M3-008 | Each method receives identical scenario IDs and disturbances; failures remain in aggregation |
| M3-010 | `TODO` | Run locked confirmatory suite exactly once for the study version | M3-009 | Report identifies frozen recipe/config/seeds and is labeled confirmatory; reruns are separately labeled |
| M3-011 | `TODO` | Generate task 1-a tables, plots, and limitations report | M3-010 | Report includes all primary/secondary metrics, distributions, failures, and paired comparisons |
| M3-012 | `TODO` | Implement `scripts/reproduce_1a.py` orchestration | M1-011, M3-011 | One command resolves external records and rebuilds data, model, evaluation, and report or identifies missing/mismatched payloads |
| M3-013 | `TODO` | Perform clean-checkout reproduction audit | M3-012 | Human records commands, environment, elapsed time, hashes, and any numerical deviation |
| M3-014 | `TODO` | Review code, experimental fairness, and documentation alignment | M3-013 | Findings are fixed or tracked with explicit IDs; plan reflects tested behavior |
| M3-GATE | `TODO` | Review and close the M3 gate | M3-014 | Independent reviewer reproduces the selected task 1-a result and accepts provenance/fairness |

## M4 — Later planar experiments (gated epics)

Do not implement these epics directly. After M3, refine the next epic into
review-sized tasks with fixed hypotheses, data splits, baselines, metrics,
perturbations, and acceptance criteria.

| ID | Status | Epic | Depends on | Exit criterion |
| --- | --- | --- | --- | --- |
| M4-001 | `TODO` | Task 1-b: several demonstrations, one target, unseen initial postures | M3-GATE | Reproducible success-region report versus model-based reaching/replay baselines |
| M4-002 | `TODO` | Multi-target reaching with one-hot target conditioning and live switching | M4-001 | Per-target/switch success and smoothness report versus matched baseline |
| M4-003 | `TODO` | Periodic circle and lemniscate drawing | M4-002 | Convergence, geometric tracking, phase drift, and perturbation-recovery report |
| M4-004 | `TODO` | Repeat qualified experiments on a 4-DOF planar arm | M4-001 | Scaling report isolates DOF/reservoir/data effects with equivalent protocols |
| M4-005 | `TODO` | Decide whether augmentation or new RC formulation is required | M4-004 | Evidence-backed decision and amended plan/tasks; failed approaches retained in reports |
| M4-GATE | `TODO` | Review and close the planar-research gate | M4-003, M4-004, M4-005 | At least one equilibrium and one periodic experiment have frozen reproducible reports |

## M5 — C++ and 7-DOF simulation (gated epic)

| ID | Status | Epic | Depends on | Exit criterion |
| --- | --- | --- | --- | --- |
| M5-001 | `TODO` | Audit `rclib` fitted-model serialization and real-time inference APIs | M3-GATE | Gap report identifies exact existing/missing APIs and ownership |
| M5-002 | `TODO` | Add versioned fitted-model serialization in a dedicated `rclib` branch/PR if needed | M5-001 | Python/C++ round-trip tests pass in `rclib`; PR is reviewable and referenced |
| M5-003 | `TODO` | Advance the `rclib` submodule and add project-level model loading | M5-002 | Separate submodule commit passes Python/C++ integration tests |
| M5-004 | `TODO` | Implement C++ target generator and derivative estimator | M5-003 | Shared fixtures match Python outputs/state resets within declared tolerance |
| M5-005 | `TODO` | Implement `rtctrl::arm::Controller` adapter without bypassing bridge/safety layers | M5-004 | Runs unchanged against `SimArm`; no direct hardware/dxl dependency |
| M5-006 | `TODO` | Benchmark allocations, latency, jitter, and missed deadlines | M5-005 | Worst-case qualified workload meets the `rtctrl` command window with approved margin |
| M5-007 | `TODO` | Add 7-DOF gravity simulation configs and paired baselines | M5-005 | Versioned config and tests verify mapping, units, limits, gravity, and telemetry |
| M5-008 | `TODO` | Tune/evaluate selected offline tasks in `rtctrl` simulation | M5-006, M5-007 | Reproducible paired report across seeds/model errors/disturbances |
| M5-009 | `TODO` | Submit any generic `rtctrl` improvements through a dedicated branch/PR | M5-008 | Library tests justify the change; project advances pin separately |
| M5-GATE | `TODO` | Review and close the C++/7-DOF simulation gate | M5-008, M5-009 | Parity, timing, safety, and simulation acceptance evidence is approved |

## M6 — Physical CRANE-X7 offline verification (gated epic)

| ID | Status | Epic | Depends on | Exit criterion |
| --- | --- | --- | --- | --- |
| M6-001 | `TODO` | Write experiment-specific hazard analysis and staged bring-up procedure | M5-GATE | Independent safety reviewer approves executable, model, config, limits, and aborts |
| M6-002 | `TODO` | Rehearse exact hardware executable/config with `rtctrl` simulation and emulator | M6-001 | Exact hashes pass all rehearsals and no code/config changes follow without re-review |
| M6-003 | `TODO` | Validate demonstration recording with torque-off/manual-guidance policy | M6-001 | Supervised procedure yields complete, synchronized, immutable data safely |
| M6-004 | `TODO` | Perform stationary/zero-output and conservative-current checks | M6-002 | Telemetry/watchdogs/abort behavior match approved bounds |
| M6-005 | `TODO` | Execute short nominal offline RC trial under supervision | M6-003, M6-004 | Full run record exists; operator deviations and aborts are documented |
| M6-006 | `TODO` | Expand duration/perturbation only after reviewing prior stage | M6-005 | Each expansion has explicit approval and remains within safety envelope |
| M6-007 | `TODO` | Produce sim-to-real and baseline comparison report | M6-006 | Report includes timing, tracking, limits, failures, model mismatch, and reproducibility |
| M6-GATE | `TODO` | Review and close the offline hardware gate | M6-007 | Scientific and safety reviewers accept evidence and limitations |

## M7 — Online adaptation (definition epic only)

Online learning is intentionally not implementation-ready. Do not write online
hardware code from this task list. First replace this epic with a separately
reviewed plan based on M6 evidence.

| ID | Status | Epic | Depends on | Exit criterion |
| --- | --- | --- | --- | --- |
| M7-001 | `TODO` | Define a task that requires adaptation and a non-adaptive control baseline | M6-GATE | Hypothesis, plant change, oracle/teacher signal, and metrics are unambiguous |
| M7-002 | `TODO` | Specify RLS/LMS update timing, supervision, bounds, freeze, rollback, and persistence | M7-001 | Mathematical and software contract is decision-complete |
| M7-003 | `TODO` | Specify stability/safety envelope independent of the learning update | M7-001 | Limits, monitors, aborts, fallback controller, and test matrix are approved |
| M7-004 | `TODO` | Benchmark update cost and numerical conditioning in Python/C++ | M7-002 | Worst-case update meets deadline/memory bounds and stress tests remain finite |
| M7-005 | `TODO` | Evaluate online adaptation in planar and 7-DOF simulation | M7-002, M7-003, M7-004 | Reproducible improvement/failure report covers adversarial and rollback cases |
| M7-006 | `TODO` | Decide whether hardware online adaptation is justified | M7-005 | Independent scientific/safety review records go/no-go decision |
| M7-GATE | `TODO` | Replace M7 with an implementation-ready plan if approved | M7-006 | Updated PLAN/TASKS leave no hardware implementation decisions unresolved |

## Cross-cutting upstream work

Use these IDs when a need is discovered before its owning milestone. Do not start
an upstream PR merely to make project-local code cleaner.

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| UP-001 | `TODO` | Track `rclib` serialization/import need | M2-006 | Local recipe unblocks Python; issue/PR scope is evidence-backed before M5 |
| UP-002 | `TODO` | Track `rclib` real-time allocation or latency findings | M5-006 | Reproducer and benchmark show generic library impact before PR work |
| UP-003 | `TODO` | Track missing generic `skelarm` extension seam | M2-009 | Existing controller/task/log registries are tested first; PR includes generic tests |
| UP-004 | `TODO` | Track missing generic `rtctrl` bridge/telemetry/safety seam | M5-005 | Existing `Arm`/`Controller` APIs are tested first; PR cannot weaken safety |

## Definition of done for every implementation task

- The task status and evidence are updated in this file.
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
