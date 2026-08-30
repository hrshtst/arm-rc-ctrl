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

- **Next task:** `M1-017`/`M1-018` — dwell and effort metrics.
- **Current milestone:** M1 — demonstration pipeline and frozen baselines.
- **Active blockers:** None.
- **Latest completed work:** M0 closed 2026-08-30 (all M0 tasks and `M0-GATE` `DONE`; PR #1); `UP-005` open.

## Milestone gates

| Milestone | Gate | Status |
| --- | --- | --- |
| M0 | Clean recursive checkout installs and passes documented Python/C++ quality commands | `DONE` |
| M1 | Demonstration preprocessing and both direct-replay baselines are reproducible | `IN PROGRESS` |
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
| M0-003 | `DONE` | Wire local/pinned Python builds of `rclib` and `skelarm` into the `uv` environment | M0-002 | `[tool.uv.sources]` builds non-editable `rclib`/`skelarm` from the submodules (`uv sync`); `src/arm_rc_ctrl/dependencies.py` reports recorded/checked-out submodule revisions, installed versions, and stale installs; `tests/unit/test_dependency_wiring.py` imports both libraries headless, checks versions/sources against the pins, and records revisions; review round 1: `python -m arm_rc_ctrl.dependencies rebuild` reinstalls both packages and stamps a build manifest (submodule commit, version, Python-source and compiled-extension digests, editable flag) in site-packages; `verify_builds()`/`nox -s deps` reject a missing manifest, a moved pin, a dirty submodule, or a changed installed file (C++-only rclib changes included); `tests/unit/test_build_identity.py`; review round 2: the build manifest must list exactly the built packages (missing, duplicate, unknown, or reordered entries are rejected on construction and on load) |
| M0-004 | `DONE` | Add Ruff, strict type checking, pytest, coverage, and pre-commit configuration | M0-001 | Ruff (upstream policy), basedpyright `strict`, pytest (`filterwarnings=error`, strict markers/config), branch coverage, pre-commit, and `noxfile.py` (`lint`, `type_check`, `tests`, `pre_commit`) configured; `tests/unit/test_quality_tools.py` proves each tool reports the planted problems in `tests/fixtures/quality/`; `uv run nox` and `uv run nox -s pre_commit` pass on the clean tree; review round 1: nox sessions run the environment tools directly (no nested `uv run`, which lost `UV_PYTHON`) and assert the interpreter when `ARM_RC_CTRL_EXPECTED_PYTHON` is set |
| M0-005 | `DONE` | Add CMake skeleton under `cpp/` with Catch2 and an empty library/application vertical slice | M0-002 | `cpp/` builds library `arm_rc_ctrl` (`version()`), app `arm_rc_ctrl_version`, and a Catch2 test; Catch2 v3.7.1 is fetched by commit `fa43b77`; `uv run nox -s cpp` configures with `-Werror`, builds, and passes 2 CTest cases headless; `tests/unit/test_cpp_skeleton.py` keeps the C++/Python versions and the Catch2 pin consistent |
| M0-006 | `DONE` | Add CI for Python lint/type/test and C++ configure/build/test | M0-003, M0-004, M0-005 | `.github/workflows/ci.yml`: jobs `python` (3.12/3.13 matrix: selective submodule init, headless Qt libs, `uv sync --locked`, `dependencies rebuild`, `nox -s deps lint type_check tests` with `ARM_RC_CTRL_EXPECTED_PYTHON` from the matrix, coverage artifact), `cpp` (configure with `-Werror`, build, ctest), `pre-commit` (dev group only, rclib/skelarm top-level for metadata); uv pinned to 0.12.5, no secrets, `permissions: contents: read`. Hosted evidence: PR #1 run 33244708306 (3 of 4 jobs passed; pre-commit job fixed in `e401faa`) and run 33285899541 (head `53f6b41`, review round 2 + Eigen mirror fallback) — all four jobs passed (Python 3.12, Python 3.13 under CPython 3.13.15 with 239 tests + 1 strict xfail, C++ configure/build/ctest, pre-commit); run 33285103191 on `86e4c13` failed only because gitlab.com refused the Eigen clone under load; the submodule step now retries and verifies completeness; runs 33285332530 (+ re-run) failed only on gitlab.com refusing the Eigen clone from every runner region despite retries; the submodule step now falls back to the GitHub Eigen mirror (`eigen-mirror/eigen`, verified to serve pinned commit `6d035ce6`), still checking out the recorded SHA |
| M0-007 | `DONE` | Add `.gitignore` and external data/artifact conventions | M0-001 | `.gitignore` excludes payload formats, external-layout directories, DVC cache/local config, `storage.toml`, builds, and tool caches while `data/records/**/*.toml`, DVC metafiles, `configs/*.example.toml`, and fixture data formats under `tests/fixtures/` stay trackable (caches/bytecode there remain ignored); `data/README.md` states the conventions; `tests/unit/test_gitignore_conventions.py` verifies 48 representative paths with `git check-ignore` |
| M0-008 | `DONE` | Add README setup, quality, and recursive-checkout commands | M0-003, M0-005 | `README.md` documents requirements, clean-checkout setup (selective submodule bootstrap, `uv sync`), the nox quality gate and raw commands, pin-advance reinstall, the external storage root, the smoke experiment, layout, licensing/citation; `AGENTS.md` commands aligned; `pyproject.toml` references the README. Clean-room walkthrough (2026-08-29, fresh clone of `m0-foundation` @ `9808640` (same tree; history was later rewritten only to strip commit trailers)) used only documented commands: bootstrap 15 s, `uv sync` 11 s, `uv run nox` 14 s, smoke run, `nox -s pre_commit` — all passed; review round 2: README documents the three-command interpreter switch (`uv sync`, `dependencies rebuild`, `nox` under `UV_PYTHON`/`ARM_RC_CTRL_EXPECTED_PYTHON`), validated 3.12→3.13→3.12; README documents the same Eigen mirror workaround for the recursive rclib init |
| M0-009 | `DONE` | Add GPL-3.0-only licensing and initial third-party notice inventory | — | Root `LICENSE`, plan policy, SPDX guidance, and `THIRD_PARTY_NOTICES.md` are committed |
| M0-010 | `DONE` | Add typed TOML loader with strict unknown-key rejection and relative path resolution | M0-001, M0-004 | `arm_rc_ctrl.config` (`load_config`, `from_mapping`, `to_mapping`, `ConfigError`) maps TOML onto frozen dataclasses with unknown-key rejection, exact types (no int/float/bool coercion, finite floats), dotted error locations, and `Path` resolution relative to the config file; `tests/unit/test_config_loader.py` (27 cases) covers valid, missing, unknown, type-invalid, nested/array locations, path resolution, round trip, and schema errors; review round 1: `Literal` matches type and value (`1.0`/`True` are not `Literal[1]`); a `ValueError` from a schema `__post_init__` is reported as a located `ConfigError`; M1-001: PEP 695 `type` aliases are unwrapped |
| M0-011 | `DONE` | Define provenance record and collection utilities | M0-002, M0-010 | `arm_rc_ctrl.provenance` defines `ProvenanceRecord` (commit, dirty flag, submodule revisions, `uv.lock` SHA-256, canonical-JSON resolved config + digest, `armrc://` artifact references with SHA-256/size, seeds, platform/package/thread-env) with strict mapping/JSON round trip, `collect_provenance`, payload `artifact_reference`/`verify_artifact`, `worktree_state`, and `require_clean_for_confirmatory`; `tests/unit/test_provenance.py` (15 cases) covers digests, artifacts, a throwaway git repo for dirty detection, the real checkout, round trips, strict validation, and the clean/dirty policy; review round 1: records carry verified `builds` identities; `collect_provenance` fails on unknown/mismatched build identity and the confirmatory policy rejects editable or dirty-source builds; records are integrity-validated on construction and load (exact schema version, 40-hex commit, 64-hex digests, canonical `config_json` matching `config_sha256`, aware-UTC second-precision timestamp, exact non-negative integer seeds, unique artifacts/submodules/builds, consistent submodule fields); `collect_provenance` rejects fractional/bool seeds; tampering tests cover each case; review round 2: a record must list exactly `SUBMODULES` and `BUILT_PACKAGES` in order, each build's `source_commit`/`source_dirty` must equal its submodule's `checked_out`/`dirty`, and its `version` must equal `platform.packages[name]` — empty or inconsistent records no longer load (tampering tests for each case) |
| M0-012 | `DONE` | Add headless deterministic smoke experiment independent of GUI/hardware | M0-003, M0-010, M0-011 | `arm_rc_ctrl.experiments.smoke` + `configs/evaluations/smoke.toml`: headless 2-DOF `skelarm` PD reach and teacher-forced `rclib` ESN (`[q,dq]→q_next`), single-OpenMP-thread guard, finite/float64 checks, transactional immutable output under `armrc://runs/<id>/` (`arrays.npz`, `summary.json` with per-array SHA-256, canonical digest, full provenance), clean-worktree policy, `python -m` entry point; `tests/integration/test_smoke_experiment.py`: two fresh-process executions are bitwise identical (tolerance 0); in-process repeat is a strict xfail pending UP-005; review round 1: config validation also covers ESN ranges (mirroring rclib), non-negative gains, and initial/target postures within joint limits |
| M0-013 | `DONE` | Add owner-approved citation and publication metadata | M0-001 | `CITATION.cff` (CFF 1.2.0; author Hiroshi Atsuta <atsuta@ieee.org>, GPL-3.0-only, repository URL, dev version, no DOI/date yet; affiliation/ORCID pending) and `docs/PUBLICATION.md` (public development, archival release + Zenodo DOI only after a reproducible milestone, external/private data); `tests/unit/test_citation_metadata.py` checks required CFF fields and consistency with `pyproject.toml` |
| M0-014 | `DONE` | Implement machine-local external storage-root resolution | M0-010 | `arm_rc_ctrl.storage` resolves env → XDG `storage.toml` → `/external/arm-rc-ctrl`, validates an existing readable root outside any known worktree (never created, never the repository), parses/renders `armrc://<bucket>/…` with traversal/absolute/charset rejection, canonicalizes targets and refuses symlink escapes, and distinguishes read (must exist/readable) from write (writable root, parents created); `tests/unit/test_storage.py` (43 cases) covers precedence, XDG fallback, invalid config, access checks, symlinks, and the committed example config; review round 1: unknown access modes raise `ValueError` without creating directories |
| M0-GATE | `DONE` | Review and close the M0 gate | M0-006, M0-007, M0-008, M0-009, M0-012, M0-013, M0-014 | Reviewer sign-off 2026-08-30 on PR #1 head `c48f3bd`: full gate 239 passed + 1 strict xfail (UP-005), coverage 92%, CTest 2/2 with `-Werror`, Ruff/format/basedpyright strict/pre-commit clean, all four hosted CI jobs passed (run 33286023074, Python 3.12 and 3.13), Eigen mirror serves the pinned commit, worktree and submodules clean. Review rounds 1–2 findings resolved in `c6ade00`…`86e4c13`; residual item UP-005 is documented and non-blocking |

## M1 — Demonstration pipeline and frozen baselines

### M1.1 Data contracts and preprocessing

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M1-001 | `DONE` | Define versioned raw-demonstration artifact records | M0-010, M0-014 | `arm_rc_ctrl.data.records`: frozen `ArtifactRecord` (content-addressed immutable ID `<kind>-<YYYYMMDD>-<12 hex of payload SHA-256>`, kind, UTC timestamp, license, access class, optional expiry/supersedes, `Payload` URI/SHA-256/size/format/schema, `Origin` command/config digest/project+dependency commits/sources/run, optional `DvcPointer`) and `RawDemonstrationRecord` (scenario config path+digest, robot/task/dof/initial posture/target, sampling period/clock/units, pseudonymous session, contiguous prime/move/dwell intervals, duration); TOML via `tomli-w` with `None` omitted, strict load, immutable atomic writes, `data/records/<kind>/<id>.toml` layout, append-only `data/catalog.toml`, payload digest/verification through the storage root; `tests/unit/test_artifact_records.py` (72 cases) covers round trips incl. the committed fixture `tests/fixtures/records/raw-20260830-2a97516c354b.toml` (byte-stable), every invariant, immutability, catalog semantics; `data/README.md` documents the layout |
| M1-002 | `DONE` | Define canonical `samples.npz` arrays and processed artifact records | M1-001 | `arm_rc_ctrl.data.samples.SampleSet` fixes the canonical arrays (`t`, `q`/`dq`/`ddq`, `tip`/`dtip`/`ddtip`, `task_code` float64; `phase` int64 with documented codes prime=0/move=1/dwell=2), consistent shapes, ≥2 samples, read-only copies, per-array SHA-256; `data.arrays` gives digest/`.npz` I/O with exact-name loading; `ProcessedDatasetRecord` adds dimensions, exact canonical SI units, exact phase encoding, preprocessing (period/smoothing/derivative), per-array specs (shape/dtype/digest), optional training-only normalization, raw `origin.sources`, payload pinned at `armrc://processed/<id>/samples.npz`, and `check_samples` drift detection; `tests/unit/test_samples.py` + `tests/unit/test_processed_records.py` (54 cases) with committed byte-stable fixture `processed-20260830-555555555555.toml`; `data.synthetic` provides deterministic fixtures |
| M1-003 | `DONE` | Implement raw `skelarm` log loader through the storage resolver | M0-003, M1-001 | `arm_rc_ctrl.data.raw.load_raw_demonstration(store, record)` resolves the `armrc://` URI through the storage root, verifies size/SHA-256, parses the `.sklog.npz` read-only (including its on-disk `schema_version`), and cross-checks the log against the record (schema version, required `q`/`dq`, declared units vs. log channels/units, joint count incl. embedded skeleton, strictly increasing time from zero, sampling period per declared clock, intervals within the recording); returns read-only arrays. Committed known fixture `tests/fixtures/raw/demo.sklog.npz` (3.9 kB, skelarm PD reach via `data.synthetic.synthetic_demonstration_log`) with record `raw-20260830-287036d83d46` and scenario `tests/fixtures/configs/planar_2dof_fixture.toml`; `tests/unit/test_raw_loader.py` (20 cases) covers loading, missing, unreadable, size/digest mismatch, corrupt/meta-less archives, and every disagreement, asserting the source bytes/mtime are untouched |
| M1-004 | `DONE` | Implement dataset validation | M1-002 | `arm_rc_ctrl.data.validate`: `ValidationSpec` (dof/task_dim/task_code_dim, period, `JointLimits` position + optional speed, phase completeness switch, tolerances; `from_record`) and pure `dataset_problems`/`validate_dataset` that report every finding together (`DatasetValidationError.problems`) and never repair: non-finite values per array with first index, time not starting at 0 / not strictly increasing / not uniform at the period, dimension mismatches, missing or returning phases, non-one-hot task-code rows, joint position and speed limit violations; `tests/unit/test_dataset_validation.py` (47 cases) incl. aggregation and spec/limit invariants |
| M1-005 | `DONE` | Implement configurable smoothing using zero-phase processing for offline demonstrations | M1-003, M1-004 | `arm_rc_ctrl.data.smoothing`: `SmoothingConfig` (`none` | `butterworth`, cutoff, order) and `smooth()` = zero-phase `sosfiltfilt` Butterworth low-pass per column; rejects non-finite input, cutoff ≥ Nyquist, and too-short signals; `label`/`parameters()` feed `Preprocessing`; `scipy` declared as a dependency with `scipy-stubs` for strict typing; `tests/unit/test_smoothing.py` (11 cases): passband tone amplitude within 0.1% and phase < 1e-4 rad, 40 Hz tone attenuated > 1000×, symmetric impulse response, DC/ramp preserved, per-column independence, validation |
| M1-006 | `DONE` | Implement resampling to the configured control period | M1-005 | `arm_rc_ctrl.data.resampling`: `uniform_grid` (`t0 + k·period`, endpoint included within 1e-6·period, final point clamped so nothing extrapolates) and `resample()` with linear (`np.interp`) or cubic (`make_interp_spline`) interpolation per column; strict checks on time base, finiteness, shapes, and coarse periods; `Preprocessing.interpolation` records the method; `tests/unit/test_resampling.py` (16 cases): constant/linear exact for both methods, endpoint inclusion/exclusion, identity on-grid, cubic exactness on cubics, per-column independence, rejections |
| M1-007 | `DONE` | Implement offline derivative generation for `dq`, `ddq`, `dtip`, and `ddtip` | M1-006 | `arm_rc_ctrl.data.derivatives`: `DerivativeConfig` (`central` = second-order central differences with second-order one-sided boundaries incl. the 4-point boundary second derivative; `spline` = not-a-knot cubic spline derivatives), `first_derivative`/`second_derivative`/`differentiate` per column, strict input checks; `tests/unit/test_derivatives.py` (13 cases): polynomials ≤ quadratic exact, `t³` shows the known `h²` central error, sinusoid interior/boundary errors within declared tolerances (central 5e-3/2e-2 first, 2e-2/2e-1 second; spline 2e-3/2e-2, 2e-2/3e-1), per-column independence, rejections |
| M1-008 | `DONE` | Implement or validate explicit prime/move/dwell interval annotation | M1-003 | `arm_rc_ctrl.data.phases`: `annotate_phases(t, intervals)` gives every sample exactly one code (half-open boundaries, inclusive dwell end within a tolerance), rejects samples outside the intervals and intervals without samples; `check_annotation` verifies an existing phase array; `intervals_from_phases` recovers boundaries; missing/overlapping/gapped/reversed intervals fail in `Intervals`; `tests/unit/test_phases.py` (23 cases) |
| M1-009 | `DONE` | Implement training-only normalization statistics | M1-004 | `arm_rc_ctrl.data.normalization`: `fit_normalization(arrays, channels, fitted_on, training_rows, near_zero)` fits per-column mean/std over the boolean training mask only, replaces std ≤ `near_zero` (default 1e-8) by 1.0 and records the indices, skips zero-width channels, rejects empty masks/non-finite training rows; `Normalizer.transform/inverse`; `tests/unit/test_normalization.py` (12 cases) incl. a leakage test (arbitrary/NaN evaluation rows leave the statistics identical) and near-zero handling |
| M1-010 | `DONE` | Add transactional preprocessing CLI as a thin wrapper around tested functions | M1-006, M1-007, M1-008, M1-009 | `arm_rc_ctrl.data.preprocess` (`preprocess_demonstration` + `python -m arm_rc_ctrl.data.preprocess --raw … --scenario … [--config …] [--exploratory]`): verifies the raw payload, checks the raw record was recorded under the given scenario (digest, dof), smooths → resamples to `dt` → derivatives → skelarm FK endpoint → phases → `SampleSet` → `validate_dataset` → normalization; writes `samples.npz` + `provenance.json` to a staging dir, digests, moves atomically to `armrc://processed/<content-addressed id>/`, then writes the record and appends the catalog; existing payload/record → `FileExistsError`, any failure removes staging, dirty worktree rejected unless exploratory, storage root inside the repo rejected; `configs/preprocessing/default.toml`; fixture scenario converted to the scenario schema; `tests/integration/test_preprocess.py` (6 cases incl. deterministic rerun, failure cleanup, CLI) |
| M1-011 | `DONE` | Initialize DVC with external per-machine cache and remote | M0-007, M0-014, M1-010 | `dvc` 3.67 added; `dvc init` metadata tracked (`.dvc/config` = analytics off, no paths; `.dvc/.gitignore`; `.dvcignore`); `arm_rc_ctrl.data.dvc` `setup` writes the ignored `.dvc/config.local` (cache `<root>/dvc-cache`, default local remote `store` = `<root>/dvc-store`) and `verify` checks the mapping, that the tracked config has no paths, and that `config.local` is ignored/untracked; `tests/integration/test_dvc_setup.py` (6 cases) incl. `dvc add`/`push` in a temporary repo placing bytes only under the storage root and only the `.dvc` metafile in Git |
| M1-012 | `DONE` | Create or select the canonical 2-DOF robot/task 1-a scenario | M0-003 | `configs/tasks/task_1a.toml` pins the planar 2-DOF robot (links 0.30/0.25 m, masses, inertias, joint limits ±3 rad, gravity-free), velocity/torque/endpoint-radius limits, initial posture (0.2, 1.2) rad, endpoint target (0.10, 0.45) m ± 0.01 m (0.46 m, elbow-up solution ≈ (0.83, 1.16) rad), `dt` = 0.01 s, and nominal prime/move/dwell intervals [0,1]/[1,4]/[4,5] s; `arm_rc_ctrl.scenario` provides the typed `ScenarioConfig` (cross-field checks: per-joint lengths, posture within limits, target within reach and radius), `load_scenario`, `joint_limits`, `build_skeleton`, and `endpoint_positions` via skelarm FK; smoke reuses the shared link/robot schema; `tests/unit/test_scenario.py` (22 cases) incl. closed-form IK/FK agreement |
| M1-013 | `TODO` | Record/import one external task 1-a demonstration and commit its immutable record | M1-010, M1-012 | Payload is absent from Git, record resolves and validates, and visual/manual review confirms intended reach/dwell |
| M1-014 | `TODO` | Add preprocessing integration/regression fixture | M1-013 | Clean reproduction through a configured external store yields expected records, arrays, and digests |

### M1.2 Metrics and run records

| ID | Status | Task | Depends on | Acceptance/evidence |
| --- | --- | --- | --- | --- |
| M1-015 | `DONE` | Define typed termination reasons and success/failure records | M0-010 | `arm_rc_ctrl.experiments.termination`: `Termination` (kinds `completed`, `invalid_state`, `invalid_output`, `limit_violation` with limit/joint/value/bound, `divergence`, `timeout`, `backend_failure`; time/step/detail validated per kind) with factories, and `Outcome` (termination + named success criteria; `success`, `failed_criteria`; `completed` criterion must agree with the termination); strict mapping round trip; `tests/unit/test_termination.py` (14 cases) |
| M1-016 | `DONE` | Implement joint RMSE with per-joint continuous-angle policy | M1-002 | `arm_rc_ctrl.metrics.joint`: `wrap_angle` to (-π, π], `JointAnglePolicy` (per-joint continuous flag; `limited(dof)`), `joint_error` (wrapping on continuous joints only), `joint_rmse` → aggregate `sqrt(Σ‖wrap(q−q_demo)‖²/(N d))` and per-joint RMSE with sample count; shape/finiteness rejection; `tests/unit/test_joint_metrics.py` (11 cases) with hand-calculated fixtures, zero-for-identical/non-negative properties, wrapping policy, and rejections |
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
| UP-005 | `TODO` | Fix in-process seed reproducibility of `rclib` `RandomSparseReservoir` (power-iteration start vector uses `Eigen::Random()`/`std::rand`, never re-seeded; `cpp_core/src/reservoirs/RandomSparseReservoir.cpp:39` at pin `a015aca`) | M0-012 | Found 2026-08-29: fresh processes reproduce exactly, repeated construction in one process drifts; documented by strict-xfail `tests/integration/test_smoke_experiment.py::test_in_process_repeat_is_reproducible`. Upstream branch/PR with a regression test, then advance the pin and drop the xfail |

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
