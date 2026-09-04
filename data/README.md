# Data and artifact conventions

Git never stores experimental payloads. Raw demonstrations, processed datasets,
run logs, trained models, full study reports, MLflow state, and Optuna databases live below a
machine-local storage root that is resolved at run time (see
[`docs/PLAN.md`](../docs/PLAN.md), section 7.1):

1. `ARM_RC_CTRL_STORAGE_ROOT` environment variable;
2. `[storage].root` in `${XDG_CONFIG_HOME:-$HOME/.config}/arm-rc-ctrl/storage.toml`
   (template: [`configs/storage.example.toml`](../configs/storage.example.toml));
3. `/external/arm-rc-ctrl`.

Tools fail before producing data when the root is absent or not writable. They
never fall back to this repository.

## What Git tracks

| Path | Content | Tracked |
| --- | --- | --- |
| `data/catalog.toml` | Index of all artifact records (introduced by M1-001) | yes |
| `data/records/raw/` | One small TOML record per raw demonstration | yes |
| `data/records/processed/` | One record per canonical processed dataset | yes |
| `data/records/runs/` | One record per curated simulation/evaluation run | yes |
| `data/records/models/` | One record per frozen model recipe | yes |
| `*.dvc`, `dvc.yaml`, `dvc.lock` | Portable DVC metafiles | yes |
| `tests/fixtures/` | Tiny synthetic or sanitized fixtures for automated tests | yes |
| Any payload (`*.npz`, `*.sklog.npz`, `*.parquet`, `*.db`, ...) | External storage only | no |
| `.dvc/config`, `.dvc/.gitignore`, `.dvcignore` | Portable DVC metadata (analytics off, no paths) | yes |
| `.dvc/config.local`, `.dvc/cache/` | Machine-specific DVC cache/remote configuration | no |
| `storage.toml` | Machine-local storage configuration | no |

## Logical URIs

Records reference payloads by logical URI, never by absolute machine path:

```text
armrc://<bucket>/<relative/path>
```

`<bucket>` is one of `raw`, `processed`, `runs`, `models`, `reports`, `mlflow`,
`optuna`, `dvc-cache`, or `dvc-store`, matching the external root layout
(`reports/` holds full study reports whose Git side is a content-addressed
pointer record next to the curated Markdown):

```text
<storage-root>/
├── raw/
├── processed/
├── runs/
├── models/
├── reports/
├── mlflow/
├── optuna/
├── dvc-cache/
└── dvc-store/
```

## Record files

Records are frozen dataclasses in `arm_rc_ctrl.data.records`, serialized as
TOML by `to_toml` and loaded through the strict configuration mapper
(`load_record`), so unknown keys, wrong types, and inconsistent values are
rejected. `tests/fixtures/records/` holds a committed example whose
serialization must stay byte-stable.

- **Artifact ID** (immutable, content-addressed):
  `<kind>-<YYYYMMDD>-<first 12 hex of the payload SHA-256>`, where `<kind>` is
  `raw`, `processed`, `run`, or `model` and the date is the record's
  `created_at` (UTC). `make_artifact_id` derives it.
- **File:** `data/records/<raw|processed|runs|models>/<artifact-id>.toml`.
- **Common `[artifact]` fields:** `schema_version`, `artifact_id`, `kind`,
  `created_at` (UTC, seconds), `license` (SPDX or `LicenseRef-...`),
  `access` (`private` | `internal` | `public`), `notes`, optional
  `expires_at` and `supersedes`; `[artifact.payload]` (`uri`, `sha256`,
  `size`, `format`, `schema_version`); `[artifact.origin]` (`command`,
  `config_sha256`, `project_commit`, `project_dirty`, `dependency_commits`,
  `sources`, optional `run_id`); optional `[artifact.dvc]` (`target`, `md5`).
- **Raw demonstration records** add `[scenario]` (repository-relative config
  path and digest, robot, task, dof, initial posture, target), `[sampling]`
  (period, clock, per-channel units), `session` (pseudonymous teacher or
  recording session), `[intervals]` (`prime`, `move`, `dwell` as contiguous
  `[start, end]` pairs starting at 0), and `duration_s`. Their payload is the
  unchanged `skelarm` log at `armrc://raw/<artifact-id>/demo.sklog.npz`.
- **Processed dataset records** (`ProcessedDatasetRecord`) add `n_samples`,
  `dof`, `task_dim`, `task_code_dim`, `[units]` (must equal the canonical SI
  units), `[phases]` (must equal `prime = 0`, `move = 1`, `dwell = 2`),
  `[preprocessing]` (resampling period, smoothing label and parameters,
  derivative method), `[arrays.<name>]` (shape, dtype, SHA-256 of every
  array), and optional `[normalization]` (training-only statistics with the
  artifacts they were fitted on). Their payload is `samples.npz` at
  `armrc://processed/<artifact-id>/samples.npz` with exactly these arrays
  (`arm_rc_ctrl.data.samples`): `t (N,)`, `q`/`dq`/`ddq (N, dof)`,
  `tip`/`dtip`/`ddtip (N, task_dim)`, `task_code (N, task_code_dim)` — all
  float64 — and `phase (N,)` int64. `origin.sources` must name the raw
  demonstration(s) the dataset was derived from.
- **Catalog:** `data/catalog.toml` lists every record (`artifact_id`, `kind`,
  record path, `uri`, `sha256`, `created_at`). It is append-only: entries are
  never changed or removed.

## Offline signal processing

`arm_rc_ctrl.data.smoothing.smooth(signal, sample_rate_hz, SmoothingConfig)`
applies a zero-phase Butterworth low-pass (forward and backward, so the
magnitude response is squared and there is no phase shift) column by column;
`method = "none"` disables it. Non-finite input, a cutoff at or above Nyquist,
and signals too short for the edge padding are errors. The configuration's
`label`/`parameters()` feed the processed record's `[preprocessing]`.

`arm_rc_ctrl.data.resampling.resample(t, values, ResamplingConfig)` moves a
signal onto the uniform grid `t[0] + k * period` (endpoint included within a
relative tolerance of 1e-6 of the period; never extrapolated) by linear or
cubic interpolation per column; the method is recorded as
`preprocessing.interpolation`.

`arm_rc_ctrl.data.derivatives.differentiate(values, period_s, DerivativeConfig)`
returns first and second derivatives on the uniform grid: `central`
(second-order central differences with second-order one-sided boundary
formulas, so every sample is O(h²)) or `spline` (analytic derivatives of a
not-a-knot cubic spline). Declared tolerances on polynomial and sinusoidal
fixtures are checked in `tests/unit/test_derivatives.py`; the scheme label is
recorded as `preprocessing.derivative_method`.

`arm_rc_ctrl.data.normalization.fit_normalization(arrays, channels, fitted_on=…,
training_rows=mask)` fits per-column mean and standard deviation on the
masked training rows only (evaluation rows can never influence them); columns
with a standard deviation at or below `near_zero` (default 1e-8) get scale 1.0
and are listed in `replaced_near_zero`. `Normalizer` applies and inverts the
recorded statistics.

## Recording and importing demonstrations

`uv run python -m arm_rc_ctrl.data.teacher --scenario configs/tasks/task_1a.toml --out demo.sklog.npz`
records the scripted minimum-jerk demonstration (prime hold, minimum-jerk move
to the closed-form joint solution of the target, dwell hold; tracked by a
high-gain computed-torque controller in skelarm). Human demonstrations come
from skelarm's `tools/trajectory_recorder.py`.

`uv run python -m arm_rc_ctrl.data.import_demo --log demo.sklog.npz --scenario … --session … --license … --access …`
copies the native log unchanged into `armrc://raw/<artifact-id>/demo.sklog.npz`
(staged, digested, moved atomically), verifies it with the raw loader against
the record it is about to write, then writes `data/records/raw/<id>.toml` and
appends the catalog. Intervals are given explicitly (`--intervals
prime_start,move_start,dwell_start,end`) or proposed from the joint-speed
profile (`--propose-intervals [--speed-threshold …] --plot review.png`), which
prints the proposal and writes a review plot; importing with a proposal
requires `--confirm` after visual review.

## Preprocessing command

```bash
uv run python -m arm_rc_ctrl.data.preprocess --raw data/records/raw/<id>.toml \
    --scenario configs/tasks/task_1a.toml [--config configs/preprocessing/default.toml]
```

Loads and verifies the raw log, checks that the raw record was recorded under
the given scenario (digest and dof), smooths joint positions, resamples onto
the scenario's `dt`, differentiates, computes endpoint kinematics, annotates
phases, validates the dataset, fits normalization statistics, and then writes
`samples.npz` (plus `provenance.json`) to a staging directory under the
`processed` bucket, digests it, moves it atomically to
`armrc://processed/<artifact-id>/`, writes `data/records/processed/<id>.toml`,
and appends the catalog. Existing payloads or records are never overwritten,
any failure removes the staging directory, and a dirty worktree is rejected
unless `--exploratory` is given (the origin records the dirty flag either way).

## Phase annotation

`arm_rc_ctrl.data.phases.annotate_phases(t, intervals)` assigns every sample
exactly one phase from the record's contiguous `[intervals]` (half-open
boundaries; the dwell end is inclusive within a small tolerance). Samples
outside the intervals or an interval without samples are errors;
`check_annotation` verifies an existing `phase` array and
`intervals_from_phases` recovers boundaries from one.

## Validating datasets

`arm_rc_ctrl.data.validate.validate_dataset(samples, spec)` checks a
`SampleSet` against a `ValidationSpec` (dimensions, sampling period, joint
position and optional speed limits; `ValidationSpec.from_record` derives it
from a processed record plus the scenario limits): non-finite values, time not
starting at zero / not strictly increasing / not uniform at the period,
dimension mismatches, missing or out-of-order prime/move/dwell phases,
task-code rows that are not one-hot, and joint-limit violations. All problems
are reported together in `DatasetValidationError.problems`; nothing is
repaired.

## DVC

After configuring the storage root, run `uv run python -m arm_rc_ctrl.data.dvc
setup` once per machine. It writes the Git-ignored `.dvc/config.local` so the
DVC cache is `<storage-root>/dvc-cache` and the default local remote `store`
is `<storage-root>/dvc-store`; `... dvc verify` checks the mapping and that
the tracked `.dvc/config` carries no machine paths. Use `dvc add --to-remote`
for large inputs to avoid a repository-local copy.

## Loading raw demonstrations

`arm_rc_ctrl.data.raw.load_raw_demonstration(store, record)` resolves the
record's URI under the storage root, verifies size and SHA-256, parses the
`.sklog.npz` read-only, and cross-checks it against the record (log schema
version, required `q`/`dq` channels, declared units versus the channels and
units in the log, joint count, strictly increasing time from zero, sampling
period per the declared clock, intervals within the recording). Missing,
unreadable, mismatched, corrupt, or disagreeing payloads fail before any data
is returned and are never modified. `tests/fixtures/raw/demo.sklog.npz` with
`tests/fixtures/records/raw-20260830-287036d83d46.toml` is the committed
known-good example.

## Immutability

Payload creation is transactional: write to an external temporary path,
validate, compute the SHA-256 digest, move atomically to the final URI, then
write the repository record. Raw recordings are never overwritten; corrections
create a new artifact ID that names the superseded one. Readers verify size and
digest and fail on missing or mismatched data. Removing a Git record never
deletes an external payload; garbage collection is a separate, explicit, audited
operation.

Each record declares its own license and access classification. A record
without that metadata describes a private, non-redistributable artifact.
