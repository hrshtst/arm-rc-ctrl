<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Data reference

[Roadmap](../PLAN.md) · [Work queue](../TASKS.md)

Detailed sections moved from the roadmap on 2026-09-22. Original section
numbers and historical implementation notes are retained so source-code
and configuration references remain interpretable. Current priorities and
gate status are in the roadmap/queue; experiment-specific decisions live
in each linked experiment plan. Read only the sections relevant to the task.

## 7. Data contracts

### 7.1 Storage location and portability

Experimental payloads do not live in Git and are not stored in the repository
working tree. This includes raw demonstrations, processed datasets, full run
logs, trained models, full per-trial study reports (Git keeps a
content-addressed pointer and the curated Markdown), MLflow state, and
Optuna databases. The only exception is
small synthetic or sanitized data under `tests/fixtures/` required for automated
tests.

All tools resolve a machine-local storage root in this order:

1. `ARM_RC_CTRL_STORAGE_ROOT` environment variable;
2. `[storage].root` in
   `${XDG_CONFIG_HOME:-$HOME/.config}/arm-rc-ctrl/storage.toml`;
3. `/external/arm-rc-ctrl`.

The committed `configs/storage.example.toml` documents the machine-local format.
If the resolved root is absent, inaccessible, or not writable for an operation
that produces data, the command fails before running. It never falls back to the
repository. Versioned metadata contains logical `armrc://` URIs, never absolute
machine paths.

The external root uses this layout:

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

### 7.2 Artifact records and raw demonstrations

Git stores only `data/catalog.toml`, one small TOML record per artifact under
`data/records/{raw,processed,runs,models}/`, and DVC metafiles when applicable.
Every artifact record contains at least:

- schema version, immutable artifact ID, kind, and logical `armrc://` URI;
- SHA-256 digest, byte size, media format, and payload schema version;
- creation timestamp, license, access classification, and optional expiry;
- producing run/command, resolved-config digest, project/dependency revisions,
  and source artifact IDs;
- DVC target/hash when DVC manages the payload.

The native `*.sklog.npz` produced by `skelarm` is retained unchanged at
`armrc://raw/<artifact-id>/demo.sklog.npz`. Its record additionally contains the
robot/scenario configuration, sampling clock and units, pseudonymous teacher or
recording-session ID, task/target/initial posture, notes, and prime/move/dwell
interval boundaries.

Recovery-protocol records additionally preserve the complete acquisition
pre-roll, proposed and confirmed motion-onset samples, detector configuration,
any human adjustment, and the raw-payload digest. Scripted records identify the
programmed onset. These annotations do not define reservoir warm-up.

Payload creation is transactional: write to an external temporary path,
validate it, compute its digest, atomically move it to the immutable final URI,
then write the repository record. Raw recordings are never overwritten. A
correction creates a new artifact ID and records the superseded ID. Readers
verify size and digest and fail on missing or mismatched data.

### 7.3 Canonical processed dataset

Each processed dataset payload is an external `samples.npz` referenced by a
Git-tracked artifact record. Arrays use `float64` and have a common leading
sample dimension:

| Array | Shape | Meaning |
| --- | --- | --- |
| `t` | `(N,)` | Monotonic time in seconds, beginning at zero |
| `q`, `dq`, `ddq` | `(N, dof)` | Demonstrated joint state |
| `tip`, `dtip`, `ddtip` | `(N, task_dim)` | Demonstrated endpoint state |
| `task_code` | `(N, task_code_dim)` | Empty for task 1-a; one-hot later |
| `phase` | `(N,)` | Versioned task phases; M3 uses `prime`/`move`/`dwell`, while recovery datasets crop pre-roll and contain `move`/`dwell` |

The artifact record also contains source IDs, filters, resampling period,
derivative method, normalization statistics, array shapes/dtypes, and checksums.
Validation rejects NaN/Inf, non-monotonic time, unexpected shapes, joint-limit
violations, missing phase intervals, or inconsistent units.

Normalization statistics are fitted on training data only and persisted in the
model recipe. Near-zero scales are replaced by `1.0` and reported.

### 7.4 Run record

Full run records are written under `armrc://runs/<run-id>/`; Git retains only
their artifact records and deliberately curated small reports/plots. Every
simulation/evaluation run records at least:

- measured `t`, `q`, `dq`, and endpoint position;
- desired `q`, raw/filtered desired derivatives, and low-level tracking error;
- active-task `generator_output_q`, optional residual
  `generator_increment_q`, and separately delimited warm-up telemetry;
- requested/applied torque when exposed by the backend;
- task code, target, disturbances, saturation, and termination reason;
- full resolved config, seeds, Git commit, dirty-tree flag, dependency commits,
  DVC hashes, Python lock hash, platform, and library versions.

A confirmatory run from a dirty worktree is rejected unless explicitly marked as
exploratory.

### 7.5 Result inspection and visualization

A verified run can be converted into a disposable `skelarm.StateLog`
(`*.sklog.npz`) and played with the `skelarm` player. The converter resolves the
Git-tracked run pointer, verifies the external payload, and preserves the robot
geometry, target and tolerance, run identity, provenance digests, disturbances,
and available telemetry. `time` and measured joint position form the playback
trajectory. Applied torque is the canonical playback torque; requested torque is
used only when applied torque is unavailable, and both original channels remain
available when present.

The exported log may contain visualization-only floating-point copies of integer
channels, but it is not an archival representation of the run. Repeated exports
must be semantically equivalent, contain no absolute machine paths, and be
written atomically without overwriting an existing file. Exported logs and ad
hoc videos are local, disposable products: they receive no artifact record or
catalog entry and are ignored by Git. A small animation may be committed only
when explicitly curated for a human report; it must name its verified source run
and generation command and remains an illustration rather than primary
experimental evidence.

A thin convenience command (`uv run --locked arm-rc-play-run --run <run-id>`;
the original `scripts/play_run.py` remains supported) exports one run to a temporary location and invokes
the pinned `third_party/skelarm/tools/player.py`; it forwards playback speed,
panel, center-of-mass, and GIF/MP4 export options and propagates failures. This
is kinematic inspection of recorded state, not controller re-execution or
simulation replay. The first version supports one run at a time; synchronized
comparisons, editing, and re-simulation are out of scope.

Runs without individual Git pointers (including M3REP runs) can be resolved
from the configured external store. `--task-clock` shifts the display to the
recorded activation time without changing the stored timestamps. CLI packaging,
usage examples, and regression coverage are tracked as TOOL-003.

Playback-only task metadata requires a generic `skelarm` enhancement. The player
must accept target/tolerance metadata without treating a partial scenario as a
rerunnable source configuration. This change is developed and tested upstream,
then adopted here with a separate submodule-pin update.

## 11. Experiment and data management

- **MLflow:** use a local store under `armrc://mlflow/` (a SQLite tracking
  database plus an artifact directory; MLflow's plain file store is in
  maintenance mode). Every curated run command logs there by default (the
  `--no-mlflow` opt-out is for scratch only): resolved parameters, dependency
  revisions and build identities, payload digests, seeds, scalar metrics,
  plots, reports, model recipes, provenance, and Optuna study summaries. A
  study is mirrored as one parent run (protocol, digest, dataset and tracker
  identities, provenance, summary, selection) with one child run per trial
  (point, objective, every component as its own metric, the running objective
  as a series, the reason, the full evaluation as an artifact), idempotent per
  trial across resumes. The Git pointer record and run directory stay
  authoritative; a tracking server remains optional.
- **DVC:** Git stores only `.dvc` metafiles plus the domain artifact records.
  Configure the cache and default local remote per machine in ignored
  `.dvc/config.local`, resolving them to `<storage-root>/dvc-cache` and
  `<storage-root>/dvc-store`. Use `dvc add --to-remote` for large inputs when it
  avoids a repository-local copy. Never commit a machine-specific absolute path.
- **Optuna:** place local SQLite studies under `armrc://optuna/` (one database
  per study). A study records its identity (protocol digest, seeded sampler,
  pruner, direction) as user attributes and resumes only when that identity
  matches; a failing trial aborts the study instead of being recorded as a
  failure. Export selected trials and study summaries to MLflow so the
  database is not the sole record.
- **Git/uv:** Git pins project/submodule revisions; `uv.lock` pins Python
  dependencies. CMake/submodules pin the C++ build inputs.

Generated payloads, temporary captures, materialized DVC data, and local storage
configuration are ignored by Git. Only artifact records, the catalog, DVC
metafiles, curated small reports/plots/tables, recipes, and documentation are
committed. Removing a Git pointer never deletes an external payload; garbage
collection is a separate, explicit, audited operation.
