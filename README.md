# arm-rc-ctrl

Reservoir-computing (echo state network) target generators for robot-arm
control. The research roadmap is [`docs/PLAN.md`](docs/PLAN.md); the
authoritative work queue is [`docs/TASKS.md`](docs/TASKS.md).

This repository owns the learning policy, research protocol, experiment
configuration, metrics, tuning, and reproducibility tooling. The domain
libraries are pinned Git submodules:

| Library | Role | Path |
| --- | --- | --- |
| [rclib](https://github.com/hrshtst/rclib) | ESN reservoirs and readouts (C++ core, Python bindings) | `third_party/rclib` |
| [skelarm](https://github.com/hrshtst/skelarm) | Planar arm kinematics/dynamics simulation and teaching logs | `third_party/skelarm` |
| [rtctrl](https://github.com/hrshtst/rtctrl) | CRANE-X7 simulation/hardware bridge (used from milestone M5) | `third_party/rtctrl` |

Pinned commits are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

**Status:** milestones M0–M2 are closed; M3 (tuning and robustness) has
tuned and frozen the task 1-a ESN recipe (v4, from the recorded search v2
and its seed-sensitivity panel) and run the locked confirmatory robustness
suite once on it (260 runs, all successful); start with the human-oriented
`docs/experiments/task_1a/overview.md`, then consult the complete results report
at `docs/experiments/task_1a/report.md` (regenerate with
`uv run python -m arm_rc_ctrl.experiments.report_1a --docs docs/experiments/task_1a
--output docs/experiments/task_1a/report.md --plots docs/experiments/task_1a/plots --force`);
`uv run python scripts/reproduce_1a.py` re-derives the result from the
committed records (submodule pins, lock digest, payload digests, dataset
rebuild, recipe refit, confirmatory rerun into a scratch store, report
rendering) and fails naming the first missing or mismatched input; the
clean-checkout audit follows. Any curated run can be exported as a
disposable `skelarm` log and inspected with the pinned player
(`uv run python scripts/play_run.py --run <run-id> --scenario configs/tasks/task_1a.toml`,
or `scripts/export_run_sklog.py` for the file; `docs/PLAN.md` section 7.5).
Recorded results live under
`docs/experiments/task_1a/` with Git-tracked records under `data/records/`;
see `docs/TASKS.md` for the ledger.

The reservoir-computing commands (`python -m arm_rc_ctrl.rc.train`,
`arm_rc_ctrl.experiments.closed_loop`, `arm_rc_ctrl.experiments.paired`,
`arm_rc_ctrl.experiments.scale_pilot`, `arm_rc_ctrl.experiments.esn_study`)
pin `OMP_NUM_THREADS=1` for their own
process so results are bitwise reproducible; when driving `arm_rc_ctrl.rc`
from an interactive interpreter, export `OMP_NUM_THREADS=1` first (a
different explicit value is rejected).

The run commands (`arm_rc_ctrl.experiments.replay`, `closed_loop`, `paired`)
also log every run to the MLflow store under `armrc://mlflow/tracking/` in the
external storage root (a SQLite database plus artifact directory; the
`--experiment` name defaults to the scenario) and print the MLflow run ID.
`--no-mlflow` skips this for scratch runs only; inspect the store with
`uv run mlflow ui --backend-store-uri sqlite:///<storage-root>/mlflow/tracking/mlflow.db`.
An ESN search (`arm_rc_ctrl.experiments.esn_study --protocol configs/studies/esn_search_1a.toml
--dataset … --report …`) keeps its Optuna study under `armrc://optuna/<name>.db`,
resumes when re-run with the same protocol (`--max-trials` bounds one
invocation), and mirrors the study as one MLflow parent run with a child run
per trial. Only trials feasible in every development scenario can be
selected; `arm_rc_ctrl.experiments.esn_stability` re-evaluates a study's
leading trials over a reservoir-seed panel, and `arm_rc_ctrl.experiments.esn_freeze`
turns the selection into versioned model and evaluation configurations.
The paired robustness suite (`arm_rc_ctrl.experiments.robustness --development
configs/evaluations/task_1a_robustness_dev_v1.toml` or `--confirmatory
configs/evaluations/task_1a_confirmatory_v2.toml`, plus `--dataset`, `--recipe`,
`--evaluation`, `--label`, `--report`) runs every arm (RC and direct replay under
each frozen tracker) on identical generated scenarios — the five classes of
PLAN section 9.2 — as persisted run records, and reports per-class outcomes with
failures counted and paired RC-minus-replay effects.

## Requirements

- Linux (x86_64 tested), Git.
- [uv](https://docs.astral.sh/uv/) 0.12.5 (the version the lock file was
  produced with; CI pins it). uv downloads the interpreter if needed.
- Python 3.12 or 3.13 (`.python-version` selects 3.12).
- A C++17 compiler and CMake ≥ 3.22 (CMake and Ninja are also pulled in by uv
  for the rclib wheel build; the compiler is not).
- For `skelarm`, which imports PyQt6, a headless-capable Qt runtime: on Debian
  or Ubuntu servers install `libegl1 libopengl0 libxkbcommon-x11-0 libdbus-1-3`.
  Tests set `QT_QPA_PLATFORM=offscreen` automatically.

## Setup from a clean checkout

```bash
git clone https://github.com/hrshtst/arm-rc-ctrl.git
cd arm-rc-ctrl

# Pinned domain libraries: rclib recursively (its wheel build needs the nested
# Eigen, pybind11, and Catch2 submodules); skelarm and rtctrl top-level only.
git submodule update --init third_party/skelarm third_party/rtctrl
git submodule update --init --recursive third_party/rclib

# Locked Python environment; builds rclib and skelarm from the submodules.
uv sync

# Reinstall both packages from the checked-out submodules and record their
# build identity (submodule commit, installed version, digests of the Python
# sources and compiled extensions) in the environment's build manifest.
uv run python -m arm_rc_ctrl.dependencies rebuild
```

rclib's nested Eigen submodule is hosted on gitlab.com, which regularly refuses
clones under load. If the recursive init fails with "GitLab is currently unable
to handle this request", point that one submodule at the GitHub mirror and
re-run the recursive init; git checks out the same recorded commit either way:

```bash
git -C third_party/rclib config submodule.cpp_core/third_party/eigen.url \
  https://github.com/eigen-mirror/eigen.git
git submodule update --init --recursive third_party/rclib
```

uv does not rebuild a path dependency when only its sources change, and a
compiled extension carries no revision of its own, so the manifest is the only
link between the installed binaries and the pins. Run the `rebuild` command
again after every submodule pin advance. `uv run nox -s deps` (part of the
default gate) and every provenance-collecting command verify the manifest and
fail when it is missing, the pin moved, a submodule is dirty, or an installed
file differs from what was stamped. Editable installs, used only for upstream
development, are recorded as such and rejected for confirmatory runs.

## Quality gate

Everything runs from the locked environment through nox:

```bash
uv run nox                 # deps, lint, type_check, tests, cpp (the full gate)
uv run nox -s deps         # verify rclib/skelarm build identity (-- --rebuild to rebuild)
uv run nox -s lint         # ruff check + ruff format --check
uv run nox -s type_check   # basedpyright, strict mode
uv run nox -s tests        # pytest with branch coverage (coverage.xml); fails below 90 %
uv run nox -s cpp          # cmake configure/build + ctest with -Werror
uv run nox -s pre_commit   # all pre-commit hooks on all files
```

The underlying commands, if you need them directly:

```bash
uv run ruff check . && uv run ruff format --check .
uv run basedpyright
uv run pytest
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DARM_RC_CTRL_WERROR=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Install the Git hooks once with `uv run pre-commit install`.

To exercise the other supported interpreter, re-create the environment, rebuild
the submodule packages (switching interpreters replaces `.venv` and with it the
environment-local build manifest), and run the gate with the interpreter
assertion, as CI does from its matrix:

```bash
UV_PYTHON=3.13 uv sync --locked
UV_PYTHON=3.13 uv run --locked python -m arm_rc_ctrl.dependencies rebuild
UV_PYTHON=3.13 ARM_RC_CTRL_EXPECTED_PYTHON=3.13 uv run --locked nox
```

Repeat the same three commands with `3.12` to switch back.
`.github/workflows/ci.yml` runs the same sessions on every pull request and on
pushes to `main`.

## External storage root

Experimental payloads (raw demonstrations, processed datasets, run logs,
models, MLflow and Optuna state) never live in this repository. Every tool
resolves one machine-local storage root, in this order:

1. `ARM_RC_CTRL_STORAGE_ROOT` (absolute path);
2. `[storage].root` in `${XDG_CONFIG_HOME:-$HOME/.config}/arm-rc-ctrl/storage.toml`
   (template: [`configs/storage.example.toml`](configs/storage.example.toml));
3. `/external/arm-rc-ctrl`.

The root must already exist, be writable, and lie outside the repository; the
layout below it and the `armrc://<bucket>/…` URIs used by Git-tracked records
are described in [`data/README.md`](data/README.md).

## Smoke experiment

A headless, deterministic end-to-end check (planar 2-DOF `skelarm` PD reach,
then a teacher-forced `rclib` ESN on the log):

```bash
uv run python -m arm_rc_ctrl.experiments.smoke --run-id smoke-001 --exploratory
```

Outputs land in `armrc://runs/smoke-001/` (`arrays.npz`, `summary.json` with
metrics, per-array digests, and full provenance). Run identifiers are
immutable. Without `--exploratory` a dirty worktree is rejected. Two fresh
processes with the same configuration produce bitwise-identical outputs; see
`UP-005` in `docs/TASKS.md` for the known in-process limitation of the pinned
rclib.

## Repository layout

```text
configs/       versioned TOML (robots, tasks, controllers, studies, evaluations)
cpp/           C++17 library/app/tests (rtctrl integration arrives in M5)
data/          Git-tracked artifact records only; payloads are external
docs/          PLAN.md, TASKS.md, PUBLICATION.md, experiment and theory notes
src/arm_rc_ctrl/  Python package (config, storage, provenance, experiments, ...)
tests/         unit, integration, regression tests and tiny fixtures
third_party/   pinned submodules
```

## Licensing and citation

Original code and documentation are licensed under GPL-3.0-only
([`LICENSE`](LICENSE)); third-party components keep their own terms
([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). Citation metadata is in
[`CITATION.cff`](CITATION.cff); the release policy is in
[`docs/PUBLICATION.md`](docs/PUBLICATION.md).
