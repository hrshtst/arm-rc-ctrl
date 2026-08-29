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

**Status:** milestone M0 (repository foundation). No experiment results yet.

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
```

After advancing a submodule pin, rebuild the affected package explicitly, as
uv does not track source changes inside path dependencies:

```bash
uv sync --reinstall-package rclib --reinstall-package skelarm
```

`tests/unit/test_dependency_wiring.py` fails when the installed copies are
stale.

## Quality gate

Everything runs from the locked environment through nox:

```bash
uv run nox                 # lint, type_check, tests, cpp (the full gate)
uv run nox -s lint         # ruff check + ruff format --check
uv run nox -s type_check   # basedpyright, strict mode
uv run nox -s tests        # pytest with branch coverage (coverage.xml)
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

Install the Git hooks once with `uv run pre-commit install`. Use
`UV_PYTHON=3.13 uv run nox` to exercise the other supported interpreter.
`.github/workflows/ci.yml` runs the same sessions on every push to `main` and
every pull request.

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
