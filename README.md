# arm-rc-ctrl

Reservoir-computing (echo state network) target generators for robot-arm
control. The research roadmap is [`docs/PLAN.md`](docs/PLAN.md); the
authoritative work-queue entry point is [`docs/TASKS.md`](docs/TASKS.md).
Read those first, then only the relevant [shared specification](docs/design/README.md)
or experiment plan. [Task history and gated backlogs](docs/tasks/README.md)
retain older IDs and evidence; they are not required reading for every task.

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
rendering) and fails naming the first missing or mismatched input. It
requires the checkout to match the evidence exactly — after a submodule pin
advance, add `--from-evidence` to reproduce inside a fresh git worktree at
the recorded audit commit, which carries the evidence's pins. The recorded audits live next to
the report. Any curated run can be exported as a
disposable `skelarm` log and inspected with the pinned player
(`uv run arm-rc-play-run --run <run-id> --scenario configs/tasks/task_1a.toml`,
or `scripts/export_run_sklog.py` for the file; `docs/design/data.md` section 7.5).
Recorded results live under
`docs/experiments/task_1a/` with Git-tracked records under `data/records/`;
see `docs/TASKS.md` for the ledger.

### Replay or regenerate an animation by run ID

After the environment and external-store setup below, run these commands from
the repository root (`uv run --locked` installs the CLI when needed). For example,
this M3REP residual-output run failed when its generated command left the joint
bounds:

```bash
# Interactive player at half speed; no simulation or refitting.
uv run --locked arm-rc-play-run --run run-20260910-4b19d25412c6 \
  --scenario configs/tasks/task_1a.toml --task-clock --speed 0.5

# Browser-viewable GIF, including the time/telemetry panel.
uv run --locked arm-rc-play-run --run run-20260910-4b19d25412c6 \
  --scenario configs/tasks/task_1a.toml --task-clock \
  --export /tmp/m3rep-residual-failure.gif --fps 12 --panel
```

Substitute any recorded run ID. The command uses its Git pointer when present,
otherwise reconstructs a temporary pointer from the configured store, and checks
the payload digests before playback. If the run records its scenario configuration,
the command uses it and checks any supplied scenario against it. Otherwise
(including M3REP runs), supply the original `--scenario` file as above; the player
checks its scenario name and records that the configuration was supplied for playback.
`--task-clock` makes warm-up negative and activation zero (omit it for runs that
do not record an activation time). Use an `.mp4` output instead for video; MP4
export requires FFmpeg. Existing output files are never overwritten: choose a
fresh filename. Interactive playback requires a graphical desktop; `--export`
runs headlessly. Use `uv run --locked arm-rc-play-run --help` for all options.
The original `uv run python scripts/play_run.py` command remains supported.

The follow-up recovery experiment is explained in the
[browser-readable recovery overview](docs/experiments/task_1a_state_conditioned_recovery/overview.html).
Open it directly in a browser for the motivation, method, development results,
and recorded-motion comparisons. It reports the accepted negative result:
134 feasible models, none eligible for freeze or confirmatory evaluation.
The page uses the adjacent committed visual assets and includes reproduction commands.

The repeated-demonstration pilot that followed is explained in the
[browser-readable repetition overview](docs/experiments/task_1a_repeated_demonstration/overview.html):
why the ridge parameter must be explicit, the six-configuration panel, the
paired outcomes, the ridge controls, the residual formulation, the changed
12 rad/s speed threshold, the historical eligibility structure applied as a
descriptive diagnostic, limitations, and measured costs, with every
time-series figure and animation on the task clock (warm-up negative,
activation at 0 s). The owner's review closed the pilot on 2026-09-10:
repetition's benefits are consistent with reduced effective regularization,
not new information from duplicate demonstrations; two configurations passed
the descriptive eligibility rule without exceeding 6 rad/s; no residual or
augmented configuration was feasible in this panel; no model is selected and
any further study requires a separate approved plan.

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
The repeated-demonstration pilot (`docs/experiments/task_1a_repeated_demonstration/plan.md`)
freezes its panel with `scripts/freeze_repetition_panel.py`, validates its ridge
equivalences and fresh-process reproducibility with
`arm_rc_ctrl.experiments.repetition_numerics validate --manifest … --output …
--markdown … --workspace …` (launched pinned through `arm_rc_ctrl.execution run`),
diagnoses any comparison that validation retained as failed with
`arm_rc_ctrl.experiments.repetition_diagnosis --validation … --manifest …
--output … --markdown …`, evaluates the panel's behavioral arms against paired
replay with `arm_rc_ctrl.experiments.repetition_evaluation run --manifest …
--evaluation configs/evaluations/task_1a_repetition_dev_v1.toml --validation …
--evidence-dir …` (resumable at run granularity; one Git pointer per model
configuration and per replay bank), measures one entry's cost with
`arm_rc_ctrl.experiments.repetition_timing smoke … --output … --markdown …`,
accounts for the executed panel with
`arm_rc_ctrl.experiments.repetition_accounting account … --output … --markdown …`,
renders the report tables, task-clock figures, and task-clock animations with
`arm_rc_ctrl.experiments.repetition_report render --docs …` (thin
`scripts/render_repetition_report.py`; the narrative `overview.html` is
hand-written against that report), reproduces the whole pilot from the committed records with
`scripts/reproduce_repetition.py` (`--from-checkout` reproduces in a fresh
worktree; `--doc005` reruns the historical task 1-a reproduction apart;
`--gates` records the quality gates; the audited run is
`docs/experiments/task_1a_repeated_demonstration/reproduction_audit_v1.md`), and records
dependency-build parity across a submodule pin advance with
`arm_rc_ctrl.experiments.build_parity probe` / `compare`. `scripts/play_run.py`
and `scripts/export_run_sklog.py` take `--task-clock` to shift a run's log onto
the task clock (warm-up negative, activation at 0 s) without changing the run record,
and look a run up in the configured store when it has no pointer record under
`data/records/runs` (pilot runs are referenced by their evidence manifests only).
The manual-demonstration experiment (`docs/experiments/task_1a_manual_demonstration/plan.md`)
records takes with the pinned `skelarm` recorder and validates them offline:
`configs/tasks/task_1a_manual_v1.toml` is the protocol's own task configuration
(the task 1-a robot, limits, and target with the continuous final-dwell rule and
the acquisition rules; `task_1a.toml` is untouched), and
`configs/preprocessing/manual_v1.toml` derives full-recording datasets on the task
grid with the hold-anchored zero-phase filter (the recorded reset hold stays exact;
the residual start shift is measured and bounded, never snapped). After the first
practice pilot, the v2 files (plan section 10, I10–I13; the v1 files are unchanged)
separate acquisition from the training grid: `configs/tasks/task_1a_manual_v2.toml`
keeps the same task and the 0.01 s training grid of every comparison arm but
declares 50 Hz acquisition with rules in actual time (`min_duration_s` and a
`max_sample_gap_s` of three acquisition periods, provisional until the second
practice pilot), and a take whose recorder declares another sample period is
rejected. `configs/preprocessing/manual_v2.toml` reconstructs each take from its
actual timestamps onto the grid by linear interpolation and smooths it with the
same zero-phase filter applied to the deviation from the first sample, extended
by its point reflection about that sample and by the repeated final value (5 s
each): the first sample stays at the reset posture up to roundoff without a
stationary hold and without a velocity or acceleration spike, pre-roll
fluctuations right after it are kept, derivatives come from the smoothed
trajectory, the final dwell remains one actual second on the grid, and processed
records carry manual schema 2.
`scripts/validate_manual_takes.py --scenario … --config … --session … --batch N
--manifest … --takes reach_001.sklog.npz …` imports every saved attempt unchanged
(raw take records), measures it against the rules (exact reset start, no sample
gaps, raw increment and processed speeds within the bound, limits, workspace,
one second of uninterrupted final dwell inside 1 cm at joint speeds no greater
than 0.05 rad/s), derives a processed dataset for each accepted take, and keeps
a versioned bank manifest: the first ten accepted takes by attempt number
(acquisition order, whatever batch they arrive in) become `D01`–`D10`, rejected
takes stay retained with their reasons, a byte-identical copy of an earlier take
is rejected without a record of its own and names the original, and an
unreadable file or an archive whose channels disagree with its timestamps is a
per-take `malformed` rejection that never aborts the batch. A batch registers
its Git-tracked records only after all of its takes, so a run without
`--exploratory` never trips over its own records, and it stages everything it
publishes (reports, records, manifest) as a journal in the external store
first: if publication fails part-way, rerunning the batch with the same takes
completes the journal without validating again (without `--exploratory` only
the journal's own outputs may be dirty), and no other batch starts meanwhile; a batch number is recorded once and never edited, and a
complete bank locks its assignments (later batches may only add later
attempts). The manifest is loaded strictly (types, unknown keys, and
assignment consistency). The command exits 2 while the bank still needs takes
(it names how many) and 0 once it is complete. Practice takes belong to a
separate session and never enter a manifest.
`scripts/record_demo.py --scenario configs/tasks/task_1a_manual_v1.toml --recording
configs/recording/task_1a_manual_v1.toml --session … --purpose {practice,study}
--output-root …` starts the pinned recorder in-process from those configurations
and verifies it before anything is recorded: the exact reset posture (no degree
round trip), the joint limits, the target marker, the sampling tick, IK guidance,
both trail overlays, the 30 s per-take timeout, and numbered outputs. Takes and a
portable `session.json` (configurations by digest, the pinned recorder commit, the
resolved options) go to `<output-root>/<purpose>/<session>/`, an absolute directory
outside the repository; a session's settings never change, and `--dry-run` prints
them without writing anything. With the v2 recording configuration the launcher also selects the
last-saved-trail display, so only the most recently saved take is drawn behind the
current trail, verifies it on the recorder window, and records it in `session.json`. With `configs/recording/task_1a_manual_v1.toml` the
sampling rate follows the task period (100 Hz); `configs/recording/task_1a_manual_v2.toml`
sets 50 Hz (a 20 ms tick), and the launcher refuses a rate that is not a whole
number of milliseconds, is faster than the task's training grid, or differs from
the task's acquisition rules; `session.json` records the rate.
`scripts/evaluate_manual_study.py run --study … --evaluation
configs/evaluations/task_1a_manual_dev_v1.toml --evidence-dir … [--entries …]
[--workers N] [--exploratory]` evaluates the frozen study's models against
direct replay of their own demonstrations under the revised protocol:
completion is judged against the configured evaluation horizon rather than a
demonstration's length, the dwell is the acquisition rule measured on the run
itself (one uninterrupted second inside 1 cm at joint speeds no greater than
0.05 rad/s, any excursion restarting the timer), the force pulse fires once the
arm actually holds the target rather than at a fixed task time and its realised
timestamp is what the run records, and every scenario is attempted
independently from a fresh reset so an unsafe run aborts alone and the next one
still runs. Each model is evaluated at its own configuration's inherited
warm-up, and the replay baselines of one demonstration are produced once and
shared by every model paired against them. Completed evidence is immutable: a
manifest is content-addressed and served rather than recomputed, an interrupted
sweep resumes at run granularity after re-verifying every completed run against
the store, and the command leaves a Git pointer to each manifest it produced or
served. `--workers N` evaluates N models at once in worker processes that
inherit the pinned environment and run one numerical thread each; a worker's
runs belong to the study only while its execution identity equals the parent's.
`scripts/smoke_manual_timing.py smoke --study … --evaluation … --evidence-dir …
--output … --markdown … [--models N] [--entries …] [--exploratory]` measures what
the full study will cost before it is executed. It evaluates a deterministic
subset through the same path the full execution uses, so what it reports is what
that execution will do, and takes the subset by position from the frozen study
rather than by a configuration's name: the six inherited configurations exist to
avoid selecting after seeing results, and their historical feasibility labels are
not known to predict performance on manual data. The report records each run's
simulate and persist time, its rows and stored bytes, each model's fit with
whether the cache served it, the peak resident memory of this process and of its
worker children, and the projection: 186 models and 60 replay banks (one per
parent per configuration) over the configured pairs, which is 24,180 RC runs
beside 7,800 replay runs at 65 scenarios and two trackers. The projection
multiplies maximum counts by means measured on a subset, so it is an estimate and
not a bound, and the report says so beside the numbers and next to the earlier
planning estimate. Timings are wall-clock in the canonical single-threaded
environment; bounded parallel execution reduces elapsed time and not storage. An
existing report is never overwritten.
`scripts/derive_manual_results.py derive --study … --evaluation …
--evidence-dir … --run-ordering … --representative-rule … --result-schema
…/result_schema_v3.json --output …/results [--workers N] [--exploratory]`
derives the executed study's machine-readable evidence, from the canonical pinned environment and after checking
every model and bank manifest against the study's trusted inputs exactly as a resume does: one row per run (its
verdict, terminal state, and diagnostic metrics read from its own verified
trajectories over the active segment), the paired comparisons per parent with
both measures (success-count differences over shared scenarios with their
denominators, and improved/worsened/tied tallies), the ten-parent summaries with
median and range, the class summaries with the all-ten arm counted once, the
accounting, the frozen representative rule applied per configuration and
tracker, and the figure inputs. The per-run and per-comparison tables go to the
store behind digest-and-size pointers; everything else is committed, and nothing
is ever overwritten. `scripts/render_manual_case.py cases|plot|animate --inputs
…/results/figure_inputs_v1.json …` lists the illustrated cases and renders one
as a figure, or one of its runs as a GIF, keeping the recorded demonstration,
the commanded reference, and the actual motion visibly apart.
`docs/experiments/task_1a_manual_demonstration/results/usage_v1.md` explains
how to read and reproduce every output.
`scripts/reproduce_manual_study.py audit --study … --evaluation … --evidence-dir …
--results …/results --docs … --output …/audit [--workers N] [--gates]
[--no-resimulate] [--exploratory]` audits that evidence from a clean checkout:
it verifies every source, demonstration, manifest, fit and stored run against
the digests the committed records keep for them, checks each manifest against
the study's trusted inputs as a resume does, recomputes every run's metrics
from its own trajectories and every comparison, summary and selection from the
per-run table, renders a case from the committed figure inputs, and
re-simulates the subset frozen before execution in a scratch store, comparing
each run's arrays digest with the stored one. Every stored run is judged again
from its own trajectories with the evaluation's own judgement, so the dwell,
effort and saturation measurements a row carries are recomputed rather than
copied from the manifest that reported them; every row of the per-run table is
then rebuilt whole from the frozen study and that judgement, and every raw
record behind a demonstration is loaded, bound to its source and checked
against its payload digest. Failures are retained rather than raised: each step
records what it checked and every disagreement, a step whose evidence cannot be
read is recorded as unavailable instead of ending the audit, a re-simulation
that did not reproduce keeps its payloads and the record cites where, the
record lists the declared tolerances and the handoff bundle, and the command
exits non-zero when any step failed. It writes
`reproduction_audit_v<version>.{json,md}` and never overwrites an existing
audit. An audit that was run is kept whether or not it passed: one superseded
by a later version moves to `audit/superseded/`, unedited and with the account
of what replaced it.

### Manual-demonstration interpretation (M3MAN-012)

The [assistant-authored report](docs/experiments/task_1a_manual_demonstration/report/report.md)
interprets the audited one-versus-ten comparison, copy controls and synthetic
variation by configuration and tracker. The owner accepted it at M3MAN-GATE on
2026-09-22 and closed the experiment with a negative finding for the
one-versus-ten question; the decision is in the ledger and
[plan section 12](docs/experiments/task_1a_manual_demonstration/plan.md#12-gate-decision-2026-09-22).
The [reproduction guide](docs/experiments/task_1a_manual_demonstration/report/reproduce.md)
and `scripts/render_manual_report.py` recreate its plots and animation from
the existing evidence without new training or simulation.

The additional [standalone HTML report](docs/experiments/task_1a_manual_demonstration/expert_report/index.html)
introduces the research for a new expert and adds eight synchronized five-arm
case studies, task-space plots and diagnostic time series. Open it locally in a
browser; its [build guide](docs/experiments/task_1a_manual_demonstration/expert_report/README.md)
and `scripts/render_manual_expert_report.py` reproduce the offline presentation.
[Export GIFs and matching plots](docs/experiments/task_1a_manual_demonstration/expert_media.md)
with `scripts/export_manual_expert_media.py` for reuse in your own report.

### Manual-demonstration ESN search (M3MS)

The [approved search](docs/experiments/task_1a_manual_esn_search/plan.md) runs
and resumes through one command, launched pinned; every invocation continues the
same study and the same trial, time and storage caps:

```sh
uv run python -m arm_rc_ctrl.execution run --policy p-cores -- \
  uv run python -m arm_rc_ctrl.experiments.manual_search_run search \
  --protocol configs/studies/manual_esn_search_v1.toml [--stop-at-trials N]
```

`--stop-at-trials` bounds a pilot inside the cap: its trials are the search's
own, and a later invocation without the bound continues after them. The timing
pilot (M3MS-004) states what it will schedule before it runs and is reported
against that statement afterwards; `preflight` never overwrites a stated plan,
and `report` exits non-zero when the records, the ledger and the Optuna study
disagree with it:

```sh
uv run python -m arm_rc_ctrl.experiments.manual_search_pilot preflight \
  --protocol configs/studies/manual_esn_search_v1.toml --stop-at-trials 10 \
  --output docs/experiments/task_1a_manual_esn_search/pilot/preflight_v1.json
uv run python -m arm_rc_ctrl.experiments.manual_search_pilot report \
  --protocol configs/studies/manual_esn_search_v1.toml \
  --preflight docs/experiments/task_1a_manual_esn_search/pilot/preflight_v1.json \
  --output docs/experiments/task_1a_manual_esn_search/pilot/timing_pilot_v2.json \
  --markdown docs/experiments/task_1a_manual_esn_search/pilot/timing_pilot_v2.md
```

Once the search has stopped, the three highest nominal scores are frozen
(M3MS-005), pinned, from a clean checkout. The freeze refuses pending trials,
a search with no spent cap, and a study that disagrees with the retained
records. It verifies each chosen trial's evidence again and never overwrites
a frozen selection:

```sh
uv run python -m arm_rc_ctrl.execution run --policy p-cores -- \
  uv run python -m arm_rc_ctrl.experiments.manual_search_freeze \
  --protocol configs/studies/manual_esn_search_v1.toml \
  --output docs/experiments/task_1a_manual_esn_search/freeze/selection_v1.json \
  --markdown docs/experiments/task_1a_manual_esn_search/freeze/selection_v1.md
```

Add `--verify` to check committed files instead of writing them. The freeze is
rebuilt from the retained trial records, with the chosen evidence verified
again, and must equal the stored record byte for byte. Anything that acts on a
freeze loads it through that same verified path.

The five-arm comparison at the frozen configurations (M3MS-006) runs and
resumes through one command, pinned, under the ceiling it shares with the
search. `status` records its progress in new files, and `publish` verifies
every complete unit again and writes the Git pointers once it has stopped:

```sh
uv run python -m arm_rc_ctrl.execution run --policy p-cores -- \
  uv run python -m arm_rc_ctrl.experiments.manual_comparison_run run \
  --protocol configs/studies/manual_esn_search_v1.toml \
  --freeze docs/experiments/task_1a_manual_esn_search/freeze/selection_v1.json
```

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

The coverage population excludes only the private-payload recovery reproduction
orchestrator. Its full clean-checkout behavior is instead covered by the signed,
regression-locked reproduction audit; reusable experiment and scientific modules
remain subject to the 90% gate.

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

### Canonical execution environment

On a hybrid CPU the core type a process starts on changes closed-loop results
at the 1e-10 level (`docs/experiments/task_1a_repeated_demonstration/execution_environment_probe_v1.md`).
Evidence generation, numerical comparisons, timing, and canonical reproduction
therefore run through the launcher, which resolves the performance cores from
`sysfs` at run time, pins the process, sets `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to `1`, and execs the command so
every child inherits the restriction:

```bash
uv run python -m arm_rc_ctrl.execution run --policy p-cores -- uv run --locked nox
uv run python -m arm_rc_ctrl.execution record --output execution.json --markdown execution.md
```

Pilot commands verify that declaration on start (`require_canonical`) and store
an execution record (affinity, core types, BLAS build and kernel, thread
counts, package versions) whose identity digest is part of their evidence and
cache keys. On a machine without hybrid core types use `--policy all` or an
explicit `--cpus` list; the record then says which environment produced a
result instead of treating environments as interchangeable.

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
pushes to `main`. Its Python jobs clone the full history (`fetch-depth: 0`):
the evidence locks verify with git that the audited reproduction commits are
ancestors of the checkout, which a depth-1 clone cannot answer.

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
docs/          short PLAN/TASKS, design references, task archives/backlogs, experiments
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
