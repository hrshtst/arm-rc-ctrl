# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reproduce the task 1-a result from the committed records (``docs/PLAN.md`` section 16; M3-012).

One command resolves the external records the committed evidence points at
and rebuilds, in a scratch store outside the repository, the processed
dataset, the frozen model, the confirmatory evaluation, and the report,
comparing each against what is committed. It fails clearly — naming the
step — when the storage configuration, a payload, a record, a submodule
pin, a build identity, or a lock digest is missing or mismatched. It never
falls back to repository copies, downloads anything, accepts a checksum
mismatch, or picks "the latest" model: every input is the one the committed
records name.

Command line::

    python scripts/reproduce_1a.py [--classes nominal ...] [--tolerance 0.0] [--scratch DIR]
        [--summary reproduce_1a.json] [--keep-going] [--exploratory]

The confirmatory rerun (step ``evaluation``) always runs under the
``confirmatory-rerun`` label from a clean checkout; ``--exploratory`` lets the
other steps run in a dirty worktree and makes that step fail clearly.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import (
    ProcessedDatasetRecord,
    RawDemonstrationRecord,
    load_catalog,
    load_record,
    verify_payload,
)
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.dependencies import submodule_revisions, verify_builds
from arm_rc_ctrl.experiments.baselines import load_frozen_baseline
from arm_rc_ctrl.experiments.confirmatory import load_confirmatory
from arm_rc_ctrl.experiments.perturbations import CLASS_ORDER, PerturbationClass
from arm_rc_ctrl.experiments.report_1a import PLOT_FILES, load_inputs, render_report
from arm_rc_ctrl.experiments.robustness import RobustnessSuite, load_suite, run_robustness
from arm_rc_ctrl.experiments.run_record import RunPointerRecord
from arm_rc_ctrl.provenance import command_line, sha256_file
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.recipe import load_recipe
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageError, StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.rc.recipe import ModelRecipe

__all__ = ["Check", "Reproduction", "ReproductionError", "audit_markdown", "compare_suites", "main", "reproduce"]

REPO: Final = repository_root()
DOCS: Final = REPO / "docs" / "experiments" / "task_1a"
CONFIRMATORY_REPORT: Final = DOCS / "robustness_confirmatory_v2_recipe_v4.json"
PREPROCESS_CONFIG: Final = REPO / "configs" / "preprocessing" / "default.toml"
STEPS: Final = ("environment", "storage", "records", "payloads", "data", "model", "evaluation", "report")
_METRIC_PATHS: Final = (
    ("joint_rmse", "aggregate"),
    ("dwell", "endpoint", "mean"),
    ("dwell", "endpoint", "rms"),
    ("dwell", "endpoint", "max"),
    ("dwell", "endpoint", "p95"),
    ("dwell", "in_tolerance_fraction"),
    ("dwell", "longest_in_tolerance_s"),
    ("dwell", "velocity_rms"),
    ("dwell", "velocity_max"),
    ("effort", "torque_rms"),
    ("effort", "torque_peak"),
    ("effort", "saturation_fraction"),
    ("effort", "effort"),
    ("move_coverage",),
    ("dwell_coverage",),
)


class ReproductionError(RuntimeError):
    """A reproduction step found a missing or mismatched input."""


@dataclass(frozen=True)
class Check:
    """Outcome of one reproduction step."""

    name: str
    ok: bool
    detail: str
    elapsed_s: float


@dataclass(frozen=True)
class Reproduction:
    """The whole reproduction: checks, environment, and inputs."""

    started_at: str
    checks: tuple[Check, ...]
    inputs: dict[str, str]
    environment: dict[str, str]
    max_deviation: float | None
    elapsed_s: float
    schema_version: int = field(default=1)

    @property
    def ok(self) -> bool:
        """Whether every step ran and passed."""
        return len(self.checks) == len(STEPS) and all(c.ok for c in self.checks)


def _scalar(report: object, path: tuple[str, ...]) -> float | None:
    value: object = report
    for name in path:
        if value is None:
            return None
        value = getattr(value, name)
    return None if value is None else float(cast("float", value))


def compare_suites(committed: RobustnessSuite, rebuilt: RobustnessSuite) -> tuple[float, list[str]]:
    """Largest absolute metric deviation over every (arm, scenario) pair, plus categorical differences."""
    by_key = {(r.arm, r.scenario_id): r for r in committed.runs}
    worst = 0.0
    differences: list[str] = []
    for run in rebuilt.runs:
        reference = by_key.get((run.arm, run.scenario_id))
        if reference is None:
            differences.append(f"{run.arm}/{run.scenario_id}: not in the committed suite")
            continue
        a, b = run.report, reference.report
        if (a.termination_kind, a.success, a.failed_criteria) != (b.termination_kind, b.success, b.failed_criteria):
            outcome = f"{a.termination_kind}/{a.success} vs {b.termination_kind}/{b.success}"
            differences.append(f"{run.arm}/{run.scenario_id}: outcome {outcome}")
        for path in _METRIC_PATHS:
            x, y = _scalar(a, path), _scalar(b, path)
            if (x is None) != (y is None):
                differences.append(f"{run.arm}/{run.scenario_id}: {'.'.join(path)} present in one report only")
            elif x is not None and y is not None:
                worst = max(worst, abs(x - y))
        if a.joint_rmse is not None and b.joint_rmse is not None:
            for x, y in zip(a.joint_rmse.per_joint, b.joint_rmse.per_joint, strict=True):
                worst = max(worst, abs(x - y))
    return worst, differences


_ABSOLUTE_PATH: Final = re.compile(r"(?<![\w./])/(?:[^\s'\"`|,;)]+/)*[^\s'\"`|,;)]+")


def _sanitize(text: str) -> str:
    """Replace absolute paths in a message by their basename (records never carry machine-specific paths)."""
    return _ABSOLUTE_PATH.sub(lambda m: Path(m.group(0)).name or "/", text)


def _fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _need[T](value: T | None, step: str) -> T:
    if value is None:
        msg = f"step {step!r} requires an earlier step that did not run"
        raise ReproductionError(msg)
    return value


@dataclass
class _Reproducer:
    """The steps, in order, sharing what they resolve."""

    scratch: Path
    classes: Sequence[PerturbationClass]
    tolerance: float
    exploratory: bool
    configured_store: StorageRoot | None
    docs: Path
    now: datetime | None
    inputs: dict[str, str] = field(default_factory=dict)
    max_deviation: float | None = None
    suite: RobustnessSuite | None = None
    store: StorageRoot | None = None
    recipe: ModelRecipe | None = None
    processed: ProcessedDatasetRecord | None = None
    raw: RawDemonstrationRecord | None = None
    samples: SampleSet | None = None
    scratch_store: StorageRoot | None = None

    def environment(self) -> str:
        """Submodule pins, build identities, and the lock digest match the committed evidence."""
        suite = load_suite(self.docs / CONFIRMATORY_REPORT.name)
        self.suite = suite
        self.inputs["confirmatory_report"] = CONFIRMATORY_REPORT.name
        builds = verify_builds(REPO)
        current = {s.name: (s.checked_out or s.recorded) for s in submodule_revisions(REPO)}
        recorded = {s.name: (s.checked_out or s.recorded) for s in suite.provenance.submodules}
        mismatched = sorted(name for name in recorded if current.get(name) != recorded[name])
        if mismatched:
            msg = f"submodule pins differ from the committed evidence: {mismatched}"
            raise ReproductionError(msg)
        lock = sha256_file(REPO / "uv.lock")
        if lock != suite.provenance.lock_sha256:
            msg = f"uv.lock digest {lock[:12]} differs from the evidence's {suite.provenance.lock_sha256[:12]}"
            raise ReproductionError(msg)
        self.inputs["lock_sha256"] = lock
        self.inputs["project_commit"] = suite.provenance.project_commit
        return f"{len(builds)} build identities verified; submodules {sorted(recorded)} and uv.lock match the evidence"

    def storage(self) -> str:
        """The external storage root resolves (never the repository)."""
        self.store = open_storage() if self.configured_store is None else self.configured_store
        self.inputs["storage_root"] = "<configured external root>"
        return "external storage root resolved"

    def records(self) -> str:
        """The recipe, dataset, raw demonstration, and every run pointer the evidence names exist."""
        suite = _need(self.suite, "records")
        recipe = load_recipe(REPO / suite.recipe_file)
        if recipe.name != suite.recipe:
            msg = f"recipe file names {recipe.name!r}, the evidence used {suite.recipe!r}"
            raise ReproductionError(msg)
        (source,) = recipe.datasets
        processed = load_record(REPO / source.record, ProcessedDatasetRecord)
        if processed.artifact.artifact_id != suite.reference_artifact:
            msg = "the recipe's dataset is not the evidence's reference artifact"
            raise ReproductionError(msg)
        (raw_id,) = processed.artifact.origin.sources
        raw = load_record(REPO / "data" / "records" / "raw" / f"{raw_id}.toml", RawDemonstrationRecord)
        catalog = load_catalog(REPO / "data" / "catalog.toml")
        missing = [
            r.run_id for r in suite.runs if catalog.find(r.run_id) is None or not (REPO / str(r.pointer)).is_file()
        ]
        if missing:
            msg = f"{len(missing)} evidence runs lack a pointer record or catalog entry (first: {missing[0]})"
            raise ReproductionError(msg)
        self.recipe, self.processed, self.raw = recipe, processed, raw
        self.inputs.update({"recipe": suite.recipe_file, "processed": source.record, "raw": raw_id})
        dataset = processed.artifact.artifact_id
        return f"recipe {recipe.name}, dataset {dataset}, raw {raw_id}, {len(suite.runs)} run pointers"

    def payloads(self) -> str:
        """Every payload resolves in the store with its recorded size and digest."""
        store = _need(self.store, "payloads")
        suite = _need(self.suite, "payloads")
        verify_payload(store, _need(self.raw, "payloads").artifact)
        verify_payload(store, _need(self.processed, "payloads").artifact)
        for run in suite.runs:
            verify_payload(store, load_record(REPO / str(run.pointer), RunPointerRecord).artifact)
        return f"raw, processed, and {len(suite.runs)} run payloads verified against their recorded digests"

    def data(self) -> str:
        """The processed dataset rebuilt from the raw payload has the committed digest."""
        store = _need(self.store, "data")
        raw = _need(self.raw, "data")
        processed = _need(self.processed, "data")
        scratch_store = StorageRoot(_fresh(self.scratch / "store"), repositories=(REPO,))
        payload = store.path(raw.artifact.payload.uri, mode="read")
        scratch_store.path(raw.artifact.payload.uri, mode="write").write_bytes(payload.read_bytes())
        records_root = _fresh(self.scratch / "repo")
        (records_root / "data" / "records" / "processed").mkdir(parents=True)
        (records_root / "data" / "records" / "runs").mkdir(parents=True)
        scenario_file = REPO / processed.scenario.config_path
        processed.check_scenario(scenario_file)
        result = preprocess_demonstration(
            REPO / "data" / "records" / "raw" / f"{raw.artifact.artifact_id}.toml",
            scenario_file,
            PREPROCESS_CONFIG,
            store=scratch_store,
            records_root=records_root,
            exploratory=True,
            now=self.now,
        )
        rebuilt = result.record.artifact.payload.sha256
        if rebuilt != processed.artifact.payload.sha256:
            committed = processed.artifact.payload.sha256[:12]
            msg = f"rebuilt dataset digest {rebuilt[:12]} differs from the committed {committed}"
            raise ReproductionError(msg)
        self.scratch_store = scratch_store
        return f"processed dataset rebuilt with digest {rebuilt[:12]} (identical)"

    def model(self) -> str:
        """The frozen recipe refits on the committed dataset and reproduces its fit report."""
        store = _need(self.store, "model")
        recipe = _need(self.recipe, "model")
        processed = _need(self.processed, "model")
        samples = load_samples(verify_payload(store, processed.artifact))
        _model, report = recipe.refit({processed.artifact.artifact_id: samples})
        self.samples = samples
        return f"recipe refitted; fit RMSE {report.rmse:.6g} rad reproduced"

    def evaluation(self) -> str:
        """The confirmatory suite re-runs in the scratch store and matches the committed metrics."""
        if self.exploratory:
            msg = "the confirmatory rerun requires a clean checkout; run without --exploratory from a clean tree"
            raise ReproductionError(msg)
        suite = _need(self.suite, "evaluation")
        store = _need(self.store, "evaluation")
        scratch_store = _need(self.scratch_store, "evaluation")
        processed = _need(self.processed, "evaluation")
        samples = _need(self.samples, "evaluation")
        recipe = _need(self.recipe, "evaluation")
        protocol = load_confirmatory(REPO / suite.protocol_file)
        scenario = load_scenario(protocol.scenario)
        trackers = {a.tracker: load_frozen_baseline(a.tracker) for a in suite.arms}
        payload = store.path(processed.artifact.payload.uri, mode="read")  # the runs' provenance needs the reference
        scratch_store.path(processed.artifact.payload.uri, mode="write").write_bytes(payload.read_bytes())
        rebuilt = run_robustness(
            protocol,
            REPO / suite.protocol_file,
            label="confirmatory-rerun",
            scenario=scenario,
            scenario_file=protocol.scenario,
            dataset=processed,
            reference=samples,
            recipe=recipe,
            recipe_file=REPO / suite.recipe_file,
            estimator=suite.estimator,
            trackers=trackers,
            training_samples={processed.artifact.artifact_id: samples},
            store=scratch_store,
            exploratory=False,
            arms=suite.arms,
            classes=self.classes,
            now=self.now,
        )
        worst, differences = compare_suites(suite, rebuilt)
        self.max_deviation = worst
        if differences:
            msg = f"{len(differences)} categorical differences (first: {differences[0]})"
            raise ReproductionError(msg)
        if worst > self.tolerance:
            msg = f"largest metric deviation {worst:.3e} exceeds the tolerance {self.tolerance:.3e}"
            raise ReproductionError(msg)
        return f"{len(rebuilt.runs)} runs re-evaluated; largest deviation {worst:.3e} (tolerance {self.tolerance:.3e})"

    def report(self) -> str:
        """The report re-renders identically from the committed evidence."""
        rendered = render_report(load_inputs(self.docs), plots=[f"plots/{name}" for name in PLOT_FILES])
        if rendered != (self.docs / "report.md").read_text(encoding="utf-8"):
            msg = "the rendered report differs from the committed report.md"
            raise ReproductionError(msg)
        return "report re-rendered identically from the committed evidence"

    def step(self, name: str) -> Check:
        """Run one step and record its outcome."""
        action = cast("Callable[[], str]", getattr(self, name))
        t0 = time.perf_counter()
        try:
            detail = action()
        except (ReproductionError, StorageError, FileNotFoundError, ValueError, RuntimeError) as exc:
            detail = _sanitize(f"{type(exc).__name__}: {exc}")
            return Check(name, ok=False, detail=detail, elapsed_s=time.perf_counter() - t0)
        return Check(name, ok=True, detail=_sanitize(detail), elapsed_s=time.perf_counter() - t0)


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
    }


def reproduce(
    *,
    scratch: Path,
    classes: Sequence[PerturbationClass] = CLASS_ORDER,
    tolerance: float = 0.0,
    keep_going: bool = False,
    exploratory: bool = False,
    store: StorageRoot | None = None,
    docs: Path = DOCS,
    now: datetime | None = None,
) -> Reproduction:
    """Run every reproduction step; ``scratch`` receives the rebuilt store and records (outside the repository)."""
    started = time.perf_counter()
    started_at = (now or datetime.now(tz=UTC)).isoformat()
    reproducer = _Reproducer(scratch, classes, tolerance, exploratory, store, docs, now)
    checks: list[Check] = []
    for name in STEPS:
        check = reproducer.step(name)
        checks.append(check)
        if not check.ok and not keep_going:
            break
    return Reproduction(
        started_at=started_at,
        checks=tuple(checks),
        inputs=dict(reproducer.inputs),
        environment=_environment(),
        max_deviation=reproducer.max_deviation,
        elapsed_s=time.perf_counter() - started,
    )


def audit_markdown(result: Reproduction, *, command: str, auditor: str = "(to be filled by the auditor)") -> str:
    """A reproduction-audit note: command, environment, per-step outcome and time, inputs, deviation."""
    deviation = "n/a" if result.max_deviation is None else f"{result.max_deviation:.3e}"
    lines = [
        "# Task 1-a reproduction audit",
        "",
        f"- Auditor: {auditor}",
        f"- Started: {result.started_at}",
        f"- Command: `{command}`",
        f"- Elapsed: {result.elapsed_s:.1f} s",
        f"- Outcome: {'PASS' if result.ok else 'FAIL'}",
        f"- Largest metric deviation of the confirmatory rerun: {deviation}",
        "",
        "## Environment",
        "",
        *[f"- {key}: {value or '-'}" for key, value in sorted(result.environment.items())],
        "",
        "## Inputs",
        "",
        *[f"- {key}: `{value or '-'}`" for key, value in sorted(result.inputs.items())],
        "",
        "## Steps",
        "",
        "| step | outcome | elapsed (s) | detail |",
        "| --- | --- | --- | --- |",
        *[f"| {c.name} | {'ok' if c.ok else 'FAILED'} | {c.elapsed_s:.2f} | {c.detail} |" for c in result.checks],
    ]
    missing = [name for name in STEPS if name not in {c.name for c in result.checks}]
    if missing:
        lines.extend(["", f"Steps not run after the first failure: {', '.join(missing)}."])
    return "\n".join(line.rstrip() for line in lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point (exit status 1 when any step fails)."""
    parser = argparse.ArgumentParser(description="Reproduce the task 1-a result from the committed records.")
    parser.add_argument("--classes", nargs="+", default=list(CLASS_ORDER), choices=list(CLASS_ORDER))
    parser.add_argument(
        "--tolerance", type=float, default=0.0, help="largest accepted metric deviation (default exact)"
    )
    parser.add_argument("--scratch", type=Path, default=None, help="scratch directory outside the repository")
    parser.add_argument("--summary", type=Path, default=None, help="write the machine-readable summary here")
    parser.add_argument("--audit", type=Path, default=None, help="write the Markdown audit note here")
    parser.add_argument("--keep-going", action="store_true", help="run every step even after a failure")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="allow a dirty worktree for the data, model, and report steps (the confirmatory rerun then fails)",
    )
    args = parser.parse_args(argv)
    ensure_single_thread()  # before rclib is imported and provenance is collected
    scratch = Path(tempfile.mkdtemp(prefix="arm-rc-ctrl-reproduce-")) if args.scratch is None else Path(args.scratch)
    result = reproduce(
        scratch=scratch,
        classes=cast("list[PerturbationClass]", args.classes),
        tolerance=float(args.tolerance),
        keep_going=bool(args.keep_going),
        exploratory=bool(args.exploratory),
    )
    summary = to_mapping(result)
    summary["ok"] = result.ok
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary is not None:
        Path(args.summary).write_text(text + "\n", encoding="utf-8")
    if args.audit is not None:
        invoked = command_line("arm_rc_ctrl.experiments.reproduce_1a", sys.argv[1:] if argv is None else argv)
        Path(args.audit).write_text(audit_markdown(result, command=invoked), encoding="utf-8")
    print(text)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
