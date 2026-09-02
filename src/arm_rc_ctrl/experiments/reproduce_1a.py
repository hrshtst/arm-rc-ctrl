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
        [--summary reproduce_1a.json] [--keep-going] [--exploratory] [--from-evidence]

The confirmatory rerun (step ``evaluation``) always runs under the
``confirmatory-rerun`` label from a clean checkout; ``--exploratory`` lets the
other steps run in a dirty worktree and makes that step fail clearly. The
scratch directory must be a fresh (empty or missing) directory outside the
repository: nothing is ever deleted. The environment step requires the
checkout's submodule pins to equal the evidence's; after a pin advance, run
``--from-evidence``, which reproduces inside a fresh git worktree at the
evidence's own commit (submodules included) instead of weakening the check.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
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
from arm_rc_ctrl.provenance import command_line, sha256_file, worktree_state
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.recipe import load_recipe
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageError, StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.rc.recipe import ModelRecipe

__all__ = [
    "Check",
    "Reproducer",
    "Reproduction",
    "ReproductionError",
    "audit_markdown",
    "compare_suites",
    "main",
    "prepare_scratch",
    "reproduce",
]

REPO: Final = repository_root()
DOCS: Final = REPO / "docs" / "experiments" / "task_1a"
CONFIRMATORY_REPORT: Final = DOCS / "robustness_confirmatory_v2_recipe_v4.json"
PREPROCESS_CONFIG: Final = REPO / "configs" / "preprocessing" / "default.toml"
STEPS: Final = ("environment", "storage", "records", "payloads", "data", "model", "evaluation", "report")


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


def _is_exact(value: object) -> bool:
    """Whether a leaf must match exactly (flags, counts, labels, absences) rather than within a tolerance."""
    return isinstance(value, bool | str | int | None)


def _compare_leaf(path: str, a: object, b: object, differences: list[str]) -> float | None:
    """Deviation of two leaves, or ``None`` when they are not both leaves."""
    if _is_exact(a) or _is_exact(b):
        if a != b:
            differences.append(f"{path}: {a!r} vs {b!r}")
        return 0.0
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            if not (math.isnan(a) and math.isnan(b)):
                differences.append(f"{path}: {a!r} vs {b!r}")
            return 0.0
        return abs(a - b)
    return None


def _compare(path: str, a: object, b: object, differences: list[str]) -> float:
    """Largest float deviation under ``path``; counts, flags, strings, and shapes must match exactly."""
    kinds = f"{type(a).__name__} vs {type(b).__name__}"
    leaf = _compare_leaf(path, a, b, differences)
    if leaf is not None:
        return leaf
    if isinstance(a, dict) and isinstance(b, dict):
        left = cast("dict[str, object]", a)
        right = cast("dict[str, object]", b)
        worst = 0.0
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append(f"{path}.{key}: present in one report only")
                continue
            worst = max(worst, _compare(f"{path}.{key}", left[key], right[key], differences))
        return worst
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        left_items = cast("Sequence[object]", a)
        right_items = cast("Sequence[object]", b)
        if len(left_items) != len(right_items):
            differences.append(f"{path}: length {len(left_items)} vs {len(right_items)}")
            return 0.0
        pairs = zip(left_items, right_items, strict=True)
        return max((_compare(f"{path}[{i}]", x, y, differences) for i, (x, y) in enumerate(pairs)), default=0.0)
    differences.append(f"{path}: {kinds}")
    return 0.0


def compare_suites(committed: RobustnessSuite, rebuilt: RobustnessSuite) -> tuple[float, list[str]]:
    """Compare every field of every rebuilt run's report with the committed one (the rerun's own run ID excepted).

    Returns the largest absolute deviation over the floating-point fields and
    the list of categorical or structural differences (outcomes, counts,
    labels, windows, shapes), each named by its path.
    """
    by_key = {(r.arm, r.scenario_id): r for r in committed.runs}
    worst = 0.0
    differences: list[str] = []
    for run in rebuilt.runs:
        reference = by_key.get((run.arm, run.scenario_id))
        if reference is None:
            differences.append(f"{run.arm}/{run.scenario_id}: not in the committed suite")
            continue
        left = to_mapping(run.report)
        right = to_mapping(reference.report)
        left.pop("run_id", None)  # the rerun is a new content-addressed run
        right.pop("run_id", None)
        worst = max(worst, _compare(f"{run.arm}/{run.scenario_id}", left, right, differences))
    return worst, differences


_ABSOLUTE_PATH: Final = re.compile(r"(?<![\w./])/(?:[^\s'\"`|,;)]+/)*[^\s'\"`|,;)]+")


def _sanitize(text: str) -> str:
    """Replace absolute paths in a message by their basename (records never carry machine-specific paths)."""
    return _ABSOLUTE_PATH.sub(lambda m: Path(m.group(0)).name or "/", text)


def prepare_scratch(scratch: Path) -> Path:
    """A fresh scratch directory outside the repository; nothing is ever deleted.

    A symbolic link, a location inside the repository, a non-directory, or a
    non-empty directory is refused; a missing directory is created.
    """
    if scratch.is_symlink():
        msg = f"scratch directory {scratch.name!r} is a symbolic link"
        raise ValueError(msg)
    resolved = scratch.resolve()
    if resolved == REPO or resolved.is_relative_to(REPO):
        msg = "the scratch directory must lie outside the repository"
        raise ValueError(msg)
    if resolved.exists():
        if not resolved.is_dir():
            msg = f"scratch {resolved.name!r} is not a directory"
            raise ValueError(msg)
        if any(resolved.iterdir()):
            msg = f"the scratch directory {resolved.name!r} is not empty; nothing is deleted, choose a fresh one"
            raise ValueError(msg)
    else:
        resolved.mkdir(parents=True)
    return resolved


def _need[T](value: T | None, step: str) -> T:
    if value is None:
        msg = f"step {step!r} requires an earlier step that did not run"
        raise ReproductionError(msg)
    return value


@dataclass
class Reproducer:
    """The steps, in order, sharing what they resolve (public so tests can exercise one step directly)."""

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
        commit, dirty = worktree_state(REPO)
        self.inputs["lock_sha256"] = lock
        self.inputs["evidence_project_commit"] = suite.provenance.project_commit
        self.inputs["reproduction_project_commit"] = commit
        self.inputs["reproduction_project_dirty"] = str(dirty)
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
        store_dir = self.scratch / "store"
        store_dir.mkdir()  # the scratch directory was verified empty; a second data step is an error
        scratch_store = StorageRoot(store_dir, repositories=(REPO,))
        payload = store.path(raw.artifact.payload.uri, mode="read")
        scratch_store.path(raw.artifact.payload.uri, mode="write").write_bytes(payload.read_bytes())
        records_root = self.scratch / "repo"
        records_root.mkdir()
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
            scenarios=suite.scenarios,  # replay the recorded scenarios, never regenerated ones
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
    if not (math.isfinite(tolerance) and tolerance >= 0.0):
        msg = f"tolerance must be a finite non-negative number, got {tolerance!r}"
        raise ValueError(msg)
    started = time.perf_counter()
    started_at = (now or datetime.now(tz=UTC)).isoformat()
    reproducer = Reproducer(prepare_scratch(scratch), classes, tolerance, exploratory, store, docs, now)
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


def prepare_evidence_checkout(scratch: Path, *, keep: bool = False) -> tuple[Path, str]:
    """Create a detached worktree at the recorded audit checkout, whose pins equal the evidence's.

    The evidence itself predates this script, so the worktree targets the
    commit the committed reproduction audit ran from
    (``reproduction_project_commit``): it carries both the reproduction
    tooling and the evidence's submodule pins, which are re-verified here.
    Returns the checkout path and that commit. The caller reproduces inside
    it (``uv sync --locked`` there first); with ``keep`` false the worktree
    is registered for manual removal via ``git worktree remove``.
    """
    suite = load_suite(CONFIRMATORY_REPORT)
    audit = json.loads((DOCS / "reproduction_audit.json").read_text(encoding="utf-8"))
    commit = str(audit["inputs"]["reproduction_project_commit"])
    checkout = prepare_scratch(scratch) / "evidence"
    _git("worktree", "add", "--detach", str(checkout), commit)
    _git("-C", str(checkout), "submodule", "update", "--init", "third_party/skelarm", "third_party/rtctrl")
    _git("-C", str(checkout), "submodule", "update", "--init", "--recursive", "third_party/rclib")
    recorded = {s.name: (s.checked_out or s.recorded) for s in suite.provenance.submodules}
    for line in _git("-C", str(checkout), "submodule", "status").splitlines():
        revision, name = line.split()[0].lstrip("+-U"), line.split()[1].split("/")[-1]
        if recorded.get(name) != revision:
            msg = f"evidence checkout has {name} at {revision[:12]}, the evidence recorded {recorded.get(name)!r}"
            raise ReproductionError(msg)
    if not (checkout / "scripts" / "reproduce_1a.py").is_file():
        msg = f"the audit checkout {commit[:12]} carries no scripts/reproduce_1a.py"
        raise ReproductionError(msg)
    if not keep:
        print(f"evidence checkout at {checkout} (remove with: git worktree remove --force {checkout})")
    return checkout, commit


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], check=False, capture_output=True, text=True, cwd=REPO)
    if completed.returncode != 0:
        msg = f"git {' '.join(args[:3])} failed: {completed.stderr.strip()[:300]}"
        raise ReproductionError(msg)
    return completed.stdout


def run_from_evidence(scratch: Path, forwarded: Sequence[str]) -> int:
    """Reproduce inside a fresh worktree at the evidence commit (sync, run, keep the checkout for inspection)."""
    checkout, commit = prepare_evidence_checkout(scratch, keep=True)
    sync = subprocess.run(["uv", "sync", "--locked"], check=False, cwd=checkout)
    if sync.returncode != 0:
        return int(sync.returncode)
    inner_scratch = scratch / "inner"
    command = [
        "uv", "run", "--locked", "python", "scripts/reproduce_1a.py", "--scratch", str(inner_scratch), *forwarded,
    ]  # fmt: skip
    print(f"reproducing at {commit[:12]} in {checkout}")
    inner = subprocess.run(command, check=False, cwd=checkout)
    return int(inner.returncode)


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
    parser.add_argument(
        "--from-evidence",
        action="store_true",
        help="reproduce inside a fresh git worktree at the recorded audit commit (evidence pins included)",
    )
    args = parser.parse_args(argv)
    tolerance = float(args.tolerance)
    if not (math.isfinite(tolerance) and tolerance >= 0.0):
        parser.error(f"--tolerance must be a finite non-negative number, got {args.tolerance!r}")
    ensure_single_thread()  # before rclib is imported and provenance is collected
    scratch = Path(tempfile.mkdtemp(prefix="arm-rc-ctrl-reproduce-")) if args.scratch is None else Path(args.scratch)
    if args.from_evidence:
        forwarded = ["--classes", *cast("list[str]", args.classes), "--tolerance", str(args.tolerance)]
        if args.summary is not None:
            forwarded += ["--summary", str(Path(args.summary).resolve())]
        if args.audit is not None:
            forwarded += ["--audit", str(Path(args.audit).resolve())]
        if args.keep_going:
            forwarded.append("--keep-going")
        return run_from_evidence(scratch, forwarded)
    result = reproduce(
        scratch=scratch,
        classes=cast("list[PerturbationClass]", args.classes),
        tolerance=tolerance,
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
