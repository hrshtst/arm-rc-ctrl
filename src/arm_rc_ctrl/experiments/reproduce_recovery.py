# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reproduce the task 1-a recovery result from the committed records (M3R-019; negative-result path).

One command resolves the external records the committed evidence points at and
rebuilds, in a scratch store outside the repository, the cropped recovery
dataset, the anchor augmentation episodes, the representative trial's recipe
and its full development evaluation, and the curated representative pairs,
comparing each against what is committed; finally the ablation, freeze, and
recovery reports re-render byte-for-byte. No recipe is frozen and no
confirmatory suite exists: the reproduction target is the development chain
behind the accepted negative result, and the pair rerun carries a distinct
reproduction command label. It fails clearly — naming the step — when the
storage configuration, a payload, a record, a submodule pin, a build identity,
or a lock digest is missing or mismatched, and it never falls back to
repository copies or accepts a checksum mismatch.

Command line::

    python scripts/reproduce_recovery.py [--tolerance 0.0] [--scratch DIR]
        [--summary reproduce_recovery.json] [--audit reproduce_recovery.md]
        [--keep-going] [--exploratory]

The pair rerun (step ``pairs``) always runs from a clean checkout;
``--exploratory`` lets the other steps run in a dirty worktree and makes that
step fail clearly. The scratch directory must be a fresh directory outside the
repository: nothing is ever deleted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.derivatives import DerivativeConfig
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_catalog, load_record, verify_payload
from arm_rc_ctrl.data.recover import recover_demonstration
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.dependencies import submodule_revisions, verify_builds
from arm_rc_ctrl.experiments.evidence import StoredReport, load_report_pointer, open_stored_report
from arm_rc_ctrl.experiments.recovery_ablation import AblationReport, load_ablation, render_ablation_markdown
from arm_rc_ctrl.experiments.recovery_freeze import FreezeRecord, load_freeze, render_freeze_markdown
from arm_rc_ctrl.experiments.recovery_objective import (
    RecoveryTrialContext,
    evaluate_recovery_point,
    train_recovery_point,
)
from arm_rc_ctrl.experiments.recovery_report import PLOT_FILES, build_report_inputs, render_recovery_report
from arm_rc_ctrl.experiments.recovery_representative import RepresentativeRecord, load_representatives
from arm_rc_ctrl.experiments.recovery_search import (
    RecoverySearchProtocol,
    load_recovery_search,
    point_from_params,
    recovery_protocol_digest,
)
from arm_rc_ctrl.experiments.recovery_slice import run_recovery_pair
from arm_rc_ctrl.experiments.reproduce_1a import Check, ReproductionError, prepare_scratch
from arm_rc_ctrl.experiments.run_record import RunPointerRecord, load_run
from arm_rc_ctrl.provenance import canonical_json, command_line, sha256_bytes, sha256_file, worktree_state
from arm_rc_ctrl.rc.augment import generate_augmentation
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.recipe import AugmentationTrainingSpec
from arm_rc_ctrl.rc.warmup import WarmupConfig
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageError, StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.studies import TrialRecord
    from arm_rc_ctrl.rc.recipe import ModelRecipe

__all__ = [
    "STEPS",
    "RecoveryReproduction",
    "Reproducer",
    "animation_names",
    "audit_markdown",
    "compare_evidence",
    "main",
    "reproduce",
]

REPO: Final = repository_root()
DOCS: Final = REPO / "docs" / "experiments" / "task_1a_state_conditioned_recovery"
DERIVE_CONFIG: Final = REPO / "configs" / "preprocessing" / "recovery_v1.toml"
STEPS: Final = ("environment", "storage", "records", "payloads", "data", "episodes", "model", "pairs", "report")
_ANCHOR_D1: Final = {"n_synthetic": 64, "sigma_rad": 0.05, "phi": 0.99, "gamma": 1.0}
_DERIVATIVE_LABELS: Final[dict[str, Literal["central"]]] = {"central-difference": "central"}
_REPRODUCTION_COMMAND: Final = "python -m arm_rc_ctrl.experiments.reproduce_recovery (reproduction rerun)"


@dataclass(frozen=True)
class RecoveryReproduction:
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


def compare_evidence(path: str, committed: object, rebuilt: object, differences: list[str]) -> float:
    """Largest float deviation between two mappings; every non-float mismatch is categorical."""
    if isinstance(committed, dict) and isinstance(rebuilt, dict):
        committed_map = cast("dict[str, object]", committed)
        rebuilt_map = cast("dict[str, object]", rebuilt)
        if set(committed_map) != set(rebuilt_map):
            differences.append(f"{path}: keys {sorted(committed_map)} != {sorted(rebuilt_map)}")
            return 0.0
        return max(
            (
                compare_evidence(f"{path}.{key}", committed_map[key], rebuilt_map[key], differences)
                for key in committed_map
            ),
            default=0.0,
        )
    if isinstance(committed, (list, tuple)) and isinstance(rebuilt, (list, tuple)):
        committed_seq = cast("Sequence[object]", committed)
        rebuilt_seq = cast("Sequence[object]", rebuilt)
        if len(committed_seq) != len(rebuilt_seq):
            differences.append(f"{path}: length {len(committed_seq)} != {len(rebuilt_seq)}")
            return 0.0
        return max(
            (
                compare_evidence(f"{path}[{i}]", a, b, differences)
                for i, (a, b) in enumerate(zip(committed_seq, rebuilt_seq, strict=True))
            ),
            default=0.0,
        )
    if isinstance(committed, bool) or isinstance(rebuilt, bool) or committed is None or rebuilt is None:
        if committed != rebuilt:
            differences.append(f"{path}: {committed!r} != {rebuilt!r}")
        return 0.0
    if isinstance(committed, (int, float)) and isinstance(rebuilt, (int, float)):
        return abs(float(committed) - float(rebuilt))
    if committed != rebuilt:
        differences.append(f"{path}: {committed!r} != {rebuilt!r}")
    return 0.0


def _need[T](value: T | None, step: str) -> T:
    if value is None:
        msg = f"step {step!r} requires an earlier step that did not run"
        raise ReproductionError(msg)
    return value


def animation_names(record: RepresentativeRecord) -> tuple[str, ...]:
    """The committed animation file names in the export order (pd_v2 pairs, RC before replay)."""
    names: list[str] = []
    for pair in record.pairs:
        if pair.tracker != "pd_v2":
            continue
        names.append(f"{pair.kind}_rc_pd.gif")
        names.append(f"{pair.kind}_replay_pd.gif")
    return tuple(names)


@dataclass
class Reproducer:
    """The steps, in order, sharing what they resolve (public so tests can exercise one step directly)."""

    scratch: Path
    tolerance: float
    exploratory: bool
    configured_store: StorageRoot | None
    docs: Path
    now: datetime | None
    inputs: dict[str, str] = field(default_factory=dict)
    max_deviation: float | None = None
    store: StorageRoot | None = None
    scratch_store: StorageRoot | None = None
    pointers: dict[str, StoredReport] | None = None
    ablation: AblationReport | None = None
    freeze: FreezeRecord | None = None
    representative: RepresentativeRecord | None = None
    dataset: RecoveryDatasetRecord | None = None
    raw: RawDemonstrationRecord | None = None
    samples: SampleSet | None = None
    protocol: RecoverySearchProtocol | None = None
    trial: TrialRecord | None = None
    context: RecoveryTrialContext | None = None
    recipe: ModelRecipe | None = None

    def environment(self) -> str:
        """Submodule pins, build identities, and the lock digest match the committed freeze record."""
        freeze = load_freeze(self.docs / "model_freeze_v2.json")
        self.freeze = freeze
        builds = verify_builds(REPO)
        current = {s.name: (s.checked_out or s.recorded) for s in submodule_revisions(REPO)}
        recorded = {s.name: (s.checked_out or s.recorded) for s in freeze.provenance.submodules}
        mismatched = sorted(name for name in recorded if current.get(name) != recorded[name])
        if mismatched:
            msg = f"submodule pins differ from the committed evidence: {mismatched}"
            raise ReproductionError(msg)
        lock = sha256_file(REPO / "uv.lock")
        if lock != freeze.provenance.lock_sha256:
            msg = f"uv.lock digest {lock[:12]} differs from the evidence's {freeze.provenance.lock_sha256[:12]}"
            raise ReproductionError(msg)
        commit, dirty = worktree_state(REPO)
        self.inputs["evidence_project_commit"] = freeze.provenance.project_commit
        self.inputs["reproduction_project_commit"] = commit
        self.inputs["reproduction_project_dirty"] = str(dirty)
        return f"{len(builds)} build identities verified; submodules {sorted(recorded)} and uv.lock match the evidence"

    def storage(self) -> str:
        """The external storage root resolves (never the repository)."""
        self.store = open_storage() if self.configured_store is None else self.configured_store
        self.inputs["storage_root"] = "<configured external root>"
        return "external storage root resolved"

    def records(self) -> str:
        """Every committed evidence record loads strictly and binds its neighbours by digest."""
        freeze = _need(self.freeze, "records")
        pointers = {f.name: load_report_pointer(f) for f in sorted(self.docs.glob("recovery_search_*_v1.toml"))}
        pointers["residual_search_1a_v1.toml"] = load_report_pointer(self.docs / "residual_search_1a_v1.toml")
        ablation = load_ablation(self.docs / "development_ablation_v2.json")
        if freeze.ablation_sha256 != sha256_file(self.docs / freeze.ablation_file):
            msg = "the freeze record's ablation digest does not match the committed ablation file"
            raise ReproductionError(msg)
        for study in freeze.studies:
            pointer = pointers.get(study.file)
            if pointer is None or pointer.protocol_sha256 != study.protocol_sha256:
                msg = f"freeze input {study.file!r} does not match the committed pointer"
                raise ReproductionError(msg)
        for pointer in pointers.values():
            config = REPO / "configs" / "studies" / f"recovery_search_1a_{pointer.formulation}_v1.toml"
            if recovery_protocol_digest(load_recovery_search(config)) != pointer.protocol_sha256:
                msg = f"committed protocol {config.name!r} no longer hashes to the study digest"
                raise ReproductionError(msg)
        representative = load_representatives(self.docs / "recovery_representative_v1.json")
        catalog = load_catalog(REPO / "data" / "catalog.toml")
        missing = [
            run_id
            for pair in representative.pairs
            for run_id in (pair.replay_run, pair.rc_run)
            if catalog.find(run_id) is None or not (REPO / "data" / "records" / "runs" / f"{run_id}.toml").is_file()
        ]
        if missing:
            msg = f"{len(missing)} representative runs lack a pointer record or catalog entry (first: {missing[0]})"
            raise ReproductionError(msg)
        dataset_file = REPO / "data" / "records" / "processed" / f"{representative.dataset}.toml"
        dataset = load_record(dataset_file, RecoveryDatasetRecord)
        (raw_id,) = dataset.artifact.origin.sources
        raw = load_record(REPO / "data" / "records" / "raw" / f"{raw_id}.toml", RawDemonstrationRecord)
        self.pointers, self.ablation, self.representative = pointers, ablation, representative
        self.dataset, self.raw = dataset, raw
        self.inputs.update({"dataset": representative.dataset, "raw": raw_id})
        return (
            f"{len(pointers)} study pointers, ablation, freeze, representative record, dataset "
            f"{representative.dataset}, raw {raw_id}, and {2 * len(representative.pairs)} run pointers verified"
        )

    def payloads(self) -> str:
        """Every payload resolves with its recorded size and digest; the source trial matches the record."""
        store = _need(self.store, "payloads")
        pointers = _need(self.pointers, "payloads")
        representative = _need(self.representative, "payloads")
        for pointer in pointers.values():
            open_stored_report(store, pointer)
        verify_payload(store, _need(self.raw, "payloads").artifact)
        verify_payload(store, _need(self.dataset, "payloads").artifact)
        for pair in representative.pairs:
            for run_id in (pair.replay_run, pair.rc_run):
                pointer_file = REPO / "data" / "records" / "runs" / f"{run_id}.toml"
                verify_payload(store, load_record(pointer_file, RunPointerRecord).artifact)
        timing = next(p for p in pointers.values() if p.formulation == "no_augmentation")
        report = open_stored_report(store, timing)
        trial = next((t for t in report.summary.trials if t.number == representative.trial), None)
        if trial is None:
            msg = f"the timing-only payload holds no trial {representative.trial}"
            raise ReproductionError(msg)
        if {k: float(v) for k, v in trial.params.items()} != representative.point_params:
            msg = "the representative record's point parameters differ from the stored trial"
            raise ReproductionError(msg)
        self.trial = trial
        return (
            f"4 study payloads, dataset, raw, and {2 * len(representative.pairs)} run payloads verified; "
            f"trial {representative.trial} bound"
        )

    def data(self) -> str:
        """The cropped recovery dataset rebuilt from the raw payload has the committed digest."""
        store = _need(self.store, "data")
        raw = _need(self.raw, "data")
        dataset = _need(self.dataset, "data")
        store_dir = self.scratch / "store"
        store_dir.mkdir()  # the scratch directory was verified empty; a second data step is an error
        scratch_store = StorageRoot(store_dir, repositories=(REPO,))
        payload = store.path(raw.artifact.payload.uri, mode="read")
        scratch_store.path(raw.artifact.payload.uri, mode="write").write_bytes(payload.read_bytes())
        records_root = self.scratch / "repo"
        (records_root / "data" / "records" / "processed").mkdir(parents=True)
        scenario_file = REPO / dataset.scenario.config_path
        dataset.check_scenario(scenario_file)
        result = recover_demonstration(
            REPO / "data" / "records" / "raw" / f"{raw.artifact.artifact_id}.toml",
            scenario_file,
            DERIVE_CONFIG,
            store=scratch_store,
            records_root=records_root,
            exploratory=True,
            now=self.now,
        )
        rebuilt = result.record.artifact.payload.sha256
        if rebuilt != dataset.artifact.payload.sha256:
            committed = dataset.artifact.payload.sha256[:12]
            msg = f"rebuilt dataset digest {rebuilt[:12]} differs from the committed {committed}"
            raise ReproductionError(msg)
        self.scratch_store = scratch_store
        return f"cropped recovery dataset rebuilt with digest {rebuilt[:12]} (identical)"

    def episodes(self) -> str:
        """The anchor augmentation configuration regenerates with the committed episode digests."""
        store = _need(self.store, "episodes")
        dataset = _need(self.dataset, "episodes")
        samples = load_samples(verify_payload(store, dataset.artifact))
        self.samples = samples
        scenario = load_scenario(REPO / dataset.scenario.config_path)
        task = dataset.crop.task  # exactly the intervals the committed validation hashed
        method = _DERIVATIVE_LABELS.get(dataset.preprocessing.derivative_method)
        if method is None:
            msg = f"unsupported derivative method {dataset.preprocessing.derivative_method!r}"
            raise ReproductionError(msg)
        spec = AugmentationTrainingSpec(
            family="contractive",
            n_synthetic=int(_ANCHOR_D1["n_synthetic"]),
            sigma_rad=float(_ANCHOR_D1["sigma_rad"]),
            phi=float(_ANCHOR_D1["phi"]),
            gamma=float(_ANCHOR_D1["gamma"]),
            seed_bank=1,
            attempt_budget=4 * int(_ANCHOR_D1["n_synthetic"]),
        )
        result = generate_augmentation(
            samples.t, samples.q, task, scenario, spec.config(), derivatives=DerivativeConfig(method=method)
        )
        digest = sha256_bytes(canonical_json(result.digests()).encode("utf-8"))
        validation = json.loads((self.docs / "augmentation_validation_v2.json").read_text(encoding="utf-8"))
        rows = cast("list[dict[str, object]]", validation["configurations"])
        matches = [
            row
            for row in rows
            if all(cast("dict[str, object]", row["config"]).get(key) == value for key, value in _ANCHOR_D1.items())
        ]
        if len(matches) != 1:
            msg = f"expected one committed validation row for the anchor configuration, found {len(matches)}"
            raise ReproductionError(msg)
        committed = str(matches[0]["digests_sha256"])
        if digest != committed:
            msg = f"anchor episode digests {digest[:12]} differ from the committed {committed[:12]}"
            raise ReproductionError(msg)
        return f"anchor augmentation episodes regenerated with digest {digest[:12]} (identical)"

    def model(self) -> str:
        """The source trial's recipe refits and its full development evaluation reproduces the stored row."""
        store = _need(self.store, "model")
        trial = _need(self.trial, "model")
        protocol = load_recovery_search(REPO / "configs" / "studies" / "recovery_search_1a_no_augmentation_v1.toml")
        representative = _need(self.representative, "model")
        dataset_file = REPO / "data" / "records" / "processed" / f"{representative.dataset}.toml"
        context = RecoveryTrialContext.load(protocol, store=store, dataset_file=dataset_file, records_root=REPO)
        point = point_from_params(protocol, trial.params)
        trained = train_recovery_point(protocol, context, point)
        if isinstance(trained, str):
            msg = f"the source trial no longer trains: {trained}"
            raise ReproductionError(msg)
        recipe, _model = trained
        evaluation = evaluate_recovery_point(protocol, context, point)
        differences: list[str] = []
        worst = compare_evidence("objective", trial.value, evaluation.objective, differences)
        stored: dict[tuple[str, str], float] = {}
        index = 0
        while f"components.{index}.kind" in trial.labels:
            prefix = f"components.{index}"
            ratio = trial.metrics.get(prefix + ".gap_ratio")
            if ratio is not None:
                stored[(trial.labels[prefix + ".scenario_id"], trial.labels[prefix + ".tracker"])] = ratio
            index += 1
        rebuilt = {(c.scenario_id, c.tracker): c.gap_ratio for c in evaluation.components if c.gap_ratio is not None}
        if set(stored) != set(rebuilt):
            msg = f"re-evaluation covers {len(rebuilt)} ratio pairs, the stored trial {len(stored)}"
            raise ReproductionError(msg)
        for key, value in stored.items():
            worst = max(worst, compare_evidence(f"gap_ratio[{key}]", value, rebuilt[key], differences))
        if differences:
            msg = f"{len(differences)} categorical differences (first: {differences[0]})"
            raise ReproductionError(msg)
        if worst > self.tolerance:
            msg = f"largest deviation {worst:.3e} exceeds the tolerance {self.tolerance:.3e}"
            raise ReproductionError(msg)
        self.protocol, self.context, self.recipe = protocol, context, recipe
        self.max_deviation = worst if self.max_deviation is None else max(self.max_deviation, worst)
        return (
            f"recipe refitted (fit RMSE {recipe.fit.rmse:.6g} rad); trial {trial.number} re-evaluated over "
            f"{len(evaluation.components)} pairs; largest deviation {worst:.3e}"
        )

    def pairs(self) -> str:
        """The representative pairs rerun under the reproduction label and match the committed runs."""
        if self.exploratory:
            msg = "the pair rerun requires a clean checkout; run without --exploratory from a clean tree"
            raise ReproductionError(msg)
        store = _need(self.store, "pairs")
        scratch_store = _need(self.scratch_store, "pairs")
        context = _need(self.context, "pairs")
        recipe = _need(self.recipe, "pairs")
        representative = _need(self.representative, "pairs")
        trial = _need(self.trial, "pairs")
        protocol = _need(self.protocol, "pairs")
        point = point_from_params(protocol, trial.params)
        estimator = point.esn.estimator(max_dt_ratio=protocol.max_dt_ratio).config(context.scenario.timing.dt)
        payload = store.path(context.dataset.artifact.payload.uri, mode="read")
        scratch_store.path(context.dataset.artifact.payload.uri, mode="write").write_bytes(payload.read_bytes())
        by_id = {scenario.scenario_id: scenario for scenario in context.scenarios}
        worst = 0.0
        differences: list[str] = []
        for pair in representative.pairs:
            scenario = by_id[pair.scenario_id]
            rebuilt = run_recovery_pair(
                context.scenario,
                context.scenario_file,
                context.dataset,
                context.reference,
                recipe,
                context.trackers[pair.tracker],
                store=scratch_store,
                warmup=WarmupConfig(point.warmup_s),
                exploratory=False,
                estimator=estimator,
                initial_q=scenario.initial_q(context.dataset.q0_ref),
                force=scenario.pulse,
                now=self.now,
                command=_REPRODUCTION_COMMAND,
            )
            committed_rc = load_run(
                store, load_record(REPO / "data" / "records" / "runs" / f"{pair.rc_run}.toml", RunPointerRecord)
            )
            committed_replay = load_run(
                store, load_record(REPO / "data" / "records" / "runs" / f"{pair.replay_run}.toml", RunPointerRecord)
            )
            if rebuilt.rc.run.arrays.specs() != committed_rc.arrays.specs():
                differences.append(f"{pair.scenario_id}/{pair.tracker}: RC array digests differ")
            if rebuilt.replay.run.arrays.specs() != committed_replay.arrays.specs():
                differences.append(f"{pair.scenario_id}/{pair.tracker}: replay array digests differ")
            committed_metrics = None if pair.recovery is None else to_mapping(pair.recovery)
            rebuilt_metrics = None if rebuilt.recovery is None else to_mapping(rebuilt.recovery)
            if (committed_metrics is None) != (rebuilt_metrics is None):
                differences.append(f"{pair.scenario_id}/{pair.tracker}: recovery metrics presence differs")
            elif committed_metrics is not None and rebuilt_metrics is not None:
                worst = max(worst, compare_evidence("recovery", committed_metrics, rebuilt_metrics, differences))
        if differences:
            msg = f"{len(differences)} categorical differences (first: {differences[0]})"
            raise ReproductionError(msg)
        if worst > self.tolerance:
            msg = f"largest metric deviation {worst:.3e} exceeds the tolerance {self.tolerance:.3e}"
            raise ReproductionError(msg)
        self.max_deviation = worst if self.max_deviation is None else max(self.max_deviation, worst)
        return (
            f"{2 * len(representative.pairs)} runs rerun under the reproduction label; array digests identical; "
            f"largest metric deviation {worst:.3e} (tolerance {self.tolerance:.3e})"
        )

    def report(self) -> str:
        """The ablation, freeze, and recovery reports re-render identically from the committed evidence."""
        store = _need(self.store, "report")
        ablation = _need(self.ablation, "report")
        freeze = _need(self.freeze, "report")
        representative = _need(self.representative, "report")
        if render_ablation_markdown(ablation) != (self.docs / "development_ablation_v2.md").read_text(encoding="utf-8"):
            msg = "the rendered ablation differs from the committed development_ablation_v2.md"
            raise ReproductionError(msg)
        if render_freeze_markdown(freeze) != (self.docs / "model_freeze_v2.md").read_text(encoding="utf-8"):
            msg = "the rendered freeze record differs from the committed model_freeze_v2.md"
            raise ReproductionError(msg)
        inputs = build_report_inputs(self.docs, store=store, records_root=REPO)
        rendered = render_recovery_report(inputs, plots=list(PLOT_FILES), animations=animation_names(representative))
        if rendered != (self.docs / "recovery_report_v1.md").read_text(encoding="utf-8"):
            msg = "the rendered report differs from the committed recovery_report_v1.md"
            raise ReproductionError(msg)
        return "ablation, freeze, and recovery reports re-rendered identically"

    def step(self, name: str) -> Check:
        """Run one step and record its outcome."""
        action = cast("Callable[[], str]", getattr(self, name))
        t0 = time.perf_counter()
        try:
            detail = action()
        except (ReproductionError, StorageError, FileNotFoundError, ValueError, RuntimeError) as exc:
            return Check(name, ok=False, detail=f"{type(exc).__name__}: {exc}", elapsed_s=time.perf_counter() - t0)
        return Check(name, ok=True, detail=detail, elapsed_s=time.perf_counter() - t0)


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
    tolerance: float = 0.0,
    keep_going: bool = False,
    exploratory: bool = False,
    store: StorageRoot | None = None,
    docs: Path = DOCS,
    now: datetime | None = None,
) -> RecoveryReproduction:
    """Run every reproduction step; ``scratch`` receives the rebuilt store and records (outside the repository)."""
    if not (math.isfinite(tolerance) and tolerance >= 0.0):
        msg = f"tolerance must be a finite non-negative number, got {tolerance!r}"
        raise ValueError(msg)
    started = time.perf_counter()
    started_at = (now or datetime.now(tz=UTC)).isoformat()
    reproducer = Reproducer(prepare_scratch(scratch), tolerance, exploratory, store, docs, now)
    checks: list[Check] = []
    for name in STEPS:
        check = reproducer.step(name)
        checks.append(check)
        if not check.ok and not keep_going:
            break
    return RecoveryReproduction(
        started_at=started_at,
        checks=tuple(checks),
        inputs=dict(reproducer.inputs),
        environment=_environment(),
        max_deviation=reproducer.max_deviation,
        elapsed_s=time.perf_counter() - started,
    )


def audit_markdown(
    result: RecoveryReproduction, *, command: str, auditor: str = "(to be filled by the auditor)"
) -> str:
    """A compact audit note of one reproduction invocation."""
    machine = f"{result.environment.get('platform', '')} / {result.environment.get('machine', '')}"
    lines = [
        "# Task 1-a recovery reproduction",
        "",
        f"- Started: {result.started_at}; elapsed {result.elapsed_s:.1f} s; ok: {result.ok}.",
        f"- Command: `{command}`",
        f"- Executor machine: `{machine}`.",
        f"- Auditor: {auditor} (cross-machine execution preferred; plan M3R-020).",
        f"- Largest metric deviation: {result.max_deviation!r}.",
        "",
        "| step | ok | elapsed (s) | detail |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {c.name} | {c.ok} | {c.elapsed_s:.1f} | {c.detail} |" for c in result.checks)
    lines += ["", "## Inputs", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in sorted(result.inputs.items()))
    lines += ["", "## Environment", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in sorted(result.environment.items()))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point (exit status 1 when any step fails)."""
    parser = argparse.ArgumentParser(description="Reproduce the task 1-a recovery result from the committed records.")
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
        help="allow a dirty worktree for the other steps (the pair rerun then fails clearly)",
    )
    args = parser.parse_args(argv)
    tolerance = float(args.tolerance)
    if not (math.isfinite(tolerance) and tolerance >= 0.0):
        parser.error(f"--tolerance must be a finite non-negative number, got {args.tolerance!r}")
    ensure_single_thread()  # before rclib is imported and provenance is collected
    scratch = (
        Path(tempfile.mkdtemp(prefix="arm-rc-ctrl-reproduce-recovery-")) if args.scratch is None else Path(args.scratch)
    )
    result = reproduce(
        scratch=scratch,
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
        invoked = command_line("arm_rc_ctrl.experiments.reproduce_recovery", sys.argv[1:] if argv is None else argv)
        Path(args.audit).write_text(audit_markdown(result, command=invoked), encoding="utf-8")
    print(text)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
