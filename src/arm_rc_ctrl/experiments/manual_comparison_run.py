# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-006: the five-arm comparison at the frozen configurations, under the shared ceiling.

The comparison evaluates S, M10, R10, C10 and Replay at each configuration the
search froze, over the 65 locked cases under both fixed trackers. Its work is
divided into **units**, each run by its own worker process:

* per configuration, the ten parents' **replay banks** first, since every
  learned model of a parent is paired against its bank;
* then the 31 **learned models**, in the closed study's arm order.

Configurations are completed one at a time, in selection order. The search and
the comparison share one elapsed-time and storage ceiling: the search's
verified spend is carried in from the freeze, and the comparison adds its own.
A unit is never started without a real deadline, and scheduling stops at the
ceiling with the evidence that exists retained, so a stopped third
configuration is reported as incomplete rather than dropped or traded for a
larger allowance.

The resume protocol is the search's. A unit is **reserved** with the identity
its evidence will be stored under before its worker starts; each attempt is
opened on disk before it runs and charged even when its parent is killed; a
worker that writes no report has been interrupted and is resumed under the same
unit; bytes are discovered from what the unit's directories hold, never
reported. A worker's report is not evidence: the parent checks the pointer
against the installed manifest, then serves the unit back through the
evaluation's own resume path, which verifies the manifest, the fits, the runs
and the replay pairing against inputs the parent reconstructs, and recounts
the verdicts over the complete scope.

The freeze is loaded only through `load_verified_freeze`, and each worker
checks the freeze file's bytes against the digest its parent verified.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments import manual_search
from arm_rc_ctrl.experiments.manual_evaluation import (
    PHASE_BUILD_BANK,
    PHASE_FIT,
    PHASE_SERVE_FIT,
    PHASE_SWEEP,
    PROGRESS_FILE,
    REPORTS_PREFIX,
    RUN_CLAIMS_FILE,
    EvidenceIntegrityError,
    bank_identity,
    evaluation_scenarios,
    load_manual_evaluation_config,
    load_manual_model_evidence,
    load_manual_replay_bank,
    model_uri,
    prepare_runner,
    read_progress_runs,
    read_run_claims,
    replay_bank_uri,
    stored_manifest,
    stored_reference,
)
from arm_rc_ctrl.experiments.manual_fits import ManualFitStore, cache_uri
from arm_rc_ctrl.experiments.manual_sampled import LEARNED_ARMS, sampled_configuration, sampled_entry
from arm_rc_ctrl.experiments.manual_search import load_manual_search
from arm_rc_ctrl.experiments.manual_search_freeze import freeze_digest, load_verified_freeze, read_freeze
from arm_rc_ctrl.experiments.manual_search_run import (
    HEADROOM_S,
    BudgetLedger,
    SearchInputs,
    SearchInvocation,
    TrialSpend,
    TrialTiming,
    read_record,
    unpublished_bytes,
    write_record,
)
from arm_rc_ctrl.experiments.manual_study import ASSIGNMENTS
from arm_rc_ctrl.experiments.perturbations import load_development_robustness
from arm_rc_ctrl.provenance import ArtifactReference, sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_evaluation import (
        ManualEvaluationRunner,
        ManualModelEvidence,
        ManualPairRecord,
        ManualReplayBank,
        ManualRunConditions,
    )
    from arm_rc_ctrl.experiments.manual_recipes import ManualArmSpec
    from arm_rc_ctrl.experiments.manual_search import ManualSearchBudget, ManualSearchProtocol
    from arm_rc_ctrl.experiments.manual_search_freeze import FrozenConfiguration, ManualSearchFreeze
    from arm_rc_ctrl.experiments.manual_study import StudyConfiguration, StudyModel

__all__ = [
    "COMPARISON_PREFIX",
    "ComparisonUnit",
    "SharedLedger",
    "UnitOutcome",
    "UnitReservation",
    "UnitResult",
    "comparison_scope_mismatches",
    "comparison_units",
    "evaluate_unit",
    "run_comparison",
    "shared_ledger",
    "shared_stopped",
    "spawn_unit",
    "unit_directory",
]

_MODULE: Final = "arm_rc_ctrl.experiments.manual_comparison_run"
_GIB: Final = 1024**3
_SHA256_HEX: Final = 64
_RUNS_BUCKET: Final = "runs"
COMPARISON_PREFIX: Final = "armrc://reports/task_1a_manual_search/comparison"
"""Where each unit's reservation, report, spend and outcome are retained, beside the search's trials."""
REPLAY: Final = "replay"
MODEL: Final = "model"

type UnitKind = Literal["replay", "model"]
REPLAY_KIND: Final[UnitKind] = "replay"
MODEL_KIND: Final[UnitKind] = "model"


@dataclass(frozen=True)
class ComparisonUnit:
    """One unit of the comparison: a parent's replay bank, or one learned model, at one configuration."""

    number: int
    rank: int
    """The configuration's rank in the freeze, which is also the order configurations are completed in."""
    kind: UnitKind
    label: str
    """A parent (``D01``) for a replay bank, an arm label (``S/D01``, ``M10``) for a model."""

    @property
    def slug(self) -> str:
        """The unit's directory name: its number first, so a listing is the schedule."""
        return f"unit-{self.number:03d}-c{self.rank}-{self.kind}-{self.label.replace('/', '-')}"


def comparison_units(n_configurations: int) -> tuple[ComparisonUnit, ...]:
    """Every unit, in the order they run: configuration by configuration, each bank before any model."""
    units: list[ComparisonUnit] = []
    for rank in range(1, n_configurations + 1):
        banks: list[tuple[UnitKind, str]] = [(REPLAY_KIND, parent) for parent in ASSIGNMENTS]
        models: list[tuple[UnitKind, str]] = [(MODEL_KIND, arm.label) for arm in LEARNED_ARMS]
        labels = banks + models
        units += [
            ComparisonUnit(number=len(units) + index, rank=rank, kind=kind, label=label)
            for index, (kind, label) in enumerate(labels)
        ]
    return tuple(units)


def unit_directory(store: StorageRoot, unit: ComparisonUnit) -> Path:
    """Where one unit's records are kept, resolved without creating anything."""
    return store.root / ArtifactUri.parse(f"{COMPARISON_PREFIX}/{unit.slug}/reservation.json").relative_path.parent


# --- the records ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitReservation:
    """The parent's commitment to one unit, written before its worker starts."""

    number: int
    rank: int
    kind: UnitKind
    label: str
    configuration: str
    freeze_sha256: str
    identity: str
    """The identity the unit's evidence is stored under: the bank's, or the model evidence's."""
    fit_identity: str | None
    """The fit a model unit is keyed by; ``None`` for a replay bank, which fits nothing."""

    def __post_init__(self) -> None:
        """A reservation names a real unit, its freeze and the identities its work will produce."""
        digests = [self.freeze_sha256, self.identity, *([] if self.fit_identity is None else [self.fit_identity])]
        if (self.kind == MODEL) != (self.fit_identity is not None) or any(len(d) != _SHA256_HEX for d in digests):
            msg = f"unit {self.number}: a reservation names its freeze and identities, got {self}"
            raise ValueError(msg)


@dataclass(frozen=True)
class UnitResult:
    """What a worker reports about one unit: where its evidence was installed and what it counted."""

    number: int
    identity: str
    pairs: int
    completed: int
    infeasible: int
    unexecuted: int
    seconds: float
    evidence: ArtifactReference | None = None
    failure: str | None = None
    timing: TrialTiming | None = None

    def __post_init__(self) -> None:
        """Counts add up, and a unit without evidence says why."""
        if min(self.pairs, self.completed, self.infeasible, self.unexecuted) < 0 or self.seconds < 0.0:
            msg = f"unit {self.number}: a report is non-negative, got {self}"
            raise ValueError(msg)
        if self.completed + self.infeasible + self.unexecuted != self.pairs:
            msg = f"unit {self.number}: {self.completed}+{self.infeasible}+{self.unexecuted} is not {self.pairs} pairs"
            raise ValueError(msg)
        if (self.failure is None) == (self.evidence is None):
            msg = f"unit {self.number}: a report carries evidence or a failure, and not both"
            raise ValueError(msg)


@dataclass(frozen=True)
class UnitOutcome:
    """A finalized unit: what the parent verified and what it charged."""

    number: int
    state: Literal["complete", "failed", "unavailable"]
    """``failed``: the unit's own work was refused; ``unavailable``: a model whose parent bank is missing."""
    identity: str
    pairs: int
    completed: int
    infeasible: int
    unexecuted: int
    seconds: float
    stored_bytes: int
    evidence: str | None = None
    failure: str | None = None
    worker_seconds: float | None = None
    verify_seconds: float | None = None

    def __post_init__(self) -> None:
        """A complete unit names its evidence; any other names its reason; spend is non-negative."""
        if (self.state == "complete") != (self.evidence is not None) or (self.state == "complete") == (
            self.failure is not None
        ):
            msg = f"unit {self.number}: state {self.state!r} with evidence {self.evidence!r}, failure {self.failure!r}"
            raise ValueError(msg)
        measured = (self.worker_seconds, self.verify_seconds)
        if self.seconds < 0.0 or self.stored_bytes < 0 or any(m is not None and m < 0.0 for m in measured):
            msg = f"unit {self.number}: spend is never negative"
            raise ValueError(msg)


# --- the shared ceiling --------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedLedger:
    """The search's verified spend and the comparison's, against the one ceiling they share."""

    search: BudgetLedger
    units: int
    """Finalized comparison units."""
    seconds: float
    """Comparison seconds: every finalized unit's charge and every pending unit's measured attempts."""
    stored_bytes: int
    """Comparison bytes, including pending units' partial work and payloads staged but not yet claimed."""

    @property
    def total_seconds(self) -> float:
        """Elapsed execution against the shared ceiling."""
        return self.search.seconds + self.seconds

    @property
    def total_bytes(self) -> int:
        """Stored bytes against the shared ceiling."""
        return self.search.stored_bytes + self.stored_bytes


def _read_optional[T](path: Path, schema: type[T]) -> T | None:
    return read_record(path, schema) if path.is_file() else None


def shared_ledger(store: StorageRoot, freeze: ManualSearchFreeze, units: Sequence[ComparisonUnit]) -> SharedLedger:
    """The spend, derived from the retained unit records and the verified freeze, never from a counter."""
    finalized, seconds, stored = 0, 0.0, 0
    for unit in units:
        directory = unit_directory(store, unit)
        outcome = _read_optional(directory / "outcome.json", UnitOutcome)
        if outcome is not None:
            finalized, seconds, stored = finalized + 1, seconds + outcome.seconds, stored + outcome.stored_bytes
            continue
        spend = _read_optional(directory / "spend.json", TrialSpend)
        if spend is not None:
            seconds += spend.seconds
        reservation = _read_optional(directory / "reservation.json", UnitReservation)
        if reservation is not None:
            stored += unit_bytes(store, reservation)
    # Staged payloads have no owner yet, so they are charged here, once, whatever the units are doing.
    return SharedLedger(
        search=freeze.ledger, units=finalized, seconds=seconds, stored_bytes=stored + unpublished_bytes(store)
    )


def _remaining_seconds(ledger: SharedLedger, budget: ManualSearchBudget) -> float:
    return budget.hours * 3600.0 - ledger.total_seconds - HEADROOM_S


def shared_stopped(ledger: SharedLedger, budget: ManualSearchBudget) -> list[str]:
    """Why no further unit may start: the shared time or storage cap, or no allowance beyond the headroom."""
    found: list[str] = []
    if ledger.total_seconds >= budget.hours * 3600.0:
        found.append(f"the shared {budget.hours:g} h cap is spent ({ledger.total_seconds / 3600.0:.2f} h)")
    if ledger.total_bytes >= budget.gib * _GIB:
        found.append(f"the shared {budget.gib:g} GiB cap is spent ({ledger.total_bytes / _GIB:.2f} GiB)")
    if not found and _remaining_seconds(ledger, budget) <= 0.0:
        found.append(f"the shared {budget.hours:g} h cap leaves no allowance beyond the persistence headroom")
    return found


def unit_bytes(store: StorageRoot, reservation: UnitReservation) -> int:
    """Every byte a unit has left in the store, discovered from its directories rather than reported.

    A model unit owns its fit cache and its evidence directory; a replay unit
    owns its bank directory. Each owns the runs its claims and its progress
    name: ownership of a run is recorded before its payload is published, so a
    run whose progress entry never landed is charged too, and a record that
    cannot be read is refused rather than counted as nothing.
    """
    if reservation.kind == MODEL:
        evidence = store.root / ArtifactUri.parse(f"{model_uri(reservation.identity)}/x").relative_path.parent
        directories = [store.root / cache_uri(cast("str", reservation.fit_identity)).relative_path, evidence]
    else:
        evidence = store.root / ArtifactUri.parse(f"{_bank_uri(reservation)}/x").relative_path.parent
        directories = [evidence]
    claims, progress = evidence / RUN_CLAIMS_FILE, evidence / PROGRESS_FILE
    if claims.is_file():
        directories += [
            store.root / _RUNS_BUCKET / run for run in read_run_claims(claims, identity=reservation.identity)
        ]
    if progress.is_file():
        directories += [
            (store.root / ArtifactUri.parse(uri).relative_path).parent for uri in read_progress_runs(progress)
        ]
    total = 0
    for directory in {path.resolve() for path in directories}:
        if directory.is_dir():
            total += sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
    return total


def _bank_uri(reservation: UnitReservation) -> str:
    """A bank's directory: the evaluation names it by the bank's identity, which the reservation carries."""
    return f"{REPORTS_PREFIX}/replay/{reservation.identity}"


# --- the scope ------------------------------------------------------------------------------------


def comparison_scope_mismatches(
    scenario_ids: Sequence[str], locked_ids: Sequence[str], trackers: Sequence[str], protocol: ManualSearchProtocol
) -> list[str]:
    """Why a runner's scope is not the comparison's, if it is not: every locked case, under both trackers.

    Checked where a worker actually receives its scope, as the search checks its
    nominal-only scope, so a narrowed or widened comparison cannot be produced
    by a worker invoked any other way.
    """
    found: list[str] = []
    if tuple(scenario_ids) != tuple(locked_ids):
        found.append(
            f"the comparison evaluates every locked case in order, got {len(scenario_ids)} of {len(locked_ids)}"
        )
    if len(locked_ids) != manual_search.REQUIRED_SCENARIOS or protocol.comparison.scenarios != len(locked_ids):
        found.append(f"the comparison has {protocol.comparison.scenarios} cases, the locked set {len(locked_ids)}")
    if set(trackers) != set(protocol.fixed.trackers) or len(trackers) != len(protocol.fixed.trackers):
        found.append(f"the comparison runs under {protocol.fixed.trackers}, got {tuple(trackers)}")
    return found


# --- the worker ----------------------------------------------------------------------------------


def _chosen(freeze: ManualSearchFreeze, unit: ComparisonUnit) -> FrozenConfiguration:
    matches = [item for item in freeze.chosen if item.rank == unit.rank]
    if len(matches) != 1:
        msg = f"the freeze holds no configuration of rank {unit.rank}"
        raise ValueError(msg)
    return matches[0]


def _configuration(protocol: ManualSearchProtocol, chosen: FrozenConfiguration) -> StudyConfiguration:
    configuration = sampled_configuration(protocol, chosen.point, trial=chosen.trial)
    if configuration.label != chosen.configuration:
        msg = f"the freeze names {chosen.configuration!r}, its point builds {configuration.label!r}"
        raise ValueError(msg)
    return configuration


def _arm(label: str) -> ManualArmSpec:
    for arm in LEARNED_ARMS:
        if arm.label == label:
            return arm
    msg = f"no learned arm {label!r}"
    raise ValueError(msg)


def _counts(pairs: Sequence[ManualPairRecord]) -> tuple[int, int, int]:
    return (
        sum(1 for pair in pairs if pair.status == "completed"),
        sum(1 for pair in pairs if pair.status == "infeasible"),
        sum(1 for pair in pairs if pair.status == "unexecuted"),
    )


def _timing(runner: ManualEvaluationRunner, label: str, *, prepare_seconds: float) -> TrialTiming:
    phases = [item for item in runner.phase_timings if item.label == label]
    model = runner.model_timings.get(label)
    runs = runner.run_timings
    return TrialTiming(
        prepare_seconds=prepare_seconds,
        fit_seconds=math.fsum(item.seconds for item in phases if item.phase in (PHASE_FIT, PHASE_SERVE_FIT)),
        fit_cache_hit=model is not None and model.fit_cache_hit,
        sweep_seconds=math.fsum(item.seconds for item in phases if item.phase in (PHASE_SWEEP, PHASE_BUILD_BANK)),
        simulate_seconds=math.fsum(run.simulate_seconds for run in runs),
        persist_seconds=math.fsum(run.persist_seconds for run in runs),
        run_bytes=sum(run.run_bytes for run in runs),
        simulated_runs=len(runs),
    )


def evaluate_unit(
    protocol: ManualSearchProtocol,
    *,
    freeze_file: Path,
    freeze_sha256: str,
    number: int,
    root: Path,
    argv: Sequence[str],
    exploratory: bool = False,
) -> UnitResult:
    """Build one bank or evaluate one model at its frozen configuration, in this (worker) process.

    The freeze is read from its file only after its bytes are checked against
    the digest the parent verified, and the configuration is rebuilt from the
    frozen point. The runner's scope is checked here, where it is received. A
    candidate the learner refuses is a failed unit; a fault in the store is not
    a verdict and propagates, leaving no report, so the unit stays recoverable.
    """
    if sha256_file(freeze_file) != freeze_sha256:
        msg = f"{freeze_file} is not the freeze the parent verified ({freeze_sha256[:12]})"
        raise ValueError(msg)
    freeze = read_freeze(freeze_file)
    units = comparison_units(freeze.n_configurations)
    if not 0 <= number < len(units):
        msg = f"there is no comparison unit {number}"
        raise ValueError(msg)
    unit = units[number]
    configuration = _configuration(protocol, _chosen(freeze, unit))
    args = argparse.Namespace(
        study=str(protocol.study),
        evaluation=str(protocol.comparison.evaluation),
        exploratory=exploratory,
        argv=list(argv),
    )
    preparing = time.perf_counter()
    prepared = prepare_runner(args, role="worker", root=root, module=_MODULE, sampled=(configuration,))
    prepare_seconds = time.perf_counter() - preparing
    runner = prepared.runner
    cutoffs = (configuration.velocity_cutoff_hz, configuration.acceleration_cutoff_hz)
    conditions = runner.conditions(configuration.warmup_s, cutoffs)
    locked = locked_scenario_ids(protocol.comparison.evaluation)
    mismatches = comparison_scope_mismatches(conditions.scenario_ids, locked, conditions.tracker_order, protocol)
    if mismatches:
        raise ValueError("; ".join(mismatches))
    started = time.perf_counter()
    entry = None if unit.kind == REPLAY else sampled_entry(prepared.context.manifest, configuration, _arm(unit.label))
    if entry is None:
        identity, timing_label = bank_identity(conditions, unit.label), unit.label
    else:
        identity, timing_label = runner.model_identity(entry, warmup_s=configuration.warmup_s), entry.label
    try:
        # Only the learner refusing a candidate is a result. It is asked first, on its own; everything
        # after it -- serving stored evidence, building a bank, the sweep -- is infrastructure, and a
        # refusal there propagates, leaving no report, so the unit stays recoverable. A bank has no
        # learner at all, so nothing about building one is ever a verdict.
        if entry is not None:
            ManualFitStore(runner.store).fit_or_load(entry, runner.inputs)
    except EvidenceIntegrityError:
        raise
    except ValueError as error:
        return UnitResult(
            number=number,
            identity=identity,
            pairs=0,
            completed=0,
            infeasible=0,
            unexecuted=0,
            seconds=time.perf_counter() - started,
            failure=f"{type(error).__name__}: {str(error)[:300]}",
            timing=_timing(runner, timing_label, prepare_seconds=prepare_seconds),
        )
    if entry is None:
        evidence: ManualReplayBank | ManualModelEvidence = runner.replay_bank(
            unit.label, warmup_s=configuration.warmup_s, replay_cutoffs=cutoffs
        )
    else:
        evidence = runner.evaluate(entry, warmup_s=configuration.warmup_s)
    uri = _evidence_uri(unit, conditions, identity)
    stored = stored_manifest(runner.store, uri)
    if stored is None:
        msg = f"unit {number}: its evidence was not installed under {uri}"
        raise ValueError(msg)
    completed, infeasible, unexecuted = _counts(evidence.pairs)
    return UnitResult(
        number=number,
        identity=evidence.identity,
        pairs=len(evidence.pairs),
        completed=completed,
        infeasible=infeasible,
        unexecuted=unexecuted,
        seconds=time.perf_counter() - started,
        evidence=stored_reference(stored, runner.store),
        timing=_timing(runner, timing_label, prepare_seconds=prepare_seconds),
    )


def locked_scenario_ids(evaluation: Path) -> tuple[str, ...]:
    """The evaluation's locked cases, derived from its configuration rather than read from a runner."""
    config = load_manual_evaluation_config(evaluation)
    cases = evaluation_scenarios(load_development_robustness(config.development), load_manual_scenario(config.scenario))
    return tuple(case.scenario_id for case in cases)


def installed_manifest(store: StorageRoot, directory_uri: str) -> Path | None:
    """The manifest installed under ``directory_uri``, looked up without creating the directory it would be in."""
    directory = store.root / ArtifactUri.parse(f"{directory_uri}/manifest.json").relative_path.parent
    return stored_manifest(store, directory_uri) if directory.is_dir() else None


def _evidence_uri(unit: ComparisonUnit, conditions: ManualRunConditions, identity: str) -> str:
    return replay_bank_uri(conditions, unit.label) if unit.kind == REPLAY else model_uri(identity)


# --- the parent ----------------------------------------------------------------------------------


class ComparisonContext:
    """What the parent derives every expectation from: the verified freeze and a runner bound to the study.

    The runner is the evaluation's own, prepared through the path a worker
    uses, so reservations are keyed exactly as a worker's evidence is, and a
    unit is verified by serving it back through the resume path that re-checks
    manifests, fits, runs and replay pairing against inputs derived here.
    """

    def __init__(
        self,
        protocol: ManualSearchProtocol,
        protocol_file: Path,
        freeze_file: Path,
        *,
        root: Path,
        exploratory: bool = False,
        module: str = _MODULE,
        argv: Sequence[str] = ("compare",),
    ) -> None:
        """Verify the freeze against the search, then bind a runner to its configurations.

        ``module`` and ``argv`` name the command actually invoked, which the
        runner's provenance and every record derived through it cite: a
        derivation or an audit is not the comparison run.
        """
        self.protocol = protocol
        self.protocol_file = protocol_file
        self.freeze_file = freeze_file
        self.store = open_storage()
        self.freeze = load_verified_freeze(
            freeze_file, protocol, self.store, inputs=SearchInputs(protocol, root=root, exploratory=exploratory)
        )
        self.freeze_sha256 = freeze_digest(self.freeze)
        self.configurations = {item.rank: _configuration(protocol, item) for item in self.freeze.chosen}
        args = argparse.Namespace(
            study=str(protocol.study),
            evaluation=str(protocol.comparison.evaluation),
            exploratory=exploratory,
            argv=list(argv),
        )
        prepared = prepare_runner(
            args, role="main", root=root, module=module, sampled=tuple(self.configurations.values())
        )
        self.manifest = prepared.context.manifest
        self.runner = prepared.runner
        self.units = tuple(
            unit for unit in comparison_units(self.freeze.n_configurations) if unit.rank in self.configurations
        )

    def conditions(self, rank: int) -> ManualRunConditions:
        """The conditions every run of one configuration is keyed by."""
        configuration = self.configurations[rank]
        cutoffs = (configuration.velocity_cutoff_hz, configuration.acceleration_cutoff_hz)
        return self.runner.conditions(configuration.warmup_s, cutoffs)

    def entry(self, unit: ComparisonUnit) -> StudyModel:
        """The learned model a model unit evaluates, bound exactly as the frozen study binds its own."""
        return sampled_entry(self.manifest, self.configurations[unit.rank], _arm(unit.label))

    def reservation(self, unit: ComparisonUnit) -> UnitReservation:
        """The unit's reservation, derived from the verified freeze and the study alone."""
        configuration = self.configurations[unit.rank]
        if unit.kind == REPLAY:
            identity, fit_identity = bank_identity(self.conditions(unit.rank), unit.label), None
        else:
            entry = self.entry(unit)
            identity = self.runner.model_identity(entry, warmup_s=configuration.warmup_s)
            fit_identity = entry.fit_identity
        return UnitReservation(
            number=unit.number,
            rank=unit.rank,
            kind=unit.kind,
            label=unit.label,
            configuration=configuration.label,
            freeze_sha256=self.freeze_sha256,
            identity=identity,
            fit_identity=fit_identity,
        )

    def installed_evidence(self, unit: ComparisonUnit, reservation: UnitReservation) -> ArtifactReference | None:
        """The manifest installed under the unit's reserved identity, or ``None`` when there is none."""
        uri = _evidence_uri(unit, self.conditions(unit.rank), reservation.identity)
        stored = installed_manifest(self.store, uri)
        return None if stored is None else stored_reference(stored, self.store)

    def installed_counts(self, unit: ComparisonUnit, reservation: UnitReservation) -> tuple[int, int, int, int] | None:
        """Pairs, completed, infeasible and unexecuted, recounted from the manifest installed under the identity."""
        uri = _evidence_uri(unit, self.conditions(unit.rank), reservation.identity)
        stored = installed_manifest(self.store, uri)
        if stored is None:
            return None
        loaded = load_manual_replay_bank(stored) if unit.kind == REPLAY else load_manual_model_evidence(stored)
        return (len(loaded.pairs), *_counts(loaded.pairs))

    def bank_unit(self, unit: ComparisonUnit) -> ComparisonUnit | None:
        """The replay unit a model is paired against, or ``None`` for the all-ten arm."""
        if unit.kind == REPLAY:
            return None
        assignment = _arm(unit.label).assignment
        if assignment is None:
            return None
        return next(u for u in self.units if u.rank == unit.rank and u.kind == REPLAY and u.label == assignment)


def spawn_unit(command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[bytes]:
    """Run one unit's worker as its own process, with a real deadline; the tests replace this seam."""
    if timeout <= 0.0:
        msg = f"a worker is never started without a deadline, got {timeout}"
        raise ValueError(msg)
    return subprocess.run(list(command), check=False, capture_output=True, timeout=timeout)


def unit_command(
    context: ComparisonContext, unit: ComparisonUnit, *, output: Path, root: Path, exploratory: bool
) -> list[str]:
    """The worker invocation for one unit: the freeze, the digest it was verified at, and the unit's number."""
    return [
        sys.executable,
        "-m",
        _MODULE,
        "evaluate-unit",
        "--protocol",
        str(context.protocol_file),
        "--freeze",
        str(context.freeze_file),
        "--freeze-sha256",
        # The digest the verified load established: the file's bytes equal the canonical freeze.
        context.freeze_sha256,
        "--unit",
        str(unit.number),
        "--output",
        str(output),
        "--root",
        str(root),
        *(["--exploratory"] if exploratory else []),
    ]


def verified_unit(
    context: ComparisonContext,
    unit: ComparisonUnit,
    reservation: UnitReservation,
    result: UnitResult,
    *,
    seconds: float,
) -> UnitOutcome:
    """Turn a worker's report into a finalized unit, from the evidence the parent serves back itself."""
    if result.number != unit.number or result.identity != reservation.identity:
        msg = f"unit {unit.number}: the report names unit {result.number} at {result.identity[:12]}"
        raise ValueError(msg)
    store = context.store
    if result.failure is not None:
        return UnitOutcome(
            number=unit.number,
            state="failed",
            identity=reservation.identity,
            pairs=result.pairs,
            completed=result.completed,
            infeasible=result.infeasible,
            unexecuted=result.unexecuted,
            seconds=seconds,
            stored_bytes=unit_bytes(store, reservation),
            failure=result.failure,
        )
    conditions = context.conditions(unit.rank)
    installed = context.installed_evidence(unit, reservation)
    if installed is None or installed != result.evidence:
        msg = f"unit {unit.number}: the reported evidence is not the manifest installed under its identity"
        raise ValueError(msg)
    served = serve_unit(context, unit, reservation)
    stored = installed
    complaints = [
        f"the served evidence is {served.identity[:12]}" if served.identity != reservation.identity else None,
        (
            f"the evidence holds {len(served.pairs)} pairs, not the {len(conditions.pairs)} of every case and tracker"
            if sorted((p.scenario_id, p.tracker) for p in served.pairs) != sorted(conditions.pairs)
            else None
        ),
        (
            "the reported counts are not the evidence's"
            if (len(served.pairs), *_counts(served.pairs))
            != (result.pairs, result.completed, result.infeasible, result.unexecuted)
            else None
        ),
    ]
    found = [item for item in complaints if item]
    if found:
        raise ValueError(f"unit {unit.number}: " + "; ".join(found))
    completed, infeasible, unexecuted = _counts(served.pairs)
    return UnitOutcome(
        number=unit.number,
        state="complete",
        identity=reservation.identity,
        pairs=len(served.pairs),
        completed=completed,
        infeasible=infeasible,
        unexecuted=unexecuted,
        seconds=seconds,
        stored_bytes=unit_bytes(store, reservation),
        evidence=stored.uri,
    )


def serve_unit(
    context: ComparisonContext, unit: ComparisonUnit, reservation: UnitReservation
) -> ManualReplayBank | ManualModelEvidence:
    """Serve a unit's stored evidence through the verifying resume path, and never compute any of it.

    The evaluation's resume path builds whatever it does not find, so
    everything it would otherwise build is required first: the unit's own
    manifest, its fit, and its parent's bank. Serving then verifies them; a
    run simulated anyway would be work outside any worker, deadline or budget,
    and is refused loudly rather than kept.
    """
    missing = _missing_inputs(context, unit, reservation)
    if context.installed_evidence(unit, reservation) is None:
        missing.insert(0, "its own evidence is not installed")
    if missing:
        raise ValueError(f"unit {unit.number}: " + "; ".join(missing))
    configuration = context.configurations[unit.rank]
    simulated = len(context.runner.run_timings)
    if unit.kind == REPLAY:
        served: ManualReplayBank | ManualModelEvidence = context.runner.replay_bank(
            unit.label,
            warmup_s=configuration.warmup_s,
            replay_cutoffs=(configuration.velocity_cutoff_hz, configuration.acceleration_cutoff_hz),
        )
    else:
        served = context.runner.evaluate(context.entry(unit), warmup_s=configuration.warmup_s)
    if len(context.runner.run_timings) != simulated:
        msg = f"unit {unit.number}: serving simulated runs; stored evidence is served, never computed, here"
        raise ValueError(msg)
    return served


def _missing_inputs(context: ComparisonContext, unit: ComparisonUnit, reservation: UnitReservation) -> list[str]:
    store = context.store
    found: list[str] = []
    if (
        reservation.fit_identity is not None
        and not (store.root / cache_uri(reservation.fit_identity).relative_path).is_dir()
    ):
        found.append("its fit is not in the cache")
    bank = context.bank_unit(unit)
    if (
        bank is not None
        and installed_manifest(store, replay_bank_uri(context.conditions(unit.rank), bank.label)) is None
    ):
        found.append(f"its replay bank {bank.label} is not installed")
    return found


def _finalize(
    context: ComparisonContext,
    unit: ComparisonUnit,
    reservation: UnitReservation,
    *,
    root: Path,
    allowance: float,
    exploratory: bool,
) -> UnitOutcome | None:
    """Run one reserved unit to a finalized outcome, or leave it pending; every attempt is charged."""
    directory = unit_directory(context.store, unit)
    output = directory / "result.json"
    output.unlink(missing_ok=True)
    spend_file = directory / "spend.json"
    spent = read_record(spend_file, TrialSpend) if spend_file.is_file() else TrialSpend(trial=unit.number, seconds=0.0)
    now = _now()
    opened = spent.recovered(at=now).opening(at=now)
    write_record(spend_file, opened)
    started = time.perf_counter()
    try:
        completed = spawn_unit(
            unit_command(context, unit, output=output, root=root, exploratory=exploratory), timeout=allowance
        )
        interrupted = None if completed.returncode == 0 else _tail(completed.stderr)
    except subprocess.TimeoutExpired:
        interrupted = f"the worker exceeded the remaining allowance of {allowance:.0f} s"
    elapsed = time.perf_counter() - started
    charged = opened.closing(elapsed)
    if interrupted is not None or not output.is_file():
        write_record(spend_file, charged)
        print(f"unit {unit.number} interrupted: {interrupted or 'no report'}", file=sys.stderr)
        return None
    verify_started = time.perf_counter()
    outcome = verified_unit(context, unit, reservation, read_record(output, UnitResult), seconds=charged.seconds)
    verified = time.perf_counter() - verify_started
    outcome = _with_time(outcome, worker=elapsed, verify=verified)
    write_record(directory / "outcome.json", outcome)
    spend_file.unlink(missing_ok=True)
    return outcome


def _with_time(outcome: UnitOutcome, *, worker: float, verify: float) -> UnitOutcome:
    """Charge the parent's verification too: it is elapsed execution under the shared ceiling."""
    return replace(outcome, seconds=outcome.seconds + verify, worker_seconds=worker, verify_seconds=verify)


def _tail(stream: bytes) -> str:
    return " | ".join(stream.decode("utf-8", "replace").strip().splitlines()[-5:])[:500]


def _now() -> datetime:
    """The wall clock an attempt's start is recorded against; the tests replace it."""
    return datetime.now(tz=UTC)


def reconcile(context: ComparisonContext) -> list[dict[str, object]]:
    """Charge every attempt whose parent was killed before it could measure it, by the wall clock."""
    repaired: list[dict[str, object]] = []
    for unit in context.units:
        spend_file = unit_directory(context.store, unit) / "spend.json"
        if not spend_file.is_file():
            continue
        spend = read_record(spend_file, TrialSpend)
        if spend.started_at is None:
            continue
        closed = spend.recovered(at=_now())
        write_record(spend_file, closed)
        repaired.append({"charged": unit.number, "seconds": closed.seconds - spend.seconds})
    return repaired


def finalized_mismatches(context: ComparisonContext, unit: ComparisonUnit) -> list[str]:
    """Why a finalized unit's records are not what the trusted schedule and the store support, if they are not.

    A finalized unit is never taken on its word: its reservation must be the
    one the verified freeze derives, and its outcome must name that identity,
    hold every case under both trackers when it is complete, and point at the
    manifest actually installed under the identity.
    """
    directory = unit_directory(context.store, unit)
    outcome = read_record(directory / "outcome.json", UnitOutcome)
    derived = context.reservation(unit)
    found: list[str] = []
    recorded = _read_optional(directory / "reservation.json", UnitReservation)
    if recorded != derived:
        found.append("its recorded reservation is not the one the verified freeze derives")
    if outcome.number != unit.number or outcome.identity != derived.identity:
        found.append(f"its outcome names unit {outcome.number} at {outcome.identity[:12]}")
    if outcome.state == "complete":
        expected = len(context.conditions(unit.rank).pairs)
        if outcome.pairs != expected:
            found.append(f"its outcome holds {outcome.pairs} pairs, not the {expected} of every case and tracker")
        installed = context.installed_evidence(unit, derived)
        if installed is None or installed.uri != outcome.evidence:
            found.append("its outcome does not name the manifest installed under its identity")
        recounted = context.installed_counts(unit, derived)
        recorded_counts = (outcome.pairs, outcome.completed, outcome.infeasible, outcome.unexecuted)
        if recounted is not None and recounted != recorded_counts:
            found.append(
                f"its outcome's breakdown {recorded_counts} is not the installed manifest's {recounted} "
                "(pairs, completed, infeasible, unexecuted)"
            )
    elif outcome.pairs:
        found.append(f"a {outcome.state} unit holds no pairs, got {outcome.pairs}")
    return found


def check_finalized(context: ComparisonContext, unit: ComparisonUnit) -> None:
    """Refuse a finalized unit whose records the schedule and the store do not support."""
    found = finalized_mismatches(context, unit)
    if found:
        raise ValueError(f"unit {unit.number}: " + "; ".join(found))


def _reserve(context: ComparisonContext, unit: ComparisonUnit) -> UnitReservation:
    """The unit's reservation: written once, and on a resume required to be exactly the one derived now."""
    derived = context.reservation(unit)
    path = unit_directory(context.store, unit) / "reservation.json"
    if path.is_file():
        recorded = read_record(path, UnitReservation)
        if recorded != derived:
            msg = f"unit {unit.number}: the recorded reservation is not the one the verified freeze derives"
            raise ValueError(msg)
        return recorded
    write_record(path, derived)
    return derived


def _unavailable(
    context: ComparisonContext, unit: ComparisonUnit, reservation: UnitReservation, bank: ComparisonUnit
) -> UnitOutcome:
    outcome = UnitOutcome(
        number=unit.number,
        state="unavailable",
        identity=reservation.identity,
        pairs=0,
        completed=0,
        infeasible=0,
        unexecuted=0,
        seconds=0.0,
        stored_bytes=0,
        failure=f"its parent's replay bank (unit {bank.number}) did not complete",
    )
    write_record(unit_directory(context.store, unit) / "outcome.json", outcome)
    return outcome


def run_comparison(
    protocol: ManualSearchProtocol,
    protocol_file: Path,
    freeze_file: Path,
    *,
    root: Path,
    exploratory: bool = False,
    argv: Sequence[str] = ("compare",),
) -> SharedLedger:
    """Run or resume the comparison, unit by unit in order, until it is complete or the ceiling stops it."""
    started_at, started = datetime.now(tz=UTC), time.perf_counter()
    context = ComparisonContext(protocol, protocol_file, freeze_file, root=root, exploratory=exploratory, argv=argv)
    before = shared_ledger(context.store, context.freeze, context.units).seconds
    touched: list[int] = []
    completed = False
    try:
        reported: list[dict[str, object]] = reconcile(context)
        stopped: list[str] = []
        for unit in context.units:
            directory = unit_directory(context.store, unit)
            if (directory / "outcome.json").is_file():
                check_finalized(context, unit)
                continue
            ledger = shared_ledger(context.store, context.freeze, context.units)
            stopped = shared_stopped(ledger, protocol.budget)
            if stopped:
                break
            reservation = _reserve(context, unit)
            touched.append(unit.number)
            bank = context.bank_unit(unit)
            if bank is not None:
                bank_outcome = _read_optional(unit_directory(context.store, bank) / "outcome.json", UnitOutcome)
                if bank_outcome is None or bank_outcome.state != "complete":
                    reported.append(
                        {"unit": unit.number, "state": _unavailable(context, unit, reservation, bank).state}
                    )
                    continue
            outcome = _finalize(
                context,
                unit,
                reservation,
                root=root,
                allowance=_remaining_seconds(ledger, protocol.budget),
                exploratory=exploratory,
            )
            reported.append({"unit": unit.number, "state": None if outcome is None else outcome.state})
            if outcome is None:
                stopped = [f"unit {unit.number} was interrupted and is retained"]
                break
        ledger = shared_ledger(context.store, context.freeze, context.units)
        print(json.dumps({"units": reported, "spent": to_mapping(ledger), "stopped": stopped}, indent=2))
        completed = True
        return ledger
    finally:
        _record_invocation(context, started_at, started, before, touched, completed=completed)


def _record_invocation(
    context: ComparisonContext,
    started_at: datetime,
    started: float,
    before: float,
    touched: list[int],
    *,
    completed: bool,
) -> None:
    """Keep this invocation's wall clock beside the charged spend, even when it raised."""
    name = f"{started_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    path = context.store.root / ArtifactUri.parse(f"{COMPARISON_PREFIX}/invocations/{name}").relative_path
    after = shared_ledger(context.store, context.freeze, context.units).seconds
    write_record(
        path,
        SearchInvocation(
            started_at=started_at.isoformat(),
            seconds=time.perf_counter() - started,
            charged_seconds=after - before,
            trials=tuple(touched),
            stop_at_trials=None,
            completed=completed,
        ),
    )


# --- status and publication ----------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigurationStatus:
    """How far one configuration's comparison has come."""

    rank: int
    trial: int
    configuration: str
    units: int
    complete: int
    failed: int
    unavailable: int
    pending: int
    """Reserved units without an outcome: interrupted work a resume finishes."""
    not_started: int
    rc_pairs: int
    replay_pairs: int
    finished: bool
    """Every unit has an outcome: the configuration's whole comparison was attempted."""


@dataclass(frozen=True)
class ComparisonStatus:
    """The comparison as its retained records hold it, against the planned scope and the shared ceiling."""

    freeze_sha256: str
    planned_units: int
    planned_runs: int
    configurations: tuple[ConfigurationStatus, ...]
    ledger: SharedLedger
    stopped: tuple[str, ...]
    finished_configurations: int


def comparison_status(context: ComparisonContext) -> ComparisonStatus:
    """Count every unit by state, per configuration, in selection order."""
    rows: list[ConfigurationStatus] = []
    for item in context.freeze.chosen:
        units = [unit for unit in context.units if unit.rank == item.rank]
        outcomes: list[UnitOutcome] = []
        pending = not_started = 0
        for unit in units:
            directory = unit_directory(context.store, unit)
            outcome = _read_optional(directory / "outcome.json", UnitOutcome)
            if outcome is not None:
                check_finalized(context, unit)
                outcomes.append(outcome)
            elif (directory / "reservation.json").is_file():
                pending += 1
            else:
                not_started += 1
        kinds = {unit.number: unit.kind for unit in units}
        rows.append(
            ConfigurationStatus(
                rank=item.rank,
                trial=item.trial,
                configuration=item.configuration,
                units=len(units),
                complete=sum(1 for o in outcomes if o.state == "complete"),
                failed=sum(1 for o in outcomes if o.state == "failed"),
                unavailable=sum(1 for o in outcomes if o.state == "unavailable"),
                pending=pending,
                not_started=not_started,
                rc_pairs=sum(o.pairs for o in outcomes if o.state == "complete" and kinds[o.number] == MODEL),
                replay_pairs=sum(o.pairs for o in outcomes if o.state == "complete" and kinds[o.number] == REPLAY),
                finished=len(outcomes) == len(units),
            )
        )
    ledger = shared_ledger(context.store, context.freeze, context.units)
    per_case = len(context.conditions(1).pairs) if context.configurations else 0
    return ComparisonStatus(
        freeze_sha256=context.freeze_sha256,
        planned_units=len(context.units),
        planned_runs=len(context.units) * per_case,
        configurations=tuple(rows),
        ledger=ledger,
        stopped=tuple(shared_stopped(ledger, context.protocol.budget)),
        finished_configurations=sum(1 for row in rows if row.finished),
    )


def render_status(status: ComparisonStatus) -> str:
    """A human-readable account of the comparison's progress; the JSON is the record."""
    ledger = status.ledger
    lines = [
        "<!-- Generated by `python -m arm_rc_ctrl.experiments.manual_comparison_run status`; do not edit. -->",
        "",
        "# M3MS-006 comparison status",
        "",
        (
            f"Freeze `{status.freeze_sha256[:12]}`. {status.finished_configurations} of {len(status.configurations)} "
            f"configurations finished; {status.planned_units} units and at most {status.planned_runs:,} runs planned."
        ),
        "",
        (
            "| Rank | Trial | Units | Complete | Failed | Unavailable | Pending | Not started | RC pairs | "
            "Replay pairs | Finished |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines += [
        f"| {r.rank} | {r.trial} | {r.units} | {r.complete} | {r.failed} | {r.unavailable} | {r.pending} | "
        f"{r.not_started} | {r.rc_pairs:,} | {r.replay_pairs:,} | {'yes' if r.finished else 'no'} |"
        for r in status.configurations
    ]
    lines += [
        "",
        (
            f"Shared ceiling: {ledger.total_seconds / 3600.0:.2f} h and {ledger.total_bytes / _GIB:.3f} GiB spent "
            f"(search {ledger.search.seconds / 3600.0:.2f} h, {ledger.search.stored_bytes / _GIB:.3f} GiB; comparison "
            f"{ledger.seconds / 3600.0:.2f} h, {ledger.stored_bytes / _GIB:.3f} GiB)."
        ),
        "",
        "Stopped: " + ("no." if not status.stopped else "; ".join(status.stopped)),
        "",
    ]
    return "\n".join(lines)


def publish_pointers(context: ComparisonContext, evidence_dir: Path) -> list[Path]:
    """Serve every complete unit back through the verifying resume path, then write the Git pointers.

    Pointers are written once the comparison has stopped: a worker refuses a
    dirty worktree, so writing them while units still run would stop the run.
    """
    for unit in context.units:
        outcome = _read_optional(unit_directory(context.store, unit) / "outcome.json", UnitOutcome)
        if outcome is None or outcome.state != "complete":
            continue
        check_finalized(context, unit)
        serve_unit(
            context, unit, read_record(unit_directory(context.store, unit) / "reservation.json", UnitReservation)
        )
    return context.runner.write_pointers(evidence_dir)


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        msg = f"{path} already exists; a recorded status is versioned, never rewritten"
        raise FileExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: run, report or publish the comparison, or evaluate one unit."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run the five-arm comparison at the frozen configurations.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    for name, text in (
        ("run", "run or resume the comparison until it is complete or the shared ceiling stops it"),
        ("status", "write the comparison's status (new files only)"),
        ("publish", "verify every complete unit again and write its Git pointer"),
    ):
        command = subparsers.add_parser(name, help=text)
        command.add_argument("--protocol", required=True, type=Path)
        command.add_argument("--freeze", required=True, type=Path)
        command.add_argument("--exploratory", action="store_true", help="tolerate a dirty worktree (fixtures)")
        if name == "status":
            command.add_argument("--output", required=True, type=Path)
            command.add_argument("--markdown", required=True, type=Path)
        if name == "publish":
            command.add_argument("--evidence-dir", required=True, type=Path)
    worker = subparsers.add_parser("evaluate-unit", help="evaluate one unit in this process (spawned by run)")
    worker.add_argument("--protocol", required=True, type=Path)
    worker.add_argument("--freeze", required=True, type=Path)
    worker.add_argument("--freeze-sha256", required=True)
    worker.add_argument("--unit", required=True, type=int)
    worker.add_argument("--output", required=True, type=Path)
    worker.add_argument("--root", required=True, type=Path)
    worker.add_argument("--exploratory", action="store_true")
    args = parser.parse_args(argv)
    protocol_file = cast("Path", args.protocol)
    protocol = load_manual_search(protocol_file)
    freeze_file = cast("Path", args.freeze)
    exploratory = bool(args.exploratory)
    if args.subcommand == "evaluate-unit":
        result = evaluate_unit(
            protocol,
            freeze_file=freeze_file,
            freeze_sha256=cast("str", args.freeze_sha256),
            number=cast("int", args.unit),
            root=cast("Path", args.root),
            argv=argv,
            exploratory=exploratory,
        )
        write_record(cast("Path", args.output), result)
        return 0
    if args.subcommand == "run":
        run_comparison(protocol, protocol_file, freeze_file, root=repository_root(), exploratory=exploratory, argv=argv)
        return 0
    context = ComparisonContext(
        protocol, protocol_file, freeze_file, root=repository_root(), exploratory=exploratory, argv=argv
    )
    if args.subcommand == "status":
        status = comparison_status(context)
        _write_new(cast("Path", args.output), json.dumps(to_mapping(status), sort_keys=True, indent=2) + "\n")
        _write_new(cast("Path", args.markdown), render_status(status))
        print(render_status(status))
        return 0
    written = publish_pointers(context, cast("Path", args.evidence_dir))
    print(f"{len(written)} pointers written")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
