# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-003: the resumable nominal-only search, its scope guards and its budget accounting.

The parent opens (or resumes) one Optuna study keyed by the protocol's identity
and runs trials **serially**. Each trial draws a point from the approved space,
turns it into a sampled configuration, and hands it to a worker **process**,
which fits the all-ten arm and evaluates the nominal case under both fixed
trackers. The worker's score is the nominal success fraction; nothing else
reaches the optimizer.

Two properties are enforced on both sides of that boundary rather than assumed:

* **Scope.** The parent restricts the evaluation to ``("nominal",)`` and the
  worker checks the scenarios it was actually given. A parent-side preflight
  alone would not survive a worker invoked by hand or by a resumed study.
* **Budget.** Trials, elapsed seconds and stored bytes accumulate on the study
  itself, so a resumed invocation continues the same accounting instead of
  starting a fresh allowance. Work stops at a cap; it is never enlarged.

Numerical work runs in the worker process, which inherits the whole canonical
affinity set with one thread each (plan section 7). Worker count is never
implemented by narrowing a worker's CPU set: the execution identity includes it,
and evidence produced under a narrowed set is not this study's.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from optuna.trial import TrialState

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.manual_evaluation import (
    PROGRESS_FILE,
    load_manual_evaluation_config,
    load_manual_model_evidence,
    load_verified_run,
    manual_conditions,
    model_uri,
    prepare_runner,
)
from arm_rc_ctrl.experiments.manual_fits import cache_uri
from arm_rc_ctrl.experiments.manual_sampled import (
    SampledPoint,
    arm_of,
    sampled_configuration,
    sampled_entry,
    sampled_point,
)
from arm_rc_ctrl.experiments.manual_search import (
    NOMINAL_SCENARIOS,
    REQUIRED_RUNS,
    load_manual_search,
    nominal_success_fraction,
    protocol_digest,
)
from arm_rc_ctrl.experiments.manual_study import StudyManifest, load_study
from arm_rc_ctrl.experiments.studies import PrunerSpec, close_study, open_study
from arm_rc_ctrl.provenance import ArtifactReference, canonical_json, sha256_bytes, verify_artifact
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optuna

    from arm_rc_ctrl.experiments.manual_evaluation import ManualEvaluationConfig
    from arm_rc_ctrl.experiments.manual_search import ManualSearchBudget, ManualSearchProtocol, ManualSearchSpace

__all__ = [
    "HEADROOM_S",
    "STUDY_NAME",
    "TRIALS_PREFIX",
    "BudgetLedger",
    "TrialOutcome",
    "TrialReservation",
    "TrialResult",
    "TrialSpend",
    "budget_complaints",
    "evaluate_trial",
    "ledger_of",
    "nominal_scope_mismatches",
    "pending_reservations",
    "read_result",
    "run_search",
    "spawn_trial",
    "suggest_sampled_point",
    "trial_command",
    "trial_directories",
    "trial_directory",
    "verified_outcome",
]

STUDY_NAME: Final = "manual-esn-search-v1"
"""The study's name in the Optuna bucket; its identity is the protocol digest, checked on resume."""
_MODULE: Final = "arm_rc_ctrl.experiments.manual_search_run"
_GIB: Final = 1024**3
_SHA256_HEX: Final = 64
TRIALS_PREFIX: Final = "armrc://reports/task_1a_manual_search/trials"
"""Where each trial's reservation, report and outcome are retained, outside the closed study's prefix."""
HEADROOM_S: Final = 60.0
"""Allowance reserved so an in-flight result can still be persisted when a cap is near."""


def nominal_scope_mismatches(scenarios: Sequence[str]) -> list[str]:
    """Why a scenario list is not the optimizer's nominal-only scope, if it is not.

    Checked by the parent before it schedules anything and by the worker with
    the list it actually received: a perturbed case reaching the objective
    would undo the separation the whole experiment rests on.
    """
    actual = tuple(scenarios)
    if actual != NOMINAL_SCENARIOS:
        return [
            (
                f"the optimizer evaluates {NOMINAL_SCENARIOS} and nothing else, got {actual}: "
                "no perturbed scenario may reach the score, a tie-break or pruning"
            )
        ]
    return []


@dataclass(frozen=True)
class BudgetLedger:
    """What this search has spent so far, across every invocation that resumed it."""

    trials: int
    seconds: float
    stored_bytes: int

    def __post_init__(self) -> None:
        """Spend is never negative, and a resumed ledger never forgets what was already spent."""
        if self.trials < 0 or self.seconds < 0.0 or self.stored_bytes < 0:
            msg = f"a ledger records non-negative spend, got {self}"
            raise ValueError(msg)

    def plus(self, *, trials: int = 0, seconds: float = 0.0, stored_bytes: int = 0) -> BudgetLedger:
        """The ledger after one more trial's spend."""
        return BudgetLedger(
            trials=self.trials + trials,
            seconds=self.seconds + seconds,
            stored_bytes=self.stored_bytes + stored_bytes,
        )


def budget_complaints(ledger: BudgetLedger, budget: ManualSearchBudget) -> list[str]:
    """Every cap this ledger has reached. A non-empty answer stops scheduling; it never raises the cap."""
    complaints: list[str] = []
    if ledger.trials >= budget.trials:
        complaints.append(f"the {budget.trials}-trial cap is spent ({ledger.trials} trials)")
    if ledger.seconds >= budget.hours * 3600.0:
        complaints.append(f"the {budget.hours:g} h cap is spent ({ledger.seconds / 3600.0:.2f} h)")
    if ledger.stored_bytes >= budget.gib * _GIB:
        complaints.append(f"the {budget.gib:g} GiB cap is spent ({ledger.stored_bytes / _GIB:.2f} GiB)")
    return complaints


def suggest_sampled_point(space: ManualSearchSpace, trial: optuna.Trial) -> SampledPoint:
    """Draw one point of the approved space, then read it back through the protocol's own validation."""
    params: dict[str, float] = {
        "n_neurons": trial.suggest_int(
            "n_neurons", space.n_neurons.low, space.n_neurons.high, step=space.n_neurons.step
        ),
        "spectral_radius": trial.suggest_float(
            "spectral_radius", space.spectral_radius.low, space.spectral_radius.high
        ),
        "sparsity": trial.suggest_float("sparsity", space.sparsity.low, space.sparsity.high),
        "leak_rate": trial.suggest_float("leak_rate", space.leak_rate.low, space.leak_rate.high, log=True),
        "input_scaling": trial.suggest_float(
            "input_scaling", space.input_scaling.low, space.input_scaling.high, log=True
        ),
        "alpha_0": trial.suggest_float("alpha_0", space.alpha_0.low, space.alpha_0.high, log=True),
        "warmup_s": trial.suggest_categorical("warmup_s", list(space.warmup_s)),
    }
    return sampled_point(space, params)


@dataclass(frozen=True)
class TrialReservation:
    """The parent's commitment to one trial, written before its worker starts.

    A reservation is what makes an interruption recoverable: it names the trial
    number, the configuration and the exact point, so a resumed search finishes
    that trial under its original identity instead of handing the work to a new
    one and spending a second slot on it.
    """

    trial: int
    configuration: str
    point: SampledPoint
    protocol_sha256: str
    fit_identity: str
    """The fit this trial's candidate is keyed by, derived from the trusted inputs before it runs."""
    evidence_identity: str
    """Where this trial's evidence will live, so its retained work is discoverable without it."""

    def __post_init__(self) -> None:
        """A reservation names a real trial, a known protocol, and the work it is about to produce."""
        digests = (self.protocol_sha256, self.fit_identity, self.evidence_identity)
        if self.trial < 0 or not self.configuration.strip() or any(len(d) != _SHA256_HEX for d in digests):
            msg = f"a reservation names its trial, configuration, protocol and identities, got {self}"
            raise ValueError(msg)


@dataclass(frozen=True)
class TrialResult:
    """What a worker reports about one candidate: verdict counts and where its evidence was left."""

    trial: int
    successes: int
    runs: int
    statuses: tuple[str, ...]
    scenarios: tuple[str, ...]
    seconds: float
    evidence: ArtifactReference | None = None
    """The model evidence manifest the worker installed, which the parent verifies before scoring."""
    failure: str | None = None
    """Why the candidate has no score: a fit failure, retained as a failed trial rather than a zero."""

    def __post_init__(self) -> None:
        """A reported result is internally consistent and stays inside the nominal scope."""
        if self.trial < 0 or self.runs < 0 or self.successes < 0 or self.successes > self.runs:
            msg = f"trial {self.trial}: {self.successes} successes of {self.runs} runs is not a result"
            raise ValueError(msg)
        if len(self.statuses) != self.runs:
            msg = f"trial {self.trial}: {len(self.statuses)} statuses for {self.runs} runs"
            raise ValueError(msg)
        if self.seconds < 0.0:
            msg = f"trial {self.trial}: spend is never negative"
            raise ValueError(msg)
        if self.failure is None and self.evidence is None:
            msg = f"trial {self.trial}: a scored candidate reports the evidence it produced"
            raise ValueError(msg)
        mismatches = nominal_scope_mismatches(self.scenarios)
        if mismatches:
            msg = f"trial {self.trial}: {'; '.join(mismatches)}"
            raise ValueError(msg)


@dataclass(frozen=True)
class TrialOutcome:
    """A finalized trial: what the parent verified, and what it charged for it."""

    trial: int
    state: Literal["scored", "failed"]
    score: float | None
    successes: int
    runs: int
    seconds: float
    stored_bytes: int
    evidence: str | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        """A scored trial has a score and a failed one has a reason; neither has both."""
        if (self.state == "scored") != (self.score is not None):
            msg = f"trial {self.trial}: state {self.state!r} with score {self.score!r}"
            raise ValueError(msg)
        if (self.state == "failed") != (self.failure is not None):
            msg = f"trial {self.trial}: state {self.state!r} with failure {self.failure!r}"
            raise ValueError(msg)
        if self.seconds < 0.0 or self.stored_bytes < 0:
            msg = f"trial {self.trial}: spend is never negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class TrialSpend:
    """What a trial has cost so far, accumulated over every attempt, including interrupted ones."""

    trial: int
    seconds: float
    attempts: int = 1

    def __post_init__(self) -> None:
        """Spend only grows, and an attempt that happened is never unattempted."""
        if self.seconds < 0.0 or self.attempts < 1:
            msg = f"trial {self.trial}: {self.attempts} attempts costing {self.seconds} s is not a spend"
            raise ValueError(msg)

    def plus(self, seconds: float) -> TrialSpend:
        """The spend after one more attempt's cost is added."""
        return TrialSpend(trial=self.trial, seconds=self.seconds + seconds, attempts=self.attempts + 1)


def trial_directory(store: StorageRoot, trial: int) -> Path:
    """Where one trial's reservation, report and outcome are kept, inside the store and retained."""
    return store.path(f"{TRIALS_PREFIX}/trial-{trial:04d}/reservation.json", mode="write").parent


def write_record(path: Path, record: object) -> None:
    """Write one search record as canonical JSON, replacing it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.staging")
    staged.write_text(canonical_json(to_mapping(record)) + "\n", encoding="utf-8")
    staged.replace(path)


def read_record[T](path: Path, schema: type[T]) -> T:
    """Read one search record strictly."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), schema)


def read_result(path: Path) -> TrialResult:
    """Read a worker's report strictly; an unreadable one is an interruption, never a zero."""
    return read_record(path, TrialResult)


def ledger_of(store: StorageRoot) -> BudgetLedger:
    """The spend, derived from the retained trial records rather than carried in a counter.

    Deriving it is what makes the accounting recoverable: an interruption
    between finalizing a trial and recording its spend cannot lose the spend,
    because there is no separate place for it to be lost from.
    """
    trials, seconds, stored = 0, 0.0, 0
    for directory in trial_directories(store):
        outcome_file, spend_file, reservation_file = (
            directory / "outcome.json",
            directory / "spend.json",
            directory / "reservation.json",
        )
        if outcome_file.is_file():
            outcome = read_record(outcome_file, TrialOutcome)
            trials, seconds, stored = trials + 1, seconds + outcome.seconds, stored + outcome.stored_bytes
            continue
        if spend_file.is_file():
            seconds += read_record(spend_file, TrialSpend).seconds
        if reservation_file.is_file():
            # An unfinished trial has still written whatever it wrote, and the ceiling counts it.
            stored += retained_bytes(store, read_record(reservation_file, TrialReservation))
    return BudgetLedger(trials=trials, seconds=seconds, stored_bytes=stored)


def trial_directories(store: StorageRoot) -> list[Path]:
    """Every retained trial directory, in trial order; the later tasks read these records too."""
    root = store.path(f"{TRIALS_PREFIX}/trial-0000/reservation.json", mode="write").parent.parent
    return sorted(root.glob("trial-*")) if root.is_dir() else []


def retained_bytes(store: StorageRoot, reservation: TrialReservation) -> int:
    """Every byte this trial has left in the store, whether or not it produced evidence.

    The reservation names where the work will live before it runs, so partial
    work is discoverable without a completed evidence object: the fit cache,
    the evidence directory, and each run the run-granular progress record
    names. A candidate that failed or was interrupted mid-evaluation is
    therefore charged for what it actually retained.
    """
    directories = [
        store.root / cache_uri(reservation.fit_identity).relative_path,
        store.path(f"{model_uri(reservation.evidence_identity)}/{PROGRESS_FILE}", mode="write").parent,
    ]
    progress = directories[1] / PROGRESS_FILE
    if progress.is_file():
        try:
            pairs = cast("list[dict[str, object]]", json.loads(progress.read_text(encoding="utf-8"))["pairs"])
        except (OSError, ValueError, KeyError):
            pairs = []
        for pair in pairs:
            run = cast("dict[str, object] | None", pair.get("run"))
            if run is not None:
                directories.append(store.path(cast("str", run["uri"]), mode="write").parent)
    total = 0
    for directory in {path.resolve() for path in directories}:
        if directory.is_dir():
            total += sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
    return total


def pending_reservations(store: StorageRoot) -> tuple[TrialReservation, ...]:
    """Every reserved trial that has no outcome yet, in trial order: the work a resume must finish."""
    root = store.path(f"{TRIALS_PREFIX}/trial-0000/reservation.json", mode="write").parent.parent
    pending: list[TrialReservation] = []
    for directory in sorted(root.glob("trial-*")) if root.is_dir() else []:
        reservation = directory / "reservation.json"
        if reservation.is_file() and not (directory / "outcome.json").is_file():
            pending.append(read_record(reservation, TrialReservation))
    return tuple(pending)


def verified_outcome(
    store: StorageRoot, reservation: TrialReservation, result: TrialResult, *, seconds: float
) -> TrialOutcome:
    """Turn a worker's report into a finalized trial, deriving the score from verified evidence.

    A report is not evidence. It must name the trial that was scheduled, and a
    scored candidate must point at a model evidence manifest that verifies
    against its digest, belongs to the configuration this trial reserved, and
    carries the nominal runs whose verdicts are recounted here.
    """
    if result.trial != reservation.trial:
        msg = f"the report names trial {result.trial}, but trial {reservation.trial} was scheduled"
        raise ValueError(msg)
    charged = seconds
    if result.failure is not None:
        return TrialOutcome(
            trial=reservation.trial,
            state="failed",
            score=None,
            successes=0,
            runs=result.runs,
            seconds=charged,
            stored_bytes=retained_bytes(store, reservation),
            evidence=None,
            failure=result.failure,
        )
    payload = cast("ArtifactReference", result.evidence)
    evidence = load_manual_model_evidence(verify_artifact(store, payload))
    expected = f"{reservation.configuration}/M10"
    complaints = [
        f"the evidence is {evidence.label!r}, not this trial's {expected!r}" if evidence.label != expected else None,
        (
            f"the evidence is {evidence.identity[:12]}, not the reserved {reservation.evidence_identity[:12]}"
            if evidence.identity != reservation.evidence_identity
            else None
        ),
        (
            f"the evidence was fitted under {None if evidence.fit is None else evidence.fit.identity[:12]}, not the "
            f"reserved {reservation.fit_identity[:12]}"
            if evidence.fit is None or evidence.fit.identity != reservation.fit_identity
            else None
        ),
        *nominal_scope_mismatches(tuple(dict.fromkeys(pair.scenario_id for pair in evidence.pairs))),
        (
            f"the evidence holds {len(evidence.pairs)} runs, not the two fixed trackers'"
            if len(evidence.pairs) != REQUIRED_RUNS
            else None
        ),
    ]
    reported = [complaint for complaint in complaints if complaint]
    if reported:
        msg = f"trial {reservation.trial}: {'; '.join(reported)}"
        raise ValueError(msg)
    # Every run is loaded through the reader a resume uses, so a corrupted archive or a verdict the
    # payload does not support is refused before the candidate is scored.
    for pair in evidence.pairs:
        load_verified_run(store, pair)
    successes = sum(1 for pair in evidence.pairs if pair.outcome is not None and pair.outcome.success)
    runs = len(evidence.pairs)
    return TrialOutcome(
        trial=reservation.trial,
        state="scored",
        score=nominal_success_fraction(successes, runs=runs),
        successes=successes,
        runs=runs,
        seconds=charged,
        stored_bytes=retained_bytes(store, reservation),
        evidence=payload.uri,
        failure=None,
    )


@dataclass(frozen=True)
class SearchInputs:
    """What the parent must load to reserve a trial: the frozen study and the bound evaluation."""

    manifest: StudyManifest
    config: ManualEvaluationConfig
    evaluation_file: Path
    root: Path


def search_inputs(protocol: ManualSearchProtocol, *, root: Path) -> SearchInputs:
    """Load what the parent needs to derive a trial's identities before it runs."""
    return SearchInputs(
        manifest=load_study(protocol.study),
        config=load_manual_evaluation_config(protocol.comparison.evaluation),
        evaluation_file=protocol.comparison.evaluation,
        root=root,
    )


def reserve_trial(
    protocol: ManualSearchProtocol, inputs: SearchInputs, point: SampledPoint, *, trial: int
) -> TrialReservation:
    """The reservation of one trial, with the identities its work will be stored under.

    Deriving them here, from the protocol and the point alone, is what lets the
    parent find a candidate's retained work and verify its evidence without
    taking a worker's word for either.
    """
    configuration = sampled_configuration(protocol, point, trial=trial)
    entry = sampled_entry(inputs.manifest, configuration, arm_of("M10"))
    conditions = manual_conditions(
        inputs.config,
        inputs.evaluation_file,
        scenario_ids=NOMINAL_SCENARIOS,
        warmup_s=configuration.warmup_s,
        replay_cutoffs=(configuration.velocity_cutoff_hz, configuration.acceleration_cutoff_hz),
        execution_identity=inputs.manifest.execution.identity,
        root=inputs.root,
    )
    return TrialReservation(
        trial=trial,
        configuration=configuration.label,
        point=point,
        protocol_sha256=protocol_digest(protocol),
        fit_identity=entry.fit_identity,
        evidence_identity=sha256_bytes(f"{entry.fit_identity}:{conditions.identity}".encode("ascii")),
    )


def spawn_trial(command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[bytes]:
    """Run one trial's worker as its own process, inheriting this process's pinned environment.

    The worker is a process rather than a thread because reservoir
    construction re-seeds a process-global generator: two constructions that
    interleave in one interpreter can differ from the same seed. ``timeout``
    is the allowance left before a cap, so a worker cannot run past a ceiling
    that is only checked between trials. It is always a real deadline: a
    non-positive allowance stops scheduling rather than starting an unbounded
    worker.
    """
    if timeout <= 0.0:
        msg = f"a worker is never started without a deadline, got {timeout}"
        raise ValueError(msg)
    return subprocess.run(list(command), check=False, capture_output=True, timeout=timeout)


def evaluate_trial(
    protocol: ManualSearchProtocol,
    *,
    trial: int,
    point: SampledPoint,
    scenarios: Sequence[str],
    root: Path,
    argv: Sequence[str],
    exploratory: bool = False,
) -> TrialResult:
    """Fit the all-ten arm at one sampled configuration and judge its nominal runs.

    This runs in the worker process. The scenarios it was actually given are
    checked here, not only where they were chosen, and the configuration is
    rebuilt from the point so the worker never depends on the parent having
    described it correctly. A fit or evaluation failure is reported as a failed
    candidate with whatever it stored; only an interruption leaves no report.
    """
    mismatches = nominal_scope_mismatches(scenarios)
    if mismatches:
        msg = "; ".join(mismatches)
        raise ValueError(msg)
    configuration = sampled_configuration(protocol, point, trial=trial)
    args = argparse.Namespace(
        study=str(protocol.study),
        evaluation=str(protocol.comparison.evaluation),
        exploratory=exploratory,
        argv=list(argv),
    )
    prepared = prepare_runner(
        args,
        role="worker",
        root=root,
        module=_MODULE,
        scenario_ids=tuple(scenarios),
        sampled=(configuration,),
    )
    manifest = prepared.context.manifest
    entry = sampled_entry(manifest, configuration, arm_of("M10"))
    started = time.perf_counter()
    try:
        evidence = prepared.runner.evaluate(entry, warmup_s=configuration.warmup_s)
    except ValueError as error:
        # A candidate the learner or the protocol refuses is a failed trial. An infrastructure
        # failure -- a storage error, a missing device, a mismatched artifact -- is not: it
        # propagates, leaving no report, so the parent retains the trial and a resume retries it.
        return TrialResult(
            trial=trial,
            successes=0,
            runs=0,
            statuses=(),
            scenarios=tuple(scenarios),
            seconds=time.perf_counter() - started,
            failure=f"{type(error).__name__}: {str(error)[:300]}",
        )
    pointer = next(item for item in prepared.runner.pointers if item.label == entry.label)
    return TrialResult(
        trial=trial,
        successes=sum(1 for pair in evidence.pairs if pair.outcome is not None and pair.outcome.success),
        runs=len(evidence.pairs),
        statuses=tuple(pair.status for pair in evidence.pairs),
        scenarios=tuple(scenarios),
        seconds=time.perf_counter() - started,
        evidence=pointer.payload,
    )


def trial_command(
    protocol_file: Path, *, trial: int, point: SampledPoint, output: Path, root: Path, exploratory: bool = False
) -> list[str]:
    """The worker invocation for one trial: its own process, its own scope, its own report.

    The worker is given the point rather than a seed for the sampler, so it
    reproduces exactly the candidate the parent drew, and it is given the
    scenarios explicitly so it can refuse anything but the nominal case.
    """
    return [
        sys.executable,
        "-m",
        _MODULE,
        "evaluate-trial",
        "--protocol",
        str(protocol_file),
        "--trial",
        str(trial),
        "--point",
        json.dumps(to_mapping(point), sort_keys=True),
        "--scenarios",
        *NOMINAL_SCENARIOS,
        "--output",
        str(output),
        "--root",
        str(root),
        *(["--exploratory"] if exploratory else []),
    ]


def _remaining_seconds(ledger: BudgetLedger, budget: ManualSearchBudget) -> float:
    """The allowance a worker may use, keeping headroom to persist an in-flight result."""
    return budget.hours * 3600.0 - ledger.seconds - HEADROOM_S


def _finalize(
    study: optuna.Study,
    store: StorageRoot,
    reservation: TrialReservation,
    *,
    protocol_file: Path,
    root: Path,
    allowance: float,
    exploratory: bool,
) -> TrialOutcome | None:
    """Run one reserved trial to a finalized outcome, or leave it pending for a resume.

    An interruption -- a crash, a kill, or the elapsed allowance running out --
    writes no outcome. The reservation stays, the work stays where it was
    written, and every attempt's measured cost is kept, so a resume finishes
    this trial under its own number instead of buying a replacement and the
    ledger charges what the retries actually cost.
    """
    directory = trial_directory(store, reservation.trial)
    output = directory / "result.json"
    command = trial_command(
        protocol_file,
        trial=reservation.trial,
        point=reservation.point,
        output=output,
        root=root,
        exploratory=exploratory,
    )
    spend_file = directory / "spend.json"
    spent = read_record(spend_file, TrialSpend) if spend_file.is_file() else None
    started = time.perf_counter()
    try:
        completed = spawn_trial(command, timeout=allowance)
        interrupted = None if completed.returncode == 0 else _tail(completed.stderr)
    except subprocess.TimeoutExpired:
        interrupted = f"the worker exceeded the remaining allowance of {allowance:.0f} s"
    elapsed = time.perf_counter() - started
    # Every attempt's cost is kept: a retry adds to the charge rather than replacing it.
    charged = TrialSpend(trial=reservation.trial, seconds=elapsed) if spent is None else spent.plus(elapsed)
    if interrupted is not None and not output.is_file():
        # Infrastructure, not a fit: no outcome, so the trial stays pending and is resumed.
        write_record(spend_file, charged)
        return None
    outcome = verified_outcome(store, reservation, read_result(output), seconds=charged.seconds)
    write_record(directory / "outcome.json", outcome)
    spend_file.unlink(missing_ok=True)
    study.tell(
        reservation.trial,
        outcome.score if outcome.state == "scored" else None,
        state=None if outcome.state == "scored" else TrialState.FAIL,
        skip_if_finished=True,
    )
    return outcome


def _tail(stream: bytes) -> str:
    """The last few lines a failed worker wrote, for the record."""
    return " | ".join(stream.decode("utf-8", "replace").strip().splitlines()[-5:])[:500]


def reconcile(
    study: optuna.Study, store: StorageRoot, inputs: SearchInputs, protocol: ManualSearchProtocol
) -> list[dict[str, object]]:
    """Bring the study and the retained records back into agreement before anything is scheduled.

    Three interruption points leave them disagreeing, and each is repaired
    here rather than worked around later:

    * an outcome was written but the study never heard of it, which would leave
      its trial running for ever while the ledger counts it finished;
    * a trial was asked for but its reservation was never published, which
      would orphan that trial and hand its work to the next number;
    * a reservation has no outcome, which is ordinary pending work.
    """
    repaired: list[dict[str, object]] = []
    finalized = {t.number for t in study.trials if t.state != TrialState.RUNNING}
    for directory in trial_directories(store):
        outcome_file = directory / "outcome.json"
        if not outcome_file.is_file():
            continue
        outcome = read_record(outcome_file, TrialOutcome)
        if outcome.trial in finalized:
            continue
        study.tell(
            outcome.trial,
            outcome.score if outcome.state == "scored" else None,
            state=None if outcome.state == "scored" else TrialState.FAIL,
            skip_if_finished=True,
        )
        repaired.append({"finalized": outcome.trial, "state": outcome.state})
    reserved = {reservation.trial for reservation in pending_reservations(store)}
    stored = {
        read_record(d / "outcome.json", TrialOutcome).trial
        for d in trial_directories(store)
        if (d / "outcome.json").is_file()
    }
    for frozen in study.trials:
        if frozen.state != TrialState.RUNNING or frozen.number in reserved or frozen.number in stored:
            continue
        # Asked for, then lost before its reservation reached the store.
        try:
            point = sampled_point(protocol.space, cast("dict[str, float]", frozen.params))
        except (ValueError, KeyError):
            study.tell(frozen.number, None, state=TrialState.FAIL, skip_if_finished=True)
            repaired.append({"abandoned": frozen.number, "reason": "no usable parameters were recorded"})
            continue
        write_record(
            trial_directory(store, frozen.number) / "reservation.json",
            reserve_trial(protocol, inputs, point, trial=frozen.number),
        )
        repaired.append({"adopted": frozen.number})
    return repaired


def run_search(
    protocol: ManualSearchProtocol, protocol_file: Path, *, root: Path, exploratory: bool = False
) -> BudgetLedger:
    """Run or resume the search: reconcile, finish every pending trial, then draw new ones until a cap.

    Pending work is recovered under its original trial identity before
    anything new is scheduled, and the ledger is derived from the retained
    records, so neither an interruption nor a resume can spend a cap twice. A
    trial is never started without a real deadline: when the elapsed allowance
    is gone, scheduling stops and the work that exists is retained.
    """
    store = open_storage()
    inputs = search_inputs(protocol, root=root)
    study = open_study(
        store,
        STUDY_NAME,
        protocol_sha256=protocol_digest(protocol),
        sampler=protocol.sampler,
        pruner=PrunerSpec(kind="none"),
        direction="maximize",
    )
    try:
        reported: list[dict[str, object]] = list(reconcile(study, store, inputs, protocol))
        while True:
            ledger = ledger_of(store)
            complaints = budget_complaints(ledger, protocol.budget)
            if complaints:
                return _report(ledger, reported, stopped=complaints)
            allowance = _remaining_seconds(ledger, protocol.budget)
            if allowance <= 0.0:
                stopped = [f"the {protocol.budget.hours:g} h cap leaves no allowance beyond the persistence headroom"]
                return _report(ledger, reported, stopped=stopped)
            pending = pending_reservations(store)
            recovering = bool(pending)
            if recovering:
                reservation = pending[0]
            else:
                trial = study.ask()
                reservation = reserve_trial(
                    protocol, inputs, suggest_sampled_point(protocol.space, trial), trial=trial.number
                )
                write_record(trial_directory(store, reservation.trial) / "reservation.json", reservation)
            outcome = _finalize(
                study,
                store,
                reservation,
                protocol_file=protocol_file,
                root=root,
                allowance=allowance,
                exploratory=exploratory,
            )
            key = "recovered" if recovering else "trial"
            reported.append({key: reservation.trial, "state": None if outcome is None else outcome.state})
            if outcome is None:
                return _report(ledger_of(store), reported, stopped=["a trial was interrupted and is retained"])
    finally:
        close_study(study)


def _report(ledger: BudgetLedger, trials: list[dict[str, object]], *, stopped: list[str] | None = None) -> BudgetLedger:
    """Print what this invocation did and what it spent, and return the spend."""
    print(json.dumps({"trials": trials, "spent": to_mapping(ledger), "stopped": stopped or []}, indent=2))
    return ledger


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: the parent search and the per-trial worker."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run or resume the manual-demonstration ESN search.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    search = subparsers.add_parser("search", help="run or resume the approved search")
    search.add_argument("--protocol", required=True, type=Path, help="the frozen search protocol")
    search.add_argument("--exploratory", action="store_true", help="tolerate a dirty worktree (fixtures, trials)")
    worker = subparsers.add_parser("evaluate-trial", help="evaluate one candidate in this process")
    worker.add_argument("--protocol", required=True, type=Path)
    worker.add_argument("--trial", required=True, type=int)
    worker.add_argument("--point", required=True, help="the sampled point as JSON")
    worker.add_argument("--scenarios", nargs="+", required=True, help="the scenarios this worker may evaluate")
    worker.add_argument("--output", required=True, type=Path)
    worker.add_argument("--root", required=True, type=Path)
    worker.add_argument("--exploratory", action="store_true", help="tolerate a dirty worktree (fixtures, trials)")
    args = parser.parse_args(argv)
    protocol = load_manual_search(Path(cast("Path", args.protocol)))
    if args.subcommand == "search":
        run_search(
            protocol,
            Path(cast("Path", args.protocol)),
            root=repository_root(),
            exploratory=bool(args.exploratory),
        )
        return 0
    point = read_record_from_text(cast("str", args.point), SampledPoint)
    result = evaluate_trial(
        protocol,
        trial=int(cast("int", args.trial)),
        point=point,
        scenarios=tuple(cast("list[str]", args.scenarios)),
        root=Path(cast("Path", args.root)),
        argv=argv,
        exploratory=bool(args.exploratory),
    )
    write_record(Path(cast("Path", args.output)), result)
    return 0


def read_record_from_text[T](text: str, schema: type[T]) -> T:
    """Read one record from a JSON string, as strictly as from a file."""
    return from_mapping(cast("dict[str, object]", json.loads(text)), schema)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
