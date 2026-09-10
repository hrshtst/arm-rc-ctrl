# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Paired pilot evaluation of the repeated-demonstration panel (M3REP-004; repetition plan sections 7.1 and 8).

Every behavioral model configuration (one frozen panel entry fitted under one
arm, served from the M3REP-003 fit cache) is evaluated on the recovery
protocol's 65 development scenarios under both frozen trackers, paired
against direct replay at the same warm-up and conditions, in the fixed
scenario-major, tracker-minor order, stopping at the first infeasible pair
exactly as recovery v1 did. The evaluation differs from recovery v1 in one
declared respect only: the simulation aborts at the evaluation config's
``simulation.velocity_abort`` (12 rad/s per joint, decision D5) instead of the
scenario's 6 rad/s limit, which stays the training-validation limit and is
reported as a non-terminating diagnostic (clarification C2).

Evidence (clarification C6): every run's arrays live in the external store as
an immutable run record; one evidence manifest per model configuration and one
per replay bank binds every constituent run's identity, digest, size, resolved
conditions, and status, and each gets one Git pointer. Progress is persisted
after every run so an interrupted sweep resumes at run granularity without
re-simulating or overwriting completed runs. A replay failure in a posture
class blocks the paired RC evaluation and stops the sweep (C7): such models
are labelled replay-blocked, the remaining pairs are marked unexecuted, and
they are reported apart from RC-gate failures. Every manifest carries the C11
caveat of the accepted numerical exception.

Command line (launch pinned through ``python -m arm_rc_ctrl.execution run --policy p-cores -- ...``)::

    python -m arm_rc_ctrl.experiments.repetition_evaluation run
        --manifest <docs>/panel_manifest_v1.json --evaluation configs/evaluations/task_1a_repetition_dev_v1.toml
        --validation <docs>/numerical_validation_v1.json --evidence-dir <docs>/evidence
        [--entries feasible-best ...] [--arms absolute/R/K17 ...]
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, load_config, to_mapping
from arm_rc_ctrl.controllers.adapter import GeneratorTrackingController
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator
from arm_rc_ctrl.controllers.tracking import LimitedTracker
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.records import load_record, write_record
from arm_rc_ctrl.data.recovery import task_intervals_from_phases
from arm_rc_ctrl.execution import ExecutionRecord, collect_execution, require_canonical
from arm_rc_ctrl.experiments.baselines import frozen_baseline_digest, load_frozen_baseline
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario, load_development_robustness, robustness_scenarios
from arm_rc_ctrl.experiments.recovery_objective import (
    RATIO_CLASSES,
    RecoveryComponent,
    ReplayComponent,
    blocked_component,
    recovery_component,
    replay_component,
)
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.experiments.recovery_slice import DEFAULT_SETTLING_BAND_RAD, HeldTaskReference, recovery_outcome
from arm_rc_ctrl.experiments.repetition_fits import CachedFit, FitInputs, FitStore
from arm_rc_ctrl.experiments.repetition_numerics import PanelContext, load_validation
from arm_rc_ctrl.experiments.repetition_panel import EXPERIMENT_LABEL, PanelEntry
from arm_rc_ctrl.experiments.repetition_recipes import AUGMENTATION_ANCHOR, ArmSpec, panel_arms
from arm_rc_ctrl.experiments.run_record import RunArrays, write_run
from arm_rc_ctrl.experiments.simulation import (
    GENERATOR_CHANNELS,
    RESIDUAL_CHANNELS,
    CheckedState,
    resolve_velocity_abort,
    simulate,
)
from arm_rc_ctrl.experiments.termination import Outcome, Termination
from arm_rc_ctrl.experiments.velocity_diagnostics import VelocityDiagnostics, velocity_diagnostics
from arm_rc_ctrl.metrics.recovery import SATURATION_BOUND
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_bytes,
    sha256_file,
)
from arm_rc_ctrl.rc.augment import generate_augmentation
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.rc.recipe import DatasetSource, derivative_config
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot, open_storage
from arm_rc_ctrl.validation import is_hex

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from arm_rc_ctrl.controllers.tracking import TrackerConfig
    from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.scenario import ScenarioConfig

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "MODEL_STATUSES",
    "PAIR_STATUSES",
    "POINTER_SCHEMA",
    "REPORTS_PREFIX",
    "AugmentationBankRecord",
    "BudgetExceededError",
    "EvaluationConditions",
    "EvidencePointer",
    "ExecutionBudget",
    "ModelEvidence",
    "ModelSweepTiming",
    "NumericalExceptionBinding",
    "PairRecord",
    "PilotRunner",
    "PreparedRun",
    "RepetitionEvaluationConfig",
    "ReplayBank",
    "ReplayConditions",
    "RunArtifact",
    "RunTiming",
    "SimulationLimits",
    "augmentation_bank_record",
    "check_bank_prefix",
    "load_evaluation_config",
    "load_model_evidence",
    "load_pointer",
    "load_replay_bank",
    "main",
    "numerical_binding",
    "pointer_name",
    "prepare_runner",
]

EVALUATION_SCHEMA_VERSION: Final = 1
REPORTS_PREFIX: Final = "armrc://reports/task_1a_repetition_v1"
POINTER_SCHEMA: Final = "task-1a-repetition-evidence"
PAIR_STATUSES: Final = ("completed", "infeasible", "replay_blocked", "unexecuted")
MODEL_STATUSES: Final = ("feasible", "rc_gate_failure", "replay_blocked", "training_failure")
PROGRESS_FILE: Final = "progress.json"
_SHA256_HEX: Final = 64
_MODULE: Final = "arm_rc_ctrl.experiments.repetition_evaluation"
C11_CAVEAT: Final = (
    "C11: the feasible-middle absolute R/K65 fit did not demonstrate numerical agreement with S-effective within "
    "the approved tolerance (largest prediction difference 2.93e-8 rad); the fit is retained unchanged, the "
    "exact-arithmetic ridge identity stands, and this small discrepancy does not establish that closed-loop "
    "behavior is unaffected."
)


# --- evaluation configuration ---------------------------------------------------------------


@dataclass(frozen=True)
class SimulationLimits:
    """The simulation-only abort limits of the pilot (D5, C2)."""

    velocity_abort: tuple[float, ...]
    """Per-joint measured speed abort (rad/s), symmetric, applied to every arm throughout the run."""

    def __post_init__(self) -> None:
        """Positive, finite bounds."""
        if not self.velocity_abort or any(not (math.isfinite(v) and v > 0) for v in self.velocity_abort):
            msg = f"simulation.velocity_abort must be positive finite per-joint bounds, got {self.velocity_abort!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RepetitionEvaluationConfig:
    """``configs/evaluations/task_1a_repetition_dev_v1.toml``: the locked development levels plus the abort."""

    name: str
    development: Path
    """The recovery development levels (scenarios, seeds, trackers' envelope); never a confirmatory file."""
    simulation: SimulationLimits

    def __post_init__(self) -> None:
        """Named, and never pointed at a confirmatory file."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if "confirmatory" in self.development.name:
            msg = f"the pilot evaluates development levels only, not {self.development.name!r}"
            raise ValueError(msg)


def load_evaluation_config(path: Path) -> RepetitionEvaluationConfig:
    """Load and validate the pilot's evaluation configuration."""
    return load_config(path, RepetitionEvaluationConfig)


# --- identities ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayConditions:
    """Everything that decides a direct-replay run: scenarios, trackers, limits, warm-up, dataset, environment.

    The identity of these conditions keys the replay bank; changing the
    velocity abort, the warm-up, a tracker, or the execution environment
    changes the key, so no replay is ever reused across conditions (plan
    section 7.1).
    """

    evaluation_name: str
    evaluation_file: str
    evaluation_sha256: str
    development_file: str
    development_sha256: str
    scenario_file: str
    scenario_sha256: str
    dataset: DatasetSource
    trackers: dict[str, str]
    """SHA-256 of each frozen tracker's gains, by name."""
    tracker_order: tuple[str, ...]
    """The trackers in evaluation order (the JSON form sorts mapping keys, so the order is explicit)."""
    warmup_s: float
    velocity_abort: tuple[float, ...]
    historical_velocity_limit: tuple[float, ...]
    saturation_bound: float
    scenario_ids: tuple[str, ...]
    execution_identity: str

    def __post_init__(self) -> None:
        """Digests and counts are well-formed."""
        for name in ("evaluation_sha256", "development_sha256", "scenario_sha256", "execution_identity"):
            if not is_hex(getattr(self, name), _SHA256_HEX):
                msg = f"{name} must be 64 lowercase hex characters"
                raise ValueError(msg)
        if not self.trackers or not self.scenario_ids or len(set(self.scenario_ids)) != len(self.scenario_ids):
            msg = "conditions need at least one tracker and distinct scenario ids"
            raise ValueError(msg)
        if len(self.tracker_order) != len(self.trackers) or set(self.tracker_order) != set(self.trackers):
            msg = "tracker_order must list every tracker exactly once"
            raise ValueError(msg)
        if len(self.velocity_abort) != len(self.historical_velocity_limit):
            msg = "velocity_abort and historical_velocity_limit must cover the same joints"
            raise ValueError(msg)

    @property
    def identity(self) -> str:
        """SHA-256 of the canonical JSON of these conditions (the replay bank key)."""
        return sha256_bytes(canonical_json(to_mapping(self)).encode("utf-8"))

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        """Every ``(scenario_id, tracker)`` pair in the deterministic evaluation order."""
        return tuple((scenario, tracker) for scenario in self.scenario_ids for tracker in self.tracker_order)


@dataclass(frozen=True)
class EvaluationConditions:
    """The replay conditions plus the RC-side settings of one model evaluation (estimator, settling band)."""

    replay: ReplayConditions
    estimator: EstimatorSpec
    settling_band_rad: float

    @property
    def identity(self) -> str:
        """SHA-256 of the canonical JSON of these conditions (with the fit identity, the model evaluation key)."""
        return sha256_bytes(canonical_json(to_mapping(self)).encode("utf-8"))

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        """The replay conditions' pairs."""
        return self.replay.pairs

    @property
    def velocity_abort(self) -> tuple[float, ...]:
        """The applied per-joint speed abort."""
        return self.replay.velocity_abort

    @property
    def warmup_s(self) -> float:
        """The warm-up of the evaluated entry."""
        return self.replay.warmup_s


# --- records --------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunArtifact:
    """One persisted run: identity, location, digests, and size."""

    artifact_id: str
    uri: str
    sha256: str
    size: int
    arrays_sha256: str

    def __post_init__(self) -> None:
        """Digests are well-formed."""
        if not is_hex(self.sha256, _SHA256_HEX) or not is_hex(self.arrays_sha256, _SHA256_HEX) or self.size < 0:
            msg = "a run artifact needs its two digests and a non-negative size"
            raise ValueError(msg)
        ArtifactUri.parse(self.uri)

    @property
    def reference(self) -> ArtifactReference:
        """The payload reference of the run summary."""
        return ArtifactReference(self.uri, self.sha256, self.size)


@dataclass(frozen=True)
class RunTiming:
    """Wall time and size of one run this runner simulated and persisted (M3REP-005 measurements)."""

    arm: str
    label: str
    """The model label (``<entry>/<arm>``) or ``replay:<bank identity prefix>``."""
    scenario_id: str
    tracker: str
    rows: int
    simulate_seconds: float
    persist_seconds: float
    run_bytes: int

    def __post_init__(self) -> None:
        """Figures are non-negative."""
        if min(self.rows, self.simulate_seconds, self.persist_seconds, self.run_bytes) < 0:
            msg = "run timings are non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class PairRecord:
    """One scenario/tracker pair of one arm with its status, run, outcome, metrics, and speed diagnostics."""

    index: int
    scenario_id: str
    kind: str
    tracker: str
    arm: str
    """``rc`` or ``replay``."""
    status: str
    run: RunArtifact | None = None
    rc: RecoveryComponent | None = None
    replay: ReplayComponent | None = None
    velocity: VelocityDiagnostics | None = None
    crossed_historical: bool | None = None
    timing: RunTiming | None = None
    """The measured cost of the run when this pair was simulated (M3REP-005); resumed pairs keep theirs."""

    def __post_init__(self) -> None:
        """The status and arm agree with the attached records."""
        if self.status not in PAIR_STATUSES or self.arm not in ("rc", "replay"):
            msg = (
                f"pair status must be one of {PAIR_STATUSES} and arm 'rc' or 'replay', got {self.status!r}/{self.arm!r}"
            )
            raise ValueError(msg)
        simulated = self.status in ("completed", "infeasible")
        attached = (self.run is not None, self.velocity is not None, self.timing is not None)
        if any(flag != simulated for flag in attached):
            msg = (
                f"{self.scenario_id} [{self.tracker}] {self.arm}: a simulated pair carries its run, diagnostics, "
                "and timing"
            )
            raise ValueError(msg)
        if (self.arm == "rc" and self.replay is not None) or (self.arm == "replay" and self.rc is not None):
            msg = "an rc pair carries an rc component only and a replay pair a replay component only"
            raise ValueError(msg)
        component = self.rc if self.arm == "rc" else self.replay
        if self.status == "unexecuted" and component is not None:
            msg = "an unexecuted pair carries no component"
            raise ValueError(msg)
        if self.status in ("completed", "infeasible", "replay_blocked") and component is None:
            msg = f"a {self.status} pair carries its component"
            raise ValueError(msg)
        if component is not None:
            feasible = component.feasible
            expected = (
                "completed" if feasible else ("replay_blocked" if self.status == "replay_blocked" else "infeasible")
            )
            if self.status != expected:
                msg = f"pair status {self.status!r} contradicts the component (feasible={feasible})"
                raise ValueError(msg)
        if self.velocity is not None and self.crossed_historical != self.velocity.crossed_historical:
            msg = "crossed_historical must equal the diagnostics' flag"
            raise ValueError(msg)


@dataclass(frozen=True)
class ReplayBank:
    """The direct-replay baselines of every scenario/tracker pair under one set of conditions."""

    experiment: str
    identity: str
    conditions: ReplayConditions
    pairs: tuple[PairRecord, ...]
    complete: bool
    execution: ExecutionRecord
    provenance: ProvenanceRecord
    schema_version: int = field(default=EVALUATION_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """The identity is the conditions' and completeness re-derives from the pairs."""
        if self.schema_version != EVALUATION_SCHEMA_VERSION or self.experiment != EXPERIMENT_LABEL:
            msg = "unsupported replay bank schema or experiment"
            raise ValueError(msg)
        if self.identity != self.conditions.identity:
            msg = "a replay bank's identity is its conditions' identity"
            raise ValueError(msg)
        keys = [(p.scenario_id, p.tracker) for p in self.pairs]
        if len(set(keys)) != len(keys) or any(p.arm != "replay" for p in self.pairs):
            msg = "a replay bank holds distinct replay pairs"
            raise ValueError(msg)
        if self.complete != (set(keys) == set(self.conditions.pairs) and all(p.run is not None for p in self.pairs)):
            msg = "complete contradicts the pairs"
            raise ValueError(msg)

    def get(self, scenario_id: str, tracker: str) -> PairRecord | None:
        """The replay pair, when already evaluated."""
        for pair in self.pairs:
            if pair.scenario_id == scenario_id and pair.tracker == tracker:
                return pair
        return None


@dataclass(frozen=True)
class AugmentationBankRecord:
    """The accepted synthetic episodes an augmented arm trained on (plan section 5.3)."""

    family: str
    n_synthetic: int
    attempt_budget: int
    attempts_used: int
    accepted_attempts: tuple[int, ...]
    rejections: int
    episode_sha256: tuple[str, ...]
    """Digest of each accepted episode's ``q`` and ``dq`` arrays, in acceptance order."""

    def __post_init__(self) -> None:
        """Counts agree."""
        if len(self.accepted_attempts) != self.n_synthetic or len(self.episode_sha256) != self.n_synthetic:
            msg = f"{self.family}: {self.n_synthetic} accepted episodes need their attempts and digests"
            raise ValueError(msg)


def augmentation_bank_record(
    samples: SampleSet, scenario: ScenarioConfig, derivative_method: str, *, family: str, n_synthetic: int
) -> AugmentationBankRecord:
    """Regenerate the fixed-anchor bank of ``family`` at ``n_synthetic`` episodes and record what was accepted."""
    spec = AUGMENTATION_ANCHOR.spec(family, n_synthetic)
    task = task_intervals_from_phases(samples.t, samples.phase)
    result = generate_augmentation(
        samples.t, samples.q, task, scenario, spec.config(), derivatives=derivative_config(derivative_method)
    )
    digests: list[str] = []
    attempts: list[int] = []
    for episode in result.episodes:
        arrays = getattr(episode, family)
        digests.append(sha256_bytes((array_digest(arrays.q) + array_digest(arrays.dq)).encode("ascii")))
        attempts.append(int(episode.attempt))
    return AugmentationBankRecord(
        family=family,
        n_synthetic=n_synthetic,
        attempt_budget=spec.attempt_budget,
        attempts_used=result.attempts_used,
        accepted_attempts=tuple(attempts),
        rejections=len(result.rejections),
        episode_sha256=tuple(digests),
    )


def check_bank_prefix(smaller: AugmentationBankRecord, larger: AugmentationBankRecord) -> bool:
    """Whether the smaller bank is a prefix of the larger one (same family, same accepted attempts and arrays)."""
    if smaller.family != larger.family or smaller.n_synthetic > larger.n_synthetic:
        return False
    n = smaller.n_synthetic
    return (
        smaller.accepted_attempts == larger.accepted_attempts[:n]
        and smaller.episode_sha256 == larger.episode_sha256[:n]
    )


@dataclass(frozen=True)
class NumericalExceptionBinding:
    """The M3REP-003 validation every manifest binds, with the accepted exception (C11)."""

    validation_file: str
    validation_sha256: str
    all_passed: bool
    comparisons: int
    comparisons_passed: int
    exception_candidate_identity: str | None
    exception_reference_identity: str | None
    caveat: str

    def __post_init__(self) -> None:
        """The exception is recorded exactly when the validation did not pass."""
        if not is_hex(self.validation_sha256, _SHA256_HEX):
            msg = "validation_sha256 must be 64 lowercase hex characters"
            raise ValueError(msg)
        if self.all_passed != (self.comparisons_passed == self.comparisons):
            msg = "all_passed contradicts the comparison counts"
            raise ValueError(msg)
        if self.all_passed == (self.exception_candidate_identity is not None) or not self.caveat.strip():
            msg = "a failed validation binds its exception identities and the caveat text"
            raise ValueError(msg)


def numerical_binding(validation_file: Path, *, root: Path) -> NumericalExceptionBinding:
    """Bind the committed numerical validation and its single accepted exception (C11)."""
    validation = load_validation(validation_file)
    failed = [c for c in validation.comparisons if not c.passed]
    if len(failed) > 1:
        msg = f"C11 accepts exactly one numerical exception; the validation retains {len(failed)} failures"
        raise ValueError(msg)
    exception = failed[0] if failed else None
    return NumericalExceptionBinding(
        validation_file=validation_file.relative_to(root).as_posix()
        if validation_file.is_relative_to(root)
        else str(validation_file),
        validation_sha256=sha256_file(validation_file),
        all_passed=validation.all_passed,
        comparisons=validation.n_comparisons,
        comparisons_passed=validation.n_comparisons_passed,
        exception_candidate_identity=None if exception is None else exception.candidate_identity,
        exception_reference_identity=None if exception is None else exception.reference_identity,
        caveat=C11_CAVEAT if exception is not None else "no numerical exception was accepted",
    )


@dataclass(frozen=True)
class FitBinding:
    """The cached fit a model evaluation used."""

    identity: str
    panel_label: str
    source_trial: int
    arm: ArmSpec
    formulation: str
    solver_alpha: float
    recipe_sha256: str
    weights_sha256: str
    fit_rmse: float
    loss_rows: int


@dataclass(frozen=True)
class ModelEvidence:
    """The evidence manifest of one model configuration (C6)."""

    experiment: str
    evaluation_identity: str
    fit: FitBinding | None
    """``None`` only for a training failure (the fit could not be produced)."""
    conditions: EvaluationConditions
    replay_bank: str
    status: str
    first_failure: str | None
    pairs: tuple[PairRecord, ...]
    n_pairs: int
    n_completed: int
    n_infeasible: int
    n_replay_blocked: int
    n_unexecuted: int
    crossed_historical: bool
    """Whether any executed RC run of this model exceeded the historical 6 rad/s limit."""
    cells: dict[str, float]
    """``<class>:<tracker>`` median early-gap ratios, only when every pair is feasible."""
    augmentation: AugmentationBankRecord | None
    augmentation_prefix_verified: bool | None
    caveats: tuple[str, ...]
    numerical: NumericalExceptionBinding
    execution: ExecutionRecord
    provenance: ProvenanceRecord
    training_failure: str | None = None
    schema_version: int = field(default=EVALUATION_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Counts, status, and the first failure re-derive from the pairs."""
        if self.schema_version != EVALUATION_SCHEMA_VERSION or self.experiment != EXPERIMENT_LABEL:
            msg = "unsupported model evidence schema or experiment"
            raise ValueError(msg)
        if self.status not in MODEL_STATUSES:
            msg = f"status must be one of {MODEL_STATUSES}, got {self.status!r}"
            raise ValueError(msg)
        if (self.status == "training_failure") != (self.fit is None) or (self.status == "training_failure") != (
            self.training_failure is not None
        ):
            msg = "a training failure has no fit and records its reason; every other status has a fit"
            raise ValueError(msg)
        self._check_pairs()
        self._check_summary()

    def _check_pairs(self) -> None:
        rc = [p for p in self.pairs if p.arm == "rc"]
        if len(rc) != len(self.pairs) or [(p.scenario_id, p.tracker) for p in rc] != list(self.conditions.pairs):
            msg = "the pairs are the rc pairs of every scenario/tracker in evaluation order"
            raise ValueError(msg)
        counts = {status: sum(1 for p in rc if p.status == status) for status in PAIR_STATUSES}
        recorded = (self.n_pairs, self.n_completed, self.n_infeasible, self.n_replay_blocked, self.n_unexecuted)
        actual = (len(rc), counts["completed"], counts["infeasible"], counts["replay_blocked"], counts["unexecuted"])
        if recorded != actual:
            msg = f"recorded pair counts {recorded} contradict the pairs {actual}"
            raise ValueError(msg)
        expected = _model_status(rc, training_failure=self.training_failure)
        if (self.status, self.first_failure) != expected:
            msg = f"status/first_failure {(self.status, self.first_failure)} contradict {expected}"
            raise ValueError(msg)
        if self.crossed_historical != any(bool(p.crossed_historical) for p in rc):
            msg = "crossed_historical contradicts the pairs"
            raise ValueError(msg)

    def _check_summary(self) -> None:
        if bool(self.cells) != (self.status == "feasible"):
            msg = "cells are recorded exactly for a feasible model"
            raise ValueError(msg)
        if not self.caveats or self.numerical.caveat not in self.caveats:
            msg = "every manifest carries the numerical validation's caveat (C11)"
            raise ValueError(msg)
        if self.fit is not None and self.fit.arm.arm.startswith("A-") != (self.augmentation is not None):
            msg = "augmented arms record their bank; other arms do not"
            raise ValueError(msg)


def _model_status(rc: Sequence[PairRecord], *, training_failure: str | None) -> tuple[str, str | None]:
    if training_failure is not None:
        return "training_failure", f"training_failure:{training_failure}"
    for pair in rc:
        if pair.status == "replay_blocked":
            reason = pair.rc.reason if pair.rc is not None else "replay_infeasible"
            return "replay_blocked", f"scenario {pair.index} [{pair.tracker}]: {reason}"
        if pair.status == "infeasible":
            reason = pair.rc.reason if pair.rc is not None else "infeasible"
            return "rc_gate_failure", f"scenario {pair.index} [{pair.tracker}]: {reason}"
        if pair.status == "unexecuted":
            msg = "an unexecuted pair precedes every failure"
            raise ValueError(msg)
    return "feasible", None


@dataclass(frozen=True)
class EvidencePointer:
    """The Git pointer of one stored manifest (model configuration or replay bank)."""

    schema: str
    experiment: str
    kind: str
    """``model`` or ``replay``."""
    identity: str
    label: str
    status: str
    payload: ArtifactReference
    n_pairs: int
    n_completed: int
    n_unexecuted: int
    first_failure: str | None = None

    def __post_init__(self) -> None:
        """Schema and kind are known."""
        if self.schema != POINTER_SCHEMA or self.experiment != EXPERIMENT_LABEL or self.kind not in ("model", "replay"):
            msg = "unsupported evidence pointer"
            raise ValueError(msg)
        if not is_hex(self.identity, _SHA256_HEX) or not self.label.strip():
            msg = "a pointer names its identity and label"
            raise ValueError(msg)


def pointer_name(kind: str, label: str) -> str:
    """The pointer file name: ``<kind>__<label with / replaced by __>.toml``."""
    return f"{kind}__{label.replace('/', '__')}.toml"


def load_pointer(path: Path) -> EvidencePointer:
    """Load a pointer record."""
    return load_record(path, EvidencePointer)


def load_model_evidence(path: Path) -> ModelEvidence:
    """Strictly rebuild a model manifest from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ModelEvidence)


def load_replay_bank(path: Path) -> ReplayBank:
    """Strictly rebuild a replay bank from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ReplayBank)


@dataclass(frozen=True)
class ModelSweepTiming:
    """Wall time of one model's sweep and how its fit was obtained."""

    fit_cache_hit: bool
    fit_seconds: float
    sweep_seconds: float


class BudgetExceededError(RuntimeError):
    """The approved execution or storage allowance would be exceeded; progress is checkpointed (C12)."""


@dataclass(frozen=True)
class ExecutionBudget:
    """The owner-approved allowances of one invocation (C12): wall time and total run/manifest storage."""

    time_seconds: float | None
    """Wall-time allowance of this invocation (``None``: unlimited)."""
    storage_bytes: int | None
    """Total run and manifest storage allowance (``None``: unlimited)."""
    storage_baseline_bytes: int = 0
    """Storage already used by earlier invocations that counts against the same allowance."""

    def __post_init__(self) -> None:
        """Allowances are non-negative."""
        negative = [
            value
            for value in (self.time_seconds, self.storage_bytes, self.storage_baseline_bytes)
            if value is not None and value < 0
        ]
        if negative:
            msg = "budget allowances must be non-negative"
            raise ValueError(msg)

    def check(self, *, elapsed_seconds: float, used_bytes: int) -> None:
        """Fail before the next run when the elapsed time or the used storage has reached an allowance."""
        if self.time_seconds is not None and elapsed_seconds >= self.time_seconds:
            msg = (
                f"the execution allowance of {self.time_seconds:.0f} s is reached after {elapsed_seconds:.0f} s; "
                "progress is checkpointed, request an extension (C12)"
            )
            raise BudgetExceededError(msg)
        total = self.storage_baseline_bytes + used_bytes
        if self.storage_bytes is not None and total >= self.storage_bytes:
            msg = (
                f"the storage allowance of {self.storage_bytes} bytes is reached at {total} bytes; progress is "
                "checkpointed, request an extension (C12)"
            )
            raise BudgetExceededError(msg)


# --- persistence --------------------------------------------------------------------------


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.staging-{os.getpid()}")
    staged.write_text(text, encoding="utf-8")
    staged.replace(path)


def _install_manifest(store: StorageRoot, directory_uri: str, text: str) -> ArtifactReference:
    """Install a content-addressed manifest next to its progress file; identical content is reused."""
    data = text.encode("utf-8")
    digest = sha256_bytes(data)
    uri = f"{directory_uri}/manifest-{digest[:12]}.json"
    target = store.path(uri, mode="write")
    if target.exists():
        if sha256_file(target) != digest:
            msg = f"{uri} exists with other content; refusing to overwrite"
            raise ValueError(msg)
    else:
        _write_atomic(target, text)
    return ArtifactReference(uri, digest, len(data))


def _existing_manifest(store: StorageRoot, directory_uri: str) -> Path | None:
    """The manifest already installed in a store directory, if any (several would be a corrupted store)."""
    directory = store.path(f"{directory_uri}/{PROGRESS_FILE}", mode="write").parent
    manifests = sorted(directory.glob("manifest-*.json"))
    if len(manifests) > 1:
        msg = f"{directory} holds {len(manifests)} manifests; completed evidence is written once"
        raise ValueError(msg)
    return manifests[0] if manifests else None


def _reference_of(path: Path, store: StorageRoot) -> ArtifactReference:
    return ArtifactReference(str(store.uri_for(path)), sha256_file(path), path.stat().st_size)


class _Progress:
    """The mutable, run-granular progress of one manifest (``progress.json`` in its store directory)."""

    def __init__(self, store: StorageRoot, directory_uri: str, identity: str) -> None:
        self.store = store
        self.directory_uri = directory_uri
        self.identity = identity
        self.path = store.path(f"{directory_uri}/{PROGRESS_FILE}", mode="write")
        self.pairs: list[PairRecord] = []
        if self.path.exists():
            mapping = cast("dict[str, object]", json.loads(self.path.read_text(encoding="utf-8")))
            if mapping.get("identity") != identity or mapping.get("schema_version") != EVALUATION_SCHEMA_VERSION:
                msg = f"{self.path} belongs to another evaluation or schema"
                raise ValueError(msg)
            self.pairs = [
                from_mapping(cast("dict[str, object]", p), PairRecord) for p in cast("list[object]", mapping["pairs"])
            ]
            self._verify_runs()

    def _verify_runs(self) -> None:
        """Every completed run in the progress still exists with its recorded digest (never re-simulated)."""
        for pair in self.pairs:
            if pair.run is None:
                continue
            path = self.store.path(pair.run.uri, mode="read")
            if path.stat().st_size != pair.run.size or sha256_file(path) != pair.run.sha256:
                msg = f"run {pair.run.artifact_id} of {pair.scenario_id} [{pair.tracker}] no longer matches its record"
                raise ValueError(msg)

    def get(self, scenario_id: str, tracker: str, arm: str) -> PairRecord | None:
        for pair in self.pairs:
            if (pair.scenario_id, pair.tracker, pair.arm) == (scenario_id, tracker, arm):
                return pair
        return None

    def add(self, pair: PairRecord) -> None:
        if self.get(pair.scenario_id, pair.tracker, pair.arm) is not None:
            msg = f"{pair.scenario_id} [{pair.tracker}] {pair.arm} is already recorded; completed evidence is immutable"
            raise ValueError(msg)
        self.pairs.append(pair)
        self.save()

    def save(self) -> None:
        mapping = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "identity": self.identity,
            "pairs": [to_mapping(p) for p in self.pairs],
        }
        _write_atomic(self.path, json.dumps(mapping, indent=1, sort_keys=True))


# --- the runner ---------------------------------------------------------------------------


type SimulateFn = Callable[..., tuple[RunArrays, Termination]]


@dataclass(frozen=True)
class _Prepared:
    entry: PanelEntry
    conditions: EvaluationConditions
    activation_s: float
    duration_s: float
    dwell_start_s: float


class PilotRunner:
    """Evaluate panel entries under arms with paired replay, persisted at run granularity."""

    def __init__(
        self,
        *,
        store: StorageRoot,
        inputs: FitInputs,
        dataset: RecoveryDatasetRecord,
        evaluation: RepetitionEvaluationConfig,
        evaluation_file: Path,
        root: Path,
        execution: ExecutionRecord,
        provenance: ProvenanceRecord,
        numerical: NumericalExceptionBinding,
        scenarios: Sequence[RobustnessScenario],
        trackers: Mapping[str, TrackerConfig],
        tracker_digests: Mapping[str, str],
        development_sha256: str,
        settling_band_rad: float = DEFAULT_SETTLING_BAND_RAD,
        saturation_bound: float = SATURATION_BOUND,
        command: str = f"python -m {_MODULE} run",
        license_label: str = "LicenseRef-Private",
        access: str = "private",
        simulate_fn: SimulateFn | None = None,
        log: Callable[[str], None] = print,
        budget: ExecutionBudget | None = None,
    ) -> None:
        """Bind the store, the panel inputs, the resolved conditions, and the persistence settings."""
        if inputs.execution_identity != execution.identity:
            msg = "the fit inputs must bind the runner's execution identity (C10)"
            raise ValueError(msg)
        if set(trackers) != set(tracker_digests) or not trackers:
            msg = "trackers and their digests must name the same frozen baselines"
            raise ValueError(msg)
        self.store = store
        self.fits = FitStore(store)
        self.inputs = inputs
        self.dataset = dataset
        self.evaluation = evaluation
        self.evaluation_file = evaluation_file
        self.root = root
        self.execution = execution
        self.provenance = provenance
        self.numerical = numerical
        self.scenarios = tuple(scenarios)
        self.trackers = dict(trackers)
        self.tracker_digests = dict(tracker_digests)
        self.development_sha256 = development_sha256
        self.settling_band_rad = settling_band_rad
        self.saturation_bound = saturation_bound
        self.command = command
        self.license_label = license_label
        self.access = access
        self.simulate: SimulateFn = simulate if simulate_fn is None else simulate_fn
        self.log = log
        self.scenario = inputs.scenario
        self.reference = inputs.samples
        self.velocity_abort = resolve_velocity_abort(self.scenario, evaluation.simulation.velocity_abort)
        self._banks: dict[str, ReplayBank] = {}
        self.pointers: list[tuple[str, str, EvidencePointer]] = []
        """``(kind, label, pointer)`` of every manifest this runner completed, written by :meth:`write_pointers`."""
        self.run_timings: list[RunTiming] = []
        """Every run this runner simulated (resumed runs are not re-timed)."""
        self.model_timings: dict[str, ModelSweepTiming] = {}
        """Sweep timing by evaluation identity for the models this runner evaluated."""
        self.manifest_bytes = 0
        """Bytes of the manifests this runner installed."""
        self._last_run_bytes = 0
        self.budget = budget
        self.started = time.perf_counter()
        self.used_bytes = 0
        """Run and manifest bytes this runner wrote."""

    # -- conditions -------------------------------------------------------------------------

    def replay_conditions(self, entry: PanelEntry) -> ReplayConditions:
        """The replay conditions of ``entry`` (its warm-up; everything else shared across the panel)."""
        return ReplayConditions(
            evaluation_name=self.evaluation.name,
            evaluation_file=_relative(self.evaluation_file, self.root),
            evaluation_sha256=sha256_file(self.evaluation_file),
            development_file=_relative(self.evaluation.development, self.root),
            development_sha256=self.development_sha256,
            scenario_file=_relative(self.inputs.scenario_file, self.root),
            scenario_sha256=sha256_file(self.inputs.scenario_file),
            dataset=self.inputs.source,
            trackers=dict(self.tracker_digests),
            tracker_order=tuple(self.trackers),
            warmup_s=entry.warmup_s,
            velocity_abort=self.velocity_abort,
            historical_velocity_limit=tuple(self.scenario.limits.velocity),
            saturation_bound=self.saturation_bound,
            scenario_ids=tuple(s.scenario_id for s in self.scenarios),
            execution_identity=self.execution.identity,
        )

    def conditions(self, entry: PanelEntry) -> EvaluationConditions:
        """The resolved model-evaluation conditions of ``entry`` (replay conditions plus its estimator)."""
        return EvaluationConditions(
            replay=self.replay_conditions(entry), estimator=entry.estimator, settling_band_rad=self.settling_band_rad
        )

    def _prepare(self, entry: PanelEntry) -> _Prepared:
        activation = entry.warmup_s
        task = task_intervals_from_phases(self.reference.t, self.reference.phase)
        return _Prepared(
            entry=entry,
            conditions=self.conditions(entry),
            activation_s=activation,
            duration_s=activation + float(self.reference.t[-1]),
            dwell_start_s=task.dwell[0],
        )

    def _start(self, case: RobustnessScenario) -> tuple[float, ...]:
        return case.initial_q(self.dataset.q0_ref)

    def _run_force(self, case: RobustnessScenario, activation: float) -> ForcePulse | None:
        pulse = case.pulse
        if pulse is None:
            return None
        return ForcePulse(start_s=pulse.start_s + activation, duration_s=pulse.duration_s, force=pulse.force)

    # -- persistence of one run -------------------------------------------------------------

    def _persist(
        self,
        arrays: RunArrays,
        termination: Termination,
        *,
        prepared: _Prepared,
        case: RobustnessScenario,
        tracker: str,
        arm: str,
        seeds: dict[str, int],
        run_force: ForcePulse | None,
    ) -> RunArtifact:
        criteria = recovery_outcome(
            self.scenario, self.reference, arrays, termination, activation_s=prepared.activation_s
        )
        provenance = replace(self.provenance, seeds=dict(seeds))
        pointer, summary, _directory = write_run(
            self.store,
            arrays,
            kind="simulation",
            method=f"{arm}+{tracker}",
            scenario=self.scenario.name,
            control_period_s=self.scenario.timing.dt,
            duration_s=prepared.duration_s,
            target=self.scenario.task.target,
            task_code=(),
            disturbances=() if run_force is None else (run_force.to_disturbance(),),
            termination=termination,
            outcome=Outcome(termination, criteria),
            provenance=provenance,
            license_label=self.license_label,
            access=cast("Any", self.access),
            command=self.command,
            sources=(self.inputs.source.artifact_id,),
            activation_s=prepared.activation_s,
            reuse_identical=True,
            notes=(
                f"{EXPERIMENT_LABEL} {arm} arm: {case.scenario_id} [{tracker}] for {prepared.entry.label}; velocity "
                f"abort {list(self.velocity_abort)} rad/s (historical limit {list(self.scenario.limits.velocity)})."
            ),
        )
        self._last_run_bytes = sum(p.stat().st_size for p in _directory.iterdir() if p.is_file())
        self.used_bytes += self._last_run_bytes
        return RunArtifact(
            artifact_id=pointer.artifact.artifact_id,
            uri=pointer.artifact.payload.uri,
            sha256=pointer.artifact.payload.sha256,
            size=pointer.artifact.payload.size,
            arrays_sha256=summary.arrays_sha256,
        )

    def _check_budget(self) -> None:
        """Refuse to start another run once an allowance is reached (every completed run is already persisted)."""
        if self.budget is not None:
            self.budget.check(elapsed_seconds=time.perf_counter() - self.started, used_bytes=self.used_bytes)

    def _time_run(
        self,
        *,
        arm: str,
        label: str,
        case: RobustnessScenario,
        tracker: str,
        rows: int,
        simulate_s: float,
        persist_s: float,
    ) -> RunTiming:
        timing = RunTiming(
            arm=arm,
            label=label,
            scenario_id=case.scenario_id,
            tracker=tracker,
            rows=rows,
            simulate_seconds=simulate_s,
            persist_seconds=persist_s,
            run_bytes=self._last_run_bytes,
        )
        self.run_timings.append(timing)
        return timing

    def _diagnostics(
        self, states: Sequence[CheckedState], termination: Termination, prepared: _Prepared
    ) -> VelocityDiagnostics:
        return velocity_diagnostics(
            states,
            termination,
            historical=self.scenario.limits.velocity,
            abort_limit=self.velocity_abort,
            activation_s=prepared.activation_s,
            dwell_start_s=prepared.dwell_start_s,
            dt=self.scenario.timing.dt,
        )

    # -- replay bank ------------------------------------------------------------------------

    def _bank_uri(self, conditions: ReplayConditions) -> str:
        return f"{REPORTS_PREFIX}/replay/{conditions.identity}"

    def _replay_pair(self, prepared: _Prepared, index: int, case: RobustnessScenario, tracker: str) -> PairRecord:
        start = self._start(case)
        run_force = self._run_force(case, prepared.activation_s)
        held = HeldTaskReference.from_samples(
            self.reference,
            activation_s=prepared.activation_s,
            interpolation=self.dataset.preprocessing.interpolation,
            hold=np.asarray(start, dtype=np.float64),
        )
        controller = LimitedTracker(cast("Any", held), self.trackers[tracker], self.scenario.limits.torque)
        self._check_budget()
        states: list[CheckedState] = []
        started = time.perf_counter()
        arrays, termination = self.simulate(
            self.scenario,
            controller,
            duration_s=prepared.duration_s,
            initial_q=start,
            force=run_force,
            velocity_abort=self.velocity_abort,
            checked_states=states,
        )
        simulated = time.perf_counter()
        component = replay_component(
            index,
            case,
            tracker,
            start,
            self.scenario,
            self.reference,
            arrays,
            termination,
            activation_s=prepared.activation_s,
            bound=self.saturation_bound,
        )
        run = self._persist(
            arrays,
            termination,
            prepared=prepared,
            case=case,
            tracker=tracker,
            arm="replay",
            seeds={},
            run_force=run_force,
        )
        timing = self._time_run(
            arm="replay",
            label=f"replay:{prepared.conditions.replay.identity[:12]}",
            case=case,
            tracker=tracker,
            rows=arrays.n_samples,
            simulate_s=simulated - started,
            persist_s=time.perf_counter() - simulated,
        )
        diagnostics = self._diagnostics(states, termination, prepared)
        return PairRecord(
            index=index,
            scenario_id=case.scenario_id,
            kind=str(case.kind),
            tracker=tracker,
            arm="replay",
            status="completed" if component.feasible else "infeasible",
            run=run,
            replay=component,
            velocity=diagnostics,
            crossed_historical=diagnostics.crossed_historical,
            timing=timing,
        )

    def replay_bank(self, entry: PanelEntry) -> ReplayBank:
        """The complete replay bank of ``entry``'s conditions, run or resumed as needed and stored."""
        prepared = self._prepare(entry)
        conditions = prepared.conditions.replay
        cached = self._banks.get(conditions.identity)
        if cached is not None:
            return cached
        progress = _Progress(self.store, self._bank_uri(conditions), conditions.identity)
        ran = False
        for index, case in enumerate(self.scenarios):
            for tracker in self.trackers:
                if progress.get(case.scenario_id, tracker, "replay") is None:
                    self.log(f"replay {conditions.identity[:12]}: {case.scenario_id} [{tracker}]")
                    progress.add(self._replay_pair(prepared, index, case, tracker))
                    ran = True
        existing = _existing_manifest(self.store, self._bank_uri(conditions))
        if not ran and existing is not None:
            stored = load_replay_bank(existing)
            if stored.identity != conditions.identity:
                msg = f"{existing} holds the bank {stored.identity[:12]}, not {conditions.identity[:12]}"
                raise ValueError(msg)
            self._register_bank(entry, stored, _reference_of(existing, self.store))
            return stored
        ordered = tuple(
            cast("PairRecord", progress.get(case.scenario_id, tracker, "replay"))
            for case in self.scenarios
            for tracker in self.trackers
        )
        bank = ReplayBank(
            experiment=EXPERIMENT_LABEL,
            identity=conditions.identity,
            conditions=conditions,
            pairs=ordered,
            complete=True,
            execution=self.execution,
            provenance=self.provenance,
        )
        payload = _install_manifest(self.store, self._bank_uri(conditions), canonical_json(to_mapping(bank)) + "\n")
        self.manifest_bytes += payload.size
        self.used_bytes += payload.size
        self._register_bank(entry, bank, payload)
        return bank

    def _register_bank(self, entry: PanelEntry, bank: ReplayBank, payload: ArtifactReference) -> None:
        label = f"warmup-{entry.warmup_s:g}s"
        self.pointers.append(
            (
                "replay",
                label,
                EvidencePointer(
                    schema=POINTER_SCHEMA,
                    experiment=EXPERIMENT_LABEL,
                    kind="replay",
                    identity=bank.identity,
                    label=label,
                    status="complete",
                    payload=payload,
                    n_pairs=len(bank.pairs),
                    n_completed=sum(1 for p in bank.pairs if p.status == "completed"),
                    n_unexecuted=0,
                ),
            )
        )
        self._banks[bank.identity] = bank

    # -- model evaluation -------------------------------------------------------------------

    def _model_uri(self, identity: str) -> str:
        return f"{REPORTS_PREFIX}/models/{identity}"

    def _controllers(self, cached: CachedFit, prepared: _Prepared) -> dict[str, GeneratorTrackingController]:
        lower = np.array([link.q_min for link in self.scenario.robot.links], dtype=np.float64)
        upper = np.array([link.q_max for link in self.scenario.robot.links], dtype=np.float64)
        controllers: dict[str, GeneratorTrackingController] = {}
        for name, gains in self.trackers.items():
            estimator = CausalDerivativeEstimator(
                prepared.entry.estimator.config(self.scenario.timing.dt), self.scenario.dof
            )
            generator = RcTargetGenerator(
                cached.model,
                cached.recipe.encoder(),
                estimator,
                position_bounds=(lower, upper),
                output=cached.recipe.output,
            )
            controllers[name] = GeneratorTrackingController(
                generator, gains, self.scenario.limits.torque, hold_until_s=prepared.activation_s
            )
        return controllers

    def _rc_pair(
        self,
        prepared: _Prepared,
        cached: CachedFit,
        controller: GeneratorTrackingController,
        index: int,
        case: RobustnessScenario,
        tracker: str,
        replay: PairRecord,
    ) -> PairRecord:
        start = self._start(case)
        run_force = self._run_force(case, prepared.activation_s)
        replay_component_record = cast("ReplayComponent", replay.replay)
        if case.kind in RATIO_CLASSES and not replay_component_record.feasible:
            component = blocked_component(index, case, tracker, start, replay_component_record)
            return PairRecord(
                index=index,
                scenario_id=case.scenario_id,
                kind=str(case.kind),
                tracker=tracker,
                arm="rc",
                status="replay_blocked",
                rc=component,
            )
        self._check_budget()
        states: list[CheckedState] = []
        channels = RESIDUAL_CHANNELS if cached.recipe.output == "increment" else GENERATOR_CHANNELS
        started = time.perf_counter()
        arrays, termination = self.simulate(
            self.scenario,
            controller,
            duration_s=prepared.duration_s,
            initial_q=start,
            force=run_force,
            channels=channels,
            velocity_abort=self.velocity_abort,
            checked_states=states,
        )
        simulated = time.perf_counter()
        component = recovery_component(
            index,
            case,
            tracker,
            start,
            (arrays, termination),
            scenario=self.scenario,
            reference=self.reference,
            activation_s=prepared.activation_s,
            bound=self.saturation_bound,
            replay=replay_component_record,
            boundary_jump=controller.boundary_jump,
            settling_band_rad=self.settling_band_rad,
        )
        run = self._persist(
            arrays,
            termination,
            prepared=prepared,
            case=case,
            tracker=tracker,
            arm="rc",
            seeds={"reservoir": cached.recipe.esn.reservoir.seed},
            run_force=run_force,
        )
        timing = self._time_run(
            arm="rc",
            label=f"{prepared.entry.label}/{cached.record.arm.label}",
            case=case,
            tracker=tracker,
            rows=arrays.n_samples,
            simulate_s=simulated - started,
            persist_s=time.perf_counter() - simulated,
        )
        diagnostics = self._diagnostics(states, termination, prepared)
        return PairRecord(
            index=index,
            scenario_id=case.scenario_id,
            kind=str(case.kind),
            tracker=tracker,
            arm="rc",
            status="completed" if component.feasible else "infeasible",
            run=run,
            rc=component,
            velocity=diagnostics,
            crossed_historical=diagnostics.crossed_historical,
            timing=timing,
        )

    def _unexecuted(self, index: int, case: RobustnessScenario, tracker: str) -> PairRecord:
        return PairRecord(
            index=index,
            scenario_id=case.scenario_id,
            kind=str(case.kind),
            tracker=tracker,
            arm="rc",
            status="unexecuted",
        )

    def _augmentation(self, arm: ArmSpec) -> tuple[AugmentationBankRecord | None, bool | None]:
        if not arm.arm.startswith("A-"):
            return None, None
        family = "non_decaying" if arm.arm == "A-non-decaying" else "contractive"
        record = augmentation_bank_record(
            self.reference,
            self.scenario,
            self.inputs.preprocessing.derivative_method,
            family=family,
            n_synthetic=arm.count - 1,
        )
        verified = True
        for other in (16, 32, 64):
            if other >= arm.count - 1:
                continue
            smaller = augmentation_bank_record(
                self.reference,
                self.scenario,
                self.inputs.preprocessing.derivative_method,
                family=family,
                n_synthetic=other,
            )
            verified = verified and check_bank_prefix(smaller, record)
        return record, verified

    def evaluate(self, entry: PanelEntry, arm: ArmSpec) -> ModelEvidence:
        """Evaluate ``entry`` under ``arm`` (resuming a persisted sweep) and store its manifest."""
        if not arm.behavioral:
            msg = f"{arm.label} is a numerical reference, not a behavioral arm"
            raise ValueError(msg)
        prepared = self._prepare(entry)
        conditions = prepared.conditions
        bank = self.replay_bank(entry)
        sweep_started = time.perf_counter()
        try:
            cached = self.fits.fit_or_load(entry, arm, self.inputs)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            identity = self._failure_identity(entry, arm, conditions)
            stored = self._stored_model(identity, entry, arm)
            if stored is not None:
                return stored
            return self._store_model(
                entry, arm, None, conditions, bank, (), training_failure=f"{type(exc).__name__}: {exc}"
            )
        identity = sha256_bytes(f"{cached.record.identity}:{conditions.identity}".encode("ascii"))
        progress = _Progress(self.store, self._model_uri(identity), identity)
        controllers = self._controllers(cached, prepared)
        pairs: list[PairRecord] = []
        stopped = False
        ran = False
        for index, case in enumerate(self.scenarios):
            for tracker in self.trackers:
                existing = progress.get(case.scenario_id, tracker, "rc")
                if existing is not None:
                    pairs.append(existing)
                    stopped = stopped or existing.status in ("infeasible", "replay_blocked")
                    continue
                if stopped:
                    pair = self._unexecuted(index, case, tracker)
                else:
                    self.log(f"{entry.label}/{arm.label}: {case.scenario_id} [{tracker}]")
                    replay = cast("PairRecord", bank.get(case.scenario_id, tracker))
                    pair = self._rc_pair(prepared, cached, controllers[tracker], index, case, tracker, replay)
                    stopped = pair.status in ("infeasible", "replay_blocked")
                progress.add(pair)
                pairs.append(pair)
                ran = True
        self.model_timings[identity] = ModelSweepTiming(
            fit_cache_hit=cached.cache_hit,
            fit_seconds=cached.record.fit_seconds,
            sweep_seconds=time.perf_counter() - sweep_started,
        )
        if not ran:
            stored = self._stored_model(identity, entry, arm)
            if stored is not None:
                return stored
        return self._store_model(entry, arm, cached, conditions, bank, tuple(pairs), training_failure=None)

    def _failure_identity(self, entry: PanelEntry, arm: ArmSpec, conditions: EvaluationConditions) -> str:
        return sha256_bytes(f"training-failure:{entry.label}/{arm.label}:{conditions.identity}".encode())

    def _stored_model(self, identity: str, entry: PanelEntry, arm: ArmSpec) -> ModelEvidence | None:
        """A model manifest already completed under ``identity`` (reused with its own provenance, never rewritten)."""
        existing = _existing_manifest(self.store, self._model_uri(identity))
        if existing is None:
            return None
        stored = load_model_evidence(existing)
        if stored.evaluation_identity != identity:
            msg = f"{existing} holds the evaluation {stored.evaluation_identity[:12]}, not {identity[:12]}"
            raise ValueError(msg)
        self._register_model(entry, arm, stored, _reference_of(existing, self.store))
        return stored

    def _store_model(
        self,
        entry: PanelEntry,
        arm: ArmSpec,
        cached: CachedFit | None,
        conditions: EvaluationConditions,
        bank: ReplayBank,
        pairs: tuple[PairRecord, ...],
        *,
        training_failure: str | None,
    ) -> ModelEvidence:
        if cached is None:
            identity = self._failure_identity(entry, arm, conditions)
            fit_binding = None
            pairs = tuple(
                self._unexecuted(index, case, tracker)
                for index, case in enumerate(self.scenarios)
                for tracker in self.trackers
            )
        else:
            identity = sha256_bytes(f"{cached.record.identity}:{conditions.identity}".encode("ascii"))
            fit_binding = FitBinding(
                identity=cached.record.identity,
                panel_label=entry.label,
                source_trial=entry.source_trial,
                arm=arm,
                formulation=arm.formulation,
                solver_alpha=cached.recipe.solver_alpha,
                recipe_sha256=cached.record.recipe_sha256,
                weights_sha256=cached.record.weights_sha256,
                fit_rmse=cached.record.fit.rmse,
                loss_rows=cached.record.fit.loss_rows,
            )
        status, first_failure = _model_status(pairs, training_failure=training_failure)
        cells: dict[str, float] = {}
        if status == "feasible":
            ratios: dict[str, list[float]] = {}
            for pair in pairs:
                component = pair.rc
                if component is not None and component.gap_ratio is not None:
                    ratios.setdefault(f"{pair.kind}:{pair.tracker}", []).append(component.gap_ratio)
            cells = {key: float(median(values)) for key, values in sorted(ratios.items())}
        augmentation, prefix = (None, None) if cached is None else self._augmentation(arm)
        caveats = [self.numerical.caveat]
        if (
            self.numerical.exception_candidate_identity is not None
            and cached is not None
            and cached.record.identity == self.numerical.exception_candidate_identity
        ):
            caveats.append(
                "This is the configuration of the accepted numerical exception (C11): its R fit is evaluated as fitted."
            )
        counts = {s: sum(1 for p in pairs if p.status == s) for s in PAIR_STATUSES}
        evidence = ModelEvidence(
            experiment=EXPERIMENT_LABEL,
            evaluation_identity=identity,
            fit=fit_binding,
            conditions=conditions,
            replay_bank=bank.identity,
            status=status,
            first_failure=first_failure,
            pairs=pairs,
            n_pairs=len(pairs),
            n_completed=counts["completed"],
            n_infeasible=counts["infeasible"],
            n_replay_blocked=counts["replay_blocked"],
            n_unexecuted=counts["unexecuted"],
            crossed_historical=any(bool(p.crossed_historical) for p in pairs),
            cells=cells,
            augmentation=augmentation,
            augmentation_prefix_verified=prefix,
            caveats=tuple(caveats),
            numerical=self.numerical,
            execution=self.execution,
            provenance=self.provenance,
            training_failure=training_failure,
        )
        payload = _install_manifest(self.store, self._model_uri(identity), canonical_json(to_mapping(evidence)) + "\n")
        self.manifest_bytes += payload.size
        self.used_bytes += payload.size
        self._register_model(entry, arm, evidence, payload)
        return evidence

    def _register_model(
        self, entry: PanelEntry, arm: ArmSpec, evidence: ModelEvidence, payload: ArtifactReference
    ) -> None:
        label = f"{entry.label}/{arm.label}"
        self.pointers.append(
            (
                "model",
                label,
                EvidencePointer(
                    schema=POINTER_SCHEMA,
                    experiment=EXPERIMENT_LABEL,
                    kind="model",
                    identity=evidence.evaluation_identity,
                    label=label,
                    status=evidence.status,
                    payload=payload,
                    n_pairs=evidence.n_pairs,
                    n_completed=evidence.n_completed,
                    n_unexecuted=evidence.n_unexecuted,
                    first_failure=evidence.first_failure,
                ),
            )
        )

    def run(self, entries: Sequence[PanelEntry], arms: Sequence[ArmSpec]) -> list[ModelEvidence]:
        """Evaluate every entry under every behavioral arm, in panel and report order."""
        return [self.evaluate(entry, arm) for entry in entries for arm in arms if arm.behavioral]

    def write_pointers(self, evidence_dir: Path) -> list[Path]:
        """Write the Git pointer of every manifest this runner completed (identical pointers are idempotent)."""
        written: list[Path] = []
        for kind, label, pointer in self.pointers:
            path = evidence_dir / pointer_name(kind, label)
            if path.exists():
                if load_pointer(path) != pointer:
                    msg = f"{path} exists with another pointer; completed evidence is never overwritten"
                    raise FileExistsError(msg)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            write_record(path, pointer)
            written.append(path)
        return written


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(root.resolve()).as_posix() if resolved.is_relative_to(root.resolve()) else path.name


# --- command line -------------------------------------------------------------------------


def _select_arms(labels: Sequence[str] | None) -> tuple[ArmSpec, ...]:
    arms = tuple(arm for arm in panel_arms() if arm.behavioral)
    if not labels:
        return arms
    by_label = {arm.label: arm for arm in arms}
    unknown = [label for label in labels if label not in by_label]
    if unknown:
        msg = f"unknown behavioral arms {unknown}; choose from {sorted(by_label)}"
        raise ValueError(msg)
    return tuple(by_label[label] for label in labels)


@dataclass(frozen=True)
class PreparedRun:
    """A runner built from the command-line arguments, with what it was built from."""

    runner: PilotRunner
    context: PanelContext
    evaluation_file: Path
    entries: tuple[PanelEntry, ...]
    arms: tuple[ArmSpec, ...]


def prepare_runner(
    args: argparse.Namespace, *, module: str = _MODULE, argv: Sequence[str] = (), budget: ExecutionBudget | None = None
) -> PreparedRun:
    """Verify the pinned environment, bind the panel, config, trackers, and validation, and build the runner.

    Shared by the ``run`` command and the M3REP-005 smoke check; the caller's
    ``module`` and ``argv`` are recorded as the launch command.
    """
    require_canonical()
    ensure_single_thread()
    for name in ("numpy", "rclib"):
        importlib.import_module(name)
    root = repository_root()
    store = open_storage()
    command = command_line(module, list(argv))
    execution = collect_execution(command=command, role="main", now=datetime.now(tz=UTC))
    execution.check_canonical()
    context = PanelContext.load(Path(cast("str", args.manifest)), store=store, root=root, execution=execution)
    evaluation_file = Path(cast("str", args.evaluation))
    evaluation = load_evaluation_config(evaluation_file)
    development_sha256 = sha256_file(evaluation.development)
    if development_sha256 != context.manifest.configs.development_sha256:
        msg = (
            f"{evaluation.development} (sha256 {development_sha256[:12]}) is not the development file the panel "
            f"manifest bound ({context.manifest.configs.development_sha256[:12]})"
        )
        raise ValueError(msg)
    levels = load_development_robustness(evaluation.development)
    lower = tuple(link.q_min for link in context.inputs.scenario.robot.links)
    upper = tuple(link.q_max for link in context.inputs.scenario.robot.links)
    scenarios = robustness_scenarios(levels, nominal=context.dataset.q0_ref, lower=lower, upper=upper)
    trackers = {name: load_frozen_baseline(name) for name in RECOVERY_TRACKERS}
    digests = {name: frozen_baseline_digest(name) for name in RECOVERY_TRACKERS}
    if digests != context.manifest.configs.trackers:
        msg = "the frozen trackers differ from the ones the panel manifest bound"
        raise ValueError(msg)
    numerical = numerical_binding(Path(cast("str", args.validation)), root=root)
    labels = cast("list[str] | None", args.entries) or list(context.manifest.rule.labels)
    entries = tuple(context.manifest.entry(label) for label in labels)
    arms = _select_arms(cast("list[str] | None", args.arms))
    resolved = {
        "manifest": context.manifest_sha256,
        "evaluation": {evaluation_file.name: sha256_file(evaluation_file)},
        "development_sha256": development_sha256,
        "velocity_abort": list(evaluation.simulation.velocity_abort),
        "entries": [e.label for e in entries],
        "arms": [a.label for a in arms],
        "numerical_validation": numerical.validation_sha256,
        "execution_identity": execution.identity,
        "command": command,
    }
    provenance = collect_provenance(
        resolved, seeds={}, artifacts=[context.payload], exploratory=bool(args.exploratory), now=datetime.now(tz=UTC)
    )
    require_clean_for_confirmatory(provenance)
    runner = PilotRunner(
        store=store,
        inputs=context.inputs,
        dataset=context.dataset,
        evaluation=evaluation,
        evaluation_file=evaluation_file,
        root=root,
        execution=execution,
        provenance=provenance,
        numerical=numerical,
        scenarios=scenarios,
        trackers=trackers,
        tracker_digests=digests,
        development_sha256=development_sha256,
        command=command,
        budget=budget,
    )
    return PreparedRun(runner, context, evaluation_file, entries, arms)


def _budget(args: argparse.Namespace) -> ExecutionBudget | None:
    time_s = cast("float | None", args.time_budget_s)
    storage = cast("int | None", args.storage_budget_bytes)
    if time_s is None and storage is None:
        return None
    return ExecutionBudget(
        time_seconds=time_s, storage_bytes=storage, storage_baseline_bytes=int(cast("int", args.storage_baseline_bytes))
    )


def _run(args: argparse.Namespace) -> int:
    prepared = prepare_runner(args, argv=cast("list[str]", args.argv), budget=_budget(args))
    runner = prepared.runner
    evidences: list[ModelEvidence] = []
    stopped: str | None = None
    try:
        for entry in prepared.entries:
            for arm in prepared.arms:
                evidences.append(runner.evaluate(entry, arm))  # noqa: PERF401 - partial results survive a budget stop
    except BudgetExceededError as exc:
        stopped = str(exc)
    written = runner.write_pointers(Path(cast("str", args.evidence_dir)))
    print(
        json.dumps(
            {
                "models": len(evidences),
                "statuses": {s: sum(1 for e in evidences if e.status == s) for s in MODEL_STATUSES},
                "pointers_written": len(written),
                "runs_simulated": len(runner.run_timings),
                "used_bytes": runner.used_bytes,
                "elapsed_seconds": round(time.perf_counter() - runner.started, 1),
                "budget_stop": stopped,
            },
            indent=2,
        )
    )
    return 3 if stopped is not None else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Paired pilot evaluation of the repeated-demonstration panel.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    run = subparsers.add_parser("run", help="evaluate panel entries under behavioral arms with paired replay")
    run.add_argument("--manifest", type=str, required=True, help="frozen panel manifest JSON")
    run.add_argument("--evaluation", type=str, required=True, help="pilot evaluation config TOML")
    run.add_argument("--validation", type=str, required=True, help="numerical validation JSON (C11 binding)")
    run.add_argument("--evidence-dir", type=str, required=True, help="directory of the Git pointer records")
    run.add_argument("--entries", type=str, nargs="*", default=None, help="panel labels (default: all six)")
    run.add_argument("--arms", type=str, nargs="*", default=None, help="behavioral arm labels (default: all)")
    run.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    run.add_argument(
        "--time-budget-s", type=float, default=None, help="wall-time allowance of this invocation in seconds (C12)"
    )
    run.add_argument(
        "--storage-budget-bytes", type=int, default=None, help="total run/manifest storage allowance in bytes (C12)"
    )
    run.add_argument(
        "--storage-baseline-bytes", type=int, default=0, help="storage earlier invocations already used against it"
    )
    args = parser.parse_args(argv)
    args.argv = argv
    return _run(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
