# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Optuna studies in the external storage root (``docs/PLAN.md`` sections 10 and 11; M3-002).

A study lives in its own SQLite database under ``armrc://optuna/<name>.db``
with a seeded sampler and pruner. The study records its identity (protocol
digest, sampler, pruner, direction) as user attributes; reopening the study
resumes it only when that identity matches, so a resumed study cannot silently
continue under a different protocol. Sampling is deterministic given the seed
and the stored trial history: two resumes from the same stored state produce
the same continuation, and a study rerun from scratch reproduces itself.
(Optuna re-seeds the sampler on every open, so an uninterrupted study and one
interrupted and resumed can differ after the interruption point; both are
recorded in full.) Trials that raise propagate immediately rather than being
recorded as failures.
"""

from __future__ import annotations

import json
import re
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, cast

import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.scalars import flatten_scalars
from arm_rc_ctrl.provenance import canonical_json

if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna.trial import FrozenTrial

    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "OPTUNA_BUCKET",
    "PrunerSpec",
    "SamplerSpec",
    "StudyMismatchError",
    "StudySummary",
    "TrialRecord",
    "close_study",
    "finished",
    "open_study",
    "run_trials",
    "select_best",
    "study_uri",
    "summarize",
    "summary_from_json",
    "summary_to_json",
]

OPTUNA_BUCKET: Final = "optuna"
_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_IDENTITY_ATTRS: Final = ("armrc.protocol_sha256", "armrc.sampler", "armrc.pruner", "armrc.direction")
_FINALIZERS: dict[int, weakref.finalize[[RDBStorage], optuna.Study]] = {}
"""Engine disposal per open study, keyed by the study object's id while the caller holds it."""


def _dispose(storage: RDBStorage) -> None:
    storage.engine.dispose()


optuna.logging.set_verbosity(optuna.logging.WARNING)  # per-trial INFO lines belong to the study summary, not stderr


class StudyMismatchError(ValueError):
    """An existing study was created under a different protocol, sampler, pruner, or direction."""


@dataclass(frozen=True)
class SamplerSpec:
    """A seeded Optuna sampler."""

    seed: int
    kind: Literal["tpe", "random"] = "tpe"
    n_startup_trials: int = 10
    """Trials sampled uniformly at random before TPE starts modelling (ignored by ``random``)."""

    def __post_init__(self) -> None:
        """Validate the seed and start-up count."""
        if self.seed < 0:
            msg = f"sampler.seed must be non-negative, got {self.seed}"
            raise ValueError(msg)
        if self.n_startup_trials < 1:
            msg = f"sampler.n_startup_trials must be >= 1, got {self.n_startup_trials}"
            raise ValueError(msg)

    def build(self) -> optuna.samplers.BaseSampler:
        """The sampler (fresh random state from the seed)."""
        if self.kind == "random":
            return optuna.samplers.RandomSampler(seed=self.seed)
        return optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=self.n_startup_trials)


@dataclass(frozen=True)
class PrunerSpec:
    """A deterministic Optuna pruner acting on the intermediate values a trial reports."""

    kind: Literal["median", "none"] = "median"
    n_startup_trials: int = 5
    """Completed trials required before pruning starts."""
    n_warmup_steps: int = 0
    """Reported steps of a trial that are never pruned."""

    def __post_init__(self) -> None:
        """Validate the counts."""
        if self.n_startup_trials < 0 or self.n_warmup_steps < 0:
            msg = "pruner counts must be non-negative"
            raise ValueError(msg)

    def build(self) -> optuna.pruners.BasePruner:
        """The pruner."""
        if self.kind == "none":
            return optuna.pruners.NopPruner()
        return optuna.pruners.MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_warmup_steps)


def study_uri(name: str) -> str:
    """Logical location of the study database."""
    if not _NAME.match(name):
        msg = f"study name {name!r} must match {_NAME.pattern}"
        raise ValueError(msg)
    return f"armrc://{OPTUNA_BUCKET}/{name}.db"


def _identity(protocol_sha256: str, sampler: SamplerSpec, pruner: PrunerSpec, direction: str) -> dict[str, str]:
    return {
        "armrc.protocol_sha256": protocol_sha256,
        "armrc.sampler": canonical_json(to_mapping(sampler)),
        "armrc.pruner": canonical_json(to_mapping(pruner)),
        "armrc.direction": direction,
    }


def open_study(
    store: StorageRoot,
    name: str,
    *,
    protocol_sha256: str,
    sampler: SamplerSpec,
    pruner: PrunerSpec,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> optuna.Study:
    """Create the study in the store or resume it when its recorded identity matches."""
    database = store.path(study_uri(name), mode="write")
    identity = _identity(protocol_sha256, sampler, pruner, direction)
    storage = RDBStorage(f"sqlite:///{database}")
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        sampler=sampler.build(),
        pruner=pruner.build(),
        direction=direction,
        load_if_exists=True,
    )
    _FINALIZERS[id(study)] = weakref.finalize(study, _dispose, storage)
    stored = {key: study.user_attrs.get(key) for key in _IDENTITY_ATTRS}
    if all(value is None for value in stored.values()):
        for key, value in identity.items():
            study.set_user_attr(key, value)
        return study
    if stored != identity:
        differing = sorted(key for key in _IDENTITY_ATTRS if stored[key] != identity[key])
        msg = f"study {name!r} in {study_uri(name)} was created under a different {', '.join(differing)}"
        raise StudyMismatchError(msg)
    return study


def close_study(study: optuna.Study) -> None:
    """Dispose of the study's database connections (done at garbage collection otherwise)."""
    finalizer = _FINALIZERS.pop(id(study), None)
    if finalizer is not None:
        finalizer()


def finished(study: optuna.Study) -> tuple[FrozenTrial, ...]:
    """Trials that reached a final state (complete or pruned), in trial order."""
    return tuple(t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED))


def run_trials(study: optuna.Study, objective: Callable[[optuna.Trial], float], *, budget: int) -> int:
    """Run the objective until ``budget`` trials have finished (resuming counts stored trials); return trials run."""
    if budget < 1:
        msg = f"budget must be >= 1, got {budget}"
        raise ValueError(msg)
    remaining = budget - len(finished(study))
    if remaining <= 0:
        return 0
    study.optimize(objective, n_trials=remaining, catch=(), gc_after_trial=False, show_progress_bar=False)
    return remaining


def select_best(study: optuna.Study, *, eligible: Callable[[FrozenTrial], bool] | None = None) -> FrozenTrial:
    """The eligible completed trial with the best value; ties go to the earliest trial.

    ``eligible`` restricts the selection (e.g. to trials flagged feasible); a
    completed trial is otherwise eligible whatever its attributes say.
    """
    complete = [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None and (eligible is None or eligible(t))
    ]
    if not complete:
        msg = f"study {study.study_name!r} has no eligible completed trial"
        raise ValueError(msg)
    sign = 1.0 if study.direction == optuna.study.StudyDirection.MINIMIZE else -1.0
    return min(complete, key=lambda t: (sign * cast("float", t.value), t.number))


@dataclass(frozen=True)
class TrialRecord:
    """One trial as stored by Optuna, with its attributes flattened to typed scalar tables."""

    number: int
    state: str
    value: float | None
    params: dict[str, float]
    """Numeric parameters (integer distributions are stored as floats)."""
    choices: dict[str, str] = field(default_factory=dict)
    """Categorical parameters."""
    metrics: dict[str, float] = field(default_factory=dict)
    """Numeric user attributes (nested structures flattened to dotted keys)."""
    flags: dict[str, bool] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    intermediate_values: dict[str, float] = field(default_factory=dict)
    """Reported intermediate values keyed by step (as a string, for TOML/JSON)."""


@dataclass(frozen=True)
class StudySummary:
    """Everything a study holds, for reports and MLflow export."""

    name: str
    storage: str
    """Logical location of the database."""
    direction: str
    identity: dict[str, str]
    """The study's recorded identity attributes (protocol digest, sampler, pruner, direction)."""
    trials: tuple[TrialRecord, ...]
    n_complete: int
    n_pruned: int
    best_number: int | None
    best_value: float | None
    selection_rule: str = "complete"
    """Which trials were eligible for ``best``: ``complete`` (any completed trial) or a named restriction."""

    def __post_init__(self) -> None:
        """Consistency of the counts with the trials."""
        states = [t.state for t in self.trials]
        if states.count("COMPLETE") != self.n_complete or states.count("PRUNED") != self.n_pruned:
            msg = "trial state counts do not match the trials"
            raise ValueError(msg)


def _record(trial: FrozenTrial) -> TrialRecord:
    params: dict[str, float] = {}
    choices: dict[str, str] = {}
    for key, value in cast("dict[str, object]", trial.params).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            choices[key] = str(value)
        else:
            params[key] = float(value)
    flat: dict[str, object] = {}
    flatten_scalars("", cast("dict[str, object]", trial.user_attrs), flat)
    metrics: dict[str, float] = {}
    flags: dict[str, bool] = {}
    labels: dict[str, str] = {}
    for key, value in flat.items():
        if isinstance(value, bool):
            flags[key] = value
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
        elif value is not None:
            labels[key] = str(value)
    return TrialRecord(
        number=trial.number,
        state=trial.state.name,
        value=None if trial.value is None else float(trial.value),
        params=params,
        choices=choices,
        metrics=metrics,
        flags=flags,
        labels=labels,
        intermediate_values={str(step): float(v) for step, v in sorted(trial.intermediate_values.items())},
    )


def summarize(
    study: optuna.Study,
    *,
    eligible: Callable[[FrozenTrial], bool] | None = None,
    selection_rule: str = "complete",
) -> StudySummary:
    """Summarize the study's stored state (queued trials that have not started are not part of it).

    ``eligible`` and ``selection_rule`` name which completed trials may be
    selected as ``best`` (see :func:`select_best`).
    """
    trials = tuple(_record(t) for t in study.trials if t.state != TrialState.WAITING)
    candidates = [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None and (eligible is None or eligible(t))
    ]
    best = select_best(study, eligible=eligible) if candidates else None
    return StudySummary(
        name=study.study_name,
        storage=study_uri(study.study_name),
        direction=study.direction.name.lower(),
        identity={key: str(study.user_attrs[key]) for key in _IDENTITY_ATTRS if key in study.user_attrs},
        trials=trials,
        n_complete=sum(1 for t in trials if t.state == "COMPLETE"),
        n_pruned=sum(1 for t in trials if t.state == "PRUNED"),
        best_number=None if best is None else best.number,
        best_value=None if best is None else cast("float", best.value),
        selection_rule=selection_rule,
    )


def summary_to_json(summary: StudySummary) -> str:
    """Canonical JSON of the summary."""
    return canonical_json(to_mapping(summary))


def summary_from_json(text: str) -> StudySummary:
    """Strictly rebuild a summary from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(text)), StudySummary)
