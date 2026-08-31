# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Versioned ESN hyperparameter search space and development-only scenarios (``docs/PLAN.md`` section 10; M3-003).

A search protocol (``configs/studies/esn_search_*.toml``) fixes the base model
configuration (input transform and readout solver stay as recorded there), the
frozen tracker, the bounds of every tuned parameter, the study seed, budget,
sampler and pruner, the objective and its infeasibility penalty, the
development scenarios, and the comparison points that are evaluated first.
The tuned parameters are the reservoir size, spectral radius, sparsity, leak
rate, input scaling, reservoir seed, ridge regularization, and the two
derivative-filter cutoffs of the causal estimator; the washout is the
demonstration's prime phase (a recipe invariant), so it is not a free
parameter. Confirmatory seeds and levels never appear here (enforced by the
regression locks).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from optuna.trial import TrialState

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.studies import PrunerSpec, SamplerSpec
from arm_rc_ctrl.experiments.tuning import DevelopmentScenarios, Feasibility
from arm_rc_ctrl.provenance import canonical_json
from arm_rc_ctrl.rc.esn import EsnConfig
from arm_rc_ctrl.rc.train import ModelConfig, load_model_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    import optuna

__all__ = [
    "PLANNED_PARAMETERS",
    "ComparisonPoint",
    "EsnObjective",
    "EsnSearchProtocol",
    "EsnSearchSpace",
    "FloatRange",
    "IntRange",
    "TrialPoint",
    "enqueue_comparisons",
    "load_esn_search",
    "point_from_params",
    "protocol_digest",
    "suggest_point",
]

PLANNED_PARAMETERS: Final = (
    "n_neurons",
    "spectral_radius",
    "sparsity",
    "leak_rate",
    "input_scaling",
    "seed",
    "alpha",
    "velocity_cutoff_hz",
    "acceleration_cutoff_hz",
)
"""The tuned parameters, in the order they are suggested."""


@dataclass(frozen=True)
class FloatRange:
    """Bounds of a continuous parameter."""

    low: float
    high: float
    log: bool = False
    step: float | None = None

    def __post_init__(self) -> None:
        """Validate the bounds."""
        if not (math.isfinite(self.low) and math.isfinite(self.high)) or self.low >= self.high:
            msg = f"range needs finite low < high, got [{self.low!r}, {self.high!r}]"
            raise ValueError(msg)
        if self.log and self.low <= 0:
            msg = f"a log range needs low > 0, got {self.low!r}"
            raise ValueError(msg)
        if self.step is not None and (self.log or self.step <= 0):
            msg = "step must be positive and cannot be combined with log"
            raise ValueError(msg)

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies within the bounds."""
        return self.low <= value <= self.high


@dataclass(frozen=True)
class IntRange:
    """Bounds of an integer parameter."""

    low: int
    high: int
    step: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        """Validate the bounds."""
        if self.low >= self.high:
            msg = f"range needs low < high, got [{self.low}, {self.high}]"
            raise ValueError(msg)
        if self.step < 1 or (self.log and self.step != 1):
            msg = "step must be >= 1 and 1 for a log range"
            raise ValueError(msg)
        if self.log and self.low <= 0:
            msg = f"a log range needs low > 0, got {self.low}"
            raise ValueError(msg)

    def contains(self, value: int) -> bool:
        """Whether ``value`` lies within the bounds on the step grid."""
        return self.low <= value <= self.high and (value - self.low) % self.step == 0


@dataclass(frozen=True)
class EsnSearchSpace:
    """Bounds of every tuned parameter (all of ``PLANNED_PARAMETERS``)."""

    n_neurons: IntRange
    spectral_radius: FloatRange
    sparsity: FloatRange
    leak_rate: FloatRange
    input_scaling: FloatRange
    seed: IntRange
    alpha: FloatRange
    velocity_cutoff_hz: FloatRange
    acceleration_cutoff_hz: FloatRange

    def __post_init__(self) -> None:
        """Bounds must stay within what the reservoir, readout, and estimator accept."""
        checks = (
            (self.n_neurons.low >= 1, "n_neurons must start at 1 or more"),
            (self.spectral_radius.low >= 0, "spectral_radius must be non-negative"),
            (self.sparsity.low >= 0 and self.sparsity.high <= 1, "sparsity must lie in [0, 1]"),
            (self.leak_rate.low > 0 and self.leak_rate.high <= 1, "leak_rate must lie in (0, 1]"),
            (self.input_scaling.low >= 0, "input_scaling must be non-negative"),
            (self.seed.low >= 0, "seed must be non-negative"),
            (self.alpha.low > 0, "alpha must be positive"),
            (self.velocity_cutoff_hz.low > 0, "velocity_cutoff_hz must be positive"),
            (self.acceleration_cutoff_hz.low > 0, "acceleration_cutoff_hz must be positive"),
        )
        for ok, message in checks:
            if not ok:
                msg = f"search.{message}"
                raise ValueError(msg)

    def contains(self, point: TrialPoint) -> bool:
        """Whether every coordinate of ``point`` lies within its bounds."""
        return (
            self.n_neurons.contains(point.n_neurons)
            and self.spectral_radius.contains(point.spectral_radius)
            and self.sparsity.contains(point.sparsity)
            and self.leak_rate.contains(point.leak_rate)
            and self.input_scaling.contains(point.input_scaling)
            and self.seed.contains(point.seed)
            and self.alpha.contains(point.alpha)
            and self.velocity_cutoff_hz.contains(point.velocity_cutoff_hz)
            and self.acceleration_cutoff_hz.contains(point.acceleration_cutoff_hz)
        )


@dataclass(frozen=True)
class TrialPoint:
    """One point of the search space."""

    n_neurons: int
    spectral_radius: float
    sparsity: float
    leak_rate: float
    input_scaling: float
    seed: int
    alpha: float
    velocity_cutoff_hz: float
    acceleration_cutoff_hz: float

    def params(self) -> dict[str, float | int]:
        """The point as Optuna parameters."""
        return {name: getattr(self, name) for name in PLANNED_PARAMETERS}

    def model_config(self, base: ModelConfig, *, name: str) -> ModelConfig:
        """The base model configuration with this point's reservoir and ridge settings."""
        reservoir = replace(
            base.esn.reservoir,
            n_neurons=self.n_neurons,
            spectral_radius=self.spectral_radius,
            sparsity=self.sparsity,
            leak_rate=self.leak_rate,
            input_scaling=self.input_scaling,
            seed=self.seed,
        )
        readout = replace(base.esn.readout, alpha=self.alpha)
        return replace(base, name=name, esn=EsnConfig(reservoir=reservoir, readout=readout))

    def estimator(self, *, max_dt_ratio: float) -> EstimatorSpec:
        """The causal estimator settings of this point."""
        return EstimatorSpec(self.velocity_cutoff_hz, self.acceleration_cutoff_hz, max_dt_ratio)


@dataclass(frozen=True)
class ComparisonPoint:
    """A labelled point evaluated before any sampled trial (e.g. the development anchor)."""

    label: str
    point: TrialPoint

    def __post_init__(self) -> None:
        """Labels are non-empty."""
        if not self.label.strip():
            msg = "comparison.label must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class EsnObjective:
    """The scalar objective and its infeasibility penalty (components are always kept separately)."""

    kind: Literal["median_move_joint_rmse"] = "median_move_joint_rmse"
    infeasible_penalty: float = 10.0

    def __post_init__(self) -> None:
        """The penalty is a finite positive radian value larger than any plausible RMSE."""
        if not math.isfinite(self.infeasible_penalty) or self.infeasible_penalty <= 0:
            msg = f"objective.infeasible_penalty must be positive and finite, got {self.infeasible_penalty!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class EsnSearchProtocol:
    """Complete ESN search protocol (``configs/studies/esn_search_*.toml``)."""

    name: str
    scenario: Path
    model: Path
    """Base model configuration: the input transform and readout solver are taken from it unchanged."""
    tracker: str
    """Frozen baseline tracker (registry name); tracker gains are not tuned."""
    budget: int
    sampler: SamplerSpec
    pruner: PrunerSpec
    search: EsnSearchSpace
    development: DevelopmentScenarios
    objective: EsnObjective = field(default_factory=EsnObjective)
    feasibility: Feasibility = field(default_factory=Feasibility)
    comparison: tuple[ComparisonPoint, ...] = ()
    max_dt_ratio: float = 3.0

    def __post_init__(self) -> None:
        """Validate the name, budget, comparison points, and the estimator bound."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if self.budget < len(self.comparison) + 1:
            msg = f"budget {self.budget} leaves no sampled trial after {len(self.comparison)} comparison points"
            raise ValueError(msg)
        labels = [c.label for c in self.comparison]
        if len(set(labels)) != len(labels):
            msg = "comparison labels must be unique"
            raise ValueError(msg)
        for comparison in self.comparison:
            if not self.search.contains(comparison.point):
                msg = f"comparison point {comparison.label!r} lies outside the search space"
                raise ValueError(msg)
        if self.max_dt_ratio < 1:
            msg = f"max_dt_ratio must be >= 1, got {self.max_dt_ratio!r}"
            raise ValueError(msg)

    def base_model(self) -> ModelConfig:
        """Load the base model configuration."""
        return load_model_config(self.model)


def load_esn_search(path: Path) -> EsnSearchProtocol:
    """Load and validate a search protocol."""
    return load_config(path, EsnSearchProtocol)


def protocol_digest(protocol: EsnSearchProtocol) -> str:
    """SHA-256 of the canonical protocol (the study identity)."""
    return hashlib.sha256(canonical_json(to_mapping(protocol)).encode("utf-8")).hexdigest()


def suggest_point(space: EsnSearchSpace, trial: optuna.Trial) -> TrialPoint:
    """Draw one point from ``space`` through ``trial`` (parameters in ``PLANNED_PARAMETERS`` order)."""

    def draw_float(name: str, bounds: FloatRange) -> float:
        return trial.suggest_float(name, bounds.low, bounds.high, log=bounds.log, step=bounds.step)

    def draw_int(name: str, bounds: IntRange) -> int:
        return trial.suggest_int(name, bounds.low, bounds.high, step=bounds.step, log=bounds.log)

    return TrialPoint(
        n_neurons=draw_int("n_neurons", space.n_neurons),
        spectral_radius=draw_float("spectral_radius", space.spectral_radius),
        sparsity=draw_float("sparsity", space.sparsity),
        leak_rate=draw_float("leak_rate", space.leak_rate),
        input_scaling=draw_float("input_scaling", space.input_scaling),
        seed=draw_int("seed", space.seed),
        alpha=draw_float("alpha", space.alpha),
        velocity_cutoff_hz=draw_float("velocity_cutoff_hz", space.velocity_cutoff_hz),
        acceleration_cutoff_hz=draw_float("acceleration_cutoff_hz", space.acceleration_cutoff_hz),
    )


def point_from_params(space: EsnSearchSpace, params: Mapping[str, float]) -> TrialPoint:
    """Rebuild a point from stored Optuna parameters (validated against ``space``)."""
    missing = sorted(set(PLANNED_PARAMETERS) - set(params))
    if missing:
        msg = f"parameters {missing} are missing"
        raise ValueError(msg)
    point = TrialPoint(
        n_neurons=int(params["n_neurons"]),
        spectral_radius=float(params["spectral_radius"]),
        sparsity=float(params["sparsity"]),
        leak_rate=float(params["leak_rate"]),
        input_scaling=float(params["input_scaling"]),
        seed=int(params["seed"]),
        alpha=float(params["alpha"]),
        velocity_cutoff_hz=float(params["velocity_cutoff_hz"]),
        acceleration_cutoff_hz=float(params["acceleration_cutoff_hz"]),
    )
    if not space.contains(point):
        msg = "stored parameters lie outside the search space"
        raise ValueError(msg)
    return point


def enqueue_comparisons(study: optuna.Study, protocol: EsnSearchProtocol) -> int:
    """Queue the protocol's comparison points the study does not hold yet; return how many were queued.

    Points already evaluated (matching parameters) are skipped, and Optuna skips
    points that are still waiting, so the call is idempotent across resumes.
    """
    evaluated = [dict(t.params) for t in study.get_trials(deepcopy=False) if t.state != TrialState.WAITING]
    queued = 0
    for comparison in protocol.comparison:
        params = comparison.point.params()
        if any(params == p for p in evaluated):
            continue
        before = len(study.get_trials(deepcopy=False))
        study.enqueue_trial(dict(params), user_attrs={"armrc.comparison": comparison.label}, skip_if_exists=True)
        queued += len(study.get_trials(deepcopy=False)) - before
    return queued
