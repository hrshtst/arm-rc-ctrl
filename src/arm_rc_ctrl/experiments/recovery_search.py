# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Recovery development search protocols: one study per generator formulation (M3R-012; plan section 9.2).

Each formulation — ``no_augmentation``, ``non_decaying``, ``contractive`` —
gets its own Optuna study with an identical trial count and an identical ESN
space; the two augmented studies share their augmentation seed bank and search
the approved D1 grids as categorical parameters. Parameters that do not apply
to a formulation are **absent**, never dummy-filled; every trial evaluates
both frozen trackers, and the tracker is never an Optuna parameter. The
development scenarios come from the locked recovery development levels; no
confirmatory seed, level, or outcome is reachable from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import optuna
from optuna.trial import TrialState

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.experiments.esn_search import EsnSearchSpace, TrialPoint
from arm_rc_ctrl.experiments.esn_search import (
    point_from_params as esn_point_from_params,
)
from arm_rc_ctrl.experiments.esn_search import (
    suggest_point as esn_suggest_point,
)
from arm_rc_ctrl.experiments.studies import PrunerSpec, SamplerSpec
from arm_rc_ctrl.experiments.tuning import Feasibility
from arm_rc_ctrl.metrics.recovery import SATURATION_BOUND
from arm_rc_ctrl.provenance import config_digest
from arm_rc_ctrl.rc.augment import (
    APPROVED_GAMMA,
    APPROVED_N_SYNTHETIC,
    APPROVED_PHI,
    APPROVED_SIGMA_RAD,
)
from arm_rc_ctrl.rc.recipe import AugmentationTrainingSpec, TrainingSpec
from arm_rc_ctrl.rc.train import ModelConfig, load_model_config
from arm_rc_ctrl.rc.warmup import APPROVED_WARMUPS_S

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "RECOVERY_TRACKERS",
    "AugmentationGrid",
    "AugmentationPoint",
    "RecoveryComparison",
    "RecoveryFormulation",
    "RecoveryObjectiveSpec",
    "RecoverySearchProtocol",
    "RecoverySpace",
    "RecoveryTrialPoint",
    "check_matched_protocols",
    "enqueue_recovery_comparisons",
    "load_recovery_search",
    "point_from_params",
    "recovery_protocol_digest",
    "suggest_recovery_point",
    "training_spec_for",
]

RECOVERY_TRACKERS: Final[tuple[str, ...]] = ("pd_v2", "computed_torque")
"""Both frozen trackers are evaluated for every trial; the tracker is never a search parameter."""

type RecoveryFormulation = Literal["no_augmentation", "non_decaying", "contractive"]
_AUGMENTED: Final = ("non_decaying", "contractive")
_AUGMENTATION_PARAMETERS: Final = ("n_synthetic", "sigma_rad", "phi", "gamma")


def _subset(
    name: str, values: tuple[float, ...] | tuple[int, ...], approved: frozenset[float] | frozenset[int]
) -> None:
    if not values or len(set(values)) != len(values):
        msg = f"{name} must be a non-empty list of distinct values, got {list(values)}"
        raise ValueError(msg)
    outside = [v for v in values if v not in approved]
    if outside:
        msg = f"{name} values {outside} lie outside the approved set {sorted(approved)}"
        raise ValueError(msg)


@dataclass(frozen=True)
class AugmentationGrid:
    """Searched subsets of the approved D1 augmentation grids (categorical dimensions)."""

    n_synthetic: tuple[int, ...]
    sigma_rad: tuple[float, ...]
    phi: tuple[float, ...]
    gamma: tuple[float, ...]

    def __post_init__(self) -> None:
        """Every listed value sits on the approved D1 grid."""
        _subset("space.n_synthetic", self.n_synthetic, APPROVED_N_SYNTHETIC)
        _subset("space.sigma_rad", self.sigma_rad, APPROVED_SIGMA_RAD)
        _subset("space.phi", self.phi, APPROVED_PHI)
        _subset("space.gamma", self.gamma, APPROVED_GAMMA)


@dataclass(frozen=True)
class RecoverySpace:
    """The formulation-specific search dimensions beyond the ESN space."""

    warmups_s: tuple[float, ...]
    n_synthetic: tuple[int, ...] = ()
    sigma_rad: tuple[float, ...] = ()
    phi: tuple[float, ...] = ()
    gamma: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Warm-ups are a subset of the approved D2 set; augmentation grids validate when present."""
        _subset("space.warmups_s", self.warmups_s, APPROVED_WARMUPS_S)
        if self.augmentation is not None:
            pass  # constructing the grid runs its validation

    @property
    def augmentation(self) -> AugmentationGrid | None:
        """The augmentation grid, or ``None`` when no augmentation dimension is listed."""
        listed = (self.n_synthetic, self.sigma_rad, self.phi, self.gamma)
        if not any(listed):
            return None
        if not all(listed):
            missing = [name for name, values in zip(_AUGMENTATION_PARAMETERS, listed, strict=True) if not values]
            msg = f"augmentation grids are incomplete: missing {missing}"
            raise ValueError(msg)
        return AugmentationGrid(self.n_synthetic, self.sigma_rad, self.phi, self.gamma)


@dataclass(frozen=True)
class RecoveryObjectiveSpec:
    """The development objective: the worst class-by-tracker cell median of the early command-gap ratio."""

    kind: Literal["worst_cell_median_gap_ratio"] = "worst_cell_median_gap_ratio"
    infeasible_penalty: float = 10.0

    def __post_init__(self) -> None:
        """The penalty is positive and finite."""
        if not (self.infeasible_penalty > 0 and self.infeasible_penalty < float("inf")):
            msg = f"objective.infeasible_penalty must be positive and finite, got {self.infeasible_penalty!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RecoveryComparison:
    """A labelled point evaluated before any sampled trial (the approved D1/D2 anchors)."""

    label: str
    point: RecoveryTrialPoint

    def __post_init__(self) -> None:
        """Labels are non-empty."""
        if not self.label.strip():
            msg = "comparison.label must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class RecoverySearchProtocol:
    """A versioned recovery development search for one generator formulation."""

    name: str
    scenario: Path
    model: Path
    formulation: RecoveryFormulation
    budget: int
    seed_bank: int
    """Shared augmentation seed bank (identical in both augmented studies by protocol)."""
    attempt_factor: int
    """Attempt budget per accepted episode: ``attempt_factor * n_synthetic``."""
    development: Path
    """The locked recovery development levels (never the confirmatory file)."""
    sampler: SamplerSpec
    pruner: PrunerSpec
    esn: EsnSearchSpace
    space: RecoverySpace
    objective: RecoveryObjectiveSpec = field(default_factory=RecoveryObjectiveSpec)
    feasibility: Feasibility = field(default_factory=lambda: Feasibility(SATURATION_BOUND))
    max_dt_ratio: float = 3.0
    comparison: tuple[RecoveryComparison, ...] = ()
    """Labelled anchor points queued before sampling; every value must lie in the searched sets."""

    def __post_init__(self) -> None:
        """Formulation-specific dimensions are present exactly when applicable; the saturation bound is fixed."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if self.budget < 1 or self.seed_bank < 0 or self.attempt_factor < 1:
            msg = (
                f"budget >= 1, seed_bank >= 0, attempt_factor >= 1 required, got "
                f"{self.budget}, {self.seed_bank}, {self.attempt_factor}"
            )
            raise ValueError(msg)
        grid = self.space.augmentation
        if self.formulation in _AUGMENTED and grid is None:
            msg = f"the {self.formulation!r} formulation requires the augmentation grids in [space]"
            raise ValueError(msg)
        if self.formulation == "no_augmentation" and grid is not None:
            msg = "the no_augmentation study carries no augmentation parameters (inapplicable, never dummy-filled)"
            raise ValueError(msg)
        if self.feasibility.max_saturation_fraction != SATURATION_BOUND:
            msg = (
                f"the recovery protocol fixes the saturation bound at {SATURATION_BOUND}; "
                f"got {self.feasibility.max_saturation_fraction!r}"
            )
            raise ValueError(msg)
        if self.max_dt_ratio < 1:
            msg = f"max_dt_ratio must be >= 1, got {self.max_dt_ratio!r}"
            raise ValueError(msg)
        labels = [comparison.label for comparison in self.comparison]
        if len(set(labels)) != len(labels):
            msg = f"comparison labels must be unique, got {labels}"
            raise ValueError(msg)
        for comparison in self.comparison:
            point_from_params(self, comparison.point.params())  # every anchor lies in the searched sets

    def base_model(self) -> ModelConfig:
        """The base model configuration (input transform and readout solver come from here)."""
        return load_model_config(self.model)


def load_recovery_search(path: Path) -> RecoverySearchProtocol:
    """Load and validate a recovery search protocol."""
    return load_config(path, RecoverySearchProtocol)


def recovery_protocol_digest(protocol: RecoverySearchProtocol) -> str:
    """Portable digest of the resolved protocol (the study identity)."""
    return config_digest(protocol)[1]


@dataclass(frozen=True)
class AugmentationPoint:
    """The sampled augmentation values of one trial."""

    n_synthetic: int
    sigma_rad: float
    phi: float
    gamma: float


@dataclass(frozen=True)
class RecoveryTrialPoint:
    """One sampled point: ESN hyperparameters, the warm-up, and (when applicable) augmentation values."""

    esn: TrialPoint
    warmup_s: float
    augmentation: AugmentationPoint | None = None
    """Absent for the no-augmentation formulation (inapplicable parameters are never dummy-filled)."""

    def params(self) -> dict[str, float | int]:
        """Every Optuna parameter of the point; inapplicable parameters are absent."""
        values: dict[str, float | int] = dict(self.esn.params())
        values["warmup_s"] = self.warmup_s
        if self.augmentation is not None:
            values["n_synthetic"] = self.augmentation.n_synthetic
            values["sigma_rad"] = self.augmentation.sigma_rad
            values["phi"] = self.augmentation.phi
            values["gamma"] = self.augmentation.gamma
        return values


def suggest_recovery_point(protocol: RecoverySearchProtocol, trial: optuna.Trial) -> RecoveryTrialPoint:
    """Sample one point of the protocol's space (categoricals for the discrete approved grids)."""
    esn = esn_suggest_point(protocol.esn, trial)
    warmup = float(trial.suggest_categorical("warmup_s", list(protocol.space.warmups_s)))
    grid = protocol.space.augmentation
    augmentation = None
    if grid is not None:
        augmentation = AugmentationPoint(
            n_synthetic=int(trial.suggest_categorical("n_synthetic", list(grid.n_synthetic))),
            sigma_rad=float(trial.suggest_categorical("sigma_rad", list(grid.sigma_rad))),
            phi=float(trial.suggest_categorical("phi", list(grid.phi))),
            gamma=float(trial.suggest_categorical("gamma", list(grid.gamma))),
        )
    return RecoveryTrialPoint(esn=esn, warmup_s=warmup, augmentation=augmentation)


def point_from_params(protocol: RecoverySearchProtocol, params: Mapping[str, float]) -> RecoveryTrialPoint:
    """Rebuild a stored point; values outside the protocol's searched sets are refused."""
    esn = esn_point_from_params(
        protocol.esn, {k: v for k, v in params.items() if k not in ("warmup_s", *(_AUGMENTATION_PARAMETERS))}
    )
    if "warmup_s" not in params:
        msg = "parameter 'warmup_s' is missing"
        raise ValueError(msg)
    warmup = float(params["warmup_s"])
    if warmup not in protocol.space.warmups_s:
        msg = f"warmup_s {warmup!r} lies outside the protocol's searched set {list(protocol.space.warmups_s)}"
        raise ValueError(msg)
    grid = protocol.space.augmentation
    if grid is None:
        extraneous = sorted(set(_AUGMENTATION_PARAMETERS) & set(params))
        if extraneous:
            msg = f"parameters {extraneous} do not apply to the {protocol.formulation!r} formulation"
            raise ValueError(msg)
        return RecoveryTrialPoint(esn=esn, warmup_s=warmup, augmentation=None)
    missing = sorted(set(_AUGMENTATION_PARAMETERS) - set(params))
    if missing:
        msg = f"parameters {missing} are missing"
        raise ValueError(msg)
    point = AugmentationPoint(
        n_synthetic=int(params["n_synthetic"]),
        sigma_rad=float(params["sigma_rad"]),
        phi=float(params["phi"]),
        gamma=float(params["gamma"]),
    )
    pairs = (
        ("n_synthetic", point.n_synthetic, grid.n_synthetic),
        ("sigma_rad", point.sigma_rad, grid.sigma_rad),
        ("phi", point.phi, grid.phi),
        ("gamma", point.gamma, grid.gamma),
    )
    for name, value, allowed in pairs:
        if value not in allowed:
            msg = f"{name} {value!r} lies outside the protocol's searched set {list(allowed)}"
            raise ValueError(msg)
    return RecoveryTrialPoint(esn=esn, warmup_s=warmup, augmentation=point)


def training_spec_for(protocol: RecoverySearchProtocol, point: RecoveryTrialPoint) -> TrainingSpec:
    """The training spec of one trial: warm-up washout plus the formulation's augmentation."""
    augmentation = None
    if point.augmentation is not None:
        family: Literal["non_decaying", "contractive"] = (
            "non_decaying" if protocol.formulation == "non_decaying" else "contractive"
        )
        augmentation = AugmentationTrainingSpec(
            family=family,
            n_synthetic=point.augmentation.n_synthetic,
            sigma_rad=point.augmentation.sigma_rad,
            phi=point.augmentation.phi,
            gamma=point.augmentation.gamma,
            seed_bank=protocol.seed_bank,
            attempt_budget=protocol.attempt_factor * point.augmentation.n_synthetic,
        )
    return TrainingSpec(washout="warmup_hold", warmup_s=point.warmup_s, augmentation=augmentation)


def check_matched_protocols(first: RecoverySearchProtocol, second: RecoverySearchProtocol) -> None:
    """Fail unless two formulation studies are matched on every dimension applicable to both.

    Identical budgets, attempt factors, ESN bounds, warm-up sets, objectives, and feasibility; the
    augmented pair additionally shares its augmentation grids and seed bank. Inapplicable dimensions
    (the no-augmentation study has no grids) are absent by construction and therefore not compared.
    """
    pairs = (
        ("budget", first.budget, second.budget),
        ("attempt_factor", first.attempt_factor, second.attempt_factor),
        ("esn space", first.esn, second.esn),
        ("space.warmups_s", first.space.warmups_s, second.space.warmups_s),
        ("objective", first.objective, second.objective),
        ("feasibility", first.feasibility, second.feasibility),
    )
    for name, a, b in pairs:
        if a != b:
            msg = f"studies {first.name!r} and {second.name!r} differ in {name}: {a!r} != {b!r}"
            raise ValueError(msg)
    if first.formulation in _AUGMENTED and second.formulation in _AUGMENTED:
        if first.space.augmentation != second.space.augmentation:
            msg = (
                f"studies {first.name!r} and {second.name!r} differ in their augmentation grids: "
                f"{first.space.augmentation!r} != {second.space.augmentation!r}"
            )
            raise ValueError(msg)
        if first.seed_bank != second.seed_bank:
            msg = f"the augmented studies must share their seed bank, got {first.seed_bank} and {second.seed_bank}"
            raise ValueError(msg)


def enqueue_recovery_comparisons(study: optuna.Study, protocol: RecoverySearchProtocol) -> int:
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
