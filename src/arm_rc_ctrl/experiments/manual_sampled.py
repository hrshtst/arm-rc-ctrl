# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-002: one sampled ESN configuration, expressed as the closed experiment already expresses one.

A trial of the manual-demonstration search samples the reservoir, the base
readout regularization and the warm-up; everything else is inherited from the
closed experiment and from the protocol's fixed policy. A sampled configuration
is therefore not a new kind of thing: it is a
:class:`~arm_rc_ctrl.experiments.manual_study.StudyConfiguration`, the same
record the frozen study uses, so the existing recipes, conditions, evaluation
and evidence paths take it unchanged and there is no second training path to
keep right.

What this module adds is the boundary. A point is read from a trial's
parameters and validated against the approved space, the fixed policy is
written into the configuration and can be re-checked afterwards, the sampled
label is drawn from a namespace the closed study's six cannot occupy, and the
arms are the closed experiment's own. Nothing here fits a model, simulates a
run, writes to the store, or reads or rewrites the frozen study's identities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.experiments.manual_recipes import CONTRACTIVE_ARM, MANUAL_ANCHOR, esn_for_arm, manual_arms
from arm_rc_ctrl.experiments.manual_study import ContractiveBank, StudyConfiguration, study_model
from arm_rc_ctrl.rc.esn import EsnConfig
from arm_rc_ctrl.rc.train import load_model_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from arm_rc_ctrl.experiments.esn_search import FloatRange, IntRange
    from arm_rc_ctrl.experiments.manual_recipes import ManualAnchor, ManualArmSpec
    from arm_rc_ctrl.experiments.manual_search import ManualSearchProtocol, ManualSearchSpace
    from arm_rc_ctrl.experiments.manual_study import StudyManifest, StudyModel
    from arm_rc_ctrl.rc.esn import ReadoutConfig

__all__ = [
    "CONFIGURATION_PREFIX",
    "LEARNED_ARMS",
    "SAMPLED_PARAMETERS",
    "SampledPoint",
    "configuration_label",
    "contractive_bank",
    "fixed_policy_mismatches",
    "point_mismatches",
    "sampled_configuration",
    "sampled_entry",
    "sampled_esn",
    "sampled_point",
]

CONFIGURATION_PREFIX: Final = "search-t"
"""Namespace of sampled configuration labels; the closed study's six are words, never ``search-t…``."""
SAMPLED_PARAMETERS: Final = (
    "n_neurons",
    "spectral_radius",
    "sparsity",
    "leak_rate",
    "input_scaling",
    "alpha_0",
    "warmup_s",
)
"""Exactly what a trial samples. Anything else in a trial's parameters is a scope error."""
LEARNED_ARMS: Final = manual_arms()
"""The closed experiment's learned arms: ten singletons, the all-ten model, ten copies, ten synthetic."""


@dataclass(frozen=True)
class SampledPoint:
    """One trial's draw: the reservoir, the base regularization and the warm-up."""

    n_neurons: int
    spectral_radius: float
    sparsity: float
    leak_rate: float
    input_scaling: float
    alpha_0: float
    warmup_s: float

    def __post_init__(self) -> None:
        """Every sampled value is a usable number; the space decides whether it is an approved one."""
        values = (
            self.spectral_radius,
            self.sparsity,
            self.leak_rate,
            self.input_scaling,
            self.alpha_0,
            self.warmup_s,
        )
        if not all(math.isfinite(value) for value in values):
            msg = f"a sampled point must be finite, got {self}"
            raise ValueError(msg)
        if self.n_neurons < 1:
            msg = f"n_neurons must be positive, got {self.n_neurons}"
            raise ValueError(msg)
        if self.warmup_s < 0.0 or self.alpha_0 <= 0.0:
            msg = f"warm-up is non-negative and alpha_0 positive, got {self.warmup_s} and {self.alpha_0}"
            raise ValueError(msg)


def configuration_label(trial: int) -> str:
    """The label of the configuration a trial defines, unique per trial and outside the study's namespace."""
    if trial < 0:
        msg = f"a trial number is non-negative, got {trial}"
        raise ValueError(msg)
    return f"{CONFIGURATION_PREFIX}{trial:04d}"


def _range_complaint(name: str, value: float, bounds: FloatRange | IntRange) -> str | None:
    """Why a value is not in an approved range, if it is not."""
    if not bounds.contains(value):  # pyright: ignore[reportArgumentType] - both ranges accept their own type
        return f"{name}: {value!r} is outside the approved range [{bounds.low}, {bounds.high}]"
    return None


def point_mismatches(space: ManualSearchSpace, point: SampledPoint) -> list[str]:
    """Every way a point departs from the approved space.

    This is the one validator. A point reaches training through a
    configuration, so the configuration checks it too: a point built directly
    is as trainable as one read from a trial's parameters, and requiring the
    caller to remember which door to use would be no protection at all.
    """
    return [
        complaint
        for complaint in (
            _range_complaint("n_neurons", point.n_neurons, space.n_neurons),
            _range_complaint("spectral_radius", point.spectral_radius, space.spectral_radius),
            _range_complaint("sparsity", point.sparsity, space.sparsity),
            _range_complaint("leak_rate", point.leak_rate, space.leak_rate),
            _range_complaint("input_scaling", point.input_scaling, space.input_scaling),
            _range_complaint("alpha_0", point.alpha_0, space.alpha_0),
            (
                None
                if point.warmup_s in space.warmup_s
                else f"warmup_s: {point.warmup_s!r} is not one of {space.warmup_s}"
            ),
        )
        if complaint is not None
    ]


def _integral(name: str, value: float) -> int:
    """``value`` as the integer it exactly is, refusing one that would be truncated."""
    number = float(value)
    if not number.is_integer():
        msg = f"{name}: {value!r} is not a whole number; truncating it would train another point"
        raise ValueError(msg)
    return int(number)


def sampled_point(space: ManualSearchSpace, params: Mapping[str, float]) -> SampledPoint:
    """Read a trial's parameters as a point of the approved space, refusing anything else.

    Optuna hands back a mapping; this is where it becomes a checked value. A
    missing parameter is never defaulted and an extra one is never ignored,
    because either would mean the trial searched something the owner did not
    approve. A reservoir size is checked for integrality before it is converted,
    so a fractional value is refused rather than quietly truncated onto the
    grid.
    """
    missing = [name for name in SAMPLED_PARAMETERS if name not in params]
    unknown = [name for name in params if name not in SAMPLED_PARAMETERS]
    if missing or unknown:
        msg = f"a trial samples exactly {list(SAMPLED_PARAMETERS)}; missing {missing}, unknown {unknown}"
        raise ValueError(msg)
    point = SampledPoint(
        n_neurons=_integral("n_neurons", params["n_neurons"]),
        spectral_radius=float(params["spectral_radius"]),
        sparsity=float(params["sparsity"]),
        leak_rate=float(params["leak_rate"]),
        input_scaling=float(params["input_scaling"]),
        alpha_0=float(params["alpha_0"]),
        warmup_s=float(params["warmup_s"]),
    )
    complaints = point_mismatches(space, point)
    if complaints:
        msg = "; ".join(complaints)
        raise ValueError(msg)
    return point


def sampled_configuration(protocol: ManualSearchProtocol, point: SampledPoint, *, trial: int) -> StudyConfiguration:
    """The configuration a trial defines: its sampled point over the protocol's fixed policy.

    The point is checked against the approved space here as well as where a
    trial's parameters are read, because this is where a point becomes
    trainable: a directly constructed point is no less real than a sampled one.

    The reservoir is the base model configuration's with the sampled
    parameters and the fixed seed, so anything the base file declares and the
    search does not touch is inherited rather than restated. The estimator
    cutoffs are the fixed policy's, which is what makes every trial comparable.
    """
    complaints = point_mismatches(protocol.space, point)
    if complaints:
        msg = "; ".join(complaints)
        raise ValueError(msg)
    base = load_model_config(protocol.model)
    reservoir = replace(
        base.esn.reservoir,
        n_neurons=point.n_neurons,
        spectral_radius=point.spectral_radius,
        sparsity=point.sparsity,
        leak_rate=point.leak_rate,
        input_scaling=point.input_scaling,
        seed=protocol.fixed.reservoir_seed,
    )
    return StudyConfiguration(
        label=configuration_label(trial),
        source_trial=trial,
        warmup_s=point.warmup_s,
        base_alpha=point.alpha_0,
        reservoir=reservoir,
        velocity_cutoff_hz=protocol.fixed.velocity_cutoff_hz,
        acceleration_cutoff_hz=protocol.fixed.acceleration_cutoff_hz,
    )


def fixed_policy_mismatches(protocol: ManualSearchProtocol, configuration: StudyConfiguration) -> list[str]:
    """Every way a configuration departs from what the protocol holds fixed.

    Sampling moves the ESN and the warm-up and nothing else, so this is
    checkable after the fact: evidence produced under a drifted filter policy
    or another reservoir seed is not this search's evidence.
    """
    fixed = protocol.fixed
    pairs = (
        ("velocity_cutoff_hz", configuration.velocity_cutoff_hz, fixed.velocity_cutoff_hz),
        ("acceleration_cutoff_hz", configuration.acceleration_cutoff_hz, fixed.acceleration_cutoff_hz),
        ("reservoir.seed", configuration.reservoir.seed, fixed.reservoir_seed),
    )
    return [
        f"{configuration.label}: {name} is {actual!r}, but the protocol fixes {expected!r}"
        for name, actual, expected in pairs
        if actual != expected
    ]


def arm_of(kind: str) -> ManualArmSpec:
    """The first learned arm of ``kind``; the arms themselves are the closed experiment's."""
    for arm in LEARNED_ARMS:
        if arm.arm == kind:
            return arm
    msg = f"no learned arm of kind {kind!r}"
    raise KeyError(msg)


def sampled_esn(
    configuration: StudyConfiguration,
    arm: ManualArmSpec,
    *,
    readout: ReadoutConfig,
    anchor: ManualAnchor = MANUAL_ANCHOR,
) -> EsnConfig:
    """The ESN one arm of a sampled configuration fits: its reservoir with the arm's scaled readout.

    This is the closed experiment's own construction (:func:`esn_for_arm`) over
    a sampled reservoir, so the readout layout, the explicit bias and the
    count-scaled ridge are whatever the study already decided; only the
    reservoir and ``alpha_0`` come from the trial.
    """
    return esn_for_arm(
        EsnConfig(reservoir=configuration.reservoir, readout=readout),
        arm,
        base_alpha=configuration.base_alpha,
        anchor=anchor,
    )


def sampled_entry(study: StudyManifest, configuration: StudyConfiguration, arm: ManualArmSpec) -> StudyModel:
    """One arm of a sampled configuration, bound exactly as the frozen study binds its own.

    The fit identity comes from the study's own derivation
    (:func:`~arm_rc_ctrl.experiments.manual_study.study_model`), so a sampled
    entry is keyed by the same rule as a frozen one and the two can never
    collide: the configuration label differs, and the label is part of the key.
    Everything the search does not sample -- the readout, the ten sources, the
    frozen transform, the validation, the anchor, the library and the
    environment -- is the closed study's.
    """
    banks: dict[str, ContractiveBank] = {}
    if arm.arm == CONTRACTIVE_ARM:
        assignment = arm.assignment
        if assignment is None:
            msg = f"{arm.label}: a contractive arm names the parent it grows from"
            raise ValueError(msg)
        banks[assignment] = contractive_bank(study, assignment)
    return study_model(
        configuration,
        arm,
        readout=study.readout,
        sources=study.sources,
        banks=banks,
        transform=study.transform.transform,
        validation=study.validation,
        anchor=study.anchor,
        rclib_commit=study.rclib.commit,
        execution_identity=study.execution.identity,
    )


def contractive_bank(study: StudyManifest, assignment: str) -> ContractiveBank:
    """The frozen contractive bank of one parent, inherited from the closed study unchanged.

    Synthetic variation is not tuned here, so a sampled configuration's ``C10``
    arm regenerates from exactly the construction the closed experiment
    accepted: the same seed bank, the same recorded dwell onset and the same
    frozen bank digest. The construction is per parent, so every configuration
    of the closed study must agree about it; a study that does not is refused
    rather than silently read from its first entry.
    """
    banks = {
        entry.contractive
        for entry in study.entries
        if entry.contractive is not None and entry.arm.assignment == assignment
    }
    if not banks:
        msg = f"the study holds no contractive bank for {assignment!r}"
        raise KeyError(msg)
    if len(banks) != 1:
        msg = f"{assignment}: the study's configurations disagree about the contractive construction: {banks}"
        raise ValueError(msg)
    return banks.pop()
