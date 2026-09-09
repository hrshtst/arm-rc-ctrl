# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Recipe construction for the repeated-demonstration pilot's arms (M3REP-002; repetition plan sections 4, 5.2, 5.3, 8).

Every behavioral configuration of ``task_1a_repetition_v1`` is one frozen panel
entry (a source trial's reservoir, readout parameter ``alpha_0``, warm-up, and
estimator cutoffs) combined with one arm at one episode count ``K``:

- ``S``: the original episode once at ``alpha_0``;
- ``R``: ``K`` exact copies at ``alpha_0``;
- ``R-scaled``: ``K`` exact copies at ``K * alpha_0``;
- ``A-non-decaying`` / ``A-contractive``: the original plus ``K - 1`` synthetic
  episodes of the fixed augmentation anchor at ``alpha_0`` (absolute only);
- ``S-effective``: the original once at ``alpha_0 / K``, the numerical
  reference of ``R``.

This module turns an arm into a schema 2 :class:`~arm_rc_ctrl.rc.recipe.ModelRecipe`
whose ridge parameter is constructed directly from the rule (never through a
search-space validator), whose training validation is bound to the scenario
file, and whose fit identity binds the execution environment
(:attr:`~arm_rc_ctrl.execution.ExecutionRecord.identity`) so later fit caches
and evidence can never mix environments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.provenance import canonical_json, sha256_bytes
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig
from arm_rc_ctrl.rc.recipe import (
    APPROVED_ADDITIONAL_REPEATS,
    AugmentationTrainingSpec,
    DatasetSource,
    ModelRecipe,
    TrainingSpec,
    TrainingValidation,
    create_recipe,
    solver_alpha,
)
from arm_rc_ctrl.rc.teacher_forcing import InputTransform

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from arm_rc_ctrl.data.records import Normalization, Preprocessing
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.repetition_panel import PanelEntry
    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.train import ModelConfig
    from arm_rc_ctrl.scenario import ScenarioConfig

__all__ = [
    "ARMS",
    "AUGMENTATION_ANCHOR",
    "BEHAVIORAL_ARMS",
    "FORMULATIONS",
    "RESIDUAL_ARMS",
    "ROWS_PER_EPISODE",
    "ArmSpec",
    "AugmentationAnchor",
    "expected_loss_rows",
    "fit_identity",
    "panel_arms",
    "readout_for_arm",
    "recipe_for_arm",
    "training_spec_for_arm",
]

FORMULATIONS: Final = ("absolute", "residual")
ARMS: Final = ("S", "R", "R-scaled", "A-non-decaying", "A-contractive", "S-effective")
RESIDUAL_ARMS: Final = ("S", "R", "R-scaled", "S-effective")
"""Residual arms use the original and its exact copies only (decision D2)."""
BEHAVIORAL_ARMS: Final = ("S", "R", "R-scaled", "A-non-decaying", "A-contractive")
"""Arms that enter the behavioral sweep; ``S-effective`` is fitted and compared numerically only."""
ROWS_PER_EPISODE: Final = 400
"""Loss rows of the recovery episode (401 task samples paired into 400 next-step rows)."""
_RULES: Final = {
    "S": "base",
    "R": "base",
    "R-scaled": "count_scaled",
    "A-non-decaying": "base",
    "A-contractive": "base",
    "S-effective": "count_divided",
}
_FAMILIES: Final = {"A-non-decaying": "non_decaying", "A-contractive": "contractive"}


@dataclass(frozen=True)
class AugmentationAnchor:
    """The fixed augmentation settings of both varied-data arms (decision D4)."""

    sigma_rad: float = 0.05
    phi: float = 0.99
    gamma: float = 1.0
    seed_bank: int = 1
    attempt_factor: int = 4

    def spec(self, family: str, n_synthetic: int) -> AugmentationTrainingSpec:
        """The training augmentation of one varied-data arm at ``n_synthetic = K - 1`` episodes."""
        if family not in ("non_decaying", "contractive"):
            msg = f"family must be 'non_decaying' or 'contractive', got {family!r}"
            raise ValueError(msg)
        return AugmentationTrainingSpec(
            family=family,
            n_synthetic=n_synthetic,
            sigma_rad=self.sigma_rad,
            phi=self.phi,
            gamma=self.gamma,
            seed_bank=self.seed_bank,
            attempt_budget=self.attempt_factor * n_synthetic,
        )


AUGMENTATION_ANCHOR: Final = AugmentationAnchor()


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the pilot at one output formulation and episode count."""

    formulation: str
    arm: str
    additional_repeats: int | None = None
    """``K - 1`` for every arm but ``S`` (which has no count)."""

    def __post_init__(self) -> None:
        """Only the approved arm/formulation/count combinations exist."""
        if self.formulation not in FORMULATIONS or self.arm not in ARMS:
            msg = (
                f"formulation must be one of {FORMULATIONS} and arm one of {ARMS}, "
                f"got {self.formulation!r}/{self.arm!r}"
            )
            raise ValueError(msg)
        if self.formulation == "residual" and self.arm not in RESIDUAL_ARMS:
            msg = f"residual arms are {RESIDUAL_ARMS}; augmentation is outside this pilot for residual output (D2)"
            raise ValueError(msg)
        if self.arm == "S":
            if self.additional_repeats is not None:
                msg = "the S arm has no episode count"
                raise ValueError(msg)
        elif self.additional_repeats not in APPROVED_ADDITIONAL_REPEATS:
            msg = (
                f"{self.arm} needs additional_repeats in {sorted(APPROVED_ADDITIONAL_REPEATS)}, "
                f"got {self.additional_repeats!r}"
            )
            raise ValueError(msg)

    @property
    def count(self) -> int:
        """The matched episode count ``K`` (1 for ``S``)."""
        return 1 + (self.additional_repeats or 0)

    @property
    def label(self) -> str:
        """``<formulation>/<arm>`` with ``/K<count>`` for counted arms, e.g. ``absolute/R/K17``."""
        return f"{self.formulation}/{self.arm}" + ("" if self.arm == "S" else f"/K{self.count}")

    @property
    def regularization_rule(self) -> str:
        """The rule deriving the solver's ridge parameter from ``alpha_0``."""
        return _RULES[self.arm]

    @property
    def target(self) -> str:
        """The training target of the formulation."""
        return "increment_q" if self.formulation == "residual" else "next_q"

    @property
    def behavioral(self) -> bool:
        """Whether the arm enters the behavioral sweep (``S-effective`` is numerical only)."""
        return self.arm in BEHAVIORAL_ARMS

    @property
    def episodes_trained(self) -> int:
        """Episodes the fit stacks: ``K`` for repeated and augmented arms, one for ``S`` and ``S-effective``."""
        return 1 if self.arm in ("S", "S-effective") else self.count


def panel_arms(*, counts: tuple[int, ...] = tuple(sorted(APPROVED_ADDITIONAL_REPEATS))) -> tuple[ArmSpec, ...]:
    """Every arm of both formulations at every approved count, in report order (6 + 12 + 6 = 24 per entry)."""
    arms: list[ArmSpec] = []
    for formulation in FORMULATIONS:
        allowed = RESIDUAL_ARMS if formulation == "residual" else ARMS
        arms.append(ArmSpec(formulation, "S"))
        arms.extend(ArmSpec(formulation, arm, repeats) for repeats in counts for arm in allowed if arm != "S")
    return tuple(arms)


def training_spec_for_arm(arm: ArmSpec, *, warmup_s: float, base_alpha: float) -> TrainingSpec:
    """The training construction of ``arm``: warm-up washout, target, repetition or augmentation, and ridge rule."""
    family = _FAMILIES.get(arm.arm)
    return TrainingSpec(
        washout="warmup_hold",
        warmup_s=warmup_s,
        target=arm.target,
        augmentation=None if family is None else AUGMENTATION_ANCHOR.spec(family, arm.count - 1),
        additional_repeats=arm.additional_repeats if arm.arm in ("R", "R-scaled") else None,
        base_alpha=base_alpha,
        regularization_rule=arm.regularization_rule,
        regularization_count=arm.count,
    )


def readout_for_arm(base: ReadoutConfig, arm: ArmSpec, *, base_alpha: float) -> ReadoutConfig:
    """The source readout with the solver parameter the arm prescribes (constructed directly, never clipped)."""
    return replace(base, alpha=solver_alpha(base_alpha, arm.regularization_rule, arm.count))


def expected_loss_rows(arm: ArmSpec, *, rows_per_episode: int = ROWS_PER_EPISODE) -> int:
    """Loss rows the arm's stacked fit holds (6800/13200/26000 for repeated arms on the recovery episode)."""
    if rows_per_episode < 1:
        msg = f"rows_per_episode must be positive, got {rows_per_episode}"
        raise ValueError(msg)
    return arm.episodes_trained * rows_per_episode


def fit_identity(
    *,
    panel_label: str,
    source_trial: int,
    arm: ArmSpec,
    warmup_s: float,
    base_alpha: float,
    solver_alpha: float,
    dataset: DatasetSource,
    transform: InputTransform,
    validation: TrainingValidation,
    rclib_commit: str,
    execution_identity: str,
) -> str:
    """SHA-256 identity of one fit for caches and evidence: every input that decides the fitted readout.

    The execution environment (:attr:`~arm_rc_ctrl.execution.ExecutionRecord.identity`)
    is part of the key, so a fit produced on another core type or thread
    setting is never served from the cache (clarification C10).
    """
    mapping = {
        "panel_label": panel_label,
        "source_trial": source_trial,
        "arm": to_mapping(arm),
        "formulation": arm.formulation,
        "warmup_s": warmup_s,
        "base_alpha": base_alpha,
        "solver_alpha": solver_alpha,
        "regularization_rule": arm.regularization_rule,
        "count": arm.count,
        "dataset": to_mapping(dataset),
        "transform": to_mapping(transform),
        "validation": to_mapping(validation),
        "rclib_commit": rclib_commit,
        "execution_identity": execution_identity,
    }
    return sha256_bytes(canonical_json(mapping).encode("utf-8"))


def recipe_for_arm(
    entry: PanelEntry,
    arm: ArmSpec,
    *,
    base: ModelConfig,
    source: DatasetSource,
    samples: Mapping[str, SampleSet],
    dof: int,
    task_code_dim: int,
    preprocessing: Preprocessing,
    normalization: Normalization,
    scenario: ScenarioConfig,
    scenario_file: Path,
    root: Path | None = None,
    name: str | None = None,
) -> tuple[ModelRecipe, EsnModel]:
    """Fit the arm for one panel entry and return the schema 2 recipe that reproduces it, plus the model.

    The reservoir comes from the entry's point, the readout from the arm's rule
    applied to the entry's ``base_alpha``, the input transform from the base
    model's policy over the dataset's recorded statistics, and the training
    validation from the scenario file the dataset is bound to.
    """
    config = entry.point.esn.model_config(base, name=name or f"{entry.label}/{arm.label}")
    esn = EsnConfig(
        reservoir=config.esn.reservoir, readout=readout_for_arm(config.esn.readout, arm, base_alpha=entry.base_alpha)
    )
    transform = InputTransform.derive(
        base.input_transform.policy, normalization, fixed_scales=base.input_transform.fixed_scales
    )
    validation = TrainingValidation.from_scenario(scenario, scenario_file, root=root)
    training = training_spec_for_arm(arm, warmup_s=entry.warmup_s, base_alpha=entry.base_alpha)
    return create_recipe(
        config.name,
        esn,
        sources=[source],
        samples=samples,
        dof=dof,
        task_code_dim=task_code_dim,
        preprocessing=preprocessing,
        transform=transform,
        training=training,
        scenario=scenario,
        validation=validation,
    )
