# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Recipe construction for the manual-demonstration arms (M3MAN-005; manual plan section 4, D4/D5, I3/I4/I8).

Every behavioral configuration of ``task_1a_manual_v1`` is one inherited ESN
configuration (reservoir, ``alpha_0``, warm-up, estimator cutoffs) combined
with one arm over the locked demonstration bank:

- ``S_i``: demonstration ``D_i`` once, at ``alpha_0``;
- ``M10``: all ten demonstrations once each, at ``10 alpha_0``;
- ``R10_i``: ten exact copies of ``D_i``, at ``10 alpha_0`` — the
  episode-count and weight-matched duplication control.

``C10_i`` (the contractive arm) is built by the augmentation task and is not
constructed here.

Every arm shares one construction, frozen in :data:`MANUAL_ANCHOR`: each
episode is a complete recording (its recorded pre-roll trains too), episodes
carry equal total loss weight through the per-row weight ``400 / L_i`` with
``solver_alpha = K alpha_0`` — equivalent to the mean-per-episode loss with
``lambda = alpha_0 / 400`` — the readout fits an explicit ones column instead
of ``rclib``'s implicit bias (I3), copies are recipe multiplicities rather than
duplicated payloads, and the input transform is copied from the historical
scripted dataset, digest-bound, never computed from a manual take (I8), so a
singleton receives no statistics from the other nine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.records import load_record
from arm_rc_ctrl.provenance import canonical_json, sha256_bytes
from arm_rc_ctrl.rc.esn import EsnConfig
from arm_rc_ctrl.rc.recipe import (
    EQUAL_EPISODE_WEIGHTING,
    DatasetSource,
    TrainingSpec,
    create_recipe,
    solver_alpha,
)
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from arm_rc_ctrl.data.records import Preprocessing
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.manual_bank import BankManifest
    from arm_rc_ctrl.rc.esn import EsnModel, ReadoutConfig
    from arm_rc_ctrl.rc.recipe import ModelRecipe, TrainingValidation
    from arm_rc_ctrl.rc.teacher_forcing import InputTransform
    from arm_rc_ctrl.scenario import ScenarioConfig

__all__ = [
    "ARMS",
    "ASSIGNMENTS",
    "BANK_SIZE",
    "COPIES",
    "MANUAL_ANCHOR",
    "ROWS_REFERENCE",
    "TRANSFORM_SOURCE",
    "ArmAccounting",
    "ManualAnchor",
    "ManualArmSpec",
    "arm_accounting",
    "arm_sources",
    "bank_sources",
    "esn_for_arm",
    "fit_identity",
    "manual_arms",
    "readout_for_arm",
    "recipe_for_arm",
    "training_spec_for_arm",
]

ARMS: Final = ("S", "M10", "R10")
"""The arms this module builds (D4); the contractive arm ``C10`` is the augmentation task's."""
BANK_SIZE: Final = 10
"""Accepted demonstrations in the locked bank, assigned ``D01`` .. ``D10``."""
ASSIGNMENTS: Final = tuple(f"D{index:02d}" for index in range(1, BANK_SIZE + 1))
COPIES: Final = 10
"""Episodes of the duplication control ``R10``: ten exact copies of one demonstration."""
ROWS_REFERENCE: Final = 400
"""The historical row-count reference ``R`` of the weight ``R / L_i``; not a required length for new takes."""
TRANSFORM_SOURCE: Final = DatasetSource(
    "processed-20260830-feaf73e6663c",
    "feaf73e6663c865cf9232c5a620bab5729ea02e6bd8c0181ad3dae5733540767",
    "data/records/processed/processed-20260830-feaf73e6663c.toml",
)
"""The historical scripted dataset whose recorded statistics supply the frozen input transform (I8)."""
_RECORD_DIRECTORY: Final = "data/records/processed"


@dataclass(frozen=True)
class ManualAnchor:
    """The frozen training construction every manual arm shares (plan section 4; I3, I4, I8)."""

    weight_reference_rows: int = ROWS_REFERENCE
    episode_weighting: str = EQUAL_EPISODE_WEIGHTING
    regularization_rule: str = "count_scaled"
    target: str = "next_q"
    """Absolute next position for v1; residual replication is a later decision."""
    washout: str = "warmup_hold"
    copies: int = COPIES
    transform_source: DatasetSource = TRANSFORM_SOURCE

    def __post_init__(self) -> None:
        """The anchor states the approved rule; none of it is a per-run choice."""
        if self.weight_reference_rows < 1 or self.copies < 1:
            msg = (
                f"weight_reference_rows and copies must be positive, got {self.weight_reference_rows} and {self.copies}"
            )
            raise ValueError(msg)
        if self.episode_weighting != EQUAL_EPISODE_WEIGHTING or self.regularization_rule != "count_scaled":
            msg = (
                f"the approved construction is {EQUAL_EPISODE_WEIGHTING!r} weighting with the 'count_scaled' ridge "
                f"rule, got {self.episode_weighting!r} and {self.regularization_rule!r}"
            )
            raise ValueError(msg)

    def regularization_lambda(self, base_alpha: float) -> float:
        """``alpha_0 / R``: the regularization scale of the declared mean-per-episode objective."""
        if not base_alpha > 0:
            msg = f"base_alpha must be positive, got {base_alpha!r}"
            raise ValueError(msg)
        return base_alpha / self.weight_reference_rows


MANUAL_ANCHOR: Final = ManualAnchor()


@dataclass(frozen=True)
class ManualArmSpec:
    """One approved arm: a singleton, the all-ten model, or a ten-copy duplication control."""

    arm: str
    assignment: str | None = None
    """``D01`` .. ``D10`` for ``S`` and ``R10``; ``None`` for ``M10``, which uses the whole bank."""

    def __post_init__(self) -> None:
        """Only the approved arm and assignment combinations exist."""
        if self.arm not in ARMS:
            msg = f"arm must be one of {ARMS}, got {self.arm!r}"
            raise ValueError(msg)
        if self.arm == "M10":
            if self.assignment is not None:
                msg = "M10 trains on the whole bank and carries no assignment"
                raise ValueError(msg)
        elif self.assignment not in ASSIGNMENTS:
            msg = f"{self.arm} needs an assignment in {list(ASSIGNMENTS)}, got {self.assignment!r}"
            raise ValueError(msg)

    @property
    def label(self) -> str:
        """``M10`` or ``<arm>/<assignment>``, e.g. ``S/D01`` or ``R10/D07``."""
        return self.arm if self.assignment is None else f"{self.arm}/{self.assignment}"

    @property
    def assignments(self) -> tuple[str, ...]:
        """The demonstrations the arm trains on, in bank order."""
        return ASSIGNMENTS if self.assignment is None else (self.assignment,)

    @property
    def copies_per_source(self) -> int:
        """Episodes each demonstration contributes: one, except the ten of the duplication control."""
        return COPIES if self.arm == "R10" else 1

    @property
    def source_counts(self) -> tuple[int, ...]:
        """The recipe's source multiplicities, aligned with :attr:`assignments`."""
        return tuple(self.copies_per_source for _ in self.assignments)

    @property
    def count(self) -> int:
        """The episode count ``K`` the ridge parameter and the weighting refer to."""
        return sum(self.source_counts)

    @property
    def unique_sources(self) -> int:
        """Distinct manual demonstrations the fit sees."""
        return len(self.assignments)

    @property
    def copies(self) -> int:
        """Episodes that are exact copies of another episode of the same fit."""
        return self.count - self.unique_sources


def manual_arms() -> tuple[ManualArmSpec, ...]:
    """Every approved arm in report order: ten singletons, the all-ten model, and ten duplication controls."""
    singles = [ManualArmSpec("S", assignment) for assignment in ASSIGNMENTS]
    repeats = [ManualArmSpec("R10", assignment) for assignment in ASSIGNMENTS]
    return (*singles, ManualArmSpec("M10"), *repeats)


def training_spec_for_arm(
    arm: ManualArmSpec, *, warmup_s: float, base_alpha: float, anchor: ManualAnchor = MANUAL_ANCHOR
) -> TrainingSpec:
    """The arm's training construction: inherited warm-up, equal-episode weighting, and its multiplicities."""
    return TrainingSpec(
        washout=anchor.washout,
        warmup_s=warmup_s,
        target=anchor.target,
        episode_weighting=anchor.episode_weighting,
        weight_reference_rows=anchor.weight_reference_rows,
        source_counts=arm.source_counts,
        base_alpha=base_alpha,
        regularization_rule=anchor.regularization_rule,
    )


def readout_for_arm(
    base: ReadoutConfig, arm: ManualArmSpec, *, base_alpha: float, anchor: ManualAnchor = MANUAL_ANCHOR
) -> ReadoutConfig:
    """The inherited readout at ``K alpha_0`` with the explicit-bias layout the weighting needs (I3)."""
    return replace(
        base,
        alpha=solver_alpha(base_alpha, anchor.regularization_rule, arm.count),
        include_bias=False,
        explicit_bias=True,
    )


def esn_for_arm(
    base: EsnConfig, arm: ManualArmSpec, *, base_alpha: float, anchor: ManualAnchor = MANUAL_ANCHOR
) -> EsnConfig:
    """The inherited reservoir with the arm's readout; reservoir weights never differ between arms."""
    return EsnConfig(
        reservoir=base.reservoir, readout=readout_for_arm(base.readout, arm, base_alpha=base_alpha, anchor=anchor)
    )


def bank_sources(bank: BankManifest, *, root: Path | None = None) -> dict[str, DatasetSource]:
    """``D01`` .. ``D10`` mapped to their digest-bound processed datasets, read from the Git-tracked records.

    The bank records which take each assignment is; the payload digest comes
    from that take's committed record, so a recipe binds the dataset bytes and
    never a machine path.
    """
    # Imported here so the arm algebra above stays importable without the manual schema's heavy dependencies.
    from arm_rc_ctrl.data.manual import ManualDatasetRecord

    if not bank.complete:
        msg = f"the bank is incomplete: {bank.shortfall} more accepted takes are needed before arms can be built"
        raise ValueError(msg)
    base = repository_root() if root is None else root
    sources: dict[str, DatasetSource] = {}
    for take in bank.takes:
        if take.assignment is None:
            continue
        artifact = cast("str", take.processed_artifact_id)
        relative = f"{_RECORD_DIRECTORY}/{artifact}.toml"
        record = load_record(base / relative, ManualDatasetRecord)
        if record.artifact.artifact_id != artifact:
            msg = f"record {relative} describes {record.artifact.artifact_id}, not {artifact}"
            raise ValueError(msg)
        sources[take.assignment] = DatasetSource(artifact, record.artifact.payload.sha256, relative)
    missing = [name for name in ASSIGNMENTS[: bank.required] if name not in sources]
    if missing:
        msg = f"the bank names no dataset for {missing}"
        raise ValueError(msg)
    return sources


def arm_sources(arm: ManualArmSpec, sources: Mapping[str, DatasetSource]) -> tuple[DatasetSource, ...]:
    """The arm's training datasets in training order, resolved through the bank's assignments."""
    missing = [name for name in arm.assignments if name not in sources]
    if missing:
        msg = f"arm {arm.label} needs datasets for {missing}"
        raise ValueError(msg)
    return tuple(sources[name] for name in arm.assignments)


@dataclass(frozen=True)
class ArmAccounting:
    """What one arm's fit is made of: sources, multiplicities, raw rows, loss weights, and the ridge scales."""

    label: str
    arm: str
    assignments: tuple[str, ...]
    unique_sources: int
    copies: int
    episodes: int
    loss_rows: tuple[int, ...]
    """Raw loss rows of every episode in training order; a copy repeats its parent's count."""
    row_weights: tuple[float, ...]
    """The per-row weight ``R / L_i`` each episode is fitted with."""
    total_loss_weight: float
    """``sum_i w_i L_i = R K``: every episode contributes the same total weight."""
    base_alpha: float
    solver_alpha: float
    regularization_lambda: float
    """``alpha_0 / R``, the scale of the equivalent mean-per-episode objective."""
    weight_reference_rows: int

    def __post_init__(self) -> None:
        """The accounting is internally consistent: one row count and weight per episode, all positive."""
        if len(self.loss_rows) != self.episodes or len(self.row_weights) != self.episodes:
            msg = f"{self.label}: one loss-row count and one weight per episode ({self.episodes}) are required"
            raise ValueError(msg)
        if self.episodes != self.unique_sources + self.copies or self.unique_sources < 1:
            msg = f"{self.label}: episodes must be the unique sources plus their copies"
            raise ValueError(msg)
        if any(rows < 1 for rows in self.loss_rows) or any(weight <= 0 for weight in self.row_weights):
            msg = f"{self.label}: loss rows and weights must be positive"
            raise ValueError(msg)


def arm_accounting(
    arm: ManualArmSpec,
    *,
    base_alpha: float,
    loss_rows: Mapping[str, int],
    anchor: ManualAnchor = MANUAL_ANCHOR,
) -> ArmAccounting:
    """Account for one arm from the raw loss-row count of every demonstration it trains on.

    ``loss_rows`` maps ``D01`` .. ``D10`` to the loss rows of that recording
    (its samples minus the one row the next-step pairing consumes); warm-up
    rows are not loss rows and never appear here.
    """
    missing = [name for name in arm.assignments if name not in loss_rows]
    if missing:
        msg = f"arm {arm.label} needs the loss rows of {missing}"
        raise ValueError(msg)
    rows: list[int] = []
    for name in arm.assignments:
        rows.extend([loss_rows[name]] * arm.copies_per_source)
    weights = tuple(anchor.weight_reference_rows / count for count in rows)
    return ArmAccounting(
        label=arm.label,
        arm=arm.arm,
        assignments=arm.assignments,
        unique_sources=arm.unique_sources,
        copies=arm.copies,
        episodes=arm.count,
        loss_rows=tuple(rows),
        row_weights=weights,
        total_loss_weight=float(sum(weight * count for weight, count in zip(weights, rows, strict=True))),
        base_alpha=base_alpha,
        solver_alpha=solver_alpha(base_alpha, anchor.regularization_rule, arm.count),
        regularization_lambda=anchor.regularization_lambda(base_alpha),
        weight_reference_rows=anchor.weight_reference_rows,
    )


def fit_identity(
    *,
    configuration: str,
    arm: ManualArmSpec,
    warmup_s: float,
    base_alpha: float,
    esn: EsnConfig,
    datasets: tuple[DatasetSource, ...],
    transform: InputTransform,
    validation: TrainingValidation,
    anchor: ManualAnchor = MANUAL_ANCHOR,
    rclib_commit: str,
    execution_identity: str,
) -> str:
    """SHA-256 identity of one fit: every input that decides the fitted readout, including the environment.

    The execution environment (:attr:`~arm_rc_ctrl.execution.ExecutionRecord.identity`)
    is part of the key, so a fit produced on another core type or thread
    setting is never served from a cache (clarification C10).
    """
    mapping = {
        "configuration": configuration,
        "arm": to_mapping(arm),
        "label": arm.label,
        "source_counts": list(arm.source_counts),
        "count": arm.count,
        "warmup_s": warmup_s,
        "base_alpha": base_alpha,
        "solver_alpha": solver_alpha(base_alpha, anchor.regularization_rule, arm.count),
        "anchor": to_mapping(anchor),
        "esn": to_mapping(esn),
        "datasets": [to_mapping(dataset) for dataset in datasets],
        "transform": to_mapping(transform),
        "validation": to_mapping(validation),
        "rclib_commit": rclib_commit,
        "execution_identity": execution_identity,
    }
    return sha256_bytes(canonical_json(mapping).encode("utf-8"))


def recipe_for_arm(
    arm: ManualArmSpec,
    *,
    esn: EsnConfig,
    sources: Mapping[str, DatasetSource],
    samples: Mapping[str, SampleSet],
    dof: int,
    task_code_dim: int,
    preprocessing: Preprocessing,
    transform: InputTransform,
    validation: TrainingValidation,
    warmup_s: float,
    base_alpha: float,
    scenario: ScenarioConfig | None = None,
    anchor: ManualAnchor = MANUAL_ANCHOR,
    name: str | None = None,
) -> tuple[ModelRecipe, EsnModel]:
    """Fit one arm over the locked bank and return the schema 3 recipe that reproduces it, plus the model.

    The reservoir is the inherited configuration's, the readout is its
    explicit-bias form at ``K alpha_0``, the transform must be the frozen one
    copied from :attr:`ManualAnchor.transform_source`, and the copies of
    ``R10`` are recorded as source multiplicities, never as duplicated data.
    """
    if transform.derived_from != (anchor.transform_source.artifact_id,):
        msg = (
            f"the input transform derives from {transform.derived_from}, not from the frozen source "
            f"{anchor.transform_source.artifact_id}; manual arms never compute statistics from a take (I8)"
        )
        raise ValueError(msg)
    datasets = arm_sources(arm, sources)
    return create_recipe(
        arm.label if name is None else name,
        esn_for_arm(esn, arm, base_alpha=base_alpha, anchor=anchor),
        sources=datasets,
        samples=samples,
        dof=dof,
        task_code_dim=task_code_dim,
        preprocessing=preprocessing,
        transform=transform,
        training=training_spec_for_arm(arm, warmup_s=warmup_s, base_alpha=base_alpha, anchor=anchor),
        scenario=scenario,
        validation=validation,
        transform_source=anchor.transform_source,
    )
