# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Deterministic model recipe (docs/PLAN.md section 8).

A recipe is the model artifact of task 1-a: the ESN hyperparameters and seeds,
the training data identity (artifact IDs and payload digests), the
preprocessing and normalization settings the inputs were built with, the
``rclib`` revision, the readout configuration, and the fit report obtained
when the recipe was created. Loading a recipe reconstructs the model and
refits it from the referenced datasets; the refit must reproduce the recorded
fit report within the declared tolerance. No pickle is ever written or read.

Schema 1 is the frozen M2/M3/M3R form. Schema 2 (M3REP-002; repetition plan
sections 4, 5.2, and 8, clarification C2) additionally binds the scenario file
and limits the training episodes were validated against, and permits the
exact-repetition construction (``additional_repeats`` literal copies of the
single source episode, each harvested from a reset reservoir into one stacked
ridge fit) with an explicit base ridge parameter and the rule that derives the
solver's parameter from it.

Schema 3 (M3MAN-005; manual plan section 4 and clarifications I3, I4, and I8)
is the manual-demonstration form: episodes of *variable* length whose recorded
pre-roll stays inside the loss, equal total loss weight per episode
(``episode_weighting``, ``weight_reference_rows``), exact copies expressed as
per-source multiplicities instead of duplicated payloads (``source_counts``),
the explicit-bias readout the weighting needs, and an input transform copied
from a dataset that is deliberately *not* among the training sources
(``transform_source``), so a singleton recipe receives no statistics from the
other takes.

Every new field defaults to ``None`` and is
stripped from the stored TOML and the recipe identity, so schema 1 records keep
their serialization and hashes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, cast

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.data.derivatives import DerivativeConfig
from arm_rc_ctrl.data.records import (
    Normalization,
    Preprocessing,
    ProcessedDatasetRecord,
    is_artifact_id,
    require_relative_posix,
)
from arm_rc_ctrl.data.records import to_toml as records_to_toml
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, task_intervals_from_phases
from arm_rc_ctrl.dependencies import submodule_revisions, submodule_version
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.rc.augment import AugmentationConfig, EpisodeArrays, generate_augmentation
from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel
from arm_rc_ctrl.rc.teacher_forcing import INPUT_CHANNELS, Episode, InputEncoder, InputTransform, build_episode
from arm_rc_ctrl.rc.training import FitReport, train_readout
from arm_rc_ctrl.rc.warmup import WarmupConfig, build_task_episode, build_task_episode_arrays
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import joint_limits
from arm_rc_ctrl.validation import COMMIT_HEX_LENGTH, SHA256_HEX_LENGTH, is_hex

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from arm_rc_ctrl.data.manual import ManualDatasetRecord
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.scenario import ScenarioConfig

    # Any processed-kind record a recipe trains from; the manual schema records no normalization by design (I8).
    type DatasetRecord = ProcessedDatasetRecord | RecoveryDatasetRecord | ManualDatasetRecord

__all__ = [
    "APPROVED_ADDITIONAL_REPEATS",
    "BINDING_SCHEMA_VERSION",
    "DERIVATIVE_METHODS",
    "EQUAL_EPISODE_WEIGHTING",
    "RECIPE_SCHEMA_VERSION",
    "REGULARIZATION_RULES",
    "SUPPORTED_RECIPE_SCHEMAS",
    "WEIGHTED_SCHEMA_VERSION",
    "AugmentationTrainingSpec",
    "DatasetSource",
    "FitTolerance",
    "ModelRecipe",
    "RclibIdentity",
    "RecipeMismatchError",
    "TrainingSpec",
    "TrainingValidation",
    "create_recipe",
    "derivative_config",
    "expected_episode_labels",
    "load_recipe",
    "solver_alpha",
    "write_recipe",
]

RECIPE_SCHEMA_VERSION: Final = 1
"""The frozen schema of every recipe written before M3REP-002 (and of legacy commands that keep writing it)."""
BINDING_SCHEMA_VERSION: Final = 2
"""The schema that binds training validation and permits repetition and ridge rules (M3REP-002, C2)."""
WEIGHTED_SCHEMA_VERSION: Final = 3
"""The schema of the manual-demonstration arms: equal-episode weighting, source multiplicities, the explicit-bias
readout, and a transform copied from outside the training sources (M3MAN-005; I3, I4, I8)."""
EQUAL_EPISODE_WEIGHTING: Final = "equal_episode"
"""The only episode-weighting rule: every episode carries the same total loss weight whatever its length."""
SUPPORTED_RECIPE_SCHEMAS: Final = (RECIPE_SCHEMA_VERSION, BINDING_SCHEMA_VERSION, WEIGHTED_SCHEMA_VERSION)
APPROVED_ADDITIONAL_REPEATS: Final[frozenset[int]] = frozenset({16, 32, 64})
"""Approved exact-copy counts (repetition plan section 4): 17, 33, or 65 episodes in total."""
REGULARIZATION_RULES: Final = ("base", "count_scaled", "count_divided")
"""How the solver's ridge parameter derives from the base parameter and an episode count (plan section 5.2)."""


def solver_alpha(base_alpha: float, rule: str, count: int) -> float:
    """The ridge parameter handed to the solver: ``base``, ``count * base`` (R-scaled), or ``base / count``.

    The ``count_divided`` rule is the S-effective control: one episode fitted at
    ``alpha_0 / K`` is the exact-arithmetic equivalent of ``K`` copies at
    ``alpha_0``. These are prescribed controls, never search draws, so no
    historical search bound applies to the result.
    """
    if rule not in REGULARIZATION_RULES:
        msg = f"regularization_rule must be one of {REGULARIZATION_RULES}, got {rule!r}"
        raise ValueError(msg)
    if not (math.isfinite(base_alpha) and base_alpha > 0):
        msg = f"base_alpha must be positive and finite, got {base_alpha!r}"
        raise ValueError(msg)
    if count < 1:
        msg = f"the episode count must be >= 1, got {count}"
        raise ValueError(msg)
    if rule == "count_scaled":
        return base_alpha * count
    if rule == "count_divided":
        return base_alpha / count
    return base_alpha


class RecipeMismatchError(RuntimeError):
    """A refit did not reproduce the recipe's recorded fit report within tolerance."""


@dataclass(frozen=True)
class DatasetSource:
    """One processed dataset the recipe was trained on."""

    artifact_id: str
    payload_sha256: str
    record: str
    """Repository-relative path of the Git-tracked record."""

    def __post_init__(self) -> None:
        """Identity fields have the canonical formats."""
        if not is_artifact_id(self.artifact_id) or not self.artifact_id.startswith("processed-"):
            msg = f"artifact_id must be a processed artifact ID, got {self.artifact_id!r}"
            raise ValueError(msg)
        if not is_hex(self.payload_sha256, SHA256_HEX_LENGTH):
            msg = f"payload_sha256 must be 64 lowercase hex characters, got {self.payload_sha256!r}"
            raise ValueError(msg)
        require_relative_posix(self.record, "record")


@dataclass(frozen=True)
class RclibIdentity:
    """The ``rclib`` revision the recipe's reservoir and readout semantics depend on."""

    version: str
    commit: str

    def __post_init__(self) -> None:
        """The commit is a full hash."""
        if not is_hex(self.commit, COMMIT_HEX_LENGTH):
            msg = f"rclib.commit must be a 40-hex commit, got {self.commit!r}"
            raise ValueError(msg)
        if not self.version.strip():
            msg = "rclib.version must not be empty"
            raise ValueError(msg)

    @classmethod
    def current(cls) -> RclibIdentity:
        """The pinned ``rclib`` submodule of this checkout."""
        (revision,) = [r for r in submodule_revisions() if r.name == "rclib"]
        return cls(submodule_version("rclib"), revision.recorded)


@dataclass(frozen=True)
class AugmentationTrainingSpec:
    """Deterministic training augmentation of one recipe (M3R-012; approved D1 values only).

    The synthetic episodes regenerate bitwise from the dataset, the scenario,
    and these values via :func:`arm_rc_ctrl.rc.augment.generate_augmentation`;
    the recipe therefore stays refittable from config plus dataset alone.
    """

    family: Literal["non_decaying", "contractive"]
    n_synthetic: int
    sigma_rad: float
    phi: float
    gamma: float
    seed_bank: int
    attempt_budget: int

    def __post_init__(self) -> None:
        """The family is one of the two matched arms and every value sits on the approved grids."""
        if self.family not in ("non_decaying", "contractive"):
            msg = f"augmentation.family must be 'non_decaying' or 'contractive', got {self.family!r}"
            raise ValueError(msg)
        self.config()  # rejects values outside the approved D1 grids

    def config(self) -> AugmentationConfig:
        """The generator configuration (validated against the approved grids)."""
        return AugmentationConfig(
            n_synthetic=self.n_synthetic,
            sigma_rad=self.sigma_rad,
            phi=self.phi,
            gamma=self.gamma,
            seed_bank=self.seed_bank,
            attempt_budget=self.attempt_budget,
        )


@dataclass(frozen=True)
class TrainingValidation:
    """The scenario file and limits the training episodes were validated against (schema 2; C2).

    Training keeps these limits even when an evaluation config relaxes its own
    abort limits: a recipe refuses to build episodes against a scenario whose
    limits differ from the ones recorded here.
    """

    scenario_file: str
    """Repository-relative path of the scenario the dataset is bound to."""
    scenario_sha256: str
    velocity_limit: tuple[float, ...]
    """Per-joint speed bound (rad/s) synthetic episodes were validated against."""
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    endpoint_radius: float

    def __post_init__(self) -> None:
        """Identity fields are well-formed and the limits are consistent."""
        require_relative_posix(self.scenario_file, "validation.scenario_file")
        if not is_hex(self.scenario_sha256, SHA256_HEX_LENGTH):
            msg = f"validation.scenario_sha256 must be 64 lowercase hex characters, got {self.scenario_sha256!r}"
            raise ValueError(msg)
        widths = {len(self.velocity_limit), len(self.joint_lower), len(self.joint_upper)}
        if len(widths) != 1 or not self.velocity_limit:
            msg = "validation limits must cover the same non-empty joint set"
            raise ValueError(msg)
        values = (*self.velocity_limit, *self.joint_lower, *self.joint_upper, self.endpoint_radius)
        if any(not math.isfinite(v) for v in values):
            msg = "validation limits must be finite"
            raise ValueError(msg)
        if any(v <= 0 for v in self.velocity_limit) or self.endpoint_radius <= 0:
            msg = "validation speed limits and endpoint radius must be positive"
            raise ValueError(msg)
        if any(lo >= hi for lo, hi in zip(self.joint_lower, self.joint_upper, strict=True)):
            msg = "validation joint limits need lower < upper per joint"
            raise ValueError(msg)

    @classmethod
    def from_scenario(
        cls, scenario: ScenarioConfig, scenario_file: Path, *, root: Path | None = None
    ) -> TrainingValidation:
        """Bind the scenario file's digest and the limits the generator validates against."""
        base = (repository_root() if root is None else root).resolve()
        resolved = scenario_file.resolve()
        if not resolved.is_relative_to(base):
            msg = f"scenario file {scenario_file} lies outside the repository root {base}"
            raise ValueError(msg)
        limits = joint_limits(scenario)
        return cls(
            scenario_file=resolved.relative_to(base).as_posix(),
            scenario_sha256=sha256_file(resolved),
            velocity_limit=tuple(float(v) for v in scenario.limits.velocity),
            joint_lower=tuple(float(v) for v in limits.lower),
            joint_upper=tuple(float(v) for v in limits.upper),
            endpoint_radius=float(scenario.limits.endpoint_radius),
        )

    def check(self, scenario: ScenarioConfig) -> None:
        """Fail unless ``scenario`` carries exactly the recorded training-validation limits."""
        limits = joint_limits(scenario)
        current = (
            tuple(float(v) for v in scenario.limits.velocity),
            tuple(float(v) for v in limits.lower),
            tuple(float(v) for v in limits.upper),
            float(scenario.limits.endpoint_radius),
        )
        recorded = (self.velocity_limit, self.joint_lower, self.joint_upper, self.endpoint_radius)
        if current != recorded:
            msg = (
                f"the scenario's limits {current} differ from the training-validation limits {recorded} bound by "
                "the recipe; training keeps its recorded limits (C2), so build episodes against the bound scenario"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class TrainingSpec:
    """How episodes are built; absolute next-position output with a versioned washout policy."""

    input_channels: tuple[str, ...] = INPUT_CHANNELS
    target: str = "next_q"
    """``next_q`` (absolute next joint position) or ``increment_q`` (the residual arm's
    ``q_{k+1} - q_k``; recovery plan section 6.1, requires the ``warmup_hold`` washout)."""
    washout: str = "prime_phase"
    """``prime_phase`` (M3: washout rows lie in the prime interval) or ``warmup_hold`` (M3R-006:
    the washout repeats the episode's encoded ``[q_0, 0]`` for the configured warm-up)."""
    warmup_s: float | None = None
    """Warm-up duration (approved D2 value) for ``warmup_hold``; must be ``None`` for ``prime_phase``."""
    augmentation: AugmentationTrainingSpec | None = None
    """Deterministic synthetic-episode augmentation; requires the ``warmup_hold`` washout."""
    additional_repeats: int | None = None
    """Exact copies of the single source episode added to training (approved 16, 32, or 64); requires the
    ``warmup_hold`` washout and excludes augmentation (repetition plan section 4)."""
    base_alpha: float | None = None
    """The source ridge parameter ``alpha_0`` the readout's solver parameter derives from (plan section 5.2)."""
    regularization_rule: str | None = None
    """How ``esn.readout.alpha`` derives from ``base_alpha``: ``base``, ``count_scaled``, or ``count_divided``."""
    regularization_count: int | None = None
    """The episode count the rule refers to; defaults to the recipe's own episode count (S-effective fits one
    episode at ``alpha_0 / K`` and therefore states ``K`` explicitly)."""
    episode_weighting: str | None = None
    """Schema 3 only: ``equal_episode``, the rule giving every episode the same total loss weight (I4)."""
    weight_reference_rows: int | None = None
    """Schema 3 only: the row-count reference ``R`` of the per-row weight ``R / L_i`` (historically 400)."""
    source_counts: tuple[int, ...] | None = None
    """Schema 3 only: how many episodes each dataset contributes, aligned 1:1 with the recipe's datasets.
    A count above one is that many exact copies of the source episode; payloads are never duplicated."""

    def __post_init__(self) -> None:
        """Only the implemented representations are accepted."""
        _check_repetition(self)
        _check_weighting(self)
        if self.input_channels != INPUT_CHANNELS or self.target not in ("next_q", "increment_q"):
            msg = (
                f"unsupported training spec {self!r}; supported: input_channels {INPUT_CHANNELS}, "
                "target 'next_q' or 'increment_q'"
            )
            raise ValueError(msg)
        if self.washout == "prime_phase":
            if self.target != "next_q":
                msg = "the 'increment_q' target requires the 'warmup_hold' washout (recovery plan section 6.1)"
                raise ValueError(msg)
            if self.warmup_s is not None:
                msg = "warmup_s is only meaningful for the 'warmup_hold' washout"
                raise ValueError(msg)
            if self.augmentation is not None:
                msg = "training augmentation requires the 'warmup_hold' washout"
                raise ValueError(msg)
        elif self.washout == "warmup_hold":
            if self.warmup_s is None:
                msg = "the 'warmup_hold' washout requires warmup_s (an approved D2 duration)"
                raise ValueError(msg)
            WarmupConfig(self.warmup_s)  # rejects durations outside the approved set
        else:
            msg = f"unsupported washout {self.washout!r}; supported: 'prime_phase', 'warmup_hold'"
            raise ValueError(msg)

    @property
    def episode_count(self) -> int:
        """Episodes the training stacks: the source multiplicities, or one source plus its copies or synthetics."""
        if self.source_counts is not None:
            return sum(self.source_counts)
        if self.additional_repeats is not None:
            return 1 + self.additional_repeats
        if self.augmentation is not None:
            return 1 + self.augmentation.n_synthetic
        return 1

    @property
    def formulation(self) -> str:
        """``absolute`` (next-position readout) or ``residual`` (next-step increment readout)."""
        return "residual" if self.target == "increment_q" else "absolute"

    @property
    def uses_binding_features(self) -> bool:
        """Whether the spec uses constructions only schema 2 recipes may carry."""
        return self.additional_repeats is not None or self.base_alpha is not None

    @property
    def uses_weighted_features(self) -> bool:
        """Whether the spec uses constructions only schema 3 recipes may carry."""
        return (
            self.episode_weighting is not None
            or self.weight_reference_rows is not None
            or self.source_counts is not None
        )


def _check_weighting(spec: TrainingSpec) -> None:
    weighting, reference, counts = spec.episode_weighting, spec.weight_reference_rows, spec.source_counts
    declared = [field for field in (weighting, reference, counts) if field is not None]
    if not declared:
        return
    if len(declared) != 3:  # noqa: PLR2004 - the three weighting fields are one construction
        msg = "episode_weighting, weight_reference_rows, and source_counts are recorded together"
        raise ValueError(msg)
    if weighting != EQUAL_EPISODE_WEIGHTING:
        msg = f"episode_weighting must be {EQUAL_EPISODE_WEIGHTING!r}, got {weighting!r}"
        raise ValueError(msg)
    if reference is None or reference < 1:
        msg = f"weight_reference_rows must be >= 1, got {reference!r}"
        raise ValueError(msg)
    if not counts or any(count < 1 for count in counts):
        msg = f"source_counts must give every dataset at least one episode, got {counts!r}"
        raise ValueError(msg)
    if spec.washout != "warmup_hold":
        msg = "equal-episode weighting requires the 'warmup_hold' washout (manual plan section 4)"
        raise ValueError(msg)
    if spec.additional_repeats is not None or spec.augmentation is not None:
        msg = (
            "equal-episode weighting expresses copies as source_counts; additional_repeats and augmentation are "
            "the separate repetition and varied-data constructions"
        )
        raise ValueError(msg)


def _check_repetition(spec: TrainingSpec) -> None:
    repeats = spec.additional_repeats
    if repeats is not None:
        if repeats not in APPROVED_ADDITIONAL_REPEATS:
            msg = (
                f"additional_repeats must be one of the approved counts {sorted(APPROVED_ADDITIONAL_REPEATS)}, "
                f"got {repeats}"
            )
            raise ValueError(msg)
        if spec.washout != "warmup_hold":
            msg = "exact repetition requires the 'warmup_hold' washout (repetition plan section 4)"
            raise ValueError(msg)
        if spec.augmentation is not None:
            msg = (
                "exact repetition and augmentation are mutually exclusive: the varied-data arms are separate recipes "
                "(and sigma = 0 is not an unlabelled substitute for repetition)"
            )
            raise ValueError(msg)
    if (spec.base_alpha is None) != (spec.regularization_rule is None):
        msg = "base_alpha and regularization_rule must be recorded together"
        raise ValueError(msg)
    if spec.regularization_rule is not None and spec.base_alpha is not None:
        solver_alpha(spec.base_alpha, spec.regularization_rule, spec.regularization_count or spec.episode_count)
    elif spec.regularization_count is not None:
        msg = "regularization_count needs base_alpha and regularization_rule"
        raise ValueError(msg)


@dataclass(frozen=True)
class FitTolerance:
    """Declared tolerance for reproducing the recorded fit report."""

    error_abs: float = 1e-9
    """Absolute tolerance (rad) on every RMSE and error figure of the fit report."""

    def __post_init__(self) -> None:
        """The tolerance is non-negative and finite."""
        if not (math.isfinite(self.error_abs) and self.error_abs >= 0):
            msg = f"error_abs must be finite and non-negative, got {self.error_abs!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ModelRecipe:
    """Everything needed to rebuild and refit the model, plus the fit it produced."""

    name: str
    esn: EsnConfig
    dof: int
    task_code_dim: int
    datasets: tuple[DatasetSource, ...]
    """Training episodes, in training order."""
    preprocessing: Preprocessing
    transform: InputTransform
    """Input transform derived from the datasets' recorded statistics under the recipe's policy."""
    training: TrainingSpec
    rclib: RclibIdentity
    fit: FitReport
    tolerance: FitTolerance = field(default_factory=FitTolerance)
    schema_version: int = field(default=RECIPE_SCHEMA_VERSION)
    validation: TrainingValidation | None = None
    """Schema 2 and 3: the scenario file and limits the training episodes were validated against (C2)."""
    transform_source: DatasetSource | None = None
    """Schema 3 only: the dataset the frozen input transform was copied from, digest-bound and deliberately outside
    ``datasets`` (I8), so a singleton recipe receives no statistics from the other takes."""

    def __post_init__(self) -> None:
        """Consistency between datasets, fit report, normalization, widths, schema, and the ridge rule."""
        _check_schema(self)
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if self.dof < 1 or self.task_code_dim < 0:
            msg = f"dof must be >= 1 and task_code_dim >= 0, got {self.dof} and {self.task_code_dim}"
            raise ValueError(msg)
        ids = tuple(d.artifact_id for d in self.datasets)
        if not ids or len(set(ids)) != len(ids):
            msg = f"datasets must be a non-empty list of distinct artifacts, got {ids}"
            raise ValueError(msg)
        expected_labels = expected_episode_labels(self.training, ids)
        if self.fit.episodes != expected_labels:
            msg = f"fit.episodes {self.fit.episodes} must equal the training episode labels {expected_labels}"
            raise ValueError(msg)
        if len(self.fit.rmse_per_joint) != self.dof:
            msg = f"fit.rmse_per_joint has {len(self.fit.rmse_per_joint)} joints, expected {self.dof}"
            raise ValueError(msg)
        if self.schema_version != WEIGHTED_SCHEMA_VERSION:
            # Schema 3 copies its transform from the bound transform_source instead (checked in _check_schema).
            unknown = sorted(set(self.transform.derived_from) - set(ids))
            if unknown:
                msg = f"transform.derived_from names datasets outside the recipe: {unknown}"
                raise ValueError(msg)
        self.encoder()  # validates the transform against the widths

    @property
    def formulation(self) -> str:
        """``absolute`` or ``residual`` (the training target's output formulation)."""
        return self.training.formulation

    @property
    def output(self) -> str:
        """The generator output mode the recipe requires: ``absolute`` or ``increment``."""
        return "increment" if self.training.target == "increment_q" else "absolute"

    @property
    def solver_alpha(self) -> float:
        """The ridge parameter handed to the solver (``esn.readout.alpha``)."""
        return self.esn.readout.alpha

    @property
    def input_dim(self) -> int:
        """Width of the ESN input."""
        return len(self.training.input_channels) * self.dof + self.task_code_dim

    @property
    def output_dim(self) -> int:
        """Width of the ESN output (one next position per joint)."""
        return self.dof

    def encoder(self) -> InputEncoder:
        """The input encoder the episodes and the runtime generator share."""
        return InputEncoder(self.transform, self.dof, self.task_code_dim)

    def check_dataset_record(self, source: DatasetSource, record: DatasetRecord) -> None:
        """Fail unless ``record`` is the dataset the recipe names and was processed the way the recipe expects.

        Schema 1 and 2 recipes derive their input transform from a training
        dataset and therefore require its recorded normalization. Schema 3
        copies the transform from its ``transform_source`` instead, so a
        manual take, which records no normalization by design (I8), is
        accepted as training data.
        """
        artifact = record.artifact
        if artifact.artifact_id != source.artifact_id or artifact.payload.sha256 != source.payload_sha256:
            msg = (
                f"record {source.record} describes {artifact.artifact_id} ({artifact.payload.sha256[:12]}), "
                f"not {source.artifact_id} ({source.payload_sha256[:12]})"
            )
            raise ValueError(msg)
        if record.dof != self.dof or record.task_code_dim != self.task_code_dim:
            msg = (
                f"dataset {source.artifact_id} has dof {record.dof} and task_code_dim {record.task_code_dim}; "
                f"the recipe expects {self.dof} and {self.task_code_dim}"
            )
            raise ValueError(msg)
        if record.preprocessing != self.preprocessing:
            msg = f"dataset {source.artifact_id} was preprocessed differently from the recipe's preprocessing"
            raise ValueError(msg)
        if record.normalization is None and self.schema_version != WEIGHTED_SCHEMA_VERSION:
            msg = f"dataset {source.artifact_id} records no normalization statistics"
            raise ValueError(msg)

    def check_transform_record(self, record: DatasetRecord) -> Normalization:
        """The statistics of the digest-bound transform source (schema 3); fail unless ``record`` is that dataset."""
        source = self.transform_source
        if source is None:
            msg = f"recipe {self.name!r} names no transform source; its transform derives from its training datasets"
            raise ValueError(msg)
        artifact = record.artifact
        if artifact.artifact_id != source.artifact_id or artifact.payload.sha256 != source.payload_sha256:
            msg = (
                f"record {source.record} describes {artifact.artifact_id} ({artifact.payload.sha256[:12]}), "
                f"not the transform source {source.artifact_id} ({source.payload_sha256[:12]})"
            )
            raise ValueError(msg)
        if record.normalization is None:
            msg = f"the transform source {source.artifact_id} records no normalization statistics"
            raise ValueError(msg)
        return record.normalization

    def check_transform_source(self, normalizations: Mapping[str, Normalization]) -> None:
        """Fail unless the transform re-derives exactly from the recorded statistics it claims to come from.

        For schema 3 those statistics belong to ``transform_source``, which is
        not a training dataset; the re-derivation is bitwise either way.
        """
        if len(self.transform.derived_from) != 1:
            msg = f"the transform must derive from exactly one dataset, got {self.transform.derived_from}"
            raise ValueError(msg)
        (source_id,) = self.transform.derived_from
        normalization = normalizations.get(source_id)
        if normalization is None:
            msg = f"no normalization statistics available for {source_id}, which the transform derives from"
            raise ValueError(msg)
        derived = InputTransform.derive(
            self.transform.policy, normalization, fixed_scales=self.transform.fixed_scales or None
        )
        if derived != self.transform:
            msg = f"the recipe's transform does not derive from the recorded normalization of {source_id}"
            raise ValueError(msg)

    def build_model(self) -> EsnModel:
        """Reconstruct the (unfitted) model from the hyperparameters and seeds."""
        return EsnModel(self.esn, input_dim=self.input_dim, output_dim=self.output_dim)

    def episodes(self, samples: Mapping[str, SampleSet], *, scenario: ScenarioConfig | None = None) -> list[Episode]:
        """Build the training episodes from the referenced datasets, in training order.

        Augmented recipes regenerate their synthetic episodes deterministically
        and need the ``scenario`` (envelope and validity limits).
        """
        missing = [d.artifact_id for d in self.datasets if d.artifact_id not in samples]
        if missing:
            msg = f"samples are missing for datasets {missing}"
            raise ValueError(msg)
        if scenario is not None and self.validation is not None:
            self.validation.check(scenario)
        return _build_episodes(
            self.training, self.datasets, samples, self.encoder(), self.preprocessing, scenario=scenario
        )

    def require_rclib(self, installed: RclibIdentity | None = None) -> None:
        """Fail unless the installed ``rclib`` (default: this checkout's pin) is the one the recipe was made with."""
        current = RclibIdentity.current() if installed is None else installed
        if current != self.rclib:
            msg = (
                f"recipe {self.name!r} was made with rclib {self.rclib.version} ({self.rclib.commit[:12]}) but "
                f"{current.version} ({current.commit[:12]}) is installed; reservoir and readout semantics may differ"
            )
            raise RecipeMismatchError(msg)

    def refit(
        self,
        samples: Mapping[str, SampleSet],
        *,
        installed: RclibIdentity | None = None,
        scenario: ScenarioConfig | None = None,
    ) -> tuple[EsnModel, FitReport]:
        """Rebuild and refit the model; fail unless the rclib pin matches and the fit report is reproduced."""
        self.require_rclib(installed)
        model = self.build_model()
        report = train_readout(
            model,
            self.episodes(samples, scenario=scenario),
            weight_reference_rows=self.training.weight_reference_rows,
        )
        mismatches = _compare_fit(report, self.fit, self.tolerance)
        if mismatches:
            msg = f"refit of recipe {self.name!r} does not reproduce its fit report: " + "; ".join(mismatches)
            raise RecipeMismatchError(msg)
        return model, report


def _check_schema(recipe: ModelRecipe) -> None:
    if recipe.schema_version not in SUPPORTED_RECIPE_SCHEMAS:
        msg = f"unsupported recipe schema version {recipe.schema_version}"
        raise ValueError(msg)
    weighted = recipe.schema_version == WEIGHTED_SCHEMA_VERSION
    bound = recipe.schema_version in (BINDING_SCHEMA_VERSION, WEIGHTED_SCHEMA_VERSION)
    if bound and recipe.validation is None:
        msg = f"schema {recipe.schema_version} recipes bind their training validation (C2); validation is missing"
        raise ValueError(msg)
    if not bound and recipe.validation is not None:
        msg = f"schema {RECIPE_SCHEMA_VERSION} recipes carry no training validation; write a schema 2 recipe"
        raise ValueError(msg)
    if not bound and recipe.training.uses_binding_features:
        msg = f"exact repetition and ridge rules need a schema {BINDING_SCHEMA_VERSION} recipe"
        raise ValueError(msg)
    if not weighted and (recipe.training.uses_weighted_features or recipe.transform_source is not None):
        msg = (
            "equal-episode weighting, source multiplicities, and a transform source outside the training data "
            f"need a schema {WEIGHTED_SCHEMA_VERSION} recipe"
        )
        raise ValueError(msg)
    if weighted:
        _check_weighted_recipe(recipe)
    spec = recipe.training
    if spec.base_alpha is not None and spec.regularization_rule is not None:
        count = spec.regularization_count or spec.episode_count
        expected = solver_alpha(spec.base_alpha, spec.regularization_rule, count)
        if recipe.esn.readout.alpha != expected:
            msg = (
                f"esn.readout.alpha {recipe.esn.readout.alpha!r} must equal {expected!r}: "
                f"{spec.regularization_rule} of base_alpha {spec.base_alpha!r} over {count} episodes"
            )
            raise ValueError(msg)


def _check_weighted_recipe(recipe: ModelRecipe) -> None:
    """The schema 3 construction: equal-episode weighting, aligned multiplicities, explicit bias, copied transform."""
    spec = recipe.training
    counts = spec.source_counts
    if spec.episode_weighting != EQUAL_EPISODE_WEIGHTING or spec.weight_reference_rows is None or counts is None:
        msg = (
            f"schema {WEIGHTED_SCHEMA_VERSION} recipes record {EQUAL_EPISODE_WEIGHTING!r} weighting with its "
            "weight_reference_rows and one source count per dataset"
        )
        raise ValueError(msg)
    if len(counts) != len(recipe.datasets):
        msg = f"source_counts {counts} must give one multiplicity per dataset, got {len(recipe.datasets)} datasets"
        raise ValueError(msg)
    override = spec.regularization_count
    if override is not None and override != spec.episode_count:
        msg = (
            f"regularization_count {override} must equal the {spec.episode_count} episodes a schema "
            f"{WEIGHTED_SCHEMA_VERSION} recipe stacks: the weighted objective scales the solver's parameter with "
            "the episodes actually fitted, so an override would let a copy control train at another "
            "regularization than the count it matches"
        )
        raise ValueError(msg)
    if spec.base_alpha is None or spec.regularization_rule != "count_scaled":
        msg = (
            f"schema {WEIGHTED_SCHEMA_VERSION} recipes derive esn.readout.alpha as 'count_scaled' of base_alpha "
            "over the episode count K (the weighted objective's regularization scale)"
        )
        raise ValueError(msg)
    source = recipe.transform_source
    if source is None:
        msg = f"schema {WEIGHTED_SCHEMA_VERSION} recipes bind the dataset their input transform was copied from (I8)"
        raise ValueError(msg)
    if source.artifact_id in tuple(d.artifact_id for d in recipe.datasets):
        msg = (
            f"transform_source {source.artifact_id} is one of the training datasets; the frozen transform is copied "
            "from a dataset outside them so a singleton receives no statistics from the other takes (I8)"
        )
        raise ValueError(msg)
    if recipe.transform.derived_from != (source.artifact_id,):
        msg = (
            f"transform.derived_from {recipe.transform.derived_from} must name exactly the transform source "
            f"{source.artifact_id}"
        )
        raise ValueError(msg)
    readout = recipe.esn.readout
    if not readout.explicit_bias or readout.include_bias:
        msg = (
            f"schema {WEIGHTED_SCHEMA_VERSION} recipes fit the explicit-bias readout: set esn.readout.explicit_bias "
            "and leave esn.readout.include_bias false, so every weighted row scales with its bias entry (I3)"
        )
        raise ValueError(msg)
    if recipe.fit.episode_loss_rows is None or recipe.fit.episode_weights is None:
        msg = (
            f"schema {WEIGHTED_SCHEMA_VERSION} recipes record the per-episode loss rows and weights their fit "
            "used, so a refit verifies the weighting itself and not only the errors it produced"
        )
        raise ValueError(msg)


def _compare_fit(actual: FitReport, expected: FitReport, tolerance: FitTolerance) -> list[str]:
    mismatches = [
        f"{name} {getattr(actual, name)!r} != {getattr(expected, name)!r}"
        for name in ("episodes", "loss_rows", "washout_rows", "episode_loss_rows", "episode_weights")
        if getattr(actual, name) != getattr(expected, name)
    ]
    pairs = [
        ("rmse", actual.rmse, expected.rmse),
        ("constant_rmse", actual.constant_rmse, expected.constant_rmse),
        ("max_abs_error", actual.max_abs_error, expected.max_abs_error),
        *(
            (f"rmse_per_joint[{i}]", a, e)
            for i, (a, e) in enumerate(zip(actual.rmse_per_joint, expected.rmse_per_joint, strict=False))
        ),
    ]
    if len(actual.rmse_per_joint) != len(expected.rmse_per_joint):
        mismatches.append("rmse_per_joint lengths differ")
    mismatches += [
        f"{name} {a!r} differs from {e!r} beyond {tolerance.error_abs:g}"
        for name, a, e in pairs
        if not math.isclose(a, e, rel_tol=0.0, abs_tol=tolerance.error_abs)
    ]
    return mismatches


def expected_episode_labels(spec: TrainingSpec, ids: tuple[str, ...]) -> tuple[str, ...]:
    """The episode labels a recipe with ``spec`` trains on, in training order."""
    if spec.source_counts is not None:
        if len(ids) != len(spec.source_counts):
            msg = f"source_counts {spec.source_counts} must give one multiplicity per dataset, got {list(ids)}"
            raise ValueError(msg)
        labels: list[str] = []
        for artifact, count in zip(ids, spec.source_counts, strict=True):
            labels.append(artifact)
            labels.extend(f"{artifact}#copy-{index:03d}" for index in range(1, count))
        return tuple(labels)
    if spec.additional_repeats is not None:
        if len(ids) != 1:
            msg = f"exact repetition uses exactly one dataset, got {list(ids)}"
            raise ValueError(msg)
        return (ids[0], *(f"{ids[0]}#repeat-{i:03d}" for i in range(1, spec.additional_repeats + 1)))
    if spec.augmentation is None:
        return tuple(ids)
    if len(ids) != 1:
        msg = f"augmented training uses exactly one dataset, got {list(ids)}"
        raise ValueError(msg)
    family = spec.augmentation.family
    return (ids[0], *(f"{ids[0]}#{family}-{i:03d}" for i in range(1, spec.augmentation.n_synthetic + 1)))


DERIVATIVE_METHODS: Final = {"central-difference": "central", "cubic-spline": "spline"}
"""Preprocessing derivative labels and the derivative schemes they name."""


def derivative_config(label: str) -> DerivativeConfig:
    """The derivative scheme a preprocessing label names (shared by recipes, probes, and augmentation banks)."""
    if label not in DERIVATIVE_METHODS:
        msg = f"unknown derivative policy label {label!r}; expected one of {sorted(DERIVATIVE_METHODS)}"
        raise ValueError(msg)
    return DerivativeConfig(method=cast('Literal["central", "spline"]', DERIVATIVE_METHODS[label]))


def _derivatives(preprocessing: Preprocessing) -> DerivativeConfig:
    return derivative_config(preprocessing.derivative_method)


def _weighted_episodes(
    spec: TrainingSpec,
    sources: Sequence[DatasetSource],
    samples: Mapping[str, SampleSet],
    encoder: InputEncoder,
    *,
    warmup: WarmupConfig,
    period_s: float,
) -> list[Episode]:
    """One warm-up-prefixed episode per dataset plus its literal copies, in ``source_counts`` order.

    The complete recording is teacher-forced: every recorded sample pairs with
    its successor and enters the loss, including the pre-roll the manual
    protocol keeps (I4, I10), so the descriptive phase annotation never decides
    which rows train. A copy re-wraps the parent's own arrays under a copy
    label: nothing is harvested twice and no payload is duplicated.
    """
    counts = cast("tuple[int, ...]", spec.source_counts)
    episodes: list[Episode] = []
    for source, count in zip(sources, counts, strict=True):
        sample_set = samples[source.artifact_id]
        original = build_task_episode_arrays(
            sample_set.t,
            sample_set.q,
            sample_set.dq,
            sample_set.task_code,
            encoder,
            source=source.artifact_id,
            warmup=warmup,
            period_s=period_s,
            target=spec.target,
        )
        episodes.append(original)
        episodes.extend(
            Episode(
                source=f"{source.artifact_id}#copy-{index:03d}",
                t=original.t,
                inputs=original.inputs,
                targets=original.targets,
                loss_rows=original.loss_rows,
            )
            for index in range(1, count)
        )
    return episodes


def _build_episodes(
    spec: TrainingSpec,
    sources: Sequence[DatasetSource],
    samples: Mapping[str, SampleSet],
    encoder: InputEncoder,
    preprocessing: Preprocessing,
    *,
    scenario: ScenarioConfig | None = None,
) -> list[Episode]:
    """Build the training episodes under the spec's washout policy (shared by training and refit)."""
    if spec.washout != "warmup_hold":
        return [build_episode(samples[s.artifact_id], encoder, source=s.artifact_id) for s in sources]
    warmup = WarmupConfig(cast("float", spec.warmup_s))
    period = preprocessing.resample_period_s
    if spec.source_counts is not None:
        return _weighted_episodes(spec, sources, samples, encoder, warmup=warmup, period_s=period)
    episodes = [
        build_task_episode(
            samples[s.artifact_id],
            encoder,
            source=s.artifact_id,
            warmup=warmup,
            period_s=period,
            target=spec.target,
        )
        for s in sources
    ]
    if spec.additional_repeats is not None:
        (original,) = episodes  # expected_episode_labels enforces exactly one dataset for repetition
        labels = expected_episode_labels(spec, tuple(s.artifact_id for s in sources))[1:]
        # Literal copies: every copy is harvested from a reset reservoir and stacked into the one ridge fit.
        copies = [
            Episode(
                source=label,
                t=original.t,
                inputs=original.inputs,
                targets=original.targets,
                loss_rows=original.loss_rows,
            )
            for label in labels
        ]
        return [original, *copies]
    augmentation = spec.augmentation
    if augmentation is None:
        return episodes
    if scenario is None:
        msg = (
            "augmented training needs the scenario (endpoint envelope and validity limits); "
            "pass scenario=... to episodes()/refit()/create_recipe()"
        )
        raise ValueError(msg)
    (source,) = sources  # expected_episode_labels enforces exactly one dataset for augmented recipes
    sample_set = samples[source.artifact_id]
    task = task_intervals_from_phases(sample_set.t, sample_set.phase)
    result = generate_augmentation(
        sample_set.t, sample_set.q, task, scenario, augmentation.config(), derivatives=_derivatives(preprocessing)
    )
    for episode in result.episodes:
        arrays: EpisodeArrays = getattr(episode, augmentation.family)
        episodes.append(
            build_task_episode_arrays(
                sample_set.t,
                arrays.q,
                arrays.dq,
                sample_set.task_code,
                encoder,
                source=f"{source.artifact_id}#{augmentation.family}-{episode.episode:03d}",
                warmup=warmup,
                period_s=period,
                target=spec.target,
            )
        )
    return episodes


def create_recipe(
    name: str,
    esn: EsnConfig,
    *,
    sources: Sequence[DatasetSource],
    samples: Mapping[str, SampleSet],
    dof: int,
    task_code_dim: int,
    preprocessing: Preprocessing,
    transform: InputTransform,
    training: TrainingSpec | None = None,
    rclib: RclibIdentity | None = None,
    tolerance: FitTolerance | None = None,
    scenario: ScenarioConfig | None = None,
    validation: TrainingValidation | None = None,
    transform_source: DatasetSource | None = None,
) -> tuple[ModelRecipe, EsnModel]:
    """Train the model on ``sources`` and return the recipe that reproduces it, plus the fitted model.

    Passing ``validation`` writes a schema 2 recipe bound to the scenario file
    and limits it names; the given ``scenario`` must carry those limits. An
    equal-episode training spec or a ``transform_source`` writes a schema 3
    recipe instead: the fit is weighted with the spec's reference row count and
    the transform is bound to the dataset it was copied from.
    """
    spec = TrainingSpec() if training is None else training
    if validation is not None and scenario is not None:
        validation.check(scenario)
    encoder = InputEncoder(transform, dof, task_code_dim)
    missing = [s.artifact_id for s in sources if s.artifact_id not in samples]
    if not sources or missing:
        msg = f"sources must be non-empty and every dataset needs samples; missing {missing}"
        raise ValueError(msg)
    episodes = _build_episodes(spec, sources, samples, encoder, preprocessing, scenario=scenario)
    model = EsnModel(esn, input_dim=encoder.input_dim, output_dim=dof)
    report = train_readout(model, episodes, weight_reference_rows=spec.weight_reference_rows)
    schema_version = RECIPE_SCHEMA_VERSION
    if spec.uses_weighted_features or transform_source is not None:
        schema_version = WEIGHTED_SCHEMA_VERSION
    elif validation is not None:
        schema_version = BINDING_SCHEMA_VERSION
    recipe = ModelRecipe(
        name=name,
        esn=esn,
        dof=dof,
        task_code_dim=task_code_dim,
        datasets=tuple(sources),
        preprocessing=preprocessing,
        transform=transform,
        training=spec,
        rclib=RclibIdentity.current() if rclib is None else rclib,
        fit=report,
        tolerance=FitTolerance() if tolerance is None else tolerance,
        schema_version=schema_version,
        validation=validation,
        transform_source=transform_source,
    )
    return recipe, model


def write_recipe(path: Path, recipe: ModelRecipe) -> None:
    """Write the recipe as TOML; an existing file is never overwritten."""
    if path.exists():
        msg = f"{path} already exists; recipes are immutable (create a new one instead)"
        raise FileExistsError(msg)
    header = (
        "# Deterministic model recipe (docs/PLAN.md section 8): rebuild and refit, never unpickle.\n"
        "# Written by arm_rc_ctrl.rc.recipe; do not edit.\n"
    )
    path.write_text(header + records_to_toml(recipe), encoding="utf-8")


def load_recipe(path: Path) -> ModelRecipe:
    """Load and validate a recipe."""
    return load_config(path, ModelRecipe)
