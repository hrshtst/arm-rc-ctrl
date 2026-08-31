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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import tomli_w

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.data.records import (
    Normalization,
    Preprocessing,
    ProcessedDatasetRecord,
    is_artifact_id,
    require_relative_posix,
)
from arm_rc_ctrl.dependencies import submodule_revisions, submodule_version
from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel
from arm_rc_ctrl.rc.teacher_forcing import INPUT_CHANNELS, Episode, InputEncoder, InputTransform, build_episode
from arm_rc_ctrl.rc.training import FitReport, train_readout
from arm_rc_ctrl.validation import COMMIT_HEX_LENGTH, SHA256_HEX_LENGTH, is_hex

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from arm_rc_ctrl.data.samples import SampleSet

__all__ = [
    "RECIPE_SCHEMA_VERSION",
    "DatasetSource",
    "FitTolerance",
    "ModelRecipe",
    "RclibIdentity",
    "RecipeMismatchError",
    "TrainingSpec",
    "create_recipe",
    "load_recipe",
    "write_recipe",
]

RECIPE_SCHEMA_VERSION: Final = 1


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
class TrainingSpec:
    """How episodes are built; task 1-a supports exactly one representation."""

    input_channels: tuple[str, ...] = INPUT_CHANNELS
    target: str = "next_q"
    """Absolute next demonstrated joint position (the only supported output representation)."""
    washout: str = "prime_phase"
    """Washout rows are those whose input sample lies in the prime interval."""

    def __post_init__(self) -> None:
        """Only the implemented representation is accepted."""
        if self.input_channels != INPUT_CHANNELS or self.target != "next_q" or self.washout != "prime_phase":
            msg = (
                f"unsupported training spec {self!r}; supported: input_channels {INPUT_CHANNELS}, "
                "target 'next_q', washout 'prime_phase'"
            )
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

    def __post_init__(self) -> None:
        """Consistency between datasets, fit report, normalization, and widths."""
        if self.schema_version != RECIPE_SCHEMA_VERSION:
            msg = f"unsupported recipe schema version {self.schema_version}"
            raise ValueError(msg)
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
        if self.fit.episodes != ids:
            msg = f"fit.episodes {self.fit.episodes} must equal the dataset order {ids}"
            raise ValueError(msg)
        if len(self.fit.rmse_per_joint) != self.dof:
            msg = f"fit.rmse_per_joint has {len(self.fit.rmse_per_joint)} joints, expected {self.dof}"
            raise ValueError(msg)
        unknown = sorted(set(self.transform.derived_from) - set(ids))
        if unknown:
            msg = f"transform.derived_from names datasets outside the recipe: {unknown}"
            raise ValueError(msg)
        self.encoder()  # validates the transform against the widths

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

    def check_dataset_record(self, source: DatasetSource, record: ProcessedDatasetRecord) -> None:
        """Fail unless ``record`` is the dataset the recipe names and was processed the way the recipe expects."""
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
        if record.normalization is None:
            msg = f"dataset {source.artifact_id} records no normalization statistics"
            raise ValueError(msg)

    def check_transform_source(self, normalizations: Mapping[str, Normalization]) -> None:
        """Fail unless the transform re-derives exactly from the recorded statistics it claims to come from."""
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

    def episodes(self, samples: Mapping[str, SampleSet]) -> list[Episode]:
        """Build the training episodes from the referenced datasets, in training order."""
        missing = [d.artifact_id for d in self.datasets if d.artifact_id not in samples]
        if missing:
            msg = f"samples are missing for datasets {missing}"
            raise ValueError(msg)
        encoder = self.encoder()
        return [build_episode(samples[d.artifact_id], encoder, source=d.artifact_id) for d in self.datasets]

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
        self, samples: Mapping[str, SampleSet], *, installed: RclibIdentity | None = None
    ) -> tuple[EsnModel, FitReport]:
        """Rebuild and refit the model; fail unless the rclib pin matches and the fit report is reproduced."""
        self.require_rclib(installed)
        model = self.build_model()
        report = train_readout(model, self.episodes(samples))
        mismatches = _compare_fit(report, self.fit, self.tolerance)
        if mismatches:
            msg = f"refit of recipe {self.name!r} does not reproduce its fit report: " + "; ".join(mismatches)
            raise RecipeMismatchError(msg)
        return model, report


def _compare_fit(actual: FitReport, expected: FitReport, tolerance: FitTolerance) -> list[str]:
    mismatches = [
        f"{name} {getattr(actual, name)!r} != {getattr(expected, name)!r}"
        for name in ("episodes", "loss_rows", "washout_rows")
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
) -> tuple[ModelRecipe, EsnModel]:
    """Train the model on ``sources`` and return the recipe that reproduces it, plus the fitted model."""
    spec = TrainingSpec() if training is None else training
    encoder = InputEncoder(transform, dof, task_code_dim)
    missing = [s.artifact_id for s in sources if s.artifact_id not in samples]
    if not sources or missing:
        msg = f"sources must be non-empty and every dataset needs samples; missing {missing}"
        raise ValueError(msg)
    episodes = [build_episode(samples[s.artifact_id], encoder, source=s.artifact_id) for s in sources]
    model = EsnModel(esn, input_dim=encoder.input_dim, output_dim=dof)
    report = train_readout(model, episodes)
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
    path.write_text(header + tomli_w.dumps(to_mapping(recipe)), encoding="utf-8")


def load_recipe(path: Path) -> ModelRecipe:
    """Load and validate a recipe."""
    return load_config(path, ModelRecipe)
