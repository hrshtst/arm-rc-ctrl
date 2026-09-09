# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Environment-bound fit cache of the repeated-demonstration pilot (M3REP-003; plan sections 6 and 8, C6, C10).

Every fit of the pilot is one panel entry combined with one arm. Its cache key
is :func:`~arm_rc_ctrl.experiments.repetition_recipes.fit_identity`, which
binds the entry, the arm, both ridge parameters, the dataset, the input
transform, the training validation, the ``rclib`` commit, and the execution
environment identity; a fit produced under another environment or library
build is never served. Each cached fit is a directory in the external store::

    armrc://models/task_1a_repetition_v1/<fit identity>/
        recipe.toml   the schema 2 recipe that refits it (rebuild, never unpickle)
        fit.json      the fit record below
        weights.npy   the fitted readout weights read through rclib's accessor

The directory is written into a staging area and renamed into place, so a
partially written fit never exists under its final name; an existing fit is
never overwritten. Fresh-process refits bypass the cache by construction:
they rebuild the model from the recipe TOML and the dataset payload and compare
their weights and fit report against this record.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.records import to_toml
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec, fit_identity, recipe_for_arm
from arm_rc_ctrl.provenance import canonical_json, sha256_bytes, sha256_file
from arm_rc_ctrl.rc.esn import EsnModel
from arm_rc_ctrl.rc.recipe import ModelRecipe, RclibIdentity, TrainingValidation, load_recipe, solver_alpha
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.rc.training import FitReport, harvest_episode
from arm_rc_ctrl.storage import AccessMode, ArtifactUri, StorageRoot
from arm_rc_ctrl.validation import is_hex, validate_utc_timestamp

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.records import Normalization, Preprocessing
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.repetition_panel import PanelEntry
    from arm_rc_ctrl.rc.recipe import DatasetSource
    from arm_rc_ctrl.rc.teacher_forcing import Episode
    from arm_rc_ctrl.rc.train import ModelConfig
    from arm_rc_ctrl.scenario import ScenarioConfig

__all__ = [
    "CACHE_BUCKET",
    "FIT_FILE",
    "FIT_SCHEMA_VERSION",
    "RECIPE_FILE",
    "WEIGHTS_FILE",
    "CachedFit",
    "FitInputs",
    "FitRecord",
    "FitStore",
    "cache_uri",
    "fit_arm",
    "fit_record_from",
    "harvested_states",
    "recipe_text_of",
]

FIT_SCHEMA_VERSION: Final = 1
CACHE_BUCKET: Final = "models"
CACHE_PREFIX: Final = "task_1a_repetition_v1"
RECIPE_FILE: Final = "recipe.toml"
FIT_FILE: Final = "fit.json"
WEIGHTS_FILE: Final = "weights.npy"
_SHA256_HEX: Final = 64


@dataclass(frozen=True)
class FitRecord:
    """What one cached fit is and what it produced (``fit.json``)."""

    identity: str
    panel_label: str
    source_trial: int
    arm: ArmSpec
    recipe_sha256: str
    """SHA-256 of ``recipe.toml`` as written."""
    fit: FitReport
    weights_sha256: str
    """:func:`~arm_rc_ctrl.data.arrays.array_digest` of the weights (dtype, shape, and bytes)."""
    weights_shape: tuple[int, ...]
    """``(n_neurons + 1, output_dim)`` (the strict mapper reads homogeneous tuples only)."""
    state_digests: tuple[str, ...]
    """Digest of every training episode's harvested state array, in training order."""
    execution_identity: str
    rclib: RclibIdentity
    fit_seconds: float
    created_at: str
    schema_version: int = field(default=FIT_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Digests are well-formed, the counts agree with the fit report, and the timing is finite."""
        if self.schema_version != FIT_SCHEMA_VERSION:
            msg = f"unsupported fit schema_version {self.schema_version}"
            raise ValueError(msg)
        for name in ("identity", "recipe_sha256", "weights_sha256", "execution_identity"):
            if not is_hex(getattr(self, name), _SHA256_HEX):
                msg = f"{name} must be 64 lowercase hex characters, got {getattr(self, name)!r}"
                raise ValueError(msg)
        if any(not is_hex(d, _SHA256_HEX) for d in self.state_digests) or len(self.state_digests) != len(
            self.fit.episodes
        ):
            msg = "state_digests must hold one 64-hex digest per training episode of the fit report"
            raise ValueError(msg)
        if len(self.weights_shape) != 2 or any(v < 1 for v in self.weights_shape):  # noqa: PLR2004
            msg = f"weights_shape must be a positive 2-D shape, got {self.weights_shape}"
            raise ValueError(msg)
        if not (self.fit_seconds >= 0 and self.fit_seconds < float("inf")):
            msg = f"fit_seconds must be finite and non-negative, got {self.fit_seconds!r}"
            raise ValueError(msg)
        if self.source_trial < 0 or not self.panel_label.strip():
            msg = "a fit record names its panel entry and a non-negative source trial"
            raise ValueError(msg)
        validate_utc_timestamp(self.created_at)


@dataclass(frozen=True)
class CachedFit:
    """A fit read back from the cache with its recipe, weights, and fitted model."""

    record: FitRecord
    recipe: ModelRecipe
    weights: NDArray[np.float64]
    model: EsnModel
    episodes: tuple[Episode, ...]
    """The training episodes rebuilt from the recipe (also the fit's stacked rows in training order)."""
    cache_hit: bool
    """``True`` when the fit was served from the store rather than fitted in this process."""


def cache_uri(identity: str) -> ArtifactUri:
    """The store directory of one fit identity."""
    if not is_hex(identity, _SHA256_HEX):
        msg = f"fit identity must be 64 lowercase hex characters, got {identity!r}"
        raise ValueError(msg)
    return ArtifactUri.parse(f"armrc://{CACHE_BUCKET}/{CACHE_PREFIX}/{identity}")


@dataclass(frozen=True)
class FitInputs:
    """Everything a fit of one panel entry needs besides the arm (loaded once per validation run)."""

    base: ModelConfig
    source: DatasetSource
    samples: SampleSet
    dof: int
    task_code_dim: int
    preprocessing: Preprocessing
    normalization: Normalization
    scenario: ScenarioConfig
    scenario_file: Path
    root: Path
    execution_identity: str
    rclib: RclibIdentity

    def __post_init__(self) -> None:
        """The execution identity is a digest."""
        if not is_hex(self.execution_identity, _SHA256_HEX):
            msg = f"execution_identity must be 64 lowercase hex characters, got {self.execution_identity!r}"
            raise ValueError(msg)

    @property
    def mapping(self) -> Mapping[str, SampleSet]:
        """Training samples keyed by dataset ID."""
        return {self.source.artifact_id: self.samples}

    def transform(self) -> InputTransform:
        """The input transform the base model's policy derives from the dataset's recorded statistics."""
        return InputTransform.derive(
            self.base.input_transform.policy, self.normalization, fixed_scales=self.base.input_transform.fixed_scales
        )

    def validation(self) -> TrainingValidation:
        """The training validation bound to the scenario file."""
        return TrainingValidation.from_scenario(self.scenario, self.scenario_file, root=self.root)

    def identity(self, entry: PanelEntry, arm: ArmSpec) -> str:
        """The cache identity of ``entry`` fitted under ``arm`` in this environment."""
        return fit_identity(
            panel_label=entry.label,
            source_trial=entry.source_trial,
            arm=arm,
            warmup_s=entry.warmup_s,
            base_alpha=entry.base_alpha,
            solver_alpha=solver_alpha(entry.base_alpha, arm.regularization_rule, arm.count),
            dataset=self.source,
            transform=self.transform(),
            validation=self.validation(),
            rclib_commit=self.rclib.commit,
            execution_identity=self.execution_identity,
        )


def fit_record_from(
    *,
    identity: str,
    entry: PanelEntry,
    arm: ArmSpec,
    recipe_text: str,
    recipe: ModelRecipe,
    weights: NDArray[np.float64],
    states: tuple[NDArray[np.float64], ...],
    execution_identity: str,
    fit_seconds: float,
    now: datetime,
) -> FitRecord:
    """Assemble the record of a fit just produced."""
    return FitRecord(
        identity=identity,
        panel_label=entry.label,
        source_trial=entry.source_trial,
        arm=arm,
        recipe_sha256=sha256_bytes(recipe_text.encode("utf-8")),
        fit=recipe.fit,
        weights_sha256=array_digest(weights),
        weights_shape=(int(weights.shape[0]), int(weights.shape[1])),
        state_digests=tuple(array_digest(s) for s in states),
        execution_identity=execution_identity,
        rclib=recipe.rclib,
        fit_seconds=fit_seconds,
        created_at=now.astimezone(UTC).isoformat(timespec="seconds"),
    )


def fit_arm(entry: PanelEntry, arm: ArmSpec, inputs: FitInputs) -> tuple[ModelRecipe, EsnModel, tuple[Episode, ...]]:
    """Fit ``entry`` under ``arm`` in this process (no cache) and return recipe, model, and training episodes."""
    recipe, model = recipe_for_arm(
        entry,
        arm,
        base=inputs.base,
        source=inputs.source,
        samples=inputs.mapping,
        dof=inputs.dof,
        task_code_dim=inputs.task_code_dim,
        preprocessing=inputs.preprocessing,
        normalization=inputs.normalization,
        scenario=inputs.scenario,
        scenario_file=inputs.scenario_file,
        root=inputs.root,
    )
    episodes = tuple(recipe.episodes(inputs.mapping, scenario=inputs.scenario))
    return recipe, model, episodes


def harvested_states(model: EsnModel, episodes: tuple[Episode, ...]) -> tuple[NDArray[np.float64], ...]:
    """The reservoir state array of every episode, each harvested from a reset."""
    return tuple(harvest_episode(model, episode).states for episode in episodes)


RECIPE_HEADER: Final = (
    "# Deterministic model recipe (docs/PLAN.md section 8): rebuild and refit, never unpickle.\n"
    "# Written by arm_rc_ctrl.experiments.repetition_fits; do not edit.\n"
)


def recipe_text_of(recipe: ModelRecipe) -> str:
    """The TOML text a cached recipe is stored as (header plus the recipe's canonical TOML form)."""
    return RECIPE_HEADER + to_toml(recipe)


class FitStore:
    """The fit cache in one storage root."""

    def __init__(self, store: StorageRoot) -> None:
        """Bind to ``store``."""
        self._store = store

    @property
    def store(self) -> StorageRoot:
        """The storage root."""
        return self._store

    def directory(self, identity: str, *, mode: AccessMode = "read") -> Path:
        """The final directory of ``identity``."""
        return self._store.path(cache_uri(identity), mode=mode)

    def exists(self, identity: str) -> bool:
        """Whether a complete fit is cached under ``identity``."""
        directory = self._store.root / cache_uri(identity).relative_path
        return all((directory / name).is_file() for name in (RECIPE_FILE, FIT_FILE, WEIGHTS_FILE))

    def read_record(self, identity: str) -> FitRecord:
        """The strictly loaded ``fit.json`` of ``identity``."""
        text = (self.directory(identity) / FIT_FILE).read_text(encoding="utf-8")
        record = from_mapping(cast("dict[str, object]", json.loads(text)), FitRecord)
        if record.identity != identity:
            msg = f"cached fit {identity[:12]} carries the identity {record.identity[:12]}"
            raise ValueError(msg)
        return record

    def read_weights(self, record: FitRecord) -> NDArray[np.float64]:
        """The digest-verified weights of ``record``."""
        weights = np.asarray(
            np.load(self.directory(record.identity) / WEIGHTS_FILE, allow_pickle=False), dtype=np.float64
        )
        if array_digest(weights) != record.weights_sha256 or weights.shape != record.weights_shape:
            msg = f"cached weights of {record.identity[:12]} do not match their recorded digest or shape"
            raise ValueError(msg)
        return weights

    def read_recipe(self, record: FitRecord) -> ModelRecipe:
        """The digest-verified recipe of ``record``."""
        path = self.directory(record.identity) / RECIPE_FILE
        if sha256_file(path) != record.recipe_sha256:
            msg = f"cached recipe of {record.identity[:12]} does not match its recorded digest"
            raise ValueError(msg)
        return load_recipe(path)

    def write(self, record: FitRecord, recipe_text: str, weights: NDArray[np.float64]) -> Path:
        """Write a complete fit transactionally; an existing fit under the identity is an error."""
        final = self.directory(record.identity, mode="write")
        if final.exists():
            msg = f"fit {record.identity[:12]} already exists in the cache; fits are immutable"
            raise FileExistsError(msg)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = final.parent / f".staging-{record.identity}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            (staging / RECIPE_FILE).write_text(recipe_text, encoding="utf-8")
            np.save(staging / WEIGHTS_FILE, np.ascontiguousarray(weights, dtype=np.float64))
            (staging / FIT_FILE).write_text(canonical_json(to_mapping(record)) + "\n", encoding="utf-8")
            staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return final

    def fit_or_load(
        self, entry: PanelEntry, arm: ArmSpec, inputs: FitInputs, *, now: datetime | None = None
    ) -> CachedFit:
        """Serve the fit of ``entry`` under ``arm`` from the cache, or fit it now and cache it.

        A cached fit is verified against its own record (recipe and weight
        digests) and its recipe is rebuilt into a fitted model through the
        recipe's refit self-check, which reproduces the recorded fit report
        within the recipe's tolerance; the weights handed back are the cached
        array, never the refit's.
        """
        identity = inputs.identity(entry, arm)
        if self.exists(identity):
            record = self.read_record(identity)
            recipe = self.read_recipe(record)
            weights = self.read_weights(record)
            model, _report = recipe.refit(inputs.mapping, scenario=inputs.scenario)
            if array_digest(model.readout_weights()) != record.weights_sha256:
                msg = (
                    f"refitting cached fit {identity[:12]} in the same environment did not reproduce its weights "
                    "bitwise; the cache identity no longer describes this environment"
                )
                raise ValueError(msg)
            episodes = tuple(recipe.episodes(inputs.mapping, scenario=inputs.scenario))
            return CachedFit(record, recipe, weights, model, episodes, cache_hit=True)
        started = time.perf_counter()
        recipe, model, episodes = fit_arm(entry, arm, inputs)
        elapsed = time.perf_counter() - started
        weights = model.readout_weights()
        recipe_text = recipe_text_of(recipe)
        record = fit_record_from(
            identity=identity,
            entry=entry,
            arm=arm,
            recipe_text=recipe_text,
            recipe=recipe,
            weights=weights,
            states=harvested_states(model, episodes),
            execution_identity=inputs.execution_identity,
            fit_seconds=elapsed,
            now=datetime.now(tz=UTC) if now is None else now,
        )
        self.write(record, recipe_text, weights)
        return CachedFit(record, recipe, weights, model, episodes, cache_hit=False)
