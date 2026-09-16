# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Environment-bound fit cache of the manual-demonstration study (M3MAN-007; manual plan sections 4 and 5).

Every fit of the study is one entry of the frozen study manifest. Its cache key
is the manifest's own
:attr:`~arm_rc_ctrl.experiments.manual_study.StudyModel.fit_identity`, which
binds the configuration, the arm, the ridge scale, the digest-bound training
datasets, the frozen input transform, the training validation, a contractive
arm's construction, the ``rclib`` commit, and the execution environment
identity (clarification C10); a fit produced under another environment or
library build is never served. Each cached fit is a directory in the external
store::

    armrc://models/task_1a_manual_v1/<fit identity>/
        recipe.toml   the schema 3 recipe that refits it (rebuild, never unpickle)
        fit.json      the fit record below
        weights.npy   the fitted readout weights read through rclib's accessor

The prefix is the manual study's own, so the repeated-demonstration pilot's
cache and this one never share a bucket. The directory is written into a
staging area and renamed into place, so a partially written fit never exists
under its final name, and an existing fit is never overwritten.

The record keeps what the numerical controls check without re-reading the
payloads: per training episode, the digests of the inputs, the targets, the
loss-row mask, and the harvested reservoir states, together with the episode's
row counts and the equal-episode loss weight it was fitted with. For a
duplication control those nine fields are identical across the ten copies, and
that is exactly the literal-repetition condition the manual plan requires to be
verified before any behavioral comparison.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.records import to_toml
from arm_rc_ctrl.experiments.manual_recipes import ManualArmSpec, recipe_for_arm, training_spec_for_arm
from arm_rc_ctrl.provenance import canonical_json, sha256_bytes, sha256_file
from arm_rc_ctrl.rc.esn import EsnConfig
from arm_rc_ctrl.rc.recipe import ModelRecipe, RclibIdentity, load_recipe
from arm_rc_ctrl.rc.training import FitReport, harvest_episode
from arm_rc_ctrl.storage import AccessMode, ArtifactUri, StorageRoot
from arm_rc_ctrl.validation import SHA256_HEX_LENGTH, is_hex, validate_utc_timestamp

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.records import Preprocessing
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.manual_recipes import ArmAccounting
    from arm_rc_ctrl.experiments.manual_study import StudyConfiguration, StudyManifest, StudyModel
    from arm_rc_ctrl.rc.augment import TaskGeometry
    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.recipe import DatasetSource
    from arm_rc_ctrl.rc.teacher_forcing import Episode

__all__ = [
    "CACHE_BUCKET",
    "CACHE_PREFIX",
    "FIT_FILE",
    "FIT_SCHEMA_VERSION",
    "RECIPE_FILE",
    "WEIGHTS_FILE",
    "CachedFit",
    "EpisodeIdentity",
    "ManualFitInputs",
    "ManualFitRecord",
    "ManualFitStore",
    "cache_uri",
    "episode_identities",
    "fit_entry",
    "fit_record_from",
    "recipe_mismatches",
    "recipe_text_of",
    "states_witness",
]

FIT_SCHEMA_VERSION: Final = 1
CACHE_BUCKET: Final = "models"
CACHE_PREFIX: Final = "task_1a_manual_v1"
"""The manual study's own cache prefix; two studies never share a bucket."""
RECIPE_FILE: Final = "recipe.toml"
FIT_FILE: Final = "fit.json"
WEIGHTS_FILE: Final = "weights.npy"
_SHAPE_DIMENSIONS: Final = 2

RECIPE_HEADER: Final = (
    "# Deterministic model recipe (docs/PLAN.md section 8): rebuild and refit, never unpickle.\n"
    "# Written by arm_rc_ctrl.experiments.manual_fits; do not edit.\n"
)


def _require_digest(name: str, value: str) -> None:
    if not is_hex(value, SHA256_HEX_LENGTH):
        msg = f"{name} must be 64 lowercase hex characters, got {value!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class EpisodeIdentity:
    """Everything one training episode contributed to a fit, as digests rather than arrays.

    A literal copy repeats its parent exactly, so all four digests, the row
    counts, the warm-up length, and the loss weight are identical across the
    copies of a duplication control (manual plan section 5).
    """

    source: str
    rows: int
    washout_rows: int
    loss_rows: int
    weight: float
    """The equal-episode loss weight ``400 / L_i`` this episode was fitted with."""
    inputs_sha256: str
    targets_sha256: str
    mask_sha256: str
    """Digest of the loss-row mask: which rows entered the ridge loss and which were warm-up."""
    states_sha256: str
    """Digest of the reservoir states harvested for this episode from a reset."""

    def __post_init__(self) -> None:
        """Counts are consistent and every digest is well-formed."""
        if self.rows < 1 or self.washout_rows < 0 or self.loss_rows < 1:
            msg = f"{self.source}: an episode needs at least one row and one loss row, got {self.rows}"
            raise ValueError(msg)
        if self.rows != self.washout_rows + self.loss_rows:
            msg = f"{self.source}: {self.rows} rows are not {self.washout_rows} warm-up plus {self.loss_rows} loss rows"
            raise ValueError(msg)
        if not (math.isfinite(self.weight) and self.weight > 0):
            msg = f"{self.source}: weight must be positive and finite, got {self.weight!r}"
            raise ValueError(msg)
        for name in ("inputs_sha256", "targets_sha256", "mask_sha256", "states_sha256"):
            _require_digest(f"{self.source}: {name}", getattr(self, name))


@dataclass(frozen=True)
class ManualFitRecord:
    """What one cached fit is and what it produced (``fit.json``)."""

    identity: str
    configuration: str
    source_trial: int
    arm: ManualArmSpec
    recipe_sha256: str
    """SHA-256 of ``recipe.toml`` as written."""
    fit: FitReport
    weights_sha256: str
    """:func:`~arm_rc_ctrl.data.arrays.array_digest` of the weights (dtype, shape, and bytes)."""
    weights_shape: tuple[int, ...]
    """``(n_neurons + 1, dof)``: the explicit-bias layout with the bias row last (I3)."""
    episodes: tuple[EpisodeIdentity, ...]
    """One identity per training episode, in training order."""
    states_sha256: str
    """One witness over every episode's state digest, in training order."""
    execution_identity: str
    rclib: RclibIdentity
    fit_seconds: float
    created_at: str
    schema_version: int = field(default=FIT_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Digests are well-formed, the episodes agree with the fit report, and the timing is finite."""
        if self.schema_version != FIT_SCHEMA_VERSION:
            msg = f"unsupported fit schema_version {self.schema_version}"
            raise ValueError(msg)
        for name in ("identity", "recipe_sha256", "weights_sha256", "execution_identity", "states_sha256"):
            _require_digest(name, getattr(self, name))
        if len(self.episodes) != len(self.fit.episodes):
            msg = "a fit record carries one identity per training episode of its fit report"
            raise ValueError(msg)
        if tuple(e.source for e in self.episodes) != self.fit.episodes:
            msg = "the recorded episode identities must follow the fit report's episode labels in training order"
            raise ValueError(msg)
        if states_witness(self.episodes) != self.states_sha256:
            msg = "states_sha256 does not witness the recorded episode state digests"
            raise ValueError(msg)
        if len(self.weights_shape) != _SHAPE_DIMENSIONS or any(v < 1 for v in self.weights_shape):
            msg = f"weights_shape must be a positive 2-D shape, got {self.weights_shape}"
            raise ValueError(msg)
        if not (self.fit_seconds >= 0 and math.isfinite(self.fit_seconds)):
            msg = f"fit_seconds must be finite and non-negative, got {self.fit_seconds!r}"
            raise ValueError(msg)
        if self.source_trial < 0 or not self.configuration.strip():
            msg = "a fit record names its configuration and a non-negative source trial"
            raise ValueError(msg)
        validate_utc_timestamp(self.created_at)

    @property
    def label(self) -> str:
        """``<configuration>/<arm>``, e.g. ``feasible-best/R10/D07``."""
        return f"{self.configuration}/{self.arm.label}"


@dataclass(frozen=True)
class CachedFit:
    """A fit read back from the cache with its recipe, weights, fitted model, and training episodes."""

    record: ManualFitRecord
    recipe: ModelRecipe
    weights: NDArray[np.float64]
    model: EsnModel
    episodes: tuple[Episode, ...]
    cache_hit: bool
    """``True`` when the fit was served from the store rather than fitted in this process."""


def cache_uri(identity: str) -> ArtifactUri:
    """The store directory of one fit identity."""
    _require_digest("fit identity", identity)
    return ArtifactUri.parse(f"armrc://{CACHE_BUCKET}/{CACHE_PREFIX}/{identity}")


def states_witness(episodes: Sequence[EpisodeIdentity]) -> str:
    """One digest over every episode's harvested-state digest, in training order."""
    return sha256_bytes("".join(episode.states_sha256 for episode in episodes).encode("utf-8"))


@dataclass(frozen=True)
class ManualFitInputs:
    """Everything the study's fits consume besides the entry itself (loaded once per validation run)."""

    manifest: StudyManifest
    samples: dict[str, SampleSet]
    """The ten demonstrations' digest-verified payloads, keyed by processed artifact ID."""
    dof: int
    task_code_dim: int
    preprocessing: Preprocessing
    scenario: TaskGeometry
    """The manual task configuration the bank was recorded and validated under."""
    root: Path
    execution_identity: str
    rclib: RclibIdentity

    def __post_init__(self) -> None:
        """The environment and the library are the ones every recorded fit identity was hashed with."""
        _require_digest("execution_identity", self.execution_identity)
        if self.execution_identity != self.manifest.execution.identity:
            msg = (
                f"the fit inputs run in environment {self.execution_identity[:12]}, while the study manifest binds "
                f"{self.manifest.execution.identity[:12]}; a fit of another environment is never this study's (C10)"
            )
            raise ValueError(msg)
        if self.rclib != self.manifest.rclib:
            msg = (
                f"the installed rclib {self.rclib.commit[:12]} is not the manifest's "
                f"{self.manifest.rclib.commit[:12]}; reservoir and readout semantics may differ"
            )
            raise ValueError(msg)
        missing = [
            record.dataset.artifact_id
            for record in self.manifest.demonstrations
            if record.dataset.artifact_id not in self.samples
        ]
        if missing:
            msg = f"samples are missing for the demonstrations {missing}"
            raise ValueError(msg)

    @property
    def sources(self) -> dict[str, DatasetSource]:
        """The ten bank positions mapped to their digest-bound datasets."""
        return self.manifest.sources

    def configuration(self, entry: StudyModel) -> StudyConfiguration:
        """The inherited configuration ``entry`` belongs to."""
        return self.manifest.configuration(entry.configuration)

    def identity(self, entry: StudyModel) -> str:
        """The cache identity of ``entry``: the manifest's own key, re-derived when the manifest was loaded."""
        return entry.fit_identity

    def accounting(self, entry: StudyModel) -> ArmAccounting:
        """What ``entry``'s fit is made of, rebuilt from the demonstrations' loss rows."""
        return self.manifest.accounting(entry)

    def base_esn(self, entry: StudyModel) -> EsnConfig:
        """The inherited reservoir with the bound readout, before the arm's ridge scale is applied."""
        return EsnConfig(reservoir=self.configuration(entry).reservoir, readout=self.manifest.readout)


def _datasets_label(datasets: Sequence[DatasetSource]) -> str:
    """The training datasets in training order, as artifact IDs with their bound payload digests."""
    return "[" + ", ".join(f"{d.artifact_id} ({d.payload_sha256[:12]})" for d in datasets) + "]"


def recipe_mismatches(entry: StudyModel, recipe: ModelRecipe, inputs: ManualFitInputs) -> tuple[str, ...]:
    """Every way ``recipe``'s construction differs from the one ``entry`` and the frozen manifest demand.

    A fit identity is a digest of the construction that was hashed into it; it
    never witnesses that the recipe *stored* under it is that construction. So
    the recipe itself is compared here with what the study entry demands: the
    fitted ESN and with it the ridge parameter, the digest-bound training
    datasets, the training construction (weighting, multiplicities, warm-up,
    and a contractive bank where there is one), the frozen input transform and
    the dataset it was copied from (I8), the training validation, the
    preprocessing, both widths, and the pinned ``rclib`` build.
    """
    manifest = inputs.manifest
    configuration = inputs.configuration(entry)
    training = training_spec_for_arm(
        entry.arm,
        warmup_s=entry.warmup_s,
        base_alpha=configuration.base_alpha,
        anchor=manifest.anchor,
        contractive=None if entry.contractive is None else entry.contractive.spec,
    )
    compared: tuple[tuple[str, object, object], ...] = (
        ("esn", recipe.esn, manifest.esn(entry)),
        ("datasets", _datasets_label(recipe.datasets), _datasets_label(manifest.datasets(entry))),
        ("training", recipe.training, training),
        ("transform", recipe.transform, manifest.transform.transform),
        ("transform_source", recipe.transform_source, manifest.anchor.transform_source),
        ("validation", recipe.validation, manifest.validation),
        ("preprocessing", recipe.preprocessing, inputs.preprocessing),
        ("dof", recipe.dof, inputs.dof),
        ("task_code_dim", recipe.task_code_dim, inputs.task_code_dim),
        ("rclib", recipe.rclib, inputs.rclib),
    )
    return tuple(
        f"{name}: the recipe has {actual!r}, the study entry demands {expected!r}"
        for name, actual, expected in compared
        if actual != expected
    )


def _require_recipe(entry: StudyModel, recipe: ModelRecipe, inputs: ManualFitInputs) -> None:
    """Fail unless ``recipe`` is the construction the frozen study entry demands."""
    mismatches = recipe_mismatches(entry, recipe, inputs)
    if mismatches:
        reported = "; ".join(mismatches)
        msg = (
            f"{entry.label}: the recipe is not the construction the study froze for this model, so a fit of it is "
            f"another model's: {reported}"
        )
        raise ValueError(msg)


def fit_entry(entry: StudyModel, inputs: ManualFitInputs) -> tuple[ModelRecipe, EsnModel, tuple[Episode, ...]]:
    """Fit one study entry in this process (no cache) and return recipe, model, and training episodes."""
    configuration = inputs.configuration(entry)
    recipe, model = recipe_for_arm(
        entry.arm,
        esn=inputs.base_esn(entry),
        sources=inputs.sources,
        samples=inputs.samples,
        dof=inputs.dof,
        task_code_dim=inputs.task_code_dim,
        preprocessing=inputs.preprocessing,
        transform=inputs.manifest.transform.transform,
        validation=inputs.manifest.validation,
        warmup_s=entry.warmup_s,
        base_alpha=configuration.base_alpha,
        scenario=inputs.scenario,
        anchor=inputs.manifest.anchor,
        name=entry.label,
        contractive=None if entry.contractive is None else entry.contractive.spec,
    )
    # Nearly vacuous here, where the recipe was just built from the entry, and the whole guard on the serve path.
    _require_recipe(entry, recipe, inputs)
    episodes = tuple(recipe.episodes(inputs.samples, scenario=inputs.scenario))
    return recipe, model, episodes


def episode_identities(
    model: EsnModel, episodes: Sequence[Episode], weights: Sequence[float]
) -> tuple[EpisodeIdentity, ...]:
    """Digest every training episode's inputs, targets, loss-row mask, and harvested states, in training order."""
    identities: list[EpisodeIdentity] = []
    for episode, weight in zip(episodes, weights, strict=True):
        harvested = harvest_episode(model, episode)
        identities.append(
            EpisodeIdentity(
                source=episode.source,
                rows=episode.n_rows,
                washout_rows=episode.washout_len,
                loss_rows=int(np.count_nonzero(episode.loss_rows)),
                weight=float(weight),
                inputs_sha256=array_digest(episode.inputs),
                targets_sha256=array_digest(episode.targets),
                mask_sha256=array_digest(episode.loss_rows),
                states_sha256=array_digest(harvested.states),
            )
        )
    return tuple(identities)


def recipe_text_of(recipe: ModelRecipe) -> str:
    """The TOML text a cached recipe is stored as (header plus the recipe's canonical TOML form)."""
    return RECIPE_HEADER + to_toml(recipe)


def fit_record_from(
    *,
    identity: str,
    entry: StudyModel,
    recipe_text: str,
    recipe: ModelRecipe,
    weights: NDArray[np.float64],
    episodes: tuple[EpisodeIdentity, ...],
    execution_identity: str,
    fit_seconds: float,
    now: datetime,
) -> ManualFitRecord:
    """Assemble the record of a fit just produced."""
    return ManualFitRecord(
        identity=identity,
        configuration=entry.configuration,
        source_trial=entry.source_trial,
        arm=entry.arm,
        recipe_sha256=sha256_bytes(recipe_text.encode("utf-8")),
        fit=recipe.fit,
        weights_sha256=array_digest(weights),
        weights_shape=(int(weights.shape[0]), int(weights.shape[1])),
        episodes=episodes,
        states_sha256=states_witness(episodes),
        execution_identity=execution_identity,
        rclib=recipe.rclib,
        fit_seconds=fit_seconds,
        created_at=now.astimezone(UTC).isoformat(timespec="seconds"),
    )


class ManualFitStore:
    """The manual study's fit cache in one storage root."""

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

    def read_record(self, identity: str) -> ManualFitRecord:
        """The strictly loaded ``fit.json`` of ``identity``."""
        text = (self.directory(identity) / FIT_FILE).read_text(encoding="utf-8")
        record = from_mapping(cast("dict[str, object]", json.loads(text)), ManualFitRecord)
        if record.identity != identity:
            msg = f"cached fit {identity[:12]} carries the identity {record.identity[:12]}"
            raise ValueError(msg)
        return record

    def read_weights(self, record: ManualFitRecord) -> NDArray[np.float64]:
        """The digest-verified weights of ``record``."""
        weights = np.asarray(
            np.load(self.directory(record.identity) / WEIGHTS_FILE, allow_pickle=False), dtype=np.float64
        )
        if array_digest(weights) != record.weights_sha256 or weights.shape != record.weights_shape:
            msg = f"cached weights of {record.identity[:12]} do not match their recorded digest or shape"
            raise ValueError(msg)
        return weights

    def read_recipe(self, record: ManualFitRecord) -> ModelRecipe:
        """The digest-verified recipe of ``record``."""
        path = self.directory(record.identity) / RECIPE_FILE
        if sha256_file(path) != record.recipe_sha256:
            msg = f"cached recipe of {record.identity[:12]} does not match its recorded digest"
            raise ValueError(msg)
        return load_recipe(path)

    def write(self, record: ManualFitRecord, recipe_text: str, weights: NDArray[np.float64]) -> Path:
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

    def fit_or_load(self, entry: StudyModel, inputs: ManualFitInputs, *, now: datetime | None = None) -> CachedFit:
        """Serve the fit of ``entry`` from the cache, or fit it now and cache it.

        A cached fit is verified against its own record (recipe and weight
        digests), its recipe's complete construction is verified against the
        one ``entry`` and the frozen manifest demand, and the recipe is rebuilt
        into a fitted model through the recipe's refit self-check, which
        reproduces the recorded fit report and its per-episode weighting; the
        refit must also reproduce the weights bitwise, or the cache identity no
        longer describes this environment. The weights handed back are the
        cached array, never the refit's.
        """
        identity = inputs.identity(entry)
        if self.exists(identity):
            return self._serve(entry, identity, inputs)
        started = time.perf_counter()
        recipe, model, episodes = fit_entry(entry, inputs)
        elapsed = time.perf_counter() - started
        weights = model.readout_weights()
        recipe_text = recipe_text_of(recipe)
        record = fit_record_from(
            identity=identity,
            entry=entry,
            recipe_text=recipe_text,
            recipe=recipe,
            weights=weights,
            episodes=episode_identities(model, episodes, _weights_of(recipe)),
            execution_identity=inputs.execution_identity,
            fit_seconds=elapsed,
            now=datetime.now(tz=UTC) if now is None else now,
        )
        self.write(record, recipe_text, weights)
        return CachedFit(record, recipe, weights, model, episodes, cache_hit=False)

    def _serve(self, entry: StudyModel, identity: str, inputs: ManualFitInputs) -> CachedFit:
        record = self.read_record(identity)
        if (record.configuration, record.arm) != (entry.configuration, entry.arm):
            msg = (
                f"cached fit {identity[:12]} records {record.label}, not {entry.label}; a fit of another model is "
                "never served under this identity"
            )
            raise ValueError(msg)
        recipe = self.read_recipe(record)
        _require_recipe(entry, recipe, inputs)
        weights = self.read_weights(record)
        model, _report = recipe.refit(inputs.samples, scenario=inputs.scenario)
        if array_digest(model.readout_weights()) != record.weights_sha256:
            msg = (
                f"refitting cached fit {identity[:12]} in the same environment did not reproduce its weights "
                "bitwise; the cache identity no longer describes this environment"
            )
            raise ValueError(msg)
        episodes = tuple(recipe.episodes(inputs.samples, scenario=inputs.scenario))
        return CachedFit(record, recipe, weights, model, episodes, cache_hit=True)


def _weights_of(recipe: ModelRecipe) -> tuple[float, ...]:
    """The per-episode loss weights a schema 3 recipe recorded for its fit."""
    weights = recipe.fit.episode_weights
    if weights is None:  # pragma: no cover - schema 3 always records them
        msg = f"recipe {recipe.name!r} records no per-episode weights; it is not a weighted manual fit"
        raise ValueError(msg)
    return weights
