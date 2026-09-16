# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The frozen manual-demonstration study manifest: all 186 model identities (M3MAN-007; manual plan sections 4, 5).

``task_1a_manual_v1`` trains the 31 approved arms over the locked demonstration
bank (ten ``S_i``, ``M10``, ten ``R10_i``, ten ``C10_i``) under each of the six
inherited configurations of the frozen repetition panel (trials 17, 136, 53, 1,
0, 28). That cross product is the approved scope of 186 models. This module
resolves it once, binds every model to the inputs that decide its fitted
readout, and freezes the result into an immutable, portable manifest.

The header binds the read-only panel with its digest, the locked take bank with
its digest, the manual task and preprocessing configurations the takes were
validated and derived under (both taken from the bank itself, so the study can
only use what the bank was built with), the model configuration the panel
already bound together with the readout solver that file declares, the
transform and the digest-bound scripted dataset it was copied from
(clarification I8), the ten locked demonstrations in bank order — each with its
digest-bound dataset and the raw loss rows it contributes — the shared seed
bank of the contractive banks, the training validation, the pinned ``rclib``
revision, the canonical
:class:`~arm_rc_ctrl.execution.ExecutionRecord` (clarification C10), and the
provenance of the run that wrote the manifest.

Every entry is a reference, never a copy. It records its configuration and
source trial, the arm, the warm-up, the contractive construction of a ``C10``
arm (seed bank, the parent's recorded dwell onset, and that parent's bank
digest), the canonical execution identity, and the fit identity. Everything
else the fit consumes is bound once in the header and rebuilt on demand
(:meth:`StudyManifest.esn`, :meth:`StudyManifest.datasets`, and
:meth:`StudyManifest.accounting`): the fitted ESN from the configuration's
reservoir and the bound readout at ``K alpha_0`` with the explicit-bias layout
(I3), the digest-bound training datasets from the arm's assignments, and the
accounting (raw loss rows per episode, row weights, total loss weight, and the
regularization scale ``alpha_0 / 400``) from the ten demonstrations under the
frozen anchor. The recorded fit identity is the witness of that rebuild: it is
hashed from the rebuilt values, so a header that no longer produces them cannot
re-derive a single one of the 186 keys.

Loading re-derives every invariant instead of trusting the document: the
entries are exactly the six configurations crossed with the 31 arms, in order
and without a duplicate pair; the header names the ten bank positions in bank
order, each with a distinct digest-bound dataset; the accounting rebuilds from
those demonstrations' loss-row counts and describes the arm it belongs to;
every solver parameter is ``K alpha_0`` and every readout fits an explicit
bias, so a fixed-alpha diagnostic is refused (and the deferred ``M100`` is not
an arm the approved algebra admits); every ``C10`` entry names its parent's
dataset and bank digest at the manifest's seed bank; every entry binds the
manifest's canonical execution identity; and every fit identity re-derives from
the rebuilt inputs and is distinct from all others.

Payloads live in the external store. The contractive bank digests are derived
from the parent arrays the caller loads through it, so nothing written here
depends on a machine path, and the module stays importable and testable without
a storage root.

Command line::

    python -m arm_rc_ctrl.experiments.manual_study
        --panel docs/experiments/task_1a_repeated_demonstration/panel_manifest_v1.json
        --bank docs/experiments/task_1a_manual_demonstration/bank/bank_v1.json
        --scenario configs/tasks/task_1a_manual_v2.toml
        --preprocessing configs/preprocessing/manual_v2.toml --seed-bank 1
        --output docs/experiments/task_1a_manual_demonstration/study_manifest_v1.json
        --markdown docs/experiments/task_1a_manual_demonstration/study_manifest_v1.md [--exploratory]
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.manual import ManualDatasetRecord, load_manual_derive_config
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record, require_relative_posix, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.execution import ExecutionRecord, collect_execution, require_canonical
from arm_rc_ctrl.experiments.manual_augmentation import (
    MANUAL_PROTOCOL,
    ManualParent,
    ParentBankRecord,
    generate_banks,
    manual_parents,
    record_digest,
)
from arm_rc_ctrl.experiments.manual_bank import load_bank_manifest
from arm_rc_ctrl.experiments.manual_recipes import (
    ASSIGNMENTS,
    BANK_SIZE,
    CONTRACTIVE_ARM,
    MANUAL_ANCHOR,
    TRANSFORM_SOURCE,
    ArmAccounting,
    ManualAnchor,
    ManualArmSpec,
    arm_accounting,
    arm_sources,
    esn_for_arm,
    fit_identity,
    manual_arms,
)
from arm_rc_ctrl.experiments.repetition_panel import EXPERIMENT_LABEL as PANEL_EXPERIMENT
from arm_rc_ctrl.experiments.repetition_panel import load_panel
from arm_rc_ctrl.provenance import (
    DirtyWorktreeError,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
    worktree_state,
)
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import (
    ContractiveTrainingSpec,
    DatasetSource,
    RclibIdentity,
    TrainingValidation,
    solver_alpha,
)
from arm_rc_ctrl.rc.teacher_forcing import INPUT_CHANNELS, ChannelTransform, InputTransform
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage
from arm_rc_ctrl.validation import SHA256_HEX_LENGTH, is_hex

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.manual_bank import BankManifest
    from arm_rc_ctrl.experiments.repetition_panel import PanelManifest
    from arm_rc_ctrl.rc.augment import TaskGeometry
    from arm_rc_ctrl.rc.teacher_forcing import TransformPolicy
    from arm_rc_ctrl.rc.train import ModelConfig
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "ARM_COUNT",
    "CONFIGURATION_COUNT",
    "EXPERIMENT_LABEL",
    "MODEL_COUNT",
    "RIDGE_RULE",
    "STUDY_SCHEMA_VERSION",
    "ContractiveBank",
    "FrozenTransform",
    "SourceFile",
    "StudyConfiguration",
    "StudyDemonstration",
    "StudyManifest",
    "StudyMismatchError",
    "StudyModel",
    "TransformChannel",
    "build_study_manifest",
    "contractive_banks",
    "frozen_transform",
    "load_study",
    "main",
    "parent_samples",
    "render_study_markdown",
    "study_configurations",
    "study_to_json",
]

STUDY_SCHEMA_VERSION: Final = 1
EXPERIMENT_LABEL: Final = MANUAL_PROTOCOL
"""The manual experiment every model of this manifest belongs to."""
CONFIGURATION_COUNT: Final = 6
"""Inherited source configurations of the frozen repetition panel (D5)."""
ARM_COUNT: Final = len(manual_arms())
"""Approved arms per configuration (D4): ten ``S_i``, ``M10``, ten ``R10_i``, ten ``C10_i``."""
MODEL_COUNT: Final = CONFIGURATION_COUNT * ARM_COUNT
"""The approved scope: 186 model identities (manual plan section 5)."""
RIDGE_RULE: Final = "count_scaled"
"""The only ridge rule of the weighted objective: the solver fits at ``K alpha_0``."""
_SHORT: Final = 12


class StudyMismatchError(ValueError):
    """The manifest does not describe the approved study scope, or one of its bindings does not re-derive."""


def _require_sha256(name: str, value: str) -> None:
    if not is_hex(value, SHA256_HEX_LENGTH):
        msg = f"{name} must be 64 lowercase hex characters, got {value!r}"
        raise ValueError(msg)


def _require_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        msg = f"{name} must be positive and finite, got {value!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class SourceFile:
    """One digest-bound input file, named by its repository-relative path (never a machine path)."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        """The path is repository-relative and the digest well-formed."""
        require_relative_posix(self.path, "source file path")
        _require_sha256(f"{self.path} sha256", self.sha256)

    @classmethod
    def resolve(cls, path: Path, root: Path) -> SourceFile:
        """Digest ``path`` and record it relative to ``root`` (or by name when it lies outside the repository)."""
        return cls(path=_relative(path, root), sha256=sha256_file(path))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    base = root.resolve()
    return resolved.relative_to(base).as_posix() if resolved.is_relative_to(base) else path.name


@dataclass(frozen=True)
class StudyConfiguration:
    """One inherited panel configuration: its source trial, warm-up, ``alpha_0``, reservoir, and estimator cutoffs."""

    label: str
    source_trial: int
    warmup_s: float
    base_alpha: float
    """The source ridge parameter ``alpha_0`` every arm of this configuration scales with its episode count."""
    reservoir: ReservoirConfig
    """The inherited reservoir; identical across the arms of the configuration."""
    velocity_cutoff_hz: float
    acceleration_cutoff_hz: float
    """Evaluation-side causal estimator cutoffs of the source trial; never recipe fields."""

    def __post_init__(self) -> None:
        """Identity fields are present and every inherited value is a usable number."""
        if not self.label.strip() or self.source_trial < 0:
            msg = f"a configuration needs its label and a non-negative source trial, got {self.label!r}"
            raise ValueError(msg)
        if not (math.isfinite(self.warmup_s) and self.warmup_s >= 0):
            msg = f"{self.label}: warmup_s must be finite and non-negative, got {self.warmup_s!r}"
            raise ValueError(msg)
        _require_positive(f"{self.label}: base_alpha", self.base_alpha)
        _require_positive(f"{self.label}: velocity_cutoff_hz", self.velocity_cutoff_hz)
        _require_positive(f"{self.label}: acceleration_cutoff_hz", self.acceleration_cutoff_hz)


@dataclass(frozen=True)
class ContractiveBank:
    """The contractive construction of one ``C10`` arm: its parent, the seed bank, and the frozen bank's digest."""

    assignment: str
    """``D01`` .. ``D10``: the parent's position in the locked bank."""
    parent: str
    """The parent's processed artifact ID; the bank's streams are seeded from it (I8)."""
    seed_bank: int
    dwell_start_s: float
    """The parent's recorded final-dwell onset, which places the envelope's terminal taper."""
    bank_sha256: str
    """Digest of the frozen bank record (M3MAN-006); the witness every configuration shares."""

    def __post_init__(self) -> None:
        """The parent is a bank position with a digest-bound frozen bank."""
        if self.assignment not in ASSIGNMENTS:
            msg = f"assignment must be a bank position in {list(ASSIGNMENTS)}, got {self.assignment!r}"
            raise ValueError(msg)
        if not self.parent.strip():
            msg = f"{self.assignment}: parent must identify the demonstration the bank was grown from"
            raise ValueError(msg)
        if self.seed_bank < 0:
            msg = f"{self.assignment}: seed_bank must be non-negative, got {self.seed_bank}"
            raise ValueError(msg)
        _require_positive(f"{self.assignment}: dwell_start_s", self.dwell_start_s)
        _require_sha256(f"{self.assignment}: bank_sha256", self.bank_sha256)

    @property
    def spec(self) -> ContractiveTrainingSpec:
        """The recipe construction the arm's nine synthetic episodes regenerate from."""
        return ContractiveTrainingSpec(seed_bank=self.seed_bank, dwell_start_s=self.dwell_start_s)

    @classmethod
    def from_record(cls, record: ParentBankRecord) -> ContractiveBank:
        """The construction of one frozen bank record."""
        return cls(
            assignment=record.assignment,
            parent=record.dataset.artifact_id,
            seed_bank=record.config.seed_bank,
            dwell_start_s=record.dwell_start_s,
            bank_sha256=record.bank_sha256,
        )


@dataclass(frozen=True)
class TransformChannel:
    """One channel of the frozen input transform, recorded in the fixed channel order."""

    name: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    fixed_scale: float | None = None
    """The channel's shared physical scale under the ``fixed_scale`` policy; ``None`` under ``training_std``."""


@dataclass(frozen=True)
class FrozenTransform:
    """The frozen input transform as an ordered record.

    :class:`~arm_rc_ctrl.rc.teacher_forcing.InputTransform` holds its channels
    in a mapping whose order is semantic, and a canonical (sorted-key) document
    cannot carry that order. The manifest therefore records one entry per
    channel in the fixed order and rebuilds the transform from it; the rebuilt
    transform is the one the fit identities hash, so no recorded fit key
    depends on this representation.
    """

    policy: str
    derived_from: tuple[str, ...]
    channels: tuple[TransformChannel, ...]

    def __post_init__(self) -> None:
        """The channels are exactly the input channels in order, and the rebuilt transform validates itself."""
        names = tuple(channel.name for channel in self.channels)
        if names != INPUT_CHANNELS:
            msg = f"transform channels must be exactly {INPUT_CHANNELS} in order, got {names}"
            raise ValueError(msg)
        self.transform  # noqa: B018 - rebuilding validates the policy, the scales, and the statistics

    @property
    def transform(self) -> InputTransform:
        """The input transform this record rebuilds (the one every fit identity was derived from)."""
        channels: dict[str, ChannelTransform] = {}
        fixed: dict[str, float] = {}
        for channel in self.channels:
            channels[channel.name] = ChannelTransform(channel.center, channel.scale)
            if channel.fixed_scale is not None:
                fixed[channel.name] = channel.fixed_scale
        return InputTransform(cast("TransformPolicy", self.policy), self.derived_from, channels, fixed)

    @classmethod
    def of(cls, transform: InputTransform) -> FrozenTransform:
        """Freeze ``transform`` into its ordered record."""
        return cls(
            policy=transform.policy,
            derived_from=transform.derived_from,
            channels=tuple(
                TransformChannel(
                    name=name,
                    center=channel.center,
                    scale=channel.scale,
                    fixed_scale=transform.fixed_scales.get(name),
                )
                for name, channel in transform.channels.items()
            ),
        )


@dataclass(frozen=True)
class StudyDemonstration:
    """One locked demonstration: its bank position, its digest-bound dataset, and the loss rows it contributes.

    The ten demonstrations are bound once, in bank order. An entry names only
    its arm, and the arm's assignments resolve through this list to the datasets
    its fit trains on and the raw row counts its weights are built from, so no
    entry carries a copy of either and every dataset a model sees stays bound to
    the payload digest of its committed record.
    """

    assignment: str
    """``D01`` .. ``D10``: the demonstration's position in the locked bank."""
    dataset: DatasetSource
    """The digest-bound processed dataset the accepted take produced."""
    loss_rows: int
    """Raw loss rows of the recording: its samples minus the one row the next-step pairing consumes."""

    def __post_init__(self) -> None:
        """The position is a bank position and the recording contributes at least one loss row."""
        if self.assignment not in ASSIGNMENTS:
            msg = f"assignment must be a bank position in {list(ASSIGNMENTS)}, got {self.assignment!r}"
            raise ValueError(msg)
        if self.loss_rows < 1:
            msg = f"{self.assignment}: loss_rows must be positive, got {self.loss_rows}"
            raise ValueError(msg)


@dataclass(frozen=True)
class StudyModel:
    """One of the 186 models: a configuration, an arm, the environment it is keyed in, and the fit identity.

    An entry stores references rather than copies. Its fitted ESN, its training
    datasets, and its accounting rebuild from the manifest header — the
    configuration's reservoir and ``alpha_0``, the bound readout, the ten
    demonstrations, and the frozen anchor — through :meth:`esn`,
    :meth:`datasets`, and :meth:`accounting`. :attr:`fit_identity` is the
    witness of that rebuild: it is hashed from the rebuilt values, so a header
    that no longer produces them cannot re-derive it.
    """

    configuration: str
    source_trial: int
    arm: ManualArmSpec
    warmup_s: float
    fit_identity: str
    execution_identity: str
    contractive: ContractiveBank | None = None

    def __post_init__(self) -> None:
        """The entry identifies one model of one configuration, and only a contractive arm carries a bank."""
        if not self.configuration.strip() or self.source_trial < 0:
            msg = f"a model needs its configuration and a non-negative source trial, got {self.configuration!r}"
            raise ValueError(msg)
        if not (math.isfinite(self.warmup_s) and self.warmup_s >= 0):
            msg = f"{self.label}: warmup_s must be finite and non-negative, got {self.warmup_s!r}"
            raise ValueError(msg)
        _require_sha256(f"{self.label}: fit_identity", self.fit_identity)
        _require_sha256(f"{self.label}: execution_identity", self.execution_identity)
        _check_model_bank(self)

    @property
    def label(self) -> str:
        """``<configuration>/<arm>``, e.g. ``feasible-best/R10/D07``."""
        return f"{self.configuration}/{self.arm.label}"

    @property
    def episodes(self) -> int:
        """The episode count ``K`` the weighting and the ridge parameter refer to."""
        return self.arm.count

    def esn(
        self, configuration: StudyConfiguration, readout: ReadoutConfig, *, anchor: ManualAnchor = MANUAL_ANCHOR
    ) -> EsnConfig:
        """The complete configuration this arm fits: the inherited reservoir and the bound readout at ``K alpha_0``.

        The readout is the one the model configuration declares, re-solved at
        the arm's episode count with the explicit-bias layout the equal-episode
        weighting needs (I3); the reservoir is the configuration's own and never
        differs between the arms of one configuration.
        """
        base = EsnConfig(reservoir=configuration.reservoir, readout=readout)
        return esn_for_arm(base, self.arm, base_alpha=configuration.base_alpha, anchor=anchor)

    def datasets(self, sources: Mapping[str, DatasetSource]) -> tuple[DatasetSource, ...]:
        """The digest-bound training datasets in training order, resolved through the bank's assignments.

        Copies are recipe multiplicities, so a duplication control resolves to
        the single dataset it repeats and never to ten copies of it.
        """
        return arm_sources(self.arm, sources)

    def accounting(
        self,
        configuration: StudyConfiguration,
        loss_rows: Mapping[str, int],
        *,
        anchor: ManualAnchor = MANUAL_ANCHOR,
    ) -> ArmAccounting:
        """What this fit is made of: rows, weights, the total loss weight, and the ridge scales of ``K alpha_0``."""
        return arm_accounting(self.arm, base_alpha=configuration.base_alpha, loss_rows=loss_rows, anchor=anchor)


def _check_model_bank(model: StudyModel) -> None:
    """A contractive arm names the bank grown from its own parent; every other arm trains on recorded episodes only.

    The parent's dataset is checked in :func:`_check_entries`, where the bank
    positions resolve to the header's demonstrations.
    """
    construction = model.contractive
    if model.arm.arm != CONTRACTIVE_ARM:
        if construction is not None:
            msg = f"{model.label}: only a {CONTRACTIVE_ARM} arm carries a contractive bank digest"
            raise StudyMismatchError(msg)
        return
    if construction is None:
        msg = f"{model.label}: a {CONTRACTIVE_ARM} arm names its parent's bank digest and seed bank"
        raise StudyMismatchError(msg)
    if construction.assignment != model.arm.assignment:
        msg = (
            f"{model.label}: the bank digest belongs to {construction.assignment}, not to the arm's parent "
            f"{model.arm.assignment}"
        )
        raise StudyMismatchError(msg)


@dataclass(frozen=True)
class StudyManifest:
    """The immutable study: digest-bound sources, six configurations, 186 models, and both environment records."""

    experiment: str
    panel: SourceFile
    """The frozen repetition panel the six configurations are read from (read-only source)."""
    bank: SourceFile
    scenario: SourceFile
    preprocessing: SourceFile
    model: SourceFile
    """The model configuration the panel bound: the readout solver and the frozen physical input transform."""
    readout: ReadoutConfig
    """The readout that model configuration declares. Every arm fits it at ``K alpha_0`` with the explicit-bias
    layout (I3), so the ``alpha`` recorded here is the configuration file's own and never a model's ridge
    parameter; :meth:`esn` re-solves it for each entry."""
    transform_source: DatasetSource
    transform: FrozenTransform
    validation: TrainingValidation
    anchor: ManualAnchor
    rclib: RclibIdentity
    seed_bank: int
    demonstrations: tuple[StudyDemonstration, ...]
    """The ten locked demonstrations in bank order; every entry resolves its datasets and rows through them."""
    configurations: tuple[StudyConfiguration, ...]
    entries: tuple[StudyModel, ...]
    execution: ExecutionRecord
    """The canonical environment every fit identity binds (clarification C10)."""
    provenance: ProvenanceRecord
    schema_version: int = field(default=STUDY_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Re-derive every invariant of the frozen scope rather than trusting the document."""
        _check_header(self)
        _check_scope(self)
        _check_demonstrations(self)
        _check_entries(self)
        _check_accounting(self)
        _check_readouts(self)
        _check_identities(self)

    def configuration(self, label: str) -> StudyConfiguration:
        """The configuration with ``label``."""
        for configuration in self.configurations:
            if configuration.label == label:
                return configuration
        msg = f"no configuration labelled {label!r}"
        raise KeyError(msg)

    def entry(self, configuration: str, arm: str) -> StudyModel:
        """The model of ``configuration`` and the arm label ``arm`` (e.g. ``C10/D03``)."""
        for model in self.entries:
            if model.configuration == configuration and model.arm.label == arm:
                return model
        msg = f"no model {configuration!r}/{arm!r}"
        raise KeyError(msg)

    @property
    def sources(self) -> dict[str, DatasetSource]:
        """The ten bank positions mapped to their digest-bound datasets."""
        return {record.assignment: record.dataset for record in self.demonstrations}

    @property
    def loss_rows(self) -> dict[str, int]:
        """The ten bank positions mapped to the raw loss rows of their recordings."""
        return {record.assignment: record.loss_rows for record in self.demonstrations}

    def esn(self, entry: StudyModel) -> EsnConfig:
        """The complete ESN configuration ``entry`` fits, rebuilt from its configuration and the bound readout."""
        return entry.esn(self.configuration(entry.configuration), self.readout, anchor=self.anchor)

    def datasets(self, entry: StudyModel) -> tuple[DatasetSource, ...]:
        """The digest-bound training datasets of ``entry``, in training order."""
        return entry.datasets(self.sources)

    def accounting(self, entry: StudyModel) -> ArmAccounting:
        """The accounting of ``entry``, rebuilt from the demonstrations' loss rows under the frozen anchor."""
        return entry.accounting(self.configuration(entry.configuration), self.loss_rows, anchor=self.anchor)

    @property
    def models_by_arm(self) -> dict[str, int]:
        """How many models each arm kind contributes, in the approved arm order."""
        counts: dict[str, int] = {}
        for model in self.entries:
            counts[model.arm.arm] = counts.get(model.arm.arm, 0) + 1
        return counts


def _check_header(manifest: StudyManifest) -> None:
    """The schema, the experiment, the frozen transform source, and the canonical environment."""
    if manifest.schema_version != STUDY_SCHEMA_VERSION:
        msg = f"unsupported study schema_version {manifest.schema_version}; expected {STUDY_SCHEMA_VERSION}"
        raise ValueError(msg)
    if manifest.experiment != EXPERIMENT_LABEL:
        msg = f"the manifest belongs to {EXPERIMENT_LABEL!r}, got {manifest.experiment!r}"
        raise ValueError(msg)
    if manifest.seed_bank < 0:
        msg = f"seed_bank must be non-negative, got {manifest.seed_bank}"
        raise ValueError(msg)
    if manifest.anchor.transform_source != manifest.transform_source:
        msg = (
            f"the manifest copies its transform from {manifest.transform_source.artifact_id}, while the frozen anchor "
            f"names {manifest.anchor.transform_source.artifact_id} (I8)"
        )
        raise StudyMismatchError(msg)
    if manifest.transform.derived_from != (manifest.transform_source.artifact_id,):
        msg = (
            f"the input transform derives from {manifest.transform.derived_from}, not from the frozen source "
            f"{manifest.transform_source.artifact_id}; manual arms never compute statistics from a take (I8)"
        )
        raise StudyMismatchError(msg)
    manifest.execution.check_canonical()
    labels = [configuration.label for configuration in manifest.configurations]
    trials = [configuration.source_trial for configuration in manifest.configurations]
    if len(labels) != CONFIGURATION_COUNT or len(set(labels)) != len(labels) or len(set(trials)) != len(trials):
        msg = f"the study inherits {CONFIGURATION_COUNT} distinct panel configurations, got {labels} and {trials}"
        raise StudyMismatchError(msg)


def _check_scope(manifest: StudyManifest) -> None:
    """The entries are the six configurations crossed with the 31 approved arms, in order and without duplicates."""
    pairs = [(model.configuration, model.arm.label) for model in manifest.entries]
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if pair in seen:
            msg = f"the model {pair[0]}/{pair[1]} appears twice; every (configuration, arm) pair is bound once"
            raise StudyMismatchError(msg)
        seen.add(pair)
    expected = [(configuration.label, arm.label) for configuration in manifest.configurations for arm in manual_arms()]
    if len(pairs) != MODEL_COUNT or pairs != expected:
        msg = (
            f"the study binds {MODEL_COUNT} models: {CONFIGURATION_COUNT} configurations crossed with the "
            f"{ARM_COUNT} approved arms in report order, got {len(pairs)} entries"
        )
        raise StudyMismatchError(msg)


def _check_demonstrations(manifest: StudyManifest) -> None:
    """The header binds the locked bank once: every position, in bank order, with its own digest-bound dataset."""
    positions = tuple(record.assignment for record in manifest.demonstrations)
    if positions != ASSIGNMENTS:
        msg = f"the study binds the {BANK_SIZE} demonstrations {list(ASSIGNMENTS)} in bank order, got {list(positions)}"
        raise StudyMismatchError(msg)
    artifacts = [record.dataset.artifact_id for record in manifest.demonstrations]
    if len(set(artifacts)) != len(artifacts):
        msg = f"every bank position names a distinct dataset, got {artifacts}"
        raise StudyMismatchError(msg)


def _check_entries(manifest: StudyManifest) -> None:
    """Every model agrees with its configuration, the locked bank, the study's seed bank, and the environment."""
    identity = manifest.execution.identity
    sources = manifest.sources
    for model in manifest.entries:
        configuration = manifest.configuration(model.configuration)
        if (model.source_trial, model.warmup_s) != (configuration.source_trial, configuration.warmup_s):
            msg = f"{model.label}: the recorded trial and warm-up differ from {configuration.label}"
            raise StudyMismatchError(msg)
        if model.execution_identity != identity:
            msg = (
                f"{model.label}: the model binds the execution identity {model.execution_identity[:_SHORT]}, not the "
                f"manifest's canonical environment {identity[:_SHORT]} (C10)"
            )
            raise StudyMismatchError(msg)
        construction = model.contractive
        if construction is None:
            continue
        if construction.seed_bank != manifest.seed_bank:
            msg = (
                f"{model.label}: the bank was grown in seed bank {construction.seed_bank}, not the study's "
                f"{manifest.seed_bank}"
            )
            raise StudyMismatchError(msg)
        parent = sources[cast("str", model.arm.assignment)].artifact_id
        if construction.parent != parent:
            msg = (
                f"{model.label}: the bank was grown from {construction.parent}, not from {parent}, the dataset the "
                f"locked bank assigns to {model.arm.assignment}"
            )
            raise StudyMismatchError(msg)


def _check_accounting(manifest: StudyManifest) -> None:
    """Every arm's rows, weights, and ridge scales rebuild from the demonstrations and describe the arm they fit.

    The document no longer states an accounting, so this guards the rebuild: the
    shared episode algebra could never hand a model the rows, multiplicities, or
    weight reference of some other arm without the study refusing to load.
    """
    for model in manifest.entries:
        accounting, arm = manifest.accounting(model), model.arm
        described = (accounting.label, accounting.arm, accounting.assignments)
        counts = (accounting.unique_sources, accounting.copies, accounting.synthetic, accounting.episodes)
        expected = (arm.unique_sources, arm.copies, arm.synthetic, arm.count)
        if described != (arm.label, arm.arm, arm.assignments) or counts != expected:
            msg = (
                f"{model.label}: the accounting rebuilt from the demonstrations' loss rows describes {described} "
                f"{counts}, not the arm {(arm.label, arm.arm, arm.assignments)} {expected}"
            )
            raise StudyMismatchError(msg)
        if accounting.weight_reference_rows != manifest.anchor.weight_reference_rows:
            msg = f"{model.label}: the weight reference differs from the frozen anchor"
            raise StudyMismatchError(msg)


def _check_readouts(manifest: StudyManifest) -> None:
    """Every rebuilt fit solves at ``K alpha_0`` with the explicit-bias layout the weighting needs (I3, D4).

    No document can state a solver parameter any more — the ridge rule is bound
    once in the frozen anchor and every readout rebuilds from it — so this
    guards the rebuild itself: a fixed-alpha diagnostic or an implicit-bias
    readout is refused whatever produced it.
    """
    for model in manifest.entries:
        configuration = manifest.configuration(model.configuration)
        readout = manifest.esn(model).readout
        expected = solver_alpha(configuration.base_alpha, RIDGE_RULE, model.episodes)
        if readout.alpha != expected or manifest.accounting(model).solver_alpha != expected:
            msg = (
                f"{model.label}: the solver parameter must be {model.episodes} x alpha_0 = {expected!r} "
                f"({RIDGE_RULE}), got readout {readout.alpha!r}; the fixed-alpha diagnostic stays deferred (D4)"
            )
            raise StudyMismatchError(msg)
        if not readout.explicit_bias or readout.include_bias:
            msg = f"{model.label}: manual arms fit the explicit-bias readout (I3), got {readout!r}"
            raise StudyMismatchError(msg)


def _check_identities(manifest: StudyManifest) -> None:
    """Every fit identity re-derives from the recorded inputs and identifies exactly one model."""
    transform = manifest.transform.transform
    seen: dict[str, str] = {}
    for model in manifest.entries:
        construction = None if model.contractive is None else model.contractive.spec
        expected = fit_identity(
            configuration=model.configuration,
            arm=model.arm,
            warmup_s=model.warmup_s,
            base_alpha=manifest.configuration(model.configuration).base_alpha,
            esn=manifest.esn(model),
            datasets=manifest.datasets(model),
            transform=transform,
            validation=manifest.validation,
            anchor=manifest.anchor,
            rclib_commit=manifest.rclib.commit,
            execution_identity=manifest.execution.identity,
            contractive=construction,
            bank_sha256=None if model.contractive is None else model.contractive.bank_sha256,
        )
        if expected != model.fit_identity:
            msg = (
                f"{model.label}: the recorded fit identity {model.fit_identity[:_SHORT]} does not re-derive from the "
                f"manifest's inputs (expected {expected[:_SHORT]})"
            )
            raise StudyMismatchError(msg)
        previous = seen.setdefault(expected, model.label)
        if previous != model.label:
            msg = f"{model.label} and {previous} share the fit identity {expected[:_SHORT]}"
            raise StudyMismatchError(msg)


def frozen_transform(model: ModelConfig, *, root: Path | None = None) -> InputTransform:
    """The frozen physical input transform, copied from the historical scripted dataset's statistics (I8).

    Raises
    ------
    ValueError
        If the committed record is not the frozen transform source, or records no statistics.
    """
    base = repository_root() if root is None else root
    record = load_record(base / TRANSFORM_SOURCE.record, ProcessedDatasetRecord)
    artifact = record.artifact
    if (
        artifact.artifact_id != TRANSFORM_SOURCE.artifact_id
        or artifact.payload.sha256 != TRANSFORM_SOURCE.payload_sha256
    ):
        msg = (
            f"record {TRANSFORM_SOURCE.record} describes {artifact.artifact_id} ({artifact.payload.sha256[:_SHORT]}), "
            f"not the frozen transform source {TRANSFORM_SOURCE.artifact_id}"
        )
        raise ValueError(msg)
    normalization = record.normalization
    if normalization is None:
        msg = f"the transform source {TRANSFORM_SOURCE.artifact_id} records no normalization statistics"
        raise ValueError(msg)
    return InputTransform.derive(
        model.input_transform.policy, normalization, fixed_scales=model.input_transform.fixed_scales
    )


def contractive_banks(
    parents: Sequence[ManualParent],
    samples: Mapping[str, SampleSet],
    scenario: TaskGeometry,
    *,
    seed_bank: int,
) -> dict[str, ParentBankRecord]:
    """Grow every parent's contractive bank at ``seed_bank`` and return the frozen records by bank position.

    Raises
    ------
    ValueError
        If any parent's bank could not be grown; the study binds ten complete
        banks or none, and the reported failures stay in the message.
    """
    generation = generate_banks(parents, samples, scenario, seed_bank=seed_bank)
    if generation.failures:
        reported = "; ".join(
            f"{failure.assignment} ({failure.dataset.artifact_id}): {failure.reason}" for failure in generation.failures
        )
        msg = f"the contractive banks of seed bank {seed_bank} are incomplete: {reported}"
        raise ValueError(msg)
    return {bank.record.assignment: bank.record for bank in generation.banks}


def parent_samples(
    parents: Sequence[ManualParent], store: StorageRoot, *, root: Path | None = None
) -> dict[str, SampleSet]:
    """Load every parent's payload from the external store, verified against its committed record."""
    base = repository_root() if root is None else root
    samples: dict[str, SampleSet] = {}
    for parent in parents:
        record = load_record(base / parent.dataset.record, ManualDatasetRecord)
        if record.artifact.payload.sha256 != parent.dataset.payload_sha256:
            msg = (
                f"{parent.assignment}: record {parent.dataset.record} carries payload "
                f"{record.artifact.payload.sha256[:_SHORT]}, not {parent.dataset.payload_sha256[:_SHORT]}"
            )
            raise ValueError(msg)
        loaded = load_samples(verify_payload(store, record.artifact))
        record.check_samples(loaded)
        samples[parent.identifier] = loaded
    return samples


def study_configurations(panel: PanelManifest, model: ModelConfig) -> tuple[StudyConfiguration, ...]:
    """The six inherited configurations: each panel entry's reservoir, warm-up, ``alpha_0``, and estimator cutoffs."""
    if panel.experiment != PANEL_EXPERIMENT or len(panel.entries) != CONFIGURATION_COUNT:
        msg = (
            f"the study inherits the {CONFIGURATION_COUNT} configurations of the frozen {PANEL_EXPERIMENT!r} panel, "
            f"got {panel.experiment!r} with {len(panel.entries)} entries"
        )
        raise StudyMismatchError(msg)
    configurations: list[StudyConfiguration] = []
    for entry in panel.entries:
        inherited = entry.point.esn.model_config(model, name=f"{EXPERIMENT_LABEL}/{entry.label}")
        configurations.append(
            StudyConfiguration(
                label=entry.label,
                source_trial=entry.source_trial,
                warmup_s=entry.warmup_s,
                base_alpha=entry.base_alpha,
                reservoir=inherited.esn.reservoir,
                velocity_cutoff_hz=entry.point.esn.velocity_cutoff_hz,
                acceleration_cutoff_hz=entry.point.esn.acceleration_cutoff_hz,
            )
        )
    return tuple(configurations)


def _check_bank(bank: BankManifest, *, scenario_file: Path, preprocessing_file: Path, root: Path) -> None:
    """The bank is the locked, complete one, and the study uses exactly the configurations it was built with."""
    if bank.protocol != EXPERIMENT_LABEL:
        msg = f"the bank belongs to {bank.protocol!r}, not to {EXPERIMENT_LABEL!r}"
        raise StudyMismatchError(msg)
    if not bank.complete or bank.required != BANK_SIZE:
        msg = f"the study needs the complete bank of {BANK_SIZE} demonstrations; it is short of {bank.shortfall}"
        raise StudyMismatchError(msg)
    for name, file, recorded, digest in (
        ("scenario", scenario_file, bank.scenario_path, bank.scenario_sha256),
        ("preprocessing", preprocessing_file, bank.derive_config_path, bank.derive_config_sha256),
    ):
        relative = _relative(file, root)
        current = sha256_file(file)
        if relative != recorded or current != digest:
            msg = (
                f"the {name} configuration {relative} ({current[:_SHORT]}) is not the one the bank was built with, "
                f"{recorded} ({digest[:_SHORT]})"
            )
            raise StudyMismatchError(msg)


def _check_parents(parents: Sequence[ManualParent], bank: BankManifest) -> dict[str, ManualParent]:
    """The parents are the bank's ten assignments, each bound to the dataset that take produced."""
    if [parent.assignment for parent in parents] != list(ASSIGNMENTS):
        msg = f"the study needs one parent per bank position {list(ASSIGNMENTS)}, got {[p.assignment for p in parents]}"
        raise StudyMismatchError(msg)
    assigned = {take.assignment: take for take in bank.takes if take.assignment is not None}
    resolved: dict[str, ManualParent] = {}
    for parent in parents:
        take = assigned.get(parent.assignment)
        if take is None or take.processed_artifact_id != parent.dataset.artifact_id:
            recorded = None if take is None else take.processed_artifact_id
            msg = (
                f"{parent.assignment}: the parent {parent.dataset.artifact_id} is not the dataset of the locked bank's "
                f"take ({recorded})"
            )
            raise StudyMismatchError(msg)
        resolved[parent.assignment] = parent
    return resolved


def _check_banks(
    banks: Mapping[str, ParentBankRecord], parents: Mapping[str, ManualParent], *, seed_bank: int
) -> dict[str, ParentBankRecord]:
    """One frozen bank per parent, at the study's seed bank, whose digest re-derives from its own record."""
    missing = [name for name in ASSIGNMENTS if name not in banks]
    if missing or len(banks) != len(ASSIGNMENTS):
        msg = f"the study needs one contractive bank per demonstration; missing {missing}"
        raise StudyMismatchError(msg)
    for assignment, record in sorted(banks.items()):
        parent = parents[assignment]
        if record.protocol != EXPERIMENT_LABEL or record.assignment != assignment:
            msg = f"{assignment}: the bank record belongs to {record.protocol!r}/{record.assignment}"
            raise StudyMismatchError(msg)
        if record.config.seed_bank != seed_bank:
            msg = (
                f"{assignment}: the bank was grown in seed bank {record.config.seed_bank}, not the study's {seed_bank}"
            )
            raise StudyMismatchError(msg)
        if record.dataset != parent.dataset or record.dwell_start_s != parent.dwell_start_s:
            msg = f"{assignment}: the bank was grown from another parent or another recorded dwell onset"
            raise StudyMismatchError(msg)
        if record.parent_q_sha256 != parent.q_sha256:
            msg = f"{assignment}: the bank binds another recording than the parent's committed arrays"
            raise StudyMismatchError(msg)
        if record.bank_sha256 != record_digest(record):
            msg = f"{assignment}: bank_sha256 does not match the record's own contents"
            raise StudyMismatchError(msg)
    return dict(banks)


def _check_model_file(panel: PanelManifest, model_file: Path, root: Path) -> None:
    """The model configuration is the one the panel bound, byte for byte."""
    relative = _relative(model_file, root)
    digest = sha256_file(model_file)
    if relative != panel.configs.model_file or digest != panel.configs.model_sha256:
        msg = (
            f"the model configuration {relative} ({digest[:_SHORT]}) is not the one the panel bound, "
            f"{panel.configs.model_file} ({panel.configs.model_sha256[:_SHORT]})"
        )
        raise StudyMismatchError(msg)


def _study_model(
    configuration: StudyConfiguration,
    arm: ManualArmSpec,
    *,
    readout: ReadoutConfig,
    sources: Mapping[str, DatasetSource],
    banks: Mapping[str, ParentBankRecord],
    transform: InputTransform,
    validation: TrainingValidation,
    anchor: ManualAnchor,
    rclib_commit: str,
    execution_identity: str,
) -> StudyModel:
    """Bind one arm of one configuration to the identity of everything its fit consumes.

    The ESN and the datasets are rebuilt here exactly as the loader rebuilds
    them, hashed into the fit identity, and then dropped: the manifest records
    the identity, and the header records the inputs it was hashed from.
    """
    esn = esn_for_arm(
        EsnConfig(reservoir=configuration.reservoir, readout=readout),
        arm,
        base_alpha=configuration.base_alpha,
        anchor=anchor,
    )
    datasets = arm_sources(arm, sources)
    construction = (
        ContractiveBank.from_record(banks[cast("str", arm.assignment)]) if arm.arm == CONTRACTIVE_ARM else None
    )
    return StudyModel(
        configuration=configuration.label,
        source_trial=configuration.source_trial,
        arm=arm,
        warmup_s=configuration.warmup_s,
        fit_identity=fit_identity(
            configuration=configuration.label,
            arm=arm,
            warmup_s=configuration.warmup_s,
            base_alpha=configuration.base_alpha,
            esn=esn,
            datasets=datasets,
            transform=transform,
            validation=validation,
            anchor=anchor,
            rclib_commit=rclib_commit,
            execution_identity=execution_identity,
            contractive=None if construction is None else construction.spec,
            bank_sha256=None if construction is None else construction.bank_sha256,
        ),
        execution_identity=execution_identity,
        contractive=construction,
    )


def build_study_manifest(
    *,
    panel: PanelManifest,
    panel_file: Path,
    bank: BankManifest,
    bank_file: Path,
    scenario_file: Path,
    preprocessing_file: Path,
    model: ModelConfig,
    model_file: Path,
    parents: Sequence[ManualParent],
    banks: Mapping[str, ParentBankRecord],
    transform: InputTransform,
    validation: TrainingValidation,
    execution: ExecutionRecord,
    provenance: ProvenanceRecord,
    seed_bank: int,
    rclib: RclibIdentity | None = None,
    anchor: ManualAnchor = MANUAL_ANCHOR,
    root: Path | None = None,
) -> StudyManifest:
    """Bind the 186 models of the approved scope to their sources, weights, banks, and environment.

    Every source is verified before it is recorded: the bank must be the locked
    one and must have been built with the given task and preprocessing
    configurations, the parents must be its ten assignments, each contractive
    bank must belong to its parent at the study's seed bank with a digest that
    re-derives from its own record, and the model configuration must be the one
    the panel bound.
    """
    base = repository_root() if root is None else root
    _check_bank(bank, scenario_file=scenario_file, preprocessing_file=preprocessing_file, root=base)
    resolved_parents = _check_parents(parents, bank)
    resolved_banks = _check_banks(banks, resolved_parents, seed_bank=seed_bank)
    _check_model_file(panel, model_file, base)
    demonstrations = tuple(
        StudyDemonstration(
            assignment=name, dataset=resolved_parents[name].dataset, loss_rows=resolved_parents[name].n_samples - 1
        )
        for name in ASSIGNMENTS
    )
    sources = {record.assignment: record.dataset for record in demonstrations}
    identity = RclibIdentity.current() if rclib is None else rclib
    configurations = study_configurations(panel, model)
    entries = tuple(
        _study_model(
            configuration,
            arm,
            readout=model.esn.readout,
            sources=sources,
            banks=resolved_banks,
            transform=transform,
            validation=validation,
            anchor=anchor,
            rclib_commit=identity.commit,
            execution_identity=execution.identity,
        )
        for configuration in configurations
        for arm in manual_arms()
    )
    return StudyManifest(
        experiment=EXPERIMENT_LABEL,
        panel=SourceFile.resolve(panel_file, base),
        bank=SourceFile.resolve(bank_file, base),
        scenario=SourceFile.resolve(scenario_file, base),
        preprocessing=SourceFile.resolve(preprocessing_file, base),
        model=SourceFile.resolve(model_file, base),
        readout=model.esn.readout,
        transform_source=anchor.transform_source,
        transform=FrozenTransform.of(transform),
        validation=validation,
        anchor=anchor,
        rclib=identity,
        seed_bank=seed_bank,
        demonstrations=demonstrations,
        configurations=configurations,
        entries=entries,
        execution=execution,
        provenance=provenance,
    )


def study_to_json(manifest: StudyManifest) -> str:
    """Canonical JSON of the manifest."""
    return canonical_json(to_mapping(manifest))


def load_study(path: Path) -> StudyManifest:
    """Strictly rebuild a manifest from JSON, re-deriving every invariant of the frozen scope."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), StudyManifest)


def _short(digest: str) -> str:
    return digest[:_SHORT]


def _demonstrations(manifest: StudyManifest) -> list[tuple[StudyDemonstration, float, ContractiveBank]]:
    """Per bank position: the demonstration, its equal-episode row weight, and the frozen contractive bank."""
    first = manifest.configurations[0].label
    banks = {
        cast("str", model.arm.assignment): cast("ContractiveBank", model.contractive)
        for model in manifest.entries
        if model.arm.arm == CONTRACTIVE_ARM and model.configuration == first
    }
    reference = manifest.anchor.weight_reference_rows
    return [(record, reference / record.loss_rows, banks[record.assignment]) for record in manifest.demonstrations]


def _configuration_rows(manifest: StudyManifest) -> list[str]:
    rows: list[str] = []
    for configuration in manifest.configurations:
        reservoir = configuration.reservoir
        cells = (
            configuration.label,
            str(configuration.source_trial),
            f"{configuration.warmup_s:g}",
            repr(configuration.base_alpha),
            str(reservoir.n_neurons),
            repr(reservoir.spectral_radius),
            repr(reservoir.sparsity),
            repr(reservoir.leak_rate),
            repr(reservoir.input_scaling),
            str(reservoir.seed),
            repr(configuration.velocity_cutoff_hz),
            repr(configuration.acceleration_cutoff_hz),
        )
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def _arm_rows(manifest: StudyManifest) -> list[str]:
    counts = manifest.models_by_arm
    rows: list[str] = []
    seen: set[str] = set()
    for arm in manual_arms():
        if arm.arm in seen:
            continue
        seen.add(arm.arm)
        scale = "alpha_0" if arm.count == 1 else f"{arm.count} alpha_0"
        cells = (
            arm.arm,
            str(counts[arm.arm]),
            str(arm.count),
            scale,
            str(arm.unique_sources),
            str(arm.copies),
            str(arm.synthetic),
        )
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def render_study_markdown(manifest: StudyManifest) -> str:
    """The Markdown rendering of the manifest (the JSON stays the exact record)."""
    execution = manifest.execution
    provenance = manifest.provenance
    lines = [
        "# Task 1-a manual study manifest (v1)",
        "",
        (
            f"Experiment `{manifest.experiment}`: the {MODEL_COUNT} models of the approved scope "
            f"({CONFIGURATION_COUNT} inherited configurations crossed with the {ARM_COUNT} arms of the locked "
            "demonstration bank), frozen before any arm is fitted (manual plan sections 4 and 5, decisions D4 and D5)."
        ),
        "",
        "## Sources",
        "",
        f"- Repetition panel `{manifest.panel.path}` (sha256 `{_short(manifest.panel.sha256)}`), read-only.",
        f"- Demonstration bank `{manifest.bank.path}` (sha256 `{_short(manifest.bank.sha256)}`).",
        (
            f"- Task configuration `{manifest.scenario.path}` (`{_short(manifest.scenario.sha256)}`) and preprocessing "
            f"`{manifest.preprocessing.path}` (`{_short(manifest.preprocessing.sha256)}`), both as the bank records "
            "them."
        ),
        (
            f"- Model configuration `{manifest.model.path}` (`{_short(manifest.model.sha256)}`), the file the panel "
            "bound: the readout solver and the frozen physical input transform."
        ),
        (
            f"- Input transform `{manifest.transform.policy}` copied from `{manifest.transform_source.artifact_id}` "
            f"(payload `{_short(manifest.transform_source.payload_sha256)}`, record "
            f"`{manifest.transform_source.record}`); no statistics come from a manual take (I8)."
        ),
        (
            f"- Training construction: {manifest.anchor.episode_weighting} weighting with "
            f"{manifest.anchor.weight_reference_rows} reference rows, the "
            f"`{manifest.anchor.regularization_rule}` ridge rule, target `{manifest.anchor.target}`, "
            f"washout `{manifest.anchor.washout}`."
        ),
        (
            f"- Training validation `{manifest.validation.scenario_file}` "
            f"(`{_short(manifest.validation.scenario_sha256)}`); contractive seed bank {manifest.seed_bank}; "
            f"rclib {manifest.rclib.version} (`{_short(manifest.rclib.commit)}`)."
        ),
        (
            f"- Execution identity `{_short(execution.identity)}` (policy `{execution.policy}`, "
            f"{'canonical' if execution.canonical else 'NOT canonical'}); every fit identity binds it (C10)."
        ),
        "",
        "## Configurations",
        "",
        (
            "The inherited reservoir, warm-up, and `alpha_0` of each source trial; the estimator cutoffs are "
            "evaluation-side settings and never recipe fields."
        ),
        "",
        (
            "| configuration | trial | warm-up (s) | alpha_0 | n_neurons | spectral_radius | sparsity | leak_rate "
            "| input_scaling | seed | velocity_cutoff_hz | acceleration_cutoff_hz |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *_configuration_rows(manifest),
        "",
        "## Arms",
        "",
        (
            "Every configuration carries the same arms. `M100` and the fixed-alpha diagnostics stay deferred (D4), so "
            "no model trains ten episodes at `alpha_0`."
        ),
        "",
        "| arm | models | episodes K | solver alpha | unique sources | copies | synthetic |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        *_arm_rows(manifest),
        "",
        "## Demonstrations",
        "",
        (
            "Each locked demonstration with the rows it contributes to a fit, its equal-episode weight "
            f"{manifest.anchor.weight_reference_rows} / L_i, and the frozen contractive bank grown from it."
        ),
        "",
        "| position | dataset | loss rows | row weight | dwell onset (s) | bank digest |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {record.assignment} | `{record.dataset.artifact_id}` | {record.loss_rows} | {weight:.6g} "
        f"| {bank.dwell_start_s:g} | `{_short(bank.bank_sha256)}` |"
        for record, weight, bank in _demonstrations(manifest)
    )
    counts = manifest.models_by_arm
    lines.extend(
        [
            "",
            "## Models",
            "",
            (
                f"- {MODEL_COUNT} models: "
                + ", ".join(f"{count} `{arm}`" for arm, count in counts.items())
                + f", one per (configuration, arm) pair over {CONFIGURATION_COUNT} configurations."
            ),
            (
                "- Every fit identity binds the arm, the datasets, the transform, the training validation, the ridge "
                "scale, the contractive construction where there is one, the pinned rclib revision, and the execution "
                "identity; all of them are distinct and re-derived when this manifest is loaded."
            ),
            (
                "- Entries reference this header instead of repeating it: each records its configuration, arm, "
                "warm-up, contractive bank, and identities, while its fitted ESN, its digest-bound datasets, and its "
                "accounting are rebuilt from the configurations, the bound readout, and the demonstrations above."
            ),
            "",
            "## Provenance",
            "",
            (
                f"- Manifest: commit `{_short(provenance.project_commit)}`"
                f"{' (dirty)' if provenance.project_dirty else ''}, created {provenance.created_at}, Python "
                f"{provenance.platform.python}, lock `{_short(provenance.lock_sha256)}`"
                f"{', exploratory' if provenance.exploratory else ''}."
            ),
            (
                f"- Execution: {execution.policy} on {execution.machine}, Python {execution.python}, "
                f"recorded {execution.created_at}."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _load_runtimes() -> None:
    """Load the numerical runtimes before the execution environment is probed.

    ``rclib`` links its own OpenMP runtime, and the probe records the library
    and its thread count only once that runtime is loaded. A manifest frozen
    without it would bind an environment no fit ever runs in, and would
    disagree with the repository's canonical execution record; the numerics
    would then refuse every fit as belonging to another environment (C10).
    """
    for name in ("numpy", "rclib"):
        importlib.import_module(name)


def _require_clean_worktree(root: Path, *, exploratory: bool) -> None:
    """Fail fast on a modified checkout before any payload is read (the provenance guard stays authoritative)."""
    if exploratory:
        return
    _commit, dirty = worktree_state(root)
    if dirty:
        msg = (
            "freezing the study manifest requires a clean checkout: the project worktree is dirty; commit the work "
            "first or pass --exploratory"
        )
        raise DirtyWorktreeError(msg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the manual-demonstration study manifest (186 models).")
    parser.add_argument("--panel", type=Path, required=True, help="frozen repetition panel manifest (read-only)")
    parser.add_argument("--bank", type=Path, required=True, help="locked demonstration bank manifest")
    parser.add_argument("--scenario", type=Path, required=True, help="manual task configuration of the bank")
    parser.add_argument("--preprocessing", type=Path, required=True, help="manual preprocessing configuration")
    parser.add_argument("--seed-bank", type=int, required=True, help="shared seed bank of the contractive banks")
    parser.add_argument("--output", type=Path, required=True, help="study manifest JSON to write (must not exist)")
    parser.add_argument("--markdown", type=Path, required=True, help="study manifest Markdown (must not exist)")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_parser().parse_args(argv)
    output, markdown = Path(args.output), Path(args.markdown)
    for target in (output, markdown):
        if target.exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    root = repository_root()
    exploratory = bool(args.exploratory)
    _require_clean_worktree(root, exploratory=exploratory)
    require_canonical()
    _load_runtimes()
    panel_file, bank_file = Path(args.panel), Path(args.bank)
    scenario_file, preprocessing_file = Path(args.scenario), Path(args.preprocessing)
    seed_bank = int(args.seed_bank)
    panel = load_panel(panel_file)
    bank = load_bank_manifest(bank_file)
    scenario = load_manual_scenario(scenario_file)
    load_manual_derive_config(preprocessing_file)  # validated here; the datasets already carry these settings
    model_file = root / panel.configs.model_file
    model = load_model_config(model_file)
    command = command_line("arm_rc_ctrl.experiments.manual_study", list(sys.argv[1:] if argv is None else argv))
    execution = collect_execution(command=command, now=datetime.now(tz=UTC))
    execution.check_canonical()
    parents = manual_parents(bank, root=root)
    banks = contractive_banks(
        parents, parent_samples(parents, open_storage(), root=root), scenario, seed_bank=seed_bank
    )
    transform = frozen_transform(model, root=root)
    validation = TrainingValidation.from_scenario(scenario, scenario_file, root=root)
    resolved: dict[str, object] = {
        "experiment": EXPERIMENT_LABEL,
        "panel": {panel_file.name: sha256_file(panel_file)},
        "bank": {bank_file.name: sha256_file(bank_file)},
        "scenario": {scenario_file.name: sha256_file(scenario_file)},
        "preprocessing": {preprocessing_file.name: sha256_file(preprocessing_file)},
        "model": {model_file.name: sha256_file(model_file)},
        "transform_source": to_mapping(TRANSFORM_SOURCE),
        "seed_bank": seed_bank,
        "execution_identity": execution.identity,
        "command": command,
    }
    provenance = collect_provenance(
        resolved, seeds={"contractive_seed_bank": seed_bank}, exploratory=exploratory, now=datetime.now(tz=UTC)
    )
    require_clean_for_confirmatory(provenance)
    manifest = build_study_manifest(
        panel=panel,
        panel_file=panel_file,
        bank=bank,
        bank_file=bank_file,
        scenario_file=scenario_file,
        preprocessing_file=preprocessing_file,
        model=model,
        model_file=model_file,
        parents=parents,
        banks=banks,
        transform=transform,
        validation=validation,
        execution=execution,
        provenance=provenance,
        seed_bank=seed_bank,
        root=root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(study_to_json(manifest) + "\n", encoding="utf-8")
    markdown.write_text(render_study_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment": manifest.experiment,
                "models": len(manifest.entries),
                "configurations": [configuration.label for configuration in manifest.configurations],
                "seed_bank": manifest.seed_bank,
                "execution_identity": manifest.execution.identity,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
