# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Numerical copy controls of the manual-demonstration study (M3MAN-007; manual plan sections 4 and 5).

Before any behavioral evaluation, the study's duplication controls are checked
on the frozen study manifest with the actual fitted weights:

1. literal repetition: every copy of a ``R10_i`` fit is verified to repeat its
   parent's inputs, targets, loss weights, loss-row masks, warm-up length, and
   harvested reservoir features bitwise, so the control differs from its
   singleton only in episode count and ridge scale;
2. ``R10_i`` (ten copies at ``10 alpha_0``) is compared with ``S_i`` (the
   recording once at ``alpha_0``) on one matched probe matrix per
   configuration: the task-row states of all ten locked demonstrations,
   harvested from a reset reservoir after the configuration's warm-up. A fit
   that trains a contractive bank is probed on that bank as well; probe-only
   trajectories never enter a fit or the frozen input transform, which is
   copied from the historical scripted dataset and never computed from a take
   (clarification I8). Absolute output is the only quantity: v1 arms predict
   the absolute next position, so no increment is ever compared with a
   position;
3. the approved tolerance is elementwise ``|a - b| <= atol + rtol |b|`` with
   ``atol = 1e-8`` rad and ``rtol = 1e-8``; maximum absolute and relative
   differences and the decision are recorded for every comparison;
4. every fit reports the conditioning of its *weighted* normal matrix and the
   normalized residual ``||A W - B||_F / (||A||_F ||W||_F + ||B||_F)`` with
   ``A = Xw^T Xw + alpha I`` and ``B = Xw^T Yw`` on the scaled design
   ``Xw = sqrt(W) [states, 1]``, ``Yw = sqrt(W) Y`` — the system the
   equal-episode weighting actually solves, with the explicit bias column
   appended exactly once and the bias row last (``0 / 0 := 0``). It must not
   exceed ``1e-10``;
5. every fit is refitted in a fresh, pinned worker process that bypasses the
   cache, verifies it runs in the parent's execution environment, and compares
   its weights, harvested states, and fit report with the cached fit.

A failed check is retained in the evidence with its figures; no tolerance is
relaxed and no arm is substituted. The predecessor's accepted numerical
exception is not blanket approval here (manual plan section 5). Every fit binds
the execution environment identity and the ``rclib`` commit of the build that
produced it.

Entries reference the study manifest instead of copying it: the manifest is
bound by digest in the header, and every count and decision of this record
re-derives from the retained rows when it is loaded, so the evidence of the
whole approved scope stays inside the repository's committable size.

Command line (launch pinned through ``python -m arm_rc_ctrl.execution run --policy p-cores -- ...``)::

    python -m arm_rc_ctrl.experiments.manual_numerics validate
        --study <docs>/study_manifest_v1.json
        --output <docs>/numerical_validation_v1.json --markdown <docs>/numerical_validation_v1.md
        --workspace <outside the worktree>
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.manual import ManualDatasetRecord
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.execution import ExecutionRecord, collect_execution, require_canonical
from arm_rc_ctrl.experiments.manual_augmentation import (
    MANUAL_PROTOCOL,
    ManualParent,
    ParentFailure,
    generate_parent_bank,
)
from arm_rc_ctrl.experiments.manual_fits import (
    RECIPE_FILE,
    CachedFit,
    ManualFitInputs,
    ManualFitRecord,
    ManualFitStore,
    episode_identities,
    states_witness,
)
from arm_rc_ctrl.experiments.manual_recipes import ASSIGNMENTS
from arm_rc_ctrl.experiments.manual_study import ContractiveBank, StudyManifest, StudyModel, load_study
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
)
from arm_rc_ctrl.rc.augment import generate_manual_augmentation
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.recipe import RclibIdentity, derivative_config, load_recipe
from arm_rc_ctrl.rc.runtime import load_training_samples
from arm_rc_ctrl.rc.training import harvest_episode, train_readout
from arm_rc_ctrl.rc.warmup import WarmupConfig, build_task_episode_arrays
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot, open_storage
from arm_rc_ctrl.validation import SHA256_HEX_LENGTH, is_hex

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.rc.augment import TaskGeometry
    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.teacher_forcing import InputEncoder

__all__ = [
    "COMPARISON_PAIR",
    "EXPERIMENT_LABEL",
    "LARGE_FILE_LIMIT_BYTES",
    "NUMERICAL_ARMS",
    "NUMERICS_SCHEMA_VERSION",
    "PROBE_BANKS",
    "TOLERANCES",
    "Comparison",
    "FitSummary",
    "FreshRefit",
    "ManualNumericalValidation",
    "ManualStudyContext",
    "NormalEquations",
    "ProbeBank",
    "ProbeMatrix",
    "Refitter",
    "RepeatIdentity",
    "Tolerances",
    "build_probes",
    "compare_fits",
    "fresh_refit",
    "load_validation",
    "main",
    "numerical_entries",
    "predict_probes",
    "refit_in_subprocess",
    "render_validation_markdown",
    "repeat_identity",
    "run_validation",
    "sha256_file",
    "validation_to_json",
    "weighted_normal_equations",
    "worker_main",
]

NUMERICS_SCHEMA_VERSION: Final = 1
EXPERIMENT_LABEL: Final = MANUAL_PROTOCOL
NUMERICAL_ARMS: Final = ("S", "R10")
"""The arms of the copy controls: the ten singletons and the ten duplication controls (manual plan section 5).

``M10`` has no count-matched repeat equivalence while ``M100`` stays deferred (D4), and ``C10`` is a synthetic
comparison rather than a repetition of recorded data, so neither is a numerical copy control."""
COMPARISON_PAIR: Final = ("R10", "S")
"""``(candidate, reference)``: the duplication control's predictions are compared against its singleton's."""
PROBE_BANKS: Final = ("manual", "contractive")
"""Probe banks in stacking order: the ten locked demonstrations, then a fit's own contractive bank if it has one."""
LARGE_FILE_LIMIT_BYTES: Final = 200 * 1024
"""The repository's ``check-added-large-files`` limit; the evidence must stay committable."""
_SHORT: Final = 12
_WORKER_MODULE: Final = "arm_rc_ctrl.experiments.manual_numerics"
_APPROVED_TOLERANCES: Final = (1e-8, 1e-8, 1e-10)


@dataclass(frozen=True)
class Tolerances:
    """The approved tolerances; any other value is refused."""

    prediction_atol_rad: float = 1e-8
    prediction_rtol: float = 1e-8
    residual_max: float = 1e-10

    def __post_init__(self) -> None:
        """Only the approved values exist."""
        if (self.prediction_atol_rad, self.prediction_rtol, self.residual_max) != _APPROVED_TOLERANCES:
            msg = (
                "the numerical tolerances are prescribed by the manual plan (section 5): prediction atol 1e-8 rad, "
                "rtol 1e-8, normal-equation residual 1e-10; they cannot be changed here"
            )
            raise ValueError(msg)


TOLERANCES: Final = Tolerances()


def numerical_entries(manifest: StudyManifest, arms: tuple[str, ...] = NUMERICAL_ARMS) -> tuple[StudyModel, ...]:
    """The study entries the copy controls validate, in manifest order."""
    return tuple(entry for entry in manifest.entries if entry.arm.arm in arms)


# --- the weighted ridge system --------------------------------------------------------------


def _frobenius(array: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.sum(array * array, dtype=np.float64)))


@dataclass(frozen=True)
class NormalEquations:
    """The normalized residual and conditioning of one fit's actual weighted ridge problem."""

    rows: int
    columns: int
    """``n_neurons + 1``: the explicit bias column is appended once and sits last (I3)."""
    alpha: float
    """``K alpha_0``, the parameter the solver was handed."""
    residual: float
    """``||A W - B||_F / (||A||_F ||W||_F + ||B||_F)`` (``0 / 0 := 0``)."""
    a_norm: float
    b_norm: float
    w_norm: float
    eigenvalue_min: float
    eigenvalue_max: float
    cond2: float
    within: bool

    @property
    def denominator(self) -> float:
        """``||A||_F ||W||_F + ||B||_F``: the residual's scale, re-derived from the recorded norms."""
        return self.a_norm * self.w_norm + self.b_norm

    def __post_init__(self) -> None:
        """Both decisions re-derive from the figures."""
        if self.within != (self.residual <= TOLERANCES.residual_max):
            msg = "within contradicts the residual"
            raise ValueError(msg)
        if self.denominator == 0.0 and self.residual != 0.0:
            msg = "an all-zero problem has residual zero"
            raise ValueError(msg)
        expected = float("inf") if self.eigenvalue_min <= 0.0 else self.eigenvalue_max / self.eigenvalue_min
        if self.cond2 != expected:
            msg = "cond2 contradicts the extreme eigenvalues"
            raise ValueError(msg)


def weighted_normal_equations(
    states: NDArray[np.float64],
    targets: NDArray[np.float64],
    row_weights: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    alpha: float,
) -> NormalEquations:
    """Evaluate the equal-episode weighted ridge system against the fitted ``W``.

    The fitted design is ``Xw = sqrt(W) [states, 1]`` and the fitted target
    ``Yw = sqrt(W) Y``, so the system is ``A = Xw^T Xw + alpha I`` and
    ``B = Xw^T Yw``. The ones column is the readout's own explicit bias, which
    the harvested ``states`` do not carry, so it is appended here exactly once.
    """
    x = np.asarray(states, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(row_weights, dtype=np.float64)
    if w.shape != (x.shape[0],) or not np.all(np.isfinite(w)) or not np.all(w > 0):
        msg = f"row_weights must hold one positive weight per row ({x.shape[0]},), got {w.shape}"
        raise ValueError(msg)
    design = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)]) * np.sqrt(w)[:, None]
    if weights.shape != (design.shape[1], y.shape[1]):
        msg = f"weights {weights.shape} do not match the design matrix {design.shape} and targets {y.shape}"
        raise ValueError(msg)
    a = design.T @ design + alpha * np.eye(design.shape[1], dtype=np.float64)
    b = design.T @ (y * np.sqrt(w)[:, None])
    numerator = _frobenius(a @ weights - b)
    a_norm, b_norm, w_norm = _frobenius(a), _frobenius(b), _frobenius(weights)
    denominator = a_norm * w_norm + b_norm
    residual = 0.0 if denominator == 0.0 else numerator / denominator
    eigenvalues = np.linalg.eigvalsh(a)
    e_min, e_max = float(eigenvalues[0]), float(eigenvalues[-1])
    return NormalEquations(
        rows=int(design.shape[0]),
        columns=int(design.shape[1]),
        alpha=alpha,
        residual=residual,
        a_norm=a_norm,
        b_norm=b_norm,
        w_norm=w_norm,
        eigenvalue_min=e_min,
        eigenvalue_max=e_max,
        cond2=float("inf") if e_min <= 0.0 else e_max / e_min,
        within=residual <= TOLERANCES.residual_max,
    )


# --- probes ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeBank:
    """One bank of probe episodes: where its rows came from and what they were."""

    bank: str
    episodes: int
    first_row: int
    """Row of the stacked probe matrix at which this bank starts (banks follow ``PROBE_BANKS`` order)."""
    episode_labels: tuple[str, ...]
    episode_rows: tuple[int, ...]
    """Task rows each episode contributed; manual recordings have unequal lengths."""
    states_sha256: str

    def __post_init__(self) -> None:
        """Counts agree, every episode contributes rows, and the digest is well-formed."""
        if self.bank not in PROBE_BANKS or self.episodes < 1 or self.first_row < 0:
            msg = f"malformed probe bank {self.bank!r}"
            raise ValueError(msg)
        if len(self.episode_labels) != self.episodes or len(self.episode_rows) != self.episodes:
            msg = f"bank {self.bank}: one label and one row count per episode are required"
            raise ValueError(msg)
        if any(rows < 1 for rows in self.episode_rows):
            msg = f"bank {self.bank}: every probe episode contributes at least one row, got {self.episode_rows}"
            raise ValueError(msg)
        if not is_hex(self.states_sha256, SHA256_HEX_LENGTH):
            msg = f"bank {self.bank}: states_sha256 must be 64 lowercase hex characters"
            raise ValueError(msg)

    @property
    def rows(self) -> int:
        """Rows the bank contributes."""
        return sum(self.episode_rows)


@dataclass(frozen=True)
class ProbeMatrix:
    """The stacked probe states of one fit, with the bank every row came from."""

    states: NDArray[np.float64]
    """``(rows, n_neurons)`` harvested task-row states."""
    banks: tuple[ProbeBank, ...]

    def bank_of(self, row: int) -> str:
        """The bank a stacked row belongs to."""
        for bank in reversed(self.banks):
            if row >= bank.first_row:
                return bank.bank
        msg = f"row {row} precedes every bank"
        raise ValueError(msg)

    @property
    def matched(self) -> NDArray[np.float64]:
        """The shared block every arm of one configuration is compared on: the ten manual trajectories."""
        return self.states[: self.banks[0].rows]


def _probe_arrays(
    entry: StudyModel, inputs: ManualFitInputs
) -> dict[str, tuple[tuple[str, SampleSet, NDArray[np.float64], NDArray[np.float64]], ...]]:
    """Per bank, every probe episode as ``(label, the recording that supplies its clock and task code, q, dq)``.

    The ten locked demonstrations probe every readout. A fit that trains a
    contractive bank is probed on that bank too, regenerated here from its
    parent's recorded arrays alone, exactly as the recipe regenerates it.
    """
    manual = tuple((name, inputs.samples[inputs.sources[name].artifact_id]) for name in ASSIGNMENTS)
    banks: dict[str, tuple[tuple[str, SampleSet, NDArray[np.float64], NDArray[np.float64]], ...]] = {
        "manual": tuple((name, samples, samples.q, samples.dq) for name, samples in manual),
        "contractive": (),
    }
    construction = entry.contractive
    if construction is None:
        return banks
    source = inputs.sources[cast("str", entry.arm.assignment)]
    parent = inputs.samples[source.artifact_id]
    result = generate_manual_augmentation(
        parent.t,
        parent.q,
        inputs.scenario,
        construction.spec.config(source.artifact_id),
        dwell_start_s=construction.dwell_start_s,
        period_s=inputs.preprocessing.resample_period_s,
        derivatives=derivative_config(inputs.preprocessing.derivative_method),
    )
    banks["contractive"] = tuple(
        (f"{source.artifact_id}#contractive-{episode.episode:03d}", parent, episode.arrays.q, episode.arrays.dq)
        for episode in result.episodes
    )
    return banks


def build_probes(model: EsnModel, encoder: InputEncoder, entry: StudyModel, inputs: ManualFitInputs) -> ProbeMatrix:
    """Harvest the probe matrix of one fit: all ten demonstrations, plus its own contractive bank if it has one.

    Every probe episode is driven from a reset reservoir through the
    configuration's warm-up, and only its task rows are kept, so the matrix
    lies in the same feature space the ridge loss used.
    """
    warmup = WarmupConfig(entry.warmup_s)
    period = inputs.preprocessing.resample_period_s
    arrays = _probe_arrays(entry, inputs)
    stacked: list[NDArray[np.float64]] = []
    banks: list[ProbeBank] = []
    first_row = 0
    for name in PROBE_BANKS:
        family = arrays[name]
        if not family:
            continue
        rows: list[NDArray[np.float64]] = []
        labels: list[str] = []
        counts: list[int] = []
        for label, samples, q, dq in family:
            episode = build_task_episode_arrays(
                samples.t, q, dq, samples.task_code, encoder, source=label, warmup=warmup, period_s=period
            )
            states = harvest_episode(model, episode).training_states
            rows.append(states)
            labels.append(label)
            counts.append(int(states.shape[0]))
        block = np.vstack(rows)
        banks.append(
            ProbeBank(
                bank=name,
                episodes=len(labels),
                first_row=first_row,
                episode_labels=tuple(labels),
                episode_rows=tuple(counts),
                states_sha256=array_digest(block),
            )
        )
        first_row += int(block.shape[0])
        stacked.append(block)
    return ProbeMatrix(np.vstack(stacked), tuple(banks))


def predict_probes(model: EsnModel, probes: ProbeMatrix) -> NDArray[np.float64]:
    """The readout's prediction on every probe state through ``rclib``'s own predict path."""
    return np.vstack([model.readout(state) for state in probes.states])


# --- literal repetition ---------------------------------------------------------------------


@dataclass(frozen=True)
class RepeatIdentity:
    """Whether every copy of a duplication control repeats its parent bitwise."""

    configuration: str
    arm: str
    episodes: int
    inputs_identical: bool
    targets_identical: bool
    weights_identical: bool
    masks_identical: bool
    warmup_identical: bool
    states_identical: bool
    identical: bool

    def __post_init__(self) -> None:
        """The verdict re-derives from the conditions."""
        conditions = (
            self.inputs_identical,
            self.targets_identical,
            self.weights_identical,
            self.masks_identical,
            self.warmup_identical,
            self.states_identical,
        )
        if self.identical != all(conditions):
            msg = "identical contradicts the recorded conditions"
            raise ValueError(msg)
        if self.episodes < 2:  # noqa: PLR2004 - a repetition needs a parent and at least one copy
            msg = f"{self.arm}: a literal repetition compares at least two episodes, got {self.episodes}"
            raise ValueError(msg)


def repeat_identity(record: ManualFitRecord) -> RepeatIdentity | None:
    """The literal-repetition verdict of a duplication control, or ``None`` for an arm without copies."""
    if record.arm.copies < 1:
        return None
    episodes = record.episodes

    def shared(name: str) -> bool:
        return len({getattr(episode, name) for episode in episodes}) == 1

    conditions = {
        "inputs_identical": shared("inputs_sha256"),
        "targets_identical": shared("targets_sha256"),
        "weights_identical": shared("weight"),
        "masks_identical": shared("mask_sha256"),
        "warmup_identical": shared("washout_rows") and shared("rows"),
        "states_identical": shared("states_sha256"),
    }
    return RepeatIdentity(
        configuration=record.configuration,
        arm=record.arm.label,
        episodes=len(episodes),
        identical=all(conditions.values()),
        **conditions,
    )


# --- comparisons ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """One equivalence comparison: ``R10_i`` against ``S_i`` on the matched probe block."""

    configuration: str
    assignment: str
    candidate: str
    reference: str
    rows: int
    max_abs: float
    max_rel: float
    """Largest ``|a - b| / |b|`` over entries with ``b != 0`` (zero when there are none)."""
    worst_row: int
    worst_bank: str
    coefficient_max_abs: float
    coefficient_max_rel: float
    coefficient_fro_rel: float
    """``||W_c - W_r||_F / max(||W_c||_F, ||W_r||_F)``."""
    passed: bool
    """``all(|a - b| <= atol + rtol |b|)`` over the matched probe rows."""

    def __post_init__(self) -> None:
        """Only the approved pair is a comparison, and both arms carry the same demonstration."""
        kinds = (self.candidate.split("/")[0], self.reference.split("/")[0])
        suffix = f"/{self.assignment}"
        if kinds != COMPARISON_PAIR or not (self.candidate.endswith(suffix) and self.reference.endswith(suffix)):
            msg = (
                f"the approved comparison is {COMPARISON_PAIR[0]}_i against {COMPARISON_PAIR[1]}_i at matched probes, "
                f"got {self.candidate} against {self.reference}"
            )
            raise ValueError(msg)
        if self.rows < 1 or self.worst_row < 0 or self.worst_bank not in PROBE_BANKS:
            msg = f"{self.candidate}: a comparison needs rows and a known worst bank, got {self.worst_bank!r}"
            raise ValueError(msg)


def compare_fits(
    candidate: CachedFit,
    reference: CachedFit,
    bank: ProbeBank,
    candidate_prediction: NDArray[np.float64],
    reference_prediction: NDArray[np.float64],
) -> Comparison:
    """Compare the duplication control against its singleton on the matched manual probe block."""
    arm_c, arm_r = candidate.record.arm, reference.record.arm
    if (arm_c.arm, arm_r.arm) != COMPARISON_PAIR or arm_c.assignment != arm_r.assignment:
        msg = f"{arm_c.label} is not compared against {arm_r.label}"
        raise ValueError(msg)
    if candidate_prediction.shape != reference_prediction.shape:
        msg = f"predictions differ in shape ({candidate_prediction.shape} vs {reference_prediction.shape})"
        raise ValueError(msg)
    diff = np.abs(candidate_prediction - reference_prediction)
    scale = np.abs(reference_prediction)
    bound = TOLERANCES.prediction_atol_rad + TOLERANCES.prediction_rtol * scale
    worst = int(np.argmax(diff - bound))
    nonzero = scale > 0.0
    wc, wr = candidate.weights, reference.weights
    coefficient = np.abs(wc - wr)
    reference_scale = np.abs(wr) > 0.0
    return Comparison(
        configuration=candidate.record.configuration,
        assignment=cast("str", arm_c.assignment),
        candidate=arm_c.label,
        reference=arm_r.label,
        rows=int(diff.shape[0]),
        max_abs=float(np.max(diff)),
        max_rel=float(np.max(diff[nonzero] / scale[nonzero])) if bool(nonzero.any()) else 0.0,
        worst_row=worst // diff.shape[1],
        worst_bank=bank.bank,
        coefficient_max_abs=float(np.max(coefficient)),
        coefficient_max_rel=float(np.max(coefficient[reference_scale] / np.abs(wr)[reference_scale]))
        if bool(reference_scale.any())
        else 0.0,
        coefficient_fro_rel=_frobenius(wc - wr) / max(_frobenius(wc), _frobenius(wr), np.finfo(np.float64).tiny),
        passed=bool(np.all(diff <= bound)),
    )


# --- fits -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitSummary:
    """One of the study's fits with its diagnostics; the complete fit report stays in the cache."""

    configuration: str
    arm: str
    identity: str
    """The study manifest's own fit identity, which already binds the environment, the arm, and the datasets."""
    cache_hit: bool
    solver_alpha: float
    """``K alpha_0``; the episode count and ``alpha_0`` themselves are the manifest's."""
    loss_rows: int
    loss_rows_expected: int
    rmse: float
    weighting_matches_accounting: bool
    """The fit's per-episode rows and weights are the ones the study manifest's accounting rebuilds."""
    training_sources_expected: bool
    """The fit trained on exactly the arm's datasets; the other demonstrations stayed probe-only."""
    transform_isolated: bool
    """The input transform was copied from the frozen scripted dataset, never from a take (I8)."""
    weights_sha256: str
    states_sha256: str
    accessor_max_abs_diff: float
    """Largest ``|predict(x) - (x W[:-1] + W[-1])|`` over the matched probes."""
    normal: NormalEquations
    fit_seconds: float

    def __post_init__(self) -> None:
        """The row count is the arm's, and the identities are digests."""
        if self.loss_rows != self.loss_rows_expected:
            msg = (
                f"{self.configuration}/{self.arm}: {self.loss_rows} loss rows, expected {self.loss_rows_expected} "
                "from the study manifest's accounting"
            )
            raise ValueError(msg)
        for name in ("identity", "weights_sha256", "states_sha256"):
            if not is_hex(getattr(self, name), SHA256_HEX_LENGTH):
                msg = f"{name} must be 64 lowercase hex characters, got {getattr(self, name)!r}"
                raise ValueError(msg)

    @property
    def label(self) -> str:
        """``<configuration>/<arm>``."""
        return f"{self.configuration}/{self.arm}"


def _summary(
    entry: StudyModel,
    cached: CachedFit,
    probes: ProbeMatrix,
    prediction: NDArray[np.float64],
    inputs: ManualFitInputs,
) -> FitSummary:
    """Diagnose one fit: the weighted ridge system, the accessor, and the construction invariants."""
    record, recipe = cached.record, cached.recipe
    accounting = inputs.accounting(entry)
    harvested = [harvest_episode(cached.model, episode) for episode in cached.episodes]
    weights = cast("tuple[float, ...]", recipe.fit.episode_weights)
    states = np.vstack([h.training_states for h in harvested])
    targets = np.vstack([h.training_targets for h in harvested])
    row_weights = np.concatenate(
        [np.full(int(np.count_nonzero(h.loss_rows)), w) for h, w in zip(harvested, weights, strict=True)]
    )
    if states_witness(episode_identities(cached.model, cached.episodes, weights)) != record.states_sha256:
        msg = f"{entry.label}: harvested states differ from the cached fit's record"
        raise ValueError(msg)
    trained = tuple(dataset.artifact_id for dataset in recipe.datasets)
    source = recipe.transform_source
    demonstrations = {record_.dataset.artifact_id for record_ in inputs.manifest.demonstrations}
    linear = probes.matched @ cached.weights[:-1] + cached.weights[-1]
    if record.execution_identity != inputs.execution_identity:
        msg = (
            f"{entry.label}: the cached fit binds environment {record.execution_identity[:_SHORT]}, not the run's "
            f"{inputs.execution_identity[:_SHORT]} (C10)"
        )
        raise ValueError(msg)
    return FitSummary(
        configuration=entry.configuration,
        arm=entry.arm.label,
        identity=record.identity,
        cache_hit=cached.cache_hit,
        solver_alpha=recipe.solver_alpha,
        loss_rows=recipe.fit.loss_rows,
        loss_rows_expected=sum(accounting.loss_rows),
        rmse=recipe.fit.rmse,
        weighting_matches_accounting=(
            recipe.fit.episode_loss_rows == accounting.loss_rows and weights == accounting.row_weights
        ),
        training_sources_expected=trained == tuple(d.artifact_id for d in inputs.manifest.datasets(entry)),
        transform_isolated=(
            source is not None
            and recipe.transform.derived_from == (source.artifact_id,)
            and source.artifact_id not in set(trained) | demonstrations
        ),
        weights_sha256=record.weights_sha256,
        states_sha256=record.states_sha256,
        accessor_max_abs_diff=float(np.max(np.abs(prediction - linear))),
        normal=weighted_normal_equations(states, targets, row_weights, cached.weights, alpha=recipe.solver_alpha),
        fit_seconds=record.fit_seconds,
    )


# --- fresh refits ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreshRefit:
    """The outcome of refitting one cached fit in a fresh, pinned process that bypassed the cache."""

    identity: str
    worker_execution_identity: str | None
    """The worker's environment when it differed from the parent's; ``None`` means it was the parent's own."""
    environment_match: bool
    weights_bitwise_equal: bool
    max_abs_weight_diff: float
    states_bitwise_equal: bool
    fit_report_equal: bool
    rmse_abs_diff: float
    seconds: float
    passed: bool

    def __post_init__(self) -> None:
        """The decision re-derives from the checks."""
        expected = (
            self.environment_match
            and self.weights_bitwise_equal
            and self.states_bitwise_equal
            and self.fit_report_equal
        )
        if self.passed != expected:
            msg = "passed contradicts the checks"
            raise ValueError(msg)
        if self.worker_execution_identity is not None and not is_hex(self.worker_execution_identity, SHA256_HEX_LENGTH):
            msg = (
                f"worker_execution_identity must be 64 lowercase hex characters, got {self.worker_execution_identity!r}"
            )
            raise ValueError(msg)


def fresh_refit(
    identity: str, *, store: StorageRoot, root: Path, parent_identity: str, execution: ExecutionRecord
) -> FreshRefit:
    """Refit the cached fit ``identity`` from its recipe and payloads only, then compare with the cache record.

    The recipe TOML is read from the cache directory; the weights are computed
    anew (never read before the comparison); the datasets come from their Git
    records and the digest-verified payloads. A foreign environment is reported
    as a failed refit, never raised.
    """
    fits = ManualFitStore(store)
    started = time.perf_counter()
    recipe = load_recipe(fits.directory(identity) / RECIPE_FILE)
    recipe.require_rclib()
    samples = load_training_samples(recipe, store, records_root=root)
    validation = recipe.validation
    scenario = None if validation is None else load_manual_scenario(root / validation.scenario_file)
    model = recipe.build_model()
    episodes = tuple(recipe.episodes(samples, scenario=scenario))
    report = train_readout(model, episodes, weight_reference_rows=recipe.training.weight_reference_rows)
    weights = model.readout_weights()
    identities = episode_identities(model, episodes, cast("tuple[float, ...]", report.episode_weights))
    elapsed = time.perf_counter() - started
    record = fits.read_record(identity)
    cached = fits.read_weights(record)
    same_shape = cached.shape == weights.shape
    environment_match = execution.identity == parent_identity == record.execution_identity
    weights_equal = same_shape and array_digest(weights) == record.weights_sha256
    states_equal = states_witness(identities) == record.states_sha256
    report_equal = report == record.fit
    return FreshRefit(
        identity=identity,
        worker_execution_identity=None if execution.identity == parent_identity else execution.identity,
        environment_match=environment_match,
        weights_bitwise_equal=weights_equal,
        max_abs_weight_diff=float(np.max(np.abs(cached - weights))) if same_shape else float("inf"),
        states_bitwise_equal=states_equal,
        fit_report_equal=report_equal,
        rmse_abs_diff=abs(report.rmse - record.fit.rmse),
        seconds=elapsed,
        passed=environment_match and weights_equal and states_equal and report_equal,
    )


def refit_in_subprocess(
    identity: str,
    *,
    root: Path,
    parent_identity: str,
    output: Path,
    python: str = sys.executable,
    env: Mapping[str, str] | None = None,
) -> FreshRefit:
    """Run :func:`worker_main` in a fresh interpreter (inheriting the pinned environment) and read its result."""
    command = [
        python,
        "-m",
        _WORKER_MODULE,
        "refit-worker",
        "--fit",
        identity,
        "--root",
        str(root),
        "--parent-identity",
        parent_identity,
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, env=None if env is None else dict(env))
    return from_mapping(cast("dict[str, object]", json.loads(output.read_text(encoding="utf-8"))), FreshRefit)


type Refitter = Callable[[str, str], FreshRefit]
"""``(fit identity, parent execution identity) -> FreshRefit``."""


# --- the study context ----------------------------------------------------------------------


@dataclass(frozen=True)
class ManualStudyContext:
    """The frozen study with its digest-verified sources and payloads, ready to fit."""

    manifest: StudyManifest
    manifest_sha256: str
    inputs: ManualFitInputs
    payloads: tuple[ArtifactReference, ...]

    @classmethod
    def load(
        cls, manifest_file: Path, *, store: StorageRoot, root: Path, execution: ExecutionRecord
    ) -> ManualStudyContext:
        """Load the manifest and verify every source file, every payload, and the execution environment."""
        manifest = load_study(manifest_file)
        for name in ("panel", "bank", "scenario", "preprocessing", "model"):
            source = getattr(manifest, name)
            actual = sha256_file(root / source.path)
            if actual != source.sha256:
                msg = (
                    f"{source.path} has digest {actual[:_SHORT]}, but the study manifest recorded "
                    f"{source.sha256[:_SHORT]}"
                )
                raise ValueError(msg)
        scenario = load_manual_scenario(root / manifest.scenario.path)
        samples, payloads, records = _load_demonstrations(manifest, store=store, root=root)
        first = records[0]
        shape = (first.preprocessing, first.dof, first.task_code_dim)
        if any((r.preprocessing, r.dof, r.task_code_dim) != shape for r in records):
            msg = "the locked demonstrations were not all derived with the same preprocessing and widths"
            raise ValueError(msg)
        _check_contractive_banks(manifest, samples, records, scenario)
        inputs = ManualFitInputs(
            manifest=manifest,
            samples=samples,
            dof=first.dof,
            task_code_dim=first.task_code_dim,
            preprocessing=first.preprocessing,
            scenario=scenario,
            root=root,
            execution_identity=execution.identity,
            rclib=RclibIdentity.current(),
        )
        return cls(manifest, sha256_file(manifest_file), inputs, payloads)


def _load_demonstrations(
    manifest: StudyManifest, *, store: StorageRoot, root: Path
) -> tuple[dict[str, SampleSet], tuple[ArtifactReference, ...], list[ManualDatasetRecord]]:
    """Every locked demonstration's committed record and digest-verified payload, in bank order."""
    samples: dict[str, SampleSet] = {}
    payloads: list[ArtifactReference] = []
    records: list[ManualDatasetRecord] = []
    for demonstration in manifest.demonstrations:
        dataset = demonstration.dataset
        record = load_record(root / dataset.record, ManualDatasetRecord)
        artifact = record.artifact
        if artifact.artifact_id != dataset.artifact_id or artifact.payload.sha256 != dataset.payload_sha256:
            msg = (
                f"{dataset.record} describes {artifact.artifact_id} ({artifact.payload.sha256[:_SHORT]}), not the "
                f"study's {dataset.artifact_id} ({dataset.payload_sha256[:_SHORT]})"
            )
            raise ValueError(msg)
        loaded = load_samples(verify_payload(store, artifact))
        record.check_samples(loaded)
        if loaded.n_samples - 1 != demonstration.loss_rows:
            msg = (
                f"{demonstration.assignment}: the payload contributes {loaded.n_samples - 1} loss rows, but the study "
                f"manifest recorded {demonstration.loss_rows}"
            )
            raise ValueError(msg)
        samples[dataset.artifact_id] = loaded
        payloads.append(ArtifactReference(artifact.payload.uri, artifact.payload.sha256, artifact.payload.size))
        records.append(record)
    return samples, tuple(payloads), records


def _check_contractive_banks(
    manifest: StudyManifest,
    samples: Mapping[str, SampleSet],
    records: Sequence[ManualDatasetRecord],
    scenario: TaskGeometry,
) -> None:
    """Grow every contractive bank the study binds again and refuse a recorded digest that does not re-derive.

    Each digest is recorded once per parent and hashed into that parent's six
    contractive fit identities, but nothing else in the study re-derives it:
    the numerical arms are the singleton and the duplication control
    (:data:`NUMERICAL_ARMS`), so no contractive arm is ever fitted and a check
    on the fit path would never run. The bank is therefore regenerated here
    from the parent's committed record, exactly as M3MAN-006 grew it, before
    any fit of the study is served.
    """
    recorded: dict[str, list[tuple[str, ContractiveBank]]] = {}
    for model in manifest.entries:
        construction = model.contractive
        if construction is not None:
            recorded.setdefault(construction.assignment, []).append((model.label, construction))
    banks: dict[str, ContractiveBank] = {}
    for assignment, entries in sorted(recorded.items()):
        first, bank = entries[0]
        differing = [label for label, other in entries[1:] if other != bank]
        if differing:
            msg = (
                f"{assignment}: {first} records the contractive construction {bank}, which {differing} do not; "
                "every configuration of a demonstration binds the same bank"
            )
            raise ValueError(msg)
        banks[assignment] = bank
    by_position = {
        demonstration.assignment: (demonstration, record)
        for demonstration, record in zip(manifest.demonstrations, records, strict=True)
    }
    for assignment, bank in sorted(banks.items()):
        demonstration, record = by_position[assignment]
        parent = ManualParent(
            assignment=assignment,
            dataset=demonstration.dataset,
            dwell_start_s=record.motion.dwell_start_s,
            n_samples=record.n_samples,
            period_s=record.preprocessing.resample_period_s,
            derivative_method=record.preprocessing.derivative_method,
            q_sha256=record.arrays["q"].sha256,
            dq_sha256=record.arrays["dq"].sha256,
        )
        outcome = generate_parent_bank(
            parent, samples[demonstration.dataset.artifact_id], scenario, seed_bank=manifest.seed_bank
        )
        if isinstance(outcome, ParentFailure):
            msg = (
                f"{assignment}: the contractive bank the study records ({bank.bank_sha256}) cannot be grown again "
                f"from {parent.identifier}: {outcome.reason}"
            )
            raise ValueError(msg)  # noqa: TRY004 - a bank that cannot be grown is a study error
        regenerated = ContractiveBank.from_record(outcome.record)
        if regenerated != bank:
            msg = (
                f"{assignment}: the study records the contractive construction {bank}, but growing the bank again "
                f"from {parent.identifier} gives {regenerated}; the contractive fit identities bind a construction "
                "this study cannot reproduce. The whole construction is compared, not only its digest, because the "
                "seed bank and the dwell onset decide which episodes the arm trains on"
            )
            raise ValueError(msg)


# --- the validation -------------------------------------------------------------------------


@dataclass(frozen=True)
class ManualNumericalValidation:
    """The committed evidence of the manual study's numerical copy controls."""

    experiment: str
    study_manifest_file: str
    study_manifest_sha256: str
    arms: tuple[str, ...]
    configurations: tuple[str, ...]
    demonstrations: tuple[str, ...]
    payloads: tuple[ArtifactReference, ...]
    execution: ExecutionRecord
    rclib: RclibIdentity
    tolerances: Tolerances
    probes: dict[str, tuple[ProbeBank, ...]]
    """Per configuration: the banks of the probe matrix its arms share."""
    probe_states_identical_across_arms: dict[str, bool]
    """Per configuration: every arm harvested the matched manual block bitwise identically."""
    fits: tuple[FitSummary, ...]
    comparisons: tuple[Comparison, ...]
    repeat_identities: tuple[RepeatIdentity, ...]
    fresh_refits: tuple[FreshRefit, ...]
    n_fits: int
    n_comparisons: int
    n_comparisons_passed: int
    n_residuals_within: int
    n_weighting_matches: int
    n_repeat_identities_identical: int
    n_fresh_refits_passed: int
    all_passed: bool
    provenance: ProvenanceRecord
    schema_version: int = field(default=NUMERICS_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Every count and the overall decision re-derive from the retained records."""
        if self.schema_version != NUMERICS_SCHEMA_VERSION:
            msg = f"unsupported numerics schema_version {self.schema_version}"
            raise ValueError(msg)
        counts = (
            (self.n_fits, len(self.fits)),
            (self.n_comparisons, len(self.comparisons)),
            (self.n_comparisons_passed, sum(1 for c in self.comparisons if c.passed)),
            (self.n_residuals_within, sum(1 for f in self.fits if f.normal.within)),
            (self.n_weighting_matches, sum(1 for f in self.fits if f.weighting_matches_accounting)),
            (self.n_repeat_identities_identical, sum(1 for r in self.repeat_identities if r.identical)),
            (self.n_fresh_refits_passed, sum(1 for r in self.fresh_refits if r.passed)),
        )
        if any(recorded != actual for recorded, actual in counts):
            msg = f"recorded counts {counts} contradict the retained records"
            raise ValueError(msg)
        if len(self.fresh_refits) != len(self.fits) or {r.identity for r in self.fresh_refits} != {
            f.identity for f in self.fits
        }:
            msg = "every fit has exactly one fresh-process refit"
            raise ValueError(msg)
        if self.all_passed != self._expected():
            msg = "all_passed contradicts the retained records"
            raise ValueError(msg)
        foreign = {r.worker_execution_identity for r in self.fresh_refits if r.worker_execution_identity is not None}
        if self.execution.identity in foreign:
            msg = "a refit recorded a foreign worker environment that is the validation's own"
            raise ValueError(msg)
        if not is_hex(self.study_manifest_sha256, SHA256_HEX_LENGTH):
            msg = "study_manifest_sha256 must be 64 lowercase hex characters"
            raise ValueError(msg)

    def _expected(self) -> bool:
        return (
            self.n_comparisons_passed == self.n_comparisons
            and self.n_residuals_within == self.n_fits
            and self.n_weighting_matches == self.n_fits
            and self.n_repeat_identities_identical == len(self.repeat_identities)
            and self.n_fresh_refits_passed == self.n_fits
            and all(f.training_sources_expected and f.transform_isolated for f in self.fits)
            and all(self.probe_states_identical_across_arms.values())
        )


def run_validation(
    entries: Sequence[StudyModel],
    inputs: ManualFitInputs,
    *,
    store: StorageRoot,
    execution: ExecutionRecord,
    refit: Refitter,
    manifest_file: str,
    manifest_sha256: str,
    provenance: ProvenanceRecord,
    payloads: tuple[ArtifactReference, ...] = (),
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> ManualNumericalValidation:
    """Fit, probe, compare, diagnose, and refit every entry; failures are retained, never repaired."""
    execution.check_canonical()
    fits = ManualFitStore(store)
    summaries: list[FitSummary] = []
    comparisons: list[Comparison] = []
    repeats: list[RepeatIdentity] = []
    refits: list[FreshRefit] = []
    probe_banks: dict[str, tuple[ProbeBank, ...]] = {}
    probe_identity: dict[str, bool] = {}
    order = [c.label for c in inputs.manifest.configurations if any(e.configuration == c.label for e in entries)]
    for configuration in order:
        here = [entry for entry in entries if entry.configuration == configuration]
        log(f"{configuration}: fitting {len(here)} arms")
        cached: dict[str, CachedFit] = {}
        predictions: dict[str, NDArray[np.float64]] = {}
        for entry in here:
            fit = fits.fit_or_load(entry, inputs, now=now)
            cached[entry.arm.label] = fit
            probes = build_probes(fit.model, fit.recipe.encoder(), entry, inputs)
            matched = probes.banks[0]
            if configuration not in probe_banks:
                probe_banks[configuration] = probes.banks
                probe_identity[configuration] = True
            elif matched.states_sha256 != probe_banks[configuration][0].states_sha256:
                probe_identity[configuration] = False
            prediction = predict_probes(fit.model, probes)[: matched.rows]
            predictions[entry.arm.label] = prediction
            summaries.append(_summary(entry, fit, probes, prediction, inputs))
            repeat = repeat_identity(fit.record)
            if repeat is not None:
                repeats.append(repeat)
        comparisons.extend(_comparisons(here, cached, predictions, probe_banks[configuration][0]))
        log(f"{configuration}: refitting {len(here)} fits in fresh processes")
        refits.extend(refit(cached[entry.arm.label].record.identity, execution.identity) for entry in here)
    return ManualNumericalValidation(
        experiment=EXPERIMENT_LABEL,
        study_manifest_file=manifest_file,
        study_manifest_sha256=manifest_sha256,
        arms=tuple(dict.fromkeys(entry.arm.arm for entry in entries)),
        configurations=tuple(order),
        demonstrations=ASSIGNMENTS,
        payloads=payloads,
        execution=execution,
        rclib=inputs.rclib,
        tolerances=TOLERANCES,
        probes=probe_banks,
        probe_states_identical_across_arms=probe_identity,
        fits=tuple(summaries),
        comparisons=tuple(comparisons),
        repeat_identities=tuple(repeats),
        fresh_refits=tuple(refits),
        n_fits=len(summaries),
        n_comparisons=len(comparisons),
        n_comparisons_passed=sum(1 for c in comparisons if c.passed),
        n_residuals_within=sum(1 for f in summaries if f.normal.within),
        n_weighting_matches=sum(1 for f in summaries if f.weighting_matches_accounting),
        n_repeat_identities_identical=sum(1 for r in repeats if r.identical),
        n_fresh_refits_passed=sum(1 for r in refits if r.passed),
        all_passed=(
            all(c.passed for c in comparisons)
            and all(f.normal.within for f in summaries)
            and all(f.weighting_matches_accounting for f in summaries)
            and all(f.training_sources_expected and f.transform_isolated for f in summaries)
            and all(r.identical for r in repeats)
            and all(r.passed for r in refits)
            and all(probe_identity.values())
        ),
        provenance=provenance,
    )


def _comparisons(
    entries: Sequence[StudyModel],
    cached: Mapping[str, CachedFit],
    predictions: Mapping[str, NDArray[np.float64]],
    bank: ProbeBank,
) -> list[Comparison]:
    """Every approved pair among ``entries``: the duplication control against the singleton of one demonstration."""
    candidate_arm, reference_arm = COMPARISON_PAIR
    found: list[Comparison] = []
    for entry in entries:
        if entry.arm.arm != candidate_arm:
            continue
        reference = f"{reference_arm}/{entry.arm.assignment}"
        if reference not in cached:
            continue
        found.append(
            compare_fits(
                cached[entry.arm.label],
                cached[reference],
                bank,
                predictions[entry.arm.label],
                predictions[reference],
            )
        )
    return found


def validation_to_json(validation: ManualNumericalValidation) -> str:
    """Canonical JSON of the evidence."""
    return canonical_json(to_mapping(validation))


def load_validation(path: Path) -> ManualNumericalValidation:
    """Strictly rebuild the evidence from JSON (re-deriving every count and decision)."""
    return from_mapping(
        cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ManualNumericalValidation
    )


# --- rendering ------------------------------------------------------------------------------


def _e(value: float) -> str:
    return f"{value:.3e}"


def _flag(value: bool) -> str:  # noqa: FBT001 - a recorded decision is rendered, not a switch
    return "pass" if value else "**FAIL**"


def _mark(value: bool, yes: str, no: str) -> str:  # noqa: FBT001 - a recorded decision is rendered, not a switch
    return yes if value else no


def _header_lines(v: ManualNumericalValidation) -> list[str]:
    t = v.tolerances
    return [
        "# Task 1-a manual numerical copy controls (v1)",
        "",
        (
            f"Experiment `{v.experiment}`: the duplication controls of the frozen study "
            f"`{v.study_manifest_file}` (sha256 `{v.study_manifest_sha256[:_SHORT]}`), rclib "
            f"`{v.rclib.version}` at `{v.rclib.commit[:_SHORT]}`, project commit "
            f"`{v.provenance.project_commit[:_SHORT]}`{' (dirty)' if v.provenance.project_dirty else ''}."
        ),
        "",
        "## Outcome",
        "",
        f"- Overall: **{'all checks passed' if v.all_passed else 'FAILURES RETAINED'}**.",
        (
            f"- Fits: {v.n_fits} ({sum(1 for f in v.fits if f.cache_hit)} served from the cache) over "
            f"{len(v.configurations)} configuration(s), arms {', '.join(v.arms)}."
        ),
        (
            f"- Equivalence comparisons ({COMPARISON_PAIR[0]}_i against {COMPARISON_PAIR[1]}_i, absolute output): "
            f"{v.n_comparisons_passed} of {v.n_comparisons} within atol {t.prediction_atol_rad:g} rad and rtol "
            f"{t.prediction_rtol:g}."
        ),
        f"- Weighted normal-equation residuals: {v.n_residuals_within} of {v.n_fits} at or below {t.residual_max:g}.",
        (
            f"- Per-episode rows and weights rebuilt from the study's accounting: {v.n_weighting_matches} of "
            f"{v.n_fits} fits."
        ),
        (
            f"- Literal repetition (inputs, targets, weights, masks, warm-up, harvested features): "
            f"{v.n_repeat_identities_identical} of {len(v.repeat_identities)} duplication controls identical."
        ),
        (
            f"- Probe isolation: {sum(1 for f in v.fits if f.training_sources_expected)} of {v.n_fits} fits trained "
            f"exactly their arm's datasets and {sum(1 for f in v.fits if f.transform_isolated)} of {v.n_fits} copied "
            "the transform from outside the takes."
        ),
        (
            f"- Fresh-process refits: {v.n_fresh_refits_passed} of {v.n_fits} reproduced weights, states, and report "
            "bitwise."
        ),
        "",
        "## Execution environment",
        "",
        (
            f"- Identity `{v.execution.identity[:_SHORT]}` "
            f"({'canonical' if v.execution.canonical else 'NOT canonical'}): policy `{v.execution.policy}`, "
            f"{len(v.execution.effective_cpus)} logical CPUs, BLAS `{v.execution.blas.corename}`, OpenMP max threads "
            f"{v.execution.openmp.max_threads}."
        ),
        (
            "- Every fit and every worker refit bound this identity; a worker whose environment differed is recorded "
            "as a failure, never merged."
        ),
    ]


def render_validation_markdown(validation: ManualNumericalValidation) -> str:
    """The Markdown rendering of the evidence (every fit, comparison, and refit; failures retained)."""
    v = validation
    lines = [
        *_header_lines(v),
        "",
        "## Probes",
        "",
        (
            "Per configuration: the task rows of all ten locked demonstrations harvested from a reset reservoir "
            "after the configuration's warm-up, plus a fit's own contractive bank where it has one. Probe-only "
            "trajectories never enter a fit or the frozen input transform."
        ),
        "",
        "| configuration | bank | episodes | rows | states sha256 | identical across arms |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for label in v.configurations:
        lines.extend(
            f"| {label} | {b.bank} | {b.episodes} | {b.rows} | `{b.states_sha256[:_SHORT]}` "
            f"| {_mark(v.probe_states_identical_across_arms[label], 'yes', '**NO**')} |"
            for b in v.probes[label]
        )
    lines += [
        "",
        "## Equivalence comparisons",
        "",
        (
            "| configuration | demonstration | candidate | reference | rows | max abs | max rel | worst bank "
            "| coef max abs | coef max rel | coef fro rel | decision |"
        ),
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {c.configuration} | {c.assignment} | {c.candidate} | {c.reference} | {c.rows} | {_e(c.max_abs)} "
        f"| {_e(c.max_rel)} | {c.worst_bank} | {_e(c.coefficient_max_abs)} | {_e(c.coefficient_max_rel)} "
        f"| {_e(c.coefficient_fro_rel)} | {_flag(c.passed)} |"
        for c in v.comparisons
    )
    lines += [
        "",
        "## Fits",
        "",
        (
            "Every fit with its ridge scale, loss rows, teacher-forced error, the normalized residual and "
            "conditioning of the weighted system A = Xw^T Xw + alpha I (explicit bias column last), the accessor "
            "check over the matched probes, and its construction invariants."
        ),
        "",
        (
            "| configuration | arm | alpha | loss rows | rmse | residual | cond2 | accessor | weighting "
            "| sources | transform | fit s | cache |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    lines.extend(
        f"| {f.configuration} | {f.arm} | {f.solver_alpha:.6g} | {f.loss_rows} | {_e(f.rmse)} "
        f"| {_e(f.normal.residual)} | {_e(f.normal.cond2)} | {_e(f.accessor_max_abs_diff)} "
        f"| {_flag(f.weighting_matches_accounting)} | {_flag(f.training_sources_expected)} "
        f"| {_flag(f.transform_isolated)} | {f.fit_seconds:.2f} | {'hit' if f.cache_hit else 'fit'} |"
        for f in v.fits
    )
    lines += [
        "",
        "## Literal repetition",
        "",
        "| configuration | arm | episodes | inputs | targets | weights | masks | warm-up | features | decision |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {r.configuration} | {r.arm} | {r.episodes} | {_mark(r.inputs_identical, 'same', '**DIFFER**')} "
        f"| {_mark(r.targets_identical, 'same', '**DIFFER**')} | {_mark(r.weights_identical, 'same', '**DIFFER**')} "
        f"| {_mark(r.masks_identical, 'same', '**DIFFER**')} | {_mark(r.warmup_identical, 'same', '**DIFFER**')} "
        f"| {_mark(r.states_identical, 'same', '**DIFFER**')} | {_flag(r.identical)} |"
        for r in v.repeat_identities
    )
    lines += [
        "",
        "## Fresh-process refits",
        "",
        "| identity | environment | weights | max abs weight diff | states | fit report | rmse diff | s | decision |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| `{r.identity[:_SHORT]}` | {_mark(r.environment_match, 'same', '**DIFFERENT**')} "
        f"| {_mark(r.weights_bitwise_equal, 'bitwise', '**DIFFER**')} | {_e(r.max_abs_weight_diff)} "
        f"| {_mark(r.states_bitwise_equal, 'bitwise', '**DIFFER**')} "
        f"| {_mark(r.fit_report_equal, 'equal', '**DIFFER**')} | {_e(r.rmse_abs_diff)} | {r.seconds:.2f} "
        f"| {_flag(r.passed)} |"
        for r in v.fresh_refits
    )
    lines += [
        "",
        "## Limitations",
        "",
        (
            "- The prediction tolerance bounds teacher-forced readout differences on fixed probes; it is not a "
            "closed-loop stability guarantee (manual plan section 5)."
        ),
        (
            "- The equivalence is a numerical identity of the weighted ridge problem; agreement here says nothing "
            "about whether ten demonstrations help, which the behavioral evaluation decides."
        ),
        (
            "- Fresh-process bitwise reproducibility holds in the recorded execution environment only; another core "
            "type, thread setting, library build, or machine is a different environment (C10)."
        ),
        (
            "- The residual is evaluated with numpy on the harvested rows; the solve itself is rclib's LDLT on its "
            "own accumulation of the same scaled rows."
        ),
        (
            "- The predecessor's accepted numerical exception is not carried over: a failure here is retained and "
            "must be diagnosed before any behavioral interpretation."
        ),
        "",
    ]
    return "\n".join(lines)


# --- command line ---------------------------------------------------------------------------


def _load_runtimes() -> None:
    """Load every numerical runtime this command uses before the execution environment is probed.

    ``skelarm`` belongs here beside ``numpy`` and ``rclib``: the manual dataset
    schema imports it, so a process that probed without it would record a
    different BLAS/OpenMP environment than one that probed with it, and the
    parent and its workers would never agree on an execution identity.
    """
    for name in ("numpy", "rclib", "skelarm"):
        importlib.import_module(name)


def worker_main(args: argparse.Namespace) -> int:
    """The fresh-process refit worker: verify the pinned environment, refit, compare, write the result."""
    require_canonical()
    ensure_single_thread()
    _load_runtimes()
    execution = collect_execution(
        command=command_line(_WORKER_MODULE, cast("list[str]", args.argv)), role="worker", now=datetime.now(tz=UTC)
    )
    result = fresh_refit(
        cast("str", args.fit),
        store=open_storage(),
        root=Path(cast("str", args.root)),
        parent_identity=cast("str", args.parent_identity),
        execution=execution,
    )
    output = Path(cast("str", args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(to_mapping(result)) + "\n", encoding="utf-8")
    return 0


def _validate(args: argparse.Namespace) -> int:
    require_canonical()
    ensure_single_thread()
    _load_runtimes()
    output, markdown = Path(cast("str", args.output)), Path(cast("str", args.markdown))
    for target in (output, markdown):
        if target.exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    root = repository_root()
    store = open_storage()
    manifest_file = Path(cast("str", args.study))
    execution = collect_execution(
        command=command_line(_WORKER_MODULE, cast("list[str]", args.argv)), role="main", now=datetime.now(tz=UTC)
    )
    execution.check_canonical()
    context = ManualStudyContext.load(manifest_file, store=store, root=root, execution=execution)
    entries = numerical_entries(context.manifest)
    resolved: dict[str, object] = {
        "study_manifest": context.manifest_sha256,
        "tolerances": to_mapping(TOLERANCES),
        "arms": list(NUMERICAL_ARMS),
        "models": len(entries),
        "execution_identity": execution.identity,
        "command": command_line(_WORKER_MODULE, cast("list[str]", args.argv)),
    }
    provenance = collect_provenance(
        resolved,
        seeds={"contractive_seed_bank": context.manifest.seed_bank},
        artifacts=context.payloads,
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
    )
    require_clean_for_confirmatory(provenance)
    workspace = Path(cast("str", args.workspace))
    workspace.mkdir(parents=True, exist_ok=True)

    def refit(identity: str, parent_identity: str) -> FreshRefit:
        return refit_in_subprocess(
            identity, root=root, parent_identity=parent_identity, output=workspace / f"{identity}.json"
        )

    validation = run_validation(
        entries,
        context.inputs,
        store=store,
        execution=execution,
        refit=refit,
        manifest_file=manifest_file.relative_to(root).as_posix()
        if manifest_file.is_relative_to(root)
        else manifest_file.name,
        manifest_sha256=context.manifest_sha256,
        provenance=provenance,
        payloads=context.payloads,
    )
    _write_evidence(validation, output=output, markdown=markdown)
    return 0


def _write_evidence(validation: ManualNumericalValidation, *, output: Path, markdown: Path) -> None:
    """Write both evidence files, then refuse silently exceeding the repository's committable size."""
    text = validation_to_json(validation) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    markdown.write_text(render_validation_markdown(validation), encoding="utf-8")
    print(
        json.dumps(
            {
                "all_passed": validation.all_passed,
                "fits": validation.n_fits,
                "comparisons_passed": f"{validation.n_comparisons_passed}/{validation.n_comparisons}",
                "residuals_within": f"{validation.n_residuals_within}/{validation.n_fits}",
                "fresh_refits_passed": f"{validation.n_fresh_refits_passed}/{validation.n_fits}",
                "output": str(output),
            },
            indent=2,
        )
    )
    size = len(text.encode("utf-8"))
    if size > LARGE_FILE_LIMIT_BYTES:
        msg = (
            f"{output} is {size} bytes, above the repository's {LARGE_FILE_LIMIT_BYTES}-byte limit for committed "
            "files; the evidence was written but cannot be committed as it stands"
        )
        raise ValueError(msg)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Numerical copy controls of the manual-demonstration study.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    validate = subparsers.add_parser("validate", help="fit, probe, compare, diagnose, and refit the copy controls")
    validate.add_argument("--study", type=str, required=True, help="frozen study manifest JSON")
    validate.add_argument("--output", type=str, required=True, help="evidence JSON to write (must not exist)")
    validate.add_argument("--markdown", type=str, required=True, help="evidence Markdown to write (must not exist)")
    validate.add_argument(
        "--workspace", type=str, required=True, help="directory for the worker result files (outside the worktree)"
    )
    validate.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    worker = subparsers.add_parser("refit-worker", help="refit one cached fit in this fresh process")
    worker.add_argument("--fit", type=str, required=True, help="fit identity (64 hex)")
    worker.add_argument("--root", type=str, required=True, help="repository root holding the records and scenario")
    worker.add_argument("--parent-identity", type=str, required=True, help="the parent's execution identity")
    worker.add_argument("--output", type=str, required=True, help="result JSON to write")
    args = parser.parse_args(argv)
    args.argv = argv
    if args.subcommand == "validate":
        return _validate(args)
    return worker_main(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
