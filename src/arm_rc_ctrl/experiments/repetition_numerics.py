# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Numerical validation of the repeated-demonstration pilot (M3REP-003; repetition plan section 6, D6, C1, C10).

Before any behavioral sweep, the pilot's ridge equivalences are checked on the
frozen panel with the actual fitted weights:

1. literal repetition: every copy of the original episode is harvested into a
   bitwise identical state array, and matched absolute/residual recipes with
   identical inputs and reservoirs harvest identical states;
2. for every entry, formulation, and count ``K``, ``R`` (``K`` copies at
   ``alpha_0``) is compared with ``S-effective`` (the original at
   ``alpha_0 / K``) and ``R-scaled`` (``K`` copies at ``K alpha_0``) with ``S``
   (the original at ``alpha_0``) on one probe matrix per entry: the harvested
   task-row states of the original episode and of both largest fixed augmented
   banks (64 synthetic episodes each, probes only, never training data).
   Residual fits are compared on their increments and on the absolute commands
   reconstructed with identical measured-posture anchors; an increment is
   never compared with an absolute position;
3. the approved tolerance is elementwise ``|a - b| <= atol + rtol |b|`` with
   ``atol = 1e-8`` rad and ``rtol = 1e-8``; maximum absolute and relative
   differences and the decision are recorded for every comparison;
4. every fit reports its coefficient differences to its counterpart, the
   conditioning of its normal matrix, and the normalized normal-equation
   residual ``||A W - B||_F / (||A||_F ||W||_F + ||B||_F)`` with
   ``A = X^T X + alpha I`` and ``B = X^T Y`` over the fit's actual loss rows,
   the bias column last as the pinned solver orders it (``0 / 0 := 0``), which
   must not exceed ``1e-10``;
5. every fit is refitted in a fresh, pinned worker process that bypasses the
   cache, verifies it runs in the parent's execution environment, and compares
   its weights, harvested states, and fit report with the cached fit.

A failed check is retained in the evidence with its figures; no tolerance is
relaxed and no arm is substituted. Every fit binds the execution environment
identity and the ``rclib`` commit of the build that produced it.

Command line (launch pinned through ``python -m arm_rc_ctrl.execution run --policy p-cores -- ...``)::

    python -m arm_rc_ctrl.experiments.repetition_numerics validate
        --manifest <docs>/panel_manifest_v1.json
        --output <docs>/numerical_validation_v1.json --markdown <docs>/numerical_validation_v1.md
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
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.derivatives import DerivativeConfig
from arm_rc_ctrl.data.records import verify_payload
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, load_processed_record, task_intervals_from_phases
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.execution import (
    ExecutionRecord,
    collect_execution,
    require_canonical,
)
from arm_rc_ctrl.experiments.repetition_fits import (
    RECIPE_FILE,
    CachedFit,
    FitInputs,
    FitStore,
    harvested_states,
)
from arm_rc_ctrl.experiments.repetition_panel import EXPERIMENT_LABEL, PanelEntry, PanelManifest, load_panel
from arm_rc_ctrl.experiments.repetition_recipes import (
    AUGMENTATION_ANCHOR,
    FORMULATIONS,
    ArmSpec,
    expected_loss_rows,
)
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
)
from arm_rc_ctrl.rc.augment import generate_augmentation
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.recipe import APPROVED_ADDITIONAL_REPEATS, DatasetSource, RclibIdentity, load_recipe
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.rc.training import FitReport, harvest_episode, train_readout
from arm_rc_ctrl.rc.warmup import WarmupConfig, build_task_episode_arrays
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageRoot, open_storage
from arm_rc_ctrl.validation import is_hex

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.records import Normalization
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.rc.augment import EpisodeArrays
    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.recipe import ModelRecipe
    from arm_rc_ctrl.rc.teacher_forcing import InputEncoder
    from arm_rc_ctrl.scenario import ScenarioConfig

__all__ = [
    "COMPARISON_PAIRS",
    "NUMERICAL_ARMS",
    "NUMERICS_SCHEMA_VERSION",
    "PROBE_BANKS",
    "PROBE_BANK_COUNT",
    "TOLERANCES",
    "Comparison",
    "FitSummary",
    "FreshRefit",
    "NormalEquations",
    "NumericalValidation",
    "PanelContext",
    "ProbeBank",
    "ProbeMatrix",
    "QuantityDifference",
    "StateIdentity",
    "Tolerances",
    "build_probes",
    "compare_fits",
    "fresh_refit",
    "load_recipe_samples",
    "load_validation",
    "main",
    "normal_equations",
    "numerical_arms",
    "predict_probes",
    "refit_in_subprocess",
    "render_validation_markdown",
    "run_validation",
    "validation_to_json",
    "worker_main",
]

NUMERICS_SCHEMA_VERSION: Final = 1
NUMERICAL_ARMS: Final = ("S", "R", "R-scaled", "S-effective")
"""The arms of the equivalence check (the augmented arms are trained in M3REP-004, not here)."""
COMPARISON_PAIRS: Final = (("R", "S-effective"), ("R-scaled", "S"))
"""``(candidate, reference)``: the candidate's predictions are compared against the reference's."""
PROBE_BANK_COUNT: Final = 64
"""Synthetic episodes per augmented probe bank: the largest fixed count (``K = 65``) of the pilot."""
PROBE_BANKS: Final = ("original", "non_decaying", "contractive")
_SHA256_HEX: Final = 64
_DERIVATIVE_METHODS: Final = {"central-difference": "central", "cubic-spline": "spline"}
_WORKER_MODULE: Final = "arm_rc_ctrl.experiments.repetition_numerics"
_COMMAND_MODULE: Final = "arm_rc_ctrl.experiments.repetition_numerics"


@dataclass(frozen=True)
class Tolerances:
    """The approved tolerances (D6); any other value is refused."""

    prediction_atol_rad: float = 1e-8
    prediction_rtol: float = 1e-8
    residual_max: float = 1e-10

    def __post_init__(self) -> None:
        """Only the approved values exist."""
        if (self.prediction_atol_rad, self.prediction_rtol, self.residual_max) != (1e-8, 1e-8, 1e-10):
            msg = "the numerical tolerances are approved under D6 and cannot be changed here"
            raise ValueError(msg)


TOLERANCES: Final = Tolerances()


def numerical_arms(*, counts: tuple[int, ...] = tuple(sorted(APPROVED_ADDITIONAL_REPEATS))) -> tuple[ArmSpec, ...]:
    """The 20 fits per entry: ``S`` plus ``R``, ``R-scaled``, and ``S-effective`` at every count, both formulations."""
    arms: list[ArmSpec] = []
    for formulation in FORMULATIONS:
        arms.append(ArmSpec(formulation, "S"))
        arms.extend(ArmSpec(formulation, arm, repeats) for repeats in counts for arm in NUMERICAL_ARMS if arm != "S")
    return tuple(arms)


# --- panel inputs -------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelContext:
    """The frozen panel with its digest-verified dataset, model, and scenario, ready to fit."""

    manifest: PanelManifest
    manifest_sha256: str
    inputs: FitInputs
    dataset: RecoveryDatasetRecord
    payload: ArtifactReference

    @classmethod
    def load(cls, manifest_file: Path, *, store: StorageRoot, root: Path, execution: ExecutionRecord) -> PanelContext:
        """Load the manifest and verify that every configured source file and payload is still the recorded one."""
        manifest = load_panel(manifest_file)
        configs = manifest.configs
        for file, digest in (
            (configs.model_file, configs.model_sha256),
            (configs.scenario_file, configs.scenario_sha256),
            (configs.dataset_record_file, configs.dataset_record_sha256),
        ):
            actual = sha256_file(root / file)
            if actual != digest:
                msg = f"{file} has digest {actual[:12]}, but the panel manifest recorded {digest[:12]}"
                raise ValueError(msg)
        dataset = load_processed_record(root / configs.dataset_record_file)
        if not isinstance(dataset, RecoveryDatasetRecord):
            msg = f"{configs.dataset_record_file} is not a recovery dataset record"
            raise TypeError(msg)
        if dataset.artifact.artifact_id != configs.dataset or dataset.artifact.payload.sha256 != (
            configs.dataset_payload_sha256
        ):
            msg = f"{configs.dataset_record_file} does not describe the panel's dataset {configs.dataset}"
            raise ValueError(msg)
        if dataset.normalization is None:  # pragma: no cover - the panel's dataset records its statistics
            msg = f"dataset {configs.dataset} records no normalization statistics"
            raise ValueError(msg)
        scenario_file = root / configs.scenario_file
        dataset.check_scenario(scenario_file)
        payload_path = verify_payload(store, dataset.artifact)
        samples = load_samples(payload_path)
        dataset.check_samples(samples)
        inputs = FitInputs(
            base=load_model_config(root / configs.model_file),
            source=DatasetSource(configs.dataset, configs.dataset_payload_sha256, configs.dataset_record_file),
            samples=samples,
            dof=dataset.dof,
            task_code_dim=dataset.task_code_dim,
            preprocessing=dataset.preprocessing,
            normalization=dataset.normalization,
            scenario=load_scenario(scenario_file),
            scenario_file=scenario_file,
            root=root,
            execution_identity=execution.identity,
            rclib=RclibIdentity.current(),
        )
        payload = ArtifactReference(
            dataset.artifact.payload.uri, dataset.artifact.payload.sha256, dataset.artifact.payload.size
        )
        return cls(manifest, sha256_file(manifest_file), inputs, dataset, payload)


def load_recipe_samples(recipe: ModelRecipe, store: StorageRoot, *, root: Path) -> dict[str, SampleSet]:
    """Resolve the recipe's dataset through its Git record (M3 or recovery schema) and the digest-verified payload."""
    samples: dict[str, SampleSet] = {}
    normalizations: dict[str, Normalization] = {}
    for source in recipe.datasets:
        record = load_processed_record(root / source.record)
        recipe.check_dataset_record(source, record)
        loaded = load_samples(verify_payload(store, record.artifact))
        record.check_samples(loaded)
        samples[source.artifact_id] = loaded
        if record.normalization is not None:
            normalizations[source.artifact_id] = record.normalization
    recipe.check_transform_source(normalizations)
    return samples


# --- probes -------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeBank:
    """One bank of probe episodes: where its rows came from and what they were."""

    bank: str
    episodes: int
    rows_per_episode: int
    first_row: int
    """Row of the stacked probe matrix at which this bank starts (banks are stacked in ``PROBE_BANKS`` order)."""
    episode_labels: tuple[str, ...]
    states_sha256: str
    anchors_sha256: str
    attempts_used: int | None = None
    """Augmented banks: attempts the deterministic generator used; ``None`` for the original."""
    rejections: int | None = None

    def __post_init__(self) -> None:
        """Counts agree and digests are well-formed."""
        if self.bank not in PROBE_BANKS or self.episodes < 1 or self.rows_per_episode < 1 or self.first_row < 0:
            msg = f"malformed probe bank {self.bank!r}"
            raise ValueError(msg)
        if len(self.episode_labels) != self.episodes:
            msg = f"bank {self.bank}: {len(self.episode_labels)} labels for {self.episodes} episodes"
            raise ValueError(msg)
        if not is_hex(self.states_sha256, _SHA256_HEX) or not is_hex(self.anchors_sha256, _SHA256_HEX):
            msg = f"bank {self.bank}: digests must be 64 lowercase hex characters"
            raise ValueError(msg)

    @property
    def rows(self) -> int:
        """Rows the bank contributes."""
        return self.episodes * self.rows_per_episode


@dataclass(frozen=True)
class ProbeMatrix:
    """The stacked probe states of one entry with the measured-posture anchor of every row."""

    states: NDArray[np.float64]
    """``(rows, n_neurons)`` harvested task-row states."""
    anchors: NDArray[np.float64]
    """``(rows, dof)`` measured joint posture ``q_k`` of the row (the residual command anchor)."""
    banks: tuple[ProbeBank, ...]

    def bank_of(self, row: int) -> str:
        """The bank a stacked row belongs to."""
        for bank in reversed(self.banks):
            if row >= bank.first_row:
                return bank.bank
        msg = f"row {row} precedes every bank"
        raise ValueError(msg)


type BankArrays = tuple[tuple[str, NDArray[np.float64], NDArray[np.float64]], ...]
"""``(label, q, dq)`` of every episode of one probe bank."""


def _derivatives(method: str) -> DerivativeConfig:
    if method not in _DERIVATIVE_METHODS:
        msg = f"unsupported derivative method {method!r}; supported: {sorted(_DERIVATIVE_METHODS)}"
        raise ValueError(msg)
    return DerivativeConfig(method=cast("Literal['central', 'spline']", _DERIVATIVE_METHODS[method]))


def _probe_arrays(
    samples: SampleSet, scenario: ScenarioConfig, derivative_method: str, *, bank_count: int
) -> tuple[dict[str, BankArrays], dict[str, tuple[int, int]]]:
    """Raw ``(label, q, dq)`` arrays of every probe bank and the generator's attempt/rejection counts."""
    arrays: dict[str, BankArrays] = {"original": (("original", samples.q, samples.dq),)}
    counts: dict[str, tuple[int, int]] = {}
    task = task_intervals_from_phases(samples.t, samples.phase)
    derivatives = _derivatives(derivative_method)
    for family in ("non_decaying", "contractive"):
        config = AUGMENTATION_ANCHOR.spec(family, bank_count).config()
        result = generate_augmentation(samples.t, samples.q, task, scenario, config, derivatives=derivatives)
        bank: list[tuple[str, NDArray[np.float64], NDArray[np.float64]]] = []
        for episode in result.episodes:
            episode_arrays: EpisodeArrays = getattr(episode, family)
            bank.append((f"{family}-{episode.episode:03d}", episode_arrays.q, episode_arrays.dq))
        arrays[family] = tuple(bank)
        counts[family] = (result.attempts_used, len(result.rejections))
    return arrays, counts


def build_probes(
    model: EsnModel,
    encoder: InputEncoder,
    samples: SampleSet,
    *,
    scenario: ScenarioConfig,
    warmup_s: float,
    period_s: float,
    derivative_method: str,
    task_code: NDArray[np.float64],
    bank_count: int = PROBE_BANK_COUNT,
) -> ProbeMatrix:
    """Harvest the probe matrix of one entry through ``model`` (the original plus both augmented banks)."""
    arrays, counts = _probe_arrays(samples, scenario, derivative_method, bank_count=bank_count)
    warmup = WarmupConfig(warmup_s)
    states: list[NDArray[np.float64]] = []
    anchors: list[NDArray[np.float64]] = []
    banks: list[ProbeBank] = []
    first_row = 0
    for name in PROBE_BANKS:
        bank_states: list[NDArray[np.float64]] = []
        bank_anchors: list[NDArray[np.float64]] = []
        labels: list[str] = []
        for label, q, dq in arrays[name]:
            episode = build_task_episode_arrays(
                samples.t, q, dq, task_code, encoder, source=label, warmup=warmup, period_s=period_s
            )
            harvested = harvest_episode(model, episode)
            bank_states.append(harvested.training_states)
            bank_anchors.append(np.asarray(q[:-1], dtype=np.float64))
            labels.append(label)
        stacked_states = np.vstack(bank_states)
        stacked_anchors = np.vstack(bank_anchors)
        attempts, rejections = counts.get(name, (None, None))
        banks.append(
            ProbeBank(
                bank=name,
                episodes=len(labels),
                rows_per_episode=int(bank_states[0].shape[0]),
                first_row=first_row,
                episode_labels=tuple(labels),
                states_sha256=array_digest(stacked_states),
                anchors_sha256=array_digest(stacked_anchors),
                attempts_used=attempts,
                rejections=rejections,
            )
        )
        first_row += int(stacked_states.shape[0])
        states.append(stacked_states)
        anchors.append(stacked_anchors)
    return ProbeMatrix(np.vstack(states), np.vstack(anchors), tuple(banks))


def predict_probes(model: EsnModel, probes: ProbeMatrix) -> NDArray[np.float64]:
    """The readout's prediction on every probe state through ``rclib``'s own predict path."""
    return np.vstack([model.readout(state) for state in probes.states])


# --- diagnostics --------------------------------------------------------------------------


def _frobenius(array: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.sum(array * array, dtype=np.float64)))


@dataclass(frozen=True)
class NormalEquations:
    """The normalized normal-equation residual and conditioning of one fit's actual ridge problem."""

    rows: int
    columns: int
    """``n_neurons + 1``: the bias column is last, as the pinned solver orders it."""
    alpha: float
    residual: float
    """``||A W - B||_F / (||A||_F ||W||_F + ||B||_F)`` (``0 / 0 := 0``)."""
    numerator: float
    denominator: float
    a_norm: float
    b_norm: float
    w_norm: float
    eigenvalue_min: float
    eigenvalue_max: float
    cond2: float
    within: bool

    def __post_init__(self) -> None:
        """The decision re-derives from the figures."""
        if self.within != (self.residual <= TOLERANCES.residual_max):
            msg = "within contradicts the residual"
            raise ValueError(msg)
        if self.denominator == 0.0 and (self.residual != 0.0 or self.numerator != 0.0):
            msg = "an all-zero problem has residual zero"
            raise ValueError(msg)


def normal_equations(
    states: NDArray[np.float64], targets: NDArray[np.float64], weights: NDArray[np.float64], *, alpha: float
) -> NormalEquations:
    """Evaluate ``A = X^T X + alpha I`` (bias column last) and ``B = X^T Y`` against the fitted ``W``."""
    x = np.hstack([states, np.ones((states.shape[0], 1), dtype=np.float64)])
    if weights.shape != (x.shape[1], targets.shape[1]):
        msg = f"weights {weights.shape} do not match the design matrix {x.shape} and targets {targets.shape}"
        raise ValueError(msg)
    a = x.T @ x + alpha * np.eye(x.shape[1], dtype=np.float64)
    b = x.T @ targets
    numerator = _frobenius(a @ weights - b)
    a_norm, b_norm, w_norm = _frobenius(a), _frobenius(b), _frobenius(weights)
    denominator = a_norm * w_norm + b_norm
    residual = 0.0 if denominator == 0.0 else numerator / denominator
    eigenvalues = np.linalg.eigvalsh(a)
    e_min, e_max = float(eigenvalues[0]), float(eigenvalues[-1])
    cond2 = float("inf") if e_min <= 0.0 else e_max / e_min
    return NormalEquations(
        rows=int(x.shape[0]),
        columns=int(x.shape[1]),
        alpha=alpha,
        residual=residual,
        numerator=numerator,
        denominator=denominator,
        a_norm=a_norm,
        b_norm=b_norm,
        w_norm=w_norm,
        eigenvalue_min=e_min,
        eigenvalue_max=e_max,
        cond2=cond2,
        within=residual <= TOLERANCES.residual_max,
    )


@dataclass(frozen=True)
class QuantityDifference:
    """Elementwise difference of one compared quantity (prediction, increment, or command)."""

    quantity: str
    rows: int
    max_abs: float
    max_rel: float
    """Largest ``|a - b| / |b|`` over entries with ``b != 0`` (zero when there are none)."""
    worst_row: int
    worst_bank: str
    passed: bool
    """``all(|a - b| <= atol + rtol |b|)``."""


def _difference(
    quantity: str, candidate: NDArray[np.float64], reference: NDArray[np.float64], probes: ProbeMatrix
) -> QuantityDifference:
    if candidate.shape != reference.shape:
        msg = f"{quantity}: shapes differ ({candidate.shape} vs {reference.shape})"
        raise ValueError(msg)
    diff = np.abs(candidate - reference)
    scale = np.abs(reference)
    bound = TOLERANCES.prediction_atol_rad + TOLERANCES.prediction_rtol * scale
    excess = diff - bound
    worst = int(np.argmax(excess))
    worst_row = worst // diff.shape[1]
    nonzero = scale > 0.0
    max_rel = float(np.max(diff[nonzero] / scale[nonzero])) if bool(nonzero.any()) else 0.0
    return QuantityDifference(
        quantity=quantity,
        rows=int(diff.shape[0]),
        max_abs=float(np.max(diff)),
        max_rel=max_rel,
        worst_row=worst_row,
        worst_bank=probes.bank_of(worst_row),
        passed=bool(np.all(diff <= bound)),
    )


@dataclass(frozen=True)
class Comparison:
    """One equivalence comparison: candidate against reference on the entry's probe matrix."""

    panel_label: str
    formulation: str
    count: int
    candidate: str
    reference: str
    candidate_identity: str
    reference_identity: str
    differences: tuple[QuantityDifference, ...]
    coefficient_max_abs: float
    coefficient_max_rel: float
    coefficient_fro_rel: float
    """``||W_c - W_r||_F / max(||W_c||_F, ||W_r||_F)``."""
    passed: bool

    def __post_init__(self) -> None:
        """The decision re-derives from the quantities, which are the formulation's."""
        expected = ("prediction",) if self.formulation == "absolute" else ("increment", "command")
        if tuple(d.quantity for d in self.differences) != expected:
            msg = f"{self.formulation} comparisons hold the quantities {expected}"
            raise ValueError(msg)
        if self.passed != all(d.passed for d in self.differences):
            msg = "passed contradicts the differences"
            raise ValueError(msg)


def compare_fits(
    entry: PanelEntry,
    candidate: CachedFit,
    reference: CachedFit,
    probes: ProbeMatrix,
    predictions: Mapping[str, NDArray[np.float64]],
) -> Comparison:
    """Compare the candidate fit against the reference on the probe matrix (predictions cached by identity)."""
    arm_c, arm_r = candidate.record.arm, reference.record.arm
    if arm_c.formulation != arm_r.formulation or (arm_c.arm, arm_r.arm) not in COMPARISON_PAIRS:
        msg = f"{arm_c.label} is not compared against {arm_r.label}"
        raise ValueError(msg)
    pc, pr = predictions[candidate.record.identity], predictions[reference.record.identity]
    if arm_c.formulation == "absolute":
        differences = (_difference("prediction", pc, pr, probes),)
    else:
        differences = (
            _difference("increment", pc, pr, probes),
            _difference("command", probes.anchors + pc, probes.anchors + pr, probes),
        )
    wc, wr = candidate.weights, reference.weights
    diff = np.abs(wc - wr)
    nonzero = np.abs(wr) > 0.0
    return Comparison(
        panel_label=entry.label,
        formulation=arm_c.formulation,
        count=arm_c.count,
        candidate=arm_c.label,
        reference=arm_r.label,
        candidate_identity=candidate.record.identity,
        reference_identity=reference.record.identity,
        differences=differences,
        coefficient_max_abs=float(np.max(diff)),
        coefficient_max_rel=float(np.max(diff[nonzero] / np.abs(wr)[nonzero])) if bool(nonzero.any()) else 0.0,
        coefficient_fro_rel=_frobenius(wc - wr) / max(_frobenius(wc), _frobenius(wr), np.finfo(np.float64).tiny),
        passed=all(d.passed for d in differences),
    )


@dataclass(frozen=True)
class StateIdentity:
    """Whether matched absolute and residual fits harvested bitwise identical states."""

    panel_label: str
    arm: str
    """``<arm>`` with ``/K<count>``, e.g. ``R/K17``; the formulations differ only in their targets."""
    absolute_identity: str
    residual_identity: str
    identical: bool


@dataclass(frozen=True)
class FitSummary:
    """One of the pilot's fits with its diagnostics."""

    panel_label: str
    source_trial: int
    arm: ArmSpec
    identity: str
    cache_hit: bool
    solver_alpha: float
    fit: FitReport
    loss_rows_expected: int
    weights_sha256: str
    weights_shape: tuple[int, ...]
    """``(n_neurons + 1, output_dim)`` (the strict mapper reads homogeneous tuples only)."""
    state_digests: tuple[str, ...]
    copies_identical: bool
    """Repeated arms: every copy's harvested states equal the original's bitwise (``True`` for single fits)."""
    accessor_max_abs_diff: float
    """Largest ``|predict(x) - (x W[:-1] + W[-1])|`` over the probes: the accessor reproduces the solver's predict."""
    normal: NormalEquations
    execution_identity: str
    rclib: RclibIdentity
    fit_seconds: float

    def __post_init__(self) -> None:
        """The row count is the arm's."""
        if self.fit.loss_rows != self.loss_rows_expected:
            msg = (
                f"{self.panel_label}/{self.arm.label}: {self.fit.loss_rows} loss rows, "
                f"expected {self.loss_rows_expected}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class FreshRefit:
    """The outcome of refitting one cached fit in a fresh, pinned process that bypassed the cache."""

    identity: str
    worker_execution_identity: str
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


def fresh_refit(
    identity: str, *, store: StorageRoot, root: Path, parent_identity: str, execution: ExecutionRecord
) -> FreshRefit:
    """Refit the cached fit ``identity`` from its recipe and dataset only, then compare with the cache record.

    The recipe TOML is read from the cache directory; the weights are computed
    anew (never read before the comparison); the dataset comes from its Git
    record and the digest-verified payload.
    """
    fits = FitStore(store)
    started = time.perf_counter()
    recipe = load_recipe(fits.directory(identity) / RECIPE_FILE)
    recipe.require_rclib()
    samples = load_recipe_samples(recipe, store, root=root)
    validation = recipe.validation
    scenario = None if validation is None else load_scenario(root / validation.scenario_file)
    model = recipe.build_model()
    episodes = tuple(recipe.episodes(samples, scenario=scenario))
    report = train_readout(model, episodes)
    weights = model.readout_weights()
    states = harvested_states(model, episodes)
    elapsed = time.perf_counter() - started
    record = fits.read_record(identity)
    cached = fits.read_weights(record)
    same_shape = cached.shape == weights.shape
    environment_match = execution.identity == parent_identity == record.execution_identity
    return FreshRefit(
        identity=identity,
        worker_execution_identity=execution.identity,
        environment_match=environment_match,
        weights_bitwise_equal=same_shape and array_digest(weights) == record.weights_sha256,
        max_abs_weight_diff=float(np.max(np.abs(cached - weights))) if same_shape else float("inf"),
        states_bitwise_equal=tuple(array_digest(s) for s in states) == record.state_digests,
        fit_report_equal=report == record.fit,
        rmse_abs_diff=abs(report.rmse - record.fit.rmse),
        seconds=elapsed,
        passed=environment_match
        and same_shape
        and array_digest(weights) == record.weights_sha256
        and tuple(array_digest(s) for s in states) == record.state_digests
        and report == record.fit,
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


# --- the validation -----------------------------------------------------------------------


@dataclass(frozen=True)
class NumericalValidation:
    """The committed evidence of M3REP-003."""

    experiment: str
    panel_manifest_file: str
    panel_manifest_sha256: str
    dataset: DatasetSource
    dataset_payload: ArtifactReference
    execution: ExecutionRecord
    tolerances: Tolerances
    probe_bank_count: int
    probes: dict[str, tuple[ProbeBank, ...]]
    """Per panel entry: the banks of its probe matrix."""
    probe_states_identical_across_formulations: dict[str, bool]
    """Per entry: the probe matrix harvested through the residual ``S`` reservoir equals the absolute one bitwise."""
    fits: tuple[FitSummary, ...]
    comparisons: tuple[Comparison, ...]
    state_identities: tuple[StateIdentity, ...]
    fresh_refits: tuple[FreshRefit, ...]
    n_fits: int
    n_comparisons: int
    n_comparisons_passed: int
    n_residuals_within: int
    n_state_identities_identical: int
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
            (self.n_state_identities_identical, sum(1 for s in self.state_identities if s.identical)),
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
        expected = (
            self.n_comparisons_passed == self.n_comparisons
            and self.n_residuals_within == self.n_fits
            and self.n_state_identities_identical == len(self.state_identities)
            and self.n_fresh_refits_passed == self.n_fits
            and all(f.copies_identical for f in self.fits)
            and all(self.probe_states_identical_across_formulations.values())
        )
        if self.all_passed != expected:
            msg = "all_passed contradicts the retained records"
            raise ValueError(msg)
        if any(f.execution_identity != self.execution.identity for f in self.fits):
            msg = "every fit binds the validation's execution identity"
            raise ValueError(msg)
        if not is_hex(self.panel_manifest_sha256, _SHA256_HEX):
            msg = "panel_manifest_sha256 must be 64 lowercase hex characters"
            raise ValueError(msg)


def _summary(entry: PanelEntry, cached: CachedFit, probes: ProbeMatrix, prediction: NDArray[np.float64]) -> FitSummary:
    record = cached.record
    harvested = [harvest_episode(cached.model, episode) for episode in cached.episodes]
    states = np.vstack([h.training_states for h in harvested])
    targets = np.vstack([h.training_targets for h in harvested])
    digests = tuple(array_digest(h.states) for h in harvested)
    if digests != record.state_digests:
        msg = f"{entry.label}/{record.arm.label}: harvested states differ from the cached fit's record"
        raise ValueError(msg)
    linear = probes.states @ cached.weights[:-1] + cached.weights[-1]
    return FitSummary(
        panel_label=entry.label,
        source_trial=entry.source_trial,
        arm=record.arm,
        identity=record.identity,
        cache_hit=cached.cache_hit,
        solver_alpha=cached.recipe.solver_alpha,
        fit=record.fit,
        loss_rows_expected=expected_loss_rows(record.arm, rows_per_episode=int(harvested[0].loss_rows.sum())),
        weights_sha256=record.weights_sha256,
        weights_shape=record.weights_shape,
        state_digests=digests,
        copies_identical=len(set(digests)) == 1,
        accessor_max_abs_diff=float(np.max(np.abs(prediction - linear))),
        normal=normal_equations(states, targets, cached.weights, alpha=cached.recipe.solver_alpha),
        execution_identity=record.execution_identity,
        rclib=record.rclib,
        fit_seconds=record.fit_seconds,
    )


type Refitter = Callable[[str, str], FreshRefit]
"""``(fit identity, parent execution identity) -> FreshRefit``."""


def run_validation(
    entries: Sequence[PanelEntry],
    inputs: FitInputs,
    *,
    store: StorageRoot,
    execution: ExecutionRecord,
    refit: Refitter,
    manifest_file: str,
    manifest_sha256: str,
    dataset_payload: ArtifactReference,
    provenance: ProvenanceRecord,
    bank_count: int = PROBE_BANK_COUNT,
    counts: tuple[int, ...] = tuple(sorted(APPROVED_ADDITIONAL_REPEATS)),
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> NumericalValidation:
    """Fit, probe, compare, diagnose, and refit every entry; failures are retained, never repaired."""
    execution.check_canonical()
    fits = FitStore(store)
    arms = numerical_arms(counts=counts)
    summaries: list[FitSummary] = []
    comparisons: list[Comparison] = []
    identities: list[StateIdentity] = []
    refits: list[FreshRefit] = []
    probe_banks: dict[str, tuple[ProbeBank, ...]] = {}
    probe_identity: dict[str, bool] = {}
    for entry in entries:
        log(f"{entry.label}: fitting {len(arms)} arms")
        cached = {arm.label: fits.fit_or_load(entry, arm, inputs, now=now) for arm in arms}
        by_formulation = {
            formulation: build_probes(
                cached[f"{formulation}/S"].model,
                cached[f"{formulation}/S"].recipe.encoder(),
                inputs.samples,
                scenario=inputs.scenario,
                warmup_s=entry.warmup_s,
                period_s=inputs.preprocessing.resample_period_s,
                derivative_method=inputs.preprocessing.derivative_method,
                task_code=inputs.samples.task_code,
                bank_count=bank_count,
            )
            for formulation in FORMULATIONS
        }
        probes = by_formulation["absolute"]
        probe_banks[entry.label] = probes.banks
        probe_identity[entry.label] = all(
            a.states_sha256 == b.states_sha256 and a.anchors_sha256 == b.anchors_sha256
            for a, b in zip(probes.banks, by_formulation["residual"].banks, strict=True)
        )
        predictions = {fit.record.identity: predict_probes(fit.model, probes) for fit in cached.values()}
        summaries.extend(
            _summary(entry, cached[arm.label], probes, predictions[cached[arm.label].record.identity]) for arm in arms
        )
        for formulation in FORMULATIONS:
            for count in counts:
                for candidate, reference in COMPARISON_PAIRS:
                    left = cached[ArmSpec(formulation, candidate, count).label]
                    right = cached[ArmSpec(formulation, reference, None if reference == "S" else count).label]
                    comparisons.append(compare_fits(entry, left, right, probes, predictions))
        for arm in arms:
            if arm.formulation != "absolute":
                continue
            counterpart = cached[ArmSpec("residual", arm.arm, arm.additional_repeats).label]
            identities.append(
                StateIdentity(
                    panel_label=entry.label,
                    arm=arm.label.split("/", 1)[1],
                    absolute_identity=cached[arm.label].record.identity,
                    residual_identity=counterpart.record.identity,
                    identical=cached[arm.label].record.state_digests == counterpart.record.state_digests,
                )
            )
        log(f"{entry.label}: refitting {len(arms)} fits in fresh processes")
        refits.extend(refit(cached[arm.label].record.identity, execution.identity) for arm in arms)
    return NumericalValidation(
        experiment=EXPERIMENT_LABEL,
        panel_manifest_file=manifest_file,
        panel_manifest_sha256=manifest_sha256,
        dataset=inputs.source,
        dataset_payload=dataset_payload,
        execution=execution,
        tolerances=TOLERANCES,
        probe_bank_count=bank_count,
        probes=probe_banks,
        probe_states_identical_across_formulations=probe_identity,
        fits=tuple(summaries),
        comparisons=tuple(comparisons),
        state_identities=tuple(identities),
        fresh_refits=tuple(refits),
        n_fits=len(summaries),
        n_comparisons=len(comparisons),
        n_comparisons_passed=sum(1 for c in comparisons if c.passed),
        n_residuals_within=sum(1 for f in summaries if f.normal.within),
        n_state_identities_identical=sum(1 for s in identities if s.identical),
        n_fresh_refits_passed=sum(1 for r in refits if r.passed),
        all_passed=(
            all(c.passed for c in comparisons)
            and all(f.normal.within for f in summaries)
            and all(s.identical for s in identities)
            and all(r.passed for r in refits)
            and all(f.copies_identical for f in summaries)
            and all(probe_identity.values())
        ),
        provenance=provenance,
    )


def validation_to_json(validation: NumericalValidation) -> str:
    """Canonical JSON of the evidence."""
    return canonical_json(to_mapping(validation))


def load_validation(path: Path) -> NumericalValidation:
    """Strictly rebuild the evidence from JSON (re-deriving every count and decision)."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), NumericalValidation)


# --- rendering ----------------------------------------------------------------------------


def _e(value: float) -> str:
    return f"{value:.3e}"


def _flag(value: bool) -> str:  # noqa: FBT001 - a recorded decision is rendered, not a switch
    return "pass" if value else "**FAIL**"


def render_validation_markdown(validation: NumericalValidation) -> str:
    """The Markdown rendering of the evidence (every fit, comparison, and refit; failures retained)."""
    v = validation
    t = v.tolerances
    lines = [
        "# Task 1-a repetition numerical validation (v1)",
        "",
        (
            f"Experiment `{v.experiment}`: ridge equivalences and fresh-process reproducibility of the pilot's "
            f"fits (repetition plan section 6, D6) on the frozen panel `{v.panel_manifest_file}` "
            f"(sha256 `{v.panel_manifest_sha256[:12]}`), dataset `{v.dataset.artifact_id}` (payload sha256 "
            f"`{v.dataset_payload.sha256[:12]}`), rclib `{v.fits[0].rclib.version}` at "
            f"`{v.fits[0].rclib.commit[:12]}`, project commit `{v.provenance.project_commit[:12]}`"
            f"{' (dirty)' if v.provenance.project_dirty else ''}."
        ),
        "",
        "## Outcome",
        "",
        f"- Overall: **{'all checks passed' if v.all_passed else 'FAILURES RETAINED'}**.",
        f"- Fits: {v.n_fits} ({sum(1 for f in v.fits if f.cache_hit)} served from the cache).",
        (
            f"- Equivalence comparisons: {v.n_comparisons_passed} of {v.n_comparisons} within atol "
            f"{t.prediction_atol_rad:g} rad and rtol {t.prediction_rtol:g}."
        ),
        f"- Normal-equation residuals: {v.n_residuals_within} of {v.n_fits} at or below {t.residual_max:g}.",
        (
            f"- Literal copies harvested bitwise identically: "
            f"{sum(1 for f in v.fits if f.copies_identical)} of {v.n_fits} fits."
        ),
        (
            f"- Matched absolute/residual state identity: {v.n_state_identities_identical} of "
            f"{len(v.state_identities)} pairs; probe matrices identical across formulations for "
            f"{sum(1 for x in v.probe_states_identical_across_formulations.values() if x)} of "
            f"{len(v.probe_states_identical_across_formulations)} entries."
        ),
        (
            f"- Fresh-process refits: {v.n_fresh_refits_passed} of {v.n_fits} reproduced weights, states, and "
            "report bitwise."
        ),
        "",
        "## Execution environment",
        "",
        (
            f"- Identity `{v.execution.identity[:12]}` ({'canonical' if v.execution.canonical else 'NOT canonical'}): "
            f"policy `{v.execution.policy}`, CPUs {v.execution.effective_cpus[0]}-{v.execution.effective_cpus[-1]} "
            f"({len(v.execution.effective_cpus)} logical CPUs), threads "
            f"{', '.join(f'{k}={val}' for k, val in sorted(v.execution.thread_environment.items()))}, "
            f"BLAS `{v.execution.blas.corename}` via `{v.execution.blas.library}`, "
            f"OpenMP max threads {v.execution.openmp.max_threads}."
        ),
        (
            "- Every fit and every worker refit bound this identity; a worker whose environment differed "
            "is recorded as a failure, never merged."
        ),
        "",
        "## Probes",
        "",
        (
            f"Per entry: the original episode plus the two largest fixed augmented banks ({v.probe_bank_count} "
            "synthetic episodes each; anchor sigma 0.05 rad, phi 0.99, gamma 1.0, seed bank 1) harvested through "
            "the entry's reservoir after its warm-up; task rows only; probes never enter a residual fit's training."
        ),
        "",
        "| entry | bank | episodes | rows | attempts | rejections | states sha256 | anchors sha256 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for label, banks in sorted(v.probes.items()):  # label order: the JSON form sorts keys
        lines.extend(
            f"| {label} | {b.bank} | {b.episodes} | {b.rows} | {'' if b.attempts_used is None else b.attempts_used} "
            f"| {'' if b.rejections is None else b.rejections} | `{b.states_sha256[:12]}` | `{b.anchors_sha256[:12]}` |"
            for b in banks
        )
    lines += [
        "",
        "## Equivalence comparisons",
        "",
        (
            "Candidate against reference on the entry's probe matrix; residual pairs compare increments and the "
            "commands reconstructed as anchor + increment with identical anchors. Coefficient columns are the "
            "differences of the fitted weights (max abs, max relative, Frobenius relative)."
        ),
        "",
        (
            "| entry | formulation | K | candidate | reference | quantity | rows | max abs | max rel | worst bank "
            "| coef max abs | coef max rel | coef fro rel | decision |"
        ),
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for c in v.comparisons:
        lines.extend(
            f"| {c.panel_label} | {c.formulation} | {c.count} | {c.candidate.split('/', 1)[1]} "
            f"| {c.reference.split('/', 1)[1]} | {d.quantity} | {d.rows} | {_e(d.max_abs)} | {_e(d.max_rel)} "
            f"| {d.worst_bank} | {_e(c.coefficient_max_abs)} | {_e(c.coefficient_max_rel)} "
            f"| {_e(c.coefficient_fro_rel)} | {_flag(d.passed)} |"
            for d in c.differences
        )
    lines += [
        "",
        "## Fits",
        "",
        (
            "Every fit with its solver parameter, loss rows, teacher-forced fit errors, normalized normal-equation "
            "residual, conditioning of A = X^T X + alpha I (bias column last), accessor check (largest difference "
            "between rclib's predict and x W over the probes), and fit time."
        ),
        "",
        (
            "| entry | arm | alpha | loss rows | rmse | max abs error | residual | cond2 | accessor | copies "
            "| fit s | cache | identity |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    lines.extend(
        f"| {f.panel_label} | {f.arm.label} | {f.solver_alpha:.6g} | {f.fit.loss_rows} | {_e(f.fit.rmse)} "
        f"| {_e(f.fit.max_abs_error)} | {_e(f.normal.residual)} | {_e(f.normal.cond2)} "
        f"| {_e(f.accessor_max_abs_diff)} | {'identical' if f.copies_identical else '**DIFFER**'} "
        f"| {f.fit_seconds:.2f} | {'hit' if f.cache_hit else 'fit'} | `{f.identity[:12]}` |"
        for f in v.fits
    )
    lines += [
        "",
        "## State identity across formulations",
        "",
        "| entry | arm | absolute | residual | identical |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {s.panel_label} | {s.arm} | `{s.absolute_identity[:12]}` | `{s.residual_identity[:12]}` "
        f"| {'yes' if s.identical else '**NO**'} |"
        for s in v.state_identities
    )
    lines += [
        "",
        "## Fresh-process refits",
        "",
        (
            "Each cached fit refitted in a new pinned interpreter from its recipe TOML and the digest-verified "
            "dataset only (the cache is consulted after the fit, for the comparison)."
        ),
        "",
        "| identity | environment | weights | max abs weight diff | states | fit report | rmse diff | s | decision |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| `{r.identity[:12]}` | {'same' if r.environment_match else '**DIFFERENT**'} "
        f"| {'bitwise' if r.weights_bitwise_equal else '**DIFFER**'} | {_e(r.max_abs_weight_diff)} "
        f"| {'bitwise' if r.states_bitwise_equal else '**DIFFER**'} "
        f"| {'equal' if r.fit_report_equal else '**DIFFER**'} "
        f"| {_e(r.rmse_abs_diff)} | {r.seconds:.2f} | {_flag(r.passed)} |"
        for r in v.fresh_refits
    )
    lines += [
        "",
        "## Limitations",
        "",
        (
            "- The prediction tolerance bounds teacher-forced readout differences on fixed probes; it is not a "
            "closed-loop stability guarantee (plan section 6)."
        ),
        (
            "- The equivalences are numerical identities of the ridge problem; agreement here says nothing about "
            "the behavioral effect of repetition, which M3REP-004 to M3REP-007 evaluate."
        ),
        (
            "- Fresh-process bitwise reproducibility holds in the recorded execution environment only; another "
            "core type, thread setting, library build, or machine is a different environment (C10, condition 5)."
        ),
        (
            "- The normal-equation residual is evaluated with numpy on the harvested rows; the solve itself is "
            "rclib's LDLT (`cholesky` option) on its own accumulation of the same rows."
        ),
        "",
    ]
    return "\n".join(lines)


# --- command line -------------------------------------------------------------------------


def worker_main(args: argparse.Namespace) -> int:
    """The fresh-process refit worker: verify the pinned environment, refit, compare, write the result."""
    require_canonical()
    ensure_single_thread()
    _load_runtimes()
    execution = collect_execution(
        command=command_line(_WORKER_MODULE, cast("list[str]", args.argv)), role="worker", now=datetime.now(tz=UTC)
    )
    store = open_storage()
    result = fresh_refit(
        cast("str", args.fit),
        store=store,
        root=Path(cast("str", args.root)),
        parent_identity=cast("str", args.parent_identity),
        execution=execution,
    )
    output = Path(cast("str", args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(to_mapping(result)) + "\n", encoding="utf-8")
    return 0


def _load_runtimes() -> None:
    """Load the numerical runtimes so the execution probes see them (as ``arm_rc_ctrl.execution record`` does)."""
    for name in ("numpy", "rclib"):
        importlib.import_module(name)


def _validate(args: argparse.Namespace) -> int:
    require_canonical()
    ensure_single_thread()
    _load_runtimes()
    for target in (args.output, args.markdown):
        if Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    root = repository_root()
    store = open_storage()
    manifest_file = Path(cast("str", args.manifest))
    execution = collect_execution(
        command=command_line(_COMMAND_MODULE, cast("list[str]", args.argv)), role="main", now=datetime.now(tz=UTC)
    )
    execution.check_canonical()
    context = PanelContext.load(manifest_file, store=store, root=root, execution=execution)
    bank_count = int(args.bank_count)
    resolved = {
        "manifest": context.manifest_sha256,
        "tolerances": to_mapping(TOLERANCES),
        "probe_bank_count": bank_count,
        "arms": [arm.label for arm in numerical_arms()],
        "execution_identity": execution.identity,
        "command": command_line(_COMMAND_MODULE, cast("list[str]", args.argv)),
    }
    provenance = collect_provenance(
        resolved,
        seeds={},
        artifacts=[context.payload],
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
        context.manifest.entries,
        context.inputs,
        store=store,
        execution=execution,
        refit=refit,
        manifest_file=manifest_file.relative_to(root).as_posix()
        if manifest_file.is_relative_to(root)
        else str(manifest_file),
        manifest_sha256=context.manifest_sha256,
        dataset_payload=context.payload,
        provenance=provenance,
        bank_count=bank_count,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(validation_to_json(validation) + "\n", encoding="utf-8")
    Path(args.markdown).write_text(render_validation_markdown(validation), encoding="utf-8")
    print(
        json.dumps(
            {
                "all_passed": validation.all_passed,
                "fits": validation.n_fits,
                "comparisons_passed": f"{validation.n_comparisons_passed}/{validation.n_comparisons}",
                "residuals_within": f"{validation.n_residuals_within}/{validation.n_fits}",
                "fresh_refits_passed": f"{validation.n_fresh_refits_passed}/{validation.n_fits}",
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Numerical validation of the repeated-demonstration pilot.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    validate = subparsers.add_parser("validate", help="fit, compare, diagnose, and refit the panel's numerical arms")
    validate.add_argument("--manifest", type=str, required=True, help="frozen panel manifest JSON")
    validate.add_argument("--output", type=str, required=True, help="evidence JSON to write (must not exist)")
    validate.add_argument("--markdown", type=str, required=True, help="evidence Markdown to write (must not exist)")
    validate.add_argument(
        "--workspace", type=str, required=True, help="directory for the worker result files (outside the worktree)"
    )
    validate.add_argument("--bank-count", type=int, default=PROBE_BANK_COUNT, help="synthetic episodes per probe bank")
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
