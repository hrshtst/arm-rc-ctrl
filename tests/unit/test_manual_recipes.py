# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-005/M3MAN-007: the approved manual arms map the locked bank onto schema 3 recipes with matched counts.

``S_i`` trains on one demonstration, ``M10`` on all ten, ``R10_i`` on ten exact
copies of one, and ``C10_i`` on one demonstration plus the nine contractive
episodes grown from it; the last three match ``M10`` on episode count and total
loss weight, not on raw sample count. Every arm copies its input transform from
the historical scripted dataset and never from a take (clarification I8).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.manual import ManualDatasetRecord
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.data.records import Normalization, Preprocessing, ProcessedDatasetRecord, load_record
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.experiments.manual_bank import load_bank_manifest
from arm_rc_ctrl.experiments.manual_recipes import (
    ARMS,
    ASSIGNMENTS,
    CONTRACTIVE_ARM,
    COPIES,
    MANUAL_ANCHOR,
    ROWS_REFERENCE,
    SYNTHETIC_EPISODES,
    TRANSFORM_SOURCE,
    ManualAnchor,
    ManualArmSpec,
    arm_accounting,
    arm_sources,
    bank_sources,
    esn_for_arm,
    fit_identity,
    manual_arms,
    readout_for_arm,
    recipe_for_arm,
    training_spec_for_arm,
)
from arm_rc_ctrl.rc.augment import MANUAL_N_SYNTHETIC, TAPER_ZERO_MARGIN_S
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import (
    EQUAL_EPISODE_WEIGHTING,
    WEIGHTED_SCHEMA_VERSION,
    ContractiveTrainingSpec,
    DatasetSource,
    RclibIdentity,
    TrainingValidation,
    expected_episode_labels,
)
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario

if TYPE_CHECKING:
    from collections.abc import Mapping

    from arm_rc_ctrl.rc.augment import TaskGeometry
    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.recipe import ModelRecipe


def _statistics(record: ProcessedDatasetRecord) -> Normalization:
    """The record's recorded statistics; the frozen scripted dataset always carries them."""
    if record.normalization is None:
        msg = f"{record.artifact.artifact_id} records no normalization statistics"
        raise ValueError(msg)
    return record.normalization


REPO_ROOT = repository_root()
BANK_FILE = REPO_ROOT / "docs" / "experiments" / "task_1a_manual_demonstration" / "bank" / "bank_v1.json"
BANK = load_bank_manifest(BANK_FILE)
SOURCES = bank_sources(BANK, root=REPO_ROOT)
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
VALIDATION = TrainingValidation.from_scenario(SCENARIO, SCENARIO_FILE, root=REPO_ROOT)
DERIVATIVES = DerivativeConfig(method="central")
HISTORICAL = load_record(REPO_ROOT / TRANSFORM_SOURCE.record, ProcessedDatasetRecord)
HISTORICAL_NORMALIZATION = _statistics(HISTORICAL)  # the frozen physical statistics live in the scripted record
TRANSFORM = InputTransform.derive("fixed_scale", HISTORICAL_NORMALIZATION, fixed_scales={"q": 0.3, "dq": 12.0})
DT = 0.01
WARMUP_S = 0.25
BASE_ALPHA = 0.02
BASE_ESN = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=20, spectral_radius=0.85, sparsity=0.9, leak_rate=0.4, input_scaling=0.4, seed=23
    ),
    readout=ReadoutConfig(alpha=0.5),
)
PREPROCESSING = Preprocessing(
    resample_period_s=DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
)
DEFAULT_ARM = ManualArmSpec("S", "D01")


def _samples(n: int) -> SampleSet:
    """A complete recording of ``n`` samples: a short hold, a smooth reach, and a final dwell."""
    t = np.arange(n, dtype=np.float64) * DT
    start = np.array(SCENARIO.task.initial_q)
    hold_end, move_end = 0.1, float(t[-1]) - 0.1
    ramp = np.clip((t - hold_end) / (move_end - hold_end), 0.0, 1.0)
    blend = ramp * ramp * (3.0 - 2.0 * ramp)
    q = start[None, :] + blend[:, None] * (np.array([0.8, 0.4]) - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = endpoint_positions(SCENARIO, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.where(t < hold_end, 0, np.where(t < move_end, 1, 2)).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((n, 0)), phase)


SAMPLES = {SOURCES[name].artifact_id: _samples(41 + 4 * index) for index, name in enumerate(ASSIGNMENTS)}
LOSS_ROWS = {name: SAMPLES[SOURCES[name].artifact_id].n_samples - 1 for name in ASSIGNMENTS}

MANUAL_SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_manual_fixture.toml"
MANUAL_SCENARIO = load_manual_scenario(MANUAL_SCENARIO_FILE)
MANUAL_VALIDATION = TrainingValidation.from_scenario(MANUAL_SCENARIO, MANUAL_SCENARIO_FILE, root=REPO_ROOT)
CONTRACTIVE_ASSIGNMENT = "D01"
PARENT = SOURCES[CONTRACTIVE_ASSIGNMENT].artifact_id
PARENT_SAMPLES = _samples(151)
"""A 1.50 s recording: the frozen 0.5 s ramp closes well before the terminal taper window opens."""
PARENT_ROWS = PARENT_SAMPLES.n_samples - 1
DWELL_START_S = float(PARENT_SAMPLES.t[-1]) - 0.1
CONTRACTIVE_SAMPLES = {PARENT: PARENT_SAMPLES}
SEED_BANK = 7
BANK_DIGEST = "1a" * 32
"""The digest of the frozen bank a ``C10`` key binds; the real one comes from the bank record (M3MAN-006)."""
OTHER_BANK_DIGEST = "2b" * 32


def _build(
    arm: ManualArmSpec,
    *,
    transform: InputTransform = TRANSFORM,
    preprocessing: Preprocessing = PREPROCESSING,
    samples: Mapping[str, SampleSet] = SAMPLES,
    validation: TrainingValidation = VALIDATION,
    scenario: TaskGeometry = SCENARIO,
    contractive: ContractiveTrainingSpec | None = None,
) -> tuple[ModelRecipe, EsnModel]:
    return recipe_for_arm(
        arm,
        esn=BASE_ESN,
        sources=SOURCES,
        samples=samples,
        dof=2,
        task_code_dim=0,
        preprocessing=preprocessing,
        transform=transform,
        validation=validation,
        warmup_s=WARMUP_S,
        base_alpha=BASE_ALPHA,
        scenario=scenario,
        contractive=contractive,
    )


def _build_contractive(
    *, seed_bank: int = SEED_BANK, assignment: str = CONTRACTIVE_ASSIGNMENT
) -> tuple[ModelRecipe, EsnModel]:
    """The ``C10`` arm over the long parent recording, validated against the manual task schema."""
    return _build(
        ManualArmSpec(CONTRACTIVE_ARM, assignment),
        samples={SOURCES[assignment].artifact_id: PARENT_SAMPLES},
        validation=MANUAL_VALIDATION,
        scenario=MANUAL_SCENARIO,
        contractive=ContractiveTrainingSpec(seed_bank=seed_bank, dwell_start_s=DWELL_START_S),
    )


def test_manual_arms_enumerate_the_approved_scope() -> None:
    """Ten singletons, the all-ten model, ten duplication controls, ten contractive arms; invalid ones refused."""
    arms = manual_arms()
    labels = [arm.label for arm in arms]
    assert len(arms) == 3 * len(ASSIGNMENTS) + 1 == 31
    assert labels[0] == "S/D01"
    assert labels[9] == "S/D10"
    assert labels[10] == "M10"
    assert labels[11] == "R10/D01"
    assert labels[20] == "R10/D10"
    assert labels[21] == "C10/D01"
    assert labels[-1] == "C10/D10"
    assert len(set(labels)) == len(labels)
    assert set(ARMS) == {"S", "M10", "R10", "C10"}
    contractive_arms = [arm for arm in arms if arm.arm == CONTRACTIVE_ARM]
    assert [arm.assignment for arm in contractive_arms] == list(ASSIGNMENTS)  # exactly one per demonstration
    assert len(contractive_arms) == len(ASSIGNMENTS)

    single = ManualArmSpec("S", "D01")
    assert (single.count, single.unique_sources, single.copies, single.synthetic) == (1, 1, 0, 0)
    assert single.source_counts == (1,)
    everything = ManualArmSpec("M10")
    assert (everything.count, everything.unique_sources, everything.copies, everything.synthetic) == (10, 10, 0, 0)
    assert everything.source_counts == (1,) * 10
    assert everything.assignments == ASSIGNMENTS
    control = ManualArmSpec("R10", "D05")
    assert (control.count, control.unique_sources, control.copies, control.synthetic) == (COPIES, 1, COPIES - 1, 0)
    assert control.source_counts == (COPIES,)
    contractive = ManualArmSpec(CONTRACTIVE_ARM, "D05")
    # One parent plus nine synthetic episodes: count-matched with M10 and R10, but none of them is a copy.
    assert (contractive.count, contractive.unique_sources, contractive.copies) == (10, 1, 0)
    assert contractive.synthetic == SYNTHETIC_EPISODES == MANUAL_N_SYNTHETIC
    assert contractive.source_counts == (1,)
    assert contractive.assignments == ("D05",)

    with pytest.raises(ValueError, match="carries no assignment"):
        ManualArmSpec("M10", "D01")
    with pytest.raises(ValueError, match="needs an assignment"):
        ManualArmSpec("S")
    with pytest.raises(ValueError, match="needs an assignment"):
        ManualArmSpec("R10", "D11")
    with pytest.raises(ValueError, match="needs an assignment"):
        ManualArmSpec(CONTRACTIVE_ARM)
    with pytest.raises(ValueError, match="needs an assignment"):
        ManualArmSpec(CONTRACTIVE_ARM, "D11")
    with pytest.raises(ValueError, match="arm must be one of"):
        ManualArmSpec("C100", "D01")  # the whole-bank duplication control stays deferred


def test_training_specs_and_readouts_follow_the_frozen_anchor() -> None:
    """Every arm gets the anchor's weighting and warm-up, ``K alpha_0``, and the explicit-bias readout."""
    for arm in manual_arms():
        construction = (
            ContractiveTrainingSpec(seed_bank=SEED_BANK, dwell_start_s=DWELL_START_S) if arm.synthetic else None
        )
        spec = training_spec_for_arm(arm, warmup_s=WARMUP_S, base_alpha=BASE_ALPHA, contractive=construction)
        assert spec.washout == "warmup_hold"
        assert spec.warmup_s == WARMUP_S
        assert spec.target == "next_q"
        assert spec.episode_weighting == EQUAL_EPISODE_WEIGHTING
        assert spec.weight_reference_rows == ROWS_REFERENCE
        assert spec.source_counts == arm.source_counts
        assert spec.contractive == construction
        assert spec.episode_count == arm.count
        assert spec.base_alpha == BASE_ALPHA
        assert spec.regularization_rule == "count_scaled"
        readout = readout_for_arm(BASE_ESN.readout, arm, base_alpha=BASE_ALPHA)
        assert readout.alpha == BASE_ALPHA * arm.count
        assert readout.include_bias is False
        assert readout.explicit_bias is True
        assert readout.solver == BASE_ESN.readout.solver
        assert esn_for_arm(BASE_ESN, arm, base_alpha=BASE_ALPHA).reservoir == BASE_ESN.reservoir
    assert readout_for_arm(BASE_ESN.readout, ManualArmSpec("S", "D01"), base_alpha=BASE_ALPHA).alpha == BASE_ALPHA
    assert readout_for_arm(BASE_ESN.readout, ManualArmSpec("M10"), base_alpha=BASE_ALPHA).alpha == 10 * BASE_ALPHA
    assert MANUAL_ANCHOR.regularization_lambda(BASE_ALPHA) == BASE_ALPHA / ROWS_REFERENCE
    assert ManualAnchor() == MANUAL_ANCHOR
    with pytest.raises(ValueError, match="approved construction"):
        ManualAnchor(episode_weighting="per_row")
    with pytest.raises(ValueError, match="must be positive"):
        ManualAnchor(weight_reference_rows=0)
    # The contractive construction belongs to C10 alone, and C10 cannot be built without it.
    with pytest.raises(ValueError, match="contractive"):
        training_spec_for_arm(ManualArmSpec(CONTRACTIVE_ARM, "D01"), warmup_s=WARMUP_S, base_alpha=BASE_ALPHA)
    with pytest.raises(ValueError, match="contractive"):
        training_spec_for_arm(
            ManualArmSpec("R10", "D01"),
            warmup_s=WARMUP_S,
            base_alpha=BASE_ALPHA,
            contractive=ContractiveTrainingSpec(seed_bank=SEED_BANK, dwell_start_s=DWELL_START_S),
        )


def test_accounting_records_rows_weights_and_ridge_scales() -> None:
    """Unequal recordings get weights ``400 / L_i``; copies get one weight; the matched arms share total weight."""
    single = arm_accounting(ManualArmSpec("S", "D02"), base_alpha=BASE_ALPHA, loss_rows=LOSS_ROWS)
    assert single.loss_rows == (LOSS_ROWS["D02"],)
    assert single.row_weights == (ROWS_REFERENCE / LOSS_ROWS["D02"],)
    assert single.solver_alpha == BASE_ALPHA
    assert single.regularization_lambda == BASE_ALPHA / ROWS_REFERENCE
    assert single.total_loss_weight == pytest.approx(ROWS_REFERENCE)

    bank = arm_accounting(ManualArmSpec("M10"), base_alpha=BASE_ALPHA, loss_rows=LOSS_ROWS)
    assert bank.loss_rows == tuple(LOSS_ROWS[name] for name in ASSIGNMENTS)
    assert bank.row_weights == tuple(ROWS_REFERENCE / LOSS_ROWS[name] for name in ASSIGNMENTS)
    assert len(set(bank.row_weights)) == len(ASSIGNMENTS)  # unequal lengths give unequal per-row weights
    assert bank.total_loss_weight == pytest.approx(ROWS_REFERENCE * 10)
    assert (bank.unique_sources, bank.copies) == (10, 0)

    control = arm_accounting(ManualArmSpec("R10", "D02"), base_alpha=BASE_ALPHA, loss_rows=LOSS_ROWS)
    assert control.loss_rows == (LOSS_ROWS["D02"],) * COPIES
    assert len(set(control.row_weights)) == 1  # equal lengths give equal weights
    assert (control.unique_sources, control.copies) == (1, COPIES - 1)
    assert control.solver_alpha == bank.solver_alpha == 10 * BASE_ALPHA
    # The controls match M10 on episode count and total loss weight, not on raw sample count.
    assert control.episodes == bank.episodes == 10
    assert control.total_loss_weight == pytest.approx(bank.total_loss_weight)
    assert sum(control.loss_rows) != sum(bank.loss_rows)

    synthetic = arm_accounting(ManualArmSpec(CONTRACTIVE_ARM, "D02"), base_alpha=BASE_ALPHA, loss_rows=LOSS_ROWS)
    # Synthetic episodes share their parent's length, so they weigh like copies but are counted apart from them.
    assert synthetic.loss_rows == (LOSS_ROWS["D02"],) * 10
    assert synthetic.row_weights == (ROWS_REFERENCE / LOSS_ROWS["D02"],) * 10
    assert (synthetic.unique_sources, synthetic.copies, synthetic.synthetic) == (1, 0, SYNTHETIC_EPISODES)
    assert synthetic.episodes == control.episodes == bank.episodes
    assert synthetic.total_loss_weight == pytest.approx(bank.total_loss_weight)
    assert synthetic.solver_alpha == 10 * BASE_ALPHA
    assert (single.synthetic, bank.synthetic, control.synthetic) == (0, 0, 0)
    with pytest.raises(ValueError, match="needs the loss rows"):
        arm_accounting(ManualArmSpec("S", "D01"), base_alpha=BASE_ALPHA, loss_rows={})


def test_bank_sources_bind_the_locked_takes_by_digest() -> None:
    """Each assignment resolves to its committed processed record, digest-bound and repository-relative."""
    assert sorted(SOURCES) == list(ASSIGNMENTS)
    assigned = {take.assignment: take for take in BANK.takes if take.assignment is not None}
    for name, source in SOURCES.items():
        record = load_record(REPO_ROOT / source.record, ManualDatasetRecord)
        assert source.artifact_id == assigned[name].processed_artifact_id == record.artifact.artifact_id
        assert source.payload_sha256 == record.artifact.payload.sha256
        assert source.record == f"data/records/processed/{source.artifact_id}.toml"
        assert not source.record.startswith("/")  # records never carry machine paths
        assert record.normalization is None  # a manual take records no statistics by design (I8)
    assert arm_sources(ManualArmSpec("M10"), SOURCES) == tuple(SOURCES[name] for name in ASSIGNMENTS)
    assert arm_sources(ManualArmSpec("R10", "D03"), SOURCES) == (SOURCES["D03"],)
    with pytest.raises(ValueError, match="needs datasets for"):
        arm_sources(ManualArmSpec("S", "D04"), {})


def test_recipe_for_arm_builds_the_weighted_schema_three_recipe() -> None:
    """M10 and R10 are schema 3 recipes at ``10 alpha_0``; the control's copies are multiplicities, not payloads."""
    recipe, model = _build(ManualArmSpec("M10"))
    assert recipe.schema_version == WEIGHTED_SCHEMA_VERSION
    assert recipe.name == "M10"
    assert recipe.solver_alpha == 10 * BASE_ALPHA
    assert recipe.transform_source == TRANSFORM_SOURCE
    assert [d.artifact_id for d in recipe.datasets] == [SOURCES[name].artifact_id for name in ASSIGNMENTS]
    assert recipe.training.source_counts == (1,) * 10
    assert recipe.fit.episode_loss_rows == tuple(LOSS_ROWS[name] for name in ASSIGNMENTS)
    assert recipe.fit.episode_weights == tuple(ROWS_REFERENCE / LOSS_ROWS[name] for name in ASSIGNMENTS)
    assert model.explicit_bias
    assert model.readout_weights().shape == (BASE_ESN.reservoir.n_neurons + 1, 2)

    control, _model = _build(ManualArmSpec("R10", "D01"))
    assert control.name == "R10/D01"
    assert control.training.source_counts == (COPIES,)
    assert len(control.datasets) == 1
    assert control.fit.episodes[1] == f"{SOURCES['D01'].artifact_id}#copy-001"
    assert control.fit.loss_rows == COPIES * LOSS_ROWS["D01"]
    assert control.solver_alpha == recipe.solver_alpha
    _refit, report = control.refit(SAMPLES, scenario=SCENARIO)
    assert report == control.fit

    single, _single_model = _build(ManualArmSpec("S", "D01"))
    assert single.solver_alpha == BASE_ALPHA
    assert single.fit.episodes == (SOURCES["D01"].artifact_id,)
    assert single.esn.reservoir == recipe.esn.reservoir  # one reservoir across the arms of a configuration


def test_a_transform_computed_from_a_take_is_refused() -> None:
    """Source isolation: the frozen transform is copied from the scripted dataset, never from a demonstration."""
    from_take = dataclasses.replace(TRANSFORM, derived_from=(SOURCES["D01"].artifact_id,))
    with pytest.raises(ValueError, match="never compute statistics from a take"):
        _build(ManualArmSpec("S", "D01"), transform=from_take)


def test_a_manual_take_is_accepted_as_training_data_without_normalization() -> None:
    """The recipe binds the committed manual record although it records no statistics of its own (I8)."""
    record = load_record(REPO_ROOT / SOURCES["D01"].record, ManualDatasetRecord)
    recipe, _model = _build(ManualArmSpec("S", "D01"), preprocessing=record.preprocessing)
    recipe.check_dataset_record(SOURCES["D01"], record)
    assert recipe.check_transform_record(HISTORICAL) == HISTORICAL_NORMALIZATION
    recipe.check_transform_source({TRANSFORM_SOURCE.artifact_id: HISTORICAL_NORMALIZATION})


def test_fit_identity_binds_every_input_including_the_environment() -> None:
    """The identity changes with the arm, the demonstration, the ridge scale, the transform, and the environment."""
    rclib = RclibIdentity.current()

    def identity(
        *,
        arm: ManualArmSpec = DEFAULT_ARM,
        warmup_s: float = WARMUP_S,
        base_alpha: float = BASE_ALPHA,
        esn: EsnConfig = BASE_ESN,
        datasets: tuple[DatasetSource, ...] = (),
        transform: InputTransform = TRANSFORM,
        execution_identity: str = "a" * 64,
    ) -> str:
        return fit_identity(
            configuration="trial-17",
            arm=arm,
            warmup_s=warmup_s,
            base_alpha=base_alpha,
            esn=esn,
            datasets=datasets or (SOURCES["D01"],),
            transform=transform,
            validation=VALIDATION,
            rclib_commit=rclib.commit,
            execution_identity=execution_identity,
        )

    base = identity()
    assert base == identity()
    assert len(base) == 64
    assert base != identity(arm=ManualArmSpec("R10", "D01"))
    assert base != identity(arm=ManualArmSpec("S", "D02"))
    assert base != identity(datasets=(SOURCES["D02"],))
    assert base != identity(base_alpha=0.03)
    assert base != identity(warmup_s=1.0)
    assert base != identity(execution_identity="b" * 64)
    assert base != identity(esn=esn_for_arm(BASE_ESN, ManualArmSpec("S", "D01"), base_alpha=BASE_ALPHA))
    assert base != identity(transform=dataclasses.replace(TRANSFORM, derived_from=("processed-20260830-999999999999",)))


def test_contractive_arm_builds_the_schema_three_bank_recipe() -> None:
    """``C10`` is one parent plus nine contractive episodes at ten alpha_0, bound to the manual task schema."""
    recipe, model = _build_contractive()
    assert recipe.schema_version == WEIGHTED_SCHEMA_VERSION
    assert recipe.name == f"{CONTRACTIVE_ARM}/{CONTRACTIVE_ASSIGNMENT}"
    assert [dataset.artifact_id for dataset in recipe.datasets] == [PARENT]
    assert recipe.training.source_counts == (1,)
    assert recipe.training.contractive == ContractiveTrainingSpec(seed_bank=SEED_BANK, dwell_start_s=DWELL_START_S)
    assert recipe.training.episode_count == 1 + MANUAL_N_SYNTHETIC == 10
    assert recipe.solver_alpha == 10 * BASE_ALPHA
    assert recipe.validation == MANUAL_VALIDATION
    assert recipe.transform_source == TRANSFORM_SOURCE
    assert recipe.fit.episode_loss_rows == (PARENT_ROWS,) * 10
    assert recipe.fit.episode_weights == (ROWS_REFERENCE / PARENT_ROWS,) * 10
    assert recipe.fit.loss_rows == 10 * PARENT_ROWS
    assert model.explicit_bias


def test_contractive_episode_labels_name_the_parent_and_its_bank() -> None:
    """The recorded parent trains first under its own artifact ID, then ``#contractive-001`` .. ``-009``."""
    recipe, _model = _build_contractive()
    labels = expected_episode_labels(recipe.training, (PARENT,))
    assert labels == recipe.fit.episodes
    assert labels[0] == PARENT
    assert labels[1] == f"{PARENT}#contractive-001"
    assert labels[-1] == f"{PARENT}#contractive-{MANUAL_N_SYNTHETIC:03d}"
    assert len(labels) == 10

    episodes = recipe.episodes(CONTRACTIVE_SAMPLES, scenario=MANUAL_SCENARIO)
    assert [episode.source for episode in episodes] == list(labels)
    single, _single_model = _build(
        ManualArmSpec("S", CONTRACTIVE_ASSIGNMENT),
        samples=CONTRACTIVE_SAMPLES,
        validation=MANUAL_VALIDATION,
        scenario=MANUAL_SCENARIO,
    )
    (recorded,) = single.episodes(CONTRACTIVE_SAMPLES, scenario=MANUAL_SCENARIO)
    # The first episode is the recorded demonstration itself, built exactly as the singleton arm builds it.
    assert np.array_equal(episodes[0].inputs, recorded.inputs)
    assert np.array_equal(episodes[0].targets, recorded.targets)
    assert np.array_equal(episodes[0].t, recorded.t)
    assert np.array_equal(episodes[0].loss_rows, recorded.loss_rows)


def test_contractive_episodes_preserve_the_parents_start_and_final_dwell() -> None:
    """The M3MAN-006 boundary guarantee reaches the fit: the recorded start and dwell train unperturbed."""
    recipe, _model = _build_contractive()
    episodes = recipe.episodes(CONTRACTIVE_SAMPLES, scenario=MANUAL_SCENARIO)
    parent, bank = episodes[0], episodes[1:]
    warmup_rows = parent.washout_len
    held = np.tile(PARENT_SAMPLES.q[0], (warmup_rows, 1))
    dwell_rows = int(np.count_nonzero(PARENT_SAMPLES.t[1:] >= DWELL_START_S - TAPER_ZERO_MARGIN_S))
    assert len(bank) == MANUAL_N_SYNTHETIC
    assert warmup_rows == round(WARMUP_S / DT)
    assert dwell_rows > 0
    for episode in bank:
        assert episode.n_rows == parent.n_rows
        assert episode.washout_len == warmup_rows
        assert int(episode.loss_rows.sum()) == PARENT_ROWS  # the recorded pre-roll trains too (I4, I10)
        # The warm-up holds the parent's exact first sample, and the recorded final dwell is untouched.
        assert np.array_equal(episode.targets[:warmup_rows], held)
        assert np.array_equal(episode.targets[-dwell_rows:], parent.targets[-dwell_rows:])
        assert not np.array_equal(episode.targets, parent.targets)  # the perturbation does reach the loss rows


def test_contractive_recipes_regenerate_bitwise_from_config_and_parent() -> None:
    """Building the same arm twice gives identical episodes and fit; another seed bank gives another bank."""
    first, _model = _build_contractive()
    second, _other = _build_contractive()
    built = first.episodes(CONTRACTIVE_SAMPLES, scenario=MANUAL_SCENARIO)
    rebuilt = second.episodes(CONTRACTIVE_SAMPLES, scenario=MANUAL_SCENARIO)
    for episode, again in zip(built, rebuilt, strict=True):
        assert episode.source == again.source
        assert np.array_equal(episode.inputs, again.inputs)
        assert np.array_equal(episode.targets, again.targets)
    assert first.fit == second.fit
    _refit, report = first.refit(CONTRACTIVE_SAMPLES, scenario=MANUAL_SCENARIO)
    assert report == first.fit

    shifted, _third = _build_contractive(seed_bank=SEED_BANK + 1)
    other_bank = shifted.episodes(CONTRACTIVE_SAMPLES, scenario=MANUAL_SCENARIO)
    assert shifted.fit != first.fit
    assert np.array_equal(other_bank[0].targets, built[0].targets)  # the parent never depends on the seed bank
    assert not np.array_equal(other_bank[1].targets, built[1].targets)


def test_a_contractive_bank_that_cannot_be_grown_names_its_parent() -> None:
    """A recording that cannot carry the frozen envelope fails loudly instead of training a short bank."""
    with pytest.raises(ValueError, match=PARENT) as excinfo:
        _build(
            ManualArmSpec(CONTRACTIVE_ARM, CONTRACTIVE_ASSIGNMENT),
            samples=SAMPLES,  # the 0.40 s recording of the shared fixture bank
            validation=MANUAL_VALIDATION,
            scenario=MANUAL_SCENARIO,
            contractive=ContractiveTrainingSpec(seed_bank=SEED_BANK, dwell_start_s=DWELL_START_S),
        )
    assert "dwell onset" in str(excinfo.value.__cause__)  # the generator's own diagnosis stays attached


def test_the_contractive_construction_binds_the_fit_identity() -> None:
    """A contractive fit key changes with the seed bank, the dwell onset, and the digest of the bank itself."""
    rclib = RclibIdentity.current()

    def identity(
        arm: ManualArmSpec, contractive: ContractiveTrainingSpec | None, bank_sha256: str | None = None
    ) -> str:
        return fit_identity(
            configuration="trial-17",
            arm=arm,
            warmup_s=WARMUP_S,
            base_alpha=BASE_ALPHA,
            esn=BASE_ESN,
            datasets=(SOURCES[CONTRACTIVE_ASSIGNMENT],),
            transform=TRANSFORM,
            validation=MANUAL_VALIDATION,
            rclib_commit=rclib.commit,
            execution_identity="a" * 64,
            contractive=contractive,
            bank_sha256=bank_sha256,
        )

    arm = ManualArmSpec(CONTRACTIVE_ARM, CONTRACTIVE_ASSIGNMENT)
    construction = ContractiveTrainingSpec(seed_bank=SEED_BANK, dwell_start_s=DWELL_START_S)
    base = identity(arm, construction, BANK_DIGEST)
    assert base == identity(arm, construction, BANK_DIGEST)
    assert base != identity(
        arm, ContractiveTrainingSpec(seed_bank=SEED_BANK + 1, dwell_start_s=DWELL_START_S), BANK_DIGEST
    )
    assert base != identity(
        arm, ContractiveTrainingSpec(seed_bank=SEED_BANK, dwell_start_s=DWELL_START_S - 0.01), BANK_DIGEST
    )
    assert base != identity(ManualArmSpec("R10", CONTRACTIVE_ASSIGNMENT), None)
    # Two banks that differ only in their recorded digest are two different models (M3MAN-006).
    assert base != identity(arm, construction, OTHER_BANK_DIGEST)
    with pytest.raises(ValueError, match="needs its construction"):
        identity(arm, None)
    with pytest.raises(ValueError, match="recorded episodes only"):
        identity(ManualArmSpec("S", CONTRACTIVE_ASSIGNMENT), construction, BANK_DIGEST)
    # The bank digest and the construction are one binding: neither is optional where the other is present.
    with pytest.raises(ValueError, match="frozen bank"):
        identity(arm, construction)
    with pytest.raises(ValueError, match="grows no contractive bank"):
        identity(ManualArmSpec("R10", CONTRACTIVE_ASSIGNMENT), None, BANK_DIGEST)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        identity(arm, construction, "not-a-digest")


def test_a_recorded_data_fit_identity_never_depends_on_a_contractive_bank() -> None:
    """The 126 recorded-data keys are hashed from a mapping the bank digest never enters."""
    rclib = RclibIdentity.current()
    keys = {
        arm.label: fit_identity(
            configuration="trial-17",
            arm=arm,
            warmup_s=WARMUP_S,
            base_alpha=BASE_ALPHA,
            esn=BASE_ESN,
            datasets=(SOURCES[CONTRACTIVE_ASSIGNMENT],),
            transform=TRANSFORM,
            validation=MANUAL_VALIDATION,
            rclib_commit=rclib.commit,
            execution_identity="a" * 64,
        )
        for arm in (ManualArmSpec("S", CONTRACTIVE_ASSIGNMENT), ManualArmSpec("R10", CONTRACTIVE_ASSIGNMENT))
    }
    assert len(set(keys.values())) == len(keys)
    # The committed values of these keys are locked against the frozen manifest in tests/unit/test_manual_study.py.
    assert all(len(key) == 64 for key in keys.values())
