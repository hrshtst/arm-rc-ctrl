# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-018: deterministic representative selection and the persisted, installed pair set."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    ArtifactRecord,
    Origin,
    Payload,
    Preprocessing,
    Scenario,
    array_specs,
    load_catalog,
    make_artifact_id,
)
from arm_rc_ctrl.data.recovery import (
    TASK_PHASE_CODES,
    BaselineCheck,
    CropWindow,
    OnsetAnnotation,
    RecoveryDatasetRecord,
    TaskIntervals,
)
from arm_rc_ctrl.data.samples import SampleSet, save_samples
from arm_rc_ctrl.experiments.esn_search import TrialPoint
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.recovery_objective import RecoveryTrialContext
from arm_rc_ctrl.experiments.recovery_representative import (
    install_representatives,
    load_representatives,
    run_representatives,
    select_scenarios,
)
from arm_rc_ctrl.experiments.recovery_search import RecoveryTrialPoint, load_recovery_search
from arm_rc_ctrl.experiments.studies import TrialRecord
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.rc.recipe import DatasetSource
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.recovery_search import RecoverySearchProtocol

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
MODEL = REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml"
TRACKER = load_config(REPO_ROOT / "configs" / "controllers" / "pd.toml", TrackerConfig)
TRACKER_CT = load_config(REPO_ROOT / "configs" / "controllers" / "task_1a_computed_torque.toml", TrackerConfig)
DEVELOPMENT = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
DERIVATIVES = DerivativeConfig(method="central")
RAW_SOURCE = "raw-20260830-2a97516c354b"
CREATED = "2026-09-04T12:00:00+00:00"
N = 101
DT = 0.01
TASK = TaskIntervals(move=(0.0, 0.8), dwell=(0.8, 1.0))
POINT = RecoveryTrialPoint(
    esn=TrialPoint(80, 0.9, 0.9, 0.3, 0.5, 31, 1e-2, 20.0, 20.0), warmup_s=0.0, augmentation=None
)

_PROTOCOL = """
name = "representative-fixture"
scenario = "{scenario}"
model = "{model}"
formulation = "no_augmentation"
budget = 4
seed_bank = 1
attempt_factor = 4
development = "{development}"

[sampler]
kind = "tpe"
seed = 13
n_startup_trials = 2

[pruner]
kind = "none"

[objective]
kind = "worst_cell_median_gap_ratio"
infeasible_penalty = 10.0

[esn]
n_neurons = {{ low = 50, high = 200, step = 10 }}
spectral_radius = {{ low = 0.8, high = 1.3 }}
sparsity = {{ low = 0.5, high = 0.98 }}
leak_rate = {{ low = 0.01, high = 0.5, log = true }}
input_scaling = {{ low = 0.02, high = 0.6, log = true }}
seed = {{ low = 1, high = 1000 }}
alpha = {{ low = 1e-3, high = 1.0, log = true }}
velocity_cutoff_hz = {{ low = 5.0, high = 30.0, log = true }}
acceleration_cutoff_hz = {{ low = 5.0, high = 30.0, log = true }}

[space]
warmups_s = [0.0, 1.0]
"""


def _samples() -> SampleSet:
    t = np.arange(N, dtype=np.float64) * DT
    start = np.array(SCENARIO.task.initial_q)
    goal = np.array([0.8, 0.4])
    s = np.clip(t / TASK.move[1], 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q = start[None, :] + blend[:, None] * (goal - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = endpoint_positions(SCENARIO, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.where(t < TASK.dwell[0], 1, 2).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((N, 0)), phase)


def _record(samples: SampleSet, payload_sha: str) -> RecoveryDatasetRecord:
    artifact_id = make_artifact_id("processed", CREATED, payload_sha)
    normalization = fit_normalization(
        samples.arrays(),
        ("q", "dq", "ddq", "tip", "dtip", "ddtip"),
        fitted_on=(artifact_id,),
        training_rows=np.ones(samples.n_samples, dtype=np.bool_),
    )
    return RecoveryDatasetRecord(
        artifact=ArtifactRecord(
            artifact_id=artifact_id,
            kind="processed",
            created_at=CREATED,
            license="LicenseRef-Private",
            access="private",
            payload=Payload(f"armrc://processed/{artifact_id}/samples.npz", payload_sha, 2048, "samples.npz", 1),
            origin=Origin(
                command="synthetic representative fixture",
                config_sha256="2" * 64,
                project_commit="a" * 40,
                project_dirty=False,
                dependency_commits={},
                sources=(RAW_SOURCE,),
            ),
        ),
        scenario=Scenario(
            config_path="tests/fixtures/configs/planar_2dof_fixture.toml",
            config_sha256=sha256_file(SCENARIO_FILE),
            robot="planar-2dof-fixture",
            task="pd-reach-fixture",
            dof=2,
            initial_q=tuple(SCENARIO.task.initial_q),
            target=tuple(SCENARIO.task.target),
        ),
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(TASK_PHASE_CODES),
        preprocessing=Preprocessing(
            resample_period_s=DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
        ),
        onset=OnsetAnnotation(
            kind="scripted",
            raw_artifact_id=RAW_SOURCE,
            raw_payload_sha256="b" * 64,
            detector="programmed",
            detector_params={},
            sampling_period_s=DT,
            proposed_onset_sample=100,
            proposed_onset_s=1.0,
            confirmed_onset_sample=100,
            confirmed_onset_s=1.0,
            confirmed_by="script",
        ),
        baseline=BaselineCheck(
            q_pre=tuple(float(v) for v in samples.q[0]), tolerance_rad=0.05, max_deviation_rad=0.0, status="passed"
        ),
        crop=CropWindow(pre_roll=(0.0, 1.0), source_duration_s=2.0, task=TASK),
        q0_ref=tuple(float(v) for v in samples.q[0]),
        arrays=array_specs(samples),
        normalization=normalization,
    )


def _scenarios() -> tuple[RobustnessScenario, ...]:
    return (
        RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
        RobustnessScenario("small-1", "posture_small", (0.02, 0.0), seed=1, draw=0, magnitude_rad=0.05),
        RobustnessScenario("small-2", "posture_small", (-0.02, 0.01), seed=1, draw=1, magnitude_rad=0.05),
        RobustnessScenario("small-3", "posture_small", (0.0, -0.03), seed=1, draw=2, magnitude_rad=0.05),
        RobustnessScenario("large-1", "posture_large", (0.05, -0.02), seed=2, draw=0, magnitude_rad=0.1),
        RobustnessScenario("large-2", "posture_large", (-0.04, 0.04), seed=2, draw=1, magnitude_rad=0.1),
        RobustnessScenario("large-3", "posture_large", (0.06, 0.03), seed=2, draw=2, magnitude_rad=0.1),
        RobustnessScenario(
            "force-000deg",
            "force",
            (0.0, 0.0),
            force_magnitude_n=3.0,
            force_start_s=0.3,
            force_duration_s=0.1,
            direction_deg=0.0,
        ),
    )


def _trial(ratios: dict[str, float] | None = None) -> TrialRecord:
    values = (
        ratios
        if ratios is not None
        else {"small-1": 0.5, "small-2": 0.6, "small-3": 0.7, "large-1": 0.4, "large-2": 0.5, "large-3": 0.8}
    )
    kinds = {
        "small-1": "posture_small",
        "small-2": "posture_small",
        "small-3": "posture_small",
        "large-1": "posture_large",
        "large-2": "posture_large",
        "large-3": "posture_large",
    }
    labels: dict[str, str] = {}
    metrics: dict[str, float] = {}
    for index, (scenario_id, ratio) in enumerate(sorted(values.items())):
        for tracker_offset, tracker in enumerate(("pd_v2", "computed_torque")):
            prefix = f"components.{index * 2 + tracker_offset}"
            labels[prefix + ".kind"] = kinds[scenario_id]
            labels[prefix + ".tracker"] = tracker
            labels[prefix + ".scenario_id"] = scenario_id
            metrics[prefix + ".gap_ratio"] = ratio
    return TrialRecord(
        number=17,
        state="COMPLETE",
        value=0.7,
        params={k: float(v) for k, v in POINT.params().items()},
        metrics=metrics,
        flags={"feasible": True},
        labels=labels,
    )


def _context(record: RecoveryDatasetRecord, samples: SampleSet) -> RecoveryTrialContext:
    artifact_id = record.artifact.artifact_id
    return RecoveryTrialContext(
        scenario=SCENARIO,
        scenario_file=SCENARIO_FILE,
        reference=samples,
        dataset=record,
        source=DatasetSource(artifact_id, record.artifact.payload.sha256, f"data/records/processed/{artifact_id}.toml"),
        trackers={"pd_v2": TRACKER, "computed_torque": TRACKER_CT},
        base_model=load_model_config(MODEL),
        scenarios=_scenarios(),
    )


def _protocol(tmp_path: Path) -> RecoverySearchProtocol:
    file = tmp_path / "representative_fixture.toml"
    file.write_text(
        _PROTOCOL.format(scenario=SCENARIO_FILE.as_posix(), model=MODEL.as_posix(), development=DEVELOPMENT.as_posix()),
        encoding="utf-8",
    )
    return load_recovery_search(file)


def test_selection_is_median_anchored_and_deterministic() -> None:
    """Posture picks sit closest to the class median with ID tie-breaks; nominal and force are fixed."""
    selected = select_scenarios(_trial(), _scenarios())
    # Odd candidate counts make the median an exact element: small-2 (0.6) and large-2 (0.5) sit on it.
    assert selected == {
        "nominal": "nominal",
        "posture_small": "small-2",
        "posture_large": "large-2",
        "force": "force-000deg",
    }
    with pytest.raises(ValueError, match="lacks stored"):
        select_scenarios(_trial({"small-1": 0.5}), _scenarios())


def test_pairs_run_persist_and_install(tmp_path: Path) -> None:
    """Eight pairs run under both trackers, persist to the store, and install pointers plus the summary."""
    samples = _samples()
    staged = tmp_path / "samples.npz"
    save_samples(staged, samples)
    record = _record(samples, sha256_file(staged))
    context = _context(record, samples)
    protocol = _protocol(tmp_path)
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = StorageRoot(store_root, repositories=(REPO_ROOT,))
    result, pointers = run_representatives(
        protocol, context, _trial(), study="representative-fixture", store=store, exploratory=True
    )
    assert len(result.pairs) == 8
    assert len(pointers) == 16
    assert len({p.artifact.artifact_id for p in pointers}) == 16
    assert {(pair.kind, pair.tracker) for pair in result.pairs} == {
        (kind, tracker)
        for kind in ("nominal", "posture_small", "posture_large", "force")
        for tracker in ("pd_v2", "computed_torque")
    }
    assert all(pair.activation_s == 0.0 for pair in result.pairs)
    records_root = tmp_path / "repo"
    (records_root / "data").mkdir(parents=True)
    output = tmp_path / "recovery_representative_v1.json"
    install_representatives(records_root, output, result, pointers)
    tracked = sorted((records_root / "data" / "records" / "runs").glob("*.toml"))
    assert len(tracked) == 16
    catalog = load_catalog(records_root / "data" / "catalog.toml")
    assert all(catalog.find(p.artifact.artifact_id) is not None for p in pointers)
    assert load_representatives(output) == result
    with pytest.raises(FileExistsError, match="refusing"):
        install_representatives(records_root, output, result, pointers)
