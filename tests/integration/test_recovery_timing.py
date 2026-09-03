# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-009: the timing-trace generator verifies the schedule contracts and writes the review set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.estimator import EstimatorConfig
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
    make_artifact_id,
    to_toml,
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
from arm_rc_ctrl.experiments.recovery_timing import (
    FORCE_PULSE_TASK,
    PERTURBATION_RAD,
    main,
    resolved_timing_config,
)
from arm_rc_ctrl.provenance import canonical_json, sha256_file
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
DERIVATIVES = DerivativeConfig(method="central")
RAW_SOURCE = "raw-20260830-2a97516c354b"
CREATED = "2026-09-03T11:00:00+00:00"
N = 101
DT = 0.01
TASK = TaskIntervals(move=(0.0, 0.8), dwell=(0.8, 1.0))

MODEL_TOML = """name = "timing-test"

[esn.reservoir]
n_neurons = 60
spectral_radius = 0.85
sparsity = 0.9
leak_rate = 0.4
input_scaling = 0.4
seed = 17
include_bias = true

[esn.readout]
alpha = 1e-6

[input_transform]
policy = "fixed_scale"
q_scale = 0.3
dq_scale = 4.0
"""

EVALUATION_TOML = """name = "timing-test-eval"
tracker = "unused.toml"

[estimator]
max_dt_ratio = 3.0
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


def _record(samples: SampleSet, payload_sha: str, payload_size: int) -> RecoveryDatasetRecord:
    artifact_id = make_artifact_id("processed", CREATED, payload_sha)
    normalization = fit_normalization(
        samples.arrays(),
        ("q", "dq", "ddq", "tip", "dtip", "ddtip"),
        fitted_on=(artifact_id,),
        training_rows=np.ones(samples.n_samples, dtype=np.bool_),
    )
    q0 = tuple(float(v) for v in samples.q[0])
    return RecoveryDatasetRecord(
        artifact=ArtifactRecord(
            artifact_id=artifact_id,
            kind="processed",
            created_at=CREATED,
            license="LicenseRef-Private",
            access="private",
            payload=Payload(
                f"armrc://processed/{artifact_id}/samples.npz", payload_sha, payload_size, "samples.npz", 1
            ),
            origin=Origin(
                command="synthetic timing fixture",
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
        baseline=BaselineCheck(q_pre=q0, tolerance_rad=0.05, max_deviation_rad=0.0, status="passed"),
        crop=CropWindow(pre_roll=(0.0, 1.0), source_duration_s=2.0, task=TASK),
        q0_ref=q0,
        arrays=array_specs(samples),
        normalization=normalization,
    )


def test_provenance_identity_binds_model_estimator_and_tracker(tmp_path: Path) -> None:
    """Changing the resolved model, estimator, or tracker changes the provenance identity."""
    samples = _samples()
    staged = tmp_path / "samples.npz"
    save_samples(staged, samples)
    record = _record(samples, sha256_file(staged), staged.stat().st_size)
    model_file = tmp_path / "model.toml"
    model_file.write_text(MODEL_TOML, encoding="utf-8")
    model = load_model_config(model_file)
    other_model_file = tmp_path / "model2.toml"
    other_model_file.write_text(MODEL_TOML.replace("seed = 17", "seed = 18"), encoding="utf-8")
    other_model = load_model_config(other_model_file)
    estimator = EstimatorConfig(nominal_dt_s=DT)
    tracker = load_config(REPO_ROOT / "configs" / "controllers" / "pd.toml", TrackerConfig)
    other_tracker = load_config(REPO_ROOT / "configs" / "controllers" / "task_1a_pd_v2.toml", TrackerConfig)
    base = resolved_timing_config(record, SCENARIO_FILE, model, estimator, tracker, "cmd")
    assert {"model", "estimator", "tracker", "dataset", "scenario"} <= set(base)
    variants = (
        resolved_timing_config(record, SCENARIO_FILE, other_model, estimator, tracker, "cmd"),
        resolved_timing_config(
            record, SCENARIO_FILE, model, EstimatorConfig(nominal_dt_s=DT, velocity_cutoff_hz=20.0), tracker, "cmd"
        ),
        resolved_timing_config(record, SCENARIO_FILE, model, estimator, other_tracker, "cmd"),
    )
    for variant in variants:
        assert canonical_json(variant) != canonical_json(base)


def test_generator_writes_verified_traces_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All four cases run, verify their schedule contracts, and produce the plot set and summary."""
    store_root = tmp_path / "store"
    store_root.mkdir()
    samples = _samples()
    staged = tmp_path / "samples.npz"
    save_samples(staged, samples)
    record = _record(samples, sha256_file(staged), staged.stat().st_size)
    store = StorageRoot(store_root, repositories=(REPO_ROOT,))
    store.path(record.artifact.payload.uri, mode="write").write_bytes(staged.read_bytes())
    record_file = tmp_path / f"{record.artifact.artifact_id}.toml"
    record_file.write_text(to_toml(record), encoding="utf-8")
    model_file = tmp_path / "model.toml"
    model_file.write_text(MODEL_TOML, encoding="utf-8")
    evaluation_file = tmp_path / "evaluation.toml"
    evaluation_file.write_text(EVALUATION_TOML, encoding="utf-8")
    tracker_file = REPO_ROOT / "configs" / "controllers" / "pd.toml"
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store_root))
    out = tmp_path / "traces"
    code = main(
        [
            "--dataset",
            str(record_file),
            "--scenario",
            str(SCENARIO_FILE),
            "--model",
            str(model_file),
            "--evaluation",
            str(evaluation_file),
            "--tracker",
            str(tracker_file),
            "--out",
            str(out),
            "--exploratory",
        ]
    )
    assert code == 0
    summary = json.loads((out / "timing_traces.json").read_text(encoding="utf-8"))
    cases = summary["cases"]
    assert sorted(cases) == ["force_tw1", "nominal_tw0", "nominal_tw1", "perturbed_tw1"]
    for entry in cases.values():
        assert (out / entry["plot"]).is_file()
        assert entry["verified"]
        assert entry["replay_criteria"]["completed"] is True
        assert entry["rc_criteria"]["completed"] is True
    assert cases["perturbed_tw1"]["initial_q"] == [record.q0_ref[j] + PERTURBATION_RAD[j] for j in range(2)]
    assert cases["force_tw1"]["force_task"]["start_s"] == FORCE_PULSE_TASK.start_s
    assert any("perturbed posture" in fact for fact in cases["perturbed_tw1"]["verified"])
    assert any("no force during the hold" in fact for fact in cases["force_tw1"]["verified"])
    provenance = summary["provenance"]
    assert record.artifact.artifact_id in json.dumps(provenance)  # the dataset is bound into provenance
    resolved = json.loads(provenance["config_json"])
    assert {"model", "estimator", "tracker"} <= set(resolved)  # resolved inputs bound, not just file names
    with pytest.raises(FileExistsError, match="immutable"):
        main(["--dataset", str(record_file), "--scenario", str(SCENARIO_FILE), "--out", str(out)])
