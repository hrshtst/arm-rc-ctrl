# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-011: the recovery safety pilot sweeps replay baselines on the recovery schedule."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

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
from arm_rc_ctrl.experiments.perturbation_pilot import select_levels, summarize_levels
from arm_rc_ctrl.experiments.recovery_pilot import (
    as_core,
    force_pulse_on_run_clock,
    load_recovery_pilot_protocol,
    load_recovery_pilot_report,
    main,
    render_recovery_markdown,
    run_recovery_pilot,
)
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
DERIVATIVES = DerivativeConfig(method="central")
RAW_SOURCE = "raw-20260830-2a97516c354b"
CREATED = "2026-09-03T12:00:00+00:00"
N = 101
DT = 0.01
TASK = TaskIntervals(move=(0.0, 0.8), dwell=(0.8, 1.0))
FIXED_TIME = datetime(2026, 9, 3, 12, 30, 0, tzinfo=UTC)


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
                command="synthetic recovery pilot fixture",
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


def _protocol_file(directory: Path) -> Path:
    file = directory / "recovery_pilot_fixture.toml"
    file.write_text(
        f'name = "recovery-pilot-fixture"\nscenario = "{SCENARIO_FILE.as_posix()}"\n'
        'baselines = ["pd", "computed_torque"]\nwarmup_s = 0.25\n'
        "[posture]\nmagnitudes = [0.005, 0.01, 0.02]\ndirections = [[1.0, 0.0], [0.0, -1.0]]\n"
        "[force]\nmagnitudes = [0.5, 2.0]\ndirections_deg = [0.0, 180.0]\nstart_s = 0.3\nduration_s = 0.05\n"
        "[selection]\nposture_recovery_min_s = 0.02\nposture_recovery_max_s = 1.2\nforce_recovery_max_s = 1.2\n"
        "force_deviation_min_m = 0.002\nforce_max_saturation_fraction = 1.0\n",
        encoding="utf-8",
    )
    return file


def test_off_approved_warmups_are_rejected(tmp_path: Path) -> None:
    """The pilot's common hold must be an approved D2 duration."""
    file = _protocol_file(tmp_path)
    text = file.read_text(encoding="utf-8").replace("warmup_s = 0.25", "warmup_s = 0.3")
    file.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        load_recovery_pilot_protocol(file)


def test_force_pulses_run_on_the_task_clock(tmp_path: Path) -> None:
    """The run-clock pulse of every force case starts at warmup + task-relative start."""
    protocol = load_recovery_pilot_protocol(_protocol_file(tmp_path))
    pulse = force_pulse_on_run_clock(protocol, 2.0, 180.0)
    assert pulse.start_s == pytest.approx(protocol.warmup_s + protocol.force.start_s)
    assert pulse.duration_s == protocol.force.duration_s


def test_pilot_sweeps_the_recovery_schedule(tmp_path: Path) -> None:
    """Every case starts from q0_ref (+ offset), levels re-derive, and the selection is reproducible."""
    samples = _samples()
    staged = tmp_path / "samples.npz"
    save_samples(staged, samples)
    record = _record(samples, sha256_file(staged), staged.stat().st_size)
    protocol = load_recovery_pilot_protocol(_protocol_file(tmp_path))
    report = run_recovery_pilot(protocol, _protocol_file(tmp_path), record, samples, exploratory=True, now=FIXED_TIME)
    assert report.warmup_s == 0.25
    assert len(report.cases) == 2 * (3 * 2 + 2 * 2)
    q0 = np.asarray(record.q0_ref)
    posture_cases = [c for c in report.cases if c.kind == "posture"]
    for case in posture_cases:
        offset = np.asarray(case.initial_q) - q0
        assert float(np.sqrt(np.sum(offset * offset))) == pytest.approx(case.magnitude)
    force_cases = [c for c in report.cases if c.kind == "force"]
    assert all(tuple(case.initial_q) == tuple(record.q0_ref) for case in force_cases)
    core = as_core(protocol)
    assert report.levels == summarize_levels(core, report.cases)
    assert report.selection == select_levels(core, report.levels)
    markdown = render_recovery_markdown(report)
    assert "recovery-pilot-fixture" in markdown
    assert "task clock" in markdown
    assert f"{protocol.warmup_s:g}" in markdown


def test_command_line_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI verifies the payload, writes the report and markdown, and refuses to overwrite."""
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
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store_root))
    report_file = tmp_path / "pilot.json"
    markdown_file = tmp_path / "pilot.md"
    code = main(
        [
            "--protocol",
            str(_protocol_file(tmp_path)),
            "--dataset",
            str(record_file),
            "--report",
            str(report_file),
            "--markdown",
            str(markdown_file),
            "--exploratory",
        ]
    )
    assert code == 0
    loaded = load_recovery_pilot_report(report_file)
    assert loaded.dataset == record.artifact.artifact_id
    assert loaded.warmup_s == 0.25
    assert markdown_file.read_text(encoding="utf-8") == render_recovery_markdown(loaded)
    payload = json.loads(report_file.read_text(encoding="utf-8"))
    assert payload["provenance"]["exploratory"] is True
    with pytest.raises(FileExistsError, match=r"refusing|exists"):
        main(
            [
                "--protocol",
                str(_protocol_file(tmp_path)),
                "--dataset",
                str(record_file),
                "--report",
                str(report_file),
            ]
        )
