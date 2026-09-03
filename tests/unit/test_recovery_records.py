# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-002: recovery dataset records bind onset, pre-roll baseline, crop, and task time immutably."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    ArtifactRecord,
    ChannelStats,
    Normalization,
    Origin,
    Payload,
    Preprocessing,
    ProcessedDatasetRecord,
    Scenario,
    array_specs,
    load_record,
    make_artifact_id,
    to_toml,
    write_record,
)
from arm_rc_ctrl.data.recovery import (
    TASK_PHASE_CODES,
    BaselineCheck,
    CropWindow,
    OnsetAnnotation,
    RecoveryDatasetRecord,
    TaskIntervals,
    load_processed_record,
    recovery_dataset_problems,
    recovery_validation_spec,
    validate_recovery_dataset,
)
from arm_rc_ctrl.data.samples import PHASE_CODES, SampleSet
from arm_rc_ctrl.data.synthetic import synthetic_arrays, synthetic_samples
from arm_rc_ctrl.data.validate import DatasetValidationError, JointLimits
from arm_rc_ctrl.provenance import sha256_file

if TYPE_CHECKING:
    from collections.abc import Callable

CREATED = "2026-09-03T04:00:00+00:00"
PAYLOAD_SHA = "5" * 64
ARTIFACT_ID = make_artifact_id("processed", CREATED, PAYLOAD_SHA)
RAW_SOURCE = "raw-20260830-2a97516c354b"
LIMITS = JointLimits(lower=(-3.0, -3.0), upper=(3.0, 3.0))


def _samples() -> SampleSet:
    arrays = synthetic_arrays(n=6, dof=2, task_dim=2, code_dim=0)
    arrays["phase"] = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    return SampleSet.from_arrays(arrays)


def _artifact(**changes: object) -> ArtifactRecord:
    base = ArtifactRecord(
        artifact_id=ARTIFACT_ID,
        kind="processed",
        created_at=CREATED,
        license="LicenseRef-Private",
        access="private",
        payload=Payload(f"armrc://processed/{ARTIFACT_ID}/samples.npz", PAYLOAD_SHA, 2048, "samples.npz", 1),
        origin=Origin(
            command="python -m arm_rc_ctrl.data.preprocess --config configs/tasks/task_1a.toml",
            config_sha256="2" * 64,
            project_commit="a" * 40,
            project_dirty=False,
            dependency_commits={"rclib": "b" * 40, "skelarm": "c" * 40, "rtctrl": "d" * 40},
            sources=(RAW_SOURCE,),
        ),
        notes="Synthetic example recovery record; the payload does not exist.",
    )
    return dataclasses.replace(base, **changes)


SCENARIO = Scenario(
    config_path="configs/tasks/task_1a.toml",
    config_sha256="2" * 64,
    robot="planar-2dof",
    task="task-1a-reach",
    dof=2,
    initial_q=(0.2, 1.2),
    target=(0.10, 0.45),
)

PREPROCESSING = Preprocessing(
    resample_period_s=0.01,
    smoothing="none",
    smoothing_params={},
    derivative_method="central-difference",
)

ONSET = OnsetAnnotation(
    kind="scripted",
    raw_artifact_id=RAW_SOURCE,
    raw_payload_sha256="b" * 64,
    detector="programmed",
    detector_params={},
    sampling_period_s=0.01,
    proposed_onset_sample=100,
    proposed_onset_s=1.0,
    confirmed_onset_sample=100,
    confirmed_onset_s=1.0,
    confirmed_by="script",
)

HUMAN_ONSET = dataclasses.replace(
    ONSET,
    kind="human",
    detector="speed-threshold",
    detector_params={"threshold_rad_s": 0.05, "hold_s": 0.1},
    proposed_onset_sample=96,
    proposed_onset_s=0.96,
    confirmed_by="human",
)


def _baseline(q_pre: tuple[float, ...], tolerance_rad: float) -> BaselineCheck:
    """A baseline whose recorded deviation is recomputed exactly as the derivation records it."""
    q0_ref = tuple(float(v) for v in _samples().q[0])
    deviation = float(np.max(np.abs(np.asarray(q_pre) - np.asarray(q0_ref))))
    status = "passed" if deviation <= tolerance_rad else "flagged"
    return BaselineCheck(q_pre=q_pre, tolerance_rad=tolerance_rad, max_deviation_rad=deviation, status=status)


_Q0 = tuple(float(v) for v in _samples().q[0])
BASELINE = _baseline((_Q0[0] + 0.0002, _Q0[1] - 0.0004), tolerance_rad=0.01)
FLAGGED = _baseline((0.7, 0.4), tolerance_rad=0.01)

CROP = CropWindow(
    pre_roll=(0.0, 1.0),
    source_duration_s=5.0,
    task=TaskIntervals(move=(0.0, 0.03), dwell=(0.03, 0.05)),
)


def _record(**changes: object) -> RecoveryDatasetRecord:
    samples = _samples()
    base = RecoveryDatasetRecord(
        artifact=_artifact(),
        scenario=SCENARIO,
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(TASK_PHASE_CODES),
        preprocessing=PREPROCESSING,
        onset=ONSET,
        baseline=BASELINE,
        crop=CROP,
        q0_ref=tuple(float(v) for v in samples.q[0]),
        arrays=array_specs(samples),
    )
    return dataclasses.replace(base, **changes)


def test_recovery_record_round_trips_through_toml(tmp_path: Path) -> None:
    """Serialize, reload strictly, and compare for equality."""
    record = _record()
    path = tmp_path / "recovery.toml"
    path.write_text(to_toml(record), encoding="utf-8")
    assert load_record(path, RecoveryDatasetRecord) == record


def _m3_record() -> ProcessedDatasetRecord:
    m3_samples = synthetic_samples()
    return ProcessedDatasetRecord(
        artifact=_artifact(),
        scenario=SCENARIO,
        n_samples=m3_samples.n_samples,
        dof=m3_samples.dof,
        task_dim=m3_samples.task_dim,
        task_code_dim=m3_samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(PHASE_CODES),
        preprocessing=PREPROCESSING,
        arrays=array_specs(m3_samples),
    )


def test_recovery_and_m3_schemas_are_mutually_exclusive(tmp_path: Path) -> None:
    """A recovery TOML never loads as an M3 record and vice versa; the M3 schema is untouched."""
    recovery_path = tmp_path / "recovery.toml"
    recovery_path.write_text(to_toml(_record()), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_record(recovery_path, ProcessedDatasetRecord)
    m3_path = tmp_path / "processed.toml"
    m3_path.write_text(to_toml(_m3_record()), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_record(m3_path, RecoveryDatasetRecord)


def test_load_processed_record_dispatches_between_the_schemas(tmp_path: Path) -> None:
    """The helper returns whichever processed schema the file satisfies and rejects files satisfying neither."""
    recovery_path = tmp_path / "recovery.toml"
    recovery_path.write_text(to_toml(_record()), encoding="utf-8")
    assert isinstance(load_processed_record(recovery_path), RecoveryDatasetRecord)
    m3_path = tmp_path / "processed.toml"
    m3_path.write_text(to_toml(_m3_record()), encoding="utf-8")
    assert isinstance(load_processed_record(m3_path), ProcessedDatasetRecord)
    neither = tmp_path / "neither.toml"
    neither.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_processed_record(neither)


def test_scripted_onset_is_locked_to_the_programmed_time() -> None:
    """A scripted onset carries the programmed time and admits no adjustment."""
    assert ONSET.adjustment_s == 0.0
    assert ONSET.confirmed_onset_s == 1.0


def test_human_onset_records_proposal_confirmation_and_adjustment(tmp_path: Path) -> None:
    """A human onset keeps the detector proposal, the confirmed sample, and their difference."""
    assert HUMAN_ONSET.proposed_onset_s == 0.96
    assert HUMAN_ONSET.confirmed_onset_s == 1.0
    assert HUMAN_ONSET.adjustment_s == pytest.approx(0.04)
    record = _record(onset=HUMAN_ONSET)
    path = tmp_path / "human.toml"
    path.write_text(to_toml(record), encoding="utf-8")
    assert load_record(path, RecoveryDatasetRecord).onset == HUMAN_ONSET


def test_q0_ref_is_the_first_cropped_sample_and_never_q_pre() -> None:
    """A record claiming q_pre as q0_ref fails against the arrays; the flag never rewrites q0_ref."""
    samples = _samples()
    # An imposter must also fake a consistent baseline deviation to get past construction;
    # check_samples still catches the substitution against the actual arrays.
    imposter = _record(q0_ref=BASELINE.q_pre, baseline=dataclasses.replace(BASELINE, max_deviation_rad=0.0))
    with pytest.raises(ValueError, match="never"):
        imposter.check_samples(samples)
    flagged = _record(baseline=FLAGGED)
    assert flagged.q0_ref == tuple(float(v) for v in samples.q[0])
    assert flagged.q0_ref != FLAGGED.q_pre
    flagged.check_samples(samples)


def test_check_samples_detects_drift_and_a_shifted_time_origin() -> None:
    """Matching samples pass; array drift and a non-zero task clock fail."""
    record = _record()
    record.check_samples(_samples())
    arrays = synthetic_arrays(n=6, dof=2, task_dim=2, code_dim=0)
    arrays["phase"] = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    arrays["q"] = arrays["q"] + 0.5
    drifted = SampleSet.from_arrays(arrays)
    with pytest.raises(ValueError, match="do not match"):
        record.check_samples(drifted)
    shifted_arrays = synthetic_arrays(n=6, dof=2, task_dim=2, code_dim=0)
    shifted_arrays["phase"] = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    shifted_arrays["t"] = shifted_arrays["t"] + 0.01
    shifted = SampleSet.from_arrays(shifted_arrays)
    shifted_record = _record(arrays=array_specs(shifted), q0_ref=tuple(float(v) for v in shifted.q[0]))
    with pytest.raises(ValueError, match=r"start at 0.0"):
        shifted_record.check_samples(shifted)


def test_check_scenario_binds_the_dataset_to_a_scenario_file(tmp_path: Path) -> None:
    """The record only accepts the scenario file whose digest it was derived under."""
    scenario_file = tmp_path / "task_1a.toml"
    scenario_file.write_text("name = 'task-1a'\n", encoding="utf-8")
    digest = sha256_file(scenario_file)
    bound = _record(scenario=dataclasses.replace(SCENARIO, config_sha256=digest))
    bound.check_scenario(scenario_file)
    with pytest.raises(ValueError, match="scenario digest"):
        _record().check_scenario(scenario_file)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"phases": dict(PHASE_CODES)}, "phases must be exactly"),
        ({"phases": {"move": 0, "dwell": 1}}, "phases must be exactly"),
        ({"recovery_schema_version": 2}, "unsupported recovery_schema_version 2"),
        ({"q0_ref": (0.2,)}, "q0_ref must have dof=2"),
        ({"units": {**CANONICAL_UNITS, "q": "deg"}}, "units must be exactly"),
        ({"n_samples": 7}, r"arrays.t.shape"),
        ({"baseline": dataclasses.replace(BASELINE, q_pre=(0.2,))}, "baseline.q_pre must have dof=2"),
        ({"crop": dataclasses.replace(CROP, pre_roll=(0.0, 0.9))}, "confirmed onset"),
        ({"onset": dataclasses.replace(ONSET, raw_artifact_id="raw-20260830-cccccccccccc")}, "origin.sources"),
        (
            {"crop": dataclasses.replace(CROP, task=TaskIntervals(move=(0.0, 3.0), dwell=(3.0, 4.0)))},
            "does not match 6 samples",
        ),
        (
            {
                "normalization": Normalization(
                    fitted_on=(ARTIFACT_ID,), channels={"q": ChannelStats(mean=(0.0,), scale=(1.0,))}
                )
            },
            "normalization.channels.q must have 2 columns",
        ),
    ],
)
def test_recovery_record_invariants(changes: dict[str, object], message: str) -> None:
    """Cross-field consistency is enforced at construction."""
    with pytest.raises(ValueError, match=message):
        _record(**changes)


def test_baseline_deviation_is_recomputed_at_construction() -> None:
    """A tampered max_deviation_rad that no longer derives from q_pre and q0_ref is rejected."""
    consistent_status = BASELINE.status
    tampered = BaselineCheck(
        q_pre=BASELINE.q_pre,
        tolerance_rad=BASELINE.tolerance_rad,
        max_deviation_rad=BASELINE.max_deviation_rad / 2,
        status=consistent_status,
    )
    with pytest.raises(ValueError, match="recomputed deviation"):
        _record(baseline=tampered)


def test_crop_task_must_agree_with_the_stored_phase_transition() -> None:
    """check_samples rejects a record whose task intervals disagree with the phase array."""
    samples = _samples()
    shifted_task = TaskIntervals(move=(0.0, 0.02), dwell=(0.02, 0.05))
    record = _record(crop=dataclasses.replace(CROP, task=shifted_task))
    with pytest.raises(ValueError, match="phase transition"):
        record.check_samples(samples)


def test_wrong_kind_is_rejected() -> None:
    """A raw artifact envelope cannot carry a recovery dataset."""
    raw_id = make_artifact_id("raw", CREATED, PAYLOAD_SHA)
    raw_artifact = _artifact(
        artifact_id=raw_id,
        kind="raw",
        payload=Payload(f"armrc://raw/{raw_id}/demo.sklog.npz", PAYLOAD_SHA, 2048, "sklog.npz", 1),
    )
    with pytest.raises(ValueError, match="kind 'processed'"):
        _record(artifact=raw_artifact)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: dataclasses.replace(ONSET, confirmed_onset_sample=101, confirmed_onset_s=1.01),
            "cannot be adjusted",
        ),
        (lambda: dataclasses.replace(ONSET, detector="speed-threshold"), "detector 'programmed'"),
        (lambda: dataclasses.replace(ONSET, detector_params={"x": 1.0}), "no detector parameters"),
        (lambda: dataclasses.replace(ONSET, confirmed_by="human"), "confirmed_by 'script'"),
        (lambda: dataclasses.replace(HUMAN_ONSET, detector="programmed"), "real detector label"),
        (lambda: dataclasses.replace(HUMAN_ONSET, confirmed_by="script"), "confirmed_by 'human'"),
        (
            lambda: dataclasses.replace(
                ONSET, proposed_onset_sample=0, proposed_onset_s=0.0, confirmed_onset_sample=0, confirmed_onset_s=0.0
            ),
            "non-empty pre-roll",
        ),
        (lambda: dataclasses.replace(ONSET, proposed_onset_s=0.5), "raw sample grid"),
        (lambda: dataclasses.replace(ONSET, sampling_period_s=0.0), "sampling_period_s must be positive"),
        (lambda: dataclasses.replace(ONSET, raw_payload_sha256="xyz"), "raw_payload_sha256"),
        (lambda: dataclasses.replace(ONSET, raw_artifact_id="processed-20260830-aaaaaaaaaaaa"), "raw artifact ID"),
        (lambda: dataclasses.replace(BASELINE, max_deviation_rad=0.02), "contradicts"),
        (lambda: dataclasses.replace(FLAGGED, max_deviation_rad=0.0004), "contradicts"),
        (lambda: dataclasses.replace(BASELINE, tolerance_rad=0.0), "tolerance_rad must be positive"),
        (lambda: dataclasses.replace(BASELINE, q_pre=()), "must not be empty"),
        (lambda: dataclasses.replace(CROP, pre_roll=(0.1, 1.0)), "start at 0.0"),
        (lambda: dataclasses.replace(CROP, pre_roll=(0.0, 5.0)), "before the recording ends"),
        (
            lambda: dataclasses.replace(CROP, task=TaskIntervals(move=(0.0, 4.0), dwell=(4.0, 4.5))),
            "cannot exceed",
        ),
    ],
)
def test_sub_record_invariants(build: Callable[[], object], message: str) -> None:
    """Onset, baseline, and crop sub-records validate themselves."""
    with pytest.raises(ValueError, match=message):
        build()


def test_recovery_records_are_immutable_on_disk(tmp_path: Path) -> None:
    """An existing record file is never overwritten; corrections supersede."""
    path = tmp_path / f"{ARTIFACT_ID}.toml"
    write_record(path, _record())
    with pytest.raises(FileExistsError, match="immutable"):
        write_record(path, _record())


def test_recovery_validation_spec_never_requires_a_prime_phase() -> None:
    """The derived spec matches the record and disables the three-phase requirement."""
    spec = recovery_validation_spec(_record(), LIMITS)
    assert spec.require_all_phases is False
    assert (spec.dof, spec.task_dim, spec.task_code_dim, spec.period_s) == (2, 2, 0, 0.01)


def test_valid_recovery_samples_have_no_problems() -> None:
    """A move/dwell episode on a zero-based uniform clock validates cleanly."""
    assert recovery_dataset_problems(_samples(), recovery_validation_spec(_record(), LIMITS)) == []


@pytest.mark.parametrize(
    ("phase", "message"),
    [
        ([0, 1, 1, 2, 2, 2], "prime"),
        ([1, 1, 1, 1, 1, 1], r"missing task phase interval\(s\): \['dwell'\]"),
        ([2, 2, 2, 2, 2, 2], r"missing task phase interval\(s\): \['move'\]"),
    ],
)
def test_invalid_recovery_phase_annotations_are_reported(phase: list[int], message: str) -> None:
    """Prime samples and missing move/dwell intervals are named problems."""
    arrays = synthetic_arrays(n=6, dof=2, task_dim=2, code_dim=0)
    arrays["phase"] = np.array(phase, dtype=np.int64)
    samples = SampleSet.from_arrays(arrays)
    problems = recovery_dataset_problems(samples, recovery_validation_spec(_record(), LIMITS))
    assert any(re.search(message, problem) for problem in problems)
    with pytest.raises(DatasetValidationError, match=message):
        validate_recovery_dataset(samples, recovery_validation_spec(_record(), LIMITS))
