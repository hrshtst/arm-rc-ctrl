# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-003: the recovery derivation crops at the programmed onset with full pre-roll filter context."""

from __future__ import annotations

import dataclasses
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.data.records import Intervals, RawDemonstrationRecord, load_catalog, load_record, to_toml
from arm_rc_ctrl.data.recover import RecoverError, main, recover_demonstration
from arm_rc_ctrl.data.recovery import TASK_PHASE_CODES, RecoveryDatasetRecord
from arm_rc_ctrl.data.samples import PHASE_DWELL, PHASE_MOVE, load_samples
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
FIXED_TIME = datetime(2026, 9, 3, 7, 0, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> StorageRoot:
    """Storage root holding the raw fixture payload at its recorded URI."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    record = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(record.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    return store


@pytest.fixture
def records_root(tmp_path: Path) -> Path:
    """A repository-like root for Git-tracked records."""
    root = tmp_path / "repo"
    (root / "data" / "records" / "processed").mkdir(parents=True)
    return root


def _config(tmp_path: Path, tolerance_rad: float) -> Path:
    path = tmp_path / f"recovery-{tolerance_rad!r}.toml"
    path.write_text(
        "[smoothing]\n"
        'method = "butterworth"\n'
        "cutoff_hz = 5.0\n"
        "order = 4\n\n"
        "[resampling]\n"
        'interpolation = "linear"\n\n'
        "[derivatives]\n"
        'method = "central"\n\n'
        "[normalization]\n"
        'channels = ["q", "dq", "ddq", "tip", "dtip", "ddtip"]\n'
        "near_zero = 1e-8\n\n"
        "[baseline]\n"
        'estimator = "median"\n'
        f"tolerance_rad = {tolerance_rad!r}\n",
        encoding="utf-8",
    )
    return path


def _run(store: StorageRoot, records_root: Path, config: Path, raw_record: Path = RAW_RECORD):  # noqa: ANN202
    return recover_demonstration(
        raw_record, SCENARIO, config, store=store, records_root=records_root, exploratory=True, now=FIXED_TIME
    )


def test_recovery_derivation_crops_at_the_programmed_onset(
    store: StorageRoot, records_root: Path, tmp_path: Path
) -> None:
    """The 31-frame fixture with a 0.1 s pre-roll becomes a 21-sample move/dwell episode at task time zero."""
    result = _run(store, records_root, _config(tmp_path, 1.0))
    record = result.record
    assert record.artifact.artifact_id.startswith("processed-20260903-")
    assert record.n_samples == 21
    assert record.phases == TASK_PHASE_CODES

    assert record.onset.kind == "scripted"
    assert record.onset.detector == "programmed"
    assert record.onset.detector_params == {}
    assert record.onset.confirmed_by == "script"
    assert record.onset.proposed_onset_sample == 10
    assert record.onset.confirmed_onset_sample == 10
    assert record.onset.confirmed_onset_s == pytest.approx(0.1)
    assert record.onset.raw_artifact_id == "raw-20260830-287036d83d46"
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    assert record.onset.raw_payload_sha256 == raw.artifact.payload.sha256

    assert record.crop.pre_roll == (0.0, record.onset.confirmed_onset_s)
    assert record.crop.source_duration_s == raw.duration_s
    assert record.crop.task.move[0] == 0.0
    assert record.crop.task.move[1] == pytest.approx(0.15)
    assert record.crop.task.duration_s == pytest.approx(0.2)

    payload = store.path(record.artifact.payload.uri, mode="read")
    samples = load_samples(payload)
    record.check_samples(samples)
    assert float(samples.t[0]) == 0.0
    assert np.allclose(np.diff(samples.t), 0.01)
    assert sorted(set(samples.phase.tolist())) == sorted(TASK_PHASE_CODES.values())
    assert int(np.count_nonzero(samples.phase == PHASE_MOVE)) == 15
    assert int(np.count_nonzero(samples.phase == PHASE_DWELL)) == 6
    assert record.q0_ref == tuple(float(v) for v in samples.q[0])

    assert record.baseline.status == "passed"
    assert len(record.baseline.q_pre) == 2
    assert record.baseline.max_deviation_rad >= 0.0
    assert record.baseline.tolerance_rad == 1.0

    assert record.normalization is not None
    assert record.normalization.fitted_on == (record.artifact.artifact_id,)
    assert (payload.parent / "provenance.json").is_file()
    assert not any(p.name.startswith("staging-") for p in (store.root / "processed").iterdir())
    assert load_record(result.record_file, RecoveryDatasetRecord) == record
    entry = load_catalog(records_root / "data" / "catalog.toml").find(record.artifact.artifact_id)
    assert entry is not None
    assert entry.uri == record.artifact.payload.uri


def test_a_material_baseline_difference_is_flagged_never_substituted(
    store: StorageRoot, records_root: Path, tmp_path: Path
) -> None:
    """A tiny tolerance flags the moving fixture pre-roll; q0_ref stays the first cropped sample."""
    result = _run(store, records_root, _config(tmp_path, 1e-9))
    record = result.record
    assert record.baseline.status == "flagged"
    assert record.baseline.max_deviation_rad > record.baseline.tolerance_rad
    samples = load_samples(store.path(record.artifact.payload.uri, mode="read"))
    assert record.q0_ref == tuple(float(v) for v in samples.q[0])
    assert record.q0_ref != record.baseline.q_pre
    record.check_samples(samples)


def test_rebuild_in_a_fresh_store_reproduces_the_digest(store: StorageRoot, records_root: Path, tmp_path: Path) -> None:
    """Identical inputs give an identical payload digest, arrays, and record in an unrelated store."""
    config = _config(tmp_path, 1.0)
    first = _run(store, records_root, config)
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    other_root = tmp_path / "store2"
    other_root.mkdir()
    other_store = StorageRoot(other_root, repositories=(REPO_ROOT,))
    other_store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    other_records = tmp_path / "repo2"
    (other_records / "data" / "records" / "processed").mkdir(parents=True)
    second = _run(other_store, other_records, config)
    assert second.record.artifact.payload.sha256 == first.record.artifact.payload.sha256
    assert second.record == first.record
    assert not second.resumed


def test_rerun_resumes_the_finalized_artifact_without_rewriting(
    store: StorageRoot, records_root: Path, tmp_path: Path
) -> None:
    """A second run over the same store and records adopts the identical artifact."""
    config = _config(tmp_path, 1.0)
    first = _run(store, records_root, config)
    before = first.record_file.read_bytes()
    second = _run(store, records_root, config)
    assert second.resumed
    assert second.record == first.record
    assert first.record_file.read_bytes() == before


def test_an_onset_off_the_raw_sample_grid_is_rejected(store: StorageRoot, records_root: Path, tmp_path: Path) -> None:
    """A programmed onset that does not land on a raw sample cannot define the crop."""
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    shifted = dataclasses.replace(raw, intervals=Intervals(prime=(0.0, 0.095), move=(0.095, 0.25), dwell=(0.25, 0.3)))
    modified = tmp_path / "raw-modified.toml"
    modified.write_text(to_toml(shifted), encoding="utf-8")
    with pytest.raises(RecoverError, match="grid"):
        _run(store, records_root, _config(tmp_path, 1.0), raw_record=modified)


def test_command_line_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI resolves the store from the environment and derives with the versioned default config."""
    root = tmp_path / "store"
    root.mkdir()
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    StorageRoot(root, repositories=(REPO_ROOT,)).path(raw.artifact.payload.uri, mode="write").write_bytes(
        RAW_LOG.read_bytes()
    )
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(root))
    fake_repo = tmp_path / "repo"
    (fake_repo / "data" / "records" / "processed").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "pyproject.toml", fake_repo / "pyproject.toml")
    (fake_repo / "src" / "arm_rc_ctrl").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", fake_repo / "configs")
    monkeypatch.setattr("arm_rc_ctrl.data.recover.repository_root", lambda: fake_repo)
    assert main(["--raw", str(RAW_RECORD), "--scenario", str(SCENARIO), "--exploratory", "--access", "internal"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["n_samples"] == 21
    written = load_record(fake_repo / printed["record"], RecoveryDatasetRecord)
    assert written.artifact.access == "internal"
    assert written.artifact.origin.command.startswith("python -m arm_rc_ctrl.data.recover --raw")
