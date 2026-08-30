# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-010: the preprocessing command writes the payload atomically, then the record; never overwrites."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.data.preprocess import PreprocessError, main, preprocess_demonstration
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, RawDemonstrationRecord, load_catalog, load_record
from arm_rc_ctrl.data.samples import PHASE_CODES, load_samples
from arm_rc_ctrl.data.validate import DatasetValidationError
from arm_rc_ctrl.provenance import ArtifactMismatchError, DirtyWorktreeError
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot, StorageRootError

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
CONFIG = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 8, 30, 7, 0, 0, tzinfo=UTC)


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


def _run(store: StorageRoot, records_root: Path):  # noqa: ANN202
    return preprocess_demonstration(
        RAW_RECORD, SCENARIO, CONFIG, store=store, records_root=records_root, exploratory=True, now=FIXED_TIME
    )


def _processed_entries(store: StorageRoot) -> list[str]:
    bucket = store.root / "processed"
    return sorted(p.name for p in bucket.iterdir()) if bucket.exists() else []


def test_preprocessing_produces_payload_record_and_catalog(store: StorageRoot, records_root: Path) -> None:
    """The fixture becomes a validated 31-sample dataset with a content-addressed record."""
    result = _run(store, records_root)
    record = result.record
    assert record.artifact.artifact_id.startswith("processed-20260830-")
    assert record.n_samples == 31
    assert (record.dof, record.task_dim, record.task_code_dim) == (2, 2, 0)
    assert record.artifact.origin.sources == ("raw-20260830-287036d83d46",)
    assert record.scenario == load_record(RAW_RECORD, RawDemonstrationRecord).scenario
    record.check_scenario(SCENARIO)
    assert record.artifact.license == "GPL-3.0-only"
    assert record.artifact.access == "public"
    assert record.preprocessing.smoothing == "butterworth-zero-phase"
    assert record.preprocessing.interpolation == "linear"
    assert record.preprocessing.derivative_method == "central-difference"
    assert record.normalization is not None
    assert record.normalization.fitted_on == (record.artifact.artifact_id,)
    assert set(record.normalization.channels) == {"q", "dq", "ddq", "tip", "dtip", "ddtip"}

    payload = store.path(record.artifact.payload.uri, mode="read")
    assert payload == result.payload_file
    samples = load_samples(payload)
    record.check_samples(samples)
    assert samples.t[0] == 0.0
    assert np.allclose(np.diff(samples.t), 0.01)
    assert sorted(set(samples.phase.tolist())) == sorted(PHASE_CODES.values())
    assert (payload.parent / "provenance.json").is_file()
    assert not any(p.name.startswith("staging-") for p in (store.root / "processed").iterdir())

    assert result.record_file == records_root / "data" / "records" / "processed" / f"{record.artifact.artifact_id}.toml"
    assert load_record(result.record_file, ProcessedDatasetRecord) == record
    catalog = load_catalog(records_root / "data" / "catalog.toml")
    entry = catalog.find(record.artifact.artifact_id)
    assert entry is not None
    assert entry.uri == record.artifact.payload.uri
    assert entry.record == f"data/records/processed/{record.artifact.artifact_id}.toml"


def test_rerun_is_deterministic_and_refuses_to_overwrite(
    store: StorageRoot, records_root: Path, tmp_path: Path
) -> None:
    """Identical inputs give identical arrays and digests; the existing artifact is never overwritten."""
    first = _run(store, records_root)
    other_records = tmp_path / "repo2"
    (other_records / "data" / "records" / "processed").mkdir(parents=True)
    other_store_root = tmp_path / "store2"
    other_store_root.mkdir()
    other_store = StorageRoot(other_store_root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    other_store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    second = _run(other_store, other_records)
    assert second.record.artifact.artifact_id == first.record.artifact.artifact_id
    assert second.record.arrays == first.record.arrays
    assert second.record.artifact.payload.sha256 == first.record.artifact.payload.sha256

    payload_before = first.payload_file.read_bytes()
    record_before = first.record_file.read_text()
    with pytest.raises(FileExistsError, match="datasets are immutable"):
        _run(store, records_root)
    assert first.payload_file.read_bytes() == payload_before
    assert first.record_file.read_text() == record_before
    assert not any(p.name.startswith("staging-") for p in (store.root / "processed").iterdir())


def test_failures_leave_no_payload_or_record_behind(store: StorageRoot, records_root: Path, tmp_path: Path) -> None:
    """Mismatched scenario, dirty worktree, tampered payload, and validation failures abort cleanly."""
    other_scenario = tmp_path / "other.toml"
    other_scenario.write_text(SCENARIO.read_text().replace("tolerance = 0.004", "tolerance = 0.005"))
    with pytest.raises(PreprocessError, match="recorded under scenario digest"):
        preprocess_demonstration(
            RAW_RECORD, other_scenario, CONFIG, store=store, records_root=records_root, exploratory=True, now=FIXED_TIME
        )

    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    payload = store.path(raw.artifact.payload.uri, mode="read")
    original = payload.read_bytes()
    payload.write_bytes(original + b"x")
    with pytest.raises(ArtifactMismatchError):
        _run(store, records_root)
    payload.write_bytes(original)

    strict = tmp_path / "strict.toml"
    strict.write_text(SCENARIO.read_text().replace("velocity = [20.0, 20.0]", "velocity = [0.001, 0.001]"))
    strict_raw = tmp_path / "raw.toml"
    text = RAW_RECORD.read_text()
    from arm_rc_ctrl.provenance import sha256_file

    strict_raw.write_text(text.replace(raw.scenario.config_sha256, sha256_file(strict)))
    with pytest.raises(DatasetValidationError, match="exceeds the speed limit"):
        preprocess_demonstration(
            strict_raw, strict, CONFIG, store=store, records_root=records_root, exploratory=True, now=FIXED_TIME
        )

    assert list((records_root / "data" / "records" / "processed").iterdir()) == []
    assert not (records_root / "data" / "catalog.toml").exists()
    assert _processed_entries(store) == []


def test_dirty_worktree_is_rejected_unless_exploratory(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clean-worktree policy applies before any output is written."""

    def dirty(_root: Path) -> tuple[str, bool]:
        return "0" * 40, True

    monkeypatch.setattr("arm_rc_ctrl.provenance.worktree_state", dirty)
    with pytest.raises(DirtyWorktreeError):
        preprocess_demonstration(
            RAW_RECORD, SCENARIO, CONFIG, store=store, records_root=records_root, exploratory=False, now=FIXED_TIME
        )
    assert _processed_entries(store) == []
    result = _run(store, records_root)
    assert result.record.artifact.origin.project_dirty is True


def test_storage_root_inside_the_repository_is_rejected() -> None:
    """There is no repository fallback for payloads."""
    inside = REPO_ROOT / "data"
    with pytest.raises(StorageRootError, match="inside the repository worktree"):
        StorageRoot(inside)


def test_command_line_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI resolves the store from the environment and writes into the repository's data/ tree."""
    root = tmp_path / "store"
    root.mkdir()
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    StorageRoot(root, repositories=(REPO_ROOT,)).path(raw.artifact.payload.uri, mode="write").write_bytes(
        RAW_LOG.read_bytes()
    )
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(root))
    # Redirect the Git-tracked outputs to a scratch copy of the repository layout.
    fake_repo = tmp_path / "repo"
    (fake_repo / "data" / "records" / "processed").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "pyproject.toml", fake_repo / "pyproject.toml")
    (fake_repo / "src" / "arm_rc_ctrl").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", fake_repo / "configs")
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.repository_root", lambda: fake_repo)
    assert main(["--raw", str(RAW_RECORD), "--scenario", str(SCENARIO), "--exploratory", "--access", "internal"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["n_samples"] == 31
    written = load_record(fake_repo / printed["record"], ProcessedDatasetRecord)
    assert written.artifact.access == "internal"
    assert written.artifact.origin.command.startswith("python -m arm_rc_ctrl.data.preprocess --raw")
