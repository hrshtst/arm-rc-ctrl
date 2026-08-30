# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-010: the preprocessing command writes the payload atomically, then the record; never overwrites."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.data.preprocess import PreprocessError, PreprocessResult, main, preprocess_demonstration
from arm_rc_ctrl.data.records import (
    Origin,
    ProcessedDatasetRecord,
    RawDemonstrationRecord,
    load_catalog,
    load_record,
)
from arm_rc_ctrl.data.samples import PHASE_CODES, load_samples
from arm_rc_ctrl.data.validate import DatasetValidationError
from arm_rc_ctrl.provenance import ArtifactMismatchError, DirtyWorktreeError, ProvenanceRecord, canonical_json
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

    payload_before = (first.payload_file.read_bytes(), first.payload_file.stat().st_mtime_ns)
    record_before = first.record_file.read_text()
    again = _run(store, records_root)  # identical inputs: verified no-op, nothing rewritten
    assert again.resumed is True
    assert again.record == first.record
    assert (first.payload_file.read_bytes(), first.payload_file.stat().st_mtime_ns) == payload_before
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


def _fail(*_args: object, **_kwargs: object) -> None:
    msg = "injected failure"
    raise RuntimeError(msg)


def test_retry_completes_after_record_write_failure(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after the payload rename leaves a verified payload; the retry writes record and catalog."""
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.write_record", _fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        _run(store, records_root)
    orphan = [p for p in _processed_entries(store) if p.startswith("processed-")]
    assert len(orphan) == 1
    assert list((records_root / "data" / "records" / "processed").iterdir()) == []
    assert not (records_root / "data" / "catalog.toml").exists()
    monkeypatch.undo()

    payload = store.root / "processed" / orphan[0] / "samples.npz"
    before = (payload.read_bytes(), payload.stat().st_mtime_ns)
    result = _run(store, records_root)
    assert result.resumed is True
    assert result.record.artifact.artifact_id == orphan[0]
    assert (payload.read_bytes(), payload.stat().st_mtime_ns) == before  # payload reused, not rewritten
    assert result.record_file.is_file()
    assert load_catalog(records_root / "data" / "catalog.toml").find(orphan[0]) is not None
    assert _processed_entries(store) == orphan  # no second copy, no staging left


def test_retry_completes_after_catalog_failure_without_rewriting_the_record(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure while appending the catalog is completed by the retry; the record bytes stay identical."""
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.write_catalog", _fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        _run(store, records_root)
    monkeypatch.undo()
    (record_file,) = (records_root / "data" / "records" / "processed").iterdir()
    before = record_file.read_bytes()
    assert not (records_root / "data" / "catalog.toml").exists()

    result = _run(store, records_root)
    assert result.resumed is True
    assert record_file.read_bytes() == before
    assert load_catalog(records_root / "data" / "catalog.toml").find(result.record.artifact.artifact_id) is not None
    # A third invocation is a clean no-op resume as well.
    again = _run(store, records_root)
    assert again.resumed is True
    assert record_file.read_bytes() == before


def test_resume_refuses_a_corrupted_or_foreign_partial_result(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a byte-identical payload (and a record describing it) can be resumed."""
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.write_record", _fail)
    with pytest.raises(RuntimeError):
        _run(store, records_root)
    monkeypatch.undo()
    (orphan,) = _processed_entries(store)
    payload = store.root / "processed" / orphan / "samples.npz"
    payload.write_bytes(payload.read_bytes() + b"\0")
    with pytest.raises(FileExistsError, match="payload differs"):
        _run(store, records_root)
    assert _processed_entries(store) == [orphan]  # staging cleaned up, orphan left for inspection


def _run_at(store: StorageRoot, records_root: Path, when: datetime, **overrides: object) -> PreprocessResult:
    return preprocess_demonstration(
        RAW_RECORD,
        SCENARIO,
        CONFIG,
        store=store,
        records_root=records_root,
        exploratory=True,
        now=when,
        **overrides,  # type: ignore[arg-type]
    )


def test_retry_at_a_later_time_keeps_the_finalized_provenance(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After an interrupted write, a later retry returns the stored provenance and a record built from it."""
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.write_record", _fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        _run_at(store, records_root, FIXED_TIME)
    monkeypatch.undo()
    (orphan,) = _processed_entries(store)
    stored = ProvenanceRecord.from_json((store.root / "processed" / orphan / "provenance.json").read_text())
    assert stored.created_at == FIXED_TIME.isoformat()

    later = _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1))
    assert later.resumed is True
    assert later.provenance == stored  # the retry's own (07:01) provenance is discarded
    assert later.record.artifact.created_at == stored.created_at
    assert later.record.artifact.origin == Origin.from_provenance(
        stored, command="python -m arm_rc_ctrl.data.preprocess", sources=(later.record.artifact.origin.sources[0],)
    )
    assert load_record(later.record_file, ProcessedDatasetRecord) == later.record
    entry = load_catalog(records_root / "data" / "catalog.toml").find(orphan)
    assert entry is not None
    assert entry.created_at == stored.created_at

    # A further retry with a record already present is a no-op that still reports the stored provenance.
    before = later.record_file.read_bytes()
    again = _run_at(store, records_root, FIXED_TIME + timedelta(minutes=2))
    assert again.resumed is True
    assert again.provenance == stored
    assert again.record == later.record
    assert later.record_file.read_bytes() == before


def test_resume_refuses_different_metadata_or_a_tampered_record(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry that would describe the same payload differently, or a record that was edited, is refused."""
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.write_catalog", _fail)
    with pytest.raises(RuntimeError):
        _run_at(store, records_root, FIXED_TIME)
    monkeypatch.undo()
    (record_file,) = (records_root / "data" / "records" / "processed").iterdir()
    with pytest.raises(
        FileExistsError, match=r"recorded with license 'GPL-3\.0-only', and this retry asks for 'CC-BY-4\.0'"
    ):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1), license_override="CC-BY-4.0")
    text = record_file.read_text()
    assert text.count('access = "public"') == 1
    record_file.write_text(text.replace('access = "public"', 'access = "internal"'))  # edited after the fact
    with pytest.raises(FileExistsError, match=r"differs in artifact\.access"):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1))


def test_resume_refuses_missing_or_foreign_stored_provenance(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finalized provenance must load strictly and describe the same inputs before it is adopted."""
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.write_record", _fail)
    with pytest.raises(RuntimeError):
        _run_at(store, records_root, FIXED_TIME)
    monkeypatch.undo()
    (orphan,) = _processed_entries(store)
    provenance_file = store.root / "processed" / orphan / "provenance.json"
    original = provenance_file.read_text()

    provenance_file.write_text(original[: len(original) // 2])  # truncated JSON
    with pytest.raises(FileExistsError, match=r"provenance\.json is missing or invalid"):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1))

    data = json.loads(original)
    data["config_sha256"] = "0" * 64  # digest no longer matches the canonical configuration
    provenance_file.write_text(json.dumps(data))
    with pytest.raises(FileExistsError, match=r"provenance\.json is missing or invalid"):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1))

    data = json.loads(original)
    config = json.loads(data["config_json"])
    config["raw_artifact"] = "raw-20260830-000000000000"
    data["config_json"] = canonical_json(config)
    data["config_sha256"] = hashlib.sha256(data["config_json"].encode("utf-8")).hexdigest()
    provenance_file.write_text(json.dumps(data))  # internally consistent, but from other inputs
    with pytest.raises(FileExistsError, match="produced from different inputs"):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1))

    provenance_file.write_text(original)
    assert _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1)).resumed is True


def test_retry_after_a_utc_date_rollover_resumes_the_finalized_artifact(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload finalized just before midnight is found and adopted by a retry on the next UTC day."""
    before_midnight = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.write_record", _fail)
    with pytest.raises(RuntimeError):
        _run_at(store, records_root, before_midnight)
    monkeypatch.undo()
    (orphan,) = _processed_entries(store)
    assert orphan.startswith("processed-20260830-")

    result = _run_at(store, records_root, before_midnight + timedelta(seconds=2))
    assert result.resumed is True
    assert result.record.artifact.artifact_id == orphan
    assert result.record.artifact.created_at == before_midnight.isoformat()
    assert result.provenance.created_at == before_midnight.isoformat()
    assert _processed_entries(store) == [orphan]  # no processed-20260831-... duplicate
    assert result.record_file.name == f"{orphan}.toml"
    assert [e.artifact_id for e in load_catalog(records_root / "data" / "catalog.toml").artifacts] == [orphan]


def test_metadata_recorded_at_finalization_wins_on_resume(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """License, access, and command come from the pending record kept with the payload, not from the retry."""
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.write_record", _fail)
    with pytest.raises(RuntimeError):
        _run_at(
            store, records_root, FIXED_TIME, license_override="CC-BY-4.0", access_override="internal", command="first"
        )
    monkeypatch.undo()
    (orphan,) = _processed_entries(store)
    pending = load_record(store.root / "processed" / orphan / "record.toml", ProcessedDatasetRecord)
    assert (pending.artifact.license, pending.artifact.access, pending.artifact.origin.command) == (
        "CC-BY-4.0",
        "internal",
        "first",
    )

    with pytest.raises(FileExistsError, match=r"recorded with license 'CC-BY-4\.0', and this retry asks for 'MIT'"):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1), license_override="MIT")
    with pytest.raises(FileExistsError, match="recorded with access 'internal', and this retry asks for 'public'"):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1), access_override="public")
    assert list((records_root / "data" / "records" / "processed").iterdir()) == []

    result = _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1), command="second")
    assert result.resumed is True
    assert result.record == pending
    assert load_record(result.record_file, ProcessedDatasetRecord) == pending
    assert result.record.artifact.origin.command == "first"


def test_resume_refuses_a_tampered_or_missing_pending_record(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pending record must load strictly and be exactly what this invocation rebuilds from the stored provenance."""
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.write_record", _fail)
    with pytest.raises(RuntimeError):
        _run_at(store, records_root, FIXED_TIME)
    monkeypatch.undo()
    (orphan,) = _processed_entries(store)
    pending_file = store.root / "processed" / orphan / "record.toml"
    original = pending_file.read_text()

    pending_file.write_text(original.replace("Processed from", "Edited after finalization:"))
    with pytest.raises(FileExistsError, match=r"record\.toml does not describe this payload and provenance"):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1))

    pending_file.write_text(original[: len(original) // 2])
    with pytest.raises(FileExistsError, match=r"record\.toml is missing or invalid"):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1))

    pending_file.unlink()
    with pytest.raises(FileExistsError, match=r"record\.toml is missing or invalid"):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1))

    pending_file.write_text(original)
    assert _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1)).resumed is True


def test_resume_refuses_several_finalized_copies(
    store: StorageRoot, records_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two finalized directories with the same payload digest are ambiguous and left for inspection."""
    monkeypatch.setattr("arm_rc_ctrl.data.preprocess.write_record", _fail)
    with pytest.raises(RuntimeError):
        _run_at(store, records_root, FIXED_TIME)
    monkeypatch.undo()
    (orphan,) = _processed_entries(store)
    shutil.copytree(
        store.root / "processed" / orphan, store.root / "processed" / orphan.replace("20260830", "20260829")
    )
    with pytest.raises(FileExistsError, match="several finalized payloads carry digest"):
        _run_at(store, records_root, FIXED_TIME + timedelta(minutes=1))
    assert len(_processed_entries(store)) == 2  # staging cleaned up, both copies left in place
