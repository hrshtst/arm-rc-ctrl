# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-003: the task 1-a recovery dataset reproduces from its raw payload through a configured store.

These tests need the machine-local storage root that holds the raw
demonstration payload; without it (e.g. in CI) they are skipped with a
reason, never silently passed.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arm_rc_ctrl.data.records import (
    RawDemonstrationRecord,
    load_catalog,
    load_record,
    verify_payload,
)
from arm_rc_ctrl.data.recover import recover_demonstration
from arm_rc_ctrl.data.recovery import TASK_PHASE_CODES, RecoveryDatasetRecord
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageError, StorageRoot, open_storage

pytestmark = [pytest.mark.regression, pytest.mark.integration]

REPO_ROOT = repository_root()
RAW_ID = "raw-20260830-b5adde395f1c"
RECOVERY_ID = "processed-20260903-ce343c8ce6a5"
RAW_RECORD = REPO_ROOT / "data" / "records" / "raw" / f"{RAW_ID}.toml"
RECOVERY_RECORD = REPO_ROOT / "data" / "records" / "processed" / f"{RECOVERY_ID}.toml"
SCENARIO = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
CONFIG = REPO_ROOT / "configs" / "preprocessing" / "recovery_v1.toml"
FIXED_TIME = datetime(2026, 9, 3, 8, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def configured_store() -> StorageRoot:
    """The machine's configured storage root, holding the raw payload; skip when unavailable."""
    try:
        store = open_storage()
        raw = load_record(RAW_RECORD, RawDemonstrationRecord)
        verify_payload(store, raw.artifact)
    except (StorageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        pytest.skip(f"configured external store with {RAW_ID} not available: {exc}")
    return store


@pytest.fixture(scope="module")
def committed_recovery() -> RecoveryDatasetRecord:
    """The Git-tracked recovery record derived from the scripted demonstration."""
    return load_record(RECOVERY_RECORD, RecoveryDatasetRecord)


def test_committed_records_are_consistent_with_each_other(committed_recovery: RecoveryDatasetRecord) -> None:
    """The recovery record binds the raw record, the programmed onset, and the move/dwell task clock."""
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.toml")
    assert catalog.find(RECOVERY_ID) is not None
    record = committed_recovery
    assert record.artifact.origin.sources == (RAW_ID,)
    assert record.scenario == raw.scenario
    record.check_scenario(SCENARIO)
    assert record.n_samples == 401
    assert record.phases == TASK_PHASE_CODES
    assert record.preprocessing.resample_period_s == 0.01
    assert record.onset.kind == "scripted"
    assert record.onset.detector == "programmed"
    assert record.onset.proposed_onset_sample == 100
    assert record.onset.confirmed_onset_sample == 100
    assert record.onset.raw_artifact_id == RAW_ID
    assert record.onset.raw_payload_sha256 == raw.artifact.payload.sha256
    assert record.crop.pre_roll == (0.0, record.onset.confirmed_onset_s)
    assert record.crop.source_duration_s == raw.duration_s
    assert record.crop.task.duration_s == pytest.approx(4.0)
    assert record.baseline.status == "passed"
    assert record.baseline.max_deviation_rad <= record.baseline.tolerance_rad
    assert record.normalization is not None
    assert record.normalization.fitted_on == (RECOVERY_ID,)
    assert record.artifact.origin.project_dirty is False


def test_committed_recovery_payload_matches_its_record(
    configured_store: StorageRoot, committed_recovery: RecoveryDatasetRecord
) -> None:
    """The stored samples.npz has the recorded digest, starts at task time zero, and begins at q0_ref."""
    payload = verify_payload(configured_store, committed_recovery.artifact)
    committed_recovery.check_samples(load_samples(payload))


def test_clean_reproduction_through_a_fresh_store_yields_identical_arrays(
    configured_store: StorageRoot, committed_recovery: RecoveryDatasetRecord, tmp_path: Path
) -> None:
    """Re-deriving from the raw payload in a fresh store reproduces the payload digest and specs."""
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    fresh_root = tmp_path / "store"
    fresh_root.mkdir()
    fresh = StorageRoot(fresh_root, repositories=(REPO_ROOT,))
    fresh.path(raw.artifact.payload.uri, mode="write").write_bytes(
        verify_payload(configured_store, raw.artifact).read_bytes()
    )
    records = tmp_path / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    result = recover_demonstration(
        RAW_RECORD, SCENARIO, CONFIG, store=fresh, records_root=records, exploratory=True, now=FIXED_TIME
    )
    reproduced = result.record
    assert reproduced.artifact.payload.sha256 == committed_recovery.artifact.payload.sha256
    assert reproduced.artifact.payload.size == committed_recovery.artifact.payload.size
    assert reproduced.arrays == committed_recovery.arrays
    # Only identity and provenance may differ between the reproduction and the committed record.
    for field in dataclasses.fields(RecoveryDatasetRecord):
        if field.name != "artifact":
            assert getattr(reproduced, field.name) == getattr(committed_recovery, field.name), field.name
