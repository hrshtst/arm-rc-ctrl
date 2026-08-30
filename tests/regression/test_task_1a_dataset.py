# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-014: the task 1-a dataset reproduces from its raw payload through a configured external store.

These tests need the machine-local storage root that holds the raw
demonstration payload; without it (e.g. in CI) they are skipped with a
reason, never silently passed.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import (
    ProcessedDatasetRecord,
    RawDemonstrationRecord,
    load_catalog,
    load_record,
    verify_payload,
)
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageError, StorageRoot, open_storage

pytestmark = [pytest.mark.regression, pytest.mark.integration]

REPO_ROOT = repository_root()
RAW_ID = "raw-20260830-b5adde395f1c"
RAW_RECORD = REPO_ROOT / "data" / "records" / "raw" / f"{RAW_ID}.toml"
SCENARIO = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 8, 30, 15, 0, 0, tzinfo=UTC)


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
def committed_processed() -> ProcessedDatasetRecord:
    """The Git-tracked processed record derived from the raw demonstration."""
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.toml")
    candidates = [
        load_record(REPO_ROOT / entry.record, ProcessedDatasetRecord)
        for entry in catalog.artifacts
        if entry.kind == "processed"
    ]
    matching = [r for r in candidates if r.artifact.origin.sources == (RAW_ID,)]
    assert len(matching) == 1, "exactly one committed processed record must derive from the demonstration"
    return matching[0]


def test_committed_records_are_consistent_with_each_other() -> None:
    """The processed record references the raw record's scenario and its payload digests are declared."""
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.toml")
    assert catalog.find(RAW_ID) is not None
    processed = [
        load_record(REPO_ROOT / e.record, ProcessedDatasetRecord) for e in catalog.artifacts if e.kind == "processed"
    ]
    (record,) = [r for r in processed if r.artifact.origin.sources == (RAW_ID,)]
    assert record.scenario == raw.scenario
    record.check_scenario(SCENARIO)
    assert record.n_samples == 501
    assert record.preprocessing.resample_period_s == 0.01
    assert record.normalization is not None
    assert record.artifact.origin.project_dirty is False
    assert raw.artifact.origin.project_dirty is False


def test_committed_processed_payload_matches_its_record(
    configured_store: StorageRoot, committed_processed: ProcessedDatasetRecord
) -> None:
    """The stored samples.npz has the recorded digest and every array matches its spec."""
    payload = verify_payload(configured_store, committed_processed.artifact)
    committed_processed.check_samples(load_samples(payload))


def test_clean_reproduction_through_a_fresh_store_yields_identical_arrays(
    configured_store: StorageRoot, committed_processed: ProcessedDatasetRecord, tmp_path: Path
) -> None:
    """Re-running preprocessing from the raw payload in a fresh store gives the same payload digest and specs."""
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    fresh_root = tmp_path / "store"
    fresh_root.mkdir()
    fresh = StorageRoot(fresh_root, repositories=(REPO_ROOT,))
    fresh.path(raw.artifact.payload.uri, mode="write").write_bytes(
        verify_payload(configured_store, raw.artifact).read_bytes()
    )
    records = tmp_path / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    result = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=fresh, records_root=records, exploratory=True, now=FIXED_TIME
    )
    reproduced = result.record
    assert reproduced.artifact.payload.sha256 == committed_processed.artifact.payload.sha256
    assert reproduced.artifact.payload.size == committed_processed.artifact.payload.size
    assert reproduced.arrays == committed_processed.arrays
    assert reproduced.normalization is not None
    assert committed_processed.normalization is not None
    assert reproduced.normalization.channels == committed_processed.normalization.channels
    assert reproduced.preprocessing == committed_processed.preprocessing
    assert reproduced.scenario == committed_processed.scenario
    assert reproduced.units == committed_processed.units
    # Only identity and provenance may differ between the reproduction and the committed record.
    stripped = {"artifact", "normalization"}
    for field in dataclasses.fields(ProcessedDatasetRecord):
        if field.name not in stripped:
            assert getattr(reproduced, field.name) == getattr(committed_processed, field.name), field.name
