# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-003: raw skelarm logs load through the storage resolver; bad inputs fail without touching the source."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
from skelarm import StateLog

from arm_rc_ctrl.data.raw import RawLogError, load_raw_demonstration, read_log_schema_version
from arm_rc_ctrl.data.records import (
    Intervals,
    RawDemonstrationRecord,
    Sampling,
    Scenario,
    load_record,
    make_artifact_id,
)
from arm_rc_ctrl.provenance import ArtifactMismatchError, sha256_bytes, sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageAccessError, StorageRoot

REPO_ROOT = repository_root()
FIXTURE_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
FIXTURE_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
IS_ROOT_USER = os.geteuid() == 0


@pytest.fixture
def record() -> RawDemonstrationRecord:
    """The committed record of the committed fixture log."""
    return load_record(FIXTURE_RECORD, RawDemonstrationRecord)


@pytest.fixture
def store(tmp_path: Path) -> StorageRoot:
    """Empty storage root outside the repository."""
    root = tmp_path / "store"
    root.mkdir()
    return StorageRoot(root, repositories=(REPO_ROOT,))


def _place(store: StorageRoot, record: RawDemonstrationRecord, data: bytes | None = None) -> Path:
    path = store.path(record.artifact.payload.uri, mode="write")
    path.write_bytes(FIXTURE_LOG.read_bytes() if data is None else data)
    return path


def _snapshot(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), path.stat().st_mtime_ns


def _record_for(record: RawDemonstrationRecord, data: bytes, **changes: object) -> RawDemonstrationRecord:
    """A record whose payload identity matches ``data`` (so verification passes), with field overrides."""
    digest = sha256_bytes(data)
    artifact_id = make_artifact_id("raw", record.artifact.created_at, digest)
    payload = dataclasses.replace(
        record.artifact.payload, uri=f"armrc://raw/{artifact_id}/demo.sklog.npz", sha256=digest, size=len(data)
    )
    artifact = dataclasses.replace(record.artifact, artifact_id=artifact_id, payload=payload)
    return dataclasses.replace(record, artifact=artifact, **changes)


def _save_log(path: Path, times: list[float], **channels: np.ndarray) -> bytes:
    log = StateLog()
    for i, t in enumerate(times):
        log.record(t, **{name: values[i] for name, values in channels.items()})
    log.save(path)
    return path.read_bytes()


# --- known fixture ------------------------------------------------------------------


def test_committed_fixture_matches_its_record(record: RawDemonstrationRecord) -> None:
    """The committed log is the payload the committed record describes."""
    assert sha256_file(FIXTURE_LOG) == record.artifact.payload.sha256
    assert FIXTURE_LOG.stat().st_size == record.artifact.payload.size
    assert read_log_schema_version(FIXTURE_LOG) == record.artifact.payload.schema_version == 1


def test_known_fixture_loads(store: StorageRoot, record: RawDemonstrationRecord) -> None:
    """The fixture resolves through the store, verifies, parses, cross-checks, and is exposed read-only."""
    path = _place(store, record)
    before = _snapshot(path)
    demo = load_raw_demonstration(store, record)
    assert demo.record is record
    assert demo.path == store.path(record.artifact.payload.uri, mode="read")
    assert demo.n_samples == 31
    assert demo.dof == 2
    assert demo.q.shape == demo.dq.shape == (31, 2)
    assert set(demo.channels) == {"q", "dq", "tau", "q_ref", "error"}
    assert demo.times[0] == 0.0
    assert abs(demo.times[-1] - record.duration_s) < 1e-9
    assert demo.log.build_skeleton().num_joints == 2
    with pytest.raises(ValueError, match="read-only"):
        demo.q[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        demo.times[0] = 1.0
    assert _snapshot(path) == before


# --- missing, inaccessible, mismatched, corrupt -----------------------------------------


def test_missing_payload_fails(store: StorageRoot, record: RawDemonstrationRecord) -> None:
    """An absent payload is a storage access error."""
    with pytest.raises(StorageAccessError, match="does not exist"):
        load_raw_demonstration(store, record)


@pytest.mark.skipif(IS_ROOT_USER, reason="permission bits are not enforced for root")
def test_unreadable_payload_fails(store: StorageRoot, record: RawDemonstrationRecord) -> None:
    """An unreadable payload is a storage access error."""
    path = _place(store, record)
    path.chmod(0o000)
    try:
        with pytest.raises(StorageAccessError, match="not readable"):
            load_raw_demonstration(store, record)
    finally:
        path.chmod(0o600)


def test_mismatched_payload_fails_without_modification(store: StorageRoot, record: RawDemonstrationRecord) -> None:
    """Size and digest mismatches are rejected before parsing; the bytes are left as found."""
    original = FIXTURE_LOG.read_bytes()
    path = _place(store, record, original + b"\0")
    before = _snapshot(path)
    with pytest.raises(ArtifactMismatchError, match="size"):
        load_raw_demonstration(store, record)
    assert _snapshot(path) == before

    flipped = bytearray(original)
    flipped[-1] ^= 0xFF
    path.write_bytes(bytes(flipped))
    before = _snapshot(path)
    with pytest.raises(ArtifactMismatchError, match="sha256"):
        load_raw_demonstration(store, record)
    assert _snapshot(path) == before


def test_corrupt_payload_fails_without_modification(
    store: StorageRoot, record: RawDemonstrationRecord, tmp_path: Path
) -> None:
    """Bytes that verify but are not a skelarm log fail with RawLogError."""
    garbage = b"this is not a zip archive"
    corrupt = _record_for(record, garbage)
    path = _place(store, corrupt, garbage)
    before = _snapshot(path)
    with pytest.raises(RawLogError, match="not a readable skelarm log"):
        load_raw_demonstration(store, corrupt)
    assert _snapshot(path) == before

    scratch = tmp_path / "nometa.npz"
    np.savez(scratch, time=np.zeros(3), q=np.zeros((3, 2)))
    no_meta = scratch.read_bytes()
    without = _record_for(record, no_meta)
    _place(store, without, no_meta)
    with pytest.raises(RawLogError, match="has no __meta__ member"):
        load_raw_demonstration(store, without)

    scratch2 = tmp_path / "badmeta.npz"
    np.savez(scratch2, time=np.zeros(3), q=np.zeros((3, 2)), __meta__=np.array("schema_version = 'one'"))
    bad_meta = scratch2.read_bytes()
    with_bad = _record_for(record, bad_meta)
    _place(store, with_bad, bad_meta)
    with pytest.raises(RawLogError, match="schema_version is missing or not an integer"):
        load_raw_demonstration(store, with_bad)


# --- unexpected data (record and log disagree) ----------------------------------------------


def _scenario(record: RawDemonstrationRecord, **changes: object) -> Scenario:
    return dataclasses.replace(record.scenario, **changes)


def _sampling(record: RawDemonstrationRecord, **changes: object) -> Sampling:
    return dataclasses.replace(record.sampling, **changes)


def _variant(record: RawDemonstrationRecord, key: str) -> dict[str, object]:
    """Record edits that make it disagree with the fixture log in one specific way."""
    if key == "schema_version":
        payload = dataclasses.replace(record.artifact.payload, schema_version=2)
        return {"artifact": dataclasses.replace(record.artifact, payload=payload)}
    if key == "dof":
        return {"scenario": _scenario(record, dof=3, initial_q=(0.0, 0.0, 0.0))}
    if key == "missing_unit":
        return {"sampling": _sampling(record, units={k: v for k, v in record.sampling.units.items() if k != "tau"})}
    if key == "wrong_unit":
        return {"sampling": _sampling(record, units={**record.sampling.units, "q": "deg"})}
    if key == "period":
        return {"sampling": _sampling(record, period_s=0.02)}
    if key == "wall_period":
        return {"sampling": _sampling(record, clock="wall", period_s=0.02)}
    if key == "intervals":
        return {"intervals": Intervals((0.0, 0.1), (0.1, 0.25), (0.25, 0.5)), "duration_s": 0.5}
    msg = f"unknown variant {key!r}"
    raise ValueError(msg)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("schema_version", "log schema_version 1 != record payload.schema_version 2"),
        ("dof", r"channel 'q' has shape \(31, 2\); expected \(31, 3\)"),
        ("dof", "embedded skeleton has 2 joints; record scenario.dof is 3"),
        ("missing_unit", "record units declare channels"),
        ("wrong_unit", "channel 'q' unit 'rad' != record unit 'deg'"),
        ("period", "simulated clock: sample interval deviates from 0.02 s"),
        ("wall_period", "wall clock: median sample interval"),
        ("intervals", "record intervals end at 0.5 s but the recording ends at 0.3"),
    ],
)
def test_record_log_disagreements_are_rejected(
    store: StorageRoot, record: RawDemonstrationRecord, key: str, message: str
) -> None:
    """Schema version, joint count, channels/units, sampling period, and interval range are cross-checked."""
    modified = dataclasses.replace(record, **_variant(record, key))
    path = _place(store, modified)
    before = _snapshot(path)
    with pytest.raises(RawLogError, match=message):
        load_raw_demonstration(store, modified)
    assert _snapshot(path) == before


def test_wall_clock_tolerates_jitter_within_bounds(store: StorageRoot, record: RawDemonstrationRecord) -> None:
    """A wall-clock record accepts the fixture's uniform 10 ms spacing."""
    wall = dataclasses.replace(record, sampling=_sampling(record, clock="wall"))
    _place(store, wall)
    assert load_raw_demonstration(store, wall).n_samples == 31


def _minimal_record(record: RawDemonstrationRecord, data: bytes, **changes: object) -> RawDemonstrationRecord:
    """Record for a hand-built log with only q and dq channels (no channel metadata)."""
    sampling = Sampling(period_s=0.01, clock="simulated", units={"t": "s", "q": "rad", "dq": "rad/s"})
    intervals = Intervals((0.0, 0.01), (0.01, 0.02), (0.02, 0.03))
    return _record_for(record, data, sampling=sampling, intervals=intervals, duration_s=0.03, **changes)


def test_log_without_required_channel_fails(store: StorageRoot, record: RawDemonstrationRecord, tmp_path: Path) -> None:
    """A log lacking dq is rejected."""
    data = _save_log(tmp_path / "q_only.npz", [0.0, 0.01, 0.02, 0.03], q=np.zeros((4, 2)))
    minimal = _minimal_record(record, data)
    _place(store, minimal, data)
    with pytest.raises(RawLogError, match="log has no 'dq' channel"):
        load_raw_demonstration(store, minimal)


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ([0.0, 0.01, 0.01, 0.02], "time is not strictly increasing"),
        ([0.5, 0.51, 0.52, 0.53], "time must start at 0"),
        ([0.0], "log has 1 frames; at least 2 are required"),
    ],
)
def test_timing_problems_are_rejected(
    store: StorageRoot, record: RawDemonstrationRecord, tmp_path: Path, times: list[float], message: str
) -> None:
    """Non-increasing time, a non-zero start, and too few frames are all rejected."""
    n = len(times)
    data = _save_log(tmp_path / "timing.npz", times, q=np.zeros((n, 2)), dq=np.zeros((n, 2)))
    minimal = _minimal_record(record, data)
    _place(store, minimal, data)
    with pytest.raises(RawLogError, match=message):
        load_raw_demonstration(store, minimal)


def test_hand_built_minimal_log_loads(store: StorageRoot, record: RawDemonstrationRecord, tmp_path: Path) -> None:
    """A log with only the required channels and no skeleton still loads when consistent."""
    data = _save_log(tmp_path / "ok.npz", [0.0, 0.01, 0.02, 0.03], q=np.zeros((4, 2)), dq=np.zeros((4, 2)))
    minimal = _minimal_record(record, data)
    _place(store, minimal, data)
    demo = load_raw_demonstration(store, minimal)
    assert demo.n_samples == 4
    assert set(demo.channels) == {"q", "dq"}
