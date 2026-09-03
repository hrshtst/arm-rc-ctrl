# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-001: raw-demonstration artifact records round-trip through TOML and reject inconsistent data."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from arm_rc_ctrl.config import ConfigError, from_mapping, to_mapping
from arm_rc_ctrl.data.records import (
    ArtifactRecord,
    Catalog,
    CatalogEntry,
    DvcPointer,
    Intervals,
    Origin,
    Payload,
    RawDemonstrationRecord,
    Sampling,
    Scenario,
    catalog_path,
    is_artifact_id,
    load_catalog,
    load_record,
    make_artifact_id,
    payload_from_store,
    record_path,
    to_toml,
    verify_payload,
    write_catalog,
    write_record,
)
from arm_rc_ctrl.data.recovery import load_processed_record
from arm_rc_ctrl.experiments.run_record import RunPointerRecord
from arm_rc_ctrl.provenance import ArtifactMismatchError, collect_provenance, sha256_bytes
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageAccessError, StorageRoot

REPO_ROOT = repository_root()
CREATED = "2026-08-30T03:00:00+00:00"
PAYLOAD_SHA = sha256_bytes(b"demo")
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "records" / f"raw-20260830-{PAYLOAD_SHA[:12]}.toml"
COMMIT = "a" * 40
CONFIG_SHA = "1" * 64


def _payload(**changes: object) -> Payload:
    base = Payload(
        uri=f"armrc://raw/raw-20260830-{PAYLOAD_SHA[:12]}/demo.sklog.npz",
        sha256=PAYLOAD_SHA,
        size=4,
        format="sklog.npz",
        schema_version=1,
    )
    return dataclasses.replace(base, **changes)


def _origin(**changes: object) -> Origin:
    base = Origin(
        command="python -m arm_rc_ctrl.data.import_demo --config configs/tasks/task_1a.toml",
        config_sha256=CONFIG_SHA,
        project_commit=COMMIT,
        project_dirty=False,
        dependency_commits={"rclib": "b" * 40, "skelarm": "c" * 40, "rtctrl": "d" * 40},
    )
    return dataclasses.replace(base, **changes)


def _artifact(**changes: object) -> ArtifactRecord:
    base = ArtifactRecord(
        artifact_id=f"raw-20260830-{PAYLOAD_SHA[:12]}",
        kind="raw",
        created_at=CREATED,
        license="LicenseRef-Private",
        access="private",
        payload=_payload(),
        origin=_origin(),
        notes="Synthetic example record; the payload does not exist.",
    )
    return dataclasses.replace(base, **changes)


def _raw(**changes: object) -> RawDemonstrationRecord:
    base = RawDemonstrationRecord(
        artifact=_artifact(),
        scenario=Scenario(
            config_path="configs/tasks/task_1a.toml",
            config_sha256=CONFIG_SHA,
            robot="planar-2dof",
            task="task-1a-reach",
            dof=2,
            initial_q=(0.3, 0.6),
            target=(0.35, 0.25),
        ),
        sampling=Sampling(period_s=0.005, clock="simulated", units={"t": "s", "q": "rad", "dq": "rad/s", "tau": "N*m"}),
        session="teacher-01-session-03",
        intervals=Intervals(prime=(0.0, 1.0), move=(1.0, 3.5), dwell=(3.5, 5.0)),
        duration_s=5.0,
    )
    return dataclasses.replace(base, **changes)


# --- identity ---------------------------------------------------------------------


def test_artifact_id_is_content_addressed_and_deterministic() -> None:
    """The ID carries kind, creation date, and the payload digest prefix."""
    assert make_artifact_id("raw", CREATED, PAYLOAD_SHA) == f"raw-20260830-{PAYLOAD_SHA[:12]}"
    assert make_artifact_id("raw", CREATED, PAYLOAD_SHA) == make_artifact_id("raw", CREATED, PAYLOAD_SHA)
    assert make_artifact_id("model", "2026-12-31T23:59:59+00:00", "f" * 64) == "model-20261231-ffffffffffff"
    with pytest.raises(ValueError, match="sha256 must be 64"):
        make_artifact_id("raw", CREATED, "abc")
    with pytest.raises(ValueError, match="timezone-aware"):
        make_artifact_id("raw", "2026-08-30T03:00:00", PAYLOAD_SHA)


@pytest.mark.parametrize(
    "value",
    [
        "raw-20260830-8f434346648f",
        "processed-20260830-000000000000",
        "run-20260101-abcdefabcdef",
        "model-20260830-1234567890ab",
    ],
)
def test_valid_artifact_ids(value: str) -> None:
    """All four kinds follow the same grammar."""
    assert is_artifact_id(value)


@pytest.mark.parametrize(
    "value",
    ["raw-2026083-8f434346648f", "raw-20260830-8F434346648F", "raw-20260830-8f43", "dataset-20260830-8f434346648f", ""],
)
def test_invalid_artifact_ids(value: str) -> None:
    """Wrong date width, uppercase, short digest, or unknown kind are rejected."""
    assert not is_artifact_id(value)


# --- round trip ---------------------------------------------------------------------


def test_raw_record_round_trips_through_toml(tmp_path: Path) -> None:
    """Dump → load reproduces an equal record; optional fields are omitted from the text when unset."""
    record = _raw()
    text = to_toml(record)
    for absent in ("expires_at", "supersedes", "dvc", "run_id"):
        assert absent not in text
    path = tmp_path / "record.toml"
    write_record(path, record)
    assert load_record(path, RawDemonstrationRecord) == record


def test_optional_fields_round_trip_when_set(tmp_path: Path) -> None:
    """expires_at, supersedes, DVC pointer, sources, and run_id survive the round trip."""
    artifact = _artifact(
        expires_at="2027-08-30T03:00:00+00:00",
        supersedes="raw-20260829-000000000000",
        dvc=DvcPointer(target="data/records/raw/demo.sklog.npz.dvc", md5="0" * 32),
        origin=_origin(sources=("raw-20260829-000000000000",), run_id="import-0001"),
    )
    record = _raw(artifact=artifact)
    path = tmp_path / "record.toml"
    write_record(path, record)
    loaded = load_record(path, RawDemonstrationRecord)
    assert loaded == record
    assert loaded.artifact.dvc == DvcPointer("data/records/raw/demo.sklog.npz.dvc", "0" * 32)
    assert loaded.artifact.origin.run_id == "import-0001"


def test_committed_example_record_is_canonical() -> None:
    """The fixture loads, equals the builder, and re-serializes byte-for-byte (stable formatting)."""
    loaded = load_record(FIXTURE, RawDemonstrationRecord)
    assert loaded == _raw()
    assert to_toml(loaded) == FIXTURE.read_text(encoding="utf-8")


def test_records_never_contain_machine_paths() -> None:
    """Only logical URIs and repository-relative paths appear in a record."""
    text = to_toml(_raw())
    assert "armrc://raw/" in text
    assert "/home/" not in text
    assert str(REPO_ROOT) not in text


def test_plain_mapping_round_trip_is_strict() -> None:
    """Unknown keys and mistyped values are rejected when a record is rebuilt from plain data."""
    mapping = json.loads(json.dumps(to_mapping(_raw())))
    assert from_mapping(mapping, RawDemonstrationRecord) == _raw()
    with pytest.raises(ConfigError, match=r"unknown key\(s\) 'teacher'"):
        from_mapping({**mapping, "teacher": "x"}, RawDemonstrationRecord)
    bad = json.loads(json.dumps(mapping))
    bad["artifact"]["payload"]["size"] = "4"
    with pytest.raises(ConfigError, match=r"artifact\.payload\.size: expected integer, got string"):
        from_mapping(bad, RawDemonstrationRecord)
    bad = json.loads(json.dumps(mapping))
    bad["artifact"]["access"] = "secret"
    with pytest.raises(ConfigError, match=r"artifact\.access: expected one of 'private', 'internal', 'public'"):
        from_mapping(bad, RawDemonstrationRecord)


# --- validation -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: _artifact(schema_version=2), "unsupported record schema_version 2"),
        (lambda: _artifact(artifact_id="raw-20260830-8F434346648F"), "must match <kind>-<YYYYMMDD>-<12 hex>"),
        (lambda: _artifact(artifact_id=f"model-20260830-{PAYLOAD_SHA[:12]}"), "does not carry kind 'raw'"),
        (lambda: _artifact(artifact_id=f"raw-20260829-{PAYLOAD_SHA[:12]}"), "date 20260829 does not match created_at"),
        (lambda: _artifact(artifact_id="raw-20260830-000000000000"), "digest prefix does not match payload.sha256"),
        (lambda: _artifact(created_at="2026-08-30T03:00:00"), "created_at must be timezone-aware in UTC"),
        (lambda: _artifact(expires_at="2026-08-30T02:00:00+00:00"), "expires_at must be later than created_at"),
        (lambda: _artifact(license="  "), "license must not be empty"),
        (
            lambda: _artifact(payload=_payload(uri=f"armrc://processed/raw-20260830-{PAYLOAD_SHA[:12]}/x.npz")),
            "bucket 'processed' does not match kind 'raw'",
        ),
        (
            lambda: _artifact(payload=_payload(uri="armrc://raw/other/demo.sklog.npz")),
            "must live under armrc://raw/raw-20260830-",
        ),
        (lambda: _artifact(supersedes=f"raw-20260830-{PAYLOAD_SHA[:12]}"), "cannot supersede itself"),
        (lambda: _artifact(supersedes="model-20260830-000000000000"), "must be an artifact ID of kind 'raw'"),
    ],
)
def test_artifact_record_invariants(build: object, message: str) -> None:
    """Identity, timestamps, license, payload placement, and supersession are validated."""
    with pytest.raises(ValueError, match=message):
        cast("object", build)()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: _payload(sha256="xyz"), "payload.sha256 must be 64"),
        (lambda: _payload(size=-1), "payload.size must be non-negative"),
        (lambda: _payload(format="SKLOG"), "payload.format must be a lowercase label"),
        (lambda: _payload(schema_version=0), "payload.schema_version must be >= 1"),
        (lambda: _payload(uri="file:///tmp/x"), "expected a armrc:// URI"),
        (lambda: _origin(command=" "), "origin.command must not be empty"),
        (lambda: _origin(config_sha256="nope"), "origin.config_sha256 must be 64"),
        (lambda: _origin(project_commit="HEAD"), "origin.project_commit must be a 40-hex commit"),
        (lambda: _origin(dependency_commits={"rclib": "short"}), r"origin.dependency_commits\['rclib'\]"),
        (lambda: _origin(sources=("not-an-id",)), "is not an artifact ID"),
        (lambda: _origin(sources=("raw-20260829-000000000000",) * 2), "origin.sources must be unique"),
        (lambda: _origin(run_id="../escape"), "invalid path segment"),
        (lambda: DvcPointer(target="/abs/x.dvc", md5="0" * 32), "dvc.target must be a repository-relative"),
        (lambda: DvcPointer(target="data/x.dvc", md5="0" * 31), "dvc.md5 must be 32"),
    ],
)
def test_payload_origin_and_dvc_invariants(build: object, message: str) -> None:
    """Sub-records validate their own fields."""
    with pytest.raises(ValueError, match=message):
        cast("object", build)()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: _raw(
                artifact=_artifact(
                    kind="model",
                    artifact_id=f"model-20260830-{PAYLOAD_SHA[:12]}",
                    payload=_payload(uri=f"armrc://models/model-20260830-{PAYLOAD_SHA[:12]}/demo.sklog.npz"),
                )
            ),
            "must have kind 'raw'",
        ),
        (
            lambda: _raw(artifact=_artifact(payload=_payload(format="samples.npz"))),
            "raw payload format must be 'sklog.npz'",
        ),
        (
            lambda: _raw(
                artifact=_artifact(payload=_payload(uri=f"armrc://raw/raw-20260830-{PAYLOAD_SHA[:12]}/teach.sklog.npz"))
            ),
            "must be retained at armrc://raw/",
        ),
        (lambda: _raw(session="Teacher 1"), r"session must match \[a-z0-9\]"),
        (lambda: _raw(duration_s=4.9), "duration_s 4.9 must equal the dwell end 5.0"),
        (
            lambda: Scenario("/abs/task.toml", CONFIG_SHA, "r", "t", 2, (0.0, 0.0), (0.1,)),
            "scenario.config_path must be a repository-relative",
        ),
        (lambda: Scenario("configs/../task.toml", CONFIG_SHA, "r", "t", 2, (0.0, 0.0), (0.1,)), "without '.' or '..'"),
        (
            lambda: Scenario("configs/task.yaml", CONFIG_SHA, "r", "t", 2, (0.0, 0.0), (0.1,)),
            "must point at a TOML file",
        ),
        (
            lambda: Scenario("configs/task.toml", CONFIG_SHA, "r", "t", 2, (0.0,), (0.1,)),
            "initial_q must have dof=2 entries",
        ),
        (
            lambda: Scenario("configs/task.toml", CONFIG_SHA, "r", "t", 2, (0.0, float("nan")), (0.1,)),
            r"initial_q\[1\] must be finite",
        ),
        (lambda: Scenario("configs/task.toml", CONFIG_SHA, "r", "t", 0, (), (0.1,)), "scenario.dof must be >= 1"),
        (
            lambda: Sampling(0.0, "simulated", {"t": "s", "q": "rad", "dq": "rad/s"}),
            "sampling.period_s must be positive",
        ),
        (lambda: Sampling(0.01, "wall", {"t": "s", "dq": "rad/s"}), r"sampling.units is missing \['q'\]"),
        (lambda: Intervals((0.0, 1.0), (1.0, 3.0), (3.0, 2.0)), "intervals.dwell must satisfy start < end"),
        (lambda: Intervals((0.5, 1.0), (1.0, 3.0), (3.0, 5.0)), "intervals.prime must start at 0.0"),
        (lambda: Intervals((0.0, 1.0), (1.5, 3.0), (3.0, 5.0)), "intervals must be contiguous"),
        (lambda: Intervals((0.0, 1.0, 2.0), (1.0, 3.0), (3.0, 5.0)), r"intervals.prime must be a \[start, end\] pair"),
    ],
)
def test_raw_demonstration_invariants(build: object, message: str) -> None:
    """Scenario, sampling, intervals, session, duration, and payload placement are validated."""
    with pytest.raises(ValueError, match=message):
        cast("object", build)()  # type: ignore[operator]


def test_loading_reports_validation_errors_with_location(tmp_path: Path) -> None:
    """Semantic failures in a record file are located ConfigErrors."""
    text = to_toml(_raw()).replace("duration_s = 5.0", "duration_s = 4.0")
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=r"bad\.toml: <root>: duration_s 4.0 must equal the dwell end 5.0"):
        load_record(path, RawDemonstrationRecord)


# --- files -----------------------------------------------------------------------------


def test_record_path_layout() -> None:
    """Records live under data/records/<kind dir>/<id>.toml."""
    record = _artifact()
    assert record_path(REPO_ROOT, record) == REPO_ROOT / "data" / "records" / "raw" / f"{record.artifact_id}.toml"
    run = _artifact(
        kind="run",
        artifact_id=f"run-20260830-{PAYLOAD_SHA[:12]}",
        payload=_payload(uri=f"armrc://runs/run-20260830-{PAYLOAD_SHA[:12]}/summary.json", format="json"),
    )
    assert record_path(REPO_ROOT, run).parent.name == "runs"
    assert catalog_path(REPO_ROOT) == REPO_ROOT / "data" / "catalog.toml"


def test_write_record_is_immutable_and_atomic(tmp_path: Path) -> None:
    """An existing record is never overwritten; no temporary file is left behind."""
    path = tmp_path / "r.toml"
    write_record(path, _raw())
    with pytest.raises(FileExistsError, match="records are immutable"):
        write_record(path, _raw())
    assert not path.with_name("r.toml.tmp").exists()
    with pytest.raises(FileNotFoundError, match="record directory"):
        write_record(tmp_path / "missing" / "r.toml", _raw())


def test_origin_from_provenance_summarizes_a_real_record() -> None:
    """The origin carries the resolved-config digest, commit, dirty flag, and submodule commits."""
    provenance = collect_provenance({"a": 1}, seeds={}, now=datetime(2026, 8, 30, 3, 0, 0, tzinfo=UTC), env={})
    origin = Origin.from_provenance(provenance, command="python -m x", sources=("raw-20260829-000000000000",))
    assert origin.config_sha256 == provenance.config_sha256
    assert origin.project_commit == provenance.project_commit
    assert origin.project_dirty == provenance.project_dirty
    assert set(origin.dependency_commits) == {"rclib", "skelarm", "rtctrl"}
    assert origin.sources == ("raw-20260829-000000000000",)


def test_payload_from_store_and_verification(tmp_path: Path) -> None:
    """Payload identity is computed from the stored bytes and verified against the record."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    artifact_id = make_artifact_id("raw", CREATED, PAYLOAD_SHA)
    uri = f"armrc://raw/{artifact_id}/demo.sklog.npz"
    store.path(uri, mode="write").write_bytes(b"demo")
    payload = payload_from_store(store, uri, format="sklog.npz", schema_version=1)
    assert payload == _payload()
    record = _artifact(payload=payload)
    assert verify_payload(store, record) == store.path(uri, mode="read")
    store.path(uri, mode="read").write_bytes(b"demo!")
    with pytest.raises(ArtifactMismatchError, match="size 5 != recorded 4"):
        verify_payload(store, record)
    store.path(uri, mode="read").unlink()
    with pytest.raises(StorageAccessError, match="does not exist"):
        verify_payload(store, record)


# --- catalog --------------------------------------------------------------------------


def test_catalog_is_append_only(tmp_path: Path) -> None:
    """Entries are appended, never duplicated, changed, or removed."""
    path = tmp_path / "catalog.toml"
    assert load_catalog(path) == Catalog()
    first = _artifact()
    catalog = Catalog().with_record(first, "data/records/raw/first.toml")
    write_catalog(path, catalog)
    assert load_catalog(path) == catalog
    assert catalog.find(first.artifact_id) is not None
    assert catalog.find("raw-20260101-000000000000") is None
    with pytest.raises(ValueError, match="already contains"):
        catalog.with_record(first, "data/records/raw/again.toml")

    second = _artifact(
        artifact_id="raw-20260830-000000000000",
        payload=_payload(uri="armrc://raw/raw-20260830-000000000000/demo.sklog.npz", sha256="0" * 64),
    )
    grown = catalog.with_record(second, "data/records/raw/second.toml")
    write_catalog(path, grown)
    assert [e.artifact_id for e in load_catalog(path).artifacts] == [first.artifact_id, second.artifact_id]
    with pytest.raises(ValueError, match="would be removed"):
        write_catalog(path, catalog)
    changed = Catalog(artifacts=(dataclasses.replace(grown.artifacts[0], record="elsewhere.toml"), grown.artifacts[1]))
    with pytest.raises(ValueError, match="would change"):
        write_catalog(path, changed)


def test_catalog_rejects_duplicates_and_bad_entries() -> None:
    """Duplicate IDs/URIs and malformed entries never form a catalog."""
    entry = CatalogEntry(
        _artifact().artifact_id, "raw", "data/records/raw/x.toml", _payload().uri, PAYLOAD_SHA, CREATED
    )
    with pytest.raises(ValueError, match="artifact IDs must be unique"):
        Catalog(artifacts=(entry, entry))
    other = dataclasses.replace(entry, artifact_id="raw-20260830-000000000000")
    with pytest.raises(ValueError, match="payload URIs must be unique"):
        Catalog(artifacts=(entry, other))
    with pytest.raises(ValueError, match="is not an artifact ID of kind 'model'"):
        dataclasses.replace(entry, kind="model")
    with pytest.raises(ValueError, match="unsupported catalog schema_version"):
        Catalog(schema_version=2)
    with pytest.raises(ValueError, match="catalog record path must be a repository-relative"):
        dataclasses.replace(entry, record="/abs/x.toml")


def test_committed_catalog_is_valid_and_consistent_with_record_files() -> None:
    """data/catalog.toml loads and every listed record file exists with a matching ID."""
    catalog = load_catalog(catalog_path(REPO_ROOT))
    schemas: dict[str, type[object]] = {
        "raw": RawDemonstrationRecord,
        "run": RunPointerRecord,
        "model": ArtifactRecord,
    }
    for entry in catalog.artifacts:
        record_file = REPO_ROOT / entry.record
        assert record_file.is_file(), entry
        loaded = (
            load_processed_record(record_file)
            if entry.kind == "processed"
            else load_record(record_file, schemas[entry.kind])
        )
        artifact = loaded if isinstance(loaded, ArtifactRecord) else cast("Any", loaded).artifact
        assert artifact.artifact_id == entry.artifact_id
        assert artifact.payload.uri == entry.uri
        assert artifact.payload.sha256 == entry.sha256
