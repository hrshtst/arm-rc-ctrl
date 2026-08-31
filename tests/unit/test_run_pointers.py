# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2 review round 2 finding 1: curated runs are tracked by Git pointer records and catalog entries."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.data.records import load_catalog, load_record
from arm_rc_ctrl.experiments.run_record import (
    REQUIRED_ARRAYS,
    RunArrays,
    RunPointerRecord,
    record_run_pointer,
    write_run,
)
from arm_rc_ctrl.experiments.termination import Outcome, completed
from arm_rc_ctrl.provenance import collect_provenance
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

N, DOF = 12, 2


def _pointer(tmp_path: Path, name: str) -> RunPointerRecord:
    (tmp_path / name).mkdir()
    store = StorageRoot(tmp_path / name, repositories=(repository_root(),))
    t = np.arange(N, dtype=np.float64) * 0.01
    joint = {k: np.zeros((N, DOF)) for k in REQUIRED_ARRAYS if k not in ("t", "tip", "task_code", "saturation")}
    arrays = RunArrays(
        {
            "t": t,
            **joint,
            "tip": np.zeros((N, 2)),
            "task_code": np.zeros((N, 0)),
            "saturation": np.zeros(N, dtype=np.int64),
        }
    )
    termination = completed(0.11, N - 1)
    pointer, _summary, _directory = write_run(
        store,
        arrays,
        kind="simulation",
        method="replay+pd",
        scenario="unit",
        control_period_s=0.01,
        duration_s=0.11,
        target=(0.1, 0.2),
        task_code=(),
        disturbances=(),
        termination=termination,
        outcome=Outcome(termination, {"completed": True}),
        provenance=collect_provenance(
            {"unit": name}, seeds={}, artifacts=[], exploratory=True, now=datetime(2026, 9, 2, tzinfo=UTC)
        ),
        license_label="LicenseRef-Private",
        access="private",
        command="unit",
    )
    return pointer


def test_pointer_is_written_and_catalogued_once(tmp_path: Path) -> None:
    """The pointer record lands under data/records/runs with a catalog entry; recording it again is a no-op."""
    records = tmp_path / "repo"
    records.mkdir()
    pointer = _pointer(tmp_path, "store")
    path = record_run_pointer(records, pointer)
    assert path == records / "data" / "records" / "runs" / f"{pointer.artifact.artifact_id}.toml"
    assert load_record(path, RunPointerRecord) == pointer
    catalog = load_catalog(records / "data" / "catalog.toml")
    entry = catalog.find(pointer.artifact.artifact_id)
    assert entry is not None
    assert (entry.kind, entry.record) == ("run", f"data/records/runs/{pointer.artifact.artifact_id}.toml")
    before = (path.read_bytes(), (records / "data" / "catalog.toml").read_bytes())
    assert record_run_pointer(records, pointer) == path
    assert (path.read_bytes(), (records / "data" / "catalog.toml").read_bytes()) == before


def test_conflicting_pointers_are_refused(tmp_path: Path) -> None:
    """A different record under the same run ID, or a disagreeing catalog entry, is an error."""
    records = tmp_path / "repo"
    records.mkdir()
    pointer = _pointer(tmp_path, "store")
    record_run_pointer(records, pointer)
    other = dataclasses.replace(pointer, success=False)
    with pytest.raises(FileExistsError, match="run records are immutable"):
        record_run_pointer(records, other)
    catalog_file = records / "data" / "catalog.toml"
    catalog_file.write_text(catalog_file.read_text().replace(pointer.artifact.payload.sha256, "0" * 64))
    with pytest.raises(ValueError, match="disagrees with the run record"):
        record_run_pointer(records, pointer)
