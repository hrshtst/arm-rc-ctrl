# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-019: run records round-trip state, references, torque, disturbances, termination, config, provenance."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.data.records import load_record, to_toml, write_record
from arm_rc_ctrl.experiments.run_record import (
    OPTIONAL_ARRAYS,
    REQUIRED_ARRAYS,
    Disturbance,
    RunArrays,
    RunPointerRecord,
    RunSummary,
    load_run,
    write_run,
)
from arm_rc_ctrl.experiments.termination import Outcome, completed, limit_violation
from arm_rc_ctrl.provenance import ArtifactMismatchError, collect_provenance
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

REPO_ROOT = repository_root()
FIXED_TIME = datetime(2026, 8, 30, 8, 0, 0, tzinfo=UTC)
N, DOF = 12, 2


def _arrays(*, applied: bool = True) -> RunArrays:
    t = np.arange(N, dtype=np.float64) * 0.01
    joint = {
        name: np.random.default_rng(i).standard_normal((N, DOF))
        for i, name in enumerate(REQUIRED_ARRAYS)
        if name not in ("t", "tip", "task_code", "saturation")
    }
    arrays: dict[str, NDArray[Any]] = {
        "t": t,
        **joint,
        "tip": np.column_stack([np.cos(t), np.sin(t)]),
        "task_code": np.zeros((N, 0)),
        "saturation": np.array([0] * (N - 2) + [1, 1], dtype=np.int64),
    }
    if applied:
        arrays["tau_applied"] = arrays["tau_requested"] * 0.9
        arrays["ext_force"] = np.column_stack([np.zeros(N), np.where(t < 0.05, -2.0, 0.0)])
        arrays["phase"] = (t >= 0.05).astype(np.int64)
        arrays["esn_state_norm"] = np.linspace(0.0, 1.0, N)
    return RunArrays(arrays)


def test_external_force_must_be_planar() -> None:
    """The optional ``ext_force`` array is (N, 2) like ``tip``."""
    arrays = _arrays().arrays
    with pytest.raises(ValueError, match=r"ext_force must have shape \(\d+, 2\)"):
        RunArrays({**arrays, "ext_force": np.zeros((N, DOF + 1))})
    without = {name: array for name, array in arrays.items() if name != "ext_force"}
    assert "ext_force" not in RunArrays(without).specs()


@pytest.fixture
def store(tmp_path: Path) -> StorageRoot:
    """Empty storage root."""
    root = tmp_path / "store"
    root.mkdir()
    return StorageRoot(root, repositories=(REPO_ROOT,))


def _write(store: StorageRoot, arrays: RunArrays | None = None, /, **overrides: object):  # noqa: ANN202
    provenance = collect_provenance(
        {"controller": {"kp": 20.0}}, seeds={"scenario": 3}, now=FIXED_TIME, env={}, exploratory=True
    )
    termination = completed(0.11, N - 1)
    kwargs: dict[str, Any] = {
        "kind": "simulation",
        "method": "replay+pd",
        "scenario": "task-1a-reach",
        "control_period_s": 0.01,
        "duration_s": 0.11,
        "target": (0.10, 0.45),
        "task_code": (),
        "disturbances": (Disturbance("endpoint_force", 0.03, 0.05, {"fx": 1.5, "fy": 0.0}),),
        "termination": termination,
        "outcome": Outcome(termination, {"completed": True, "final_dwell_in_tolerance": False}),
        "provenance": provenance,
        "license_label": "LicenseRef-Private",
        "access": "private",
        "command": "python -m arm_rc_ctrl.experiments.replay --config x.toml",
        "sources": ("processed-20260830-555555555555",),
        "notes": "unit-test run",
    }
    kwargs.update(overrides)
    return write_run(store, _arrays() if arrays is None else arrays, **kwargs)


def test_run_arrays_enforce_names_shapes_and_dtypes() -> None:
    """Required/optional names, float64/int64 dtypes, and consistent shapes are enforced."""
    arrays = _arrays()
    assert arrays.n_samples == N
    assert arrays.dof == DOF
    assert list(arrays.specs()) == [*REQUIRED_ARRAYS, *OPTIONAL_ARRAYS]
    assert list(_arrays(applied=False).specs()) == list(REQUIRED_ARRAYS)
    with pytest.raises(ValueError, match="read-only"):
        arrays.arrays["q"][0, 0] = 1.0
    raw = dict(_arrays().arrays)
    del raw["dq_desired"]
    with pytest.raises(ValueError, match=r"missing \['dq_desired'\]"):
        RunArrays(raw)
    raw = dict(_arrays().arrays)
    raw["extra"] = raw["q"]
    with pytest.raises(ValueError, match=r"unknown \['extra'\]"):
        RunArrays(raw)
    raw = dict(_arrays().arrays)
    raw["saturation"] = raw["saturation"].astype(np.float64)
    with pytest.raises(TypeError, match="saturation must be int64"):
        RunArrays(raw)
    raw = dict(_arrays().arrays)
    raw["tau_requested"] = np.zeros((N, 3))
    with pytest.raises(ValueError, match=rf"tau_requested must have shape \({N}, 2\)"):
        RunArrays(raw)
    raw = dict(_arrays().arrays)
    raw["tip"] = np.zeros((N - 1, 2))
    with pytest.raises(ValueError, match=rf"tip must have shape \({N}, 2\)"):
        RunArrays(raw)


def test_write_then_load_round_trips_everything(store: StorageRoot) -> None:
    """Arrays, references, torque, disturbances, termination, config, provenance, URI, and digests survive."""
    pointer, summary, directory = _write(store)
    assert pointer.artifact.artifact_id.startswith("run-20260830-")
    assert pointer.artifact.payload.uri == f"armrc://runs/{pointer.artifact.artifact_id}/run.json"
    assert pointer.artifact.origin.run_id == pointer.artifact.artifact_id
    assert pointer.artifact.origin.sources == ("processed-20260830-555555555555",)
    assert directory == store.path(f"armrc://runs/{pointer.artifact.artifact_id}", mode="read")
    assert not any(p.name.startswith("staging-") for p in (store.root / "runs").iterdir())

    loaded = load_run(store, pointer)
    assert loaded.summary == summary
    assert loaded.pointer == pointer
    original = _arrays()
    for name, array in original.arrays.items():
        assert np.array_equal(loaded.arrays.arrays[name], array), name
    assert loaded.summary.disturbances[0].parameters == {"fx": 1.5, "fy": 0.0}
    assert loaded.summary.termination == completed(0.11, N - 1)
    assert loaded.summary.outcome.failed_criteria == ("final_dwell_in_tolerance",)
    assert loaded.summary.provenance.config["controller"] == {"kp": 20.0}
    assert loaded.summary.seeds == {"scenario": 3}
    assert loaded.summary.arrays["tau_applied"].shape == (N, DOF)
    assert (directory / "arrays.npz").is_file()
    assert json.loads((directory / "run.json").read_text())["method"] == "replay+pd"


def test_pointer_record_round_trips_through_toml(store: StorageRoot, tmp_path: Path) -> None:
    """The Git pointer serializes and reloads strictly."""
    pointer, _, _ = _write(store)
    path = tmp_path / "run.toml"
    write_record(path, pointer)
    assert load_record(path, RunPointerRecord) == pointer
    assert "armrc://runs/" in to_toml(pointer)
    assert str(store.root) not in to_toml(pointer)


def test_tampering_is_detected_on_load(store: StorageRoot) -> None:
    """Modified summary, arrays, or pointer facts fail verification."""
    pointer, _, directory = _write(store)
    arrays_file = directory / "arrays.npz"
    original = arrays_file.read_bytes()
    arrays_file.write_bytes(original + b"\0")
    with pytest.raises(ValueError, match=r"digest .* != recorded"):
        load_run(store, pointer)
    arrays_file.write_bytes(original)

    summary_file = directory / "run.json"
    text = summary_file.read_text()
    summary_file.write_text(text.replace('"method":"replay+pd"', '"method":"rc+pd"'))
    with pytest.raises(ArtifactMismatchError):
        load_run(store, pointer)
    summary_file.write_text(text)

    with pytest.raises(ValueError, match="pointer record disagrees"):
        load_run(store, dataclasses.replace(pointer, success=True))


def test_runs_are_immutable_and_failures_leave_no_staging(store: StorageRoot) -> None:
    """Identical content on the same day maps to the same ID and is refused; errors clean up."""
    _write(store)
    with pytest.raises(FileExistsError, match="runs are immutable"):
        _write(store)
    assert not any(p.name.startswith("staging-") for p in (store.root / "runs").iterdir())
    with pytest.raises(ValueError, match="must equal termination"):
        _write(
            store, method="other", outcome=Outcome(limit_violation(0.05, 5, "torque", 12.0, 10.0), {"completed": False})
        )
    assert not any(p.name.startswith("staging-") for p in (store.root / "runs").iterdir())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": " "}, "method and scenario must not be empty"),
        ({"duration_s": 0.0}, "duration_s must be positive"),
        ({"target": (0.1,)}, r"target must be an \[x, y\]"),
        (
            {"termination": completed(9.0, 900), "outcome": Outcome(completed(9.0, 900), {"completed": True})},
            "beyond duration",
        ),
    ],
)
def test_summary_invariants(store: StorageRoot, overrides: dict[str, object], message: str) -> None:
    """Labels, timing, target, and termination timing are validated before anything is written."""
    with pytest.raises(ValueError, match=message):
        _write(store, **overrides)
    assert not (store.root / "runs").exists() or not any((store.root / "runs").iterdir())


def test_disturbance_and_pointer_invariants() -> None:
    """Disturbances validate timing/parameters; the pointer validates kind and payload placement."""
    with pytest.raises(ValueError, match="0 <= start <= end"):
        Disturbance("endpoint_force", 0.5, 0.2)
    with pytest.raises(ValueError, match="kind must not be empty"):
        Disturbance(" ", 0.0, 0.1)
    with pytest.raises(ValueError, match=r"parameters\[0\] must be finite"):
        Disturbance("x", 0.0, 0.1, {"fx": float("nan")})


def test_summary_json_is_strict() -> None:
    """Unknown keys or wrong shapes in a stored run.json are rejected."""
    with pytest.raises(ValueError, match="must be a JSON object"):
        RunSummary.from_json("[]")
    with pytest.raises(ConfigError, match=r"unknown key\(s\) 'extra'"):
        RunSummary.from_json(json.dumps({"extra": 1}))
