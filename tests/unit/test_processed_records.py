# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-002: processed dataset records tie samples.npz arrays to sources, units, phases, and digests."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    ArraySpec,
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
from arm_rc_ctrl.data.samples import PHASE_CODES, SampleSet
from arm_rc_ctrl.data.synthetic import synthetic_arrays as make_arrays
from arm_rc_ctrl.data.synthetic import synthetic_samples as make_samples
from arm_rc_ctrl.repo import repository_root

REPO_ROOT = repository_root()
CREATED = "2026-08-30T04:00:00+00:00"
PAYLOAD_SHA = "5" * 64
ARTIFACT_ID = make_artifact_id("processed", CREATED, PAYLOAD_SHA)
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "records" / f"{ARTIFACT_ID}.toml"
RAW_SOURCE = "raw-20260830-2a97516c354b"


def _samples() -> SampleSet:
    return make_samples(n=6, dof=2, task_dim=2, code_dim=0)


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
        notes="Synthetic example record; the payload does not exist.",
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


def _record(**changes: object) -> ProcessedDatasetRecord:
    samples = _samples()
    base = ProcessedDatasetRecord(
        artifact=_artifact(),
        scenario=SCENARIO,
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(PHASE_CODES),
        preprocessing=Preprocessing(
            resample_period_s=0.01,
            smoothing="butterworth-zero-phase",
            smoothing_params={"order": 4.0, "cutoff_hz": 5.0},
            derivative_method="central-difference",
        ),
        arrays=array_specs(samples),
    )
    return dataclasses.replace(base, **changes)


def _normalization() -> Normalization:
    return Normalization(
        fitted_on=(ARTIFACT_ID,),
        channels={
            "q": ChannelStats(mean=(0.1, 0.2), scale=(1.0, 0.5), replaced_near_zero=(0,)),
            "tip": ChannelStats(mean=(0.3, 0.4), scale=(0.2, 0.3)),
        },
    )


# --- round trip ----------------------------------------------------------------------


def test_processed_record_round_trips_through_toml(tmp_path: Path) -> None:
    """Dump → load reproduces an equal record, with and without normalization."""
    for record in (_record(), _record(normalization=_normalization())):
        path = tmp_path / f"{id(record)}.toml"
        write_record(path, record)
        assert load_record(path, ProcessedDatasetRecord) == record


def test_committed_example_record_is_canonical() -> None:
    """The fixture loads, equals the builder, and re-serializes byte-for-byte."""
    loaded = load_record(FIXTURE, ProcessedDatasetRecord)
    assert loaded == _record()
    assert to_toml(loaded) == FIXTURE.read_text(encoding="utf-8")


def test_record_matches_its_samples_and_detects_drift() -> None:
    """check_samples accepts the described arrays and reports shape/dtype/digest drift."""
    record = _record()
    record.check_samples(_samples())
    arrays = make_arrays(n=6, dof=2, task_dim=2, code_dim=0)
    arrays["q"] = arrays["q"] + 1e-9
    with pytest.raises(ValueError, match=r"samples do not match the record:\nq: recorded"):
        record.check_samples(SampleSet.from_arrays(arrays))
    with pytest.raises(ValueError, match="t: recorded"):
        record.check_samples(make_samples(n=7, dof=2, task_dim=2, code_dim=0))


# --- invariants ------------------------------------------------------------------------


def _with_payload(**changes: object) -> ArtifactRecord:
    payload = dataclasses.replace(_artifact().payload, **changes)
    return _artifact(payload=payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"artifact": _artifact(origin=dataclasses.replace(_artifact().origin, sources=()))},
            "at least one raw source",
        ),
        (
            {
                "artifact": _artifact(
                    origin=dataclasses.replace(_artifact().origin, sources=("model-20260830-000000000000",))
                )
            },
            "at least one raw source",
        ),
        ({"artifact": _with_payload(format="sklog.npz")}, "processed payload format must be 'samples.npz'"),
        (
            {"artifact": _with_payload(uri=f"armrc://processed/{ARTIFACT_ID}/data.npz")},
            "must be stored at armrc://processed/",
        ),
        ({"units": {**CANONICAL_UNITS, "q": "deg"}}, "units must be exactly"),
        ({"units": {k: v for k, v in CANONICAL_UNITS.items() if k != "phase"}}, "units must be exactly"),
        ({"phases": {"prime": 0, "move": 2, "dwell": 1}}, "phases must be exactly"),
        ({"n_samples": 1}, "n_samples >= 2"),
        ({"scenario": dataclasses.replace(SCENARIO, dof=3, initial_q=(0.0, 0.0, 0.0))}, "scenario.dof 3 != dof 2"),
        ({"task_code_dim": -1}, "task_code_dim >= 0"),
        ({"n_samples": 7}, r"arrays.t.shape must be \[7\], got \[6\]"),
        ({"dof": 3}, "scenario.dof 2 != dof 3"),
        (
            {"dof": 3, "scenario": dataclasses.replace(SCENARIO, dof=3, initial_q=(0.0, 0.0, 0.0))},
            r"arrays.q.shape must be \[6, 3\], got \[6, 2\]",
        ),
        ({"arrays": {k: v for k, v in array_specs(_samples()).items() if k != "phase"}}, "arrays must be exactly"),
        ({"arrays": dict(reversed(array_specs(_samples()).items()))}, "arrays must be exactly .* in order"),
        (
            {
                "arrays": {
                    **array_specs(_samples()),
                    "phase": dataclasses.replace(array_specs(_samples())["phase"], dtype="float64"),
                }
            },
            "arrays.phase.dtype must be 'int64'",
        ),
        (
            {
                "arrays": {
                    **array_specs(_samples()),
                    "t": dataclasses.replace(array_specs(_samples())["t"], dtype="int64"),
                }
            },
            "arrays.t.dtype must be 'float64'",
        ),
        (
            {"normalization": Normalization((ARTIFACT_ID,), {"q": ChannelStats((0.0,), (1.0,))})},
            "normalization.channels.q must have 2 columns",
        ),
    ],
)
def test_processed_record_invariants(changes: dict[str, object], message: str) -> None:
    """Sources, payload placement, units, phases, dimensions, array specs, and normalization are validated."""
    with pytest.raises(ValueError, match=message):
        _record(**changes)


def test_wrong_kind_is_rejected() -> None:
    """A raw artifact cannot back a processed record."""
    raw_id = make_artifact_id("raw", CREATED, PAYLOAD_SHA)
    artifact = _artifact(
        kind="raw",
        artifact_id=raw_id,
        payload=Payload(f"armrc://raw/{raw_id}/samples.npz", PAYLOAD_SHA, 1, "samples.npz", 1),
    )
    with pytest.raises(ValueError, match="must have kind 'processed'"):
        _record(artifact=artifact)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: ArraySpec(shape=(), dtype="float64", sha256="0" * 64), "non-empty with non-negative dimensions"),
        (lambda: ArraySpec(shape=(6, -1), dtype="float64", sha256="0" * 64), "non-negative dimensions"),
        (lambda: ArraySpec(shape=(6,), dtype="float64", sha256="zz"), "array sha256 must be 64"),
        (lambda: Preprocessing(0.0, "none", {}, "central-difference"), "resample_period_s must be positive"),
        (lambda: Preprocessing(0.01, "Butterworth", {}, "central-difference"), "smoothing must be a lowercase label"),
        (lambda: Preprocessing(0.01, "none", {"cutoff": float("inf")}, "x"), r"smoothing_params\[0\] must be finite"),
        (lambda: ChannelStats((0.0, 0.0), (1.0,)), "same non-zero length"),
        (lambda: ChannelStats((0.0,), (0.0,)), "scale entries must be positive"),
        (lambda: ChannelStats((0.0, 0.0), (1.0, 2.0), (1,)), "replaced near-zero scales must equal 1.0"),
        (lambda: ChannelStats((0.0, 0.0), (1.0, 1.0), (0, 0)), "unique column indices"),
        (lambda: Normalization((), {}), "fitted_on must list at least one artifact ID"),
        (lambda: Normalization((ARTIFACT_ID,), {"t": ChannelStats((0.0,), (1.0,))}), r"unknown arrays \['t'\]"),
    ],
)
def test_sub_record_invariants(build: object, message: str) -> None:
    """Array specs, preprocessing, and normalization statistics validate their own fields."""
    with pytest.raises(ValueError, match=message):
        cast("object", build)()  # type: ignore[operator]


def test_loading_reports_located_errors(tmp_path: Path) -> None:
    """A processed record file with drifted metadata fails to load with a location."""
    text = to_toml(_record()).replace('q = "rad"', 'q = "deg"')
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=r"bad\.toml: <root>: units must be exactly"):
        load_record(path, ProcessedDatasetRecord)


def test_array_specs_describe_every_array() -> None:
    """array_specs mirrors shapes, dtypes, and digests of the sample set."""
    samples = _samples()
    specs = array_specs(samples)
    assert list(specs) == list(samples.arrays())
    assert specs["phase"] == ArraySpec((6,), "int64", samples.digests()["phase"])
    assert specs["task_code"].shape == (6, 0)
    assert np.all([spec.dtype == "float64" for name, spec in specs.items() if name != "phase"])


def test_check_scenario_binds_the_dataset_to_a_scenario_file(tmp_path: Path) -> None:
    """The record accepts only the scenario file whose digest it was derived under."""
    record = _record()
    other = tmp_path / "task.toml"
    other.write_text("name = 'x'\n")
    with pytest.raises(ValueError, match="was derived under scenario digest 222222222222"):
        record.check_scenario(other)
    from arm_rc_ctrl.provenance import sha256_file

    bound = dataclasses.replace(record, scenario=dataclasses.replace(SCENARIO, config_sha256=sha256_file(other)))
    bound.check_scenario(other)
