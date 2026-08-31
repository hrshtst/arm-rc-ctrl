# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-027: the frozen task 1-a baselines replay deterministically and reproduce committed expectations.

Two datasets are covered. The fixture dataset (built from the committed raw
fixture) runs everywhere, including CI; the committed task 1-a dataset needs
the machine-local store and is skipped with a reason when it is unavailable.
Expectations live under ``tests/fixtures/regression`` and are rewritten with
``pytest --update-baselines`` after an intentional change.
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
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
from arm_rc_ctrl.experiments.baselines import (
    FROZEN_BASELINES,
    BaselineExpectations,
    baseline_method,
    build_expectations,
    compare_snapshots,
    frozen_baseline_digest,
    load_expectations,
    load_frozen_baseline,
    snapshot,
    write_expectations,
)
from arm_rc_ctrl.experiments.replay import ReplayResult, run_replay
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import StorageError, StorageRoot, open_storage

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.data.samples import SampleSet

pytestmark = [pytest.mark.regression, pytest.mark.integration]

REPO_ROOT = repository_root()
METHODS = tuple(sorted(FROZEN_BASELINES))
EXPECTATIONS_DIR = REPO_ROOT / "tests" / "fixtures" / "regression"
FIXTURE_EXPECTATIONS = EXPECTATIONS_DIR / "baselines_fixture.toml"
TASK_1A_EXPECTATIONS = EXPECTATIONS_DIR / "baselines_task_1a.toml"
FIXTURE_RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
FIXTURE_RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
FIXTURE_SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
TASK_1A_RAW_ID = "raw-20260830-b5adde395f1c"
TASK_1A_SCENARIO = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
STUDY_REPORTS = REPO_ROOT / "docs" / "experiments" / "task_1a"
FIXED_TIME = datetime(2026, 8, 31, 9, 0, 0, tzinfo=UTC)
_CLOCK = [0]


def _now() -> datetime:
    """Distinct timestamps keep run IDs unique within a shared store."""
    _CLOCK[0] += 1
    return FIXED_TIME + timedelta(minutes=_CLOCK[0])


class Dataset(NamedTuple):
    """A processed dataset bound to its scenario, plus a store that receives the replay runs."""

    runs: StorageRoot
    samples: SampleSet
    record: ProcessedDatasetRecord
    scenario: ScenarioConfig
    scenario_file: Path


def _replay(dataset: Dataset, method: str, *, initial_q: tuple[float, ...] | None = None) -> ReplayResult:
    return run_replay(
        dataset.scenario,
        dataset.scenario_file,
        dataset.record,
        dataset.samples,
        load_frozen_baseline(method),
        store=dataset.runs,
        exploratory=True,
        now=_now(),
        initial_q=initial_q,
    )


def _check_or_update(path: Path, actual: BaselineExpectations, *, update: bool) -> BaselineExpectations:
    """Compare against the committed expectations, or rewrite them when asked to."""
    if update:
        write_expectations(path, actual)
        pytest.skip(f"rewrote {path.relative_to(REPO_ROOT)}; review the diff and re-run without --update-baselines")
    expected = load_expectations(path)
    assert expected.scenario == actual.scenario
    assert expected.dataset == actual.dataset
    assert expected.gains == actual.gains, "frozen gains changed; the expectations were taken with other gains"
    assert set(expected.runs) == set(METHODS)
    mismatches = [
        f"{method}: {message}"
        for method in METHODS
        for message in compare_snapshots(actual.runs[method], expected.runs[method], expected.tolerances)
    ]
    assert not mismatches, "\n".join(mismatches)
    return expected


@pytest.fixture(scope="module")
def fixture_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """The fixture dataset preprocessed into a module-scoped temporary store."""
    base = tmp_path_factory.mktemp("baselines-fixture")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(FIXTURE_RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(FIXTURE_RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    result = preprocess_demonstration(
        FIXTURE_RAW_RECORD,
        FIXTURE_SCENARIO,
        PREPROCESS,
        store=store,
        records_root=records,
        exploratory=True,
        now=FIXED_TIME,
    )
    return Dataset(store, result.samples, result.record, load_scenario(FIXTURE_SCENARIO), FIXTURE_SCENARIO)


@pytest.fixture(scope="module")
def task_1a_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """The committed task 1-a dataset read from the configured store; runs go to a temporary store."""
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.toml")
    processed = [
        load_record(REPO_ROOT / entry.record, ProcessedDatasetRecord)
        for entry in catalog.artifacts
        if entry.kind == "processed"
    ]
    (record,) = [r for r in processed if r.artifact.origin.sources == (TASK_1A_RAW_ID,)]
    try:
        payload = verify_payload(open_storage(), record.artifact)
    except (StorageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        pytest.skip(f"configured external store with {record.artifact.artifact_id} not available: {exc}")
    root = tmp_path_factory.mktemp("baselines-task-1a") / "store"
    root.mkdir()
    runs = StorageRoot(root, repositories=(REPO_ROOT,))
    return Dataset(runs, load_samples(payload), record, load_scenario(TASK_1A_SCENARIO), TASK_1A_SCENARIO)


@pytest.mark.parametrize("method", METHODS)
def test_frozen_gains_are_the_ones_selected_by_their_study(method: str) -> None:
    """The frozen files hold exactly the gains the committed study reports selected."""
    report = json.loads((STUDY_REPORTS / f"baseline_gains_{method}.json").read_text())
    best = report["result"]["best"]
    assert best["feasible"] is True
    config = load_frozen_baseline(method)
    assert best["gains"] == {"type": baseline_method(method), "kp": list(config.kp), "kd": list(config.kd)}
    for expectations in (FIXTURE_EXPECTATIONS, TASK_1A_EXPECTATIONS):
        assert load_expectations(expectations).gains[method] == frozen_baseline_digest(method), expectations.name


@pytest.mark.parametrize("method", METHODS)
def test_fixture_replays_are_bitwise_deterministic(fixture_dataset: Dataset, method: str) -> None:
    """Two replays of the same inputs produce identical telemetry, persisted digests, and snapshots."""
    first = _replay(fixture_dataset, method)
    second = _replay(fixture_dataset, method)
    assert first.pointer.artifact.artifact_id != second.pointer.artifact.artifact_id  # distinct runs ...
    assert first.summary.arrays == second.summary.arrays  # ... with bitwise-identical persisted arrays
    for name, array in first.run.arrays.arrays.items():
        assert np.array_equal(array, second.run.arrays.arrays[name]), name
    assert snapshot(first) == snapshot(second)


def test_fixture_replays_match_committed_expectations(fixture_dataset: Dataset, *, update_baselines: bool) -> None:
    """Both frozen baselines reproduce the committed fixture snapshots within the declared tolerances."""
    results = {method: _replay(fixture_dataset, method) for method in METHODS}
    actual = build_expectations(fixture_dataset.scenario.name, fixture_dataset.record.artifact.artifact_id, results)
    _check_or_update(FIXTURE_EXPECTATIONS, actual, update=update_baselines)


def test_snapshot_requires_every_metric(fixture_dataset: Dataset) -> None:
    """A run that stops early has no complete metrics and cannot be snapshotted silently."""
    strict = dataclasses.replace(
        fixture_dataset.scenario, limits=dataclasses.replace(fixture_dataset.scenario.limits, velocity=(0.3, 0.3))
    )
    result = _replay(fixture_dataset._replace(scenario=strict), "pd")
    assert result.summary.termination.kind == "limit_violation"
    with pytest.raises(ValueError, match="terminated with 'limit_violation' before every metric"):
        snapshot(result)


def test_task_1a_replays_match_committed_expectations(task_1a_dataset: Dataset, *, update_baselines: bool) -> None:
    """On the committed task 1-a dataset both frozen baselines reproduce their snapshots and succeed."""
    results = {method: _replay(task_1a_dataset, method) for method in METHODS}
    actual = build_expectations(task_1a_dataset.scenario.name, task_1a_dataset.record.artifact.artifact_id, results)
    for method, result in results.items():
        assert result.summary.outcome.success is True, method
        assert result.summary.outcome.criteria == {
            "completed": True,
            "dwell_in_tolerance": True,
            "dwell_stationary": True,
        }
        assert result.summary.provenance.artifacts[0].sha256 == task_1a_dataset.record.artifact.payload.sha256
    _check_or_update(TASK_1A_EXPECTATIONS, actual, update=update_baselines)


def test_task_1a_paired_baselines_rank_consistently(task_1a_dataset: Dataset) -> None:
    """Paired on the same reference and posture, computed torque tracks better than PD by the committed margin."""
    pd = snapshot(_replay(task_1a_dataset, "pd"))
    ct = snapshot(_replay(task_1a_dataset, "computed_torque"))
    assert ct.joint_rmse < pd.joint_rmse
    expected = load_expectations(TASK_1A_EXPECTATIONS)
    expected_ratio = expected.runs["computed_torque"].joint_rmse / expected.runs["pd"].joint_rmse
    assert math.isclose(ct.joint_rmse / pd.joint_rmse, expected_ratio, rel_tol=2 * expected.tolerances.metric_rel)
    assert ct.n_samples == pd.n_samples == task_1a_dataset.samples.n_samples
