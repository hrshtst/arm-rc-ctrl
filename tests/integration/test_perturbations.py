# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""A force scenario replays with its pulse recorded in the run's channels and disturbance list (M3-008)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments.confirmatory import ForcePulseLevels
from arm_rc_ctrl.experiments.perturbations import force_scenarios
from arm_rc_ctrl.experiments.replay import ReplayResult, run_replay
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
DEV_PD = REPO_ROOT / "configs" / "controllers" / "pd.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 3, 13, 0, 0, tzinfo=UTC)


def test_force_scenario_replays_with_the_pulse_in_the_run_record(tmp_path: Path) -> None:
    """The ext_force channel equals the pulse inside its window and is zero elsewhere; the summary lists it."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = tmp_path / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    scenario = load_scenario(SCENARIO)
    levels = ForcePulseLevels(magnitude_n=2.0, start_s=0.1, duration_s=0.05, directions_deg=(0.0, 90.0))
    (along_x, along_y) = force_scenarios(levels, scenario.dof)
    results: dict[str, ReplayResult] = {}
    for robustness in (along_x, along_y):
        assert robustness.pulse is not None
        results[robustness.scenario_id] = run_replay(
            scenario,
            SCENARIO,
            processed.record,
            processed.samples,
            load_config(DEV_PD, TrackerConfig),
            store=store,
            exploratory=True,
            now=FIXED_TIME,
            initial_q=robustness.initial_q(scenario.task.initial_q),
            force=robustness.pulse,
        )
    assert set(results) == {"force-2N-000deg", "force-2N-090deg"}
    for scenario_id, result in results.items():
        arrays = result.run.arrays.arrays
        t = arrays["t"]
        ext = arrays["ext_force"]
        pulse = along_x.pulse if scenario_id.endswith("000deg") else along_y.pulse
        assert pulse is not None
        inside = np.array([pulse.active(float(ti)) for ti in t])
        assert 0.04 <= float(t[inside].max() - t[inside].min()) <= 0.06  # the 0.05 s window on the 4 ms grid
        expected = np.array([2.0, 0.0]) if scenario_id.endswith("000deg") else np.array([0.0, 2.0])
        assert np.allclose(ext[inside], expected)
        assert not ext[~inside].any()
        (disturbance,) = result.summary.disturbances
        assert disturbance.start_s == 0.1
        assert disturbance.end_s == pytest.approx(0.15)
        assert disturbance.parameters["magnitude_n"] == pytest.approx(2.0)
    x_ids = results["force-2N-000deg"].pointer.artifact.artifact_id
    y_ids = results["force-2N-090deg"].pointer.artifact.artifact_id
    assert x_ids != y_ids  # the pulse direction changes the run and therefore its content-addressed ID
