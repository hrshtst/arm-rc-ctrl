# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""TOOL-001: the canonical nominal RC+PD v2 run exports as a playable log with the exact recorded state."""

from __future__ import annotations

import numpy as np
import pytest

from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.experiments.playback import export_run_sklog, resolve_pointer
from arm_rc_ctrl.experiments.run_record import RunPointerRecord, load_run
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageError, open_storage

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
CANONICAL = "run-20260831-3fef7abca22a"
"""The nominal RC+PD v2 paired run the report's playback section names."""
SCENARIO = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
FRAMES = 501


def test_canonical_run_exports_with_exact_state_and_task(tmp_path: object) -> None:
    """With the configured store the export carries 501 frames of exact state/telemetry (skips without it)."""
    from pathlib import Path

    from skelarm import StateLog

    pointer_file = resolve_pointer(CANONICAL, REPO_ROOT)
    pointer = load_record(pointer_file, RunPointerRecord)
    try:
        store = open_storage()
        verify_payload(store, pointer.artifact)
    except (StorageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        pytest.skip(f"configured external store with {CANONICAL} not available: {exc}")
    out = export_run_sklog(store, pointer_file, Path(str(tmp_path)) / "canonical.sklog.npz", scenario_file=SCENARIO)
    log = StateLog.load(out)
    run = load_run(store, pointer)
    scenario = load_scenario(SCENARIO)
    assert len(log) == FRAMES
    arrays = run.arrays.arrays
    assert np.array_equal(log.times, arrays["t"])
    for name in ("q", "dq", "tip", "q_desired", "dq_desired", "tracking_error", "tau_requested", "esn_state_norm"):
        assert np.array_equal(log.channel(name), arrays[name]), name
    assert np.array_equal(log.channel("tau"), arrays["tau_applied"])
    task = log.extra["playback"]["task"]
    assert task["target"]["pos"] == [float(v) for v in scenario.task.target]
    assert task["target"]["tolerance"] == scenario.task.tolerance
    assert log.extra["run"]["id"] == CANONICAL
    assert log.extra["run"]["method"] == "rc+pd"
