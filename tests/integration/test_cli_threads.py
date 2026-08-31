# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2 review round 2 finding 2: the RC commands pin OMP_NUM_THREADS themselves; a plain shell reproduces the results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
MODEL = REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"


def test_ensure_single_thread_pins_or_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset -> pinned to 1; 1 -> accepted; anything else -> rejected."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    ensure_single_thread()
    assert os.environ["OMP_NUM_THREADS"] == "1"
    ensure_single_thread()
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS=1"):
        ensure_single_thread()


def test_training_command_runs_without_the_variable(tmp_path: Path) -> None:
    """A subprocess without OMP_NUM_THREADS trains, and its provenance records the pinned value."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = tmp_path / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD,
        SCENARIO,
        PREPROCESS,
        store=store,
        records_root=records,
        exploratory=True,
        now=datetime(2026, 9, 2, tzinfo=UTC),
    )
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    env["ARM_RC_CTRL_STORAGE_ROOT"] = str(root)
    env["QT_QPA_PLATFORM"] = "offscreen"
    report = tmp_path / "training.json"
    recipe = tmp_path / "recipe.toml"
    argv = [
        sys.executable,
        "-m",
        "arm_rc_ctrl.rc.train",
        "--model",
        str(MODEL),
        "--dataset",
        str(processed.record_file),
        "--report",
        str(report),
        "--recipe",
        str(recipe),
        "--records-root",
        str(records),
        "--exploratory",
    ]
    completed = subprocess.run(argv, check=False, capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=600)
    assert completed.returncode == 0, completed.stderr[-2000:]
    printed = json.loads(completed.stdout)
    assert printed["refit_verified"] is True
    provenance = json.loads(report.read_text())["provenance"]
    assert provenance["platform"]["thread_environment"]["OMP_NUM_THREADS"] == "1"
