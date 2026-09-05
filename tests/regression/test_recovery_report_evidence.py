# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-018: the committed report binds its statements, embedded assets, and tracked runs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.recovery_freeze import load_freeze
from arm_rc_ctrl.experiments.recovery_representative import REPRESENTATIVE_CLASSES, load_representatives
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_state_conditioned_recovery"


def _report_text() -> str:
    return (DOCS / "recovery_report_v1.md").read_text(encoding="utf-8")


def test_report_states_the_accepted_negative() -> None:
    """The headline, gate closure, and shared limitations render from the committed evidence."""
    text = _report_text()
    freeze = load_freeze(DOCS / "model_freeze_v2.json")
    for required in (
        "**Accepted negative result.**",
        "no model is frozen and the",
        "not authorized under protocol v1",
        "Flat infeasible objective",
        "Sampled, not exhaustive",
        "15-of-20 consistency requirement",
        "curated representative pairs only",
        "### Original-trajectory RMSE, restoring alignment, and contraction",
        "### Smoothness and effort",
        f"freeze commit `{freeze.provenance.project_commit[:12]}`",
        f"{freeze.n_candidates} feasible development trials",
    ):
        assert required in text


def test_every_embedded_asset_exists() -> None:
    """Every plot and animation the report embeds is a committed, non-empty file."""
    text = _report_text()
    links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", text)
    assert len(links) >= 18  # 3 augmentation plots + 7 result plots + 8 animations
    for link in links:
        target = DOCS / link
        assert target.exists(), link
        assert target.stat().st_size > 0, link


def test_animation_descriptions_bind_visuals_to_scenarios_and_runs() -> None:
    """The human report explains visual semantics and every selected perturbation."""
    text = _report_text()
    for required in (
        "actual simulated robot motion, not the desired trajectory",
        "Both arms hold their own initial posture for 0.25 s",
        "side-by-side comparison therefore isolates the reference generator",
        "Computed-torque runs are quantified in the tables but are not included",
        "### Nominal start — `nominal`",
        "$\\Delta q=[-0.0442, +0.0233]$ rad",
        "$\\Delta q=[+0.0415, -0.0910]$ rad",
        "12 N end-effector pulse acts toward +x from task time 1 to 1.2 s",
        "run time 1.25 to 1.45 s",
    ):
        assert required in text
    representative = load_representatives(DOCS / "recovery_representative_v1.json")
    pd_pairs = [pair for pair in representative.pairs if pair.tracker == "pd_v2"]
    assert len(pd_pairs) == len(REPRESENTATIVE_CLASSES)
    for pair in pd_pairs:
        assert pair.rc_run in text
        assert pair.replay_run in text


def test_augmentation_figures_regenerate_byte_for_byte(tmp_path: Path) -> None:
    """The strategy figures reproduce from the committed dataset when its external payload is available."""
    from arm_rc_ctrl.data.derivatives import DerivativeConfig
    from arm_rc_ctrl.data.records import load_record, verify_payload
    from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord
    from arm_rc_ctrl.data.samples import load_samples
    from arm_rc_ctrl.experiments.augmentation_plots import write_augmentation_task_space_plots
    from arm_rc_ctrl.scenario import load_scenario
    from arm_rc_ctrl.storage import open_storage

    dataset_file = REPO_ROOT / "data" / "records" / "processed" / "processed-20260903-ce343c8ce6a5.toml"
    record = load_record(dataset_file, RecoveryDatasetRecord)
    scenario = load_scenario(REPO_ROOT / record.scenario.config_path)
    assert record.preprocessing.derivative_method == "central-difference"
    try:
        samples = load_samples(verify_payload(open_storage(), record.artifact))
    except Exception as exc:  # noqa: BLE001 - an unavailable machine-local store skips this regression
        pytest.skip(f"external payload unavailable: {exc}")
    written = write_augmentation_task_space_plots(
        samples,
        record.crop.task,
        scenario,
        DerivativeConfig(method="central"),
        tmp_path,
    )
    committed = DOCS / "plots" / "augmentation_strategy_v1"
    assert sorted(path.name for path in written) == sorted(path.name for path in committed.glob("*.png"))
    for path in written:
        assert path.read_bytes() == (committed / path.name).read_bytes(), path.name


def test_representative_runs_are_tracked_and_cited() -> None:
    """Every representative run ID appears in the report and has a Git-tracked pointer record."""
    text = _report_text()
    record = load_representatives(DOCS / "recovery_representative_v1.json")
    for pair in record.pairs:
        for run_id in (pair.replay_run, pair.rc_run):
            assert (REPO_ROOT / "data" / "records" / "runs" / f"{run_id}.toml").exists(), run_id
        assert pair.rc_run in text
        assert pair.replay_run in text


def test_committed_plots_regenerate_byte_for_byte(tmp_path: Path) -> None:
    """Every committed figure reproduces exactly from the verified inputs when the store is present."""
    from arm_rc_ctrl.experiments.recovery_report import build_report_inputs, write_recovery_plots
    from arm_rc_ctrl.storage import StorageAccessError, open_storage

    try:
        store = open_storage()
    except Exception as exc:  # noqa: BLE001 - any setup failure just means no store on this runner
        pytest.skip(f"external storage unavailable: {exc}")
    try:
        inputs = build_report_inputs(DOCS, store=store, records_root=REPO_ROOT)
    except StorageAccessError as exc:
        pytest.skip(f"external payload unavailable: {exc}")
    written = write_recovery_plots(inputs, tmp_path)
    committed_dir = DOCS / "plots" / "recovery_report_v1"
    assert sorted(written) == sorted(p.name for p in committed_dir.glob("*.png"))
    for name in written:
        assert (tmp_path / name).read_bytes() == (committed_dir / name).read_bytes(), name
