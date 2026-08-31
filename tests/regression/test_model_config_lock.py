# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-018: the versioned model configuration adopts the input-scale pilot's selection and nothing else changes."""

from __future__ import annotations

import dataclasses

import pytest

from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_catalog, load_record
from arm_rc_ctrl.experiments.scale_pilot import (
    load_protocol,
    load_scale_pilot_report,
    render_markdown,
    select_anchor,
    summarize_cells,
)
from arm_rc_ctrl.rc.train import InputTransformSpec, load_model_config
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
PROTOCOL = REPO_ROOT / "configs" / "studies" / "input_scale_pilot_1a.toml"
REPORT = REPO_ROOT / "docs" / "experiments" / "task_1a" / "input_scale_pilot.json"
MARKDOWN = REPO_ROOT / "docs" / "experiments" / "task_1a" / "input_scale_pilot.md"
ANCHOR = REPO_ROOT / "configs" / "models" / "esn_task_1a.toml"
SELECTED = REPO_ROOT / "configs" / "models" / "esn_task_1a_v1.toml"


def test_selected_config_adopts_the_pilot_selection() -> None:
    """The v1 model config differs from the development anchor only by the pilot-selected input scales."""
    report = load_scale_pilot_report(REPORT)
    assert report.selection is not None
    anchor = load_model_config(ANCHOR)
    selected = load_model_config(SELECTED)
    assert selected.input_transform == InputTransformSpec(
        "fixed_scale", report.selection.q_scale, report.selection.dq_scale
    )
    assert selected.esn == anchor.esn
    assert selected.name == "esn-task-1a-v1"
    assert dataclasses.replace(anchor, name=selected.name, input_transform=selected.input_transform) == selected


def test_pilot_is_reproducible_from_its_stored_cases() -> None:
    """The committed report re-derives its cells and selection from the stored variants under the committed protocol."""
    protocol = load_protocol(PROTOCOL)
    report = load_scale_pilot_report(REPORT)
    assert report.protocol == protocol.name
    assert report.rules == protocol.selection
    assert report.model_config == "configs/models/esn_task_1a.toml"
    assert set(report.trackers) == set(protocol.trackers)
    assert len(report.variants) == (
        len(protocol.grid.q_scales)
        * len(protocol.grid.dq_scales)
        * len(protocol.variants.ridge_alphas)
        * len(protocol.variants.reservoir_seeds)
        * len(protocol.trackers)
    )
    assert report.cells == summarize_cells(protocol, report.variants)
    assert report.selection is not None
    assert report.selection == select_anchor(protocol, report.cells)
    assert render_markdown(report) == MARKDOWN.read_text(encoding="utf-8")


def test_pilot_is_development_grade_and_clean() -> None:
    """The pilot ran from a clean checkout on the committed dataset with development seeds only."""
    report = load_scale_pilot_report(REPORT)
    assert report.provenance.project_dirty is False
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.toml")
    processed = [
        load_record(REPO_ROOT / e.record, ProcessedDatasetRecord) for e in catalog.artifacts if e.kind == "processed"
    ]
    (dataset,) = [r for r in processed if r.artifact.origin.sources == ("raw-20260830-b5adde395f1c",)]
    assert report.dataset == dataset.artifact.artifact_id
    confirmatory_seeds = {20260901, 20260902, 20260903, 20260904, 20260905}
    assert not confirmatory_seeds & set(report.provenance.seeds.values())
