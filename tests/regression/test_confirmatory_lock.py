# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-028: the locked confirmatory protocol matches the committed pilot and shares nothing with development."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_catalog, load_record
from arm_rc_ctrl.experiments.baselines import load_frozen_baseline
from arm_rc_ctrl.experiments.confirmatory import check_against_pilot, load_confirmatory
from arm_rc_ctrl.experiments.perturbation_pilot import (
    load_pilot_report,
    load_protocol,
    render_markdown,
    select_levels,
    summarize_levels,
)
from arm_rc_ctrl.experiments.tuning import load_protocol as load_tuning_protocol
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
CONFIRMATORY = REPO_ROOT / "configs" / "evaluations" / "task_1a_confirmatory.toml"
PILOT_PROTOCOL = REPO_ROOT / "configs" / "studies" / "perturbation_pilot_1a.toml"
GAIN_STUDY = REPO_ROOT / "configs" / "studies" / "baseline_gains_1a.toml"
PILOT_REPORT = REPO_ROOT / "docs" / "experiments" / "task_1a" / "perturbation_pilot.json"
PILOT_MARKDOWN = REPO_ROOT / "docs" / "experiments" / "task_1a" / "perturbation_pilot.md"


def test_lock_matches_the_committed_pilot() -> None:
    """Locked levels equal the pilot's selection, which itself follows from the stored cases and rules."""
    protocol = load_confirmatory(CONFIRMATORY)
    report = load_pilot_report(PILOT_REPORT)
    pilot = load_protocol(PILOT_PROTOCOL)
    check_against_pilot(protocol, report)
    assert protocol.locked is True
    assert protocol.pilot_report == PILOT_REPORT
    assert protocol.scenario == pilot.scenario == REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
    assert report.protocol == pilot.name
    assert report.rules == pilot.selection
    assert report.baselines == {method: load_frozen_baseline(method) for method in pilot.baselines}
    assert report.levels == summarize_levels(pilot, report.cases)
    assert report.selection == select_levels(pilot, report.levels)
    assert len(report.cases) == len(pilot.baselines) * (
        len(pilot.posture.magnitudes) * len(pilot.posture.directions)
        + len(pilot.force.magnitudes) * len(pilot.force.directions_deg)
    )


def test_pilot_is_confirmatory_grade() -> None:
    """The pilot ran from a clean checkout on the committed task 1-a dataset."""
    report = load_pilot_report(PILOT_REPORT)
    assert report.provenance.project_dirty is False
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.toml")
    processed = [
        load_record(REPO_ROOT / e.record, ProcessedDatasetRecord) for e in catalog.artifacts if e.kind == "processed"
    ]
    (dataset,) = [r for r in processed if r.artifact.origin.sources == ("raw-20260830-b5adde395f1c",)]
    assert report.dataset == dataset.artifact.artifact_id
    assert [a.sha256 for a in report.provenance.artifacts] == [dataset.artifact.payload.sha256]
    assert render_markdown(report) == PILOT_MARKDOWN.read_text(encoding="utf-8")


def test_locked_levels_sit_at_the_pilot_safety_boundary() -> None:
    """Each locked level is safe in the pilot and the next grid level of its kind is not."""
    protocol = load_confirmatory(CONFIRMATORY)
    report = load_pilot_report(PILOT_REPORT)
    posture = {lv.magnitude: lv for lv in report.levels if lv.kind == "posture"}
    force = {lv.magnitude: lv for lv in report.levels if lv.kind == "force"}
    for magnitude in (protocol.posture.small_magnitude_rad, protocol.posture.large_magnitude_rad):
        assert posture[magnitude].safe
        assert posture[magnitude].nontrivial
    assert force[protocol.force.magnitude_n].safe
    assert force[protocol.force.magnitude_n].nontrivial
    assert min(m for m, lv in posture.items() if not lv.safe) > protocol.posture.large_magnitude_rad
    assert min(m for m, lv in force.items() if not lv.safe) > protocol.force.magnitude_n
    assert protocol.force.start_s >= 1.0  # the pulse lies inside the movement interval [1, 4] s
    assert protocol.force.start_s + protocol.force.duration_s <= 4.0


def test_confirmatory_seeds_and_levels_are_separate_from_development() -> None:
    """No confirmatory seed is a development seed; no development offset coincides with a locked level."""
    protocol = load_confirmatory(CONFIRMATORY)
    study = load_tuning_protocol(GAIN_STUDY)
    protocol.forbid_seeds([study.sampler_seed], "the baseline gain study")
    norms = [float(np.linalg.norm(offset)) for offset in study.development.initial_posture_offsets]
    for level in (protocol.posture.small_magnitude_rad, protocol.posture.large_magnitude_rad):
        assert not any(math.isclose(norm, level, rel_tol=1e-9) for norm in norms), level
