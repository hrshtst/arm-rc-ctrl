# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-011: the recovery evaluation locks match the committed pilot and share nothing with development."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.data.records import load_catalog
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, load_processed_record
from arm_rc_ctrl.experiments.confirmatory import load_confirmatory
from arm_rc_ctrl.experiments.esn_search import load_esn_search
from arm_rc_ctrl.experiments.perturbation_pilot import select_levels, summarize_levels
from arm_rc_ctrl.experiments.perturbations import (
    load_development_robustness,
    robustness_scenarios,
)
from arm_rc_ctrl.experiments.recovery_pilot import (
    as_core,
    load_recovery_pilot_protocol,
    load_recovery_pilot_report,
    render_recovery_markdown,
)
from arm_rc_ctrl.experiments.tuning import load_protocol as load_tuning_protocol
from arm_rc_ctrl.rc.augment import SEED_NAMESPACE
from arm_rc_ctrl.rc.warmup import APPROVED_WARMUPS_S
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
EXPERIMENT = REPO_ROOT / "docs" / "experiments" / "task_1a_state_conditioned_recovery"
CONFIRMATORY = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_confirmatory_v1.toml"
DEVELOPMENT = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
PILOT_PROTOCOL = REPO_ROOT / "configs" / "studies" / "task_1a_recovery_pilot_v1.toml"
PILOT_REPORT = EXPERIMENT / "recovery_pilot_v1.json"
PILOT_MARKDOWN = EXPERIMENT / "recovery_pilot_v1.md"
RECOVERY_DATASET = "processed-20260903-ce343c8ce6a5"

M3_CONFIRMATORY_SEEDS = frozenset({20260901, 20260902, 20260903, 20260904, 20260905})
M3_DEVELOPMENT_FILES = (
    REPO_ROOT / "configs" / "evaluations" / "task_1a_robustness_dev_v1.toml",
    REPO_ROOT / "configs" / "evaluations" / "task_1a_robustness_dev_v2.toml",
)
M3_STUDY_FILES = (
    REPO_ROOT / "configs" / "studies" / "baseline_gains_1a.toml",
    REPO_ROOT / "configs" / "studies" / "baseline_gains_1a_v2.toml",
)
M3_ESN_FILES = (
    REPO_ROOT / "configs" / "studies" / "esn_search_1a.toml",
    REPO_ROOT / "configs" / "studies" / "esn_search_1a_v2.toml",
)


def test_lock_matches_the_committed_recovery_pilot() -> None:
    """Locked levels equal the pilot's selection, which re-derives from the stored cases and rules."""
    protocol = load_confirmatory(CONFIRMATORY)
    report = load_recovery_pilot_report(PILOT_REPORT)
    pilot = load_recovery_pilot_protocol(PILOT_PROTOCOL)
    assert protocol.locked is True
    assert protocol.pilot_report == PILOT_REPORT
    assert protocol.scenario == pilot.scenario == REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
    selection = report.selection
    assert protocol.posture.small_magnitude_rad == selection.posture_small_rad
    assert protocol.posture.large_magnitude_rad == selection.posture_large_rad
    assert protocol.force.magnitude_n == selection.force_magnitude_n
    assert protocol.force.start_s == selection.force_start_s
    assert protocol.force.duration_s == selection.force_duration_s
    assert protocol.force.directions_deg == selection.force_directions_deg
    assert report.protocol == pilot.name
    assert report.rules == pilot.selection
    assert report.warmup_s == pilot.warmup_s
    assert report.warmup_s in APPROVED_WARMUPS_S
    core = as_core(pilot)
    assert report.levels == summarize_levels(core, report.cases)
    assert report.selection == select_levels(core, report.levels)
    assert len(report.cases) == len(pilot.baselines) * (
        len(pilot.posture.magnitudes) * len(pilot.posture.directions)
        + len(pilot.force.magnitudes) * len(pilot.force.directions_deg)
    )
    assert render_recovery_markdown(report) == PILOT_MARKDOWN.read_text(encoding="utf-8")


def test_pilot_is_confirmatory_grade_and_bound_to_the_recovery_dataset() -> None:
    """The pilot ran from a clean checkout on the committed recovery dataset."""
    report = load_recovery_pilot_report(PILOT_REPORT)
    assert report.provenance.project_dirty is False
    record = load_processed_record(REPO_ROOT / "data" / "records" / "processed" / f"{RECOVERY_DATASET}.toml")
    assert isinstance(record, RecoveryDatasetRecord)
    assert report.dataset == RECOVERY_DATASET
    assert [a.sha256 for a in report.provenance.artifacts] == [record.artifact.payload.sha256]
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.toml")
    assert catalog.find(RECOVERY_DATASET) is not None


def test_locked_levels_sit_at_the_pilot_safety_boundary() -> None:
    """Each locked level is safe in the pilot and the next grid level of its kind is not."""
    protocol = load_confirmatory(CONFIRMATORY)
    report = load_recovery_pilot_report(PILOT_REPORT)
    posture = {lv.magnitude: lv for lv in report.levels if lv.kind == "posture"}
    force = {lv.magnitude: lv for lv in report.levels if lv.kind == "force"}
    for magnitude in (protocol.posture.small_magnitude_rad, protocol.posture.large_magnitude_rad):
        assert posture[magnitude].safe
        assert posture[magnitude].nontrivial
    assert force[protocol.force.magnitude_n].safe
    assert force[protocol.force.magnitude_n].nontrivial
    assert min(m for m, lv in posture.items() if not lv.safe) > protocol.posture.large_magnitude_rad
    assert min(m for m, lv in force.items() if not lv.safe) > protocol.force.magnitude_n
    # The task-relative pulse lies inside the movement interval [0, 3] s of the cropped episode.
    assert protocol.force.start_s > 0.0
    assert protocol.force.start_s + protocol.force.duration_s <= 3.0


def test_development_shares_the_envelope_with_disjoint_seeds() -> None:
    """One method-independent envelope; development and confirmatory differ only in their seeds."""
    protocol = load_confirmatory(CONFIRMATORY)
    development = load_development_robustness(DEVELOPMENT)
    assert development.posture == protocol.posture
    assert development.force == protocol.force
    assert development.scenario == protocol.scenario
    assert not set(development.seeds) & set(protocol.seeds)


def test_allocation_is_m3_shaped_for_both_splits() -> None:
    """Both seed sets generate exactly 1 + 20 + 20 + 4 + 20 = 65 scenarios within the joint limits."""
    scenario = load_scenario(REPO_ROOT / "configs" / "tasks" / "task_1a.toml")
    record = load_processed_record(REPO_ROOT / "data" / "records" / "processed" / f"{RECOVERY_DATASET}.toml")
    assert isinstance(record, RecoveryDatasetRecord)
    lower = [link.q_min for link in scenario.robot.links]
    upper = [link.q_max for link in scenario.robot.links]
    for levels in (load_confirmatory(CONFIRMATORY), load_development_robustness(DEVELOPMENT)):
        scenarios = robustness_scenarios(levels, nominal=record.q0_ref, lower=lower, upper=upper)
        by_class: dict[str, int] = {}
        for item in scenarios:
            by_class[item.kind] = by_class.get(item.kind, 0) + 1
        assert by_class == {"nominal": 1, "posture_small": 20, "posture_large": 20, "force": 4, "combined": 20}
        assert len(scenarios) == 65
        assert len({s.scenario_id for s in scenarios}) == 65


def test_seed_namespaces_are_mutually_disjoint() -> None:
    """Augmentation, recovery development, and recovery confirmatory namespaces never overlap M3 or each other."""
    confirmatory = set(load_confirmatory(CONFIRMATORY).seeds)
    development = set(load_development_robustness(DEVELOPMENT).seeds)
    assert not confirmatory & development
    assert not (confirmatory | development) & M3_CONFIRMATORY_SEEDS
    assert SEED_NAMESPACE not in confirmatory | development
    m3_development: set[int] = set()
    for file in M3_DEVELOPMENT_FILES:
        m3_development |= set(load_development_robustness(file).seeds)
    for file in M3_STUDY_FILES:
        m3_development.add(load_tuning_protocol(file).sampler_seed)
    for file in M3_ESN_FILES:
        m3_development.add(load_esn_search(file).sampler.seed)
    assert not (confirmatory | development) & m3_development


def test_no_confirmatory_control_run_exists() -> None:
    """The lock is definitional: no recovery confirmatory suite evidence exists anywhere."""
    protocol = load_confirmatory(CONFIRMATORY)
    assert protocol.locked is True
    # The recovery experiment directory holds pilot, augmentation, and timing evidence only;
    # a confirmatory suite report would be a *confirmatory* file and requires separate
    # owner authorization (M3R-017).
    assert not [p.name for p in EXPERIMENT.iterdir() if "confirmatory" in p.name.lower()]
