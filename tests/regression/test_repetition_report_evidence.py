# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-007: the committed report derives from the committed evidence and names assets that exist."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.execution import load_execution
from arm_rc_ctrl.experiments.repetition_evaluation import C11_CAVEAT, load_model_evidence, load_pointer, pointer_name
from arm_rc_ctrl.experiments.repetition_panel import APPROVED_RULE
from arm_rc_ctrl.experiments.repetition_recipes import panel_arms
from arm_rc_ctrl.experiments.repetition_report import (
    ANIMATION_DIR,
    PLOT_DIR,
    REPRESENTATIVE_ARMS,
    REPRESENTATIVE_RULE,
    load_report,
    render_report_markdown,
)
from arm_rc_ctrl.provenance import sha256_file, verify_artifact
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_repeated_demonstration"
EVIDENCE = DOCS / "repetition_report_v1.json"
MARKDOWN = DOCS / "repetition_report_v1.md"


def test_report_binds_its_sources_and_covers_the_panel() -> None:
    """Every source digest matches its committed file; the tables cover all 120 configurations and 72 comparisons."""
    report = load_report(EVIDENCE)
    for name, digest in report.sources.items():
        assert sha256_file(DOCS / name) == digest, name
    assert report.canonical_execution_identity == load_execution(DOCS / "execution_environment_v1.json").identity
    assert not report.provenance.project_dirty
    assert not report.provenance.exploratory
    behavioral = [arm for arm in panel_arms() if arm.behavioral]
    assert [(o.panel_label, o.arm) for o in report.outcomes] == [
        (label, arm.label) for label in APPROVED_RULE.labels for arm in behavioral
    ]
    assert report.n_feasible + report.n_rc_gate_failure == 120
    assert len(report.equivalence) == 108  # 36 absolute comparisons plus 36 residual ones with two quantities
    assert sum(1 for e in report.equivalence if e.accepted_exception) == 1
    assert len(report.speeds) == 120
    assert len(report.costs) == 26
    assert all(c.models == 6 and c.fits_timed == 6 for c in report.costs)
    assert report.c11_caveat == C11_CAVEAT
    assert report.representative_rule == REPRESENTATIVE_RULE
    assert len(report.representatives) == 6 * len(REPRESENTATIVE_ARMS)
    assert report.velocity_abort == (12.0, 12.0)
    assert report.historical_velocity_limit == (6.0, 6.0)
    # Every within-formulation and cross-formulation comparison of plan section 7.2 at every entry and count.
    assert len(report.paired) == 6 * (3 * 6 + 1 + 3 + 3)
    assert all(p.shared_pairs <= 130 for p in report.paired)
    scaled = [p for p in report.paired if p.comparison.startswith("absolute/R-scaled/")]
    assert all(not p.verdict_changed for p in scaled)  # R-scaled shares S's verdict everywhere
    assert render_report_markdown(report) == MARKDOWN.read_text(encoding="utf-8")


def test_report_assets_exist_and_representatives_name_stored_runs() -> None:
    """Every plot and animation exists; the representatives' runs are the manifests' first evaluated pairs."""
    report = load_report(EVIDENCE)
    assert len(report.plots) == 4 + len(report.representatives)
    for plot in report.plots:
        assert (DOCS / PLOT_DIR / plot).stat().st_size > 0
    assert len(report.animations) == 3
    for animation in report.animations:
        assert (DOCS / ANIMATION_DIR / animation).stat().st_size > 0
    try:
        store = open_storage()
    except Exception as exc:  # noqa: BLE001 - any setup failure just means no store on this runner
        pytest.skip(f"external storage unavailable: {exc}")
    for rep in report.representatives:
        pointer = load_pointer(DOCS / "evidence" / pointer_name("model", f"{rep.panel_label}/{rep.arm}"))
        evidence = load_model_evidence(verify_artifact(store, pointer.payload))
        first = evidence.pairs[0]
        assert (rep.scenario_id, rep.tracker, rep.status) == (first.scenario_id, first.tracker, first.status)
        assert rep.rc_run == (None if first.run is None else first.run.artifact_id)
        assert rep.plot is not None
