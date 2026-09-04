# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-018: the negative-result report renders its tables, statements, and pair figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.data.synthetic import synthetic_task_samples
from arm_rc_ctrl.experiments.evidence import StoredReport
from arm_rc_ctrl.experiments.recovery_ablation import AblationReport, ArmSummary, CandidateCell, CandidateTrial
from arm_rc_ctrl.experiments.recovery_freeze import FreezeRecord, StudyInput
from arm_rc_ctrl.experiments.recovery_report import (
    ReportInputs,
    plot_recovery_pair,
    render_recovery_report,
)
from arm_rc_ctrl.experiments.recovery_representative import (
    SELECTION_RULE,
    PairOutcome,
    RepresentativeRecord,
)
from arm_rc_ctrl.provenance import ArtifactReference, collect_provenance

if TYPE_CHECKING:
    from pathlib import Path

CELL_NAMES = (
    "posture_small:pd_v2",
    "posture_small:computed_torque",
    "posture_large:pd_v2",
    "posture_large:computed_torque",
)


def _candidate(number: int) -> CandidateTrial:
    cells = {
        name: CandidateCell(gap_median=0.5, jump_median=1.5, improving_both=0, n=20, passes=False)
        for name in CELL_NAMES
    }
    return CandidateTrial(study="study-timing", number=number, value=0.5, warmup_s=0.0, cells=cells, eligible=False)


def _ablation() -> AblationReport:
    arm = ArmSummary(
        study="study-timing",
        formulation="no_augmentation",
        file="recovery_search_no_augmentation_v1.toml",
        protocol_sha256="c" * 64,
        budget=4,
        trials_stored=4,
        n_feasible=2,
        best_number=17,
        best_value=0.5,
        reasons={"dwell": 2},
        feasible_by_warmup={"0": 2},
    )
    return AblationReport(
        dataset="processed-test",
        arms=(arm,),
        candidates=(_candidate(17), _candidate(3)),
        n_eligible=0,
        improving_rule="15 of 20",
        provenance=collect_provenance({}, seeds={}, artifacts=[], exploratory=True),
    )


def _freeze() -> FreezeRecord:
    studies = (
        StudyInput(
            file="recovery_search_no_augmentation_v1.toml",
            study="study-timing",
            formulation="no_augmentation",
            protocol_sha256="c" * 64,
            n_feasible=2,
            included=True,
        ),
        StudyInput(
            file="residual_search_1a_v1.toml",
            study="study-residual",
            formulation="residual",
            protocol_sha256="d" * 64,
            n_feasible=0,
            included=False,
            note="exploratory per D4",
        ),
    )
    return FreezeRecord(
        dataset="processed-test",
        rule="rule",
        studies=studies,
        ablation_file="development_ablation_v2.json",
        ablation_sha256="e" * 64,
        n_candidates=2,
        n_eligible=0,
        eligible_trials=(),
        outcome="negative",
        selection=None,
        panel=None,
        provenance=collect_provenance({}, seeds={}, artifacts=[], exploratory=True),
    )


def _pointer(study: str, formulation: str, n_feasible: int) -> StoredReport:
    return StoredReport(
        schema="recovery-study-report",
        study=study,
        formulation=formulation,
        protocol_sha256="c" * 64,
        dataset="processed-test",
        budget=4,
        trials_stored=4,
        n_feasible=n_feasible,
        payload=ArtifactReference("armrc://reports/task_1a_state_conditioned_recovery/x-abcdef.json", "0" * 64, 1),
        best_number=17 if n_feasible else None,
        best_value=0.5 if n_feasible else None,
    )


def _representative() -> RepresentativeRecord:
    pairs = tuple(
        PairOutcome(
            scenario_id=scenario_id,
            kind=kind,
            tracker=tracker,
            replay_run=f"run-replay-{kind}-{tracker}",
            rc_run=f"run-rc-{kind}-{tracker}",
            activation_s=0.25,
            replay_completed=True,
            rc_completed=False,
            recovery=None,
        )
        for kind, scenario_id in (
            ("nominal", "nominal"),
            ("posture_small", "small-2"),
            ("posture_large", "large-2"),
            ("force", "force-000deg"),
        )
        for tracker in ("pd_v2", "computed_torque")
    )
    return RepresentativeRecord(
        study="study-timing",
        trial=17,
        point_params={"warmup_s": 0.25},
        warmup_s=0.25,
        dataset="processed-test",
        selection_rule=SELECTION_RULE,
        scenarios={
            "nominal": "nominal",
            "posture_small": "small-2",
            "posture_large": "large-2",
            "force": "force-000deg",
        },
        pairs=pairs,
        provenance=collect_provenance({}, seeds={}, artifacts=[], exploratory=True),
    )


def _inputs() -> ReportInputs:
    pointers = {
        "recovery_search_no_augmentation_v1.toml": _pointer("study-timing", "no_augmentation", 2),
        "recovery_search_non_decaying_v1.toml": _pointer("study-nd", "non_decaying", 0),
        "recovery_search_contractive_v1.toml": _pointer("study-c", "contractive", 0),
        "residual_search_1a_v1.toml": _pointer("study-residual", "residual", 0),
    }
    return ReportInputs(
        pointers=pointers,
        ablation=_ablation(),
        freeze=_freeze(),
        representative=_representative(),
        reference=synthetic_task_samples(),
        runs={},
        effort={},
    )


def test_report_renders_the_negative_result_and_every_section() -> None:
    """The report states the accepted negative, renders the tables, and embeds plots and animations."""
    markdown = render_recovery_report(_inputs(), plots=("cell_gap_medians.png",), animations=("nominal_rc_pd.gif",))
    for required in (
        "# Task 1-a state-conditioned recovery: development results (v1)",
        "Accepted negative result",
        "no model is frozen and the",
        "## Study outcomes",
        "| study-timing | no_augmentation | 4 | 4 | 2 | 0.5 |",
        "## Paired distributions",
        "15-of-20 consistency requirement",
        "## Representative pairs",
        SELECTION_RULE,
        "| nominal | nominal | pd_v2 | run-rc-nominal-pd_v2 | run-replay-nominal-pd_v2 | n/a",
        "## Failures",
        "Flat infeasible objective",
        "Sampled, not exhaustive",
        "plots/recovery_report_v1/cell_gap_medians.png",
        "animations/nominal_rc_pd.gif",
        "scripts/play_run.py --run <run-id>",
    ):
        assert required in markdown


def test_pair_plot_masks_the_hold_and_refuses_overwrites(tmp_path: Path) -> None:
    """The trajectory figure accepts the NaN-masked RC output, writes once, and validates its inputs."""
    n = 40
    t = np.arange(n, dtype=np.float64) * 0.01
    finite = np.zeros((n, 2))
    rc_output = np.full((n, 2), np.nan)
    rc_output[10:] = 0.1
    out = tmp_path / "pair.png"
    written = plot_recovery_pair(
        t,
        finite,
        finite,
        rc_output,
        finite,
        out,
        title="pair",
        boundaries=(0.1, 0.2),
        xlabel="run time (s)",
    )
    assert written == out
    assert out.stat().st_size > 0
    assert not list(tmp_path.glob("*.tmp.png"))
    with pytest.raises(FileExistsError, match="refusing"):
        plot_recovery_pair(t, finite, finite, rc_output, finite, out, title="pair", boundaries=(), xlabel="x")
    with pytest.raises(ValueError, match="reference must"):
        plot_recovery_pair(
            t, np.zeros((n, 2, 1)), finite, rc_output, finite, tmp_path / "b.png", title="t", boundaries=(), xlabel="x"
        )
    bad = finite.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="replay_actual must be finite"):
        plot_recovery_pair(t, finite, bad, rc_output, finite, tmp_path / "c.png", title="t", boundaries=(), xlabel="x")
