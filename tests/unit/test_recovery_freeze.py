# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-015: the freeze record — rule application, consistency binding, and the negative outcome."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.recovery_ablation import AblationReport, ArmSummary, CandidateCell, CandidateTrial
from arm_rc_ctrl.experiments.recovery_freeze import (
    FreezeRecord,
    StudyInput,
    build_freeze,
    freeze_to_json,
    load_freeze,
    render_freeze_markdown,
)
from arm_rc_ctrl.provenance import collect_provenance

if TYPE_CHECKING:
    from pathlib import Path

CELL_NAMES = (
    "posture_small:pd_v2",
    "posture_small:computed_torque",
    "posture_large:pd_v2",
    "posture_large:computed_torque",
)


def _cell(*, passes: bool) -> CandidateCell:
    if passes:
        return CandidateCell(gap_median=0.5, jump_median=0.5, improving_both=15, n=20, passes=True)
    return CandidateCell(gap_median=0.5, jump_median=1.5, improving_both=0, n=20, passes=False)


def _candidate(study: str, number: int, *, eligible: bool) -> CandidateTrial:
    return CandidateTrial(
        study=study,
        number=number,
        value=0.5,
        warmup_s=0.0,
        cells={name: _cell(passes=eligible) for name in CELL_NAMES},
        eligible=eligible,
    )


def _arm(study: str, formulation: str, *, n_feasible: int, stored: int = 4) -> ArmSummary:
    return ArmSummary(
        study=study,
        formulation=formulation,
        file=f"{study}.json",
        protocol_sha256="c" * 64,
        budget=4,
        trials_stored=stored,
        n_feasible=n_feasible,
        best_number=0 if n_feasible else None,
        best_value=0.5 if n_feasible else None,
        reasons={"dwell": stored - n_feasible},
        feasible_by_warmup={"0": n_feasible} if n_feasible else {},
    )


def _ablation(*, eligible: bool) -> AblationReport:
    candidates = (_candidate("study-timing", 0, eligible=eligible), _candidate("study-timing", 1, eligible=False))
    return AblationReport(
        dataset="processed-test",
        arms=(_arm("study-timing", "no_augmentation", n_feasible=2),),
        candidates=candidates,
        n_eligible=sum(1 for c in candidates if c.eligible),
        improving_rule="ceil(0.75 * n)",
        provenance=collect_provenance({}, seeds={}, artifacts=[], exploratory=True),
    )


def _inputs(*, digest: str = "c" * 64, n_feasible: int = 2) -> list[StudyInput]:
    return [
        StudyInput(
            file="study-timing.json",
            study="study-timing",
            formulation="no_augmentation",
            protocol_sha256=digest,
            n_feasible=n_feasible,
            included=True,
        ),
        StudyInput(
            file="residual.json",
            study="study-residual",
            formulation="residual",
            protocol_sha256="d" * 64,
            n_feasible=0,
            included=False,
            note="exploratory per D4",
        ),
    ]


def test_negative_outcome_is_recorded_with_no_selection(tmp_path: Path) -> None:
    """Zero eligible candidates yield the negative record, its markdown, and a strict roundtrip."""
    provenance = collect_provenance({}, seeds={}, artifacts=[], exploratory=True)
    record = build_freeze(
        _ablation(eligible=False),
        ablation_file="development_ablation_v1.json",
        ablation_sha256="e" * 64,
        studies=_inputs(),
        provenance=provenance,
    )
    assert record.outcome == "negative"
    assert record.n_candidates == 2
    assert record.n_eligible == 0
    assert record.selection is None
    assert record.panel is None
    markdown = render_freeze_markdown(record)
    for required in (
        "# Task 1-a recovery model freeze",
        "NEGATIVE RESULT",
        "confirmatory gate stays closed",
        "exploratory per D4",
        "reservoir-seed-panel stability precedes any freeze",
    ):
        assert required in markdown
    file = tmp_path / "freeze.json"
    file.write_text(freeze_to_json(record) + "\n", encoding="utf-8")
    assert load_freeze(file) == record
    tampered = json.loads(file.read_text(encoding="utf-8"))
    tampered["outcome"] = "frozen"
    file.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="eligible"):
        load_freeze(file)


def test_eligible_candidates_refuse_the_negative_path_builder() -> None:
    """With an eligible candidate the builder demands the explicit panel-backed freeze."""
    provenance = collect_provenance({}, seeds={}, artifacts=[], exploratory=True)
    with pytest.raises(ValueError, match="panel"):
        build_freeze(
            _ablation(eligible=True),
            ablation_file="development_ablation_v1.json",
            ablation_sha256="e" * 64,
            studies=_inputs(),
            provenance=provenance,
        )


def test_inputs_must_bind_the_ablation() -> None:
    """Digest, feasible-count, exclusion, and note inconsistencies are all refused."""
    provenance = collect_provenance({}, seeds={}, artifacts=[], exploratory=True)
    ablation = _ablation(eligible=False)
    with pytest.raises(ValueError, match="does not match"):
        build_freeze(
            ablation,
            ablation_file="a.json",
            ablation_sha256="e" * 64,
            studies=_inputs(digest="f" * 64),
            provenance=provenance,
        )
    with pytest.raises(ValueError, match="contradicts"):
        build_freeze(
            ablation,
            ablation_file="a.json",
            ablation_sha256="e" * 64,
            studies=_inputs(n_feasible=1),
            provenance=provenance,
        )
    bad_exclusion = _inputs()
    bad_exclusion[1] = StudyInput(
        file="study-timing-2.json",
        study="study-timing",
        formulation="no_augmentation",
        protocol_sha256="c" * 64,
        n_feasible=0,
        included=False,
        note="wrongly excluded",
    )
    with pytest.raises(ValueError, match="must not appear"):
        build_freeze(
            ablation, ablation_file="a.json", ablation_sha256="e" * 64, studies=bad_exclusion, provenance=provenance
        )
    with pytest.raises(ValueError, match="state why"):
        StudyInput(
            file="x.json",
            study="x",
            formulation="residual",
            protocol_sha256="d" * 64,
            n_feasible=0,
            included=False,
        )


def test_frozen_schema_path_validates_selection_and_panel() -> None:
    """The frozen outcome (unused under protocol v1) still enforces its own invariants."""
    provenance = collect_provenance({}, seeds={}, artifacts=[], exploratory=True)

    def frozen(selection: str, panel: str | None) -> FreezeRecord:
        return FreezeRecord(
            dataset="processed-test",
            rule="rule",
            studies=tuple(_inputs()),
            ablation_file="a.json",
            ablation_sha256="e" * 64,
            n_candidates=2,
            n_eligible=1,
            eligible_trials=("study-timing:0",),
            outcome="frozen",
            selection=selection,
            panel=panel,
            provenance=provenance,
        )

    record = frozen("study-timing:0", "stability_v1.json")
    assert record.outcome == "frozen"
    with pytest.raises(ValueError, match="selects one of"):
        frozen("study-timing:9", "stability_v1.json")
    with pytest.raises(ValueError, match="panel"):
        frozen("study-timing:0", None)
