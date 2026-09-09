# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-003: the committed diagnosis supports floating-point sensitivity behind the one retained failure (C11)."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.repetition_diagnosis import EPS64, load_diagnosis, render_diagnosis_markdown
from arm_rc_ctrl.experiments.repetition_numerics import TOLERANCES, load_validation
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

DOCS = repository_root() / "docs" / "experiments" / "task_1a_repeated_demonstration"
VALIDATION = DOCS / "numerical_validation_v1.json"
EVIDENCE = DOCS / "numerical_validation_v1_diagnosis.json"
MARKDOWN = DOCS / "numerical_validation_v1_diagnosis.md"


def test_committed_diagnosis_binds_the_validation_and_covers_its_failure() -> None:
    """The diagnosis names the committed validation, its execution identity, the failed pair, and two contrasts."""
    diagnosis = load_diagnosis(EVIDENCE)
    validation = load_validation(VALIDATION)
    assert diagnosis.validation_file == "docs/experiments/task_1a_repeated_demonstration/numerical_validation_v1.json"
    assert diagnosis.validation_sha256 == sha256_file(VALIDATION)
    assert diagnosis.execution_identity == validation.execution.identity
    assert not diagnosis.provenance.project_dirty
    assert not diagnosis.provenance.exploratory
    assert diagnosis.n_failed == 1
    failed = [d for d in diagnosis.diagnoses if d.failed]
    (failure,) = failed
    recorded = next(c for c in validation.comparisons if not c.passed)
    assert (failure.candidate_identity, failure.reference_identity) == (
        recorded.candidate_identity,
        recorded.reference_identity,
    )
    assert failure.observed_max_abs == recorded.differences[0].max_abs
    assert failure.observed_coefficient_fro_rel == recorded.coefficient_fro_rel
    assert [(d.formulation, d.candidate.split("/")[1], d.failed) for d in diagnosis.diagnoses] == [
        ("absolute", "R", True),
        ("absolute", "R-scaled", False),
        ("residual", "R", False),
    ]
    assert all(d.panel_label == "feasible-middle" and d.count == 65 for d in diagnosis.diagnoses)
    assert render_diagnosis_markdown(diagnosis) == MARKDOWN.read_text(encoding="utf-8")


def test_failure_is_roundoff_under_the_panel_s_worst_conditioning() -> None:
    """The gap is accumulation plus each fit's distance from its reference solution under cond2 above 1e9."""
    diagnosis = load_diagnosis(EVIDENCE)
    failure = next(d for d in diagnosis.diagnoses if d.failed)
    assert failure.cond2_reference > 1e9
    assert failure.sensitivity == pytest.approx(failure.cond2_reference * EPS64)
    # The stacked problem is K times the single one up to float64 accumulation roundoff.
    assert failure.accumulation_a_rel < 1e-14
    assert failure.accumulation_b_rel < 1e-14
    # Each fit's weights are within a small multiple of the first-order sensitivity from its reference solution
    # (assembly differences between rclib's Eigen path and the NumPy reconstruction included).
    assert failure.candidate_solve.normal_residual_extended < 1e-16
    assert failure.reference_solve.normal_residual_extended < 1e-16
    assert failure.candidate_solve.coefficient_fro_rel < 100.0 * failure.sensitivity
    assert failure.reference_solve.coefficient_fro_rel < 100.0 * failure.sensitivity
    # The observed gap is accounted for by the accumulation gap plus both reference-solution gaps.
    parts = (
        failure.accumulation_prediction_max_abs
        + failure.candidate_solve.prediction_max_abs
        + failure.reference_solve.prediction_max_abs
    )
    assert failure.observed_max_abs <= parts * (1.0 + 1e-6)
    assert failure.observed_max_abs > TOLERANCES.prediction_atol_rad
    # Contrasts: the residual counterpart shares the conditioning but its outputs are two orders smaller; the
    # R-scaled/S pair at the same entry and count is far better conditioned and far closer.
    residual = next(d for d in diagnosis.diagnoses if d.formulation == "residual")
    scaled = next(d for d in diagnosis.diagnoses if d.candidate.endswith("R-scaled/K65"))
    assert residual.cond2_reference == failure.cond2_reference
    assert residual.output_scale < failure.output_scale / 10.0
    assert residual.observed_max_abs < failure.observed_max_abs / 100.0
    assert scaled.cond2_reference < failure.cond2_reference / 10.0
    assert scaled.observed_max_abs < failure.observed_max_abs / 10.0
