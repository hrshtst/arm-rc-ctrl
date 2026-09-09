# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-003: the committed numerical validation is complete, bound to the panel and environment, and reproducible."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.execution import load_execution
from arm_rc_ctrl.experiments.repetition_fits import FitStore
from arm_rc_ctrl.experiments.repetition_numerics import (
    COMPARISON_PAIRS,
    PROBE_BANK_COUNT,
    PROBE_BANKS,
    TOLERANCES,
    PanelContext,
    load_validation,
    numerical_arms,
    render_validation_markdown,
)
from arm_rc_ctrl.experiments.repetition_panel import APPROVED_RULE, load_panel
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec, expected_loss_rows
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.rc.recipe import RclibIdentity
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_repeated_demonstration"
EVIDENCE = DOCS / "numerical_validation_v1.json"
MARKDOWN = DOCS / "numerical_validation_v1.md"
MANIFEST = DOCS / "panel_manifest_v1.json"
EXECUTION = DOCS / "execution_environment_v1.json"


FAILED_COMPARISON = ("feasible-middle", "absolute", 65, "absolute/R/K65", "absolute/S-effective/K65")
"""The one comparison the validation retained as failed (diagnosed in ``numerical_validation_v1_diagnosis``)."""


def test_committed_validation_covers_the_panel_and_retains_its_one_failure() -> None:
    """120 fits, 72 comparisons (71 within tolerance, one retained failure), 60 state identities, 120 fresh refits."""
    v = load_validation(EVIDENCE)
    assert v.experiment == "task_1a_repetition_v1"
    assert not v.all_passed  # one comparison exceeds the approved tolerance and is retained, never relaxed
    assert v.n_fits == 120
    assert v.n_comparisons == 72
    assert v.n_comparisons_passed == 71
    assert v.n_residuals_within == 120
    assert len(v.state_identities) == v.n_state_identities_identical == 60
    assert len(v.fresh_refits) == v.n_fresh_refits_passed == 120
    assert v.tolerances == TOLERANCES
    assert v.probe_bank_count == PROBE_BANK_COUNT == 64
    labels = tuple(APPROVED_RULE.labels)
    assert set(v.probes) == set(labels)
    assert all(v.probe_states_identical_across_formulations[label] for label in labels)
    arms = numerical_arms()
    assert [(f.panel_label, f.arm) for f in v.fits] == [(label, arm) for label in labels for arm in arms]
    for fit in v.fits:
        assert fit.fit.loss_rows == expected_loss_rows(fit.arm)  # 400 task rows per episode
        assert fit.copies_identical
        assert fit.normal.within
        assert fit.normal.columns == fit.weights_shape[0]
        assert fit.weights_shape[1] == 2
        assert fit.accessor_max_abs_diff < 1e-12
        assert fit.rclib == RclibIdentity.current()
        assert fit.execution_identity == v.execution.identity
    for label in labels:
        banks = v.probes[label]
        assert [b.bank for b in banks] == list(PROBE_BANKS)
        assert [b.episodes for b in banks] == [1, 64, 64]
        assert all(b.rows_per_episode == 400 for b in banks)
        assert banks[1].attempts_used == banks[2].attempts_used == 64  # every attempt accepted
    pairs = {(c.candidate.split("/")[1], c.reference.split("/")[1]) for c in v.comparisons}
    assert pairs == set(COMPARISON_PAIRS)
    assert {(c.panel_label, c.formulation, c.count) for c in v.comparisons} == {
        (label, formulation, count)
        for label in labels
        for formulation in ("absolute", "residual")
        for count in (17, 33, 65)
    }
    failed = [c for c in v.comparisons if not c.passed]
    assert [(c.panel_label, c.formulation, c.count, c.candidate, c.reference) for c in failed] == [FAILED_COMPARISON]
    (failure,) = failed
    assert failure.differences[0].quantity == "prediction"
    assert 2.9e-8 < failure.differences[0].max_abs < 3.0e-8
    assert failure.differences[0].worst_bank == "non_decaying"
    for c in v.comparisons:
        for d in c.differences:
            assert d.rows == 129 * 400
            if c.passed:
                assert d.max_abs <= TOLERANCES.prediction_atol_rad + TOLERANCES.prediction_rtol
    # The failed pair is the worst-conditioned ridge problem of the panel (its S-effective alpha is alpha_0 / 65).
    reference = next(f for f in v.fits if f.identity == failure.reference_identity)
    assert reference.normal.cond2 == max(f.normal.cond2 for f in v.fits)
    assert reference.normal.cond2 > 1e9
    assert reference.solver_alpha < 5e-5
    for r in v.fresh_refits:
        assert r.passed
        assert r.environment_match
        assert r.max_abs_weight_diff == 0.0
        assert r.rmse_abs_diff == 0.0


def test_committed_validation_binds_the_panel_environment_and_sources() -> None:
    """The evidence names the committed manifest, the canonical execution record, and the panel's dataset."""
    v = load_validation(EVIDENCE)
    manifest = load_panel(MANIFEST)
    assert v.panel_manifest_file == MANIFEST.relative_to(REPO_ROOT).as_posix()
    assert v.panel_manifest_sha256 == sha256_file(MANIFEST)
    assert v.dataset.artifact_id == manifest.configs.dataset
    assert v.dataset.payload_sha256 == manifest.configs.dataset_payload_sha256 == v.dataset_payload.sha256
    assert v.dataset.record == manifest.configs.dataset_record_file
    canonical = load_execution(EXECUTION)
    assert v.execution.identity == canonical.identity
    assert v.execution.canonical
    assert v.execution.policy == "p-cores"
    assert not v.provenance.project_dirty
    assert not v.provenance.exploratory
    assert v.provenance.artifacts == (v.dataset_payload,)
    assert render_validation_markdown(v) == MARKDOWN.read_text(encoding="utf-8")


def test_one_cached_fit_rebuilds_from_the_store(tmp_path: Path) -> None:
    """On the recording machine one cached fit re-derives its identity, record, and weights (needs the store)."""
    v = load_validation(EVIDENCE)
    try:
        store = open_storage()
        context = PanelContext.load(MANIFEST, store=store, root=REPO_ROOT, execution=v.execution)
    except Exception as exc:  # noqa: BLE001 - any setup failure just means no store on this runner
        pytest.skip(f"external storage unavailable: {exc}")
    fits = FitStore(store)
    entry = context.manifest.entries[0]
    summary = next(f for f in v.fits if f.panel_label == entry.label and f.arm == ArmSpec("absolute", "S"))
    assert context.inputs.identity(entry, summary.arm) == summary.identity
    if not fits.exists(summary.identity):
        pytest.skip("the fit cache of the recording machine is not present")
    record = fits.read_record(summary.identity)
    assert record.fit == summary.fit
    assert record.weights_sha256 == summary.weights_sha256
    assert array_digest(fits.read_weights(record)) == summary.weights_sha256
    cached = fits.fit_or_load(entry, summary.arm, context.inputs)
    assert cached.cache_hit
    assert cached.record == record
    del tmp_path
