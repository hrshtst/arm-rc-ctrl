# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-016 finding 3: the committed ablation rebuilds exactly from the payloads and replay baselines."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.evidence import load_report_pointer, open_stored_report
from arm_rc_ctrl.experiments.recovery_ablation import (
    build_ablation,
    load_ablation,
    render_ablation_markdown,
    replay_jump_table,
)
from arm_rc_ctrl.experiments.recovery_objective import RecoveryTrialContext
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS, load_recovery_search
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageAccessError, open_storage

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_state_conditioned_recovery"
DATASET = REPO_ROOT / "data" / "records" / "processed" / "processed-20260903-ce343c8ce6a5.toml"


def test_committed_ablation_rebuilds_exactly() -> None:
    """Payloads + recomputed replay baselines reproduce the committed JSON and Markdown byte for byte."""
    committed = load_ablation(DOCS / "development_ablation_v2.json")
    pointer_files = sorted(DOCS.glob("recovery_search_*_v1.toml"))
    pointers = [(f.name, load_report_pointer(f)) for f in pointer_files]
    try:
        store = open_storage()
    except Exception as exc:  # noqa: BLE001 - any setup failure just means no store on this runner
        pytest.skip(f"external storage unavailable: {exc}")
    try:
        inputs = [(name, open_stored_report(store, pointer)) for name, pointer in pointers]
        protocol_file = REPO_ROOT / next(
            report.protocol_file for _name, report in inputs if report.formulation == "no_augmentation"
        )
        context = RecoveryTrialContext.load(
            load_recovery_search(protocol_file), store=store, dataset_file=DATASET, records_root=REPO_ROOT
        )
    except StorageAccessError as exc:
        pytest.skip(f"external payload unavailable: {exc}")
    pairs = [
        (tracker, float(trial.params["warmup_s"]))
        for _name, report in inputs
        for trial in report.summary.trials
        if trial.flags.get("feasible") is True
        for tracker in RECOVERY_TRACKERS
    ]
    jumps = replay_jump_table(context, pairs)
    rebuilt = build_ablation(inputs, jumps, dataset=committed.dataset, provenance=committed.provenance)
    assert rebuilt == committed
    markdown = render_ablation_markdown(rebuilt)
    assert markdown == (DOCS / "development_ablation_v2.md").read_text(encoding="utf-8")
