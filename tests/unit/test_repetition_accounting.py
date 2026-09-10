# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-006: the execution accounting lists every configuration, present or missing, with its manifest facts."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments import repetition_accounting
from arm_rc_ctrl.experiments.repetition_accounting import (
    ModelAccount,
    account_pilot,
    load_accounting,
    main,
    render_accounting_markdown,
)
from arm_rc_ctrl.experiments.repetition_fixture import (
    DOCS,
    CraftedSimulator,
    PlanarFixture,
    build_pilot_runner,
    exploratory_provenance,
)
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec
from arm_rc_ctrl.provenance import sha256_file

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST = DOCS / "panel_manifest_v1.json"
VALIDATION = DOCS / "numerical_validation_v1.json"


def test_accounting_lists_present_and_missing_configurations(fixture: PlanarFixture, tmp_path: Path) -> None:
    """Two fixture evaluations of feasible-best appear present; the other 118 configurations are listed missing."""
    f = fixture
    runner = build_pilot_runner(f, velocity_abort=(47.0, 47.0), simulate_fn=CraftedSimulator(f.samples))
    runner.evaluate(f.entry, ArmSpec("absolute", "S"))
    runner.evaluate(f.entry, ArmSpec("residual", "R", 16))
    evidence_dir = tmp_path / "evidence"
    runner.write_pointers(evidence_dir)
    accounting = account_pilot(
        store=f.store,
        evidence_dir=evidence_dir,
        manifest_file=MANIFEST,
        validation_file=VALIDATION,
        provenance=exploratory_provenance(),
    )
    assert accounting.n_configurations == 120
    assert accounting.n_present == 2
    assert accounting.n_missing == 118
    assert "feasible-middle/absolute/S" in accounting.missing
    assert accounting.statuses["feasible"] == 2
    assert accounting.n_rc_runs == 20
    assert accounting.n_replay_runs == 10
    assert len(accounting.banks) == 1
    assert accounting.banks[0].warmup_s == 0.25
    assert accounting.numerical_reference_fits == 36
    assert accounting.all_carry_c11
    assert not accounting.all_bind_canonical_execution  # the fixture ran in the test process, not the canonical one
    assert not accounting.complete
    present = [m for m in accounting.models if m.present]
    assert [(m.panel_label, m.arm) for m in present] == [
        ("feasible-best", "absolute/S"),
        ("feasible-best", "residual/R/K17"),
    ]
    assert present[0].payload is not None
    assert present[0].fit_identity is not None
    assert set(present[0].cells) == {
        "posture_small:pd_v2",
        "posture_small:computed_torque",
        "posture_large:pd_v2",
        "posture_large:computed_torque",
    }
    # Order follows the panel and report order: entry-major, arm-minor.
    assert [m.arm for m in accounting.models[:3]] == ["absolute/S", "absolute/R/K17", "absolute/R-scaled/K17"]
    markdown = render_accounting_markdown(accounting)
    assert "**missing**" in markdown
    assert "accounting complete: **False**" in markdown
    file = tmp_path / "accounting.json"
    file.write_text(json.dumps(__import__("arm_rc_ctrl.config", fromlist=["to_mapping"]).to_mapping(accounting)))
    assert load_accounting(file) == accounting
    with pytest.raises(ValueError, match="contradicts"):
        replace(accounting, n_present=3)
    with pytest.raises(ValueError, match="complete contradicts"):
        replace(accounting, complete=True)
    with pytest.raises(ValueError, match="present configuration carries"):
        replace(present[0], payload=None)
    with pytest.raises(ValueError, match="unknown status"):
        ModelAccount(panel_label="x", arm="absolute/S", formulation="absolute", count=1, present=False, status="other")
    # A pointer whose manifest identity differs is refused.
    pointer_file = next(evidence_dir.glob("model__feasible-best__absolute__S.toml"))
    text = pointer_file.read_text(encoding="utf-8")
    tampered = re.sub(r'^identity = "[0-9a-f]{64}"$', f'identity = "{"f" * 64}"', text, count=1, flags=re.MULTILINE)
    assert tampered != text
    pointer_file.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="is not the pointer's"):
        account_pilot(
            store=f.store,
            evidence_dir=evidence_dir,
            manifest_file=MANIFEST,
            validation_file=VALIDATION,
            provenance=exploratory_provenance(),
        )
    pointer_file.write_text(text, encoding="utf-8")


def test_account_command_writes_the_evidence(
    fixture: PlanarFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command reads the pointers through the store and writes JSON and Markdown once."""
    f = fixture
    runner = build_pilot_runner(f, velocity_abort=(48.0, 48.0), simulate_fn=CraftedSimulator(f.samples))
    runner.evaluate(f.entry, ArmSpec("absolute", "R-scaled", 32))
    evidence_dir = tmp_path / "evidence"
    runner.write_pointers(evidence_dir)
    monkeypatch.setattr(repetition_accounting, "open_storage", lambda: f.store)
    output, markdown = tmp_path / "pilot_execution.json", tmp_path / "pilot_execution.md"
    argv = [
        "account",
        "--manifest",
        str(MANIFEST),
        "--validation",
        str(VALIDATION),
        "--evidence-dir",
        str(evidence_dir),
        "--output",
        str(output),
        "--markdown",
        str(markdown),
        "--exploratory",
    ]
    assert main(argv) == 0
    accounting = load_accounting(output)
    assert accounting.n_present == 1
    assert accounting.panel_manifest_sha256 == sha256_file(MANIFEST)
    assert markdown.read_text(encoding="utf-8") == render_accounting_markdown(accounting)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)
