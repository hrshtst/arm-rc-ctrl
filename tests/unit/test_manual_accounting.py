# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: accounting for every model and replay bank of the study, including the ones that are absent.

The freeze the ledger asks for is a complete ordering and accounting. Complete
means the study's own 186 entries are walked in manifest order and a model with
no evidence is listed as missing rather than omitted, so a partial execution is
visible as partial. Every present line is read from the manifest the pointer
resolves to, verified by size and digest, because a pointer summarising itself
is the one thing this experiment has learned not to trust.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_accounting import (
    BankAccount,
    ModelAccount,
    StudyAccounting,
    account_study,
)
from arm_rc_ctrl.experiments.manual_evaluation import (
    ManualEvaluationConfig,
    ManualEvaluationRunner,
    load_manual_evaluation_config,
)
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.provenance import ArtifactReference
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_study import StudyModel

REPO_ROOT = repository_root()
DEVELOPMENT_SOURCE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
TRACKER = TrackerConfig(type="pd", kp=(10.0, 5.0), kd=(1.5, 0.8))
HOLD_S, PULSE_S, HORIZON_S, WARMUP_S = 0.05, 0.02, 1.0, 0.25
CONFIGURATION, ARM_LABEL = "feasible-best", "S/D01"
SCENARIOS = (RobustnessScenario("nominal", "nominal", (0.0, 0.0)),)


def _evaluation(f: ManualFixture) -> tuple[ManualEvaluationConfig, Path]:
    """The fixture's own evaluation configuration."""
    evaluations = f.root / "configs" / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    development = evaluations / DEVELOPMENT_SOURCE.name
    shutil.copyfile(DEVELOPMENT_SOURCE, development)
    scenario = f.scenario_file
    limits = ", ".join(f"{v}" for v in load_manual_scenario(scenario).limits.velocity)
    target = evaluations / "task_1a_manual_dev_fixture.toml"
    target.write_text(
        f'name = "task-1a-manual-dev-fixture"\n'
        f'development = "{development.as_posix()}"\n'
        f'scenario = "{scenario.as_posix()}"\n'
        f"horizon_s = {HORIZON_S}\n\n"
        f"[trigger]\nhold_s = {HOLD_S}\nduration_s = {PULSE_S}\nmagnitude_n = 3.0\n\n"
        f"[simulation]\nvelocity_abort = [{limits}]\n",
        encoding="utf-8",
    )
    return load_manual_evaluation_config(target), target


def _runner(f: ManualFixture) -> ManualEvaluationRunner:
    config, file = _evaluation(f)
    return ManualEvaluationRunner(
        store=f.store,
        inputs=f.inputs,
        config=config,
        evaluation_file=file,
        scenarios=SCENARIOS,
        trackers={"pd_v2": TRACKER, "computed_torque": TRACKER},
        root=f.root,
        execution=f.execution,
        provenance=f.provenance,
    )


def _entry(f: ManualFixture) -> StudyModel:
    return next(e for e in f.manifest.entries if (e.configuration, e.arm.label) == (CONFIGURATION, ARM_LABEL))


def _evidence_of_one_model(f: ManualFixture, tmp_path: Path) -> Path:
    """Evaluate a single model and leave its pointers in a directory of its own."""
    runner = _runner(f)
    runner.evaluate(_entry(f), warmup_s=WARMUP_S)
    evidence_dir = tmp_path / "evidence"
    runner.write_pointers(evidence_dir)
    return evidence_dir


# --- completeness ------------------------------------------------------------------------------


def test_every_study_model_gets_a_line_present_or_not(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """A partial execution is visible as partial: 186 lines, one per entry, in manifest order."""
    f = manual_fixture
    accounting = account_study(
        store=f.store, evidence_dir=_evidence_of_one_model(f, tmp_path), manifest=f.manifest, provenance=f.provenance
    )
    assert len(accounting.models) == len(f.manifest.entries) == 186
    assert [line.label for line in accounting.models] == [entry.label for entry in f.manifest.entries]
    assert accounting.n_present == 1
    assert accounting.n_missing == 185
    assert accounting.complete is False


def test_the_missing_models_are_named_rather_than_inferred(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """The record lists what is absent, so a reader never has to subtract to find out."""
    f = manual_fixture
    accounting = account_study(
        store=f.store, evidence_dir=_evidence_of_one_model(f, tmp_path), manifest=f.manifest, provenance=f.provenance
    )
    evaluated = f"{CONFIGURATION}/{ARM_LABEL}"
    assert evaluated not in accounting.missing
    assert len(accounting.missing) == 185
    assert all(not line.present for line in accounting.models if line.label in set(accounting.missing))


def test_a_present_line_is_read_from_the_manifest_not_the_pointer(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """The counts and identity come from the verified manifest; a pointer never summarises itself here."""
    f = manual_fixture
    accounting = account_study(
        store=f.store, evidence_dir=_evidence_of_one_model(f, tmp_path), manifest=f.manifest, provenance=f.provenance
    )
    line = next(line for line in accounting.models if line.present)
    assert line.label == f"{CONFIGURATION}/{ARM_LABEL}"
    assert line.status in {"feasible", "infeasible"}
    assert line.n_pairs == len(SCENARIOS) * 2
    assert line.n_completed + line.n_infeasible == line.n_pairs
    assert line.execution_identity == f.execution.identity
    assert line.fit_identity is not None


def test_the_replay_banks_are_accounted_with_their_parent_and_policy(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """A bank belongs to a parent under one derivative policy, so both are recorded, not parsed from a name."""
    f = manual_fixture
    accounting = account_study(
        store=f.store, evidence_dir=_evidence_of_one_model(f, tmp_path), manifest=f.manifest, provenance=f.provenance
    )
    assert len(accounting.banks) == 1
    bank = accounting.banks[0]
    assert bank.assignment == "D01"
    assert bank.warmup_s == WARMUP_S
    assert bank.velocity_cutoff_hz > 0
    assert bank.acceleration_cutoff_hz > 0
    assert bank.execution_identity == f.execution.identity


def test_the_totals_re_derive_from_the_lines(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """A record that disagrees with its own lines is refused rather than reported."""
    f = manual_fixture
    accounting = account_study(
        store=f.store, evidence_dir=_evidence_of_one_model(f, tmp_path), manifest=f.manifest, provenance=f.provenance
    )
    assert accounting.n_rc_runs == sum(line.n_pairs for line in accounting.models if line.present)
    assert accounting.n_replay_runs == sum(bank.n_pairs for bank in accounting.banks)
    assert accounting.all_bind_canonical_execution is True
    assert accounting.canonical_execution_identity == f.execution.identity


def test_an_empty_evidence_directory_accounts_for_everything_as_missing(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """Nothing executed is a complete answer too, and never mistaken for nothing to do."""
    f = manual_fixture
    empty = tmp_path / "empty"
    empty.mkdir()
    accounting = account_study(store=f.store, evidence_dir=empty, manifest=f.manifest, provenance=f.provenance)
    assert accounting.n_present == 0
    assert accounting.n_missing == len(f.manifest.entries)
    assert accounting.banks == ()
    assert accounting.complete is False


# --- the guards on the record itself -----------------------------------------------------------


def _reference() -> ArtifactReference:
    return ArtifactReference("armrc://reports/task_1a_manual_v1/model/x/manifest-abc.json", "a" * 64, 10)


def _present_line(**overrides: object) -> ModelAccount:
    arguments: dict[str, object] = {
        "label": "feasible-best/S/D01",
        "configuration": "feasible-best",
        "arm": "S/D01",
        "present": True,
        "identity": "b" * 64,
        "status": "feasible",
        "n_pairs": 2,
        "n_completed": 2,
        "n_infeasible": 0,
        "execution_identity": "c" * 64,
        "payload": _reference(),
    }
    arguments.update(overrides)
    return ModelAccount(**arguments)  # type: ignore[arg-type]


def test_a_present_line_must_carry_the_facts_that_make_it_present() -> None:
    """Present means its manifest was read; a line claiming that without them is not evidence of anything."""
    with pytest.raises(ValueError, match="present model carries"):
        ModelAccount(label="feasible-best/S/D01", configuration="feasible-best", arm="S/D01", present=True)


def test_a_line_whose_counts_do_not_sum_is_refused() -> None:
    """Completed, infeasible and unexecuted together are the pairs; anything else is a miscount."""
    with pytest.raises(ValueError, match="is not"):
        _present_line(n_pairs=4, n_completed=1, n_infeasible=1)


def test_a_bank_whose_counts_do_not_sum_is_refused() -> None:
    """The same arithmetic on the baseline side."""
    with pytest.raises(ValueError, match="is not"):
        BankAccount(
            identity="d" * 64,
            assignment="D01",
            warmup_s=0.25,
            velocity_cutoff_hz=6.7,
            acceleration_cutoff_hz=5.6,
            n_pairs=4,
            n_completed=1,
            n_infeasible=1,
            execution_identity="c" * 64,
            payload=_reference(),
        )


def _accounting(manual_fixture: ManualFixture, **overrides: object) -> StudyAccounting:
    line = _present_line()
    arguments: dict[str, object] = {
        "experiment": EXPERIMENT_LABEL,
        "canonical_execution_identity": "c" * 64,
        "models": (line,),
        "banks": (),
        "n_models": 1,
        "n_present": 1,
        "n_missing": 0,
        "missing": (),
        "statuses": {"feasible": 1},
        "n_rc_runs": 2,
        "n_replay_runs": 0,
        "all_bind_canonical_execution": True,
        "complete": True,
        "provenance": manual_fixture.provenance,
    }
    arguments.update(overrides)
    return StudyAccounting(**arguments)  # type: ignore[arg-type]


def test_a_consistent_accounting_is_accepted(manual_fixture: ManualFixture) -> None:
    """The record these guard cases mutate is itself valid, or they would prove nothing."""
    assert _accounting(manual_fixture).complete is True


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("n_present", 0), ("n_missing", 1), ("missing", ("x",)), ("n_rc_runs", 99), ("complete", False)],
)
def test_an_accounting_that_contradicts_its_lines_is_refused(
    manual_fixture: ManualFixture, field_name: str, value: object
) -> None:
    """Totals are a claim about the lines; a record that disagrees with its own is not reportable."""
    with pytest.raises(ValueError, match="re-derive"):
        _accounting(manual_fixture, **{field_name: value})


def test_an_accounting_of_another_experiment_or_schema_is_refused(manual_fixture: ManualFixture) -> None:
    """Evidence names the experiment it belongs to."""
    with pytest.raises(ValueError, match=r"schema|experiment"):
        _accounting(manual_fixture, experiment="task_1a_repetition_v1")


def test_a_canonical_claim_that_contradicts_the_lines_is_refused(manual_fixture: ManualFixture) -> None:
    """Whether every manifest binds the canonical environment is re-derived, never asserted."""
    with pytest.raises(ValueError, match="canonical"):
        _accounting(manual_fixture, canonical_execution_identity="e" * 64)


def test_a_pointer_naming_another_manifest_is_refused(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """A pointer resolves to a manifest; if it claims a different identity, one of them is not this evidence.

    The pointer and the manifest each state an identity, and nothing else
    compares them. Reading the manifest through a pointer that names something
    else would report evidence the repository does not actually point at.
    """
    f = manual_fixture
    evidence_dir = _evidence_of_one_model(f, tmp_path)
    pointer_path = next(evidence_dir.glob("model__*.toml"))
    # Only the identity line: the same digest also appears inside payload.uri, and replacing that
    # too would point at a manifest that does not exist and fail before this guard is reached.
    lines = pointer_path.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = [f'identity = "{"f" * 64}"\n' if line.startswith("identity") else line for line in lines]
    assert changed != lines, "the pointer must carry an identity line to alter"
    pointer_path.write_text("".join(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="is not the pointer"):
        account_study(store=f.store, evidence_dir=evidence_dir, manifest=f.manifest, provenance=f.provenance)
